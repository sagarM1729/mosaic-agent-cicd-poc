"""
Phase 3: Mosaic AI LangChain Agent — Multi–Genie-Space Router
═══════════════════════════════════════════════════════════════
Uses LangChain ReAct agent to intelligently decide between:
  • Sales_Expert Genie Space   (revenue, profit, invoices, customers)
  • Inventory_Expert Genie Space (stock, bins, reorder, colours, brands)

The LLM (Claude 3.7 Sonnet via Databricks Model Serving) acts as the
"brain" that reads the user's question, picks the right Genie Space,
and formats the final answer.

Includes: Security guardrails, RAI output validation, cost tracking,
          MLflow tracing for full observability.
"""

import os
import re
import time

import mlflow

# ── Fix 8: Enable MLflow tracing — shows full internal call graph ─────────────
# In the MLflow UI → Experiments → any run → "Traces" tab you'll see:
#   ▼ agent_run → guardrail_check → genie_tool_call → claude_llm_call → output_filter
mlflow.tracing.enable()

# ── CONFIG ────────────────────────────────────────────────────────────────────
from agents.config import MODEL_ENDPOINT

# ── PROMPT ────────────────────────────────────────────────────────────────────
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt")


def _load_prompt() -> tuple[str, str]:
    """
    Load system prompt from file.
    Returns (prompt_text, version_tag).
    """
    try:
        with open(os.path.abspath(_PROMPT_PATH), "r") as f:
            content = f.read()
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load system prompt from {_PROMPT_PATH}: {e}\n"
            "Ensure prompts/system_prompt.txt exists in the project root."
        )
    lines = content.splitlines()
    version = "unknown"
    if lines and lines[0].startswith("# PROMPT_VERSION:"):
        version = lines[0].split("PROMPT_VERSION:")[-1].split("|")[0].strip()
        content = "\n".join(lines[1:]).lstrip("\n")
    return content, version


SYSTEM_PROMPT, PROMPT_VERSION = _load_prompt()
print(f"[mosaic_agent] Loaded prompt version: {PROMPT_VERSION}")

# ── SPARK injection (set by notebook after importlib load) ────────────────────
spark = None

# ── TOKEN USAGE TRACKING (cost governance) ────────────────────────────────────
_last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# ══════════════════════════════════════════════════════════════════════════════
#  LANGCHAIN AGENT SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _build_agent():
    """
    Build the LangChain ReAct agent with two Genie Space tools.
    Uses ChatDatabricks as the LLM (routes to Databricks Model Serving).
    """
    from langchain_core.prompts import PromptTemplate
    from langchain_databricks import ChatDatabricks

    # AgentExecutor / create_react_agent moved across langchain versions
    try:
        from langchain.agents import AgentExecutor, create_react_agent
    except ImportError:
        from langchain.agents.agent import AgentExecutor
        from langchain.agents.react.agent import create_react_agent

    try:
        from langchain.tools import Tool
    except ImportError:
        from langchain_core.tools import Tool

    # ── Import our Genie tools ────────────────────────────────────────────
    from agents.tools import inventory_genie_tool, sales_genie_tool

    # ── LLM: Databricks Foundation Model ──────────────────────────────────
    llm = ChatDatabricks(
        endpoint=MODEL_ENDPOINT,
        temperature=0.1,
        max_tokens=1024,
    )

    # ── Define Tools ──────────────────────────────────────────────────────
    tools = [
        Tool(
            name="Sales_Expert",
            func=sales_genie_tool,
            description=(
                "Use this tool for ALL questions about sales, revenue, profit, "
                "tax, invoices, transactions, customers, salespersons, cities, "
                "package types, quantities sold, or anything related to "
                "cicd.gold.fact_sale, cicd.gold.dim_customer, cicd.gold.dim_date "
                "(in a sales context). "
                "Input should be the original user question as-is."
            ),
        ),
        Tool(
            name="Inventory_Expert",
            func=inventory_genie_tool,
            description=(
                "Use this tool for ALL questions about stock items, inventory, "
                "stock holdings, quantity on hand, bin locations, reorder levels, "
                "target stock levels, cost prices, colors, brands, unit packages, "
                "or anything related to cicd.gold.dim_stock_item, "
                "cicd.gold.fact_stock_holding. "
                "Input should be the original user question as-is."
            ),
        ),
    ]

    # ── ReAct Prompt ──────────────────────────────────────────────────────
    # SYSTEM_PROMPT is loaded from prompts/system_prompt.txt so that changes
    # to the prompt file actually affect agent behaviour (SQL patterns, month
    # filters, domain rules, etc.)
    react_template = """You are a data analytics assistant with access to two specialized tools.
Your job is to answer the user's question by choosing the RIGHT tool.

== ROUTING RULES & DOMAIN KNOWLEDGE ==
{system_prompt}
== END ROUTING RULES ==

You have access to the following tools:

{{tools}}

Tool names: {{tool_names}}

Use the following format:

Question: the input question you must answer
Thought: I need to figure out which domain this question belongs to
Action: the tool to use, must be one of [{{tool_names}}]
Action Input: the question to pass to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: the final answer to the original question (return ONLY the value, no explanation)

IMPORTANT: Return ONLY the raw value as the Final Answer. No sentences, no explanation.
If the tool returns a number, return just the number.
If the tool returns a name or text, return just that text.

Begin!

Question: {{input}}
Thought: {{agent_scratchpad}}""".format(system_prompt=SYSTEM_PROMPT)

    prompt = PromptTemplate(
        input_variables=["input", "agent_scratchpad", "tools", "tool_names"],
        template=react_template,
    )

    # ── Build Agent ───────────────────────────────────────────────────────
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=3,              # Prevent infinite loops
        return_intermediate_steps=True,
    )

    return agent_executor


# Cache the built agent (avoid rebuilding on every call)
_agent_executor = None


def _get_agent():
    """Lazy-load the agent (imports happen only on first call)."""
    global _agent_executor
    if _agent_executor is None:
        _agent_executor = _build_agent()
    return _agent_executor


# ══════════════════════════════════════════════════════════════════════════════
#  GUARDRAILS — imported from agents/guardrails.py (testable without Databricks)
# ══════════════════════════════════════════════════════════════════════════════
from agents.guardrails import validate_output_safety, validate_query_safety

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(question: str) -> dict:
    """
    Main prediction function — LangChain Agent routes to Genie Spaces.

    Input : question (str)
    Output: {
        "question", "answer", "generated_sql", "source_tool",
        "prompt_version", "model", "sql_safe", "output_safe", "total_tokens"
    }
    """
    global _last_token_usage

    agent = _get_agent()
    start_time = time.time()

    try:
        from langchain.callbacks.base import BaseCallbackHandler
        
        class TokenUsageCallback(BaseCallbackHandler):
            def on_llm_end(self, response, **kwargs):
                global _last_token_usage
                if response.llm_output and "token_usage" in response.llm_output:
                    usage = response.llm_output["token_usage"]
                    _last_token_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
                    _last_token_usage["completion_tokens"] += usage.get("completion_tokens", 0)
                    _last_token_usage["total_tokens"] += usage.get("total_tokens", 0)

        # Reset counters for this incoming request
        _last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        # ── Invoke LangChain agent with token tracking callback ───────────
        result = agent.invoke(
            {"input": question},
            config={"callbacks": [TokenUsageCallback()]}
        )
        answer = result.get("output", "ERROR: No output from agent")

        # ── Determine which tool was used ─────────────────────────────────
        source_tool = "unknown"
        generated_sql = None
        intermediate_steps = result.get("intermediate_steps", [])
        if intermediate_steps:
            # Each step is (AgentAction, observation)
            last_action = intermediate_steps[-1][0]
            source_tool = getattr(last_action, "tool", "unknown")

        # ── Extract generated SQL from tool observations ──────────────────
        # tools.py embeds the Genie-generated SQL as [SQL_USED: ...] in the
        # tool return string. Extract it for the Dual-Validation Gate.
        for _action, observation in reversed(intermediate_steps):
            obs = str(observation)
            if "[SQL_USED:" in obs:
                sql_start = obs.index("[SQL_USED:") + len("[SQL_USED:")
                # Find the closing "] You must now" marker (robust against ] in SQL)
                end_marker = "] You must now"
                end_pos = obs.find(end_marker, sql_start)
                if end_pos == -1:
                    # Fallback: find first ] after sql_start
                    end_pos = obs.index("]", sql_start)
                generated_sql = obs[sql_start:end_pos].strip()
                if generated_sql == "N/A":
                    generated_sql = None
                break

        # ── FIX: Extract answer from tool output if agent looped out ──────
        # When the LLM hits max_iterations without outputting "Final Answer:",
        # LangChain returns "Agent stopped due to iteration limit".
        # But the tool DID return the correct value — extract it.
        if "Agent stopped" in answer and intermediate_steps:
            for _action, observation in reversed(intermediate_steps):
                obs = str(observation)
                # Our tool wraps results as:
                # "The database returned this exact result: X. [SQL_USED: ...] You must now..."
                if "The database returned this exact result:" in obs:
                    extracted = obs.split("The database returned this exact result:")[1]
                    # Split on ". [SQL_USED" to avoid breaking decimals like "2902877.0"
                    extracted = extracted.split(". [SQL_USED")[0].strip()
                    answer = extracted
                    print(f"[mosaic_agent] Extracted answer from tool output: {answer}")
                    break

    except Exception as e:
        print(f"[mosaic_agent] ERROR: {e}")
        answer = f"AGENT_ERROR: {str(e)}"
        source_tool = "error"
        generated_sql = None

    elapsed = round(time.time() - start_time, 2)

    # ── Guardrail checks (production-grade) ─────────────────────────────────
    # Validate the full context: user question + generated SQL + final answer
    full_context_to_validate = f"{question} {generated_sql or ''} {answer}"
    security_result = validate_query_safety(full_context_to_validate)
    rai_result = validate_output_safety(answer)

    # If output is unsafe, replace with blocked message
    if not security_result["passed"] or not rai_result["passed"]:
        blocked_flags = security_result["flags"] + rai_result["flags"]
        safe_answer = f"BLOCKED: guardrail violation — {', '.join(blocked_flags[:3])}"
        print(f"[mosaic_agent] ⛔ BLOCKED: {blocked_flags}")
    else:
        safe_answer = answer

    # ── Fallback Token Tracking (if LangChain drops LLM metadata) ───────────
    if _last_token_usage.get("total_tokens", 0) == 0:
        total_chars = len(SYSTEM_PROMPT) + len(str(question)) + len(str(answer)) + len(str(generated_sql or ""))
        _last_token_usage["total_tokens"] = max(10, total_chars // 4)

    # ── Log final metrics to MLflow (Traceability & Cost Monitoring) ────────
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric("response_time_seconds", elapsed)
            mlflow.log_metric("total_tokens", _last_token_usage.get("total_tokens", 0))
    except Exception:
        pass

    return {
        "question":        question,
        "sql":             f"[Genie Space: {source_tool}]",
        "generated_sql":   generated_sql,
        "answer":          safe_answer,
        "source_tool":     source_tool,
        "prompt_version":  PROMPT_VERSION,
        "model":           MODEL_ENDPOINT,
        # Structured guardrail results (for CI gate evaluation)
        "security_gate":   security_result,    # {passed, score, flags, details}
        "rai_gate":        rai_result,          # {passed, score, flags, details}
        "sql_safe":        security_result["passed"],   # Backward compat
        "output_safe":     rai_result["passed"],        # Backward compat
        "total_tokens":    _last_token_usage.get("total_tokens", 0),
        "elapsed_seconds": elapsed,
    }
