"""
Phase 3: Mosaic AI LangChain Agent — Multi–Genie-Space Router
═══════════════════════════════════════════════════════════════
Uses LangChain ReAct agent to intelligently decide between:
  • Sales_Expert Genie Space   (revenue, profit, invoices, customers)
  • Inventory_Expert Genie Space (stock, bins, reorder, colours, brands)

The LLM (Claude 3.7 Sonnet via Databricks Model Serving) acts as the
"brain" that reads the user's question, picks the right Genie Space,
and formats the final answer.

Includes: Security guardrails, RAI output validation, cost tracking.
"""

import os
import re
import time

# ── CONFIG ────────────────────────────────────────────────────────────────────
from agents.config import MODEL_ENDPOINT, QUALITY_GATE_THRESHOLD

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
    from langchain_community.chat_models import ChatDatabricks
    from langchain_core.prompts import PromptTemplate

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
    from agents.tools import sales_genie_tool, inventory_genie_tool

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
    # This instructs the LLM how to reason step-by-step and pick tools.
    react_template = """You are a data analytics assistant with access to two specialized tools.
Your job is to answer the user's question by choosing the RIGHT tool.

DECISION RULES:
- If the question is about sales, revenue, profit, tax, invoices, customers,
  salesperson, cities, package types, or quantities sold → use Sales_Expert
- If the question is about stock items, inventory, stock holdings, bin locations,
  reorder levels, cost prices, colors, brands, packaging → use Inventory_Expert
- If the question is about dates, calendar, weekends, years, months (general) →
  use Sales_Expert (dim_date is available in both, but prefer Sales_Expert for general date questions)
- NEVER make up an answer. ALWAYS use one of the tools.

You have access to the following tools:

{tools}

Tool names: {tool_names}

Use the following format:

Question: the input question you must answer
Thought: I need to figure out which domain this question belongs to
Action: the tool to use, must be one of [{tool_names}]
Action Input: the question to pass to the tool
Observation: the result from the tool
Thought: I now know the final answer
Final Answer: the final answer to the original question (return ONLY the value, no explanation)

IMPORTANT: Return ONLY the raw value as the Final Answer. No sentences, no explanation.
If the tool returns a number, return just the number.
If the tool returns a name or text, return just that text.

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

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
        max_iterations=3,                 # Prevent infinite loops
        early_stopping_method="generate", # Force LLM to output Final Answer if limit hit
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
#  GUARDRAILS (Security + RAI)
# ══════════════════════════════════════════════════════════════════════════════

def validate_query_safety(sql_or_answer: str) -> bool:
    """
    SECURITY GATE: Checks the agent output for dangerous patterns.
    Since Genie Space handles SQL generation internally, we validate
    the final answer rather than raw SQL.
    Returns True if the output is safe.
    """
    text_upper = str(sql_or_answer).upper().strip()

    # Block DML / DDL keywords in case they leak through
    forbidden = [
        "DROP", "DELETE", "UPDATE", "INSERT",
        "GRANT", "TRUNCATE", "ALTER", "CREATE",
    ]
    if any(word in text_upper for word in forbidden):
        return False

    return True


def validate_output_safety(answer: str) -> bool:
    """
    RAI GATE: Prevents PII leakage, massive hallucinated data dumps,
    and enforces response size limits.
    Returns True if the output is safe.
    """
    answer_str = str(answer)

    # 1. Block massive text dumps (hallucination / unbounded query)
    if len(answer_str) > 500:
        return False

    # 2. Block SSN/PII patterns (e.g., 123-45-6789)
    if re.search(r'\d{3}-\d{2}-\d{4}', answer_str):
        return False

    # 3. Block if response looks like a data dump (multiple newlines = rows)
    if answer_str.count('\n') > 20:
        return False

    return True


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN PREDICTION FUNCTION
# ══════════════════════════════════════════════════════════════════════════════

def predict(question: str) -> dict:
    """
    Main prediction function — LangChain Agent routes to Genie Spaces.

    Input : question (str)
    Output: {
        "question", "answer", "source_tool", "prompt_version", "model",
        "sql_safe", "output_safe", "total_tokens"
    }
    """
    global _last_token_usage

    agent = _get_agent()
    start_time = time.time()

    try:
        # ── Invoke LangChain agent ────────────────────────────────────────
        result = agent.invoke({"input": question})
        answer = result.get("output", "ERROR: No output from agent")

        # ── Determine which tool was used ─────────────────────────────────
        source_tool = "unknown"
        intermediate_steps = result.get("intermediate_steps", [])
        if intermediate_steps:
            # Each step is (AgentAction, observation)
            last_action = intermediate_steps[-1][0]
            source_tool = getattr(last_action, "tool", "unknown")

    except Exception as e:
        print(f"[mosaic_agent] ERROR: {e}")
        answer = f"AGENT_ERROR: {str(e)}"
        source_tool = "error"

    elapsed = round(time.time() - start_time, 2)

    # ── Token tracking (from LangChain callbacks if available) ────────────
    # LangChain's ChatDatabricks doesn't always expose token usage.
    # We track what we can; for full cost tracking, use MLflow metrics.
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric("response_time_seconds", elapsed)
    except Exception:
        pass

    # ── Guardrail checks ──────────────────────────────────────────────────
    sql_safe = validate_query_safety(answer)
    output_safe = validate_output_safety(answer)

    # If output is unsafe, replace with blocked message
    if not sql_safe or not output_safe:
        safe_answer = "BLOCKED: unsafe output detected by guardrails"
    else:
        safe_answer = answer

    return {
        "question":       question,
        "sql":            f"[Genie Space: {source_tool}]",  # SQL is internal to Genie
        "answer":         safe_answer,
        "source_tool":    source_tool,
        "prompt_version": PROMPT_VERSION,
        "model":          MODEL_ENDPOINT,
        "sql_safe":       sql_safe,
        "output_safe":    output_safe,
        "total_tokens":   _last_token_usage.get("total_tokens", 0),
        "elapsed_seconds": elapsed,
    }
