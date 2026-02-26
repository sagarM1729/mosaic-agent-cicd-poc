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
#  GUARDRAILS (Production-Grade: Security + RAI + Toxicity)
# ══════════════════════════════════════════════════════════════════════════════

# ── Response size config (not a hard cutoff — scored proportionally) ──────────
MAX_RESPONSE_LENGTH = 2000   # Soft limit: answers above this get flagged
CRITICAL_RESPONSE_LENGTH = 5000  # Hard limit: answers above this are blocked

# ── Toxicity keyword list (lowercase) ─────────────────────────────────────────
# These are common terms that should NEVER appear in a data analytics response.
# Kept intentionally minimal to avoid false positives on legitimate data values.
_TOXIC_KEYWORDS = [
    "kill", "murder", "suicide", "rape", "terrorist", "bomb",
    "hate speech", "racial slur", "white supremac", "nazi",
    "profanity", "f**k", "sh*t", "damn", "bastard",
    "exploit", "hack", "malware", "ransomware",
]

# ── DML / DDL / Injection patterns ────────────────────────────────────────────
_FORBIDDEN_SQL_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "GRANT", "TRUNCATE",
    "ALTER", "CREATE", "EXEC", "EXECUTE", "UNION SELECT",
    "OR 1=1", "OR '1'='1'", "--", ";--", "/*", "*/",
    "INFORMATION_SCHEMA", "SYSOBJECTS", "SYSCOLUMNS",
]

# ── Credential / secret patterns ──────────────────────────────────────────────
_CREDENTIAL_PATTERNS = [
    r'(?i)(password|passwd|pwd)\s*[:=]\s*\S+',
    r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+',
    r'(?i)(bearer\s+[a-zA-Z0-9\-_.]+)',
    r'dapi[a-f0-9]{32}',                   # Databricks PAT token pattern
    r'(?i)BEGIN\s+(RSA\s+)?PRIVATE\s+KEY',  # Private keys
]

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (r'\d{3}-\d{2}-\d{4}',                       "SSN"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  "Email"),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',           "Phone"),       # US phone
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b', "CreditCard"), # 16-digit card
]   


def validate_query_safety(sql_or_answer: str) -> dict:
    """
    SECURITY GATE: Production-grade security validation.
    
    Checks for: SQL injection, DML/DDL leakage, credential exposure,
    path traversal, and command injection patterns.
    
    Returns:
        dict with keys: passed (bool), score (0.0-1.0), flags (list), details (str)
        score = 1.0 means fully safe, 0.0 means critical violation.
    """
    text = str(sql_or_answer).strip()
    text_upper = text.upper()
    flags = []
    
    # ── Check 1: DML / DDL / SQL Injection ────────────────────────────────
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        # If the keyword is purely alphabetic (like DROP, SELECT), use word boundaries
        # to prevent catching 'executive' (EXEC) or 'alternative' (ALTER).
        if re.match(r'^[A-Za-z\s]+$', keyword):
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"SECURITY:SQL_INJECTION:{keyword}")
        else:
            # For symbols like '--', '/*', 'OR 1=1', use simple substring match
            if keyword.upper() in text_upper:
                flags.append(f"SECURITY:SQL_INJECTION:{keyword}")
    
    # ── Check 2: Credential / secret leakage ──────────────────────────────
    for pattern in _CREDENTIAL_PATTERNS:
        if re.search(pattern, text):
            flags.append(f"SECURITY:CREDENTIAL_LEAK:{pattern[:30]}")
    
    # ── Check 3: Path traversal ───────────────────────────────────────────
    if "../" in text or "..\\" in text:
        flags.append("SECURITY:PATH_TRAVERSAL")
    
    # ── Check 4: Command injection markers ────────────────────────────────
    cmd_patterns = [r'\$\(', r'`[^`]+`', r'\|\s*\w+', r';\s*\w+']
    for pattern in cmd_patterns:
        if re.search(pattern, text):
            flags.append(f"SECURITY:CMD_INJECTION:{pattern[:20]}")
    
    # ── Score calculation ─────────────────────────────────────────────────
    if not flags:
        score = 1.0
    elif any("SQL_INJECTION" in f for f in flags):
        score = 0.0  # Critical: zero tolerance for SQL injection
    elif any("CREDENTIAL_LEAK" in f for f in flags):
        score = 0.1  # Critical: credential exposure
    else:
        score = max(0.0, 1.0 - (len(flags) * 0.3))
    
    passed = len(flags) == 0
    details = "Security check passed" if passed else f"Violations: {', '.join(flags)}"
    
    return {"passed": passed, "score": score, "flags": flags, "details": details}


def validate_output_safety(answer: str) -> dict:
    """
    RAI GATE: Production-grade Responsible AI validation.
    
    Checks for: PII leakage, toxicity, data dumps, response size,
    and content safety.
    
    Returns:
        dict with keys: passed (bool), score (0.0-1.0), flags (list), details (str)
        score = 1.0 means fully safe, 0.0 means critical violation.
    """
    answer_str = str(answer)
    answer_lower = answer_str.lower()
    flags = []
    deductions = 0.0
    
    # ── Check 1: PII Detection ────────────────────────────────────────────
    for pattern, pii_type in _PII_PATTERNS:
        matches = re.findall(pattern, answer_str)
        if matches:
            flags.append(f"RAI:PII_{pii_type}:found_{len(matches)}")
            deductions += 0.4  # Severe: PII leakage
    
    # ── Check 2: Toxicity / Harmful Content ───────────────────────────────
    toxic_found = []
    for keyword in _TOXIC_KEYWORDS:
        if keyword in answer_lower:
            toxic_found.append(keyword)
    if toxic_found:
        flags.append(f"RAI:TOXICITY:{','.join(toxic_found[:5])}")
        deductions += 0.5  # Severe: toxic content
    
    # ── Check 3: Response Size (scored, not hard cutoff) ──────────────────
    length = len(answer_str)
    if length > CRITICAL_RESPONSE_LENGTH:
        flags.append(f"RAI:RESPONSE_TOO_LARGE:{length}_chars")
        deductions += 0.5  # Hard block for massive responses
    elif length > MAX_RESPONSE_LENGTH:
        flags.append(f"RAI:RESPONSE_LARGE:{length}_chars")
        deductions += 0.2  # Warning: unusually large response
    
    # ── Check 4: Data Dump Detection ──────────────────────────────────────
    newline_count = answer_str.count('\n')
    if newline_count > 50:
        flags.append(f"RAI:DATA_DUMP:massive_{newline_count}_lines")
        deductions += 0.5
    elif newline_count > 20:
        flags.append(f"RAI:DATA_DUMP:{newline_count}_lines")
        deductions += 0.2
    
    # ── Check 5: Repetitive content (hallucination indicator) ─────────────
    words = answer_lower.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            flags.append(f"RAI:HALLUCINATION:repetitive_content_{unique_ratio:.2f}")
            deductions += 0.3
    
    # ── Score calculation ─────────────────────────────────────────────────
    score = max(0.0, 1.0 - deductions)
    passed = len(flags) == 0
    details = "RAI check passed" if passed else f"Flags: {', '.join(flags)}"
    
    return {"passed": passed, "score": score, "flags": flags, "details": details}


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
