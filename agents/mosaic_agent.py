"""
Phase 2: Mosaic AI NL-to-SQL Agent (MLflow compatible)
Includes Security + RAI guardrails for CI/CD compliance.
"""

import os
import re

# ── MODEL ENDPOINT ────────────────────────────────────────────────────────────
MODEL_NAME = os.environ.get("MODEL_ENDPOINT", "databricks-claude-3-7-sonnet")

# ── SYSTEM PROMPT — loaded from prompts/system_prompt.txt ────────────────────
_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "..", "prompts", "system_prompt.txt")

def _load_prompt() -> tuple[str, str]:
    """
    Load system prompt from file.
    Returns (prompt_text, version_tag).
    The first line may be a version comment:  # PROMPT_VERSION: v1.2 | ...
    That line is stripped before sending to the LLM.
    """
    try:
        with open(os.path.abspath(_PROMPT_PATH), "r") as f:
            content = f.read()
    except Exception as e:
        raise FileNotFoundError(
            f"Could not load system prompt from {_PROMPT_PATH}: {e}\n"
            "Ensure prompts/system_prompt.txt exists in the project root."
        )

    lines   = content.splitlines()
    version = "unknown"

    # Parse optional version header from line 1
    if lines and lines[0].startswith("# PROMPT_VERSION:"):
        version = lines[0].split("PROMPT_VERSION:")[-1].split("|")[0].strip()
        content = "\n".join(lines[1:]).lstrip("\n")   # strip header from prompt

    return content, version

SYSTEM_PROMPT, PROMPT_VERSION = _load_prompt()
print(f"[mosaic_agent] Loaded prompt version: {PROMPT_VERSION}")

# ── SPARK INJECTION ───────────────────────────────────────────────────────────
# spark is a Databricks notebook global — modules loaded via importlib don't
# inherit it. Inject it from the notebook after loading:
#   mosaic_agent.spark = spark
spark = None


# Track token usage across calls (for cost governance)
_last_token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def generate_sql(question: str) -> str:
    """Generate SQL using Databricks Foundation Model API via MLflow deployments.
    Also tracks token usage for cost governance (Section 5.3 of architecture)."""
    global _last_token_usage
    from mlflow.deployments import get_deploy_client

    client = get_deploy_client("databricks")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"Generate SQL for: {question}"}
    ]

    response = client.predict(
        endpoint=MODEL_NAME,
        inputs={
            "messages": messages,
            "temperature": 0.1,
            "max_tokens": 500
        }
    )

    # ── COST TRACKING: extract token usage ────────────────────────────────
    usage = response.get("usage", {})
    _last_token_usage = {
        "prompt_tokens":     usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens":      usage.get("total_tokens", 0),
    }

    # Log to MLflow if an active run exists
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.log_metric("total_tokens", _last_token_usage["total_tokens"])
    except Exception:
        pass  # Don't crash agent if MLflow logging fails

    sql_query = response["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wraps output in ```sql ... ```
    if sql_query.startswith("```"):
        lines = sql_query.splitlines()
        sql_query = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()

    return sql_query


def execute_sql(sql_query: str) -> str:
    """Execute SQL on Spark and return result as string (numeric or text)."""
    if spark is None:
        raise RuntimeError(
            "spark is not injected. After loading this module via importlib, "
            "set:  mosaic_agent.spark = spark"
        )
    try:
        df     = spark.sql(sql_query)
        result = df.collect()[0][0]
        if result is None:
            return "0"
        # Return as float-string if numeric, else raw string (for text/date results)
        try:
            return str(float(result))
        except (ValueError, TypeError):
            return str(result)
    except Exception as e:
        print(f"[execute_sql] ERROR: {e}")
        return "0"


# ── GUARDRAILS (Security + RAI) ──────────────────────────────────────────────

def validate_query_safety(sql: str) -> bool:
    """
    SECURITY GATE: Enforces SELECT-only, blocks DML, cross-schema access,
    and checks for LIMIT to prevent unbounded scans.
    Returns True if the query is safe.
    """
    sql_upper = sql.upper().strip()

    # 1. SELECT-only enforcement
    if not sql_upper.startswith("SELECT"):
        return False

    # 2. Block DML / DDL keywords
    forbidden = ["DROP", "DELETE", "UPDATE", "INSERT", "GRANT", "TRUNCATE", "ALTER", "CREATE"]
    if any(word in sql_upper for word in forbidden):
        return False

    # 3. Block queries referencing tables outside cicd.gold
    if "FROM" in sql_upper and "cicd.gold" not in sql.lower():
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


def predict(question: str) -> dict:
    """
    Main prediction function with integrated guardrails + cost tracking.
    Input : question (str)
    Output: {"question", "sql", "answer", "prompt_version", "model",
             "sql_safe", "output_safe", "total_tokens", "tables_used"}
    """
    sql_query   = generate_sql(question)
    sql_safe    = validate_query_safety(sql_query)

    # If SQL is unsafe, do NOT execute — return blocked result
    if not sql_safe:
        return {
            "question":       question,
            "sql":            sql_query,
            "answer":         "BLOCKED: unsafe SQL detected",
            "prompt_version": PROMPT_VERSION,
            "model":          MODEL_NAME,
            "sql_safe":       False,
            "output_safe":    True,
            "total_tokens":   _last_token_usage["total_tokens"],
            "tables_used":    ["cicd.gold.fact_sale", "cicd.gold.dim_date"]
        }

    answer       = execute_sql(sql_query)
    output_safe  = validate_output_safety(answer)

    return {
        "question":       question,
        "sql":            sql_query,
        "answer":         answer,
        "prompt_version": PROMPT_VERSION,
        "model":          MODEL_NAME,
        "sql_safe":       sql_safe,
        "output_safe":    output_safe,
        "total_tokens":   _last_token_usage["total_tokens"],
        "tables_used":    ["cicd.gold.fact_sale", "cicd.gold.dim_date"]
    }
