"""
tools.py — Genie Space tools for the LangChain agent.

Correct 4-step Genie API flow:
  Step 1: POST  /spaces/{id}/start-conversation
          → returns conversation_id + message_id
  Step 2: GET   /spaces/{id}/conversations/{conv}/messages/{msg}
          → poll until status == COMPLETED
          → also returns attachments with attachment_id for each query
  Step 3: GET   /spaces/{id}/conversations/{conv}/messages/{msg}/attachments/{att}/query-result
          → returns statement_response → result → data_typed_array
          → this is the ACTUAL SQL result data

Two tools:
  1. sales_genie_tool     → Sales_Expert Genie Space
  2. inventory_genie_tool → Inventory_Expert Genie Space
"""

import os
import time
import json
import requests

from agents.config import (
    SALES_GENIE_SPACE_ID,
    INVENTORY_GENIE_SPACE_ID,
)


# ── DATABRICKS AUTH ──────────────────────────────────────────────────────────
# The calling notebook (test.py / deploy.py / register_model.py) extracts
# credentials from dbutils and sets them as env vars BEFORE loading this module.
# This module simply reads DATABRICKS_HOST + DATABRICKS_TOKEN.

def _get_auth():
    """
    Returns (host, headers) for Databricks REST API calls.
    Reads env vars set by the calling notebook (which has dbutils access).
    """
    host  = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")

    if host and token:
        print(f"[tools] Auth via env vars ✅  host={host}")
        return host, {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    raise RuntimeError(
        "DATABRICKS_HOST or DATABRICKS_TOKEN not set.\n"
        "The calling notebook must inject these from dbutils before loading tools.py.\n"
        "Example:  os.environ['DATABRICKS_TOKEN'] = ctx.apiToken().get()"
    )


# ── GENIE API — CORRECT 4-STEP FLOW ─────────────────────────────────────────

def _call_genie_space(space_id: str, question: str, max_wait: int = 120) -> dict:
    """
    Call a Databricks Genie Space via REST API.

    Correct flow:
      Step 1: POST  start-conversation              → conversation_id, message_id
      Step 2: GET   messages/{msg_id}               → poll until COMPLETED
                    also extracts attachment_id from each query attachment
      Step 3: GET   messages/{msg_id}/attachments/{att_id}/query-result
                    → statement_response.result.data_typed_array
                    → extract first value
      Step 4: Fallback to text attachment if no query result

    Returns a dict with 'answer' (str) and 'sql' (str or None).
    On failure, 'answer' contains a GENIE_ERROR string.
    """
    try:
        host, headers = _get_auth()
    except RuntimeError as e:
        return {"answer": f"GENIE_ERROR: Auth failed: {e}", "sql": None}

    # ── Step 1: Start conversation ────────────────────────────────────────
    start_url = f"{host}/api/2.0/genie/spaces/{space_id}/start-conversation"
    try:
        resp = requests.post(
            start_url,
            headers=headers,
            json={"content": question},
            timeout=30,
        )
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return {"answer": f"GENIE_ERROR: Failed to start conversation: {e}", "sql": None}

    data            = resp.json()
    conversation_id = data.get("conversation_id")
    message_id      = data.get("message_id")

    if not conversation_id or not message_id:
        return {"answer": f"GENIE_ERROR: Missing ids in start-conversation response: {json.dumps(data)}", "sql": None}

    print(f"  [Genie] conversation={conversation_id}  message={message_id}")

    # ── Step 2: Poll message until COMPLETED ──────────────────────────────
    poll_url = (
        f"{host}/api/2.0/genie/spaces/{space_id}"
        f"/conversations/{conversation_id}/messages/{message_id}"
    )

    msg_data = None
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            poll_resp = requests.get(poll_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return {"answer": f"GENIE_ERROR: Polling failed: {e}", "sql": None}

        msg_data = poll_resp.json()
        status   = msg_data.get("status", "UNKNOWN")
        print(f"  [Genie] status={status}")

        if status == "COMPLETED":
            break
        elif status in ("FAILED", "CANCELLED"):
            error_msg = msg_data.get("error", "Unknown Genie error")
            return {"answer": f"GENIE_ERROR: {status}: {error_msg}", "sql": None}

        time.sleep(3)
    else:
        return {"answer": "GENIE_ERROR: Timed out waiting for Genie Space response.", "sql": None}

    # ── Step 3: Fetch query result from each query attachment ─────────────
    # The message attachments array tells us which attachments have SQL results.
    # We must call a SEPARATE endpoint to get the actual data rows.
    attachments = msg_data.get("attachments", [])
    print(f"  [Genie] {len(attachments)} attachment(s) in message")

    for attachment in attachments:
        attachment_id = attachment.get("attachment_id") or attachment.get("id")

        # ── 3a: Query attachment → call query-result endpoint ─────────────
        if attachment.get("query") is not None and attachment_id:
            # Extract the generated SQL from the query attachment
            generated_sql = attachment.get("query", {}).get("query", None)
            if generated_sql:
                print(f"  [Genie] generated SQL: {generated_sql[:120]}...")

            query_result_url = (
                f"{host}/api/2.0/genie/spaces/{space_id}"
                f"/conversations/{conversation_id}"
                f"/messages/{message_id}"
                f"/attachments/{attachment_id}/query-result"
            )
            try:
                qr_resp = requests.get(query_result_url, headers=headers, timeout=30)
                qr_resp.raise_for_status()
                qr_data = qr_resp.json()
            except requests.exceptions.RequestException as e:
                print(f"  [Genie] query-result fetch failed: {e}")
                continue

            print(f"  [Genie] query-result raw keys: {list(qr_data.keys())}")

            # Navigate: statement_response → result → data_typed_array
            stmt_resp = qr_data.get("statement_response", {})
            result    = stmt_resp.get("result", {})

            # data_typed_array: list of {values: [{str: "..."} or {null: {}}]}
            data_typed = result.get("data_typed_array", [])
            if data_typed:
                first_row    = data_typed[0]
                values       = first_row.get("values", [])
                if values:
                    # Each value is {"str": "123.45"} or {"null": {}}
                    cell = values[0]
                    answer = None
                    if "str" in cell:
                        answer = cell["str"]
                    elif "i64" in cell:
                        answer = str(cell["i64"])
                    elif "f64" in cell:
                        answer = str(cell["f64"])
                    elif "bool" in cell:
                        answer = str(cell["bool"])
                    elif "null" in cell:
                        answer = "NULL"
                    else:
                        answer = str(cell)
                    if answer is not None:
                        return {"answer": answer, "sql": generated_sql}

            # Fallback: try data_array (older API response format)
            data_array = result.get("data_array", [])
            if data_array:
                return {"answer": str(data_array[0][0]), "sql": generated_sql}

            # Fallback: check manifest for column names + try other paths
            manifest = stmt_resp.get("manifest", {})
            print(f"  [Genie] manifest schema: {manifest.get('schema', {})}")

        # ── 3b: Text attachment → return the text directly ────────────────
        text_content = attachment.get("text", {}).get("content", "")
        if text_content:
            return {"answer": text_content.strip(), "sql": None}

    # ── Step 4: Last resort — top-level content ───────────────────────────
    content = msg_data.get("content", "")
    if content:
        return {"answer": content.strip(), "sql": None}

    return {"answer": f"GENIE_ERROR: Could not extract answer. Attachment keys: {[list(a.keys()) for a in attachments]}", "sql": None}


# ── LANGCHAIN TOOL DEFINITIONS ───────────────────────────────────────────────

def sales_genie_tool(question: str) -> str:
    """
    Ask the Sales_Expert Genie Space a question about sales data.
    Use this tool for questions about:
    - Revenue, profit, sales amounts, tax
    - Invoices, transactions, orders
    - Customers, salespersons, cities
    - Package types, quantities sold
    - Any question involving fact_sale, dim_customer, or dim_date (sales context)

    Tables available: cicd.gold.fact_sale, cicd.gold.dim_date, cicd.gold.dim_customer
    """
    print(f"[TOOL] 📊 Routing to Sales_Expert Genie Space: {question}")
    genie_result = _call_genie_space(SALES_GENIE_SPACE_ID, question)
    answer = genie_result["answer"]
    sql    = genie_result.get("sql") or "N/A"
    print(f"[TOOL] 📊 Sales_Expert result: {answer}")
    print(f"[TOOL] 📊 Sales_Expert SQL: {sql}")
    
    # ── FIX: Prevent infinite ReAct loops ──────────────────────────────────────
    # If we return a naked number (like '116830'), the agent gets confused, thinks 
    # it still hasn't answered the question, and asks the tool again endlessly.
    # By giving explicit format instructions, we force it to terminate.
    # Also embed the SQL for downstream extraction by mosaic_agent.py.
    return f"The database returned this exact result: {answer}. [SQL_USED: {sql}] You must now output 'Final Answer: {answer}'"


def inventory_genie_tool(question: str) -> str:
    """
    Ask the Inventory_Expert Genie Space a question about inventory/stock data.
    Use this tool for questions about:
    - Stock items, stock holdings, quantity on hand
    - Bin locations, reorder levels, target stock levels
    - Cost prices, colors, brands, packaging types
    - Any question involving dim_stock_item or fact_stock_holding

    Tables available: cicd.gold.dim_stock_item, cicd.gold.dim_date, cicd.gold.fact_stock_holding
    """
    print(f"[TOOL] 📦 Routing to Inventory_Expert Genie Space: {question}")
    genie_result = _call_genie_space(INVENTORY_GENIE_SPACE_ID, question)
    answer = genie_result["answer"]
    sql    = genie_result.get("sql") or "N/A"
    print(f"[TOOL] 📦 Inventory_Expert result: {answer}")
    print(f"[TOOL] 📦 Inventory_Expert SQL: {sql}")
    
    # ── FIX: Prevent infinite ReAct loops ──────────────────────────────────────
    return f"The database returned this exact result: {answer}. [SQL_USED: {sql}] You must now output 'Final Answer: {answer}'"
