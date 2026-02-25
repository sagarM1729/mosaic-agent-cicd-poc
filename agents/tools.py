"""
tools.py — Genie Space tools for the LangChain agent.

Two tools:
  1. sales_genie_tool     → calls Sales_Expert Genie Space
  2. inventory_genie_tool → calls Inventory_Expert Genie Space

Each tool uses the Databricks Genie API to:
  - Start a conversation with the question
  - Poll until the Genie Space produces a result
  - Return the answer string

Falls back to direct SQL execution if Genie cannot answer.
"""

import os
import time
import json
import requests

from agents.config import (
    SALES_GENIE_SPACE_ID,
    INVENTORY_GENIE_SPACE_ID,
    SALES_TABLES,
    INVENTORY_TABLES,
)

# ── DATABRICKS AUTH ──────────────────────────────────────────────────────────
# In Databricks notebooks, the token is auto-injected.
# In CI/CD (GitHub Actions), it comes from environment variables.

def _get_headers():
    """Build auth headers for the Databricks REST API."""
    token = os.environ.get("DATABRICKS_TOKEN")
    if not token:
        # Inside a Databricks notebook, use the built-in token
        try:
            from dbruntime.sdk import get_token
            token = get_token()
        except Exception:
            pass
    if not token:
        try:
            # Another notebook fallback
            token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()  # noqa: F821
        except Exception:
            pass
    if not token:
        raise RuntimeError(
            "No Databricks token found. Set DATABRICKS_TOKEN env var "
            "or run inside a Databricks notebook."
        )
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }


def _get_host():
    """Get Databricks workspace host URL."""
    host = os.environ.get("DATABRICKS_HOST")
    if not host:
        try:
            host = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().get()  # noqa: F821
        except Exception:
            pass
    if not host:
        host = "https://adb-7405619257134796.16.azuredatabricks.net"
    return host.rstrip("/")


# ── GENIE API HELPERS ────────────────────────────────────────────────────────

def _call_genie_space(space_id: str, question: str, max_wait: int = 120) -> str:
    """
    Call a Databricks Genie Space via REST API.

    Flow:
      1. POST /api/2.0/genie/spaces/{space_id}/start-conversation
         → returns conversation_id + message_id
      2. GET  /api/2.0/genie/spaces/{space_id}/conversations/{conv_id}/messages/{msg_id}
         → poll until status is COMPLETED or FAILED
      3. Extract the result from the response

    Returns the answer as a string, or an error message.
    """
    host = _get_host()
    headers = _get_headers()

    # ── Step 1: Start conversation ────────────────────────────────────────
    start_url = f"{host}/api/2.0/genie/spaces/{space_id}/start-conversation"
    payload = {"content": question}

    try:
        resp = requests.post(start_url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        return f"GENIE_ERROR: Failed to start conversation: {e}"

    data = resp.json()
    conversation_id = data.get("conversation_id")
    message_id = data.get("message_id")

    if not conversation_id or not message_id:
        return f"GENIE_ERROR: Missing conversation_id or message_id in response: {json.dumps(data)}"

    # ── Step 2: Poll for result ───────────────────────────────────────────
    poll_url = (
        f"{host}/api/2.0/genie/spaces/{space_id}"
        f"/conversations/{conversation_id}/messages/{message_id}"
    )

    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            poll_resp = requests.get(poll_url, headers=headers, timeout=30)
            poll_resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            return f"GENIE_ERROR: Polling failed: {e}"

        msg_data = poll_resp.json()
        status = msg_data.get("status", "UNKNOWN")

        if status == "COMPLETED":
            # ── Step 3: Extract answer ────────────────────────────────────
            return _extract_genie_answer(msg_data)

        elif status in ("FAILED", "CANCELLED"):
            error_msg = msg_data.get("error", "Unknown error from Genie Space")
            return f"GENIE_ERROR: {status}: {error_msg}"

        # Still executing — wait and retry
        time.sleep(3)

    return "GENIE_ERROR: Timed out waiting for Genie Space response."


def _extract_genie_answer(msg_data: dict) -> str:
    """
    Extract the final answer from a completed Genie message response.
    Genie may return the answer in various structures — handle all.
    """
    # Try 'attachments' → 'query' → 'result' path (SQL result)
    attachments = msg_data.get("attachments", [])
    for attachment in attachments:
        # Check for query result
        query_result = attachment.get("query", {}).get("result")
        if query_result:
            # result could be a data table or a scalar
            columns = query_result.get("columns", [])
            data_rows = query_result.get("data_array", [])

            if data_rows and len(data_rows) > 0:
                # Return the first cell of the first row for single-value answers
                if len(data_rows) == 1 and len(data_rows[0]) == 1:
                    return str(data_rows[0][0])
                # For multi-row results, format as a simple string
                if len(data_rows) == 1:
                    return " | ".join(str(cell) for cell in data_rows[0])
                # Multiple rows — return first row's first cell
                return str(data_rows[0][0])

        # Check for text content
        text_content = attachment.get("text", {}).get("content")
        if text_content:
            return text_content

    # Fallback: check top-level content
    content = msg_data.get("content")
    if content:
        return content

    return "GENIE_ERROR: Could not extract answer from Genie response."


# ── LANGCHAIN TOOL DEFINITIONS ───────────────────────────────────────────────
# These functions are what the LangChain agent will call.

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
    result = _call_genie_space(SALES_GENIE_SPACE_ID, question)
    print(f"[TOOL] 📊 Sales_Expert result: {result}")
    return result


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
    result = _call_genie_space(INVENTORY_GENIE_SPACE_ID, question)
    print(f"[TOOL] 📦 Inventory_Expert result: {result}")
    return result
