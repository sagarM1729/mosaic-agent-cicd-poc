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
# 3-layer auth strategy (in order of preference):
#   1. SparkContext  — always available on Databricks cluster, no dbutils needed
#   2. Databricks SDK credential provider — works for some auth configs
#   3. Environment variables — for CI/CD / local development

def _get_auth():
    """
    Returns (host, headers) for Databricks REST API calls.
    Works inside job clusters (via SparkContext), notebooks, and CI/CD.
    """

    # ── Layer 1: SparkContext ─────────────────────────────────────────────
    # On every Databricks cluster, SparkContext holds the cluster token.
    try:
        from pyspark import SparkContext
        sc    = SparkContext.getOrCreate()
        token = sc._conf.get("spark.databricks.token", "")
        url   = sc._conf.get("spark.databricks.workspaceUrl", "")
        if token and url:
            host = f"https://{url}".rstrip("/")
            print(f"[tools] Auth via SparkContext ✅  host={host}")
            return host, {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            }
    except Exception as e:
        print(f"[tools] SparkContext auth failed: {e}")

    # ── Layer 2: Databricks SDK credential provider ───────────────────────
    try:
        from databricks.sdk import WorkspaceClient
        w = WorkspaceClient()
        host = w.config.host.rstrip("/")
        # Use the SDK's credential provider to get live headers
        creds_provider = w.config.credentials_provider()
        if creds_provider:
            sdk_headers = creds_provider(w.config)
            if sdk_headers and callable(sdk_headers):
                sdk_headers = sdk_headers()
            if sdk_headers:
                sdk_headers["Content-Type"] = "application/json"
                print(f"[tools] Auth via SDK credential provider ✅  host={host}")
                return host, sdk_headers
        # Last resort: try config.token (PAT / env var token)
        token = w.config.token
        if token:
            print(f"[tools] Auth via SDK token ✅  host={host}")
            return host, {
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            }
    except Exception as e:
        print(f"[tools] SDK auth failed: {e}")

    # ── Layer 3: Explicit environment variables ────────────────────────────
    host  = os.environ.get("DATABRICKS_HOST", "https://adb-7405619257134796.16.azuredatabricks.net").rstrip("/")
    token = os.environ.get("DATABRICKS_TOKEN", "")
    if token:
        print(f"[tools] Auth via env vars ✅  host={host}")
        return host, {
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        }

    raise RuntimeError(
        "No Databricks credentials found. Tried: SparkContext, SDK, env vars.\n"
        "  On cluster: pyspark must be available (it always is on Databricks).\n"
        "  Locally:    set DATABRICKS_HOST + DATABRICKS_TOKEN env vars."
    )


# ── GENIE API — CORRECT 4-STEP FLOW ─────────────────────────────────────────

def _call_genie_space(space_id: str, question: str, max_wait: int = 120) -> str:
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

    Returns the answer as a string, or a GENIE_ERROR string on failure.
    """
    try:
        host, headers = _get_auth()
    except RuntimeError as e:
        return f"GENIE_ERROR: Auth failed: {e}"

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
        return f"GENIE_ERROR: Failed to start conversation: {e}"

    data            = resp.json()
    conversation_id = data.get("conversation_id")
    message_id      = data.get("message_id")

    if not conversation_id or not message_id:
        return f"GENIE_ERROR: Missing ids in start-conversation response: {json.dumps(data)}"

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
            return f"GENIE_ERROR: Polling failed: {e}"

        msg_data = poll_resp.json()
        status   = msg_data.get("status", "UNKNOWN")
        print(f"  [Genie] status={status}")

        if status == "COMPLETED":
            break
        elif status in ("FAILED", "CANCELLED"):
            error_msg = msg_data.get("error", "Unknown Genie error")
            return f"GENIE_ERROR: {status}: {error_msg}"

        time.sleep(3)
    else:
        return "GENIE_ERROR: Timed out waiting for Genie Space response."

    # ── Step 3: Fetch query result from each query attachment ─────────────
    # The message attachments array tells us which attachments have SQL results.
    # We must call a SEPARATE endpoint to get the actual data rows.
    attachments = msg_data.get("attachments", [])
    print(f"  [Genie] {len(attachments)} attachment(s) in message")

    for attachment in attachments:
        attachment_id = attachment.get("attachment_id") or attachment.get("id")

        # ── 3a: Query attachment → call query-result endpoint ─────────────
        if attachment.get("query") is not None and attachment_id:
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
                    if "str" in cell:
                        return cell["str"]
                    if "i64" in cell:
                        return str(cell["i64"])
                    if "f64" in cell:
                        return str(cell["f64"])
                    if "bool" in cell:
                        return str(cell["bool"])
                    if "null" in cell:
                        return "NULL"
                    # Unknown format — stringify whole cell
                    return str(cell)

            # Fallback: try data_array (older API response format)
            data_array = result.get("data_array", [])
            if data_array:
                return str(data_array[0][0])

            # Fallback: check manifest for column names + try other paths
            manifest = stmt_resp.get("manifest", {})
            print(f"  [Genie] manifest schema: {manifest.get('schema', {})}")

        # ── 3b: Text attachment → return the text directly ────────────────
        text_content = attachment.get("text", {}).get("content", "")
        if text_content:
            return text_content.strip()

    # ── Step 4: Last resort — top-level content ───────────────────────────
    content = msg_data.get("content", "")
    if content:
        return content.strip()

    return f"GENIE_ERROR: Could not extract answer. Attachment keys: {[list(a.keys()) for a in attachments]}"


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
