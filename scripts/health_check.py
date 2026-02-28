"""
health_check.py — Post-deploy endpoint validation
═══════════════════════════════════════════════════
Fix 7: Confirms the serving endpoint is live and responding after deployment.
Sends one real question and fails the pipeline if no answer comes back.

Usage:
    MODEL_ENDPOINT_NAME="mosaic-nl-sql-agent" python scripts/health_check.py
"""

import os
import time

from databricks.sdk import WorkspaceClient

# ── CONFIG ────────────────────────────────────────────────────────────────────
endpoint_name = os.environ.get("MODEL_ENDPOINT_NAME", "mosaic-nl-sql-agent")
MAX_WAIT_SECONDS = 300  # 5 minutes max to wait for endpoint readiness

w = WorkspaceClient()

# ── Wait for endpoint to finish updating (max 5 min) ─────────────────────────
print(f"🔍 Checking endpoint: {endpoint_name}")
state = "UNKNOWN"
for attempt in range(30):
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        state = endpoint.state.config_update.value
        if state == "NOT_UPDATING":
            print(f"✅ Endpoint is ready (state={state})")
            break
        print(f"⏳ Waiting for endpoint... ({state}) [{attempt+1}/30]")
    except Exception as e:
        print(f"⏳ Endpoint not found yet, retrying... [{attempt+1}/30]: {e}")
    time.sleep(10)
else:
    raise Exception(
        f"❌ Endpoint '{endpoint_name}' still updating after {MAX_WAIT_SECONDS}s. "
        f"Last state: {state}"
    )

# ── Send one real test question ───────────────────────────────────────────────
print("🧪 Sending health check query...")
try:
    response = w.serving_endpoints.query(
        name=endpoint_name,
        dataframe_records=[{
            "messages": [{
                "role": "user",
                "content": "What is total revenue this month?"
            }]
        }]
    )
    assert response.predictions, "Health check FAILED — empty response!"
    print("✅ Health check PASSED — endpoint is live and responding")
    print(f"   Response preview: {str(response.predictions)[:200]}")
except Exception as e:
    raise Exception(
        f"❌ Health check FAILED — endpoint '{endpoint_name}' returned error: {e}"
    )
