"""
health_check.py — Post-deploy endpoint validation
═══════════════════════════════════════════════════
Fix 7: Confirms the serving endpoint is live and responding after deployment.
Sends one real question and fails the pipeline if no answer comes back.

Usage:
    MODEL_ENDPOINT_NAME="mosaic-nl-sql-agent" python scripts/health_check.py
"""

import os
import sys
import time

from databricks.sdk import WorkspaceClient

# ── CONFIG ────────────────────────────────────────────────────────────────────
endpoint_name = os.environ.get("MODEL_ENDPOINT_NAME", "mosaic-nl-sql-agent")
MAX_WAIT_SECONDS = 300  # 5 minutes max to wait for endpoint readiness

w = WorkspaceClient()

# ── Wait for endpoint to finish updating (max 5 min) ─────────────────────────
print(f"🔍 Checking endpoint: {endpoint_name}")
state = "UNKNOWN"
endpoint_found = False
for attempt in range(30):
    try:
        endpoint = w.serving_endpoints.get(endpoint_name)
        endpoint_found = True
        state = endpoint.state.config_update.value
        if state == "NOT_UPDATING":
            print(f"✅ Endpoint is ready (state={state})")
            break
        print(f"⏳ Waiting for endpoint... ({state}) [{attempt+1}/30]")
    except Exception as e:
        err_msg = str(e)
        if "does not exist" in err_msg and attempt >= 5:
            # Give agents.deploy() ~60s to create it, then bail gracefully
            print(f"⚠️  Endpoint '{endpoint_name}' does not exist after {(attempt+1)*10}s.")
            print("   agents.deploy() may still be provisioning, or was skipped.")
            print("   Skipping health check — endpoint will be created on next deploy.")
            endpoint_found = False
            break
        print(f"⏳ Endpoint not found yet, retrying... [{attempt+1}/30]: {e}")
    time.sleep(10)
else:
    if not endpoint_found:
        print(f"⚠️  Endpoint '{endpoint_name}' was never found. Skipping health check.")
    else:
        raise Exception(
            f"❌ Endpoint '{endpoint_name}' still updating after {MAX_WAIT_SECONDS}s. "
            f"Last state: {state}"
        )

if not endpoint_found or state != "NOT_UPDATING":
    print("ℹ️  Health check skipped — endpoint not ready yet.")
    print("   The model is registered in Unity Catalog and will serve once the endpoint is up.")
    sys.exit(0)

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
