"""
rollback_prod.py
────────────────────────────────────────────────────────────────────────────
Called by GitHub Actions ONLY when the registration step fails.
Re-points the @PROD alias to the last known-good version so that
production traffic is never broken by a failed deployment.

Usage:
    python scripts/rollback_prod.py --version 3
    python scripts/rollback_prod.py --version none   # no-op (first deploy failed)
"""

import argparse
import os
import sys

UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "cicd.gold.mosaic_nl_sql_agent")

parser = argparse.ArgumentParser()
parser.add_argument("--version", required=True,
                    help="The UC model version number to restore @PROD to, or 'none'.")
args = parser.parse_args()

if args.version == "none":
    print("ℹ️  No previous @PROD version recorded — nothing to roll back to.")
    print("   (This was likely the first-ever deployment attempt. @PROD was never set.)")
    sys.exit(0)

try:
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    client.set_registered_model_alias(UC_MODEL_NAME, "PROD", args.version)
    print(f"✅ ROLLBACK COMPLETE: @PROD → version {args.version}")
    print(f"   Model: {UC_MODEL_NAME}")
    print("   Production traffic restored to the last known-good version.")

except Exception as e:
    # Rollback itself failed — alert loudly but don't mask the original failure
    print(f"🚨 ROLLBACK FAILED: could not restore @PROD to version {args.version}: {e}",
          file=sys.stderr)
    sys.exit(1)
