"""
get_prod_version.py
────────────────────────────────────────────────────────────────────────────
Called by GitHub Actions BEFORE registering a new model version.
Reads the current @PROD alias version from Unity Catalog so the pipeline
can roll back to it if the new registration step fails.

Prints ONLY the version number (or "none") to stdout — captured by:
    echo "prod_version=$(python scripts/get_prod_version.py)" >> $GITHUB_OUTPUT
"""

import os
import sys

UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "cicd.gold.mosaic_nl_sql_agent")

try:
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    # get_model_version_by_alias is available in mlflow >= 2.3
    mv = client.get_model_version_by_alias(UC_MODEL_NAME, "PROD")
    print(mv.version)

except Exception as e:
    err = str(e).lower()
    if "resource does not exist" in err or "does not exist" in err or "not found" in err:
        # No @PROD alias yet — first deploy, rollback is a safe no-op
        print("none")
    else:
        # Unexpected error — surface but don't block the pipeline
        print(f"error: {e}", file=sys.stderr)
        print("none")
