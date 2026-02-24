"""
get_prod_version.py
────────────────────────────────────────────────────────────────────────────
Called by GitHub Actions BEFORE registering a new model version.
Reads the current @PROD alias version from Unity Catalog so the pipeline
can roll back to it if the new registration step fails.

Prints ONLY the version number (or "none") to stdout — captured by:
    echo "prod_version=$(python scripts/get_prod_version.py)" >> $GITHUB_OUTPUT
"""

import os, sys

UC_MODEL_NAME = os.environ.get("UC_MODEL_NAME", "cicd.gold.mosaic_nl_sql_agent")

try:
    import mlflow
    mlflow.set_registry_uri("databricks-uc")
    client = mlflow.MlflowClient()

    alias_obj = client.get_registered_model_alias(UC_MODEL_NAME, "PROD")
    # alias_obj.version is a string like "3"
    print(alias_obj.version)

except mlflow.exceptions.RestException as e:
    # RESOURCE_DOES_NOT_EXIST → no @PROD alias set yet (first deploy is safe)
    if "RESOURCE_DOES_NOT_EXIST" in str(e) or "does not exist" in str(e).lower():
        print("none")
    else:
        # Unexpected API error — surface it but don't block the pipeline
        print(f"error: {e}", file=sys.stderr)
        print("none")   # treat as no previous version; rollback will be a no-op

except Exception as e:
    print(f"error: {e}", file=sys.stderr)
    print("none")
