# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

"""
register_model.py — Called by CI/CD pipeline AFTER tests pass.
Registers the agent as an MLflow pyfunc model in Unity Catalog.
Tags each version with the Git SHA for full traceability.
"""

import sys, os, importlib.util, argparse, mlflow

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────
try:
    notebook_path = (
        dbutils.notebook.entry_point
        .getDbutils().notebook().getContext()
        .notebookPath().get()
    )
    project_root = "/Workspace/" + "/".join(notebook_path.strip("/").split("/")[:-1])
except Exception:
    project_root = "/Workspace/Users/sagarmeshram1729@gmail.com/databricks-cicd"

# ── PARSE CLI ARGS (used when called from GitHub Actions) ─────────────────────
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="cicd.gold.mosaic_nl_sql_agent")
    parser.add_argument("--git-sha", default="local")
    args = parser.parse_args()
    UC_MODEL_NAME = args.model_name
    GIT_SHA       = args.git_sha
except SystemExit:
    # Running as a Databricks notebook (argparse fails on notebook args)
    UC_MODEL_NAME = "cicd.gold.mosaic_nl_sql_agent"
    GIT_SHA       = "notebook-run"

print(f"Model name : {UC_MODEL_NAME}")
print(f"Git SHA    : {GIT_SHA}")
print(f"Project    : {project_root}")

# COMMAND ----------

# ── LOAD AGENT MODULE ─────────────────────────────────────────────────────────
agent_path = os.path.join(project_root, "agents", "mosaic_agent.py")
spec        = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
mosaic_agent = importlib.util.module_from_spec(spec)
sys.modules["mosaic_agent"] = mosaic_agent
spec.loader.exec_module(mosaic_agent)

# ── DEFINE PYFUNC WRAPPER ─────────────────────────────────────────────────────
class MosaicNLSQLAgent(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import sys, os, importlib.util
        from pyspark.sql import SparkSession

        agent_path = os.path.join(context.artifacts["agents_dir"], "mosaic_agent.py")
        spec        = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
        self.agent  = importlib.util.module_from_spec(spec)
        sys.modules["mosaic_agent"] = self.agent
        spec.loader.exec_module(self.agent)
        self.agent.spark = SparkSession.getActiveSession()

    def predict(self, context, model_input):
        results = []
        for _, row in model_input.iterrows():
            results.append(self.agent.predict(row["question"]))
        return results

# COMMAND ----------

# ── REGISTER TO UNITY CATALOG ────────────────────────────────────────────────
import re

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/sagarmeshram1729@gmail.com/mosaic-agent-deploy")

# Read prompt version for tagging
prompt_path = os.path.join(project_root, "prompts", "system_prompt.txt")
with open(prompt_path) as f:
    first_line = f.readline()
version_match = re.search(r"PROMPT_VERSION:\s*([\w\.]+)", first_line)
prompt_version = version_match.group(1) if version_match else "unknown"

with mlflow.start_run(run_name=f"cicd_{GIT_SHA[:8]}") as run:
    mlflow.set_tag("prompt_version", prompt_version)
    mlflow.set_tag("git_sha", GIT_SHA)
    mlflow.set_tag("model_endpoint", "databricks-claude-3-7-sonnet")

    model_info = mlflow.pyfunc.log_model(
        artifact_path         = "mosaic_agent",
        python_model          = MosaicNLSQLAgent(),
        artifacts             = {
            "agents_dir":  os.path.join(project_root, "agents"),
            "prompts_dir": os.path.join(project_root, "prompts"),
        },
        registered_model_name = UC_MODEL_NAME,
        pip_requirements      = ["mlflow", "databricks-sdk"],
        input_example         = {"question": "What is the total profit in July?"},
    )

print(f"✅ Registered : {UC_MODEL_NAME}")
print(f"✅ Run ID     : {run.info.run_id}")
print(f"✅ Git SHA    : {GIT_SHA}")
print(f"✅ Prompt     : {prompt_version}")

# COMMAND ----------

# ── SET @PROD ALIAS (Rollback-friendly) ──────────────────────────────────────
client   = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
latest   = str(max(int(v.version) for v in versions))

client.set_registered_model_alias(UC_MODEL_NAME, "PROD", latest)
print(f"✅ Alias @PROD → version {latest}")
print(f"   Load via: mlflow.pyfunc.load_model('models:/{UC_MODEL_NAME}@PROD')")
