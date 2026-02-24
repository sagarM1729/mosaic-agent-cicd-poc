# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

import sys, os, importlib.util, mlflow, pandas as pd

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

print(f"Project root: {project_root}")

# COMMAND ----------

# ── STEP 1: DEFINE PYFUNC WRAPPER ─────────────────────────────────────────────
# MLflow PythonModel wraps our agent so it can be:
#   - Stored + versioned in Unity Catalog
#   - Loaded and called from any notebook in the workspace
# ─────────────────────────────────────────────────────────────────────────────

class MosaicNLSQLAgent(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for the WWI NL-to-SQL agent.

    Input : pandas DataFrame with column 'question'
    Output: list of dicts with keys: question, sql, answer, prompt_version, model
    """

    def load_context(self, context):
        """Called once when the model is loaded — sets up the agent module."""
        import sys, os, importlib.util
        from pyspark.sql import SparkSession

        agent_path = os.path.join(context.artifacts["agents_dir"], "mosaic_agent.py")
        spec        = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
        self.agent  = importlib.util.module_from_spec(spec)
        sys.modules["mosaic_agent"] = self.agent
        spec.loader.exec_module(self.agent)

        # Inject the active Spark session
        self.agent.spark = SparkSession.getActiveSession()

    def predict(self, context, model_input):
        results = []
        for _, row in model_input.iterrows():
            results.append(self.agent.predict(row["question"]))
        return results


# COMMAND ----------

# ── STEP 2: LOG & REGISTER TO UNITY CATALOG ──────────────────────────────────
# Packages the agent + artifacts into MLflow and registers it in UC.
# Each run creates a new version — full version history is kept.
# ─────────────────────────────────────────────────────────────────────────────
import re

UC_MODEL_NAME = "cicd.gold.mosaic_nl_sql_agent"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/sagarmeshram1729@gmail.com/mosaic-agent-deploy")

# Read prompt version for tagging
prompt_path = os.path.join(project_root, "prompts", "system_prompt.txt")
with open(prompt_path) as f:
    first_line = f.readline()
prompt_version = re.search(r"PROMPT_VERSION:\s*([\w\.]+)", first_line)
prompt_version = prompt_version.group(1) if prompt_version else "unknown"

with mlflow.start_run(run_name=f"deploy_{prompt_version}") as run:
    mlflow.set_tag("prompt_version", prompt_version)
    mlflow.set_tag("model_endpoint", "databricks-claude-3-7-sonnet")
    mlflow.set_tag("deployment_mode", "unity_catalog_direct")

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
print(f"✅ Prompt     : {prompt_version}")

# COMMAND ----------

# ── STEP 3: LOAD FROM UC & INVOKE DIRECTLY ───────────────────────────────────
# Load the latest registered version from Unity Catalog.
# No serving endpoint needed — $0 cost, instant, works on any workspace tier.
# ─────────────────────────────────────────────────────────────────────────────

# Get latest version
client   = mlflow.MlflowClient()
versions = client.search_model_versions(f"name='{UC_MODEL_NAME}'")
latest_version = str(max(int(v.version) for v in versions))

model_uri    = f"models:/{UC_MODEL_NAME}/{latest_version}"
loaded_model = mlflow.pyfunc.load_model(model_uri)

print(f"✅ Loaded : {model_uri}")
print(f"   Prompt : {prompt_version}")
print()

# ── RUN INFERENCE ─────────────────────────────────────────────────────────────
questions = pd.DataFrame([
    {"question": "What is the total profit in July?"},
    {"question": "What is the total sales quantity in July?"},
    {"question": "What is the average tax rate in July?"},
    {"question": "What is the total number of invoices in July?"},
    {"question": "What is the most common package type sold in July?"},
])

results = loaded_model.predict(questions)

print("=" * 60)
print(f"  INFERENCE RESULTS  (model v{latest_version} | prompt {prompt_version})")
print("=" * 60)
for r in results:
    print(f"\nQ  : {r['question']}")
    print(f"SQL: {r['sql'].strip()}")
    print(f"A  : {r['answer']}")
print("=" * 60)
