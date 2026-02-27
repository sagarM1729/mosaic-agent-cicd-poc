# Databricks notebook source
# MAGIC %pip install mlflow[databricks]>=3.1 langchain>=0.3.0 langchain-databricks>=0.1.0 langchain-community>=0.3.0 langchain-core>=0.3.0 databricks-sdk>=0.20.0 databricks-agents>=1.0.0 requests
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

"""
register_model.py — Called by CI/CD pipeline AFTER tests pass.
Registers the LangChain Multi–Genie-Space agent as an MLflow pyfunc model
in Unity Catalog. Tags each version with the Git SHA for full traceability.
"""

import argparse
import importlib.util
import os
import sys

import mlflow
from databricks import agents  # Fix 6: agents.deploy() for serving + inference tables

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────
try:
    notebook_path = (
        dbutils.notebook.entry_point
        .getDbutils().notebook().getContext()
        .notebookPath().get()
    )
    project_root = "/Workspace/" + "/".join(notebook_path.strip("/").split("/")[:-2])
except Exception:
    project_root = "/Workspace/Users/sagarmeshram1729@gmail.com/databricks-cicd"

# ── INJECT DATABRICKS CREDENTIALS AS ENV VARS ────────────────────────────────
try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
    os.environ["DATABRICKS_HOST"]  = ctx.apiUrl().get()
    print("[register] ✅ Injected credentials from dbutils")
except Exception as e:
    print(f"[register] ⚠️  Could not inject credentials: {e}")

# Add project root to sys.path so `from agents.config import ...` works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── PARSE CLI ARGS (used when called from GitHub Actions) ─────────────────────
try:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="cicd.prod.mosaic_nl_sql_agent")
    parser.add_argument("--git-sha", default="local")
    args = parser.parse_args()
    UC_MODEL_NAME = args.model_name
    GIT_SHA       = args.git_sha
except SystemExit:
    UC_MODEL_NAME = "cicd.prod.mosaic_nl_sql_agent"
    GIT_SHA       = "notebook-run"

print(f"Model name : {UC_MODEL_NAME}")
print(f"Git SHA    : {GIT_SHA}")
print(f"Project    : {project_root}")

# COMMAND ----------

# ── DEFINE PYFUNC WRAPPER ─────────────────────────────────────────────────────
class MosaicLangChainAgent(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for the LangChain Multi–Genie-Space Agent.
    """
    def load_context(self, context):
        import importlib.util
        import os
        import sys
        import types

        agents_dir = context.artifacts["agents_dir"]
        project_dir = os.path.dirname(agents_dir)
        if project_dir not in sys.path:
            sys.path.insert(0, project_dir)

        # Register 'agents' as a package so `from agents.config` works
        if "agents" not in sys.modules:
            agents_pkg = types.ModuleType("agents")
            agents_pkg.__path__ = [agents_dir]
            sys.modules["agents"] = agents_pkg

        # 1. Load config first (no dependencies)
        config_path = os.path.join(agents_dir, "config.py")
        config_spec = importlib.util.spec_from_file_location("agents.config", config_path)
        config_mod = importlib.util.module_from_spec(config_spec)
        sys.modules["agents.config"] = config_mod
        config_spec.loader.exec_module(config_mod)

        # 2. Load tools (depends on config)
        tools_path = os.path.join(agents_dir, "tools.py")
        tools_spec = importlib.util.spec_from_file_location("agents.tools", tools_path)
        tools_mod = importlib.util.module_from_spec(tools_spec)
        sys.modules["agents.tools"] = tools_mod
        tools_spec.loader.exec_module(tools_mod)

        # 3. Load main agent (depends on config + tools)
        agent_path = os.path.join(agents_dir, "mosaic_agent.py")
        spec = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
        self.agent = importlib.util.module_from_spec(spec)
        sys.modules["mosaic_agent"] = self.agent
        spec.loader.exec_module(self.agent)

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
    mlflow.set_tag("agent_type", "langchain_react_multi_genie")

    model_info = mlflow.pyfunc.log_model(
        artifact_path         = "mosaic_agent",
        python_model          = MosaicLangChainAgent(),
        artifacts             = {
            "agents_dir":  os.path.join(project_root, "agents"),
            "prompts_dir": os.path.join(project_root, "prompts"),
        },
        registered_model_name = UC_MODEL_NAME,
        pip_requirements      = [
            "mlflow[databricks]>=3.1",
            "databricks-sdk>=0.20.0",
            "databricks-agents>=1.0.0",
            "langchain>=0.3.0",
            "langchain-databricks>=0.1.0",
            "langchain-community>=0.3.0",
            "requests",
        ],
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

# COMMAND ----------

# ── FIX 6: DEPLOY WITH agents.deploy() ───────────────────────────────────────
# Replaces manual serving_endpoints.create() and auto-creates inference tables
# that log every production request/response — free, no extra code needed.
try:
    deployment = agents.deploy(
        model_name=UC_MODEL_NAME,
        model_version=latest,
        scale_to_zero=True,  # costs $0 when no traffic
    )
    print(f"✅ Deployed to endpoint: {deployment.endpoint_name}")
    print(f"✅ Inference table:      {deployment.inference_table_name}")
    print("   ^ Auto-created Delta table logging all prod requests/responses")
except Exception as e:
    # Don't fail the pipeline if agents.deploy() isn't available (e.g., trial limits)
    print(f"⚠️  agents.deploy() skipped: {e}")
    print("   Endpoint may need manual creation or already exists.")
