# Databricks notebook source
# MAGIC %pip install mlflow[databricks]>=3.1.0,<3.10.0 langchain==0.3.25 langchain-databricks==0.1.1 langchain-community==0.3.24 langchain-core==0.3.59 requests
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

# ── INJECT DATABRICKS CREDENTIALS AS ENV VARS ────────────────────────────────
try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
    os.environ["DATABRICKS_HOST"]  = ctx.apiUrl().get()
    print(f"[deploy] ✅ Injected credentials from dbutils")
except Exception as e:
    print(f"[deploy] ⚠️  Could not inject credentials: {e}")

# Add project root to sys.path so `from agents.config import ...` works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# COMMAND ----------

# ── STEP 1: DEFINE PYFUNC WRAPPER ─────────────────────────────────────────────
# MLflow PythonModel wraps our LangChain agent so it can be:
#   - Stored + versioned in Unity Catalog
#   - Loaded and called from any notebook in the workspace
# ─────────────────────────────────────────────────────────────────────────────

class MosaicLangChainAgent(mlflow.pyfunc.PythonModel):
    """
    MLflow PythonModel wrapper for the LangChain Multi–Genie-Space Agent.

    Input : pandas DataFrame with column 'question'
    Output: list of dicts with keys: question, answer, source_tool, prompt_version, model, etc.
    """

    def load_context(self, context):
        """Called once when the model is loaded — sets up the agent module."""
        import sys, os, types, importlib.util

        # Add agents dir parent to path so relative imports work
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

# ── STEP 2: LOG & REGISTER TO UNITY CATALOG ──────────────────────────────────
import re, subprocess

UC_MODEL_NAME = "cicd.prod.mosaic_nl_sql_agent"
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment("/Users/sagarmeshram1729@gmail.com/mosaic-agent-deploy")

# Read prompt version for tagging
prompt_path = os.path.join(project_root, "prompts", "system_prompt.txt")
with open(prompt_path) as f:
    first_line = f.readline()
prompt_version = re.search(r"PROMPT_VERSION:\s*([\w\.]+)", first_line)
prompt_version = prompt_version.group(1) if prompt_version else "unknown"

# ── GIT SHA — for traceability ────────────────────────────────────────────────
GIT_SHA = os.environ.get("GITHUB_SHA") or os.environ.get("GIT_SHA")
if not GIT_SHA:
    try:
        GIT_SHA = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        GIT_SHA = "unknown"
print(f"[deploy] Git SHA: {GIT_SHA}")

with mlflow.start_run(run_name=f"deploy_{prompt_version}_{str(GIT_SHA)[:8]}") as run:
    mlflow.set_tag("prompt_version",   prompt_version)
    mlflow.set_tag("git_sha",          GIT_SHA)
    mlflow.set_tag("model_endpoint",   "databricks-claude-3-7-sonnet")
    mlflow.set_tag("deployment_mode",  "unity_catalog_direct")
    mlflow.set_tag("agent_type",       "langchain_react_multi_genie")

    model_info = mlflow.pyfunc.log_model(
        artifact_path         = "mosaic_agent",
        python_model          = MosaicLangChainAgent(),
        artifacts             = {
            "agents_dir":  os.path.join(project_root, "agents"),
            "prompts_dir": os.path.join(project_root, "prompts"),
        },
        registered_model_name = UC_MODEL_NAME,
        pip_requirements      = [
            "mlflow[databricks]>=3.1.0,<3.10.0",
            "langchain==0.3.25",
            "langchain-databricks==0.1.1",
            "langchain-community==0.3.24",
            "langchain-core==0.3.59",
            "requests",
        ],
        input_example         = {"question": "What is the total profit in July?"},
    )

print(f"✅ Registered : {UC_MODEL_NAME}")
print(f"✅ Run ID     : {run.info.run_id}")
print(f"✅ Prompt     : {prompt_version}")
print(f"✅ Git SHA    : {GIT_SHA}")

# COMMAND ----------

# ── STEP 3: LOAD FROM UC & INVOKE DIRECTLY ───────────────────────────────────
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
    {"question": "What is the maximum quantity on hand?"},
    {"question": "How many unique stock items are there?"},
])

results = loaded_model.predict(questions)

print("=" * 65)
print(f"  INFERENCE RESULTS  (model v{latest_version} | prompt {prompt_version})")
print("=" * 65)
for r in results:
    print(f"\nQ     : {r['question']}")
    print(f"Tool  : {r.get('source_tool', 'N/A')}")
    print(f"Answer: {r['answer']}")
print("=" * 65)
