# Databricks notebook source
# MAGIC %pip install mlflow langchain==0.2.16 langchain-community==0.2.17 langchain-core==0.2.41 databricks-sdk requests
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

"""
tests/test.py — Quality Gate for CI/CD Pipeline
Runs golden_set against the LangChain Multi–Genie-Space agent.
If accuracy < 80%, exits with sys.exit(1) to block deployment.
"""

import sys
import os
import importlib.util

# ── GET PROJECT ROOT ──────────────────────────────────────────────────────────
try:
    notebook_path = (
        dbutils.notebook.entry_point
        .getDbutils().notebook().getContext()
        .notebookPath().get()
    )
    project_root = "/Workspace/" + "/".join(notebook_path.strip("/").split("/")[:-2])
except Exception:
    project_root = "/Workspace/Users/sagarmeshram1729@gmail.com/databricks-cicd"

print(f"Project root: {project_root}")

# Add project root to sys.path so `from agents.config import ...` works
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── LOAD AGENT MODULES (config → tools → agent) ──────────────────────────────
import types
agents_dir = os.path.join(project_root, "agents")

# Register 'agents' as a package so `from agents.config` works
if "agents" not in sys.modules:
    agents_pkg = types.ModuleType("agents")
    agents_pkg.__path__ = [agents_dir]
    sys.modules["agents"] = agents_pkg

# 1. Load config
config_path = os.path.join(agents_dir, "config.py")
config_spec = importlib.util.spec_from_file_location("agents.config", config_path)
config_mod = importlib.util.module_from_spec(config_spec)
sys.modules["agents.config"] = config_mod
config_spec.loader.exec_module(config_mod)

# 2. Load tools
tools_path = os.path.join(agents_dir, "tools.py")
tools_spec = importlib.util.spec_from_file_location("agents.tools", tools_path)
tools_mod = importlib.util.module_from_spec(tools_spec)
sys.modules["agents.tools"] = tools_mod
tools_spec.loader.exec_module(tools_mod)

# 3. Load main agent
agent_path = os.path.join(agents_dir, "mosaic_agent.py")
spec = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
mosaic_agent = importlib.util.module_from_spec(spec)
sys.modules["mosaic_agent"] = mosaic_agent
spec.loader.exec_module(mosaic_agent)

predict = mosaic_agent.predict

# COMMAND ----------

# ── GUARDRAIL VALIDATION ──────────────────────────────────────────────────────
validate_query_safety  = mosaic_agent.validate_query_safety
validate_output_safety = mosaic_agent.validate_output_safety

# COMMAND ----------

QUALITY_GATE_THRESHOLD = 80.0  # Pipeline fails if golden accuracy < this

def run_evaluation(dataset_name: str, df):
    """
    Runs predict() on every row, compares answer to expected, prints results
    and returns accuracy %. Also validates guardrails on every query.
    """
    rows      = df.collect()
    total     = len(rows)
    correct   = 0
    guardrail_fails = 0

    print(f"\n{'='*65}")
    print(f"  EVALUATING: {dataset_name}  ({total} questions)")
    print(f"{'='*65}\n")

    for i, row in enumerate(rows, 1):
        question = row.question
        expected = str(row.expected_answer).strip()

        result     = predict(question)
        agent_ans  = result["answer"]
        agent_str  = str(agent_ans).strip()
        source     = result.get("source_tool", "N/A")

        # ── Guardrail checks ──
        sql_safe    = result.get("sql_safe", True)
        output_safe = result.get("output_safe", True)
        if not sql_safe or not output_safe:
            guardrail_fails += 1

        # ── Tolerant accuracy match ──
        try:
            match = abs(float(agent_str) - float(expected)) < 0.01
        except (ValueError, TypeError):
            agent_norm    = agent_str.split(" ")[0].lower().strip()
            expected_norm = expected.split(" ")[0].lower().strip()
            match = agent_norm == expected_norm

        status = "✅ PASS" if match else "❌ FAIL"
        if match:
            correct += 1

        # ── Running accuracy ──
        running_acc = (correct / i) * 100

        guardrail_status = "" if (sql_safe and output_safe) else " ⚠️ GUARDRAIL"
        print(f"[{i:02d}/{total}] {status}{guardrail_status}")
        print(f"  Q       : {question}")
        print(f"  Expected: {expected}")
        print(f"  Agent   : {agent_str}")
        print(f"  Tool    : {source}")
        print(f"  📈 Running Accuracy: {correct}/{i} → {running_acc:.1f}%")
        print()

    accuracy = (correct / total) * 100
    print(f"{'='*65}")
    print(f"  {dataset_name} ACCURACY     : {correct}/{total}  →  {accuracy:.1f}%")
    print(f"  {dataset_name} GUARDRAIL    : {guardrail_fails} failures")
    print(f"{'='*65}\n")
    return accuracy

# COMMAND ----------

# ── GOLDEN SET ────────────────────────────────────────────────────────────────
golden_df  = spark.table("cicd.gold.golden_set")
golden_acc = run_evaluation("GOLDEN SET", golden_df)

# COMMAND ----------

# ── FINAL SUMMARY + QUALITY GATE ─────────────────────────────────────────────
print("=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Agent Type          : LangChain ReAct (Multi–Genie-Space)")
print(f"  Golden Set Accuracy : {golden_acc:.1f}%")
print(f"  Quality Gate        : {'PASS ✅' if golden_acc >= QUALITY_GATE_THRESHOLD else 'FAIL 🚨'}")
print(f"  Threshold           : {QUALITY_GATE_THRESHOLD}%")
print("=" * 65)

# ── HARD GATE: block CI/CD deployment if accuracy is too low ──────────────────
if golden_acc < QUALITY_GATE_THRESHOLD:
    print(f"\n🚨 Quality Gate FAILED! {golden_acc:.1f}% < {QUALITY_GATE_THRESHOLD}%")
    print("   Blocking deployment. Fix the prompt or agent logic.")
    import traceback
    traceback.print_exc()
    sys.exit(1)
else:
    print(f"\n✅ Quality Gate PASSED. Safe to deploy.")
