# Databricks notebook source
# MAGIC %pip install mlflow
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

"""
tests/test.py — Quality Gate for CI/CD Pipeline
Runs smoke_set (5 Q) + golden_set (30 Q) against the agent.
If golden_set accuracy < 80%, exits with sys.exit(1) to block deployment.
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
    # notebook is at /Users/.../databricks-cicd/tests/test
    # project root is one level up from tests/
    project_root = "/Workspace/" + "/".join(notebook_path.strip("/").split("/")[:-2])
except Exception:
    project_root = "/Workspace/Users/sagarmeshram1729@gmail.com/databricks-cicd"

print(f"Project root: {project_root}")

# ── LOAD AGENT MODULE ─────────────────────────────────────────────────────────
agent_path = os.path.join(project_root, "agents", "mosaic_agent.py")
spec        = importlib.util.spec_from_file_location("mosaic_agent", agent_path)
mosaic_agent = importlib.util.module_from_spec(spec)
sys.modules["mosaic_agent"] = mosaic_agent
spec.loader.exec_module(mosaic_agent)

# Inject Databricks spark session into the agent module
mosaic_agent.spark = spark

predict = mosaic_agent.predict

# COMMAND ----------

# ── GUARDRAIL VALIDATION ──────────────────────────────────────────────────────
# Reuse the exact functions from the agent — single source of truth.
# Tests exercise the same guardrail logic that runs in production.
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
        agent_sql  = result["sql"]

        # ── Guardrail checks ──
        sql_safe    = validate_query_safety(agent_sql)
        output_safe = validate_output_safety(agent_str)
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

        guardrail_status = "" if (sql_safe and output_safe) else " ⚠️ GUARDRAIL"
        print(f"[{i:02d}/{total}] {status}{guardrail_status}")
        print(f"  Q       : {question}")
        print(f"  Expected: {expected}")
        print(f"  Agent   : {agent_str}")
        print(f"  SQL     : {agent_sql}")
        print()

    accuracy = (correct / total) * 100
    print(f"{'='*65}")
    print(f"  {dataset_name} ACCURACY     : {correct}/{total}  →  {accuracy:.1f}%")
    print(f"  {dataset_name} GUARDRAIL    : {guardrail_fails} failures")
    print(f"{'='*65}\n")
    return accuracy

# COMMAND ----------

# ── SMOKE TEST ────────────────────────────────────────────────────────────────
smoke_df   = spark.table("cicd.gold.smoke_set")
smoke_acc  = run_evaluation("SMOKE SET", smoke_df)

# COMMAND ----------

# ── GOLDEN SET ────────────────────────────────────────────────────────────────
golden_df  = spark.table("cicd.gold.golden_set")
golden_acc = run_evaluation("GOLDEN SET", golden_df)

# COMMAND ----------

# ── FINAL SUMMARY + QUALITY GATE ─────────────────────────────────────────────
print("=" * 65)
print("  FINAL SUMMARY")
print("=" * 65)
print(f"  Smoke  Set Accuracy : {smoke_acc:.1f}%")
print(f"  Golden Set Accuracy : {golden_acc:.1f}%")
print(f"  Quality Gate        : {'PASS ✅' if golden_acc >= QUALITY_GATE_THRESHOLD else 'FAIL 🚨'}")
print(f"  Threshold           : {QUALITY_GATE_THRESHOLD}%")
print("=" * 65)

# ── HARD GATE: block CI/CD deployment if accuracy is too low ──────────────────
if golden_acc < QUALITY_GATE_THRESHOLD:
    print(f"\n🚨 Quality Gate FAILED! {golden_acc:.1f}% < {QUALITY_GATE_THRESHOLD}%")
    print("   Blocking deployment. Fix the prompt or agent logic.")
    sys.exit(1)
else:
    print(f"\n✅ Quality Gate PASSED. Safe to deploy.")
