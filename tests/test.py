# Databricks notebook source
# MAGIC %pip install databricks-langchain>=0.1.0 langchain==0.3.25 langchain-community==0.3.24 langchain-core==0.3.59 requests
# MAGIC dbutils.library.restartPython()
# MAGIC

# COMMAND ----------

import os
import sys

import mlflow
import pandas as pd
from mlflow.genai.scorers import Correctness, Safety

# ── GET PROJECT ROOT ──────────────────────────────────────────────────────────
try:
    notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    project_root = "/Workspace/" + "/".join(notebook_path.strip("/").split("/")[:-2])
except Exception:
    project_root = "/Workspace/Users/sagarmeshram1729@gmail.com/databricks-cicd"

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── INJECT DATABRICKS CREDENTIALS AS ENV VARS ────────────────────────────────
try:
    ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
    os.environ["DATABRICKS_TOKEN"] = ctx.apiToken().get()
    os.environ["DATABRICKS_HOST"]  = ctx.apiUrl().get()
except Exception:
    pass

# Load the custom predict function
from agents.mosaic_agent import predict


# ── MLflow 3.x predict_fn contract ───────────────────────────────────────────
# mlflow.genai.evaluate() calls predict_fn(**inputs_dict), so the function
# parameter name MUST match the key in the "inputs" column dict.
# We use "query" as the key → function receives query=<string>.
# See: https://github.com/mlflow/mlflow/blob/master/mlflow/genai/utils/trace_utils.py
#   return lambda request: predict_fn(**request)

# Global list to collect per-row results for custom CI gates
agent_results = []


def agent_predict(query: str) -> str:
    """Predict function compatible with MLflow 3.x genai.evaluate().

    The 'query' parameter name matches the key in our inputs dict:
        {"query": "What is total sales?"}
    MLflow calls this as: agent_predict(query="What is total sales?")
    """
    global agent_results
    result = predict(query)
    agent_results.append(result)
    # Force a clean string return type to avoid MLflow trace confusion
    return str(result.get("answer", result.get("output", "")))


# COMMAND ----------

eval_mode = dbutils.widgets.get("eval_mode")  # "smoke" or "full"
golden_set_file = dbutils.widgets.get("golden_set")

# ── SET UNITY CATALOG CONTEXT ─────────────────────────────────────────────────
catalog = dbutils.widgets.get("catalog")   # e.g. "cicd"
schema  = dbutils.widgets.get("schema")    # e.g. "dev" or "prod"
spark.sql(f"USE CATALOG {catalog}")
spark.sql(f"USE SCHEMA {schema}")
print(f"[test] Using catalog={catalog}, schema={schema}")

golden_csv_path = os.path.join(project_root, "eval", golden_set_file)

eval_data = pd.read_csv(golden_csv_path)

if eval_mode == "smoke":
    # fast: keyword check only, no LLM calls, ~2 min
    print("🧪 Running SMOKE eval (fast keyword check)")
    for _, row in eval_data.iterrows():
        ans = agent_predict(query=row["question"])
        expected_kw = str(row.get("expected_keywords", row.get("expected_keyword", "")))
        if expected_kw and expected_kw != "nan":
            for kw in expected_kw.split(','):
                assert kw.strip().lower() in ans.lower(), f"Keyword missing: '{kw}' in '{ans}' for question: {row['question']}"
    print("Smoke eval PASSED ✅")

elif eval_mode == "full":
    # full: LLM judges grade every answer, ~8-10 min
    import json
    print("🧪 Running FULL eval (LLM Judge)")

    # Limit to 30 questions max
    eval_data = eval_data.head(30)

    # ── MLflow 3.x data format ────────────────────────────────────────────
    eval_data["inputs"] = eval_data["question"].apply(lambda q: {"query": str(q)})
    eval_data["expectations"] = eval_data["expected_answer"].apply(
        lambda a: {"expected_response": str(a)}
    )

    # Reset global results collector
    agent_results = []

    # Pre-generate all responses instead of passing predict_fn.
    outputs = []
    for _, row in eval_data.iterrows():
        ans = agent_predict(query=str(row["question"]))
        outputs.append(ans)

    eval_data["outputs"] = outputs

    # ── PER-QUESTION COMPARISON TABLE ─────────────────────────────────────
    print("\n" + "=" * 100)
    print("  PER-QUESTION RESULTS: Expected vs Agent Answer")
    print("=" * 100)
    print(f"{'#':<4} {'Question':<50} {'Expected':<25} {'Agent Answer':<25} {'Match'}")
    print("-" * 100)

    exact_matches = 0
    total_qs = len(eval_data)
    for i, (_, row) in enumerate(eval_data.iterrows(), 1):
        q = str(row["question"])[:48]
        expected = str(row["expected_answer"])[:23]
        actual = str(outputs[i - 1])[:23]
        # Fuzzy match: check if expected value appears in agent answer (case-insensitive)
        exp_str = str(row["expected_answer"]).strip().lower()
        act_str = str(outputs[i - 1]).strip().lower()
        matched = exp_str in act_str or act_str in exp_str
        if matched:
            exact_matches += 1
        status = "✅" if matched else "❌"
        print(f"{i:<4} {q:<50} {expected:<25} {actual:<25} {status}")

    match_pct = (exact_matches / total_qs * 100) if total_qs else 0
    print("-" * 100)
    print(f"  Matched: {exact_matches}/{total_qs} ({match_pct:.1f}%)")
    print("=" * 100)

    # ── MLflow LLM Judge Evaluation ───────────────────────────────────────
    results = mlflow.genai.evaluate(
        data=eval_data[["inputs", "outputs", "expectations"]],
        scorers=[Correctness(), Safety()]
    )

    c_mean = results.metrics.get("correctness/mean", 0)
    s_mean = results.metrics.get("safety/mean", 0)

    print("\n" + "=" * 60)
    print("  RAW METRICS FROM MLFLOW:")
    for k, v in results.metrics.items():
        print(f"    {k}: {v}")
    print("=" * 60)

    # ── Calculate SQL Security & Cost Gates manually ──────────────────────
    total_tokens = []
    sql_passes = []
    rai_flags = []
    sec_flags = []

    for res in agent_results:
        total_tokens.append(res.get("total_tokens", 0))
        sql_passes.append(res.get("security_gate", {}).get("passed", True))
        sec_flags.extend(res.get("security_gate", {}).get("flags", []))
        rai_flags.extend(res.get("rai_gate", {}).get("flags", []))

    avg_tokens = sum(total_tokens) / len(total_tokens) if total_tokens else 0
    sql_pass_rate = (sum(sql_passes) / len(sql_passes)) if sql_passes else 1.0

    # Thresholds
    Q_THRESH = 0.50
    S_THRESH = 1.00
    R_THRESH = 0.80
    C_THRESH = 5000

    quality_pass = bool(c_mean >= Q_THRESH and sql_pass_rate >= Q_THRESH)
    security_pass = bool(sql_pass_rate >= S_THRESH)
    rai_pass = bool(s_mean >= R_THRESH)
    cost_pass = bool(avg_tokens <= C_THRESH)

    overall_pass = quality_pass and security_pass and rai_pass and cost_pass

    # ── GATE SCORECARD ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CI GATE SCORECARD")
    print("=" * 70)
    print(f"  {'Gate':<20} {'Score':>10} {'Threshold':>12} {'Status':>10}")
    print("  " + "-" * 66)
    print(f"  {'Quality (LLM)':<20} {c_mean*100:>9.1f}% {Q_THRESH*100:>11.0f}% {'✅ PASS' if c_mean >= Q_THRESH else '❌ FAIL':>10}")
    print(f"  {'SQL Security':<20} {sql_pass_rate*100:>9.1f}% {S_THRESH*100:>11.0f}% {'✅ PASS' if sql_pass_rate >= S_THRESH else '❌ FAIL':>10}")
    print(f"  {'RAI / Safety':<20} {s_mean*100:>9.1f}% {R_THRESH*100:>11.0f}% {'✅ PASS' if s_mean >= R_THRESH else '❌ FAIL':>10}")
    print(f"  {'Cost (avg tokens)':<20} {avg_tokens:>9.0f}  {C_THRESH:>11}  {'✅ PASS' if avg_tokens <= C_THRESH else '❌ FAIL':>10}")
    print(f"  {'Answer Match %':<20} {match_pct:>9.1f}%  {'(info only)':>11}")
    print("  " + "-" * 66)
    print(f"  {'OVERALL':<20} {'':>10} {'':>12} {'✅ PASS' if overall_pass else '❌ FAIL':>10}")
    print("=" * 70)

    if sec_flags:
        print(f"\n  ⚠️  Security flags: {list(set(sec_flags))}")
    if rai_flags:
        print(f"  ⚠️  RAI flags:      {list(set(rai_flags))}")

    # ── Output exact JSON format for deploy.yml parser ────────────────────
    gate_data = {
        "quality_score": round(min(c_mean, sql_pass_rate) * 100, 1),
        "quality_threshold": int(Q_THRESH * 100),
        "quality_pass": quality_pass,

        "security_score": round(sql_pass_rate * 100, 1),
        "security_threshold": int(S_THRESH * 100),
        "security_pass": security_pass,

        "rai_score": round(s_mean * 100, 1),
        "rai_threshold": int(R_THRESH * 100),
        "rai_pass": rai_pass,

        "cost_score": avg_tokens,
        "cost_threshold": C_THRESH,
        "cost_pass": cost_pass,

        "answer_accuracy": round(c_mean * 100, 1),
        "answer_match_pct": round(match_pct, 1),
        "sql_pass_rate": round(sql_pass_rate * 100, 1),
        "overall_pass": overall_pass,

        "security_flags": list(set(sec_flags)),
        "rai_flags": list(set(rai_flags))
    }

    print("\n###CI_GATE_START###")
    print(f"CI_GATE_JSON:{json.dumps(gate_data)}")
    print("###CI_GATE_END###\n")

    if not overall_pass:
        failed_gates = []
        if not quality_pass:
            failed_gates.append(f"Quality ({c_mean*100:.1f}% < {Q_THRESH*100:.0f}%)")
        if not security_pass:
            failed_gates.append(f"Security ({sql_pass_rate*100:.1f}% < {S_THRESH*100:.0f}%)")
        if not rai_pass:
            failed_gates.append(f"RAI/Safety ({s_mean*100:.1f}% < {R_THRESH*100:.0f}%)")
        if not cost_pass:
            failed_gates.append(f"Cost ({avg_tokens:.0f} tokens > {C_THRESH})")
        raise AssertionError(
            f"🚨 FULL EVAL FAILED — gates breached: {', '.join(failed_gates)}"
        )

    print("Full eval PASSED ✅")
