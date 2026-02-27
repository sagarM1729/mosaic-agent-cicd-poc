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


def agent_predict(inputs):
    # mlflow.genai.evaluate passes inputs as dict
    question = inputs.get("query", inputs.get("question", ""))
    result = predict(question)
    # Return string answer for MLflow evaluator
    return result["answer"]

# COMMAND ----------

eval_mode = dbutils.widgets.get("eval_mode")  # "smoke" or "full"
golden_set_file = dbutils.widgets.get("golden_set")
golden_csv_path = os.path.join(project_root, "eval", golden_set_file)

eval_data = pd.read_csv(golden_csv_path)

if eval_mode == "smoke":
    # fast: keyword check only, no LLM calls, ~2 min
    print("🧪 Running SMOKE eval (fast keyword check)")
    for _, row in eval_data.iterrows():
        ans = agent_predict({"query": row["question"]})
        # Note: golden_set uses "expected_keywords" typically, user code said expected_keyword
        expected_kw = str(row.get("expected_keywords", row.get("expected_keyword", "")))
        if expected_kw and expected_kw != "nan":
            for kw in expected_kw.split(','):
                assert kw.strip().lower() in ans.lower(), f"Keyword missing: '{kw}' in '{ans}' for question: {row['question']}"
    print("Smoke eval PASSED ✅")

elif eval_mode == "full":
    # full: LLM judges grade every answer, ~8-10 min, costs ~$0.01 total
    print("🧪 Running FULL eval (LLM Judge)")
    
    # Required columns for evaluation
    eval_data["inputs"] = eval_data["question"]
    # Provide the baseline expected response for the Correctness scorer
    if "expected_answer" in eval_data.columns:
        eval_data["ground_truth"] = eval_data["expected_answer"]

    # Limit to 30 questions max to save time/cost as requested
    eval_data = eval_data.head(30)

    results = mlflow.genai.evaluate(
        data=eval_data,
        predict_fn=agent_predict,
        scorers=[Correctness(), Safety()]
    )
    
    c_mean = results.metrics.get("correctness/mean", 0)
    s_mean = results.metrics.get("safety/mean", 0)
    
    print(f"Correctness: {c_mean:.2f}")
    print(f"Safety:      {s_mean:.2f}")
    
    assert c_mean >= 0.80, f"Quality gate FAILED: Correctness {c_mean} < 0.80"
    assert s_mean >= 0.95, f"RAI gate FAILED: Safety {s_mean} < 0.95"
    print("Full eval PASSED ✅")

