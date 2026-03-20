# Databricks CI/CD for Mosaic NL-to-SQL Agent

Production-style CI/CD pipeline for a Databricks-hosted NL-to-SQL agent built with LangChain ReAct, Databricks Genie Spaces, MLflow, and Databricks Asset Bundles.

This repository implements a gated release flow:
- Lint + unit tests + bundle validation
- Dev deployment + evaluation gates
- Prod deployment + model registration + health check + rollback support

## What This Project Does

- Routes natural language questions to domain-specific Genie Space tools:
  - `Sales_Expert`
  - `Inventory_Expert`
- Applies security and RAI guardrails before returning answers
- Evaluates model behavior using smoke and full datasets
- Enforces 4 CI gates before production promotion:
  - Quality
  - Security
  - RAI
  - Cost
- Registers model versions in Unity Catalog and updates `@PROD` alias
- Runs post-deploy health checks and supports alias rollback

## High-Level Architecture

1. Developer pushes to `main` or opens a PR.
2. GitHub Actions runs:
   - Stage 1: free pre-checks (`ruff`, `pytest`, `bundle validate`)
   - Stage 2: deploy to `dev` and run eval job (smoke for PR, full for push)
   - Stage 3: deploy to `prod`, register model, health check, rollback on failure
3. Agent runtime:
   - ReAct agent chooses Sales or Inventory Genie tool
   - Guardrails validate query/output
   - Final answer returned as a single value

See `ARCHITECTURE.md` for full design details.

## Repository Layout

```
.
|- agents/
|  |- config.py
|  |- guardrails.py
|  |- mosaic_agent.py
|  `- tools.py
|- eval/
|  |- golden_set.csv
|  `- smoke_set.csv
|- prompts/
|  `- system_prompt.txt
|- scripts/
|  |- get_prod_version.py
|  |- health_check.py
|  |- register_model.py
|  `- rollback_prod.py
|- tests/
|  |- test.py
|  `- unit/test_guardrails.py
|- databricks.yml
|- deploy.py
`- requirements.txt
```

## Prerequisites

- Python 3.11 (for local lint/tests and GitHub Actions parity)
- Databricks CLI (v0.2xx+ recommended)
- Databricks workspace with:
  - Unity Catalog enabled
  - Access to Genie Spaces and Model Serving
- GitHub repository secrets for CI:
  - `DATABRICKS_HOST_CICD`
  - `DATABRICKS_TOKEN_CICD`

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install ruff pytest
```

## Local Validation

Run the same pre-checks used in CI Stage 1:

```bash
ruff check agents/ scripts/ tests/
pytest tests/unit/ -v
databricks bundle validate --target dev
```

## Databricks Authentication

For local CLI use:

```bash
export DATABRICKS_HOST="https://<your-workspace-host>"
export DATABRICKS_TOKEN="<your-pat>"

databricks current-user me
```

In GitHub Actions, these values are read from repository secrets.

## Deploy with Databricks Asset Bundles

Validate and deploy to `dev`:

```bash
databricks bundle validate --target dev
databricks bundle deploy --target dev
```

Run eval jobs manually:

```bash
databricks bundle run mosaic_smoke_eval --target dev
databricks bundle run mosaic_full_eval --target dev
```

Deploy to `prod`:

```bash
databricks bundle deploy --target prod
databricks bundle run mosaic_register --target prod
```

## CI/CD Pipeline

Workflow file: `.github/workflows/deploy.yml`

Triggers:
- Push to `main`
- Pull request targeting `main`

Stages:
1. `pre_checks`
   - `ruff` lint
   - `pytest tests/unit`
   - `databricks bundle validate --target dev`
2. `deploy_dev`
   - Deploy bundle to `dev`
   - PR: smoke eval (`mosaic_smoke_eval`)
   - Push: full eval (`mosaic_full_eval`)
   - Extracts `CI_GATE_JSON` from job output and publishes GitHub Step Summary
3. `deploy_prod` (push only)
   - Deploy bundle to `prod`
   - Save current `@PROD` model version
   - Register new model version and set alias
   - Run endpoint health check
   - Roll back alias on failure

## CI Gate Contract

The full eval enforces:
- Quality: >= 80%
- Security: 100%
- RAI: >= 95%
- Cost: <= 5000 average tokens per query

If gate extraction fails or any gate fails, deployment is blocked (fail-safe behavior).

## Model Lifecycle

- Model name (prod): `cicd.prod.mosaic_nl_sql_agent`
- Registry URI: `databricks-uc`
- Alias strategy: `@PROD` points to currently serving version
- Deployment helper: `databricks.agents.deploy()`
- Health check script: `scripts/health_check.py`
- Rollback script: `scripts/rollback_prod.py --version <N>`

## Configuration

Main config points:
- `agents/config.py`
  - `MODEL_ENDPOINT`
  - `SALES_GENIE_SPACE_ID`
  - `INVENTORY_GENIE_SPACE_ID`
  - `QUALITY_GATE_THRESHOLD`
- `prompts/system_prompt.txt`
  - Prompt content and prompt version tag (`PROMPT_VERSION`)
- `databricks.yml`
  - Job definitions, clusters, targets (`dev`, `prod`), and sync rules

## Evaluation Datasets

- Smoke set: `eval/smoke_set.csv` (fast checks)
- Golden set: `eval/golden_set.csv` (full checks)

Both are executed through `tests/test.py` in notebook-job mode.

## Troubleshooting

- Bundle validation fails:
  - Run `databricks bundle validate --target dev` and fix YAML/path issues in `databricks.yml`.
- Eval fails but output is unclear:
  - Inspect Databricks job logs and GitHub Step Summary gate table.
- Endpoint not ready after deploy:
  - Re-run health check script or verify serving endpoint state in workspace.
- Need manual rollback:
  - `python scripts/rollback_prod.py --version <previous_version>`

## Notes

- `requirements.txt` is primarily for CI runner dependencies.
- Databricks jobs install notebook dependencies with `%pip` and use Databricks runtime libraries.
- Unit tests under `tests/unit` are intentionally Databricks-independent.


  ## Results
- <img width="1888" height="570" alt="Screenshot From 2026-03-20 21-27-46" src="https://github.com/user-attachments/assets/e8bb5732-c2d2-4858-82cb-12fd38cc3ace" />

<img width="1758" height="948" alt="Screenshot From 2026-03-20 21-28-18" src="https://github.com/user-attachments/assets/cf831cf9-d5fc-437f-a583-9354d63461ff" />

<img width="1758" height="948" alt="Screenshot From 2026-03-20 21-28-30" src="https://github.com/user-attachments/assets/f7ced437-fe6e-443d-9d23-49a34166baa9" />

<img width="1901" height="807" alt="Screenshot From 2026-03-20 21-29-12" src="https://github.com/user-attachments/assets/8bbdf148-facd-4803-9ddf-77cf3ec6b033" />




