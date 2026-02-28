# Architecture Design: Mosaic AI Agent — Databricks CI/CD

**Version:** 11.1 (Code-Verified)  
**Owner:** Sagar Meshram  
**Last Updated:** 2026-02-28  
**Scope:** POC on Databricks Premium 14-day trial

---

## 1. Executive Summary

A **serverless, CI/CD-gated NL-to-SQL agent** on Databricks that routes natural language questions to two Genie Spaces (Sales + Inventory), validates answers through 4 automated gates, and deploys to Unity Catalog with instant rollback.

**Stack:** Databricks Asset Bundles + LangChain ReAct + Genie Space API + MLflow 3.x + GitHub Actions

---

## 2. Architecture Flow

```
Push to main
  → GitHub Actions (3 stages)
    ├─ Stage 1: ruff lint + 23 unit tests + bundle validate (FREE — no DBUs)
    ├─ Stage 2: Deploy to dev → Full eval (28 questions, 4 gates)
    │            └─ If gates fail → raise Exception → block pipeline
    └─ Stage 3: Deploy to prod → Register model → Set @PROD alias
                 → agents.deploy() → Health check → Rollback on failure
```

```
User Question
  → LangChain ReAct Agent (Claude 3.7 Sonnet via Databricks Model Serving)
    → Input Guardrail (SQL injection, DML/DDL block)
    → Tool Selection (Sales_Expert OR Inventory_Expert)
    → Genie Space REST API (4-step: start → poll → query-result → extract)
    → Output Guardrail (PII, toxicity, size, hallucination check)
    → Final Answer (single value)
```

---

## 3. Infrastructure

### 3.1 Cluster Configuration

| Setting | Value |
|---------|-------|
| Runtime | DBR 16.4 LTS (Spark 3.5, Scala 2.12) |
| Node | Standard_D4s_v3 |
| Mode | Single-node (`num_workers: 0`) |
| Security | SINGLE_USER mode |
| MLflow | Pre-installed (3.x) — NOT pip-installed |
| Python | 3.11 (CI runner), 3.10 (cluster) |

### 3.2 Authentication

| Context | Method |
|---------|--------|
| GitHub Actions → Databricks | PAT stored in GitHub Secrets (`DATABRICKS_HOST_CICD`, `DATABRICKS_TOKEN_CICD`) |
| Notebooks → REST APIs | `dbutils.notebook.getContext().apiToken()` injected as env vars |
| MLflow Registry | `databricks-uc` URI |

### 3.3 Environment Isolation

| Target | Catalog | Schema | Purpose |
|--------|---------|--------|---------|
| `dev` (default) | `cicd` | `dev` | Evaluation runs |
| `prod` | `cicd` | `prod` | Model registry + serving |

Data lives in `cicd.gold.*` — shared read-only across both targets.

### 3.4 Workspace

- Host: `adb-7405619257134796.16.azuredatabricks.net`
- Bundle name: `mosaic-nl-sql-agent`

---

## 4. Agent Design

### 4.1 LLM

| Setting | Value |
|---------|-------|
| Model | Claude 3.7 Sonnet |
| Endpoint | `databricks-claude-3-7-sonnet` (Databricks Foundation Model) |
| Temperature | 0.1 |
| Max Tokens | 1024 |

### 4.2 ReAct Agent

- Framework: LangChain `create_react_agent` + `AgentExecutor`
- `max_iterations`: 3 (prevents infinite loops)
- `handle_parsing_errors`: True
- `verbose`: False (silent in production)
- Includes iteration-limit fallback: extracts the tool's answer directly if the LLM loops out without saying "Final Answer"

### 4.3 Tools (Genie Spaces)

| Tool | Genie Space | Domain | Tables |
|------|-------------|--------|--------|
| `Sales_Expert` | `01f111...ec759e` | Revenue, profit, invoices, customers, dates | `fact_sale`, `dim_date`, `dim_customer` |
| `Inventory_Expert` | `01f112...fe86b7` | Stock, bins, reorder, colors, brands | `dim_stock_item`, `fact_stock_holding` |

**Genie API Flow (4 steps):**

1. `POST /start-conversation` → get `conversation_id` + `message_id`
2. `GET /messages/{id}` → poll until `COMPLETED` (timeout: 120s)
3. `GET /attachments/{id}/query-result` → extract first cell from `data_typed_array`
4. Fallback: text attachment → top-level content

**Tool return format:** `"The database returned this exact result: {answer}. [SQL_USED: {sql}] You must now output 'Final Answer: {answer}'"` — forces the LLM to terminate cleanly.

### 4.4 Prompt System

- External file: `prompts/system_prompt.txt`
- Version tag: `# PROMPT_VERSION: v2.2`
- Content: Full schema definitions, JOIN keys, exact query patterns for both domains
- Injected into ReAct template at build time
- This is the real accuracy engine — the prompt has ~50 pre-written SQL patterns

### 4.5 Guardrails (`agents/guardrails.py`)

**Security Gate (`validate_query_safety`):**

- DML/DDL keyword blocking (DROP, DELETE, UPDATE, INSERT, GRANT, TRUNCATE, ALTER, CREATE)
- SQL injection patterns (`OR 1=1`, `--`, `UNION SELECT`, `/*`)
- Credential leak detection (PAT tokens, API keys, private keys, passwords)
- Path traversal (`../`)
- Command injection (`$(...)`)

**RAI Gate (`validate_output_safety`):**

- PII detection (SSN, email, phone, credit card — regex patterns)
- Toxicity filtering (keyword list)
- Response size limit (2000 soft, 5000 hard)
- Data dump detection (>50 newlines = blocked)
- Hallucination indicator (repetitive content, unique word ratio <0.3)

**Enforcement:** If either gate fails, the answer is replaced with `"BLOCKED: guardrail violation — {flags}"`.

Unit tested: **23 tests** running on GitHub's free runner (no Databricks cost).

---

## 5. CI/CD Pipeline

### 5.1 Three Stages

```
┌──────────────────────┐     ┌───────────────────────────┐     ┌───────────────────────────┐
│  STAGE 1: pre_checks │────→│  STAGE 2: deploy_dev      │────→│  STAGE 3: deploy_prod     │
│  (FREE — no DBUs)    │     │  (Dev deploy + eval)      │     │  (main only)              │
├──────────────────────┤     ├───────────────────────────┤     ├───────────────────────────┤
│  • ruff lint         │     │  • bundle deploy --dev    │     │  • bundle deploy --prod   │
│  • pytest (23 tests) │     │  • PR → smoke (5 Q)       │     │  • Save @PROD version     │
│  • bundle validate   │     │  • main → full (28 Q)     │     │  • Register model in UC   │
│                      │     │  • 4-Gate validation       │     │  • agents.deploy()        │
│                      │     │  • GitHub Step Summary     │     │  • Health check           │
│                      │     │                            │     │  • Rollback on failure    │
└──────────────────────┘     └────────────────────────────┘     └───────────────────────────┘
```

**Trigger:** Push to `main` (all 3 stages) or PR to `main` (Stages 1+2 only, smoke eval).

### 5.2 The 4-Gate Contract

| Gate | Metric | Threshold | How Measured |
|------|--------|-----------|-------------|
| **Quality** | Dual-gate accuracy (answer match + SQL logic) | ≥ 80% | MLflow `Correctness()` judge + numeric fuzzy matching (0.5% tolerance) |
| **Security** | SQL injection / credential scan pass rate | 100% | `validate_query_safety()` on every response |
| **RAI** | PII / toxicity / size violations | ≥ 95% | `validate_output_safety()` on every response |
| **Cost** | Average tokens per query | ≤ 5000 | Token callback + char-length fallback estimate |

**Fail behavior:** When gates fail, `test.py` raises an `Exception` (not `dbutils.notebook.exit()`), which causes the Databricks task to fail with non-zero exit. The exception message embeds `CI_GATE_JSON:{...}` so the pipeline can still extract gate data for the GitHub summary.

### 5.3 Gate JSON Extraction (5 strategies)

The pipeline extracts gate results from Databricks job output using cascading strategies:

1. Grep `CI_GATE_JSON:` tag from CLI output
2. Pattern match `Notebook exited:` / `Output:` lines
3. `get-run-output` API → `notebook_output.result` (pass case) or `error_trace` (fail case)
4. PCRE grep from full error output
5. Python catch-all: find any JSON with `overall_pass` key

If all 5 fail → **fail-safe blocks deployment** (never deploys with unknown gate status).

### 5.4 GitHub Step Summary

Every run publishes a markdown table to the GitHub Actions UI:

- Gate scorecard (Quality / Security / RAI / Cost with thresholds and pass/fail)
- Detailed breakdown (answer accuracy, SQL pass rate, security flags, RAI flags)
- Pipeline metadata (SHA, branch, timestamp)

---

## 6. Model Lifecycle

### 6.1 Registration

- MLflow PyFunc wrapper (`MosaicLangChainAgent` class)
- Artifacts: `agents/` directory + `prompts/` directory
- Registered to: `cicd.prod.mosaic_nl_sql_agent` in Unity Catalog
- Tagged with: `git_sha`, `prompt_version`, `model_endpoint`, `agent_type`
- Pip requirements pinned: `langchain==0.3.25`, `langchain-core==0.3.59`, etc.

### 6.2 Alias Strategy

| Alias | Purpose |
|-------|---------|
| `@PROD` | Points to the live version. Updated on every successful deploy. |

`get_prod_version.py` saves the current `@PROD` version before registration. If anything fails downstream, `rollback_prod.py --version {N}` restores it in one CLI call.

### 6.3 Serving

- `agents.deploy()` creates/updates the endpoint `mosaic-nl-sql-agent`
- `scale_to_zero=True` — $0 when idle
- Auto-creates inference tables (logs all prod requests/responses as a Delta table)

### 6.4 Health Check

- Waits up to 5 min for endpoint to reach `NOT_UPDATING` state
- Sends one test query as a smoke test
- Gracefully skips if endpoint doesn't exist after 60s (first deploy)

---

## 7. Observability

| Layer | Tool | What It Captures |
|-------|------|-----------------|
| Agent internals | `mlflow.tracing.enable()` | Full step-by-step: guardrail check → tool selection → Genie API call → LLM response → output filter |
| Experiment tracking | MLflow runs | Response time, token count, prompt version, git SHA |
| CI results | GitHub Step Summary | 4-gate scorecard per pipeline run |
| Production traffic | Inference tables (via `agents.deploy()`) | Every request/response auto-logged to Delta |
| Evaluation | MLflow Evaluation UI | Correctness + Safety assessments per question, traces view |

---

## 8. Evaluation Design

### 8.1 Datasets

| Set | File | Questions | When Used |
|-----|------|-----------|-----------|
| Smoke | `eval/smoke_set.csv` | 5 | PR checks (fast, keyword-only) |
| Full | `eval/golden_set.csv` | 28 | Main merges (LLM judge + all 4 gates) |

Both cover Sales + Inventory domains. Questions are single-value answers (aggregates, lookups, counts).

### 8.2 Scoring

**Smoke eval:** Keyword matching only (checks expected keywords appear in generated SQL or answer). No LLM judge.

**Full eval (LLM-as-Judge):**

- **LLM-as-Judge via MLflow:** `mlflow.genai.evaluate()` with `Correctness()` + `Safety()` scorers — these are Databricks-hosted LLM judges that evaluate each answer against the expected answer and return per-row Pass/Fail + numeric scores
- `Correctness()` — judges whether the agent's answer semantically matches the expected answer (powers the Quality gate)
- `Safety()` — judges whether the response is safe and appropriate (powers the RAI gate alongside app-level guardrails)
- Numeric fuzzy matching: 0.5% tolerance for floating-point differences (e.g., `6638955.75` vs `6638955.749999988`)
- Fallback scoring: If MLflow judge returns all zeros (API issue), falls back to answer-match percentage (for quality) and guardrail checks (for RAI)

---

## 9. Repository Structure

```
databricks-cicd/
├── .github/workflows/
│   └── deploy.yml              # 3-stage CI/CD pipeline (~420 lines)
├── agents/
│   ├── config.py               # Endpoints, Genie IDs, thresholds
│   ├── guardrails.py           # Security + RAI validation (194 lines)
│   ├── mosaic_agent.py         # ReAct agent + predict() (336 lines)
│   └── tools.py                # Genie Space API integration (262 lines)
├── eval/
│   ├── golden_set.csv          # 28 questions (full eval)
│   └── smoke_set.csv           # 5 questions (PR smoke)
├── prompts/
│   └── system_prompt.txt       # Versioned prompt with schema + SQL patterns (119 lines)
├── scripts/
│   ├── get_prod_version.py     # Read current @PROD alias version
│   ├── health_check.py         # Post-deploy endpoint validation
│   ├── register_model.py       # UC registration + agents.deploy()
│   └── rollback_prod.py        # Restore @PROD to previous version
├── tests/
│   ├── test.py                 # Databricks notebook: smoke/full eval runner (~355 lines)
│   └── unit/
│       └── test_guardrails.py  # 23 unit tests (runs free on GitHub runner)
├── databricks.yml              # DAB config: jobs, targets, sync
├── deploy.py                   # Interactive notebook: register + test inference
├── requirements.txt            # CI runner dependencies (NOT used on cluster)
└── ruff.toml                   # Lint config (E, F, I rules)
```

---

## 10. Dependencies

### CI Runner (GitHub Actions — `requirements.txt`)

| Package | Version | Purpose |
|---------|---------|---------|
| `langchain` | 0.3.25 | Agent framework |
| `databricks-langchain` | ≥0.1.0 | ChatDatabricks LLM |
| `langchain-community` | 0.3.24 | Community tools |
| `langchain-core` | 0.3.59 | Core abstractions |
| `mlflow-tracing` | ≥3.1.0 | Lightweight tracing (NOT full mlflow — avoids SDK conflict) |
| `requests` | 2.32.3 | Genie API calls |

### Databricks Cluster (pip install in notebooks)

| Package | Version | Purpose |
|---------|---------|---------|
| `databricks-langchain` | ≥0.1.0 | ChatDatabricks LLM |
| `langchain` | 0.3.25 | Agent framework |
| `langchain-community` | 0.3.24 | Community tools |
| `langchain-core` | 0.3.59 | Core abstractions |
| `databricks-agents` | latest | `agents.deploy()` + MLflow scorers |
| `mlflow` | Pre-installed (DBR 16.4) | NOT pip-installed — avoids conflicts |

---

## 11. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| ReAct over function-calling | Transparent reasoning chain, debuggable via traces |
| Genie Space over direct SQL gen | Databricks-managed SQL generation with built-in table access control |
| `mlflow-tracing` on CI runner (not full `mlflow`) | Full mlflow conflicts with `databricks-sdk` on the CI runner |
| Single-node clusters | POC workload — no distributed compute needed |
| `raise Exception` over `dbutils.notebook.exit()` on failure | `notebook.exit()` always returns exit code 0 — exceptions actually fail the task |
| Prompt-driven accuracy (not fine-tuning) | System prompt with exact SQL patterns is cheaper and faster to iterate than model training |
| `scale_to_zero=True` on serving | $0 cost when idle — critical for 14-day trial budget |
| Separate smoke vs full eval | PRs get fast feedback (5Q, <1 min); main merges get thorough validation (28Q, all gates) |

---

## 12. Known Limitations (POC Scope)

| Limitation | Impact | Fix Complexity |
|-----------|--------|---------------|
| Genie tools return only first row of multi-row results | List-type questions get incomplete answers | Medium — need to iterate `data_typed_array` |
| Token tracking is approximate | Cost gate checks an estimate, not real usage | Low — ChatDatabricks doesn't expose token counts reliably |
| `deploy.py` and `register_model.py` duplicate the PyFunc class | Maintenance risk if one is changed without the other | Low — extract to shared module |
| No retry on Genie API failures | One transient timeout fails the entire question | Low — add 1 retry with 5s backoff |
| Health check question doesn't match data timeframe | Asks about "this month" but data is from 2016 | Trivial — change the question |
| No staging environment | Dev → Prod with no intermediate | Medium — add a `staging` target in `databricks.yml` |

---

## 13. What Makes This Strong for a POC

1. **3-stage CI/CD with automated quality gates** — most POCs are a single notebook with manual testing
2. **4 independent gates** (quality, security, RAI, cost) — not just "does it work"
3. **Instant rollback** via `@PROD` alias — production-safe from day one
4. **Full MLflow tracing** — every agent step is visible in the UI without extra infra
5. **Inference tables auto-created** — production monitoring out of the box
6. **Prompt versioning + Git SHA tagging** — full auditability of what ran when
7. **23 unit tests running for free** on GitHub's runner — guardrail logic validated without any Databricks cost
8. **Fail-safe pipeline** — unknown gate results block deployment (never deploys blind)
