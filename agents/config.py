"""
config.py — Central configuration for the Mosaic AI Agent.
All Genie Space IDs, model endpoints, and table allowlists live here.
"""

import os

# ── MODEL ENDPOINT ───────────────────────────────────────────────────────────
MODEL_ENDPOINT = os.environ.get("MODEL_ENDPOINT", "databricks-claude-3-7-sonnet")

# ── GENIE SPACE IDs ──────────────────────────────────────────────────────────
SALES_GENIE_SPACE_ID = os.environ.get(
    "SALES_GENIE_SPACE_ID", "01f111696b8416bd9d27c7f116ec759e"
)
INVENTORY_GENIE_SPACE_ID = os.environ.get(
    "INVENTORY_GENIE_SPACE_ID", "01f1120edf871c6c816f18ddacfe86b7"
)

# ── ALLOWED TABLES (for SQL guardrails) ──────────────────────────────────────
SALES_TABLES = [
    "cicd.gold.fact_sale",
    "cicd.gold.dim_date",
    "cicd.gold.dim_customer",
]

INVENTORY_TABLES = [
    "cicd.gold.dim_stock_item",
    "cicd.gold.dim_date",
    "cicd.gold.fact_stock_holding",
]

ALL_ALLOWED_TABLES = list(set(SALES_TABLES + INVENTORY_TABLES))

# ── QUALITY GATE ─────────────────────────────────────────────────────────────
QUALITY_GATE_THRESHOLD = 80.0  # Pipeline blocks if accuracy < this %
