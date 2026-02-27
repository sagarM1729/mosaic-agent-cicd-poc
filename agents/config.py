"""
config.py — Central configuration for the Mosaic AI Agent.
Genie Space IDs, model endpoint, and CI gate thresholds.

NOTE: Tables are managed directly inside Genie Spaces (not here).
      The guardrail checks SQL patterns/injection — not table allowlists.
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

# ── QUALITY GATE ─────────────────────────────────────────────────────────────
QUALITY_GATE_THRESHOLD = 80.0  # Pipeline blocks if accuracy < this %
