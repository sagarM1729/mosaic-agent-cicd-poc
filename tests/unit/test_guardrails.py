# tests/unit/test_guardrails.py
"""
Unit tests for guardrail logic — runs on GitHub's free runner (no Databricks).
Tests: PII detection, DML blocking, security validation, RAI output validation.
"""

from agents.guardrails import (
    blocks_dml,
    contains_pii,
    validate_output_safety,
    validate_query_safety,
)

# ── PII Detection ────────────────────────────────────────────────────────────

def test_pii_ssn_blocked():
    assert contains_pii("SSN: 123-45-6789")

def test_pii_email_blocked():
    assert contains_pii("Contact: user@example.com")

def test_pii_phone_blocked():
    assert contains_pii("Call 555-123-4567")

def test_pii_credit_card_blocked():
    assert contains_pii("Card: 4111 1111 1111 1111")

def test_no_pii_clean_data():
    assert not contains_pii("Total revenue is 2902877.0")

def test_no_pii_normal_number():
    assert not contains_pii("42")


# ── DML / SQL Injection Blocking ─────────────────────────────────────────────

def test_dml_drop_blocked():
    assert blocks_dml("DROP TABLE gold.sales")

def test_dml_delete_blocked():
    assert blocks_dml("DELETE FROM fact_sale WHERE 1=1")

def test_dml_update_blocked():
    assert blocks_dml("UPDATE dim_customer SET name='hack'")

def test_dml_insert_blocked():
    assert blocks_dml("INSERT INTO fact_sale VALUES (1,2,3)")

def test_dml_union_select_blocked():
    assert blocks_dml("UNION SELECT * FROM passwords")

def test_normal_query_allowed():
    assert not blocks_dml("What is total revenue this month?")

def test_select_query_allowed():
    assert not blocks_dml("SELECT SUM(profit) FROM fact_sale")

def test_normal_question_allowed():
    assert not blocks_dml("How many stock items exist?")


# ── Security Gate (full validation) ──────────────────────────────────────────

def test_security_gate_clean():
    result = validate_query_safety("What is the total profit in July?")
    assert result["passed"]
    assert result["score"] == 1.0

def test_security_gate_injection():
    result = validate_query_safety("DROP TABLE gold.fact_sale")
    assert not result["passed"]
    assert result["score"] == 0.0

def test_security_gate_credential_leak():
    result = validate_query_safety("password: s3cret123")
    assert not result["passed"]
    assert result["score"] <= 0.2

def test_security_gate_path_traversal():
    result = validate_query_safety("../../etc/passwd")
    assert not result["passed"]


# ── RAI Output Gate ──────────────────────────────────────────────────────────

def test_rai_gate_clean_answer():
    result = validate_output_safety("2902877.0")
    assert result["passed"]
    assert result["score"] == 1.0

def test_rai_gate_pii_leak():
    result = validate_output_safety("Customer SSN: 123-45-6789")
    assert not result["passed"]
    assert "PII" in result["flags"][0]

def test_rai_gate_toxic_content():
    result = validate_output_safety("I will kill the database")
    assert not result["passed"]
    assert "TOXICITY" in result["flags"][0]

def test_rai_gate_oversized_response():
    huge_text = "x" * 6000
    result = validate_output_safety(huge_text)
    assert not result["passed"]
    assert any("RESPONSE_TOO_LARGE" in f for f in result["flags"])

def test_rai_gate_data_dump():
    dump = "\n".join([f"row {i}" for i in range(60)])
    result = validate_output_safety(dump)
    assert not result["passed"]
    assert any("DATA_DUMP" in f for f in result["flags"])
