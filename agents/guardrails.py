"""
guardrails.py — Extracted guardrail logic for Security + RAI validation.
═══════════════════════════════════════════════════════════════════════════
Standalone module so it can be unit-tested WITHOUT Databricks dependencies.
Used by mosaic_agent.py (predict()) and tested by tests/unit/test_guardrails.py.
"""

import re

# ── Response size config (not a hard cutoff — scored proportionally) ──────────
MAX_RESPONSE_LENGTH = 2000       # Soft limit: answers above this get flagged
CRITICAL_RESPONSE_LENGTH = 5000  # Hard limit: answers above this are blocked

# ── Toxicity keyword list (lowercase) ─────────────────────────────────────────
_TOXIC_KEYWORDS = [
    "kill", "murder", "suicide", "rape", "terrorist", "bomb",
    "hate speech", "racial slur", "white supremac", "nazi",
    "profanity", "f**k", "sh*t", "damn", "bastard",
    "exploit", "hack", "malware", "ransomware",
]

# ── DML / DDL / Injection patterns ────────────────────────────────────────────
_FORBIDDEN_SQL_KEYWORDS = [
    "DROP", "DELETE", "UPDATE", "INSERT", "GRANT", "TRUNCATE",
    "ALTER", "CREATE", "EXEC", "EXECUTE", "UNION SELECT",
    "OR 1=1", "OR '1'='1'", "--", ";--", "/*", "*/",
    "INFORMATION_SCHEMA", "SYSOBJECTS", "SYSCOLUMNS",
]

# ── Credential / secret patterns ──────────────────────────────────────────────
_CREDENTIAL_PATTERNS = [
    r'(?i)(password|passwd|pwd)\s*[:=]\s*\S+',
    r'(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*\S+',
    r'(?i)(bearer\s+[a-zA-Z0-9\-_.]+)',
    r'dapi[a-f0-9]{32}',                     # Databricks PAT token pattern
    r'(?i)BEGIN\s+(RSA\s+)?PRIVATE\s+KEY',    # Private keys
]

# ── PII patterns ──────────────────────────────────────────────────────────────
_PII_PATTERNS = [
    (r'\d{3}-\d{2}-\d{4}',                         "SSN"),
    (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', "Email"),
    (r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',             "Phone"),         # US phone
    (r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',  "CreditCard"),   # 16-digit card
]


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — used by mosaic_agent.py and unit tests
# ══════════════════════════════════════════════════════════════════════════════

def contains_pii(text: str) -> bool:
    """Check if text contains PII patterns (SSN, email, phone, credit card)."""
    for pattern, _pii_type in _PII_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def blocks_dml(text: str) -> bool:
    """Check if text contains forbidden DML/DDL/injection patterns."""
    text_upper = text.upper()
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.match(r'^[A-Za-z\s]+$', keyword):
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                return True
        else:
            if keyword.upper() in text_upper:
                return True
    return False


def validate_query_safety(sql_or_answer: str) -> dict:
    """
    SECURITY GATE: Production-grade security validation.

    Checks for: SQL injection, DML/DDL leakage, credential exposure,
    path traversal, and command injection patterns.

    Returns:
        dict with keys: passed (bool), score (0.0-1.0), flags (list), details (str)
        score = 1.0 means fully safe, 0.0 means critical violation.
    """
    text = str(sql_or_answer).strip()
    text_upper = text.upper()
    flags = []

    # ── Check 1: DML / DDL / SQL Injection ────────────────────────────────
    for keyword in _FORBIDDEN_SQL_KEYWORDS:
        if re.match(r'^[A-Za-z\s]+$', keyword):
            pattern = r'\b' + keyword + r'\b'
            if re.search(pattern, text, re.IGNORECASE):
                flags.append(f"SECURITY:SQL_INJECTION:{keyword}")
        else:
            if keyword.upper() in text_upper:
                flags.append(f"SECURITY:SQL_INJECTION:{keyword}")

    # ── Check 2: Credential / secret leakage ──────────────────────────────
    for pattern in _CREDENTIAL_PATTERNS:
        if re.search(pattern, text):
            flags.append(f"SECURITY:CREDENTIAL_LEAK:{pattern[:30]}")

    # ── Check 3: Path traversal ───────────────────────────────────────────
    if "../" in text or "..\\" in text:
        flags.append("SECURITY:PATH_TRAVERSAL")

    # ── Check 4: Command injection markers ────────────────────────────────
    cmd_patterns = [r'\$\(']
    for pattern in cmd_patterns:
        if re.search(pattern, text):
            flags.append(f"SECURITY:CMD_INJECTION:{pattern[:20]}")

    # ── Score calculation ─────────────────────────────────────────────────
    if not flags:
        score = 1.0
    elif any("SQL_INJECTION" in f for f in flags):
        score = 0.0
    elif any("CREDENTIAL_LEAK" in f for f in flags):
        score = 0.1
    else:
        score = max(0.0, 1.0 - (len(flags) * 0.3))

    passed = len(flags) == 0
    details = "Security check passed" if passed else f"Violations: {', '.join(flags)}"

    return {"passed": passed, "score": score, "flags": flags, "details": details}


def validate_output_safety(answer: str) -> dict:
    """
    RAI GATE: Production-grade Responsible AI validation.

    Checks for: PII leakage, toxicity, data dumps, response size,
    and content safety.

    Returns:
        dict with keys: passed (bool), score (0.0-1.0), flags (list), details (str)
        score = 1.0 means fully safe, 0.0 means critical violation.
    """
    answer_str = str(answer)
    answer_lower = answer_str.lower()
    flags = []
    deductions = 0.0

    # ── Check 1: PII Detection ────────────────────────────────────────────
    for pattern, pii_type in _PII_PATTERNS:
        matches = re.findall(pattern, answer_str)
        if matches:
            flags.append(f"RAI:PII_{pii_type}:found_{len(matches)}")
            deductions += 0.4

    # ── Check 2: Toxicity / Harmful Content ───────────────────────────────
    toxic_found = []
    for keyword in _TOXIC_KEYWORDS:
        if keyword in answer_lower:
            toxic_found.append(keyword)
    if toxic_found:
        flags.append(f"RAI:TOXICITY:{','.join(toxic_found[:5])}")
        deductions += 0.5

    # ── Check 3: Response Size ────────────────────────────────────────────
    length = len(answer_str)
    if length > CRITICAL_RESPONSE_LENGTH:
        flags.append(f"RAI:RESPONSE_TOO_LARGE:{length}_chars")
        deductions += 0.5
    elif length > MAX_RESPONSE_LENGTH:
        flags.append(f"RAI:RESPONSE_LARGE:{length}_chars")
        deductions += 0.2

    # ── Check 4: Data Dump Detection ──────────────────────────────────────
    newline_count = answer_str.count('\n')
    if newline_count > 50:
        flags.append(f"RAI:DATA_DUMP:massive_{newline_count}_lines")
        deductions += 0.5
    elif newline_count > 20:
        flags.append(f"RAI:DATA_DUMP:{newline_count}_lines")
        deductions += 0.2

    # ── Check 5: Repetitive content (hallucination indicator) ─────────────
    words = answer_lower.split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            flags.append(f"RAI:HALLUCINATION:repetitive_content_{unique_ratio:.2f}")
            deductions += 0.3

    # ── Score calculation ─────────────────────────────────────────────────
    score = max(0.0, 1.0 - deductions)
    passed = len(flags) == 0
    details = "RAI check passed" if passed else f"Flags: {', '.join(flags)}"

    return {"passed": passed, "score": score, "flags": flags, "details": details}
