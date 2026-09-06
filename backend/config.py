"""
Configuration for the Privacy-Safe OCR + PII Anonymization Service.

All tuneable parameters live here — no magic numbers scattered through the code.
"""

from __future__ import annotations

import os

# ── Server ──────────────────────────────────────────────────────────────────
HOST: str = os.getenv("HOST", "127.0.0.1")
PORT: int = int(os.getenv("PORT", "5005"))
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

# ── Upload limits ────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_BYTES: int = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp"}
)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp"})

# ── PII detection ────────────────────────────────────────────────────────────
# Confidence threshold: entities with score < this are discarded.
PII_SCORE_THRESHOLD: float = float(os.getenv("PII_SCORE_THRESHOLD", "0.5"))

# Entity types to analyse.  Add more as needed; Presidio will skip those it
# cannot handle with the loaded recognisers.
ENTITIES_TO_DETECT: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "LOCATION",
    "ORGANIZATION",
    "CREDIT_CARD",
    "US_SSN",
    "IP_ADDRESS",
    "URL",
]

# ── Anonymization ─────────────────────────────────────────────────────────────
# Maps each entity type to its placeholder label.
ANONYMIZATION_LABELS: dict[str, str] = {
    "PERSON":        "[PERSON]",
    "EMAIL_ADDRESS": "[EMAIL_ADDRESS]",
    "PHONE_NUMBER":  "[PHONE_NUMBER]",
    "LOCATION":      "[LOCATION]",
    "ORGANIZATION":  "[ORGANIZATION]",
    "CREDIT_CARD":   "[CREDIT_CARD]",
    "US_SSN":        "[US_SSN]",
    "IP_ADDRESS":    "[IP_ADDRESS]",
    "URL":           "[URL]",
}
DEFAULT_REDACTION_LABEL: str = "[REDACTED]"

# ── CORS ──────────────────────────────────────────────────────────────────────
# Comma-separated list of allowed origins (env override supported).
_raw_origins: str = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost,http://127.0.0.1,chrome-extension://",
)
ALLOWED_ORIGINS: list[str] = [o.strip() for o in _raw_origins.split(",") if o.strip()]

# ── spaCy model ───────────────────────────────────────────────────────────────
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "en_core_web_lg")
