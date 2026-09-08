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
DEBUG_PII_REPORT: bool = os.getenv("DEBUG_PII_REPORT", "false").lower() in ("true", "1", "t", "yes")


# ── Upload limits ────────────────────────────────────────────────────────────
MAX_IMAGE_SIZE_BYTES: int = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))  # 10 MB

ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/jpg", "image/webp"}
)
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp"})

# ── PII detection ────────────────────────────────────────────────────────────
# Confidence threshold: entities with score < this are discarded.
PII_SCORE_THRESHOLD: float = float(os.getenv("PII_SCORE_THRESHOLD", "0.5"))

# Debug threshold (optional, default -1 = disabled).
# When set to a value >= 0 (e.g. PII_DEBUG_THRESHOLD=0.3), the analyzer logs
# entity_type and score for any candidate that scores ABOVE this value,
# even if it falls below the production PII_SCORE_THRESHOLD.  This lets
# developers tune threshold sensitivity without changing production behaviour.
#
# Privacy guarantee: only entity_type and score are ever logged — raw text
# and matched PII values are NEVER included in debug output.
PII_DEBUG_THRESHOLD: float = float(os.getenv("PII_DEBUG_THRESHOLD", "-1.0"))

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
    "USERNAME",       # Custom: label-prefixed / context-aware usernames
    "PASSWORD",       # Custom: context-aware passwords
    "SOCIAL_HANDLE",  # Custom: bare @handles (e.g. "@dev_jane")
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
    "USERNAME":      "[USERNAME]",   # Custom recognizer
    "PASSWORD":      "[PASSWORD]",   # Custom recognizer
    "SOCIAL_HANDLE": "[HANDLE]",     # Custom recognizer
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

# ── Report output ─────────────────────────────────────────────────────────────
# Directory (relative to CWD) where .txt privacy reports are written.
# Override with OUTPUT_DIR env var, e.g.:  OUTPUT_DIR=/tmp/reports python backend/app.py
REPORT_OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "output")

# ── Face detection ───────────────────────────────────────────────────────────
# Confidence threshold: detections with score < this are discarded.
# Default 0.5 biases toward recall over precision (prioritises not missing faces).
FACE_DETECTION_CONFIDENCE: float = float(os.getenv("FACE_DETECTION_CONFIDENCE", "0.5"))

# Generous percentage padding applied to each detected face bounding box
# to capture hair, forehead, ears, and chin margins (default 20%).
FACE_PADDING_RATIO: float = float(os.getenv("FACE_PADDING_RATIO", "0.20"))

# Base Gaussian blur radius for faces — strong enough to obliterate facial details
FACE_BLUR_RADIUS: int = int(os.getenv("FACE_BLUR_RADIUS", "30"))

# Path to OpenCV SSD Caffe model files
FACE_MODEL_DIR: str = os.getenv(
    "FACE_MODEL_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"),
)
FACE_PROTO_PATH: str = os.path.join(FACE_MODEL_DIR, "deploy.prototxt")
FACE_MODEL_PATH: str = os.path.join(FACE_MODEL_DIR, "res10_300x300_ssd_iter_140000.caffemodel")

