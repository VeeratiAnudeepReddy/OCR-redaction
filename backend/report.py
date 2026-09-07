"""
report.py — Privacy-safe text report writer.

Generates a human-readable .txt report for every processed image.

Privacy guarantee:
  * This module ONLY ever receives already-anonymised data:
      - safe_text  (PII already replaced with placeholders)
      - entities   (type + score only — no raw matched values)
  * It never receives, stores, or writes raw OCR text.
  * It never receives, stores, or writes raw PII values.

Report layout
─────────────
  ============================================================
  OCR + PII PRIVACY REPORT
  ============================================================
  INPUT IMAGE: <filename>

  1. EXTRACTED TEXT        (safe_text — PII already anonymised)
  2. DETECTED PRIVATE DATA (entity type, confidence, token)
  3. SANITIZED TEXT        (safe_text repeated for inspection)
  4. PRIVACY CHECK         (processing status flags)
  ============================================================

Output location:
  <REPORT_OUTPUT_DIR>/<stem>_report.txt
  e.g.  output/screenshot_report.txt
"""

from __future__ import annotations

import logging
import pathlib
import re
from datetime import datetime, timezone
from typing import List

import backend.config as config
from backend.pii import AnalysisResult, DetectedEntity

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Replacement-token lookup  (mirrors config.ANONYMIZATION_LABELS)
# ---------------------------------------------------------------------------

_ENTITY_TOKEN: dict[str, str] = {
    **config.ANONYMIZATION_LABELS,
}
_DEFAULT_TOKEN: str = config.DEFAULT_REDACTION_LABEL


def _token_for(entity_type: str) -> str:
    """Return the placeholder token for a given entity type."""
    return _ENTITY_TOKEN.get(entity_type, _DEFAULT_TOKEN)


# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

def _report_stem(image_filename: str) -> str:
    """
    Derive a safe report filename stem from the input image filename.

    Examples:
        "screenshot.png"                      -> "screenshot_report"
        "WhatsApp Image 2026-09-06.jpeg"      -> "WhatsApp_Image_2026-09-06_report"
        ""                                    -> "image_report"
    """
    stem = pathlib.Path(image_filename).stem if image_filename else "image"
    # Replace whitespace / unsafe chars with underscores
    stem = re.sub(r"[^\w\-.]", "_", stem)
    stem = re.sub(r"_+", "_", stem).strip("_") or "image"
    return f"{stem}_report"


def _report_path(image_filename: str) -> pathlib.Path:
    """Return the full Path where the report will be written."""
    output_dir = pathlib.Path(config.REPORT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{_report_stem(image_filename)}.txt"


# ---------------------------------------------------------------------------
# Report builder
# ---------------------------------------------------------------------------

_LINE = "=" * 60
_DIVIDER = "-" * 60


def _build_report(
    image_filename: str,
    safe_text: str,
    entities: List[DetectedEntity],
    ocr_ok: bool = True,
    pii_ok: bool = True,
    debug_mode: bool = False,
) -> str:
    """
    Build the full report string.

    Args:
        image_filename: Original filename of the processed image.
        safe_text:      Anonymised OCR text (PII already replaced).
        entities:       List of detected PII entities.
        ocr_ok:         Whether OCR completed successfully.
        pii_ok:         Whether PII anonymisation completed successfully.
        debug_mode:     Whether DEBUG_PII_REPORT is enabled.

    Returns:
        A multi-line string ready to be written to disk.
    """
    lines: list[str] = []

    def ln(s: str = "") -> None:
        lines.append(s)

    # Header
    ln(_LINE)
    ln("OCR + PII PRIVACY REPORT".center(60))
    ln(_LINE)
    ln()
    ln("INPUT IMAGE:")
    ln(pathlib.Path(image_filename).name if image_filename else "image")
    ln()

    # Section 1 — Extracted Text
    ln(_LINE)
    ln("1. EXTRACTED TEXT")
    ln(_LINE)
    ln()
    if safe_text and safe_text.strip():
        for row in safe_text.splitlines():
            ln(row)
    else:
        ln("(no text extracted)")
    ln()

    # Section 2 — Detected Personal / Private Data
    ln(_LINE)
    ln("2. DETECTED PERSONAL / PRIVATE DATA")
    ln(_LINE)
    ln()
    if entities:
        ln(f"Total entities detected: {len(entities)}")
        ln()
        for i, entity in enumerate(entities, start=1):
            val_str = entity.value if (debug_mode and entity.value is not None) else "[HIDDEN]"
            ln(f"{i}. Type: {entity.entity_type}")
            ln(f"   Value: {val_str}")
            ln(f"   Confidence: {entity.score:.2f}")
            ln(f"   Replacement: {_token_for(entity.entity_type)}")
            ln()
    else:
        ln("Total entities detected: 0")
        ln()

    # Section 3 — Sanitized Text
    ln(_LINE)
    ln("3. SANITIZED TEXT")
    ln(_LINE)
    ln()
    if safe_text and safe_text.strip():
        for row in safe_text.splitlines():
            ln(row)
    else:
        ln("(no text extracted)")
    ln()

    # Section 4 — Privacy Check
    ln(_LINE)
    ln("4. PRIVACY CHECK")
    ln(_LINE)
    ln()
    ln(f"OCR completed: {'YES' if ocr_ok else 'NO'}")
    ln(f"PII detection completed: {'YES' if pii_ok else 'NO'}")
    ln(f"Anonymization completed: {'YES' if pii_ok else 'NO'}")
    ln()
    ln(f"DEBUG PII REPORT: {'ENABLED' if debug_mode else 'DISABLED'}")
    ln()
    raw_pii_written = "YES" if (debug_mode and any(e.value is not None for e in entities)) else "NO"
    ln(f"Raw PII written to report: {raw_pii_written}")
    ln()
    ln(_LINE)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_report(
    image_filename: str,
    result: AnalysisResult,
    debug_pii_report: Optional[bool] = None,
) -> str:
    """
    Write a privacy-safe .txt report for a processed image.

    Args:
        image_filename: Original filename of the processed image.
        result:         AnalysisResult from the pipeline.
        debug_pii_report: Optional override for DEBUG_PII_REPORT mode.

    Returns:
        Relative path of the written report file as a string,
        e.g. ``"output/screenshot_report.txt"``.
    """
    if debug_pii_report is None:
        debug_pii_report = config.DEBUG_PII_REPORT

    report_path = _report_path(image_filename)
    content = _build_report(
        image_filename=image_filename,
        safe_text=result.safe_text,
        entities=result.entities,
        ocr_ok=True,
        pii_ok=True,
        debug_mode=debug_pii_report,
    )

    try:
        report_path.write_text(content, encoding="utf-8")
        logger.info("Privacy report written (debug_mode=%s): %s", debug_pii_report, report_path)
    except OSError as exc:
        logger.error(
            "Failed to write privacy report (%s): %s",
            type(exc).__name__,
            report_path,
        )
        raise

    return str(pathlib.Path(config.REPORT_OUTPUT_DIR) / report_path.name)

