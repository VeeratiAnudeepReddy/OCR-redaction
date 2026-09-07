"""
Processor module — orchestrates the full privacy pipeline:

    PIL Image  →  OCR (Tesseract)  →  PII Detection (Presidio Analyzer)
               →  Anonymisation (Presidio Anonymizer)  →  AnalysisResult

This module owns the pipeline logic.  It delegates OCR to ``ocr.py``
and PII processing to ``pii.py``.

Privacy guarantee: raw OCR text never leaves this module.
"""

from __future__ import annotations

import io
import logging
from typing import Optional

from PIL import Image, UnidentifiedImageError

import backend.config as config
from backend.ocr import extract_text_from_image
from backend.pii import AnalysisResult, process_text

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

class ImageValidationError(ValueError):
    """Raised when the supplied image data fails validation."""

    def __init__(self, message: str, http_status: int = 400) -> None:
        super().__init__(message)
        self.http_status = http_status


def _validate_image_bytes(data: bytes, filename: str = "") -> None:
    """
    Validate raw image bytes before processing.

    Checks:
      - Non-empty payload
      - Size within MAX_IMAGE_SIZE_BYTES
      - File extension (when filename is provided)
      - Actual decodable image (via Pillow)

    Raises:
        ImageValidationError: with an appropriate HTTP status code.
    """
    if not data:
        raise ImageValidationError("No image data received.", http_status=400)

    if len(data) > config.MAX_IMAGE_SIZE_BYTES:
        max_mb = config.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
        raise ImageValidationError(
            f"Image exceeds maximum allowed size of {max_mb:.0f} MB.",
            http_status=413,
        )

    if filename:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        if ext not in config.ALLOWED_EXTENSIONS:
            raise ImageValidationError(
                f"Unsupported file type '.{ext}'. "
                f"Allowed: {', '.join(sorted(config.ALLOWED_EXTENSIONS))}",
                http_status=415,
            )


def _decode_image(data: bytes) -> Image.Image:
    """
    Attempt to decode raw bytes into a PIL Image.

    Raises:
        ImageValidationError(400): if data cannot be decoded as an image.
    """
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()  # detect truncated / corrupted files
        # Re-open after verify (verify() consumes the file object)
        img = Image.open(io.BytesIO(data))
        return img
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageValidationError(
            "Could not decode image. File may be corrupted or unsupported.",
            http_status=400,
        ) from exc


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def process_image_bytes(
    data: bytes,
    filename: str = "",
    preprocess: bool = True,
    debug_pii_report: Optional[bool] = None,
) -> AnalysisResult:
    """
    Full privacy pipeline: raw image bytes → sanitised AnalysisResult.

    Steps:
        1. Validate image bytes (size, extension, decodability).
        2. Decode to PIL Image.
        3. Run Tesseract OCR (with optional preprocessing).
        4. Run Presidio detection + anonymisation.
        5. Return AnalysisResult (safe_text + entity metadata).

    Args:
        data: Raw bytes of the uploaded image (PNG / JPEG / WEBP).
        filename: Original filename for extension validation (optional).
        preprocess: Whether to run image preprocessing before OCR.
        debug_pii_report: Whether debug report mode is enabled.

    Returns:
        :class:`~backend.pii.AnalysisResult` — sanitised text + entity list.
        Raw OCR text is discarded after anonymisation.

    Raises:
        ImageValidationError: for invalid / oversized / corrupted images.
        RuntimeError: for unexpected processing failures.

    Privacy note: raw OCR text is a local variable only; it is NEVER
    returned, logged, or stored.
    """
    # Step 1 — validate
    _validate_image_bytes(data, filename)

    # Step 2 — decode
    image = _decode_image(data)
    logger.info(
        "Processing image: mode=%s size=%dx%d filename=%r",
        image.mode,
        image.width,
        image.height,
        filename or "<unnamed>",
    )

    # Step 3 — OCR (raw text stays local; NEVER logged)
    raw_text: str = extract_text_from_image(image, preprocess=preprocess)

    # Step 4 — Presidio pipeline
    result: AnalysisResult = process_text(raw_text, debug_pii_report=debug_pii_report)

    # raw_text goes out of scope here — not stored, not returned, not logged.
    logger.info(
        "Pipeline complete. Entities found: %d. Text length (sanitised): %d chars.",
        len(result.entities),
        len(result.safe_text),
    )
    return result


def process_image_object(
    image: Image.Image,
    preprocess: bool = True,
    debug_pii_report: Optional[bool] = None,
) -> AnalysisResult:
    """
    Convenience overload: accept a PIL Image directly instead of raw bytes.

    Useful for testing and programmatic usage.
    """
    logger.info(
        "Processing PIL Image: mode=%s size=%dx%d",
        image.mode,
        image.width,
        image.height,
    )

    raw_text: str = extract_text_from_image(image, preprocess=preprocess)
    result: AnalysisResult = process_text(raw_text, debug_pii_report=debug_pii_report)

    logger.info(
        "Pipeline complete. Entities found: %d.",
        len(result.entities),
    )
    return result

