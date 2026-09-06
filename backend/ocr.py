"""
OCR module — responsible ONLY for extracting text from images via Tesseract.

Privacy guarantee: raw OCR output is NEVER logged.
"""

from __future__ import annotations

import logging


from PIL import Image, ImageEnhance, ImageFilter
import pytesseract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def _to_rgb(image: Image.Image) -> Image.Image:
    """Ensure the image is in RGB colour space."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _to_grayscale(image: Image.Image) -> Image.Image:
    """Convert to grayscale (improves Tesseract accuracy on many images)."""
    return image.convert("L")


def _upscale_if_small(image: Image.Image, min_width: int = 1000) -> Image.Image:
    """
    Upscale images that are smaller than *min_width* pixels wide.

    Tesseract performs poorly on very small/low-resolution images.
    We use LANCZOS resampling which preserves sharpness.
    """
    width, height = image.size
    if width < min_width:
        scale = min_width / width
        new_size = (int(width * scale), int(height * scale))
        image = image.resize(new_size, Image.LANCZOS)
        logger.debug("Upscaled image from %dx%d to %dx%d", width, height, *new_size)
    return image


def _enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Boost contrast slightly to help OCR differentiate text from background."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def _sharpen(image: Image.Image) -> Image.Image:
    """Apply a mild sharpening filter."""
    return image.filter(ImageFilter.SHARPEN)


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply a lightweight preprocessing pipeline to improve Tesseract accuracy.

    Steps (in order):
        1. Convert to RGB
        2. Upscale if image is very small
        3. Convert to grayscale
        4. Enhance contrast
        5. Sharpen

    Returns a new PIL Image ready for OCR.
    """
    image = _to_rgb(image)
    image = _upscale_if_small(image)
    image = _to_grayscale(image)
    image = _enhance_contrast(image)
    image = _sharpen(image)
    return image


# ---------------------------------------------------------------------------
# OCR entry-point
# ---------------------------------------------------------------------------

def extract_text_from_image(
    image: Image.Image,
    preprocess: bool = True,
    lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.

    Args:
        image: A PIL Image object.
        preprocess: Whether to run the preprocessing pipeline before OCR.
        lang: Tesseract language code(s), e.g. "eng" or "eng+fra".
        tesseract_config: Tesseract config flags.
            --oem 3  → use both LSTM and legacy engine (best accuracy).
            --psm 6  → assume a single uniform block of text (good default).

    Returns:
        Extracted text as a plain string.  May be empty if no text is found.

    Raises:
        pytesseract.TesseractNotFoundError: if Tesseract binary is not installed.
        pytesseract.TesseractError: if Tesseract returns a non-zero exit code.
        RuntimeError: for unexpected OCR-related failures.

    Privacy note: the returned text is NEVER logged by this function.
    """
    if preprocess:
        image = preprocess_image(image)

    try:
        # pytesseract.image_to_string returns a UTF-8 decoded string
        raw_text: str = pytesseract.image_to_string(
            image,
            lang=lang,
            config=tesseract_config,
        )
        char_count = len(raw_text.strip())
        logger.info("OCR completed. Extracted %d characters (content not logged).", char_count)
        return raw_text
    except pytesseract.TesseractNotFoundError:
        logger.error("Tesseract binary not found. Install Tesseract and ensure it is on PATH.")
        raise
    except pytesseract.TesseractError as exc:
        # Strip the message to avoid leaking any embedded text
        logger.error("Tesseract returned an error (details suppressed for privacy).")
        raise RuntimeError("OCR processing failed.") from exc
    except Exception as exc:  # noqa: BLE001
        logger.error("Unexpected error during OCR (details suppressed for privacy).")
        raise RuntimeError("OCR processing failed unexpectedly.") from exc
