"""
OCR module — responsible ONLY for extracting text from images via Tesseract.

Privacy guarantee: raw OCR output is NEVER logged.

Preprocessing strategy (multi-variant):
    Dark-background screenshots (e.g. YouTube Studio, VS Code dark theme) produce
    garbled OCR because Tesseract expects dark text on a light background.  This
    module generates several preprocessed variants of each image and picks the one
    that produces the highest-quality OCR result using an objective heuristic
    (ratio of recognisable word-like tokens), without ever logging the text itself.
"""

from __future__ import annotations

import logging
import os
import shutil
from typing import List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract binary auto-discovery (Windows fallback)
# ---------------------------------------------------------------------------

if not shutil.which("tesseract"):
    _possible_paths = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for _p in _possible_paths:
        if os.path.isfile(_p):
            pytesseract.pytesseract.tesseract_cmd = _p
            _tessdata = os.path.join(os.path.dirname(_p), "tessdata")
            if os.path.isdir(_tessdata) and "TESSDATA_PREFIX" not in os.environ:
                os.environ["TESSDATA_PREFIX"] = _tessdata
            break


# ---------------------------------------------------------------------------
# Internal helpers — image analysis
# ---------------------------------------------------------------------------

def _is_dark_background(image: Image.Image) -> bool:
    """
    Return True if the image has a predominantly dark background.

    Converts to grayscale and checks whether the mean pixel value of the
    central 50% crop is below 128.  Dark-background images need inversion
    before Tesseract can read them reliably.
    """
    gray = image.convert("L") if image.mode != "L" else image
    w, h = gray.size
    cx, cy = w // 2, h // 2
    strip_w = max(w // 2, 1)
    strip_h = max(h // 2, 1)
    box = (
        cx - strip_w // 2,
        cy - strip_h // 2,
        cx + strip_w // 2,
        cy + strip_h // 2,
    )
    sample = gray.crop(box)
    from PIL import ImageStat
    mean = ImageStat.Stat(sample).mean[0]
    return mean < 128



# ---------------------------------------------------------------------------
# Internal helpers — image transforms
# ---------------------------------------------------------------------------

def _to_rgb(image: Image.Image) -> Image.Image:
    """Normalise any image mode to RGB (handles L, RGBA, P, CMYK …)."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _to_grayscale(image: Image.Image) -> Image.Image:
    """Convert to grayscale (mode L)."""
    return image.convert("L")


def _smart_upscale(image: Image.Image, min_long_edge: int = 1500) -> Image.Image:
    """
    Upscale so the *long edge* is at least *min_long_edge* pixels.

    Using the long edge (rather than width only) correctly handles portrait
    screenshots and square crops.  LANCZOS gives the sharpest upscale.
    """
    w, h = image.size
    long_edge = max(w, h)
    if long_edge < min_long_edge:
        scale = min_long_edge / long_edge
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
        logger.debug(
            "Upscaled image from %dx%d to %dx%d (long-edge rule)",
            w, h, *new_size,
        )
    return image


def _upscale_if_small(image: Image.Image, min_width: int = 1000) -> Image.Image:
    """
    Upscale images that are smaller than *min_width* pixels wide.

    Kept for backward compatibility.  New code should prefer :func:`_smart_upscale`.
    """
    w, h = image.size
    if w < min_width:
        scale = min_width / w
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
        logger.debug("Upscaled image from %dx%d to %dx%d", w, h, *new_size)
    return image


def _enhance_contrast(image: Image.Image, factor: float = 1.5) -> Image.Image:
    """Boost contrast to help OCR separate text from background."""
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)


def _sharpen(image: Image.Image) -> Image.Image:
    """Apply a mild sharpening filter."""
    return image.filter(ImageFilter.SHARPEN)


# ---------------------------------------------------------------------------
# preprocess_image — public single-image pipeline (used by tests & fallback)
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply a smart single-pass preprocessing pipeline and return a grayscale image.

    Steps:
        1. Normalise to RGB (handles L, RGBA, P, …)
        2. Smart-upscale so the long edge >= 1500 px
        3. Convert to grayscale
        4. Auto-invert if the image has a dark background
        5. Enhance contrast (factor 1.4)
        6. Sharpen

    Returns:
        A PIL Image in mode ``"L"`` (grayscale), always.
    """
    image = _to_rgb(image)
    image = _smart_upscale(image)          # uses long-edge rule
    image = _to_grayscale(image)
    if _is_dark_background(image):
        image = ImageOps.invert(image)
        logger.debug("Dark background detected — inverted image for OCR.")
    image = _enhance_contrast(image, factor=1.4)
    image = _sharpen(image)
    return image


# ---------------------------------------------------------------------------
# Multi-variant candidate generation
# ---------------------------------------------------------------------------

def _build_variants(image: Image.Image) -> List[Image.Image]:
    """
    Return up to 4 preprocessed grayscale candidates.

    Variant 0 — plain grayscale (good for clean light-background images)
    Variant 1 — grayscale + invert (good for dark-background / UI screenshots)
    Variant 2 — upscaled + contrast (good for small / low-res images)
    Variant 3 — upscaled + invert + contrast (dark + small combined)
    """
    rgb = _to_rgb(image)

    v0 = _to_grayscale(rgb)
    v1 = ImageOps.invert(v0.copy())

    v2_base = _smart_upscale(_to_grayscale(rgb.copy()))
    v2 = _sharpen(_enhance_contrast(v2_base.copy(), factor=1.5))
    v3 = _sharpen(_enhance_contrast(ImageOps.invert(v2_base.copy()), factor=1.5))

    return [v0, v1, v2, v3]


# ---------------------------------------------------------------------------
# OCR quality heuristic
# ---------------------------------------------------------------------------

def _ocr_quality_score(text: str) -> float:
    """
    Return an objective quality score in [0, 1] for an OCR result string.

    Higher is better. Evaluates the ratio of clean word/number-like tokens after
    stripping punctuation, with a bonus for more extracted text content.
    """
    import string
    tokens = text.split()
    if not tokens:
        return 0.0

    punc = string.punctuation + "—_«»©®"

    def _wordlike(tok: str) -> bool:
        c = tok.strip(punc)
        if not c:
            return False
        if c.isdigit() and len(c) <= 6:
            return True
        alpha = sum(ch.isalpha() for ch in c)
        return len(c) >= 2 and alpha / len(c) >= 0.6

    content_tokens = [t for t in tokens if any(ch.isalnum() for ch in t)]
    if not content_tokens:
        return 0.0

    word_ratio = sum(1 for t in content_tokens if _wordlike(t)) / len(content_tokens)
    length_bonus = min(len(content_tokens) / 100.0, 1.0) * 0.2
    return min(word_ratio * 0.8 + length_bonus, 1.0)


# ---------------------------------------------------------------------------
# Safe single-variant OCR runner
# ---------------------------------------------------------------------------

def _run_ocr(image: Image.Image, config_str: str, lang: str) -> Tuple[str, float]:
    """
    Run Tesseract on *image* and return ``(text, quality_score)``.

    Returns ``("", 0.0)`` on any error so a single bad variant never aborts
    the whole multi-variant process.
    """
    try:
        text: str = pytesseract.image_to_string(image, lang=lang, config=config_str)
        return text, _ocr_quality_score(text)
    except Exception:  # noqa: BLE001
        return "", 0.0


# ---------------------------------------------------------------------------
# Public OCR entry-point
# ---------------------------------------------------------------------------

def extract_text_from_image(
    image: Image.Image,
    preprocess: bool = True,
    lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.

    When *preprocess* is ``True`` (default), the function generates 4 image
    variants and tests each with 2 PSM modes (``--psm 6`` block / ``--psm 11``
    sparse), picking the combination with the highest quality score.

    When *preprocess* is ``False``, a single OCR pass is run on the raw image
    using *tesseract_config* as supplied (backward-compatible).

    Args:
        image: A PIL Image object.
        preprocess: Whether to run the multi-variant preprocessing pipeline.
        lang: Tesseract language code(s), e.g. ``"eng"`` or ``"eng+fra"``.
        tesseract_config: Tesseract flags used when ``preprocess=False`` or as
            a fallback if all multi-variant attempts score 0.

    Returns:
        Extracted text as a plain string.  May be empty if no text is found.

    Raises:
        pytesseract.TesseractNotFoundError: if the Tesseract binary is not installed.
        RuntimeError: for unexpected OCR-related failures.

    Privacy note:
        Raw OCR text is NEVER logged.  Only aggregate metadata (character count,
        winning variant label, quality score) is logged at INFO level.
    """
    if not preprocess:
        # ── Single-pass fallback (backward-compatible) ────────────────────────
        try:
            raw_text: str = pytesseract.image_to_string(
                image, lang=lang, config=tesseract_config,
            )
            logger.info(
                "OCR (single-pass) completed. Extracted %d characters (content not logged).",
                len(raw_text.strip()),
            )
            return raw_text
        except pytesseract.TesseractNotFoundError:
            logger.error(
                "Tesseract binary not found. Install Tesseract and ensure it is on PATH."
            )
            raise
        except pytesseract.TesseractError as exc:
            logger.error("Tesseract returned an error (details suppressed for privacy).")
            raise RuntimeError("OCR processing failed.") from exc
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error during OCR (details suppressed for privacy).")
            raise RuntimeError("OCR processing failed unexpectedly.") from exc

    # ── Multi-variant path ────────────────────────────────────────────────────
    _PSM_CONFIGS = [
        "--oem 3 --psm 6",   # uniform block of text
        "--oem 3 --psm 11",  # sparse text — best for UI screenshots
    ]
    _VARIANT_LABELS = [
        "gray",
        "gray+inv",
        "upscale+contrast",
        "upscale+inv+contrast",
    ]

    try:
        variants = _build_variants(image)
    except pytesseract.TesseractNotFoundError:
        logger.error(
            "Tesseract binary not found. Install Tesseract and ensure it is on PATH."
        )
        raise
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to build image variants (details suppressed for privacy).")
        raise RuntimeError("OCR preprocessing failed.") from exc

    best_text: str = ""
    best_score: float = -1.0
    best_label: str = "none"

    for vi, variant_img in enumerate(variants):
        for cfg in _PSM_CONFIGS:
            psm_num = cfg.split("--psm ")[-1].strip()
            label = f"v{vi}({_VARIANT_LABELS[vi]})+psm{psm_num}"
            text, score = _run_ocr(variant_img, cfg, lang)
            if score > best_score:
                best_score = score
                best_text = text
                best_label = label

    # If every variant scored 0, fall back to the standard preprocess_image path
    if best_score <= 0.0:
        fallback_img = preprocess_image(image)
        best_text, _ = _run_ocr(fallback_img, tesseract_config, lang)
        best_label = "fallback(preprocess_image)"

    logger.info(
        "OCR (multi-variant) completed. Winner: %s | quality=%.3f | chars=%d.",
        best_label,
        max(best_score, 0.0),
        len(best_text.strip()),
    )
    return best_text
