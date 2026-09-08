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
from dataclasses import dataclass, field
from typing import List, Tuple

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public OCR Data Structures
# ---------------------------------------------------------------------------

@dataclass
class OCRWord:
    """Bounding box and line location of an OCR word with character offsets."""
    text: str
    left: int
    top: int
    width: int
    height: int
    line_num: int
    block_num: int = 0
    start: int = 0
    end: int = 0


@dataclass
class OCRResult:
    """Structured OCR result containing full text and word-level bounding boxes."""
    text: str
    words: List[OCRWord] = field(default_factory=list)


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


def _extract_words_and_text(
    image: Image.Image,
    lang: str,
    config_str: str,
) -> Tuple[str, List[OCRWord]]:
    """
    Extract OCRWord bounding boxes and construct the full text string in a single pass.

    Builds the text by grouping words by (block_num, par_num, line_num), joining words
    with a single space and lines with a newline. Every OCRWord is assigned its exact
    character start and end offsets in that string, guaranteeing by construction that
    text[word.start:word.end] == word.text.
    """
    try:
        data = pytesseract.image_to_data(
            image, lang=lang, config=config_str, output_type=pytesseract.Output.DICT
        )
        n = len(data.get("text", []))
        if n == 0:
            return "", []

        lines: dict[Tuple[int, int, int], List[OCRWord]] = {}
        for i in range(n):
            w_text = data["text"][i].strip()
            if not w_text:
                continue
            b = data.get("block_num", [0] * n)[i]
            p = data.get("par_num", [0] * n)[i]
            l = data["line_num"][i]
            key = (b, p, l)
            word = OCRWord(
                text=w_text,
                left=data["left"][i],
                top=data["top"][i],
                width=data["width"][i],
                height=data["height"][i],
                line_num=l,
                block_num=b,
            )
            lines.setdefault(key, []).append(word)

        if not lines:
            return "", []

        full_text_parts: List[str] = []
        all_words: List[OCRWord] = []
        curr_offset = 0

        for key, line_words in lines.items():
            for j, w in enumerate(line_words):
                w.start = curr_offset
                w.end = curr_offset + len(w.text)
                all_words.append(w)
                curr_offset = w.end
                if j < len(line_words) - 1:
                    curr_offset += 1  # space between words

            line_str = " ".join(w.text for w in line_words)
            full_text_parts.append(line_str)
            curr_offset += 1  # newline after line

        final_text = "\n".join(full_text_parts) + "\n" if full_text_parts else ""
        return final_text, all_words
    except Exception:  # noqa: BLE001
        return "", []


def _extract_words(image: Image.Image, lang: str, config_str: str) -> List[OCRWord]:
    """Helper to extract OCRWord bounding boxes via image_to_data."""
    _, words = _extract_words_and_text(image, lang=lang, config_str=config_str)
    return words


def extract_ocr_result(
    image: Image.Image,
    preprocess: bool = True,
    lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> OCRResult:
    """
    Extract text and word-level bounding box data from a PIL Image.

    Returns:
        :class:`OCRResult` containing full text string and structured word positions
        with exact character start and end offsets.
    """
    if not preprocess:
        try:
            raw_text, words = _extract_words_and_text(
                image, lang=lang, config_str=tesseract_config,
            )
            logger.info(
                "OCR (single-pass) completed. Extracted %d characters (content not logged).",
                len(raw_text.strip()),
            )
            return OCRResult(text=raw_text, words=words)
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
        "--oem 3 --psm 11",  # sparse text — best for UI screenshots
        "--oem 3 --psm 6",   # uniform block of text
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

    best_score: float = -1.0
    best_label: str = "none"
    best_variant_img: Image.Image = image
    best_cfg: str = tesseract_config

    for vi, variant_img in enumerate(variants):
        for cfg in _PSM_CONFIGS:
            psm_num = cfg.split("--psm ")[-1].strip()
            label = f"v{vi}({_VARIANT_LABELS[vi]})+psm{psm_num}"
            text, score = _run_ocr(variant_img, cfg, lang)
            if score > best_score:
                best_score = score
                best_label = label
                best_variant_img = variant_img
                best_cfg = cfg

    # If every variant scored 0, fall back to the standard preprocess_image path
    if best_score <= 0.0:
        fallback_img = preprocess_image(image)
        best_variant_img = fallback_img
        best_cfg = tesseract_config
        best_label = "fallback(preprocess_image)"

    # Extract words and synchronized text from the winning variant in a single pass
    best_text, words = _extract_words_and_text(best_variant_img, lang=lang, config_str=best_cfg)

    # ── Coordinate normalisation ──────────────────────────────────────────────
    # _extract_words_and_text returns pixel coordinates in the space of best_variant_img,
    # which may have been upscaled relative to the original *image*. Divide bounding-box
    # values by the scale factor while preserving exact text offsets (start, end).
    orig_w, orig_h = image.size
    var_w, var_h = best_variant_img.size
    if var_w != orig_w or var_h != orig_h:
        scale_x = orig_w / var_w
        scale_y = orig_h / var_h
        words = [
            OCRWord(
                text=w.text,
                left=round(w.left * scale_x),
                top=round(w.top * scale_y),
                width=max(1, round(w.width * scale_x)),
                height=max(1, round(w.height * scale_y)),
                line_num=w.line_num,
                block_num=w.block_num,
                start=w.start,
                end=w.end,
            )
            for w in words
        ]
        logger.debug(
            "Scaled OCR word coordinates from variant size %dx%d back to original %dx%d.",
            var_w, var_h, orig_w, orig_h,
        )

    logger.info(
        "OCR (multi-variant) completed. Winner: %s | quality=%.3f | chars=%d | words_found=%d.",
        best_label,
        max(best_score, 0.0),
        len(best_text.strip()),
        len(words),
    )
    return OCRResult(text=best_text, words=words)



def extract_text_from_image(
    image: Image.Image,
    preprocess: bool = True,
    lang: str = "eng",
    tesseract_config: str = "--oem 3 --psm 6",
) -> str:
    """
    Extract text from a PIL Image using Tesseract OCR.
    """
    return extract_ocr_result(
        image=image,
        preprocess=preprocess,
        lang=lang,
        tesseract_config=tesseract_config,
    ).text

