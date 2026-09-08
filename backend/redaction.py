"""
backend/redaction.py — Visual PII redaction of PIL images.

Two methods are supported:
  - "blackbox" : Solid black rectangle (default)
  - "blur"     : Gaussian blur over detected PII regions

Privacy guarantees:
  * Raw OCR text is NEVER logged.
  * PII values are NEVER logged.
  * Only aggregate metadata (entity count, method, region counts) is logged.
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFilter

from backend.ocr import OCRResult, OCRWord
from backend.pii import DetectedEntity

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Padding added around each detected PII bounding box before blurring, so
#: text glyphs near the boundary edge do not remain legible.  Tune this
#: value if text bleeds through at the edges.
REDACTION_PADDING_PX: int = 5

#: Gaussian blur radius — must be strong enough to make text unreadable.
#: Radius 15 produces heavy blur; increase to 20+ for very small text.
BLUR_RADIUS: int = 15

#: Colour used for the blackbox method.
BLACKBOX_FILL: Tuple[int, int, int] = (0, 0, 0)

#: Valid redaction method identifiers.
VALID_METHODS = {"blur", "blackbox"}
DEFAULT_METHOD = "blackbox"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class PixelBox(NamedTuple):
    """A pixel-space bounding box (left, top, right, bottom)."""
    left: int
    top: int
    right: int
    bottom: int


def _text_span_to_pixel_boxes(
    text: str,
    start: int,
    end: int,
    words: List[OCRWord],
) -> List[PixelBox]:
    """
    Map a [start, end) character offset in *text* to pixel bounding boxes
    from the OCR word list.

    The strategy:
      1. Reconstruct which characters in *text* correspond to which OCRWord
         by replaying the text token-by-token.
      2. Collect all OCRWord objects whose character span overlaps [start, end).
      3. Return one PixelBox per matched word.

    This handles multi-word PII entities (e.g. "Peter Parker") by returning
    separate boxes for each word, which are individually blurred.  The caller
    may union them if desired.

    Returns an empty list if no words map into the span.
    """
    boxes: List[PixelBox] = []

    # Walk through the reconstructed text char-by-char using the same
    # tokenisation that Tesseract produces.  We cannot use str.find() because
    # the same word can appear multiple times, so we must track offsets.
    cursor = 0
    for word in words:
        # Each word occupies exactly len(word.text) characters in the full
        # text string, but the text string may have whitespace/newlines
        # between words.  Find this word's occurrence starting at cursor.
        word_text = word.text
        idx = text.find(word_text, cursor)
        if idx == -1:
            # Word not found at or after cursor — skip (can happen if OCR
            # word extraction used a different image variant than text extraction).
            continue

        w_start = idx
        w_end = idx + len(word_text)

        # Advance cursor past this word occurrence.
        cursor = w_end

        # Check for overlap with the PII span [start, end)
        if w_start < end and w_end > start:
            boxes.append(PixelBox(
                left=word.left,
                top=word.top,
                right=word.left + word.width,
                bottom=word.top + word.height,
            ))

    return boxes


def _pad_box(box: PixelBox, padding: int, img_width: int, img_height: int) -> PixelBox:
    """Expand a PixelBox by *padding* pixels, clamped to image dimensions."""
    return PixelBox(
        left=max(0, box.left - padding),
        top=max(0, box.top - padding),
        right=min(img_width, box.right + padding),
        bottom=min(img_height, box.bottom + padding),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def apply_blur_redaction(
    image: Image.Image,
    entities: Sequence[DetectedEntity],
    ocr_result: OCRResult,
    blur_radius: int = BLUR_RADIUS,
    padding: int = REDACTION_PADDING_PX,
) -> Tuple[Image.Image, int, int]:
    """
    Apply Gaussian blur over detected PII regions.

    For each entity that maps to one or more pixel bounding boxes, the
    corresponding region is cropped, blurred, and pasted back in-place.
    Regions without mappable bounding boxes are skipped with a WARNING log.

    Args:
        image:       Original PIL Image (any mode).
        entities:    PII entities with text offsets.
        ocr_result:  OCR result containing text and word-level bounding boxes.
        blur_radius: Gaussian blur radius (higher = stronger blur).
        padding:     Extra pixels added around each box before blurring.

    Returns:
        Tuple of (redacted_image, blurred_region_count, skipped_entity_count).
        Pixel values outside PII regions are pixel-for-pixel identical to the
        original image (no compression artefacts because we never re-encode here).
    """
    # Work on a copy so the original is untouched
    redacted = image.copy().convert("RGB")
    img_w, img_h = redacted.size
    text = ocr_result.text
    words = ocr_result.words

    blurred_count = 0
    skipped_count = 0

    for entity in entities:
        boxes = _text_span_to_pixel_boxes(text, entity.start, entity.end, words)

        if not boxes:
            logger.warning(
                "No pixel bounding box found for entity type=%s — visual redaction skipped for this entity.",
                entity.entity_type,
            )
            skipped_count += 1
            continue

        for box in boxes:
            padded = _pad_box(box, padding, img_w, img_h)

            # Skip degenerate (zero-area) boxes
            if padded.left >= padded.right or padded.top >= padded.bottom:
                continue

            region = redacted.crop((padded.left, padded.top, padded.right, padded.bottom))
            blurred_region = region.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            redacted.paste(blurred_region, (padded.left, padded.top))
            blurred_count += 1

    logger.info(
        "Blur redaction complete: method=blur | entities=%d | blurred_boxes=%d | skipped=%d.",
        len(entities),
        blurred_count,
        skipped_count,
    )
    return redacted, blurred_count, skipped_count


def apply_blackbox_redaction(
    image: Image.Image,
    entities: Sequence[DetectedEntity],
    ocr_result: OCRResult,
    fill: Tuple[int, int, int] = BLACKBOX_FILL,
    padding: int = REDACTION_PADDING_PX,
) -> Tuple[Image.Image, int, int]:
    """
    Apply solid black-box redaction over detected PII regions.

    Functionally equivalent to the old ImageRedactorEngine.redact() path
    but uses our own OCR word map so bounding boxes are consistent with
    the entities we detected.

    Returns:
        Tuple of (redacted_image, drawn_box_count, skipped_entity_count).
    """
    redacted = image.copy().convert("RGB")
    img_w, img_h = redacted.size
    text = ocr_result.text
    words = ocr_result.words
    draw = ImageDraw.Draw(redacted)

    drawn_count = 0
    skipped_count = 0

    for entity in entities:
        boxes = _text_span_to_pixel_boxes(text, entity.start, entity.end, words)

        if not boxes:
            logger.warning(
                "No pixel bounding box found for entity type=%s — visual redaction skipped.",
                entity.entity_type,
            )
            skipped_count += 1
            continue

        for box in boxes:
            padded = _pad_box(box, padding, img_w, img_h)
            if padded.left >= padded.right or padded.top >= padded.bottom:
                continue
            draw.rectangle(
                [padded.left, padded.top, padded.right, padded.bottom],
                fill=fill,
            )
            drawn_count += 1

    logger.info(
        "Blackbox redaction complete: method=blackbox | entities=%d | drawn_boxes=%d | skipped=%d.",
        len(entities),
        drawn_count,
        skipped_count,
    )
    return redacted, drawn_count, skipped_count


def redact_image(
    image: Image.Image,
    entities: Sequence[DetectedEntity],
    ocr_result: OCRResult,
    method: str = DEFAULT_METHOD,
) -> Image.Image:
    """
    Public entry point: redact detected PII regions in *image*.

    Args:
        image:      Original PIL Image.
        entities:   PII entities from :func:`backend.pii.process_text`.
        ocr_result: OCR result from :func:`backend.ocr.extract_ocr_result`.
        method:     ``"blur"`` (default) or ``"blackbox"``.

    Returns:
        Redacted PIL Image in RGB mode, same dimensions as input.

    Raises:
        ValueError: if *method* is not in :data:`VALID_METHODS`.
    """
    if method not in VALID_METHODS:
        raise ValueError(
            f"Unknown redaction method {method!r}. Valid options: {sorted(VALID_METHODS)}"
        )

    if method == "blur":
        result_image, _, _ = apply_blur_redaction(image, entities, ocr_result)
    else:  # "blackbox"
        result_image, _, _ = apply_blackbox_redaction(image, entities, ocr_result)

    return result_image
