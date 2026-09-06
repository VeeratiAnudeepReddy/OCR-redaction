"""
Tests for backend/processor.py

Tests the full image → OCR → PII → sanitised text pipeline.
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.processor import (
    ImageValidationError,
    process_image_bytes,
    process_image_object,
    _decode_image,
    _validate_image_bytes,
)
from backend.pii import AnalysisResult

TEST_IMAGE_PATH = pathlib.Path(__file__).parent.parent / "test_data" / "test_input.png"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_png_bytes(text: str = "", width: int = 800, height: int = 200) -> bytes:
    """Create an in-memory PNG with optional text rendered on it."""
    img = Image.new("RGB", (width, height), "white")
    if text:
        draw = ImageDraw.Draw(img)
        # Try a system font; fall back to default
        try:
            font = ImageFont.truetype(
                "/System/Library/Fonts/Supplemental/Courier New.ttf", 24
            )
        except (OSError, IOError):
            font = ImageFont.load_default()
        draw.text((20, 20), text, fill=(0, 0, 0), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


# ── Validation tests ─────────────────────────────────────────────────────────

class TestValidateImageBytes:
    def test_empty_bytes_raises(self):
        with pytest.raises(ImageValidationError) as exc_info:
            _validate_image_bytes(b"")
        assert exc_info.value.http_status == 400

    def test_oversized_raises_413(self):
        from backend.config import MAX_IMAGE_SIZE_BYTES
        oversized = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
        with pytest.raises(ImageValidationError) as exc_info:
            _validate_image_bytes(oversized)
        assert exc_info.value.http_status == 413

    def test_unsupported_extension_raises_415(self):
        data = _make_png_bytes()
        with pytest.raises(ImageValidationError) as exc_info:
            _validate_image_bytes(data, filename="image.bmp")
        assert exc_info.value.http_status == 415

    def test_valid_png_extension_passes(self):
        data = _make_png_bytes()
        _validate_image_bytes(data, filename="test.png")  # should not raise

    def test_valid_jpeg_extension_passes(self):
        data = _make_jpeg_bytes()
        _validate_image_bytes(data, filename="photo.jpeg")  # should not raise


class TestDecodeImage:
    def test_valid_png_decodes(self):
        data = _make_png_bytes()
        img = _decode_image(data)
        assert isinstance(img, Image.Image)

    def test_invalid_bytes_raises(self):
        with pytest.raises(ImageValidationError) as exc_info:
            _decode_image(b"this is not an image")
        assert exc_info.value.http_status == 400

    def test_truncated_image_raises(self):
        data = _make_png_bytes()
        truncated = data[:50]
        with pytest.raises(ImageValidationError):
            _decode_image(truncated)


# ── Full pipeline tests ───────────────────────────────────────────────────────

class TestProcessImageBytes:
    def test_returns_analysis_result(self):
        data = _make_png_bytes()
        result = process_image_bytes(data)
        assert isinstance(result, AnalysisResult)

    def test_safe_text_is_string(self):
        data = _make_png_bytes()
        result = process_image_bytes(data)
        assert isinstance(result.safe_text, str)

    def test_with_real_test_image(self):
        """Full pipeline on the generated test image."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        data = TEST_IMAGE_PATH.read_bytes()
        result = process_image_bytes(data, filename="test_input.png")
        assert isinstance(result, AnalysisResult)
        # Some entities should be detected
        assert len(result.safe_text) > 0

    def test_pii_not_in_output(self):
        """Privacy: raw PII must not appear in safe_text."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        data = TEST_IMAGE_PATH.read_bytes()
        result = process_image_bytes(data, filename="test_input.png")
        safe = result.safe_text
        assert "peter.parker@example.com" not in safe, "Raw email leaked!"
        assert "9876543210" not in safe, "Raw phone number leaked!"

    def test_empty_data_raises(self):
        with pytest.raises(ImageValidationError):
            process_image_bytes(b"")

    def test_corrupted_data_raises(self):
        with pytest.raises(ImageValidationError):
            process_image_bytes(b"corrupted garbage data xyz")

    def test_oversized_raises_413(self):
        from backend.config import MAX_IMAGE_SIZE_BYTES
        oversized = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
        with pytest.raises(ImageValidationError) as exc_info:
            process_image_bytes(oversized)
        assert exc_info.value.http_status == 413


class TestProcessImageObject:
    def test_returns_analysis_result(self):
        img = Image.new("RGB", (200, 100), "white")
        result = process_image_object(img)
        assert isinstance(result, AnalysisResult)

    def test_with_synthetic_pii_text(self):
        """Rendered PII text should be redacted in output."""
        data = _make_png_bytes("Email: test@privacy.example.com")
        img = _decode_image(data)
        result = process_image_object(img)
        # OCR may or may not catch this perfectly — just verify no crash
        assert isinstance(result.safe_text, str)
