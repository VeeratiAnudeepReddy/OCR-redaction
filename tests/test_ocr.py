"""
Tests for backend/ocr.py

Tests Tesseract OCR functionality using the synthetic test image.
"""

from __future__ import annotations

import pathlib
import sys

import pytest
from PIL import Image

# Ensure project root is on the path when running with pytest from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.ocr import extract_text_from_image, preprocess_image

TEST_IMAGE_PATH = pathlib.Path(__file__).parent.parent / "test_data" / "test_input.png"


class TestPreprocessing:
    def test_rgb_conversion(self):
        """Grayscale image should be converted without error."""
        gray = Image.new("L", (100, 100), 128)
        result = preprocess_image(gray)
        assert result is not None

    def test_rgba_conversion(self):
        """RGBA image should be handled."""
        rgba = Image.new("RGBA", (100, 100), (255, 255, 255, 255))
        result = preprocess_image(rgba)
        assert result is not None

    def test_output_is_grayscale(self):
        """Preprocessed image should be grayscale (mode 'L')."""
        rgb = Image.new("RGB", (200, 200), (200, 200, 200))
        result = preprocess_image(rgb)
        assert result.mode == "L"

    def test_small_image_upscaled(self):
        """Images narrower than 1000 px should be upscaled."""
        small = Image.new("RGB", (100, 50), "white")
        result = preprocess_image(small)
        assert result.width >= 1000


class TestExtractText:
    def test_test_image_readable(self):
        """The synthetic test image should produce non-empty OCR output."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated — run create_test_image.py first")
        img = Image.open(TEST_IMAGE_PATH)
        text = extract_text_from_image(img)
        assert isinstance(text, str)
        assert len(text.strip()) > 10, "Expected meaningful OCR output, got very short text"

    def test_test_image_contains_expected_tokens(self):
        """OCR should recover key tokens from the test image."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        img = Image.open(TEST_IMAGE_PATH)
        text = extract_text_from_image(img).lower()
        # Check for non-PII tokens that should be preserved
        assert "computer science" in text or "course" in text, \
            f"Expected 'Computer Science' or 'Course' in OCR output"

    def test_extract_without_preprocess(self):
        """Should work without preprocessing."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        img = Image.open(TEST_IMAGE_PATH)
        text = extract_text_from_image(img, preprocess=False)
        assert isinstance(text, str)

    def test_blank_image_returns_string(self):
        """Blank white image should return empty/near-empty string, not raise."""
        blank = Image.new("RGB", (200, 200), "white")
        text = extract_text_from_image(blank)
        assert isinstance(text, str)
        # May return whitespace only — that's fine
        assert len(text) < 50, "Blank image should produce little/no text"
