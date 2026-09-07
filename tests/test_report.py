"""
tests/test_report.py — Tests for backend/report.py and the /process-image
report-generation integration.

Tests cover:
  1.  Report file is created on disk
  2.  Report contains sanitized text
  3.  Report contains detected entity types
  4.  Report contains confidence scores
  5.  Raw PII values are NOT in the report (core privacy assertion)
  6.  /process-image JSON still returns success, safe_text, entities_found
  7.  /process-image JSON now also returns report_file
  8.  /redact-image still works (no regression)
  9.  Empty / blank image produces a report with "(no text extracted)"
  10. Multiple PII entities all appear correctly in report
"""

from __future__ import annotations

import io
import json
import os
import pathlib
import sys
import textwrap
import tempfile

import pytest
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import backend.config as config
from backend.pii import AnalysisResult, DetectedEntity
from backend.report import _build_report, _report_stem, write_report

# ---------------------------------------------------------------------------
# Synthetic PII constants — fictional values only
# ---------------------------------------------------------------------------

SYNTHETIC_NAME = "John Smith"
SYNTHETIC_EMAIL = "john@example.com"
SYNTHETIC_PHONE = "+91 9876543210"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_png_bytes(width: int = 800, height: int = 200) -> bytes:
    """Create a blank white PNG image."""
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_result(
    safe_text: str = "",
    entities: list[DetectedEntity] | None = None,
) -> AnalysisResult:
    """Build a synthetic AnalysisResult for unit tests."""
    return AnalysisResult(safe_text=safe_text, entities=entities or [])


# ---------------------------------------------------------------------------
# Flask test client
# ---------------------------------------------------------------------------

from backend.app import app as flask_app


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["MAX_CONTENT_LENGTH"] = None
    with flask_app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Unit tests for backend/report.py
# ---------------------------------------------------------------------------


class TestReportStem:
    def test_simple_name(self):
        assert _report_stem("screenshot.png") == "screenshot_report"

    def test_jpeg_extension(self):
        assert _report_stem("photo.jpeg") == "photo_report"

    def test_spaces_replaced(self):
        stem = _report_stem("WhatsApp Image 2026.jpeg")
        assert " " not in stem
        assert stem.endswith("_report")

    def test_empty_filename(self):
        assert _report_stem("") == "image_report"


class TestBuildReport:
    """Test _build_report() string output directly (no disk I/O)."""

    def _report(self, safe_text="", entities=None, debug_mode=False):
        return _build_report(
            image_filename="test.png",
            safe_text=safe_text,
            entities=entities or [],
            debug_mode=debug_mode,
        )

    # ── Section headers ─────────────────────────────────────────────────────

    def test_header_present(self):
        r = self._report()
        assert "OCR + PII PRIVACY REPORT" in r

    def test_all_sections_present(self):
        r = self._report()
        assert "1. EXTRACTED TEXT" in r
        assert "2. DETECTED PERSONAL / PRIVATE DATA" in r
        assert "3. SANITIZED TEXT" in r
        assert "4. PRIVACY CHECK" in r

    def test_input_filename_in_report(self):
        r = self._report()
        assert "test.png" in r

    # ── Safe text ────────────────────────────────────────────────────────────

    def test_safe_text_appears_in_report(self):
        r = self._report(safe_text="Hello [PERSON] your email is [EMAIL_ADDRESS]")
        assert "[PERSON]" in r
        assert "[EMAIL_ADDRESS]" in r

    def test_no_text_placeholder_when_empty(self):
        r = self._report(safe_text="")
        assert "(no text extracted)" in r

    def test_whitespace_only_shows_no_text(self):
        r = self._report(safe_text="   \n  ")
        assert "(no text extracted)" in r

    # ── Entity listing ───────────────────────────────────────────────────────

    def test_entity_type_in_report(self):
        entities = [DetectedEntity(entity_type="PERSON", score=0.85, start=0, end=5)]
        r = self._report(entities=entities)
        assert "PERSON" in r

    def test_entity_confidence_in_report(self):
        entities = [DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=0, end=10)]
        r = self._report(entities=entities)
        assert "1.00" in r

    def test_entity_replacement_token_in_report(self):
        entities = [DetectedEntity(entity_type="PHONE_NUMBER", score=0.75, start=0, end=15)]
        r = self._report(entities=entities)
        assert "[PHONE_NUMBER]" in r

    def test_multiple_entities_all_listed(self):
        entities = [
            DetectedEntity(entity_type="PERSON", score=0.85, start=0, end=5),
            DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=10, end=30),
            DetectedEntity(entity_type="PHONE_NUMBER", score=0.75, start=40, end=55),
        ]
        r = self._report(entities=entities)
        assert "PERSON" in r
        assert "EMAIL_ADDRESS" in r
        assert "PHONE_NUMBER" in r
        assert "Total entities detected: 3" in r

    def test_no_entities_message(self):
        r = self._report(entities=[])
        assert "Total entities detected: 0" in r

    # ── Privacy check section ────────────────────────────────────────────────

    def test_privacy_check(self):
        r = self._report()
        assert "DEBUG PII REPORT: DISABLED" in r
        assert "Raw PII written to report: NO" in r
        assert "PII detection completed: YES" in r
        assert "OCR completed: YES" in r



class TestWriteReport:
    """Test write_report() file creation."""

    def test_file_is_created(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        result = _make_result(safe_text="Hello [PERSON]")
        path = write_report("test.png", result)
        assert pathlib.Path(path).exists()

    def test_returned_path_ends_with_report_txt(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        result = _make_result(safe_text="Hello")
        path = write_report("my_screenshot.png", result)
        assert path.endswith("_report.txt")

    def test_report_contains_safe_text(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        result = _make_result(safe_text="[PERSON] visited [URL]")
        path = write_report("test.png", result)
        content = pathlib.Path(path).read_text(encoding="utf-8")
        assert "[PERSON]" in content
        assert "[URL]" in content

    def test_report_contains_entity_types(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        entities = [
            DetectedEntity(entity_type="PERSON", score=0.85, start=0, end=5),
            DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=10, end=30),
        ]
        result = _make_result(
            safe_text="[PERSON] email [EMAIL_ADDRESS]",
            entities=entities,
        )
        path = write_report("test.png", result)
        content = pathlib.Path(path).read_text(encoding="utf-8")
        assert "PERSON" in content
        assert "EMAIL_ADDRESS" in content

    def test_report_contains_confidence_scores(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        entities = [DetectedEntity(entity_type="PERSON", score=0.85, start=0, end=5)]
        result = _make_result(safe_text="[PERSON]", entities=entities)
        path = write_report("test.png", result)
        content = pathlib.Path(path).read_text(encoding="utf-8")
        assert "0.85" in content

    # ── Core privacy assertion ───────────────────────────────────────────────

    def test_raw_pii_not_in_report(self, tmp_path, monkeypatch):
        """
        CRITICAL PRIVACY TEST.

        The report is built from already-anonymised data.  Raw PII values
        must NEVER appear — they are never passed into write_report().
        """
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        # Simulate: safe_text already has PII replaced
        safe_text = (
            "Name: [PERSON]\n"
            "Email: [EMAIL_ADDRESS]\n"
            "Phone: [PHONE_NUMBER]\n"
        )
        entities = [
            DetectedEntity(entity_type="PERSON", score=0.85, start=5, end=16),
            DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=23, end=42),
            DetectedEntity(entity_type="PHONE_NUMBER", score=0.75, start=50, end=65),
        ]
        result = _make_result(safe_text=safe_text, entities=entities)
        path = write_report("test.png", result)
        content = pathlib.Path(path).read_text(encoding="utf-8")

        # Placeholders must be present
        assert "[PERSON]" in content
        assert "[EMAIL_ADDRESS]" in content
        assert "[PHONE_NUMBER]" in content

        # Raw PII must NOT appear (they were never passed in)
        assert SYNTHETIC_NAME not in content, "PRIVACY VIOLATION: raw name in report"
        assert SYNTHETIC_EMAIL not in content, "PRIVACY VIOLATION: raw email in report"
        assert SYNTHETIC_PHONE not in content, "PRIVACY VIOLATION: raw phone in report"
        # Partial values
        assert "john@example.com" not in content
        assert "9876543210" not in content

    def test_empty_image_report_has_no_text_placeholder(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        result = _make_result(safe_text="")
        path = write_report("blank.png", result)
        content = pathlib.Path(path).read_text(encoding="utf-8")
        assert "(no text extracted)" in content


# ---------------------------------------------------------------------------
# Debug PII Report tests (DEBUG_PII_REPORT=false vs true)
# ---------------------------------------------------------------------------


class TestDebugPiiReportDisabled:
    """Verify behavior when DEBUG_PII_REPORT=false (default)."""

    def test_report_replaces_values_with_hidden(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        entities = [
            DetectedEntity(entity_type="PERSON", score=0.85, start=6, end=16, value="John Smith"),
            DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=24, end=40, value="john@example.com"),
            DetectedEntity(entity_type="PHONE_NUMBER", score=0.92, start=48, end=63, value="+91 9876543210"),
        ]
        result = _make_result(
            safe_text="Name: [PERSON]\nEmail: [EMAIL_ADDRESS]\nPhone: [PHONE_NUMBER]",
            entities=entities,
        )
        path = write_report("test.png", result, debug_pii_report=False)
        content = pathlib.Path(path).read_text(encoding="utf-8")

        assert "Value: [HIDDEN]" in content
        assert "John Smith" not in content
        assert "john@example.com" not in content
        assert "+91 9876543210" not in content
        assert "DEBUG PII REPORT: DISABLED" in content
        assert "Raw PII written to report: NO" in content


class TestDebugPiiReportEnabled:
    """Verify behavior when DEBUG_PII_REPORT=true (local dev / debug mode)."""

    def test_report_includes_raw_pii_values(self, tmp_path, monkeypatch):
        monkeypatch.setattr(config, "REPORT_OUTPUT_DIR", str(tmp_path))
        entities = [
            DetectedEntity(entity_type="PERSON", score=0.85, start=6, end=16, value="John Smith"),
            DetectedEntity(entity_type="EMAIL_ADDRESS", score=1.0, start=24, end=40, value="john@example.com"),
            DetectedEntity(entity_type="PHONE_NUMBER", score=0.92, start=48, end=63, value="+91 9876543210"),
        ]
        result = _make_result(
            safe_text="Name: [PERSON]\nEmail: [EMAIL_ADDRESS]\nPhone: [PHONE_NUMBER]",
            entities=entities,
        )
        path = write_report("test.png", result, debug_pii_report=True)
        content = pathlib.Path(path).read_text(encoding="utf-8")

        assert "Value: John Smith" in content
        assert "Value: john@example.com" in content
        assert "Value: +91 9876543210" in content
        assert "PERSON" in content
        assert "EMAIL_ADDRESS" in content
        assert "PHONE_NUMBER" in content
        assert "0.85" in content
        assert "[PERSON]" in content
        assert "[EMAIL_ADDRESS]" in content
        assert "[PHONE_NUMBER]" in content
        assert "DEBUG PII REPORT: ENABLED" in content
        assert "Raw PII written to report: YES" in content


class TestApiPrivacyGuarantee:
    """Verify /process-image API response NEVER exposes raw PII values even when debug mode is enabled."""

    def test_api_json_does_not_contain_raw_pii_values(self, client, monkeypatch):
        monkeypatch.setenv("DEBUG_PII_REPORT", "true")
        resp = client.post(
            "/process-image",
            headers={"X-Debug-PII-Report": "true"},
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
        for entity in data.get("entities_found", []):
            assert "value" not in entity



# ---------------------------------------------------------------------------
# Integration tests — /process-image API
# ---------------------------------------------------------------------------


class TestProcessImageReportIntegration:
    """Verify /process-image generates a report and includes report_file in JSON."""

    def test_json_still_has_success_key(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True

    def test_json_still_has_safe_text(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        assert "safe_text" in data

    def test_json_still_has_entities_found(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        assert "entities_found" in data

    def test_json_includes_report_file(self, client):
        """NEW: /process-image must now return report_file in JSON."""
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        assert "report_file" in data, (
            "report_file missing from /process-image response"
        )
        assert data["report_file"].endswith("_report.txt")

    def test_report_file_is_created_on_disk(self, client):
        """Report file referenced in JSON must actually exist on disk."""
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "test_disk.png")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        report_path = data.get("report_file", "")
        assert report_path, "No report_file in response"
        assert pathlib.Path(report_path).exists(), (
            f"Report file not found on disk: {report_path}"
        )

    def test_report_raw_pii_not_in_json_response(self, client):
        """Existing privacy check: raw PII must not appear in JSON."""
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        response_text = resp.data.decode("utf-8")
        forbidden = {"raw_text", "original_text", "pii_values", "ocr_text"}
        assert not any(k in response_text for k in forbidden)

    # ── /redact-image regression ─────────────────────────────────────────────

    def test_redact_image_still_works(self, client):
        """Ensure /redact-image is unaffected by the report changes."""
        resp = client.post(
            "/redact-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "test.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code in (200, 501)  # 501 if presidio-image-redactor absent

    # ── Multiple PII entities ────────────────────────────────────────────────

    def test_report_multiple_entities_present(self, client):
        """
        When a real test image with multiple PII types exists, verify all
        entity types appear in the report file.
        """
        test_img = pathlib.Path(__file__).parent.parent / "test_data" / "test_input.png"
        if not test_img.exists():
            pytest.skip("test_input.png not generated yet")

        with open(test_img, "rb") as fh:
            resp = client.post(
                "/process-image",
                data={"image": (fh, "test_input.png")},
                content_type="multipart/form-data",
            )

        data = json.loads(resp.data)
        assert data["success"] is True

        report_path = data.get("report_file", "")
        assert report_path, "No report_file in response"

        content = pathlib.Path(report_path).read_text(encoding="utf-8")

        # All detected entity types must appear in the report
        detected_types = {e["type"] for e in data["entities_found"]}
        for etype in detected_types:
            assert etype in content, (
                f"Entity type '{etype}' missing from report"
            )

    def test_report_does_not_contain_raw_email(self, client):
        """
        PRIVACY: report must not contain the raw synthetic email from test_input.png.
        """
        test_img = pathlib.Path(__file__).parent.parent / "test_data" / "test_input.png"
        if not test_img.exists():
            pytest.skip("test_input.png not generated yet")

        with open(test_img, "rb") as fh:
            resp = client.post(
                "/process-image",
                data={"image": (fh, "test_input.png")},
                content_type="multipart/form-data",
            )

        data = json.loads(resp.data)
        report_path = data.get("report_file", "")
        assert report_path, "No report_file in response"

        content = pathlib.Path(report_path).read_text(encoding="utf-8")
        assert "peter.parker@example.com" not in content, (
            "PRIVACY VIOLATION: raw email found in report file"
        )
        assert "9876543210" not in content, (
            "PRIVACY VIOLATION: raw phone found in report file"
        )
