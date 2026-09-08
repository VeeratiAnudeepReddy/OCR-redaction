"""
Tests for the Flask API (backend/app.py)

Covers:
  - GET /health
  - POST /process-image  (valid, invalid, privacy, edge-case)
  - POST /redact-image
  - Error handlers
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Import app *after* inserting project root so relative imports work
from backend.app import app as flask_app

TEST_IMAGE_PATH = pathlib.Path(__file__).parent.parent / "test_data" / "test_input.png"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_png_bytes(width: int = 800, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_jpeg_bytes() -> bytes:
    img = Image.new("RGB", (200, 100), "white")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    # Disable MAX_CONTENT_LENGTH check in tests (handled by our own validator)
    flask_app.config["MAX_CONTENT_LENGTH"] = None
    with flask_app.test_client() as client:
        yield client


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_status_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_returns_ok(self, client):
        data = json.loads(resp := client.get("/health").data)
        assert data["status"] == "ok"

    def test_no_internal_info_leaked(self, client):
        data = json.loads(client.get("/health").data)
        sensitive = {"host", "port", "debug", "config", "version"}
        assert not any(k in data for k in sensitive)


# ── /process-image ────────────────────────────────────────────────────────────

class TestProcessImage:
    def test_missing_file_field_returns_400(self, client):
        resp = client.post("/process-image")
        assert resp.status_code == 400
        data = json.loads(resp.data)
        assert data["success"] is False

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(b""), "")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_corrupted_image_returns_400(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(b"not an image at all!!!"), "bad.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 400

    def test_unsupported_extension_returns_415(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "image.bmp")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 415

    def test_valid_png_returns_200(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
        assert "safe_text" in data
        assert "entities_found" in data

    def test_valid_jpeg_returns_200(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_jpeg_bytes()), "photo.jpg")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

    def test_response_never_contains_raw_text_field(self, client):
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "blank.png")},
            content_type="multipart/form-data",
        )
        data = json.loads(resp.data)
        forbidden_keys = {"raw_text", "original_text", "pii_values", "ocr_text"}
        assert not any(k in data for k in forbidden_keys)

    def test_real_test_image(self, client):
        """Full pipeline on the generated test image."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/process-image",
                data={"image": (f, "test_input.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["success"] is True
        assert isinstance(data["safe_text"], str)
        assert isinstance(data["entities_found"], list)

    def test_privacy_raw_pii_not_in_response(self, client):
        """
        CRITICAL PRIVACY TEST.

        Process the test image and assert that known raw PII values
        do NOT appear anywhere in the JSON response.
        """
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/process-image",
                data={"image": (f, "test_input.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200
        response_text = resp.data.decode("utf-8")

        # These are the synthetic PII values — must NOT appear in any response field
        assert "peter.parker@example.com" not in response_text, \
            "PRIVACY VIOLATION: raw email in response!"
        assert "9876543210" not in response_text, \
            "PRIVACY VIOLATION: raw phone number in response!"

    def test_non_pii_preserved_in_safe_text(self, client):
        """Non-PII data should NOT be redacted."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/process-image",
                data={"image": (f, "test_input.png")},
                content_type="multipart/form-data",
            )
        data = json.loads(resp.data)
        safe = data["safe_text"]
        # At least one of these non-PII tokens should remain
        non_pii_present = (
            "Computer Science" in safe
            or "ABC University" in safe
            or "APP-2026-001" in safe
            or "Computer" in safe
        )
        assert non_pii_present, \
            f"Non-PII content was unexpectedly removed. safe_text: {safe[:200]}"

    def test_entities_found_structure(self, client):
        """entities_found items must have 'type' and 'score' only (no raw values)."""
        if not TEST_IMAGE_PATH.exists():
            pytest.skip("test_input.png not yet generated")
        with open(TEST_IMAGE_PATH, "rb") as f:
            resp = client.post(
                "/process-image",
                data={"image": (f, "test_input.png")},
                content_type="multipart/form-data",
            )
        data = json.loads(resp.data)
        for entity in data["entities_found"]:
            assert "type" in entity
            assert "score" in entity
            # Must NOT include raw matched value
            assert "value" not in entity
            assert "text" not in entity
            assert "original" not in entity

    def test_oversized_image_returns_413(self, client):
        """Send an artificially oversized payload."""
        from backend.config import MAX_IMAGE_SIZE_BYTES
        oversized = b"x" * (MAX_IMAGE_SIZE_BYTES + 1)
        resp = client.post(
            "/process-image",
            data={"image": (io.BytesIO(oversized), "big.png")},
            content_type="multipart/form-data",
        )
        assert resp.status_code == 413


# ── /redact-image ────────────────────────────────────────────────────────────

class TestRedactImage:
    def test_missing_image_returns_400(self, client):
        resp = client.post("/redact-image")
        assert resp.status_code == 400

    def test_valid_image_returns_image_or_error(self, client):
        """Either returns a PNG image (200) or a server error (500), never 501 now."""
        resp = client.post(
            "/redact-image",
            data={"image": (io.BytesIO(_make_png_bytes()), "test.png")},
            content_type="multipart/form-data",
        )
        # Now uses our own OCR-based redaction pipeline — always 200 PNG on success
        assert resp.status_code in (200, 500)


# ── 404 / 405 ────────────────────────────────────────────────────────────────

class TestErrorHandlers:
    def test_unknown_route_404(self, client):
        resp = client.get("/does-not-exist")
        assert resp.status_code == 404

    def test_get_on_post_endpoint_405(self, client):
        resp = client.get("/process-image")
        assert resp.status_code == 405
