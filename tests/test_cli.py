"""
Tests for tools/test_image.py

Uses pytest-mock to mock HTTP calls — the real backend does NOT need to be running.
"""

from __future__ import annotations

import io
import json
import pathlib
import sys

import pytest
from PIL import Image
from unittest.mock import MagicMock, patch

# Ensure project root on path
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from tools.test_image import (
    _fmt_size,
    _guess_type,
    call_backend,
    main,
    render_report,
    validate_image_path,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_png(tmp_path: pathlib.Path) -> pathlib.Path:
    """A real, valid 200×100 PNG file in a temp directory."""
    img = Image.new("RGB", (200, 100), "white")
    p = tmp_path / "test.png"
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    p.write_bytes(buf.getvalue())
    return p


@pytest.fixture
def tmp_jpeg(tmp_path: pathlib.Path) -> pathlib.Path:
    img = Image.new("RGB", (200, 100), "white")
    p = tmp_path / "photo.jpg"
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    p.write_bytes(buf.getvalue())
    return p


def _make_response(
    status_code: int = 200,
    body: dict | None = None,
    raise_exc: Exception | None = None,
) -> MagicMock:
    """Build a mock requests.Response."""
    mock = MagicMock()
    mock.status_code = status_code
    if body is not None:
        mock.json.return_value = body
        mock.text = json.dumps(body)
    else:
        mock.json.side_effect = ValueError("no JSON")
        mock.text = "bad"
    return mock


# ── Helper tests ──────────────────────────────────────────────────────────────

class TestHelpers:
    def test_fmt_size_bytes(self):
        assert _fmt_size(512) == "512 B"

    def test_fmt_size_kb(self):
        assert _fmt_size(2048) == "2 KB"

    def test_fmt_size_mb(self):
        assert _fmt_size(1024 * 1024 * 5) == "5.0 MB"

    def test_guess_type_png(self, tmp_png):
        assert _guess_type(tmp_png) == "PNG"

    def test_guess_type_jpg(self, tmp_jpeg):
        assert _guess_type(tmp_jpeg) == "JPG"

    def test_guess_type_unknown(self, tmp_path):
        p = tmp_path / "file"
        p.write_text("")
        assert _guess_type(p) == "UNKNOWN"


# ── validate_image_path ───────────────────────────────────────────────────────

class TestValidateImagePath:
    def test_valid_png(self, tmp_png, capsys):
        result = validate_image_path(str(tmp_png))
        assert result == tmp_png

    def test_valid_jpeg(self, tmp_jpeg, capsys):
        result = validate_image_path(str(tmp_jpeg))
        assert result == tmp_jpeg

    def test_nonexistent_file_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            validate_image_path("/nonexistent/path/image.png")
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "FAIL" in out

    def test_unsupported_extension_exits(self, tmp_path, capsys):
        p = tmp_path / "image.bmp"
        p.write_bytes(b"fake content")
        with pytest.raises(SystemExit) as exc_info:
            validate_image_path(str(p))
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "Unsupported" in out or "FAIL" in out

    def test_directory_path_exits(self, tmp_path, capsys):
        with pytest.raises(SystemExit) as exc_info:
            validate_image_path(str(tmp_path))
        assert exc_info.value.code == 1

    def test_tilde_expansion(self, tmp_path, monkeypatch, capsys):
        """~ in path should be expanded (not treated as literal character)."""
        img = Image.new("RGB", (10, 10), "white")
        p = tmp_path / "home.png"
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        p.write_bytes(buf.getvalue())
        # monkeypatch HOME and USERPROFILE so ~ resolves to tmp_path cross-platform
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        result = validate_image_path("~/home.png")
        assert result.name == "home.png"


# ── call_backend ──────────────────────────────────────────────────────────────

class TestCallBackend:
    def test_successful_response(self, tmp_png):
        """Valid 200 response with correct JSON is returned as dict."""
        payload = {
            "success": True,
            "safe_text": "Name: [PERSON]",
            "entities_found": [{"type": "PERSON", "score": 0.85}],
        }
        mock_resp = _make_response(200, payload)
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            result = call_backend(tmp_png)
        assert result == payload

    def test_connection_refused_exits(self, tmp_png, capsys):
        """ConnectionError → clean error message + sys.exit(1)."""
        from requests.exceptions import ConnectionError as ReqConnErr
        with patch("tools.test_image.requests.post", side_effect=ReqConnErr()):
            with pytest.raises(SystemExit) as exc_info:
                call_backend(tmp_png)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "connect" in out.lower() or "backend" in out.lower()

    def test_timeout_exits(self, tmp_png, capsys):
        from requests.exceptions import Timeout
        with patch("tools.test_image.requests.post", side_effect=Timeout()):
            with pytest.raises(SystemExit) as exc_info:
                call_backend(tmp_png)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "timed out" in out.lower() or "FAIL" in out

    def test_http_400_exits(self, tmp_png, capsys):
        mock_resp = _make_response(400, {"error": "Bad image"})
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            with pytest.raises(SystemExit) as exc_info:
                call_backend(tmp_png)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "400" in out or "FAIL" in out

    def test_http_500_exits(self, tmp_png, capsys):
        mock_resp = _make_response(500, {"error": "Internal error"})
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            with pytest.raises(SystemExit) as exc_info:
                call_backend(tmp_png)
        assert exc_info.value.code == 1

    def test_malformed_json_exits(self, tmp_png, capsys):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.side_effect = ValueError("bad json")
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            with pytest.raises(SystemExit) as exc_info:
                call_backend(tmp_png)
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "malformed" in out.lower() or "JSON" in out or "FAIL" in out


# ── render_report ─────────────────────────────────────────────────────────────

class TestRenderReport:
    def test_success_true_prints_pass(self, tmp_png, capsys):
        data = {
            "success": True,
            "safe_text": "Name: [PERSON]",
            "entities_found": [{"type": "PERSON", "score": 0.85}],
        }
        result = render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert result is True
        assert "1. COMPLETE OCR EXTRACTED TEXT" in out

    def test_success_false_prints_fail(self, tmp_png, capsys):
        data = {"success": False, "error": "something went wrong"}
        result = render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert result is False

    def test_no_pii_entities_message(self, tmp_png, capsys):
        data = {
            "success": True,
            "safe_text": "No sensitive content here.",
            "entities_found": [],
        }
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert "No PII entities detected." in out

    def test_multiple_entities_count(self, tmp_png, capsys):
        data = {
            "success": True,
            "safe_text": "Name: [PERSON]\nEmail: [EMAIL_ADDRESS]\nPhone: [PHONE_NUMBER]",
            "entities_found": [
                {"type": "PERSON",        "score": 0.85},
                {"type": "EMAIL_ADDRESS", "score": 1.00},
                {"type": "PHONE_NUMBER",  "score": 0.75},
            ],
        }
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert "Total entities detected: 3" in out

    def test_raw_pii_not_injected(self, tmp_png, capsys):
        """The report must NOT add PII fields that the backend doesn't return."""
        data = {
            "success": True,
            "safe_text": "Email: [EMAIL_ADDRESS]",
            "entities_found": [{"type": "EMAIL_ADDRESS", "score": 1.0}],
        }
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        # Confirm raw email never shows up (the backend already redacted it)
        assert "example.com" not in out
        assert "@" not in out or "[EMAIL_ADDRESS]" in out

    def test_file_metadata_displayed(self, tmp_png, capsys):
        data = {"success": True, "safe_text": "", "entities_found": []}
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert "test.png" in out

    def test_missing_safe_text_key(self, tmp_png, capsys):
        """If backend omits safe_text the report should still print cleanly."""
        data = {"success": True, "entities_found": []}
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert "1. COMPLETE OCR EXTRACTED TEXT" in out

    def test_missing_entities_key(self, tmp_png, capsys):
        """If entities_found is absent the report should still print cleanly."""
        data = {"success": True, "safe_text": "Some text."}
        render_report(tmp_png, data)
        out = capsys.readouterr().out
        assert "No PII entities detected." in out




# ── main() integration ────────────────────────────────────────────────────────

class TestMain:
    def test_main_success(self, tmp_png, capsys):
        payload = {
            "success": True,
            "safe_text": "Name: [PERSON]",
            "entities_found": [{"type": "PERSON", "score": 0.85}],
        }
        mock_resp = _make_response(200, payload)
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            exit_code = main([str(tmp_png)])
        assert exit_code == 0
        out = capsys.readouterr().out
        assert "1. COMPLETE OCR EXTRACTED TEXT" in out

    def test_main_invalid_path_exits(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main(["/absolutely/nonexistent/file.png"])
        assert exc_info.value.code == 1

    def test_main_backend_down_exits(self, tmp_png, capsys):
        from requests.exceptions import ConnectionError as ReqConnErr
        with patch("tools.test_image.requests.post", side_effect=ReqConnErr()):
            with pytest.raises(SystemExit) as exc_info:
                main([str(tmp_png)])
        assert exc_info.value.code == 1
        out = capsys.readouterr().out
        assert "backend" in out.lower() or "connect" in out.lower()

    def test_main_returns_1_on_backend_failure(self, tmp_png, capsys):
        payload = {"success": False, "error": "Bad image format"}
        mock_resp = _make_response(200, payload)
        with patch("tools.test_image.requests.post", return_value=mock_resp):
            exit_code = main([str(tmp_png)])
        assert exit_code == 1
