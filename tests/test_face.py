"""
tests/test_face.py — Unit and integration tests for face detection and redaction.

Test Matrix:
  1. Single-face image → detect_faces() returns exactly 1 region with plausible coordinates.
  2. Multi-face image → detect_faces() returns 3 regions.
  3. No-face image (test_input.png) → detect_faces() returns empty list (no false positives).
  4. /redact-image?method=blur on single-face → face region pixels changed, outside pixels unchanged.
  5. /redact-image?method=blackbox on single-face → face region covered in solid black.
  6. Combined image with face + text PII → both face and text are redacted in one output.
  7. Zero disk persistence: no cropped face images or facial byte dumps on disk.
  8. X-Faces-Detected header returned on /redact-image response.
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.app import app as flask_app
from backend.face import FaceRegion, detect_faces

# ---------------------------------------------------------------------------
# Test Paths
# ---------------------------------------------------------------------------

TEST_DATA = pathlib.Path(__file__).parent.parent / "test_data"
FACE_SINGLE = TEST_DATA / "face_single.png"
FACE_MULTI = TEST_DATA / "face_multi.png"
FACE_WITH_PII = TEST_DATA / "face_with_pii.png"
TEST_INPUT = TEST_DATA / "test_input.png"


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["MAX_CONTENT_LENGTH"] = None
    with flask_app.test_client() as c:
        yield c


def _open_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


# ---------------------------------------------------------------------------
# 1. Direct Face Detection Unit Tests
# ---------------------------------------------------------------------------

class TestFaceDetectionModule:
    def test_single_face_detected(self):
        """Single-face image returns exactly 1 FaceRegion with plausible coordinates."""
        assert FACE_SINGLE.exists(), f"Missing test asset: {FACE_SINGLE}"
        img = Image.open(str(FACE_SINGLE))
        w, h = img.size

        faces = detect_faces(img)
        assert len(faces) == 1
        face = faces[0]

        assert isinstance(face, FaceRegion)
        assert face.confidence >= 0.5
        # Coordinates must be within image dimensions
        assert 0 <= face.left < w
        assert 0 <= face.top < h
        assert face.width > 0
        assert face.height > 0
        assert face.right <= w
        assert face.bottom <= h

    def test_multi_face_detected(self):
        """Multi-face image with 3 people returns exactly 3 FaceRegions."""
        assert FACE_MULTI.exists(), f"Missing test asset: {FACE_MULTI}"
        img = Image.open(str(FACE_MULTI))
        w, h = img.size

        faces = detect_faces(img)
        assert len(faces) == 3

        for face in faces:
            assert face.confidence >= 0.5
            assert 0 <= face.left < w
            assert 0 <= face.top < h
            assert face.right <= w
            assert face.bottom <= h

    def test_no_face_control(self):
        """Negative control: plain text document produces 0 face detections."""
        assert TEST_INPUT.exists(), f"Missing test asset: {TEST_INPUT}"
        img = Image.open(str(TEST_INPUT))

        faces = detect_faces(img)
        assert len(faces) == 0

    def test_empty_image_returns_empty_list(self):
        img = Image.new("RGB", (0, 0))
        faces = detect_faces(img)
        assert faces == []

    def test_face_region_properties(self):
        f = FaceRegion(left=10, top=20, width=30, height=40, confidence=0.95)
        assert f.right == 40
        assert f.bottom == 60


# ---------------------------------------------------------------------------
# 2. Endpoint /redact-image Integration Tests
# ---------------------------------------------------------------------------

class TestRedactImageFaces:
    def test_blur_face_on_single_face_image(self, client):
        """Face region is blurred (pixels modified) while outer pixels are untouched."""
        img = Image.open(str(FACE_SINGLE))
        w, h = img.size
        faces = detect_faces(img)
        assert len(faces) == 1
        f = faces[0]

        with open(str(FACE_SINGLE), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (fh, "face_single.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        assert resp.headers.get("X-Faces-Detected") == "1"

        out_img = _open_png(resp.data)
        assert out_img.size == (w, h)

        # Centre of face region MUST have changed
        cx = f.left + f.width // 2
        cy = f.top + f.height // 2
        orig_px = img.convert("RGB").getpixel((cx, cy))
        out_px = out_img.getpixel((cx, cy))
        assert orig_px != out_px, "Face center pixel should have been modified by blur."

        # Far corner (background) MUST remain identical
        orig_bg = img.convert("RGB").getpixel((10, 10))
        out_bg = out_img.getpixel((10, 10))
        assert orig_bg == out_bg, "Background pixels outside face must remain identical."

    def test_blackbox_face_on_single_face_image(self, client):
        """Face region is filled with solid black when method=blackbox is requested."""
        img = Image.open(str(FACE_SINGLE))
        faces = detect_faces(img)
        assert len(faces) == 1
        f = faces[0]

        with open(str(FACE_SINGLE), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blackbox",
                data={"image": (fh, "face_single.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        assert resp.headers.get("X-Faces-Detected") == "1"

        out_img = _open_png(resp.data)
        cx = f.left + f.width // 2
        cy = f.top + f.height // 2
        out_px = out_img.getpixel((cx, cy))
        assert out_px == (0, 0, 0), f"Expected black pixel (0,0,0) in blackbox redaction, got {out_px}"

    def test_multi_face_blur(self, client):
        """All faces in a multi-face image are blurred, header reports correct count."""
        img = Image.open(str(FACE_MULTI))
        faces = detect_faces(img)
        assert len(faces) == 3

        with open(str(FACE_MULTI), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (fh, "face_multi.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        assert resp.headers.get("X-Faces-Detected") == "3"

        out_img = _open_png(resp.data)
        for f in faces:
            cx = f.left + f.width // 2
            cy = f.top + f.height // 2
            orig_px = img.convert("RGB").getpixel((cx, cy))
            out_px = out_img.getpixel((cx, cy))
            assert orig_px != out_px, f"Face at ({cx}, {cy}) should be blurred."

    def test_control_image_zero_faces_header(self, client):
        """Images without faces report X-Faces-Detected: 0."""
        with open(str(TEST_INPUT), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (fh, "test_input.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        assert resp.headers.get("X-Faces-Detected") == "0"

    def test_combined_face_and_text_pii(self, client):
        """Image with both a human face and text PII has both redacted."""
        img = Image.open(str(FACE_WITH_PII))
        faces = detect_faces(img)
        assert len(faces) == 1
        face = faces[0]

        with open(str(FACE_WITH_PII), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (fh, "face_with_pii.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        assert resp.headers.get("X-Faces-Detected") == "1"

        out_img = _open_png(resp.data)

        # 1. Face must be blurred
        cx = face.left + face.width // 2
        cy = face.top + face.height // 2
        orig_face_px = img.convert("RGB").getpixel((cx, cy))
        out_face_px = out_img.getpixel((cx, cy))
        assert orig_face_px != out_face_px, "Face in combined image should be blurred."

        # 2. Text PII area (in the badge card) must also have changed pixels
        # Text line 'sarah.jenkins@company.com' is in the card area
        w, h = img.size
        # Sample inside the card text area
        card_text_y = h - 115
        card_text_x = 250
        orig_card_px = img.convert("RGB").getpixel((card_text_x, card_text_y))
        out_card_px = out_img.getpixel((card_text_x, card_text_y))
        # Verify that the card text area was modified by PII blur
        assert orig_card_px != out_card_px or any(
            img.convert("RGB").getpixel((x, card_text_y)) != out_img.getpixel((x, card_text_y))
            for x in range(100, 400, 10)
        ), "Text PII in combined image badge should be blurred."

    def test_process_image_endpoint_untouched(self, client):
        """Confirm /process-image is unaffected and does not return face data."""
        with open(str(FACE_SINGLE), "rb") as fh:
            resp = client.post(
                "/process-image",
                data={"image": (fh, "face_single.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        import json
        body = json.loads(resp.data)
        assert "safe_text" in body
        assert "entities_found" in body
        # No face entity or face fields injected into /process-image
        for ent in body["entities_found"]:
            assert ent["type"] != "FACE"
            assert ent["type"] != "HUMAN_FACE"
        assert "X-Faces-Detected" not in resp.headers


# ---------------------------------------------------------------------------
# 3. Privacy & Zero-Persistence Guarantee
# ---------------------------------------------------------------------------

class TestFacePrivacyGuarantees:
    def test_no_cropped_faces_persisted_to_disk(self, tmp_path, client):
        """Verify no face crop image files are created on disk during face redaction."""
        # Record file list before
        workspace = pathlib.Path(__file__).parent.parent
        files_before = set(workspace.rglob("*.png")) | set(workspace.rglob("*.jpg"))

        with open(str(FACE_SINGLE), "rb") as fh:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (fh, "face_single.png")},
                content_type="multipart/form-data",
            )
        assert resp.status_code == 200

        files_after = set(workspace.rglob("*.png")) | set(workspace.rglob("*.jpg"))
        new_files = files_after - files_before
        # Absolutely no temporary cropped face image files should be written to disk
        assert len(new_files) == 0, f"Unexpected image files created on disk: {new_files}"
