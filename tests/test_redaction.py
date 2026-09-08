"""
tests/test_redaction.py — Tests for the Gaussian blur and blackbox
visual redaction pipeline (backend/redaction.py + /redact-image endpoint).

Test matrix:
  - blur method returns valid PNG, same dimensions
  - blackbox method returns valid PNG, same dimensions (regression)
  - omitting method param defaults to blur
  - unknown method returns 400
  - pixels OUTSIDE PII regions are unchanged
  - pixels INSIDE PII regions ARE changed (blur was applied)
  - entity with no bounding box does not crash (graceful skip)
  - test_input.png redaction (email/person/phone)
"""

from __future__ import annotations

import io
import pathlib
import sys

import pytest
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from backend.app import app as flask_app

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TEST_DATA = pathlib.Path(__file__).parent.parent / "test_data"
LOGIN_PNG = TEST_DATA / "login_screenshot.png"
INPUT_PNG = TEST_DATA / "test_input.png"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_png_bytes(width: int = 400, height: int = 200) -> bytes:
    img = Image.new("RGB", (width, height), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _open_png_from_response(data: bytes) -> Image.Image:
    """Parse response bytes as a PIL PNG image."""
    return Image.open(io.BytesIO(data))


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["MAX_CONTENT_LENGTH"] = None
    with flask_app.test_client() as c:
        yield c


def _post_redact(client, image_bytes: bytes, filename: str, method: str | None = None) -> object:
    url = "/redact-image"
    if method is not None:
        url += f"?method={method}"
    return client.post(
        url,
        data={"image": (io.BytesIO(image_bytes), filename)},
        content_type="multipart/form-data",
    )


# ---------------------------------------------------------------------------
# 1. Basic endpoint tests
# ---------------------------------------------------------------------------

class TestRedactImageEndpoint:
    def test_missing_image_returns_400(self, client):
        resp = client.post("/redact-image")
        assert resp.status_code == 400

    def test_unknown_method_returns_400(self, client):
        resp = _post_redact(client, _make_png_bytes(), "img.png", method="watermark")
        assert resp.status_code == 400
        import json
        body = json.loads(resp.data)
        assert body["success"] is False
        assert "Unknown redaction method" in body["error"]

    def test_blur_returns_200_and_png(self, client):
        resp = _post_redact(client, _make_png_bytes(), "img.png", method="blur")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        img = _open_png_from_response(resp.data)
        assert img.format == "PNG"

    def test_blackbox_returns_200_and_png(self, client):
        resp = _post_redact(client, _make_png_bytes(), "img.png", method="blackbox")
        assert resp.status_code == 200
        assert resp.content_type == "image/png"
        img = _open_png_from_response(resp.data)
        assert img.format == "PNG"

    def test_default_method_is_blackbox(self, client):
        """Omitting ?method= should default to blackbox (200 + PNG returned)."""
        resp = _post_redact(client, _make_png_bytes(), "img.png", method=None)
        assert resp.status_code == 200
        assert "image/png" in resp.content_type


# ---------------------------------------------------------------------------
# 2. Dimension / format invariants
# ---------------------------------------------------------------------------

class TestDimensionInvariance:
    @pytest.mark.skipif(not LOGIN_PNG.exists(), reason="login_screenshot.png not present")
    def test_blur_preserves_dimensions_login(self, client):
        original = Image.open(str(LOGIN_PNG))
        orig_size = original.size

        with open(str(LOGIN_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "login_screenshot.png", method="blur")

        assert resp.status_code == 200
        result = _open_png_from_response(resp.data)
        assert result.size == orig_size, (
            f"Dimensions changed: {orig_size} → {result.size}"
        )

    @pytest.mark.skipif(not LOGIN_PNG.exists(), reason="login_screenshot.png not present")
    def test_blackbox_preserves_dimensions_login(self, client):
        original = Image.open(str(LOGIN_PNG))
        orig_size = original.size

        with open(str(LOGIN_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "login_screenshot.png", method="blackbox")

        assert resp.status_code == 200
        result = _open_png_from_response(resp.data)
        assert result.size == orig_size

    @pytest.mark.skipif(not INPUT_PNG.exists(), reason="test_input.png not present")
    def test_blur_preserves_dimensions_test_input(self, client):
        original = Image.open(str(INPUT_PNG))
        orig_size = original.size

        with open(str(INPUT_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "test_input.png", method="blur")

        assert resp.status_code == 200
        result = _open_png_from_response(resp.data)
        assert result.size == orig_size


# ---------------------------------------------------------------------------
# 3. Pixel-level correctness
# ---------------------------------------------------------------------------

class TestPixelCorrectness:
    @pytest.mark.skipif(not LOGIN_PNG.exists(), reason="login_screenshot.png not present")
    def test_pii_region_is_changed_by_blur(self, client):
        """
        The USERNAME region must differ between original and blurred output.

        Actual OCR-detected USERNAME bbox for login_screenshot.png (verified
        via diagnostic run):
          PixelBox(left=405, top=130, right=484, bottom=148)
        With REDACTION_PADDING_PX=5 the padded region is x=400-489, y=125-153.
        We sample conservatively inside the original bbox to confirm blur.
        """
        original = Image.open(str(LOGIN_PNG)).convert("RGB")

        with open(str(LOGIN_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "login_screenshot.png", method="blur")

        assert resp.status_code == 200
        redacted = _open_png_from_response(resp.data).convert("RGB")

        # Sample from inside the actual OCR-detected USERNAME bbox:
        # OCR gives x=405-484, y=130-148; with 5px padding -> x=400-489, y=125-153.
        # We scan conservatively inside the core bbox to confirm blur was applied.
        changed = False
        for x in range(408, 482, 4):
            for y in range(131, 147, 2):
                orig_px = original.getpixel((x, y))
                redc_px = redacted.getpixel((x, y))
                if orig_px != redc_px:
                    changed = True
                    break
            if changed:
                break

        assert changed, (
            "Expected blur to change pixels inside the PII region "
            "(USERNAME bbox x=408-482, y=131-147), but pixels appear identical "
            "to the original. Check that OCR correctly maps the username value."
        )

    @pytest.mark.skipif(not LOGIN_PNG.exists(), reason="login_screenshot.png not present")
    def test_non_pii_region_is_unchanged_by_blur(self, client):
        """
        The top navigation bar area (far above the login form) should be
        completely unchanged by blur redaction.
        """
        original = Image.open(str(LOGIN_PNG)).convert("RGB")

        with open(str(LOGIN_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "login_screenshot.png", method="blur")

        assert resp.status_code == 200
        redacted = _open_png_from_response(resp.data).convert("RGB")

        # The very top row of pixels (y=0..2) is outside any detected PII box
        # in the login screenshot — the detected entities are at y≥125 (Login)
        # and the username/password regions are at y≥170.
        all_same = True
        for x in range(0, min(original.width, 200), 4):
            for y in range(0, 3):
                if original.getpixel((x, y)) != redacted.getpixel((x, y)):
                    all_same = False
                    break
        assert all_same, (
            "Non-PII region (top of image) should be pixel-for-pixel identical "
            "after blur redaction, but pixels differ."
        )

    @pytest.mark.skipif(not LOGIN_PNG.exists(), reason="login_screenshot.png not present")
    def test_blackbox_pii_region_is_black(self, client):
        """
        Blackbox method should draw black pixels in the PII region.

        Actual OCR-detected USERNAME bbox for login_screenshot.png (verified
        via diagnostic run):
          PixelBox(left=405, top=130, right=484, bottom=148)
        With REDACTION_PADDING_PX=5 the padded region is x=400-489, y=125-153.
        """
        with open(str(LOGIN_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "login_screenshot.png", method="blackbox")

        assert resp.status_code == 200
        redacted = _open_png_from_response(resp.data).convert("RGB")

        # The username area (OCR bbox x=405-484, y=130-148; padded to x=400-489, y=125-153)
        # should contain black pixels if blackbox was applied.
        found_black = False
        for x in range(403, 487, 4):
            for y in range(127, 151, 3):
                px = redacted.getpixel((x, y))
                if px == (0, 0, 0):
                    found_black = True
                    break
            if found_black:
                break

        assert found_black, (
            "Expected black pixels in PII region after blackbox redaction. "
            "Checked USERNAME area (x=403-487, y=127-151) -- no black pixels found."
        )


# ---------------------------------------------------------------------------
# 4. test_input.png (standard synthetic PII image)
# ---------------------------------------------------------------------------

class TestStandardPIIImage:
    @pytest.mark.skipif(not INPUT_PNG.exists(), reason="test_input.png not present")
    def test_blur_on_test_input_returns_png(self, client):
        with open(str(INPUT_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "test_input.png", method="blur")
        assert resp.status_code == 200
        assert "image/png" in resp.content_type
        img = _open_png_from_response(resp.data)
        assert img.size[0] > 0 and img.size[1] > 0

    @pytest.mark.skipif(not INPUT_PNG.exists(), reason="test_input.png not present")
    def test_blackbox_on_test_input_returns_png(self, client):
        with open(str(INPUT_PNG), "rb") as f:
            resp = _post_redact(client, f.read(), "test_input.png", method="blackbox")
        assert resp.status_code == 200
        assert "image/png" in resp.content_type


# ---------------------------------------------------------------------------
# 5. Edge-case: entity without a bounding box
# ---------------------------------------------------------------------------

class TestNoBoundingBoxEdgeCase:
    def test_entity_without_bbox_does_not_crash(self, client):
        """
        If an entity cannot be mapped to a pixel bbox, the endpoint must NOT
        crash — it should skip that entity and return 200.
        This is tested by sending a blank-white image with no OCR text,
        which produces no bounding boxes in the OCR word list but may still
        detect zero entities (so no skip path), and crucially does not 500.
        """
        # Pure white image — Tesseract finds nothing, Presidio finds nothing,
        # no bounding box mapping needed — must return 200.
        resp = _post_redact(client, _make_png_bytes(), "blank.png", method="blur")
        assert resp.status_code == 200
        img = _open_png_from_response(resp.data)
        assert img.format == "PNG"

    def test_entity_without_bbox_graceful_skip(self):
        """
        Unit-level: _text_span_to_pixel_boxes returns [] for a span that
        maps to no words → no crash in apply_blur_redaction.
        """
        from backend.redaction import apply_blur_redaction, apply_blackbox_redaction
        from backend.ocr import OCRResult
        from backend.pii import DetectedEntity

        img = Image.new("RGB", (300, 100), "white")
        ocr = OCRResult(text="hello world", words=[])  # no word boxes
        entities = [DetectedEntity(entity_type="PERSON", score=0.85, start=0, end=5)]

        # Must not raise — should return the original image with skipped_count=1
        result_img, blurred, skipped = apply_blur_redaction(img, entities, ocr)
        assert skipped == 1
        assert blurred == 0
        assert result_img.size == img.size

        result_img2, drawn, skipped2 = apply_blackbox_redaction(img, entities, ocr)
        assert skipped2 == 1
        assert drawn == 0


# ---------------------------------------------------------------------------
# 6. Unit tests for the redaction module directly
# ---------------------------------------------------------------------------

class TestRedactionModule:
    def test_invalid_method_raises(self):
        from backend.redaction import redact_image
        from backend.ocr import OCRResult
        from backend.pii import DetectedEntity

        img = Image.new("RGB", (100, 50), "white")
        ocr = OCRResult(text="", words=[])
        with pytest.raises(ValueError, match="Unknown redaction method"):
            redact_image(img, [], ocr, method="invalid")

    def test_blur_output_is_rgb(self):
        from backend.redaction import apply_blur_redaction
        from backend.ocr import OCRResult

        img = Image.new("RGB", (200, 100), "white")
        ocr = OCRResult(text="", words=[])
        result, _, _ = apply_blur_redaction(img, [], ocr)
        assert result.mode == "RGB"

    def test_blackbox_output_is_rgb(self):
        from backend.redaction import apply_blackbox_redaction
        from backend.ocr import OCRResult

        img = Image.new("RGB", (200, 100), "white")
        ocr = OCRResult(text="", words=[])
        result, _, _ = apply_blackbox_redaction(img, [], ocr)
        assert result.mode == "RGB"

    def test_blur_actually_blurs_a_known_region(self):
        """
        Synthetic: place a coloured rectangle in an image, add a matching
        OCRWord, attach a DetectedEntity spanning it, apply blur — verify
        the pixel in the rectangle changed.
        """
        from backend.redaction import apply_blur_redaction
        from backend.ocr import OCRResult, OCRWord
        from backend.pii import DetectedEntity

        # Image: white with a red rectangle at x=10,y=10 → x=90,y=40
        img = Image.new("RGB", (200, 100), "white")
        from PIL import ImageDraw
        d = ImageDraw.Draw(img)
        d.rectangle([10, 10, 90, 40], fill=(200, 0, 0))

        word = OCRWord(text="REDWORD", left=10, top=10, width=80, height=30, line_num=1)
        ocr = OCRResult(text="REDWORD", words=[word])
        entity = DetectedEntity(entity_type="PERSON", score=0.9, start=0, end=7)

        result, blurred, skipped = apply_blur_redaction(img, [entity], ocr, blur_radius=10)
        assert blurred == 1
        assert skipped == 0

        # The pixel at centre of the rectangle must have changed
        orig_px = img.getpixel((50, 25))
        result_px = result.getpixel((50, 25))
        assert orig_px != result_px, (
            f"Blur should have changed the pixel at (50,25) but orig={orig_px} result={result_px}"
        )

    def test_bbox_mapping_finds_correct_words(self):
        """Unit test _text_span_to_pixel_boxes mapping logic."""
        from backend.redaction import _text_span_to_pixel_boxes
        from backend.ocr import OCRWord

        text = "Peter Parker works here"
        words = [
            OCRWord(text="Peter",  left=10, top=5,  width=40, height=15, line_num=1),
            OCRWord(text="Parker", left=60, top=5,  width=50, height=15, line_num=1),
            OCRWord(text="works",  left=120, top=5, width=40, height=15, line_num=1),
            OCRWord(text="here",   left=170, top=5, width=30, height=15, line_num=1),
        ]

        # "Peter Parker" → start=0, end=12
        boxes = _text_span_to_pixel_boxes(text, 0, 12, words)
        assert len(boxes) == 2
        assert boxes[0].left == 10   # Peter
        assert boxes[1].left == 60   # Parker

        # "works" → start=13, end=18
        boxes2 = _text_span_to_pixel_boxes(text, 13, 18, words)
        assert len(boxes2) == 1
        assert boxes2[0].left == 120

    def test_pad_box_clamps_to_image(self):
        from backend.redaction import _pad_box, PixelBox
        b = PixelBox(left=2, top=2, right=50, bottom=50)
        padded = _pad_box(b, padding=5, img_width=100, img_height=100)
        assert padded.left == 0   # clamped from -3
        assert padded.top == 0
        assert padded.right == 55
        assert padded.bottom == 55

    def test_rgba_image_handled(self):
        """RGBA input should not crash — output must be RGB."""
        from backend.redaction import redact_image
        from backend.ocr import OCRResult

        img = Image.new("RGBA", (100, 50), (255, 255, 255, 255))
        ocr = OCRResult(text="", words=[])
        result = redact_image(img, [], ocr, method="blur")
        assert result.mode == "RGB"


# ---------------------------------------------------------------------------
# 5. Offset-to-Pixel Alignment & Union Bounding Box Tests
# ---------------------------------------------------------------------------

class TestOffsetBoundingBoxAlignment:
    """
    Tests ensuring exact synchronization between OCR text offsets and pixel
    bounding boxes, preventing mislocated redactions.
    """

    def test_github_profile_nav_tabs_unblurred(self, client):
        """
        Verify that navigation tabs near the top of the GitHub profile screenshot
        remain unblurred while actual PII below is correctly redacted.
        """
        gh_path = TEST_DATA / "github_profile_screenshot.png"
        if not gh_path.exists():
            pytest.skip("github_profile_screenshot.png not found")

        orig_img = Image.open(str(gh_path)).convert("RGB")

        with open(str(gh_path), "rb") as f:
            resp = client.post(
                "/redact-image?method=blur",
                data={"image": (f, "github_profile_screenshot.png")},
                content_type="multipart/form-data",
            )

        assert resp.status_code == 200
        redacted_img = _open_png_from_response(resp.data)

        # Nav bar region for 'Repositories 12', 'Projects', 'Packages' (x: 100 to 330, y: 44 to 56)
        # Verify that pixels in the nav bar remain identical to original
        nav_orig = orig_img.crop((105, 44, 325, 56))
        nav_redacted = redacted_img.crop((105, 44, 325, 56))
        from PIL import ImageChops
        diff = ImageChops.difference(nav_orig, nav_redacted)
        assert diff.getbbox() is None, "Navigation tabs should remain sharp and unblurred!"

        # Lower PII region (education / name info around y: 430) must be blurred
        edu_orig = orig_img.crop((550, 430, 780, 442))
        edu_redacted = redacted_img.crop((550, 430, 780, 442))
        diff_edu = ImageChops.difference(edu_orig, edu_redacted)
        assert diff_edu.getbbox() is not None, "Education PII must be redacted!"

    def test_multi_word_entity_union_box(self):
        """
        Multi-word entity spanning multiple OCR words on a line gets a single
        union bounding box enclosing all of them.
        """
        from backend.redaction import _text_span_to_pixel_boxes
        from backend.ocr import OCRWord

        # Construct a line of words with known coordinates and offsets
        text = "Education: Vignana Jyothi Institute of Technology"
        words = [
            OCRWord(text="Education:", left=20, top=100, width=80, height=20, line_num=1, start=0, end=10),
            OCRWord(text="Vignana", left=110, top=100, width=60, height=20, line_num=1, start=11, end=18),
            OCRWord(text="Jyothi", left=175, top=100, width=50, height=20, line_num=1, start=19, end=25),
            OCRWord(text="Institute", left=230, top=100, width=70, height=20, line_num=1, start=26, end=35),
            OCRWord(text="of", left=305, top=100, width=20, height=20, line_num=1, start=36, end=38),
            OCRWord(text="Technology", left=330, top=100, width=80, height=20, line_num=1, start=39, end=49),
        ]

        # Entity spans "Vignana Jyothi Institute of Technology" (start=11, end=49)
        union_boxes = _text_span_to_pixel_boxes(text, 11, 49, words, union=True)
        assert len(union_boxes) == 1
        box = union_boxes[0]

        # Encloses all words from Vignana (left=110) to Technology (right=330+80=410)
        assert box.left == 110
        assert box.top == 100
        assert box.right == 410
        assert box.bottom == 120

    def test_multiple_entities_independent_boxes(self):
        """
        Multiple separate entities in different areas get distinct, independent
        bounding boxes without cross-contamination.
        """
        from backend.redaction import _collect_redaction_boxes
        from backend.ocr import OCRResult, OCRWord
        from backend.pii import DetectedEntity

        text = "Name: Sarah Connor\nEmail: sarah@skynet.org"
        words = [
            OCRWord(text="Name:", left=10, top=50, width=40, height=15, line_num=1, start=0, end=5),
            OCRWord(text="Sarah", left=60, top=50, width=40, height=15, line_num=1, start=6, end=11),
            OCRWord(text="Connor", left=105, top=50, width=50, height=15, line_num=1, start=12, end=18),
            OCRWord(text="Email:", left=10, top=90, width=40, height=15, line_num=2, start=19, end=25),
            OCRWord(text="sarah@skynet.org", left=60, top=90, width=120, height=15, line_num=2, start=26, end=42),
        ]
        ocr = OCRResult(text=text, words=words)

        ent1 = DetectedEntity(entity_type="PERSON", score=0.9, start=6, end=18)
        ent2 = DetectedEntity(entity_type="EMAIL_ADDRESS", score=0.95, start=26, end=42)

        boxes, skipped = _collect_redaction_boxes([ent1, ent2], ocr, img_w=500, img_h=300, padding=0)
        assert skipped == 0
        assert len(boxes) == 2

        # Box 1 is Sarah Connor at y=50
        assert boxes[0].left == 60
        assert boxes[0].right == 155
        assert boxes[0].top == 50
        assert boxes[0].bottom == 65

        # Box 2 is Email at y=90
        assert boxes[1].left == 60
        assert boxes[1].right == 180
        assert boxes[1].top == 90
        assert boxes[1].bottom == 105

    def test_sanity_check_assertion_detects_mismatch(self, caplog):
        """
        If offsets point to completely mismatched words, verify_entity_box_alignment
        returns False and logs a warning.
        """
        import logging
        from backend.redaction import verify_entity_box_alignment
        from backend.ocr import OCRWord

        words = [
            OCRWord(text="Overview", left=10, top=10, width=50, height=15, line_num=1),
            OCRWord(text="Repositories", left=65, top=10, width=80, height=15, line_num=1),
        ]

        with caplog.at_level(logging.WARNING):
            is_valid = verify_entity_box_alignment(
                expected_text="Acme Corporation Ltd",
                matched_words=words,
                entity_type="ORGANIZATION",
            )

        assert is_valid is False
        assert "Sanity check warning" in caplog.text
        assert "overview repositories" in caplog.text.lower()

    def test_regression_login_and_face_boxes(self, client):
        """
        Re-run login screenshot (USERNAME/PASSWORD) and single face through
        the fixed pipeline to ensure zero regression in existing functionality.
        """
        # 1. Login screenshot
        with open(str(LOGIN_PNG), "rb") as f:
            resp_login = client.post(
                "/redact-image?method=blur",
                data={"image": (f, "login_screenshot.png")},
                content_type="multipart/form-data",
            )
        assert resp_login.status_code == 200
        img_login = _open_png_from_response(resp_login.data)
        assert img_login.format == "PNG"

        # 2. Face test
        face_path = TEST_DATA / "face_single.png"
        if face_path.exists():
            with open(str(face_path), "rb") as f:
                resp_face = client.post(
                    "/redact-image?method=blur",
                    data={"image": (f, "face_single.png")},
                    content_type="multipart/form-data",
                )
            assert resp_face.status_code == 200
            assert resp_face.headers.get("X-Faces-Detected") == "1"
