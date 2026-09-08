"""
Flask application — privacy-safe OCR + PII anonymisation service.

Endpoints:
    GET  /health         → liveness probe
    POST /process-image  → main pipeline: image → sanitised JSON
    POST /redact-image   → visual redaction pipeline (returns image)

Privacy guarantees enforced here:
    * Raw OCR text is never returned in any response.
    * PII values are never returned in any response.
    * Uploaded images are processed in memory and never written to disk.
"""

from __future__ import annotations

import os
import sys

# Ensure the project root is on sys.path so `import backend.*` works whether
# this file is run as `python backend/app.py` (from project root) or as a
# module.  We insert the directory two levels up from this file.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import io
import logging
import traceback

from flask import Flask, jsonify, request, Response
from flask_cors import CORS

import backend.config as config
from backend.processor import ImageValidationError, process_image_bytes
from backend.report import write_report

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.DEBUG if config.DEBUG else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Silence noisy third-party loggers to avoid leaking internal details
for noisy in ("presidio-analyzer", "presidio_analyzer", "spacy"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__)

# Limit upload size at the Flask/Werkzeug level as a secondary guard
app.config["MAX_CONTENT_LENGTH"] = config.MAX_IMAGE_SIZE_BYTES

# ── CORS ──────────────────────────────────────────────────────────────────────
# Allow only specific origins instead of "*".
# chrome-extension:// is whitelisted so the browser extension can call us.
CORS(
    app,
    origins=config.ALLOWED_ORIGINS,
    methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(413)
def request_entity_too_large(error: Exception) -> tuple[Response, int]:
    max_mb = config.MAX_IMAGE_SIZE_BYTES / (1024 * 1024)
    return jsonify({"success": False, "error": f"Image exceeds maximum size of {max_mb:.0f} MB."}), 413


@app.errorhandler(404)
def not_found(error: Exception) -> tuple[Response, int]:
    return jsonify({"success": False, "error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(error: Exception) -> tuple[Response, int]:
    return jsonify({"success": False, "error": "Method not allowed."}), 405


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health() -> tuple[Response, int]:
    """
    Liveness probe.

    Returns:
        200 OK with ``{"status": "ok"}``

    Note: does NOT expose internal configuration or version details.
    """
    return jsonify({"status": "ok"}), 200


@app.route("/process-image", methods=["POST"])
def process_image() -> tuple[Response, int]:
    """
    Main endpoint: image → sanitised text + entity metadata.

    Accepts:
        multipart/form-data with field ``image``.

    Returns:
        200: ``{"success": true, "safe_text": "...", "entities_found": [...]}``
        400: validation errors (missing/corrupted/empty image)
        413: image too large
        415: unsupported file type
        500: unexpected internal error
    """
    # ── Input validation ────────────────────────────────────────────────────
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request."}), 400

    file = request.files["image"]

    if file.filename == "" or file.filename is None:
        return jsonify({"success": False, "error": "No file selected."}), 400

    image_bytes: bytes = file.read()
    if not image_bytes:
        return jsonify({"success": False, "error": "Uploaded image is empty."}), 400

    # ── Pipeline ─────────────────────────────────────────────────────────────
    debug_report_env = os.getenv("DEBUG_PII_REPORT", "false").lower() in ("true", "1", "t", "yes")
    debug_report_hdr = request.headers.get("X-Debug-PII-Report", "").lower() in ("true", "1", "t", "yes")
    debug_pii_report = debug_report_env or debug_report_hdr or config.DEBUG_PII_REPORT

    try:
        result = process_image_bytes(
            data=image_bytes,
            filename=file.filename,
            preprocess=True,
            debug_pii_report=debug_pii_report,
        )
    except ImageValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), exc.http_status
    except RuntimeError as exc:
        # Surface only a safe generic message — no raw text
        logger.error("RuntimeError during processing (details suppressed).")
        return jsonify({"success": False, "error": "Internal processing error."}), 500
    except Exception:  # noqa: BLE001
        logger.error("Unhandled exception during /process-image (details suppressed).")
        return jsonify({"success": False, "error": "Unexpected server error."}), 500

    # ── Write privacy report ──────────────────────────────────────────
    report_file: str | None = None
    try:
        report_file = write_report(
            image_filename=file.filename,
            result=result,
            debug_pii_report=debug_pii_report,
        )
    except Exception:  # noqa: BLE001
        # Report failure is non-fatal — pipeline result is still returned.
        logger.error("Failed to write privacy report (details suppressed).")

    # ── Response — NEVER include raw OCR or PII values ──────────────────────
    entities_payload = [
        {
            "type": e.entity_type,
            "score": e.score,
        }
        for e in result.entities
    ]

    response_body: dict = {
        "success": True,
        "safe_text": result.safe_text,
        "entities_found": entities_payload,
    }
    if report_file is not None:
        response_body["report_file"] = report_file

    return jsonify(response_body), 200



@app.route("/redact-image", methods=["POST"])
def redact_image() -> tuple[Response, int]:
    """
    Visual redaction endpoint: image → redacted image (PNG).

    Returns the original image with PII regions visually redacted.

    Accepts:
        multipart/form-data with field ``image``.

    Query Parameters:
        method: ``"blackbox"`` (default) or ``"blur"``.
            - ``blackbox`` — Solid black rectangle drawn over PII regions (default).
            - ``blur``     — Gaussian blur applied over PII regions; rest of
                             image is pixel-for-pixel unchanged.

    Returns:
        200: PNG image with redacted regions (Content-Type: image/png)
        Various 4xx/5xx on failure.
    """
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No 'image' field in request."}), 400

    file = request.files["image"]
    if file.filename == "" or file.filename is None:
        return jsonify({"success": False, "error": "No file selected."}), 400

    image_bytes: bytes = file.read()
    if not image_bytes:
        return jsonify({"success": False, "error": "Uploaded image is empty."}), 400

    # ── Resolve redaction method from query param or form field ───────────────
    from backend.redaction import VALID_METHODS, DEFAULT_METHOD, redact_image as _redact_image

    method = (
        request.args.get("method")
        or request.form.get("method")
        or DEFAULT_METHOD
    ).strip().lower()

    if method not in VALID_METHODS:
        return jsonify({
            "success": False,
            "error": f"Unknown redaction method {method!r}. Valid options: {sorted(VALID_METHODS)}",
        }), 400

    try:
        from backend.processor import _validate_image_bytes, _decode_image
        from backend.ocr import extract_ocr_result
        from backend.pii import process_text

        _validate_image_bytes(image_bytes, file.filename)
        pil_image = _decode_image(image_bytes)

        # Run OCR — get full text AND word-level bounding boxes
        ocr_result = extract_ocr_result(pil_image, preprocess=True)

        # Run PII detection on the OCR text
        pii_result = process_text(text=ocr_result.text, ocr_words=ocr_result.words)

        # Apply visual redaction using the requested method
        redacted = _redact_image(
            image=pil_image,
            entities=pii_result.entities,
            ocr_result=ocr_result,
            method=method,
        )

        # Serialise to PNG in memory — same dimensions, same format
        buf = io.BytesIO()
        redacted.save(buf, format="PNG")
        buf.seek(0)

        logger.info(
            "Image redaction completed. method=%s entities=%d.",
            method,
            len(pii_result.entities),
        )
        return Response(buf.read(), status=200, mimetype="image/png")

    except ImageValidationError as exc:
        return jsonify({"success": False, "error": str(exc)}), exc.http_status
    except Exception:  # noqa: BLE001
        logger.error("Unhandled exception during /redact-image (details suppressed).")
        return jsonify({"success": False, "error": "Image redaction failed."}), 500



# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logger.info("Starting Privacy-Safe OCR Service on %s:%d", config.HOST, config.PORT)
    app.run(
        host=config.HOST,
        port=config.PORT,
        debug=config.DEBUG,
        # Disable the Werkzeug reloader in production mode so the module-level
        # Presidio engines are not initialised twice.
        use_reloader=config.DEBUG,
    )
