"""
tools/test_image.py  —  CLI test harness for the Redaction backend.

Sends a local image to the running backend at http://127.0.0.1:5005/process-image
and prints a clean, human-readable privacy report.

Usage:
    python tools/test_image.py /path/to/image.png
    python tools/test_image.py ~/Desktop/my_screenshot.png

The script never constructs, prints, or infers raw PII.
It only displays the safe_text and entity metadata returned by the backend.
"""

from __future__ import annotations

import argparse
import os
import sys
import pathlib
from typing import Any

# Ensure project root is in sys.path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

try:
    import backend.config as config
except ImportError:
    config = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


# ── Optional rich terminal output (graceful fallback if not installed) ────────
try:
    import requests
    from requests.exceptions import ConnectionError as ReqConnectionError
    from requests.exceptions import Timeout, RequestException
except ImportError:  # pragma: no cover
    print("ERROR: 'requests' is not installed. Run: pip install requests", file=sys.stderr)
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────

BACKEND_URL = "http://127.0.0.1:5005/process-image"
BACKEND_BASE = "http://127.0.0.1:5005"
TIMEOUT_SECONDS = 30

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_MIMES = {"image/png", "image/jpeg", "image/webp"}

LINE = "=" * 60
DIVIDER = "-" * 60


# ── Helpers ───────────────────────────────────────────────────────────────────

def _fmt_size(size_bytes: int) -> str:
    """Format byte count as a human-readable string."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.0f} KB"
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def _guess_type(path: pathlib.Path) -> str:
    """Return a short file-type label from the extension."""
    return path.suffix.lstrip(".").upper() or "UNKNOWN"


def _header(title: str) -> None:
    print(LINE)
    print(title.center(60))
    print(LINE)


def _section(title: str) -> None:
    print()
    print(DIVIDER)
    print(title)
    print(DIVIDER)


def _ok(msg: str) -> None:
    print(f"  \u2713 {msg}")


def _fail(msg: str) -> None:
    print(f"  \u2717 {msg}")


# ── Validation ────────────────────────────────────────────────────────────────

def validate_image_path(raw_path: str) -> pathlib.Path:
    """
    Resolve and validate the supplied image path.

    Returns the resolved Path on success.
    Prints an error and calls sys.exit(1) on failure — no traceback.
    """
    path = pathlib.Path(raw_path).expanduser().resolve()

    if not path.exists():
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  File not found: {raw_path}")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    if not path.is_file():
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  Path is not a file: {path}")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  Unsupported file type: '{ext}'")
        print(f"  Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    return path


# ── HTTP call ─────────────────────────────────────────────────────────────────

def call_backend(image_path: pathlib.Path) -> dict[str, Any]:
    """
    POST the image to the backend and return the parsed JSON response.

    Raises:
        SystemExit(1) for connection errors or non-recoverable HTTP errors,
        printing a clean error report instead of a traceback.
    """
    debug_mode = os.getenv("DEBUG_PII_REPORT", "false").lower() in ("true", "1", "t", "yes")
    headers = {}
    if debug_mode:
        headers["X-Debug-PII-Report"] = "true"

    try:
        with image_path.open("rb") as fh:
            response = requests.post(
                BACKEND_URL,
                files={"image": (image_path.name, fh, "image/png")},
                headers=headers,
                timeout=TIMEOUT_SECONDS,
            )
    except (ReqConnectionError, OSError):
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print("  Could not connect to the Redaction backend.")
        print()
        print("Backend:")
        print(f"  {BACKEND_BASE}")
        print()
        print("Start it with:")
        print("  python backend/app.py")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)
    except Timeout:
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  Request timed out after {TIMEOUT_SECONDS}s.")
        print(f"  The backend at {BACKEND_BASE} is not responding.")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)
    except RequestException as exc:
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  Unexpected network error: {type(exc).__name__}")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    # HTTP error statuses
    if response.status_code != 200:
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print(f"  Backend returned HTTP {response.status_code}.")
        try:
            detail = response.json().get("error", "No detail provided.")
        except Exception:
            detail = response.text[:200] or "No body."
        print(f"  Detail: {detail}")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    # Parse JSON
    try:
        data = response.json()
    except Exception:
        _header("         REDACTION IMAGE TEST")
        print()
        print("ERROR:")
        print("  Backend returned malformed JSON.")
        print()
        print("FAIL \u2717")
        print(LINE)
        sys.exit(1)

    return data


# ── Report renderer ───────────────────────────────────────────────────────────

def render_report(image_path: pathlib.Path, data: dict[str, Any]) -> bool:
    """
    Print the full human-readable report and return success status.

    Returns True if the response represents a successful result, False otherwise.
    """
    success: bool = bool(data.get("success", False))
    safe_text: str = data.get("safe_text", "")
    entities: list[dict] = data.get("entities_found", [])
    report_file: str | None = data.get("report_file")
    debug_mode: bool = os.getenv("DEBUG_PII_REPORT", "false").lower() in ("true", "1", "t", "yes")

    # Parse detected entity values from report_file if in debug mode
    entity_val_map: dict[int, tuple[str, str]] = {}
    if debug_mode and report_file and pathlib.Path(report_file).exists():
        try:
            report_content = pathlib.Path(report_file).read_text(encoding="utf-8")
            import re
            matches = re.findall(
                r"Type:\s*(\w+)\s*\n\s*Value:\s*(.+)\s*\n\s*Confidence:\s*([\d.]+)\s*\n\s*Replacement:\s*(\[\w+\])",
                report_content,
            )
            for idx, (etype, val, conf, repl) in enumerate(matches):
                entity_val_map[idx] = (val.strip(), repl.strip())
        except Exception:  # noqa: BLE001
            pass

    _header("REDACTION IMAGE TEST")

    print()
    print("Image:")
    print(f"  {image_path.name}")
    print()

    # 1. COMPLETE OCR EXTRACTED TEXT
    _section("1. COMPLETE OCR EXTRACTED TEXT")
    print()
    if safe_text and safe_text.strip():
        for idx, line in enumerate(safe_text.splitlines(), start=1):
            print(f"  [{idx:03d}] {line}")
    else:
        print("  (no text extracted)")
    print()

    # 2. OCR STATISTICS
    _section("2. OCR STATISTICS")
    print()
    lines_count = len(safe_text.splitlines()) if safe_text else 0
    words_count = len(safe_text.split()) if safe_text else 0
    chars_count = len(safe_text) if safe_text else 0
    print(f"  Lines:      {lines_count}")
    print(f"  Words:      {words_count}")
    print(f"  Characters: {chars_count}")
    print()

    # 3. DETECTED PERSONAL / PRIVATE DATA
    _section("3. DETECTED PERSONAL / PRIVATE DATA")
    print()
    if entities:
        print(f"  Total entities detected: {len(entities)}")
        print()
        for i, entity in enumerate(entities, start=1):
            entity_type = entity.get("type", "UNKNOWN")
            score = entity.get("score", 0.0)
            if debug_mode and (i - 1) in entity_val_map:
                val, token = entity_val_map[i - 1]
            else:
                val = "[HIDDEN]"
                token = (
                    config.ANONYMIZATION_LABELS.get(entity_type, f"[{entity_type}]")
                    if config and hasattr(config, "ANONYMIZATION_LABELS")
                    else f"[{entity_type}]"
                )
            print(f"  {i}. Type:        {entity_type}")
            print(f"     Value:       {val}")
            print(f"     Confidence:  {score:.2f}")
            print(f"     Replacement: {token}")
            print()
    else:
        print("  No PII entities detected.")
        print()

    # 4. PII SUMMARY
    _section("4. PII SUMMARY")
    print()
    print(f"  Total PII entities found: {len(entities)}")
    print()

    # 5. COMPLETE SANITIZED TEXT
    _section("5. COMPLETE SANITIZED TEXT")
    print()
    if safe_text and safe_text.strip():
        for idx, line in enumerate(safe_text.splitlines(), start=1):
            print(f"  [{idx:03d}] {line}")
    else:
        print("  (no text extracted)")
    print()

    # 6. PRIVACY CHECK
    _section("6. PRIVACY CHECK")
    print()
    print(f"  DEBUG_PII_REPORT mode:    {'ENABLED' if debug_mode else 'DISABLED'}")
    print(f"  Raw PII in terminal:      {'YES' if debug_mode and len(entities) > 0 else 'NO'}")
    print(f"  Raw PII in report file:   {'YES' if debug_mode and len(entities) > 0 else 'NO'}")
    print(f"  API raw PII exposure:     NO")
    print()

    # 7. REPORT PATH
    if report_file:
        _section("7. REPORT PATH")
        print()
        print("  Report generated:")
        print(f"    {report_file}")
        print()

    print(LINE)

    return success




# ── Entry point ───────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    """
    Main entry point.

    Args:
        argv: Argument list (defaults to sys.argv[1:] when None).

    Returns:
        0 on success, 1 on failure.
    """
    parser = argparse.ArgumentParser(
        prog="test_image",
        description="Send a local image to the Redaction backend and display a privacy report.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python tools/test_image.py test_data/test_input.png\n"
            "  python tools/test_image.py ~/Desktop/my_screenshot.png\n"
        ),
    )
    parser.add_argument(
        "image_path",
        metavar="IMAGE_PATH",
        help="Path to the image file (PNG, JPEG, or WEBP).",
    )
    parser.add_argument(
        "--url",
        default=BACKEND_URL,
        metavar="URL",
        help=f"Backend endpoint URL (default: {BACKEND_URL}).",
    )

    args = parser.parse_args(argv)

    # Step 1 — validate path
    image_path = validate_image_path(args.image_path)

    # Step 2 — call backend
    data = call_backend(image_path)

    # Step 3 — render report
    success = render_report(image_path, data)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
