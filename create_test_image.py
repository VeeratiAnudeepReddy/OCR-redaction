"""
Script to generate test_data/test_input.png using synthetic PII data only.

Run once:  python create_test_image.py
"""

from __future__ import annotations

import pathlib
import textwrap

from PIL import Image, ImageDraw, ImageFont


def create_test_image(output_path: str | pathlib.Path) -> None:
    """
    Create a clean, high-resolution test image with synthetic PII content.

    The image is intentionally high-contrast (dark text on white) with a large
    font to give Tesseract the best chance of accurate recognition.
    """
    content = textwrap.dedent("""\
        PRIVACY TEST APPLICATION

        Name: Peter Parker
        Email: peter.parker@example.com
        Phone: +91 9876543210

        Course: Computer Science
        College: ABC University

        Application ID: APP-2026-001
        Date: 2026-09-06

        Username: testuser123
        Handle: @sample_handle
    """)

    # Image dimensions chosen for clarity
    width, height = 900, 500
    margin = 40
    line_height = 38
    font_size = 28

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Try to load a system monospace font; fall back to default bitmap font
    font = None
    candidate_fonts = [
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ]
    for path in candidate_fonts:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except (OSError, IOError):
            continue

    if font is None:
        # PIL default bitmap font — not as nice but always available
        font = ImageFont.load_default()

    y = margin
    for line in content.splitlines():
        draw.text((margin, y), line, fill=(0, 0, 0), font=font)
        y += line_height

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path), "PNG", dpi=(200, 200))
    print(f"Test image saved to: {output_path}")


if __name__ == "__main__":
    create_test_image("test_data/test_input.png")
