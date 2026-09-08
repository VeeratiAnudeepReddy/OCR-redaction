# Test Data Directory

This directory contains test images for visual redaction, OCR, PII detection, and face detection tests.

## Test Image Inventory & Licensing

1. **`test_input.png`**:
   - **Description**: Synthetic document image with fictional PII created by `create_test_image.py`.
   - **Content**: Plain text with synthetic name, email, phone, usernames, handles. Contains 0 faces (used as negative control for face detection).
   - **License**: Dedicated to the public domain / Project-generated synthetic data.

2. **`login_screenshot.png`**:
   - **Description**: Synthetic login UI screenshot with credential fields.
   - **License**: Project-generated synthetic data.

3. **`github_profile_screenshot.png`**:
   - **Description**: Synthetic profile UI screenshot.
   - **License**: Project-generated synthetic data.

4. **`face_single.png`**:
   - **Description**: High-resolution synthetic portrait photo of a single adult person.
   - **Content**: Exactly 1 clearly visible frontal face on a neutral background. No real private individual's identity or likeness is used.
   - **License**: Synthetically generated AI photo / Free for testing and project use.

5. **`face_multi.png`**:
   - **Description**: High-resolution synthetic photo of three diverse colleagues standing side-by-side in an office setting.
   - **Content**: Exactly 3 clearly visible human faces.
   - **License**: Synthetically generated AI photo / Free for testing and project use.

6. **`face_with_pii.png`**:
   - **Description**: Combined test image based on `face_single.png` with a prominent employee badge card overlay containing synthetic text PII (name, email, phone number).
   - **Content**: 1 human face + structured text PII.
   - **License**: Synthetically generated AI photo and graphics / Free for testing and project use.
