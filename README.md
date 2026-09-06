# Privacy-Safe Screenshot OCR + PII Detection & Anonymization Service

A **local, privacy-first** backend service that extracts text from screenshots using Tesseract OCR, detects personally identifiable information (PII) using Microsoft Presidio, and returns only sanitised text — keeping all raw data strictly on your machine.

---

## Overview

```
Screenshot / Image
        ↓
  Tesseract OCR          (local — no cloud)
        ↓
  Raw OCR Text           (never logged, never returned)
        ↓
  Presidio Analyzer      (local NLP — spaCy en_core_web_lg)
        ↓
  PII Detection          (entity types + confidence only)
        ↓
  Presidio Anonymizer    (replace PII with [PLACEHOLDERS])
        ↓
  Safe / Sanitised Text
        ↓
  JSON API Response
```

**Privacy guarantee**: Raw OCR text and detected PII values are *never* returned in API responses, *never* written to logs, and *never* sent to external services.

---

## Architecture

```
project-root/
├── backend/
│   ├── __init__.py       — package marker
│   ├── app.py            — Flask application & routes
│   ├── config.py         — all tuneable configuration
│   ├── ocr.py            — Tesseract OCR + preprocessing
│   ├── pii.py            — Presidio Analyzer + Anonymizer (singleton engines)
│   └── processor.py      — full pipeline orchestration
│
├── tests/
│   ├── test_ocr.py       — OCR unit tests
│   ├── test_pii.py       — PII detection / anonymisation tests
│   ├── test_processor.py — full pipeline tests
│   ├── test_api.py       — Flask API tests (includes privacy tests)
│   └── test_cli.py       — CLI tool unit tests (mocked HTTP)
│
├── tools/
│   └── test_image.py     — CLI: send any image and view a privacy report
│
├── test_data/
│   └── test_input.png    — synthetic test image (fictional PII only)
│
├── create_test_image.py  — generates test_data/test_input.png
├── requirements.txt
├── run_server.sh
└── README.md
```

---

## Installation

### Prerequisites

| Dependency | Version | Install |
|------------|---------|---------|
| Python | 3.11 | `brew install python@3.11` |
| Tesseract | 5.x | `brew install tesseract` |

### Setup

```bash
# 1. Create and activate virtual environment
python3.11 -m venv venv
source venv/bin/activate

# 2. Install Python dependencies
pip install -r requirements.txt

# 3. Install spaCy language model
python -m spacy download en_core_web_lg

# 4. Generate the test image
python create_test_image.py
```

---

## Running

### Option 1 — run_server.sh

```bash
chmod +x run_server.sh
./run_server.sh
```

### Option 2 — directly

```bash
source venv/bin/activate
python backend/app.py
```

The server starts at **http://127.0.0.1:5005**

---

## API

### `GET /health`

Liveness probe.

```bash
curl http://127.0.0.1:5005/health
```

```json
{"status": "ok"}
```

---

### `POST /process-image`

Main endpoint — returns sanitised text with PII replaced by placeholders.

**Request**: `multipart/form-data` with field `image` (PNG / JPEG / WEBP, max 10 MB).

```bash
curl -X POST http://127.0.0.1:5005/process-image \
     -F "image=@test_data/test_input.png"
```

**Success response** (200):

```json
{
  "success": true,
  "safe_text": "Name: [PERSON]\nEmail: [EMAIL_ADDRESS]\nPhone: [PHONE_NUMBER]\n\nCourse: Computer Science\nCollege: ABC University\nApplication ID: APP-2026-001",
  "entities_found": [
    {"type": "PERSON",        "score": 0.85},
    {"type": "EMAIL_ADDRESS", "score": 0.99},
    {"type": "PHONE_NUMBER",  "score": 0.75}
  ]
}
```

**Error responses**:

| Status | Cause |
|--------|-------|
| 400 | Missing/empty/corrupted image |
| 413 | Image exceeds 10 MB |
| 415 | Unsupported file type |
| 500 | Internal processing error |

---

### `POST /redact-image`

Visual redaction — returns the original image with PII regions blacked out (PNG).

```bash
curl -X POST http://127.0.0.1:5005/redact-image \
     -F "image=@test_data/test_input.png" \
     --output redacted.png
```

Returns `image/png` on success.

---

## PII Entities Detected

| Entity | Placeholder |
|--------|-------------|
| Person name | `[PERSON]` |
| Email address | `[EMAIL_ADDRESS]` |
| Phone number | `[PHONE_NUMBER]` |
| Location | `[LOCATION]` |
| Organisation | `[ORGANIZATION]` |
| Credit card | `[CREDIT_CARD]` |
| US Social Security Number | `[US_SSN]` |
| IP address | `[IP_ADDRESS]` |
| URL | `[URL]` |
| Any other detected entity | `[REDACTED]` |

---

## Browser Extension Integration

The API is designed to work directly with a Chrome extension's `fetch` call:

```javascript
const formData = new FormData();
formData.append("image", screenshotBlob, "screenshot.png");

const response = await fetch("http://127.0.0.1:5005/process-image", {
  method: "POST",
  body: formData,
});

const result = await response.json();

if (result.success) {
  console.log("Safe text:", result.safe_text);
  console.log("Entities:", result.entities_found);
}
```

CORS is pre-configured to allow `chrome-extension://` origins.

---

## CLI Testing Tool

`tools/test_image.py` is a local testing utility that sends any image on your Mac
to the running backend and prints a clean, human-readable privacy report.
It displays the **safe text** (PII already anonymised by the backend) and a list of
detected entity types with confidence scores.  No raw PII is ever shown.

### 1. Start the backend

```bash
source venv/bin/activate
python backend/app.py
```

The server starts at `http://127.0.0.1:5005`.

### 2. Test an image

```bash
# Synthetic test image (included in the repo)
python tools/test_image.py test_data/test_input.png

# Any image on your Mac
python tools/test_image.py ~/Desktop/my_screenshot.png
```

### 3. Example output

```
============================================================
             REDACTION IMAGE TEST
============================================================

Image:
  /Users/.../test_input.png

File:
  Name: test_input.png
  Size: 30 KB
  Type: PNG

------------------------------------------------------------
OCR / SAFE TEXT RESULT
------------------------------------------------------------

  PRIVACY TEST APPLICATION

  Name: [PERSON]
  Email: [EMAIL_ADDRESS]
  Phone: [PHONE_NUMBER]

  Course: [ORGANIZATION]: [ORGANIZATION]: APP-2026-001

------------------------------------------------------------
DETECTED PII ENTITIES
------------------------------------------------------------

  Total entities detected: 7

  1. EMAIL_ADDRESS
     Confidence: 1.00

  2. PERSON
     Confidence: 0.85
  ...

------------------------------------------------------------
PRIVACY CHECK
------------------------------------------------------------

  ✓ Processing successful
  ✓ Safe text generated
  ✓ PII detection completed

------------------------------------------------------------
FINAL RESULT
------------------------------------------------------------

  PASS ✓

============================================================
```

### 4. What the tool does

- Validates that the file exists and is a supported image type.
- Posts the image to `POST /process-image` on the local backend (no cloud calls).
- Displays the `safe_text` returned by the backend (PII already replaced with placeholders).
- Lists detected entity types and confidence scores (no raw PII values).
- Exits with code `0` on success, `1` on failure.

### 5. Error handling

If the backend is not running:

```
ERROR:
  Could not connect to the Redaction backend.

Backend:
  http://127.0.0.1:5005

Start it with:
  python backend/app.py

FAIL ✗
```

---

## Running Tests


```bash
source venv/bin/activate

# Run all tests
pytest tests/ -v

# Run a specific test file
pytest tests/test_api.py -v

# Run only the privacy test
pytest tests/test_api.py::TestProcessImage::test_privacy_raw_pii_not_in_response -v
```

---

## Privacy Model

| Data | Fate |
|------|------|
| Uploaded image | Processed in memory; never written to disk |
| Raw OCR text | Local variable only; never returned, logged, or stored |
| Detected PII values | Replaced by placeholders; originals discarded |
| Entity types & scores | Returned in API response (no raw values) |
| Processed image | Not stored after request completes |

**No external calls**: Tesseract and Presidio run fully locally. No data is sent to OpenAI, Google Cloud, Azure, AWS, or any third-party service.

---

## Configuration

All settings are in `backend/config.py` and can be overridden via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `127.0.0.1` | Bind address (keep local) |
| `PORT` | `5005` | Listen port |
| `DEBUG` | `false` | Enable Flask debug mode |
| `MAX_IMAGE_SIZE_BYTES` | `10485760` | Max upload size (10 MB) |
| `PII_SCORE_THRESHOLD` | `0.5` | Minimum Presidio confidence score |
| `ALLOWED_ORIGINS` | localhost + chrome-extension | CORS allowed origins |
| `SPACY_MODEL` | `en_core_web_lg` | spaCy model name |

---

## Troubleshooting

### Tesseract not found
```
pytesseract.TesseractNotFoundError
```
Install Tesseract: `brew install tesseract`  
Verify: `which tesseract && tesseract --version`

### spaCy model missing
```
OSError: Can't find model 'en_core_web_lg'
```
```bash
source venv/bin/activate
python -m spacy download en_core_web_lg
```

### Presidio initialization error
Ensure all three packages are installed in the same venv:
```bash
pip install presidio-analyzer presidio-anonymizer presidio-image-redactor
```

### Poor OCR quality
- Use high-resolution images (≥ 300 DPI recommended)
- Dark text on white/light background works best
- The preprocessing pipeline auto-upscales small images

### CORS errors from browser extension
Add your extension ID to `ALLOWED_ORIGINS` in `backend/config.py`:
```python
ALLOWED_ORIGINS = ["chrome-extension://your-extension-id-here"]
```

### Port already in use
```bash
lsof -ti:5005 | xargs kill -9
# then restart the server
```
Or change the port:
```bash
PORT=5006 python backend/app.py
```
