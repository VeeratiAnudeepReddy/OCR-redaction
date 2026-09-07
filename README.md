# 🔒 Privacy-Safe Screenshot OCR + PII Detection & Anonymization Service

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-000000?style=flat&logo=flask&logoColor=white)](https://flask.palletsprojects.org/)
[![Tesseract OCR](https://img.shields.io/badge/Tesseract_OCR-5.x-5C2D91?style=flat)](https://github.com/tesseract-ocr/tesseract)
[![Presidio](https://img.shields.io/badge/Microsoft_Presidio-2.2+-0078D4?style=flat&logo=microsoft&logoColor=white)](https://microsoft.github.io/presidio/)
[![Privacy-First](https://img.shields.io/badge/Privacy-100%25_Local_No_Cloud-green?style=flat)]()

A **local, privacy-first** backend service that extracts text from screenshots using **Tesseract OCR**, detects Personally Identifiable Information (PII) using **Microsoft Presidio**, and returns only sanitised, anonymised text or visually redacted images — ensuring all raw data remains 100% on your machine.

---

## 📑 Table of Contents

- [Overview & Privacy Guarantees](#-overview--privacy-guarantees)
- [Architecture & Data Flow](#-architecture--data-flow)
- [Repository Structure](#-repository-structure)
- [Quick Start](#-quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation & Setup](#installation--setup)
- [Running the Backend](#-running-the-backend)
- [REST API Reference](#-rest-api-reference)
  - [`GET /health`](#get-health)
  - [`POST /process-image`](#post-process-image)
  - [`POST /redact-image`](#post-redact-image)
- [Multi-Variant OCR Engine](#-multi-variant-ocr-engine)
- [CLI Testing Tool](#-cli-testing-tool)
- [Supported PII Entity Types](#-supported-pii-entity-types)
- [Browser Extension Integration](#-browser-extension-integration)
- [Configuration Reference](#-configuration-reference)
- [Testing](#-testing)
- [Troubleshooting & FAQ](#-troubleshooting--faq)

---

## 🛡️ Overview & Privacy Guarantees

When taking screenshots of web apps, dashboards, or sensitive user interfaces (e.g. YouTube Studio, email clients, billing portals), screenshots often contain sensitive user data (names, emails, phone numbers, API keys).

This service acts as an **on-device privacy shield**. It extracts text from any uploaded image, detects PII entities, and sanitises the text before it leaves the local environment.

```
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Screenshot / UI │ ──> │ Multi-Variant OCR     │ ──> │ Microsoft Presidio    │
│ Upload (Memory) │     │ (Local Tesseract 5.x) │     │ (Local spaCy NLP)     │
└─────────────────┘     └───────────────────────┘     └───────────────────────┘
                                                                  │
                                                                  ▼
┌─────────────────┐     ┌───────────────────────┐     ┌───────────────────────┐
│ Safe JSON /     │ <── │ Anonymizer / Redactor │ <── │ PII Entity Detection  │
│ Redacted PNG    │     │ (Replace with Token)  │     │ (Type + Score Only)   │
└─────────────────┘     └───────────────────────┘     └───────────────────────┘
```

### Core Privacy Guarantees
> [!IMPORTANT]
> - **100% On-Device Execution**: Tesseract OCR and Microsoft Presidio run strictly on your local hardware. No data is ever sent to OpenAI, Google Cloud, AWS, or any external service.
> - **Zero Disk Persistence**: Uploaded images are processed entirely in-memory and discarded after the request completes.
> - **Zero Raw Text Logging**: Raw OCR text and original PII values are **never logged**, **never cached**, and **never returned** in API responses.
> - **Sanitised Outputs Only**: The API returns only anonymised `safe_text` (with placeholders like `[PERSON]`, `[EMAIL_ADDRESS]`) and metadata of detected entity types and confidence scores.

---

## 🏗️ Architecture & Data Flow

```
+-----------------------------------------------------------------------------------+
|                                  FLASK APP (app.py)                               |
|                                                                                   |
|  POST /process-image ─────────────────────────────┐                               |
|  POST /redact-image  ──────────────────────────┐  │                               |
+------------------------------------------------│--│-------------------------------+
                                                 │  │
                                                 ▼  ▼
+-----------------------------------------------------------------------------------+
|                              PROCESSOR (processor.py)                             |
|                                                                                   |
|  1. Validate image format, byte size & dimensions                                 |
|  2. Decode image in-memory via PIL                                                |
+-----------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+-----------------------------------------------------------------------------------+
|                                OCR ENGINE (ocr.py)                                |
|                                                                                   |
|  Multi-Variant Candidate Pipeline:                                                |
|    • Candidate 0: Grayscale                                                       |
|    • Candidate 1: Grayscale + Inverted (handles dark mode UI themes)              |
|    • Candidate 2: Smart-Upscaled + Contrast Boost                                 |
|    • Candidate 3: Smart-Upscaled + Inverted + Contrast Boost                      |
|                                                                                   |
|  Evaluation Heuristic:                                                            |
|    • Evaluates all candidates across PSM 6 (blocks) and PSM 11 (sparse UI text)   |
|    • Selects winning variant based on ratio of valid word-like tokens             |
+-----------------------------------------------------------------------------------+
                                                 │
                                                 ▼
+-----------------------------------------------------------------------------------+
|                              PII ENGINE (pii.py)                                  |
|                                                                                   |
|  1. Presidio Analyzer (spaCy en_core_web_lg model)                                |
|  2. Confidence threshold filtering (default: >= 0.5)                              |
|  3. Presidio Anonymizer -> Replaces PII with brackets (e.g. [PHONE_NUMBER])        |
|  4. Presidio Image Redactor -> Draws black bounding boxes (for /redact-image)     |
+-----------------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```
Redaction/
├── backend/
│   ├── __init__.py        # Package marker
│   ├── app.py             # Flask Web Server, API endpoints, CORS, error handlers
│   ├── config.py          # Centralised configuration & environment variable loading
│   ├── ocr.py             # Multi-variant Tesseract OCR engine + quality heuristic
│   ├── pii.py             # Presidio Analyzer, Anonymizer & Image Redactor engines
│   └── processor.py       # End-to-end processing pipeline & validation
│
├── tests/
│   ├── __init__.py        # Package marker
│   ├── test_api.py        # Flask HTTP endpoints, CORS & privacy boundary tests
│   ├── test_cli.py        # CLI test tool unit tests
│   ├── test_ocr.py        # OCR preprocessing & multi-variant engine unit tests
│   ├── test_pii.py        # PII detection & text anonymization unit tests
│   └── test_processor.py  # Image validation & processor pipeline tests
│
├── tools/
│   ├── __init__.py        # Package marker
│   └── test_image.py      # CLI utility: test any local image and print a privacy report
│
├── test_data/
│   ├── test_input.png     # Synthetic test screenshot with sample PII
│   └── login_screenshot.png # Synthetic login screen test asset
│
├── create_test_image.py   # Script to generate synthetic test images
├── requirements.txt       # Python dependencies
├── run_server.sh          # One-click shell script to start the local backend
└── README.md              # Project documentation
```

---

## 🚀 Quick Start

### Prerequisites

| Component | Minimum Version | Installation Command |
| :--- | :--- | :--- |
| **Python** | 3.11+ | `brew install python@3.11` *(macOS)* |
| **Tesseract OCR** | 5.x | `brew install tesseract` *(macOS)* or `sudo apt install tesseract-ocr` *(Ubuntu)* |

> [!TIP]
> **Windows Users**: Install Tesseract via [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki). The service automatically detects standard Windows install paths (`C:\Program Files\Tesseract-OCR\tesseract.exe`).

---

### Installation & Setup

1. **Clone the repository & enter the directory**:
   ```bash
   git clone https://github.com/VeeratiAnudeepReddy/OCR-redaction.git
   cd OCR-redaction
   ```

2. **Create and activate a Python 3.11 virtual environment**:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the spaCy NLP model** (required by Microsoft Presidio):
   ```bash
   python -m spacy download en_core_web_lg
   ```

5. **Generate the synthetic test image**:
   ```bash
   python create_test_image.py
   ```

---

## 🏃 Running the Backend

### Option 1 — Using the Startup Script (Recommended)

```bash
chmod +x run_server.sh
./run_server.sh
```

### Option 2 — Running Directly via Python

```bash
source venv/bin/activate
python backend/app.py
```

The Flask server will start locally at **`http://127.0.0.1:5005`**.

To verify the server is running, send a health probe:
```bash
curl http://127.0.0.1:5005/health
# Output: {"status": "ok"}
```

---

## 📡 REST API Reference

### `GET /health`
Liveness probe to verify server availability.

- **URL**: `http://127.0.0.1:5005/health`
- **Method**: `GET`
- **Response**: `200 OK`
  ```json
  {
    "status": "ok"
  }
  ```

---

### `POST /process-image`
Extracts text from an image, sanitises all detected PII, and returns anonymised text with detected entity metadata.

- **URL**: `http://127.0.0.1:5005/process-image`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `image` *(File: PNG, JPEG, WEBP; Max 10 MB)*

#### Example Request (`curl`):
```bash
curl -X POST http://127.0.0.1:5005/process-image \
     -F "image=@test_data/test_input.png"
```

#### Success Response (`200 OK`):
```json
{
  "success": true,
  "safe_text": "PRIVACY TEST APPLICATION\n\nName: [PERSON]\nEmail: [EMAIL_ADDRESS]\nPhone: [PHONE_NUMBER]\n\nCourse: Computer Science\nCollege: ABC University\n\nApplication ID: APP-2026-001",
  "entities_found": [
    {
      "type": "EMAIL_ADDRESS",
      "score": 1.0
    },
    {
      "type": "PERSON",
      "score": 0.85
    },
    {
      "type": "PHONE_NUMBER",
      "score": 0.75
    }
  ]
}
```

---

### `POST /redact-image`
Generates a visually redacted image by painting solid black rectangles over detected PII regions on the original image.

- **URL**: `http://127.0.0.1:5005/redact-image`
- **Method**: `POST`
- **Content-Type**: `multipart/form-data`
- **Form Field**: `image` *(File: PNG, JPEG, WEBP; Max 10 MB)*
- **Response**: `image/png` binary stream.

#### Example Request (`curl`):
```bash
curl -X POST http://127.0.0.1:5005/redact-image \
     -F "image=@test_data/test_input.png" \
     --output redacted_output.png
```

---

### HTTP Status Codes & Error Responses

| Status Code | Reason | Example Response |
| :--- | :--- | :--- |
| `400 Bad Request` | Missing file field, empty file, or unreadable image | `{"success": false, "error": "No image file provided."}` |
| `413 Payload Too Large` | Image file size exceeds 10 MB limit | `{"success": false, "error": "Image exceeds maximum size of 10 MB."}` |
| `415 Unsupported Media Type` | File extension or MIME type not supported | `{"success": false, "error": "Unsupported image format '.gif'."}` |
| `404 Not Found` | Non-existent route requested | `{"success": false, "error": "Endpoint not found."}` |
| `405 Method Not Allowed` | Incorrect HTTP verb used | `{"success": false, "error": "Method not allowed."}` |
| `500 Internal Server Error` | Unexpected processing failure | `{"success": false, "error": "An internal error occurred."}` |

---

## 🧠 Multi-Variant OCR Engine

Standard Tesseract OCR often produces garbled output when given screenshots with dark themes (e.g. YouTube Studio, VS Code dark mode, dark terminal windows) because Tesseract is trained primarily on dark text on white paper.

To solve this, `backend/ocr.py` implements a **multi-variant candidate pipeline**:

```
                       Input Image (PIL)
                              │
       ┌──────────────────────┼──────────────────────┐
       ▼                      ▼                      ▼
  Candidate 0            Candidate 1            Candidate 2 & 3
  Grayscale              Grayscale              Smart-Upscaled (Lanczos)
                         Inverted               + Contrast Boost (1.5x)
                         (Dark UI -> Light UI)  + Inverted Variants
       │                      │                      │
       └──────────────────────┼──────────────────────┘
                              ▼
               Evaluate against PSM Modes
               • PSM 6  (Uniform block of text)
               • PSM 11 (Sparse UI text)
                              │
                              ▼
                 Objective Quality Heuristic
              _ocr_quality_score(text: str) -> float
                              │
                              ▼
                Highest Scoring Result Chosen
```

### Objective Quality Heuristic (`_ocr_quality_score`)
The engine scores each candidate OCR result based on:
1. **Word-like Token Ratio**: Percentage of tokens containing recognizable alphanumeric patterns vs random noise/symbols.
2. **Content Length Bonus**: Reward for extracting coherent textual content.
3. **Privacy Safety**: The evaluation score is calculated purely on structural patterns — **raw text is never logged during candidate selection**.

---

## 🛠️ CLI Testing Tool

The project includes a command-line utility ([`tools/test_image.py`](file:///Users/anudeepreddyveerati/Redaction/tools/test_image.py)) to quickly test any screenshot on your Mac or PC against the running local server and output a clean privacy report.

### Usage

1. **Start the backend server in one terminal**:
   ```bash
   source venv/bin/activate
   python backend/app.py
   ```

2. **Run the CLI tool in another terminal**:
   ```bash
   # Test synthetic input image
   python tools/test_image.py test_data/test_input.png

   # Test any custom screenshot on your machine
   python tools/test_image.py ~/Desktop/my_screenshot.png
   ```

### Sample Output

```
============================================================
             REDACTION IMAGE TEST
============================================================

Image:
  /Users/.../Redaction/test_data/test_input.png

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

  Course: Computer Science
  College: ABC University

  Application ID: APP-2026-001

------------------------------------------------------------
DETECTED PII ENTITIES
------------------------------------------------------------

  Total entities detected: 3

  1. EMAIL_ADDRESS
     Confidence: 1.00

  2. PERSON
     Confidence: 0.85

  3. PHONE_NUMBER
     Confidence: 0.75

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

---

## 🏷️ Supported PII Entity Types

The Presidio PII Engine detects and anonymises the following entities:

| Entity Type | Description | Replacement Token |
| :--- | :--- | :--- |
| `PERSON` | Full names & individual names | `[PERSON]` |
| `EMAIL_ADDRESS` | Email addresses | `[EMAIL_ADDRESS]` |
| `PHONE_NUMBER` | International & national phone numbers | `[PHONE_NUMBER]` |
| `LOCATION` | Cities, countries, street addresses | `[LOCATION]` |
| `ORGANIZATION` | Company, institution, or organization names | `[ORGANIZATION]` |
| `CREDIT_CARD` | Credit card numbers | `[CREDIT_CARD]` |
| `US_SSN` | US Social Security Numbers | `[US_SSN]` |
| `IP_ADDRESS` | IPv4 and IPv6 addresses | `[IP_ADDRESS]` |
| `URL` | Web addresses and links | `[URL]` |
| `USERNAME` | Label-prefixed usernames (e.g. `Username: bob99`, `Login: admin`) | `[USERNAME]` |
| `SOCIAL_HANDLE` | Bare @handles anywhere in text (e.g. `@dev_jane`) | `[HANDLE]` |
| *Other Entities* | Any additional entity recognized by Presidio | `[REDACTED]` |

---

## 🌐 Browser Extension Integration

The backend is pre-configured with **CORS support** allowing direct `fetch()` requests from Manifest V3 Chrome Extensions (`chrome-extension://` origins).

### Example Content Script / Extension Code (`js`):

```javascript
async function sendScreenshotToRedactor(imageBlob) {
  const formData = new FormData();
  formData.append("image", imageBlob, "screenshot.png");

  try {
    const response = await fetch("http://127.0.0.1:5005/process-image", {
      method: "POST",
      body: formData,
    });

    const data = await response.json();

    if (data.success) {
      console.log("Sanitised Safe Text:", data.safe_text);
      console.log("Detected Entities:", data.entities_found);
      return data.safe_text;
    } else {
      console.error("Redaction Error:", data.error);
    }
  } catch (error) {
    console.error("Could not connect to Redaction backend:", error);
  }
}
```

---

## ⚙️ Configuration Reference

All tuneable settings are managed in [`backend/config.py`](file:///Users/anudeepreddyveerati/Redaction/backend/config.py) and can be customized via environment variables:

| Environment Variable | Default Value | Description |
| :--- | :--- | :--- |
| `HOST` | `127.0.0.1` | Server bind host (kept local for privacy) |
| `PORT` | `5005` | Server port |
| `DEBUG` | `false` | Enable/disable Flask debug mode |
| `MAX_IMAGE_SIZE_BYTES` | `10485760` | Maximum allowed image upload size in bytes (10 MB) |
| `PII_SCORE_THRESHOLD` | `0.5` | Minimum Presidio NLP detection confidence score (0.0 to 1.0) |
| `SPACY_MODEL` | `en_core_web_lg` | spaCy model used by Presidio Analyzer |
| `ALLOWED_ORIGINS` | localhost + `chrome-extension://*` | Whitelisted CORS origins |

---

## 🧪 Testing

The repository contains a full test suite with **95+ unit tests** covering Flask API endpoints, OCR preprocessing, PII anonymization, image validation, CLI tools, and privacy boundary enforcement.

### Run All Unit Tests
```bash
source venv/bin/activate
pytest tests/ -v
```

### Run Specific Test Modules
```bash
# Test API endpoints & CORS
pytest tests/test_api.py -v

# Test OCR multi-variant pipeline
pytest tests/test_ocr.py -v

# Test PII detection & anonymization
pytest tests/test_pii.py -v

# Test CLI tool logic
pytest tests/test_cli.py -v

# Test privacy boundary (verify raw text is NEVER returned)
pytest tests/test_api.py::TestProcessImage::test_privacy_raw_pii_not_in_response -v
```

---

## ❓ Troubleshooting & FAQ

### 1. `pytesseract.TesseractNotFoundError`
- **Cause**: Tesseract OCR binary is not installed or not in your system `PATH`.
- **Fix (macOS)**: Run `brew install tesseract`. Verify with `tesseract --version`.
- **Fix (Windows)**: Install Tesseract via UB-Mannheim installer. Ensure `C:\Program Files\Tesseract-OCR` is added to your Environment `PATH`.

---

### 2. `OSError: Can't find model 'en_core_web_lg'`
- **Cause**: spaCy language model has not been downloaded into the active virtual environment.
- **Fix**: Run:
  ```bash
  source venv/bin/activate
  python -m spacy download en_core_web_lg
  ```

---

### 3. Port `5005` is already in use
- **Cause**: Another process or background instance of the backend is already running on port 5005.
- **Fix**: Kill the existing process:
  ```bash
  lsof -ti:5005 | xargs kill -9
  ```
  Or launch on a custom port:
  ```bash
  PORT=5006 python backend/app.py
  ```

---

### 4. Low OCR Accuracy on Dark-Theme Screenshots
- **Solution**: The backend automatically runs dark-theme detection and inverts background colors before OCR. For best results, ensure screenshot resolution is reasonably high (at least 800px wide).

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
hon
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
