"""
Face detection module — OpenCV SSD (Caffe) based face detection.

Privacy guarantees:
  * Face detection runs 100% locally/offline — no external or cloud APIs.
  * No cropped face images, biometric landmarks, or facial image byte dumps
    are logged or persisted to disk.
  * Only bounding box coordinates and confidence scores exist transiently
    in memory during processing.

Design:
  * The Caffe SSD net is initialised ONCE at import time (module-level singleton)
    mirroring the pattern used in pii.py.
  * Generous percentage-based padding (default 20%) is applied around detected
    faces to cover hair, forehead, ears, and chin.
  * Low confidence threshold (default 0.5) biases toward high recall over precision
    to avoid false negatives (missed faces).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np
from PIL import Image

import backend.config as config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FaceRegion:
    """
    A detected face bounding box in pixel coordinates.

    Matches the bounding box shape used in the redaction pipeline
    (left, top, width, height) with confidence score.
    """
    left: int
    top: int
    width: int
    height: int
    confidence: float
    raw_left: Optional[int] = None
    raw_top: Optional[int] = None
    raw_width: Optional[int] = None
    raw_height: Optional[int] = None

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


# ---------------------------------------------------------------------------
# Model Initialisation (Singleton)
# ---------------------------------------------------------------------------

def _load_face_detector() -> cv2.dnn.Net:
    """
    Construct and return the OpenCV Caffe DNN face detector network.
    Loaded once at import time for performance.
    """
    proto_path = config.FACE_PROTO_PATH
    model_path = config.FACE_MODEL_PATH

    if not os.path.isfile(proto_path) or not os.path.isfile(model_path):
        logger.error(
            "Face detection model files missing: proto=%s, model=%s",
            proto_path,
            model_path,
        )
        raise FileNotFoundError(
            f"Face detection model files not found at {proto_path} or {model_path}"
        )

    logger.info("Initialising OpenCV DNN Face Detector from %s ...", config.FACE_MODEL_DIR)
    try:
        net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
        logger.info("OpenCV DNN Face Detector initialised successfully.")
        return net
    except Exception as exc:
        logger.error("Failed to initialise OpenCV DNN Face Detector: %s", type(exc).__name__)
        raise


_face_net: cv2.dnn.Net = _load_face_detector()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_faces(
    image: Image.Image,
    confidence_threshold: Optional[float] = None,
    padding_ratio: Optional[float] = None,
) -> List[FaceRegion]:
    """
    Detect human faces in a PIL Image using the OpenCV Caffe SSD model.

    Args:
        image: PIL Image to inspect (any mode).
        confidence_threshold: Minimum confidence score (0.0 - 1.0).
            Defaults to config.FACE_DETECTION_CONFIDENCE (0.5).
        padding_ratio: Percentage of width/height added as padding
            around each face. Defaults to config.FACE_PADDING_RATIO (0.20).

    Returns:
        List of FaceRegion objects with coordinates clamped to image bounds.
        Empty list if no faces detected or image is empty.

    Privacy note:
        Zero face images, crops, or embeddings are logged or saved to disk.
        Only counts, bounding box coordinates, and confidence scores are processed.
    """
    if confidence_threshold is None:
        confidence_threshold = config.FACE_DETECTION_CONFIDENCE
    if padding_ratio is None:
        padding_ratio = config.FACE_PADDING_RATIO

    # Normalise image to RGB
    rgb_image = image.convert("RGB")
    img_w, img_h = rgb_image.size

    if img_w <= 0 or img_h <= 0:
        return []

    # Convert RGB PIL Image to BGR NumPy array for OpenCV Caffe model
    rgb_np = np.array(rgb_image)
    bgr_np = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)

    # Construct input blob (300x300, BGR mean subtraction [104, 177, 123])
    blob = cv2.dnn.blobFromImage(
        bgr_np,
        scalefactor=1.0,
        size=(300, 300),
        mean=(104.0, 177.0, 123.0),
        swapRB=False,
        crop=False,
    )

    _face_net.setInput(blob)
    detections = _face_net.forward()

    faces: List[FaceRegion] = []
    num_detections = detections.shape[2]

    for i in range(num_detections):
        confidence = float(detections[0, 0, i, 2])
        if confidence < confidence_threshold:
            continue

        # Normalized coordinates [0.0, 1.0]
        raw_box = detections[0, 0, i, 3:7] * np.array([img_w, img_h, img_w, img_h])
        x1, y1, x2, y2 = raw_box.astype(int)

        # Clamp raw coordinates to image boundaries
        x1 = max(0, min(img_w, x1))
        y1 = max(0, min(img_h, y1))
        x2 = max(0, min(img_w, x2))
        y2 = max(0, min(img_h, y2))

        raw_w = max(0, x2 - x1)
        raw_h = max(0, y2 - y1)

        # Ignore zero-area detections
        if raw_w == 0 or raw_h == 0:
            continue

        # Calculate generous padding (e.g. 20% width and height)
        pad_w = int(raw_w * padding_ratio)
        pad_h = int(raw_h * padding_ratio)

        padded_left = max(0, x1 - pad_w)
        padded_top = max(0, y1 - pad_h)
        padded_right = min(img_w, x2 + pad_w)
        padded_bottom = min(img_h, y2 + pad_h)

        padded_w = padded_right - padded_left
        padded_h = padded_bottom - padded_top

        faces.append(
            FaceRegion(
                left=padded_left,
                top=padded_top,
                width=padded_w,
                height=padded_h,
                confidence=round(confidence, 4),
                raw_left=x1,
                raw_top=y1,
                raw_width=raw_w,
                raw_height=raw_h,
            )
        )

    logger.info(
        "Face detection complete: faces_found=%d (threshold=%.2f, padding=%.0f%%).",
        len(faces),
        confidence_threshold,
        padding_ratio * 100,
    )
    return faces
