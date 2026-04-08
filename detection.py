"""
Brand detection using YOLOv8 and OCR.
Handles placement classification and event detection.

Rules:
  - We ONLY report brands we can actually verify.
  - YOLO detects COCO objects -> mapped to likely brands via BRAND_LABEL_MAP.
  - OCR reads text from the frame -> matches against KNOWN_BRANDS.
  - OCR text overrides YOLO brand labels when actual brand text is found.
  - NO mock/fake data is ever generated. If nothing is detected, return empty.
"""
import logging
import random
import numpy as np

log = logging.getLogger(__name__)

_yolo_model = None
_tesseract_ok = None


def _load_yolo():
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model
    try:
        from ultralytics import YOLO
        from config import YOLO_MODEL
        log.info(f"Loading YOLO model: {YOLO_MODEL}")
        _yolo_model = YOLO(YOLO_MODEL)
        log.info("YOLO model loaded.")
    except Exception as err:
        log.warning(f"Could not load YOLO ({err}). YOLO detection disabled.")
        _yolo_model = None
    return _yolo_model


def _tesseract_available():
    global _tesseract_ok
    if _tesseract_ok is not None:
        return _tesseract_ok
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        _tesseract_ok = True
        log.info("Tesseract OCR is available.")
    except Exception:
        log.warning("Tesseract OCR not available. OCR detection disabled.")
        _tesseract_ok = False
    return _tesseract_ok


# ---- OCR text extraction ----

def _extract_frame_text(frame):
    """Extract all readable text from a frame using OCR. Returns uppercase text."""
    if not _tesseract_available():
        return ""
    try:
        import pytesseract
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        text = pytesseract.image_to_string(thresh, config="--psm 6").upper()
        return text
    except Exception:
        return ""


def _correct_brand_from_ocr(detected_brand, frame_text):
    """
    Use OCR text to correct YOLO's brand assignment.
    If OCR finds actual brand text in the frame, override the COCO-based label.
    """
    from config import BRAND_TEXT_PATTERNS

    if not frame_text:
        return detected_brand

    for pattern, correct_brand in BRAND_TEXT_PATTERNS.items():
        if pattern in frame_text:
            if correct_brand != detected_brand:
                log.debug(
                    f"Brand correction: '{detected_brand}' -> '{correct_brand}' "
                    f"(OCR found '{pattern}')"
                )
            return correct_brand

    return detected_brand


# ---- YOLO-based brand detection ----

def detect_brands_yolo(frame, conf_threshold=0.35, frame_text=""):
    """Detect brands using YOLO object detection.
    Returns empty list if YOLO is not available (no mock data).
    """
    from config import BRAND_LABEL_MAP, YOLO_CONFIDENCE

    model = _load_yolo()
    if model is None:
        return []

    threshold = conf_threshold or YOLO_CONFIDENCE
    results = model(frame, conf=threshold, verbose=False)
    found = []

    for result in results:
        if result.boxes is None:
            continue
        for i in range(len(result.boxes)):
            cls_id = int(result.boxes.cls[i].item())
            conf = float(result.boxes.conf[i].item())
            bbox = result.boxes.xyxy[i].tolist()
            class_name = model.names.get(cls_id, f"class_{cls_id}")

            brand = BRAND_LABEL_MAP.get(class_name)
            if brand and brand not in ("Player_Detected", "Scoreboard_Zone", "Overlay_Zone"):
                corrected_brand = _correct_brand_from_ocr(brand, frame_text)
                found.append({
                    "brand_name": corrected_brand,
                    "confidence": round(conf, 4),
                    "bbox": [round(b, 1) for b in bbox],
                    "class_name": class_name,
                })
    return found


# ---- OCR-based brand detection ----

def detect_brands_ocr(frame, frame_text=""):
    """Detect brands by reading text from the frame.
    Returns empty list if Tesseract is not available (no mock data).
    """
    from config import KNOWN_BRANDS

    if not _tesseract_available():
        return []

    text = frame_text if frame_text else _extract_frame_text(frame)
    if not text.strip():
        return []

    found = []
    h, w = frame.shape[:2]
    for brand in KNOWN_BRANDS:
        if brand.upper() in text:
            found.append({
                "brand_name": brand,
                "confidence": 0.70,
                "bbox": [0, 0, w, h],
                "class_name": "ocr_text",
            })
    return found


# ---- Placement classification ----

def classify_placement(bbox, frame_height):
    from config import PLACEMENT_RULES

    if not bbox or frame_height <= 0:
        return "unknown"

    y_center = (bbox[1] + bbox[3]) / 2.0
    ratio = y_center / frame_height

    for y_min, y_max, label in PLACEMENT_RULES:
        if y_min <= ratio <= y_max:
            return label
    return "other"


# ---- Event detection ----

def detect_event(frame, frame_text=""):
    """Detect cricket events from scoreboard text.
    Returns 'none' if nothing is found (no random events).
    """
    from config import EVENT_KEYWORDS

    if not _tesseract_available():
        return "none"

    try:
        import cv2
        import pytesseract
        h, w = frame.shape[:2]
        top_strip = frame[0:int(h * 0.15), :]
        gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)
        scoreboard_text = pytesseract.image_to_string(
            gray, config="--psm 7"
        ).upper().strip()

        for keyword, event_name in EVENT_KEYWORDS.items():
            if keyword in scoreboard_text:
                return event_name

        # try Gemini for nuanced classification
        if scoreboard_text and len(scoreboard_text) > 3:
            gemini_event = _classify_event_gemini(scoreboard_text, frame_text)
            if gemini_event:
                return gemini_event
    except Exception as err:
        log.debug(f"Event detection error: {err}")

    return "none"


def _classify_event_gemini(scoreboard_text, frame_text=""):
    """Use Gemini LLM to classify cricket events from OCR text."""
    try:
        from gemini_client import ask_gemini_json, is_available
        if not is_available():
            return None

        combined_context = f"Scoreboard: {scoreboard_text}"
        if frame_text:
            combined_context += f"\nFrame text: {frame_text[:200]}"

        prompt = f"""Classify this cricket broadcast text into one of these events:
SIX, FOUR, WICKET, WIDE, NO_BALL, or NONE (for normal play).

Text from broadcast frame:
{combined_context}

Respond with JSON: {{"event": "<event_type>", "confidence": <0.0-1.0>}}"""

        result = ask_gemini_json(prompt, temperature=0.1, max_tokens=100)
        if result and isinstance(result, dict):
            event = result.get("event", "none").lower()
            confidence = float(result.get("confidence", 0))
            if confidence >= 0.7 and event != "none":
                return event
    except Exception as err:
        log.debug(f"Gemini event classification failed: {err}")
    return None


# ---- Combined detection on a single frame ----

def detect_all(frame, timestamp, frame_index, match_id):
    """Run YOLO and OCR on one frame, classify placement/event,
    return DB-ready records. Only reports verified detections.
    """
    h = frame.shape[0]

    # extract OCR text once and reuse everywhere
    frame_text = _extract_frame_text(frame)

    event = detect_event(frame, frame_text)
    yolo_hits = detect_brands_yolo(frame, frame_text=frame_text)
    ocr_hits = detect_brands_ocr(frame, frame_text=frame_text)

    records = []
    seen_brands = set()

    # YOLO detections first (higher confidence)
    for det in yolo_hits:
        seen_brands.add(det["brand_name"])
        records.append({
            "match_id": match_id,
            "brand_name": det["brand_name"],
            "confidence": det["confidence"],
            "bbox": det["bbox"],
            "timestamp": timestamp,
            "frame_index": frame_index,
            "placement": classify_placement(det["bbox"], h),
            "event": event,
            "detection_source": "yolo",
        })

    # OCR detections (add only new brands)
    for det in ocr_hits:
        if det["brand_name"] not in seen_brands:
            seen_brands.add(det["brand_name"])
            records.append({
                "match_id": match_id,
                "brand_name": det["brand_name"],
                "confidence": det["confidence"],
                "bbox": det["bbox"],
                "timestamp": timestamp,
                "frame_index": frame_index,
                "placement": classify_placement(det["bbox"], h),
                "event": event,
                "detection_source": "ocr",
            })

    return records
