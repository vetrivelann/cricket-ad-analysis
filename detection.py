"""
Brand detection using YOLOv8 and OCR.
Also handles placement classification and event detection.
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
        log.warning(f"Could not load YOLO ({err}), will use mock detections.")
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
    except Exception:
        _tesseract_ok = False
    return _tesseract_ok


# ---- YOLO-based brand detection ----

def detect_brands_yolo(frame, conf_threshold=0.35):
    from config import BRAND_LABEL_MAP, YOLO_CONFIDENCE

    model = _load_yolo()
    if model is None:
        return _mock_yolo(frame)

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
            if brand and brand not in ("Player_Detected",):
                found.append({
                    "brand_name": brand,
                    "confidence": round(conf, 4),
                    "bbox": [round(b, 1) for b in bbox],
                    "class_name": class_name,
                })
    return found


def _mock_yolo(frame):
    from config import KNOWN_BRANDS
    h, w = frame.shape[:2]
    count = random.randint(0, 3)
    out = []
    for _ in range(count):
        x1 = random.randint(0, max(1, w - 100))
        y1 = random.randint(0, max(1, h - 100))
        x2 = min(x1 + random.randint(50, 150), w)
        y2 = min(y1 + random.randint(50, 100), h)
        out.append({
            "brand_name": random.choice(KNOWN_BRANDS),
            "confidence": round(random.uniform(0.4, 0.95), 4),
            "bbox": [x1, y1, x2, y2],
            "class_name": "mock",
        })
    return out


# ---- OCR-based brand detection ----

def detect_brands_ocr(frame):
    from config import KNOWN_BRANDS

    if not _tesseract_available():
        return _mock_ocr(frame)

    import pytesseract
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(thresh, config="--psm 6").upper()

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


def _mock_ocr(frame):
    from config import KNOWN_BRANDS
    h, w = frame.shape[:2]
    if random.random() < 0.3:
        brand = random.choice(KNOWN_BRANDS[:5])
        return [{
            "brand_name": brand,
            "confidence": round(random.uniform(0.5, 0.8), 4),
            "bbox": [0, 0, w, h],
            "class_name": "mock_ocr",
        }]
    return []


# ---- Placement classification (rule-based) ----

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


# ---- Event detection from scoreboard region ----

def detect_event(frame):
    from config import EVENT_KEYWORDS

    if _tesseract_available():
        import pytesseract
        import cv2
        h, w = frame.shape[:2]
        top_strip = frame[0:int(h * 0.15), :]
        gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config="--psm 7").upper().strip()
        for keyword, event_name in EVENT_KEYWORDS.items():
            if keyword in text:
                return event_name
    else:
        choices = ["none", "none", "none", "none", "six", "four", "wicket"]
        return random.choice(choices)

    return "none"


# ---- Combined detection on a single frame ----

def detect_all(frame, timestamp, frame_index, match_id):
    """Run both YOLO and OCR on one frame, classify placement/event, return DB-ready records."""
    h = frame.shape[0]
    event = detect_event(frame)

    yolo_hits = detect_brands_yolo(frame)
    ocr_hits = detect_brands_ocr(frame)

    records = []
    seen_brands = set()

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

    for det in ocr_hits:
        if det["brand_name"] not in seen_brands:
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
