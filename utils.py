import os
import re
import uuid
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cricket_ads")


def generate_match_id():
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    short = uuid.uuid4().hex[:6]
    return f"match_{ts}_{short}"


def seconds_to_timestamp(sec):
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:05.2f}"


def sanitize_filename(name):
    return re.sub(r'[^\w\-.]', '_', name)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def format_duration(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def calculate_visibility_ratio(brand_dur, total_dur):
    if total_dur <= 0:
        return 0.0
    return round((brand_dur / total_dur) * 100, 2)


def merge_intervals(intervals, gap=2.0):
    """Merge overlapping or close time intervals together."""
    if not intervals:
        return []
    sorted_iv = sorted(intervals, key=lambda x: x[0])
    merged = [sorted_iv[0]]
    for start, end in sorted_iv[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def flatten_detections_for_rag(detections):
    """Turn detection dicts into readable text strings for embedding."""
    texts = []
    for d in detections:
        ts = seconds_to_timestamp(d.get("timestamp", 0))
        line = (
            f"Brand '{d.get('brand_name', 'unknown')}' detected at {ts} "
            f"with {d.get('confidence', 0):.0%} confidence. "
            f"Placement: {d.get('placement', 'unknown')}. "
            f"Event: {d.get('event', 'none')}. "
            f"Source: {d.get('detection_source', 'yolo')}."
        )
        texts.append(line)
    return texts
