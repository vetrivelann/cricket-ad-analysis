"""
Video processing pipeline.
Handles frame extraction, running detection, aggregation, and chunk cutting.
"""
import os
import logging
import cv2
import numpy as np
from collections import defaultdict

from config import (
    FRAMES_DIR, CHUNKS_DIR, FRAME_RATE,
    FRAME_WIDTH, FRAME_HEIGHT, CHUNK_PADDING,
)
from detection import detect_all
from utils import (
    merge_intervals, sanitize_filename, ensure_dir,
    calculate_visibility_ratio, seconds_to_timestamp,
)

log = logging.getLogger(__name__)


# ---- Frame extraction from video ----

def extract_frames(video_path, fps=None, match_id=""):
    fps = fps or FRAME_RATE
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    vid_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frame_count / vid_fps if vid_fps > 0 else 0.0
    step = max(1, int(vid_fps / fps))

    log.info(f"Video: {os.path.basename(video_path)} | "
             f"native fps={vid_fps:.1f} | duration={duration:.1f}s | "
             f"sampling every {step} frames")

    frames = []
    idx = 0
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            ts = idx / vid_fps
            resized = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frames.append((count, ts, resized))
            count += 1
        idx += 1

    cap.release()
    log.info(f"Extracted {count} frames from video.")
    return frames, duration


# ---- Run detection on all extracted frames ----

def run_detection_pipeline(video_path, match_id, fps=None):
    frames, duration = extract_frames(video_path, fps=fps, match_id=match_id)
    all_dets = []

    for i, ts, frame in frames:
        dets = detect_all(frame, ts, i, match_id)
        all_dets.extend(dets)
        if i > 0 and i % 10 == 0:
            log.info(f"  frame {i}/{len(frames)} done, {len(all_dets)} detections so far")

    log.info(f"Detection finished: {len(all_dets)} detections from {len(frames)} frames")
    return all_dets, duration


# ---- Aggregate detections per brand ----

def aggregate_detections(detections, video_duration, match_id):
    groups = defaultdict(list)
    for d in detections:
        groups[d["brand_name"]].append(d)

    results = []
    for brand, dets in groups.items():
        intervals = [(d["timestamp"], d["timestamp"] + (1.0 / FRAME_RATE)) for d in dets]
        merged = merge_intervals(intervals, gap=2.0)
        total_dur = sum(e - s for s, e in merged)

        placements = defaultdict(int)
        events = defaultdict(int)
        for d in dets:
            placements[d.get("placement", "unknown")] += 1
            ev = d.get("event", "none")
            if ev != "none":
                events[ev] += 1

        confidences = [d["confidence"] for d in dets]
        avg_conf = float(np.mean(confidences)) if confidences else 0.0

        results.append({
            "match_id": match_id,
            "brand_name": brand,
            "total_duration": round(total_dur, 2),
            "visibility_ratio": calculate_visibility_ratio(total_dur, video_duration),
            "detection_count": len(dets),
            "avg_confidence": round(avg_conf, 4),
            "placement_distribution": dict(placements),
            "event_distribution": dict(events),
            "chunk_paths": [],
            "start_time": merged[0][0] if merged else 0.0,
            "end_time": merged[-1][1] if merged else 0.0,
        })

    log.info(f"Aggregated {len(results)} brands for {match_id}")
    return results


# ---- Video chunk extraction ----

def extract_chunks(video_path, aggregates, match_id):
    """Cut out video segments for each brand's appearance."""
    try:
        from moviepy.editor import VideoFileClip
        return _chunks_moviepy(video_path, aggregates, match_id)
    except ImportError:
        log.warning("moviepy not installed, using OpenCV for chunks (lower quality).")
        return _chunks_opencv(video_path, aggregates, match_id)


def _chunks_moviepy(video_path, aggregates, match_id):
    from moviepy.editor import VideoFileClip
    clip = VideoFileClip(video_path)
    vid_dur = clip.duration

    for agg in aggregates:
        brand = sanitize_filename(agg["brand_name"])
        out_dir = ensure_dir(os.path.join(CHUNKS_DIR, brand, match_id))
        paths = []

        t_start = max(0, agg["start_time"] - CHUNK_PADDING)
        t_end = min(vid_dur, agg["end_time"] + CHUNK_PADDING)
        if t_end - t_start < 0.5:
            continue

        fname = f"{t_start:.1f}_{t_end:.1f}.mp4"
        out_path = os.path.join(out_dir, fname)
        try:
            sub = clip.subclip(t_start, t_end)
            sub.write_videofile(out_path, codec="libx264",
                                audio=False, logger=None, verbose=False)
            paths.append(out_path)
            log.info(f"  chunk saved: {out_path}")
        except Exception as err:
            log.error(f"  chunk failed for {brand}: {err}")

        agg["chunk_paths"] = paths

    clip.close()
    return aggregates


def _chunks_opencv(video_path, aggregates, match_id):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    for agg in aggregates:
        brand = sanitize_filename(agg["brand_name"])
        out_dir = ensure_dir(os.path.join(CHUNKS_DIR, brand, match_id))
        paths = []

        t_start = max(0, agg["start_time"] - CHUNK_PADDING)
        t_end = agg["end_time"] + CHUNK_PADDING
        if t_end - t_start < 0.5:
            continue

        fname = f"{t_start:.1f}_{t_end:.1f}.mp4"
        out_path = os.path.join(out_dir, fname)

        frame_start = int(t_start * fps)
        frame_end = int(t_end * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_start)
        writer = cv2.VideoWriter(out_path, fourcc, fps, (FRAME_WIDTH, FRAME_HEIGHT))

        for _ in range(frame_end - frame_start):
            ret, frm = cap.read()
            if not ret:
                break
            frm = cv2.resize(frm, (FRAME_WIDTH, FRAME_HEIGHT))
            writer.write(frm)

        writer.release()
        paths.append(out_path)
        agg["chunk_paths"] = paths
        log.info(f"  chunk saved (opencv): {out_path}")

    cap.release()
    return aggregates


# ---- Full end-to-end pipeline ----

def process_video(video_path, match_id, fps=None, extract_video_chunks=True):
    """
    Complete pipeline: extract frames -> detect brands -> aggregate ->
    cut chunks -> store in database. Returns a summary dict.
    """
    from database import SessionLocal, create_match, insert_detections, insert_aggregates

    log.info(f"Starting pipeline for {match_id}")

    db = SessionLocal()
    try:
        create_match(db, match_id, video_path=video_path, status="processing")
    finally:
        db.close()

    # detection
    detections, video_duration = run_detection_pipeline(video_path, match_id, fps=fps)

    # aggregation
    aggregates = aggregate_detections(detections, video_duration, match_id)

    # chunk extraction
    if extract_video_chunks and detections:
        aggregates = extract_chunks(video_path, aggregates, match_id)

    # save to database
    db = SessionLocal()
    try:
        create_match(db, match_id, video_path=video_path,
                     status="completed", video_duration=video_duration)
        if detections:
            insert_detections(db, detections)
        if aggregates:
            insert_aggregates(db, aggregates)
        log.info(f"Pipeline done for {match_id}: "
                 f"{len(detections)} detections, {len(aggregates)} brands")
    finally:
        db.close()

    return {
        "match_id": match_id,
        "video_duration": video_duration,
        "total_detections": len(detections),
        "brands_found": len(aggregates),
        "detections": detections,
        "aggregates": aggregates,
    }
