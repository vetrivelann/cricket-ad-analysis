"""
Generate HTML and CSV reports for a processed match.
"""
import os
import csv
import logging
from datetime import datetime

from config import REPORTS_DIR
from utils import ensure_dir, seconds_to_timestamp, format_duration

log = logging.getLogger(__name__)


def generate_html_report(match_id):
    from database import SessionLocal, get_match, get_detections, get_aggregates

    db = SessionLocal()
    try:
        match = get_match(db, match_id)
        dets = get_detections(db, match_id)
        aggs = get_aggregates(db, match_id)
    finally:
        db.close()

    if not match:
        raise ValueError(f"Match '{match_id}' not found")

    out_dir = ensure_dir(os.path.join(REPORTS_DIR, match_id))
    path = os.path.join(out_dir, f"report_{match_id}.html")

    # build aggregate table rows
    agg_rows = ""
    for a in aggs:
        pl = ", ".join(f"{k}: {v}" for k, v in (a.placement_distribution or {}).items())
        ev = ", ".join(f"{k}: {v}" for k, v in (a.event_distribution or {}).items()) or "—"
        chunks = ""
        if a.chunk_paths and isinstance(a.chunk_paths, list):
            chunks = "<br>".join(os.path.basename(p) for p in a.chunk_paths) or "—"
        else:
            chunks = "—"

        agg_rows += f"""
        <tr>
            <td><strong>{a.brand_name}</strong></td>
            <td>{a.total_duration:.2f}s</td>
            <td>{a.visibility_ratio:.2f}%</td>
            <td>{a.detection_count}</td>
            <td>{a.avg_confidence:.2%}</td>
            <td>{pl}</td>
            <td>{ev}</td>
            <td>{chunks}</td>
        </tr>"""

    # build detection rows (limit 200)
    det_rows = ""
    for d in dets[:200]:
        det_rows += f"""
        <tr>
            <td>{d.brand_name}</td>
            <td>{d.confidence:.2%}</td>
            <td>{seconds_to_timestamp(d.timestamp)}</td>
            <td>{d.placement}</td>
            <td>{d.event}</td>
            <td>{d.detection_source}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Report - {match_id}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Inter', sans-serif; background: #f4f6fb; color: #1a1a2e; padding: 40px; }}
        .header {{
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: white; padding: 40px; border-radius: 16px;
            margin-bottom: 32px; text-align: center;
        }}
        .header h1 {{ font-size: 2rem; margin-bottom: 8px; }}
        .header p {{ opacity: 0.8; }}
        .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-bottom: 32px; }}
        .card {{
            background: white; border-radius: 12px; padding: 20px;
            text-align: center; box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        }}
        .card .num {{ font-size: 1.8rem; font-weight: 700; color: #667eea; }}
        .card .txt {{ font-size: 0.85rem; color: #666; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }}
        h2 {{ font-size: 1.3rem; color: #302b63; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 3px solid #667eea; display: inline-block; }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 24px; }}
        th {{ background: #302b63; color: white; padding: 12px 16px; text-align: left; font-size: 0.85rem; text-transform: uppercase; }}
        td {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }}
        tr:nth-child(even) {{ background: #fafbfe; }}
        tr:hover {{ background: #f0f2ff; }}
        .footer {{ text-align: center; color: #999; font-size: 0.8rem; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Cricket Ad Detection Report</h1>
        <p>{match.team_a} vs {match.team_b} | {match.match_type} | {match.location}</p>
    </div>

    <div class="cards">
        <div class="card"><div class="num">{format_duration(match.video_duration)}</div><div class="txt">Duration</div></div>
        <div class="card"><div class="num">{len(dets)}</div><div class="txt">Detections</div></div>
        <div class="card"><div class="num">{len(aggs)}</div><div class="txt">Brands</div></div>
    </div>

    <h2>Brand Visibility Summary</h2>
    <table>
        <thead><tr><th>Brand</th><th>Duration</th><th>Visibility %</th><th>Detections</th><th>Avg Confidence</th><th>Placement</th><th>Events</th><th>Chunks</th></tr></thead>
        <tbody>{agg_rows}</tbody>
    </table>

    <h2>Detection Details (up to 200)</h2>
    <table>
        <thead><tr><th>Brand</th><th>Confidence</th><th>Timestamp</th><th>Placement</th><th>Event</th><th>Source</th></tr></thead>
        <tbody>{det_rows}</tbody>
    </table>

    <div class="footer">
        Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')} |
        Cricket Ad Detection System | Jio Hotstar
    </div>
</body>
</html>"""

    with open(path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"HTML report: {path}")
    return path


def generate_csv_report(match_id):
    from database import SessionLocal, get_detections

    db = SessionLocal()
    try:
        dets = get_detections(db, match_id)
    finally:
        db.close()

    out_dir = ensure_dir(os.path.join(REPORTS_DIR, match_id))
    path = os.path.join(out_dir, f"detections_{match_id}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["brand_name", "confidence", "timestamp", "timestamp_fmt",
                         "frame_index", "placement", "event", "source", "bbox"])
        for d in dets:
            writer.writerow([
                d.brand_name, f"{d.confidence:.4f}", f"{d.timestamp:.2f}",
                seconds_to_timestamp(d.timestamp), d.frame_index,
                d.placement, d.event, d.detection_source, str(d.bbox),
            ])

    log.info(f"CSV report: {path}")
    return path


def generate_aggregates_csv(match_id):
    from database import SessionLocal, get_aggregates

    db = SessionLocal()
    try:
        aggs = get_aggregates(db, match_id)
    finally:
        db.close()

    out_dir = ensure_dir(os.path.join(REPORTS_DIR, match_id))
    path = os.path.join(out_dir, f"aggregates_{match_id}.csv")

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["brand_name", "total_duration_s", "visibility_pct",
                         "detection_count", "avg_confidence", "placements",
                         "events", "start_time", "end_time", "chunks"])
        for a in aggs:
            writer.writerow([
                a.brand_name, f"{a.total_duration:.2f}", f"{a.visibility_ratio:.2f}",
                a.detection_count, f"{a.avg_confidence:.4f}",
                str(a.placement_distribution), str(a.event_distribution),
                f"{a.start_time:.2f}", f"{a.end_time:.2f}", str(a.chunk_paths),
            ])

    log.info(f"Aggregates CSV: {path}")
    return path
