"""
Generate HTML and CSV reports for a processed match.
Enhanced with Gemini LLM for executive summaries, key insights, and sponsor recommendations.
"""
import os
import csv
import logging
from datetime import datetime

from config import REPORTS_DIR
from utils import ensure_dir, seconds_to_timestamp, format_duration

log = logging.getLogger(__name__)


def _generate_executive_summary(match, dets, aggs):
    """Generate an executive summary using Gemini LLM. Falls back to template."""
    try:
        from gemini_client import ask_gemini, is_available
        if not is_available():
            return _template_summary(match, dets, aggs)

        brand_list = ", ".join(a.brand_name for a in aggs)
        top_brand = max(aggs, key=lambda a: a.detection_count) if aggs else None

        prompt = f"""Generate a brief executive summary (3-4 sentences) for a cricket advertisement 
detection report with these stats:

Match: {match.team_a} vs {match.team_b} ({match.match_type}) at {match.location}
Video Duration: {format_duration(match.video_duration)}
Total Detections: {len(dets)}
Brands Detected: {brand_list}
Top Brand: {top_brand.brand_name if top_brand else 'N/A'} ({top_brand.detection_count if top_brand else 0} detections, {top_brand.visibility_ratio:.1f}% visibility)

Write in professional, concise analytical tone. Do not use markdown formatting."""

        result = ask_gemini(prompt, temperature=0.3, max_tokens=300)
        if result:
            return result
    except Exception as e:
        log.debug(f"Gemini summary generation failed: {e}")

    return _template_summary(match, dets, aggs)


def _template_summary(match, dets, aggs):
    """Fallback template-based executive summary."""
    top = max(aggs, key=lambda a: a.detection_count) if aggs else None
    top_str = f"{top.brand_name} dominated with {top.detection_count} detections and {top.visibility_ratio:.1f}% screen visibility" if top else "No dominant brand identified"
    return (
        f"Analysis of the {match.team_a} vs {match.team_b} {match.match_type} match "
        f"at {match.location} revealed {len(dets)} brand detections across {len(aggs)} unique brands. "
        f"{top_str}. "
        f"The video duration was {format_duration(match.video_duration)}."
    )


def _generate_key_insights(dets, aggs):
    """Generate key insights using Gemini. Falls back to rule-based."""
    try:
        from gemini_client import ask_gemini, is_available
        if not is_available():
            return _template_insights(dets, aggs)

        agg_data = "\n".join(
            f"- {a.brand_name}: {a.detection_count} detections, "
            f"{a.total_duration:.1f}s duration, {a.visibility_ratio:.1f}% visibility, "
            f"placements: {a.placement_distribution}, events: {a.event_distribution}"
            for a in aggs
        )

        prompt = f"""Based on this brand detection data from a cricket broadcast, 
provide exactly 5 key insights as numbered bullet points. Be specific with numbers.

{agg_data}

Write concise, data-driven insights. Do not use markdown formatting, just plain numbered list."""

        result = ask_gemini(prompt, temperature=0.3, max_tokens=500)
        if result:
            return result
    except Exception as e:
        log.debug(f"Gemini insights failed: {e}")

    return _template_insights(dets, aggs)


def _template_insights(dets, aggs):
    """Fallback rule-based key insights."""
    lines = []
    if aggs:
        sorted_aggs = sorted(aggs, key=lambda a: a.detection_count, reverse=True)
        lines.append(f"1. {sorted_aggs[0].brand_name} is the most visible brand with {sorted_aggs[0].detection_count} detections.")
        if len(sorted_aggs) > 1:
            lines.append(f"2. {sorted_aggs[-1].brand_name} has the lowest visibility with only {sorted_aggs[-1].detection_count} detections.")

        placement_total = {}
        for a in aggs:
            for pl, cnt in (a.placement_distribution or {}).items():
                placement_total[pl] = placement_total.get(pl, 0) + cnt
        if placement_total:
            top_pl = max(placement_total, key=placement_total.get)
            lines.append(f"3. '{top_pl}' is the most common ad placement zone ({placement_total[top_pl]} appearances).")

        event_brands = []
        for a in aggs:
            if a.event_distribution:
                event_brands.append(a.brand_name)
        if event_brands:
            lines.append(f"4. Brands appearing during cricket events: {', '.join(event_brands)}.")
        else:
            lines.append("4. No brands were specifically detected during key cricket events (sixes, fours, wickets).")

        avg_conf = sum(a.avg_confidence for a in aggs) / len(aggs) if aggs else 0
        lines.append(f"5. Average detection confidence across all brands: {avg_conf:.1%}.")

    return "\n".join(lines)


def _generate_sponsor_recommendations(dets, aggs):
    """Generate sponsor recommendations using Gemini. Falls back to template."""
    try:
        from gemini_client import ask_gemini, is_available
        if not is_available():
            return _template_recommendations(aggs)

        agg_data = "\n".join(
            f"- {a.brand_name}: {a.total_duration:.1f}s screen time, "
            f"{a.visibility_ratio:.1f}% visibility, avg confidence {a.avg_confidence:.1%}, "
            f"placements: {a.placement_distribution}"
            for a in aggs
        )

        prompt = f"""As a sports marketing consultant, provide 4 actionable sponsor recommendations 
based on this cricket broadcast brand visibility data:

{agg_data}

Format as a numbered list. Focus on ROI, placement optimization, and event targeting.
Do not use markdown formatting."""

        result = ask_gemini(prompt, temperature=0.4, max_tokens=500)
        if result:
            return result
    except Exception as e:
        log.debug(f"Gemini recommendations failed: {e}")

    return _template_recommendations(aggs)


def _template_recommendations(aggs):
    """Fallback template-based recommendations."""
    lines = []
    if aggs:
        sorted_aggs = sorted(aggs, key=lambda a: a.visibility_ratio, reverse=True)
        lines.append(f"1. {sorted_aggs[0].brand_name} should maintain its strong placement strategy.")
        if len(sorted_aggs) > 1:
            lines.append(f"2. {sorted_aggs[-1].brand_name} should explore higher-frequency ad placements to increase visibility.")
        lines.append("3. Brands should target key cricket events (sixes, wickets) for maximum viewer engagement.")
        lines.append("4. Consider boundary-level placements for consistent background visibility throughout the match.")
    return "\n".join(lines)


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

    # generate LLM-powered sections
    exec_summary = _generate_executive_summary(match, dets, aggs)
    key_insights = _generate_key_insights(dets, aggs)
    recommendations = _generate_sponsor_recommendations(dets, aggs)

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

    # convert newlines to <br> for HTML rendering
    exec_summary_html = exec_summary.replace("\n", "<br>")
    key_insights_html = key_insights.replace("\n", "<br>")
    recommendations_html = recommendations.replace("\n", "<br>")

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
        .insight-box {{
            background: linear-gradient(135deg, #f5f7ff, #eef2ff);
            border-left: 4px solid #667eea;
            border-radius: 0 12px 12px 0;
            padding: 20px 24px;
            margin: 12px 0 24px 0;
            line-height: 1.7;
            box-shadow: 0 2px 8px rgba(102, 126, 234, 0.1);
        }}
        .recommend-box {{
            background: linear-gradient(135deg, #f0fff4, #e6ffed);
            border-left: 4px solid #38a169;
            border-radius: 0 12px 12px 0;
            padding: 20px 24px;
            margin: 12px 0 24px 0;
            line-height: 1.7;
            box-shadow: 0 2px 8px rgba(56, 161, 105, 0.1);
        }}
        .summary-box {{
            background: linear-gradient(135deg, #fffff0, #fffbeb);
            border-left: 4px solid #d69e2e;
            border-radius: 0 12px 12px 0;
            padding: 20px 24px;
            margin: 12px 0 24px 0;
            line-height: 1.7;
            box-shadow: 0 2px 8px rgba(214, 158, 46, 0.1);
        }}
        table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 24px; }}
        th {{ background: #302b63; color: white; padding: 12px 16px; text-align: left; font-size: 0.85rem; text-transform: uppercase; }}
        td {{ padding: 10px 16px; border-bottom: 1px solid #f0f0f0; font-size: 0.9rem; }}
        tr:nth-child(even) {{ background: #fafbfe; }}
        tr:hover {{ background: #f0f2ff; }}
        .footer {{ text-align: center; color: #999; font-size: 0.8rem; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0; }}
        .ai-badge {{
            display: inline-block; background: linear-gradient(135deg, #667eea, #764ba2);
            color: white; padding: 3px 10px; border-radius: 12px; font-size: 0.7rem;
            font-weight: 600; margin-left: 8px; vertical-align: middle;
        }}
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

    <h2>Executive Summary <span class="ai-badge">AI Generated</span></h2>
    <div class="summary-box">{exec_summary_html}</div>

    <h2>Key Insights <span class="ai-badge">AI Generated</span></h2>
    <div class="insight-box">{key_insights_html}</div>

    <h2>Sponsor Recommendations <span class="ai-badge">AI Generated</span></h2>
    <div class="recommend-box">{recommendations_html}</div>

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
        Cricket Ad Detection System — AI-Powered Analytics | Jio Hotstar
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
