"""
Smart analytics and insights module powered by Gemini LLM.
Generates strategic recommendations from brand detection data.
"""
import logging
from collections import defaultdict
from gemini_client import ask_gemini, is_available as gemini_available

log = logging.getLogger(__name__)


def generate_insights(detections, aggregates, match_info=None):
    """
    Analyze brand detection data and generate strategic insights.
    Uses Gemini when available, otherwise falls back to rule-based analysis.
    Returns a dict with structured insight sections.
    """
    stats = _compute_stats(detections, aggregates)

    if gemini_available():
        return _gemini_insights(stats, match_info)
    return _rule_based_insights(stats, match_info)


def _compute_stats(detections, aggregates):
    """Distill raw detections into a compact stats dict for LLM consumption."""
    brand_counts = defaultdict(int)
    brand_events = defaultdict(lambda: defaultdict(int))
    brand_placements = defaultdict(lambda: defaultdict(int))
    brand_confidence = defaultdict(list)

    for d in detections:
        brand = d.get("brand_name") if isinstance(d, dict) else d.brand_name
        brand_counts[brand] += 1

        conf = d.get("confidence") if isinstance(d, dict) else d.confidence
        brand_confidence[brand].append(float(conf))

        event = d.get("event") if isinstance(d, dict) else d.event
        if event and event != "none":
            brand_events[brand][event] += 1

        placement = d.get("placement") if isinstance(d, dict) else d.placement
        brand_placements[brand][placement] += 1

    # build ranking by total presence
    brand_ranking = sorted(brand_counts.items(), key=lambda x: -x[1])

    agg_info = []
    for a in aggregates:
        name = a.get("brand_name") if isinstance(a, dict) else a.brand_name
        dur = a.get("total_duration") if isinstance(a, dict) else a.total_duration
        vis = a.get("visibility_ratio") if isinstance(a, dict) else a.visibility_ratio
        avg_c = a.get("avg_confidence") if isinstance(a, dict) else a.avg_confidence
        agg_info.append({
            "brand": name,
            "duration_seconds": round(float(dur), 2),
            "visibility_pct": round(float(vis), 2),
            "avg_confidence": round(float(avg_c), 4),
        })

    return {
        "brand_ranking": brand_ranking,
        "brand_events": {k: dict(v) for k, v in brand_events.items()},
        "brand_placements": {k: dict(v) for k, v in brand_placements.items()},
        "aggregates": agg_info,
        "total_detections": sum(brand_counts.values()),
    }


def _gemini_insights(stats, match_info):
    """Use Gemini to generate rich, context-aware insights."""
    match_ctx = ""
    if match_info:
        match_ctx = (
            f"Match: {match_info.get('team_a', 'Team A')} vs "
            f"{match_info.get('team_b', 'Team B')} "
            f"({match_info.get('match_type', 'T20')}) at "
            f"{match_info.get('location', 'Unknown')}\n"
        )

    prompt = f"""You are a senior sports marketing analytics consultant. Analyze the following 
brand advertisement detection data from a live cricket broadcast and provide actionable insights.

{match_ctx}

BRAND DETECTION SUMMARY:
Total detections: {stats['total_detections']}

Brand Ranking (by detection count):
{_format_ranking(stats['brand_ranking'])}

Brand Visibility Metrics:
{_format_aggregates(stats['aggregates'])}

Brand Placement Distribution:
{_format_dict_of_dicts(stats['brand_placements'])}

Brand Appearances During Cricket Events:
{_format_dict_of_dicts(stats['brand_events'])}

Provide your analysis in the following sections:
1. EXECUTIVE SUMMARY (2-3 sentences overview)
2. TOP PERFORMING BRANDS (which brands dominated and why)
3. AD PLACEMENT EFFECTIVENESS (which placement zones drive highest visibility)
4. EVENT IMPACT ANALYSIS (how cricket events like sixes/fours/wickets affect brand visibility)
5. STRATEGIC RECOMMENDATIONS (3-5 actionable recommendations for advertisers)

Be specific, reference actual numbers, and focus on actionable intelligence."""

    response = ask_gemini(prompt, temperature=0.4, max_tokens=1500)

    if response:
        return {"source": "gemini", "content": response, "stats": stats}

    return _rule_based_insights(stats, match_info)


def _rule_based_insights(stats, match_info=None):
    """Fallback rule-based insight generation when Gemini is unavailable."""
    sections = []

    ranking = stats["brand_ranking"]
    total = stats["total_detections"]

    # executive summary
    if ranking:
        top = ranking[0]
        sections.append(
            f"**Executive Summary**: Analyzed {total} brand detections across "
            f"{len(ranking)} brands. {top[0]} leads with {top[1]} detections "
            f"({top[1] / total * 100:.1f}% of all appearances)."
        )

    # top brands
    sections.append("\n**Top Performing Brands:**")
    for brand, count in ranking[:5]:
        pct = count / total * 100 if total > 0 else 0
        sections.append(f"  • {brand}: {count} detections ({pct:.1f}%)")

    # placement analysis
    all_placements = defaultdict(int)
    for brand_pl in stats["brand_placements"].values():
        for pl, cnt in brand_pl.items():
            all_placements[pl] += cnt
    if all_placements:
        sections.append("\n**Placement Distribution:**")
        for pl, cnt in sorted(all_placements.items(), key=lambda x: -x[1]):
            sections.append(f"  • {pl}: {cnt} detections")

    # event impact
    all_events = defaultdict(int)
    for brand_ev in stats["brand_events"].values():
        for ev, cnt in brand_ev.items():
            all_events[ev] += cnt
    if all_events:
        sections.append("\n**Event Impact:**")
        for ev, cnt in sorted(all_events.items(), key=lambda x: -x[1]):
            sections.append(f"  • During {ev}: {cnt} brand appearances")

    # recommendations
    sections.append("\n**Recommendations:**")
    if ranking:
        sections.append(f"  • {ranking[0][0]} has strongest presence — maintain current strategy")
    if len(ranking) > 1:
        weak = ranking[-1]
        sections.append(f"  • {weak[0]} has lowest visibility — consider higher-frequency placements")
    if all_events:
        top_event = max(all_events, key=all_events.get)
        sections.append(f"  • Brands appearing during '{top_event}' events get high engagement — target these moments")

    return {"source": "rule_based", "content": "\n".join(sections), "stats": stats}


def _format_ranking(ranking):
    lines = []
    for i, (brand, count) in enumerate(ranking, 1):
        lines.append(f"  {i}. {brand}: {count} detections")
    return "\n".join(lines) if lines else "  No data"


def _format_aggregates(aggs):
    lines = []
    for a in aggs:
        lines.append(
            f"  {a['brand']}: {a['duration_seconds']}s screen time, "
            f"{a['visibility_pct']}% visibility, "
            f"{a['avg_confidence']:.2%} avg confidence"
        )
    return "\n".join(lines) if lines else "  No data"


def _format_dict_of_dicts(data):
    lines = []
    for key, sub in data.items():
        parts = ", ".join(f"{k}: {v}" for k, v in sub.items())
        lines.append(f"  {key}: {parts}")
    return "\n".join(lines) if lines else "  No data"
