"""Dashboard generator for Pipeline v2.2 — Temporal Contiguity Analysis.

Generates a self-contained HTML report with:
  1. Executive Summary (scores, grade, zone distribution)
  2. Timeline View (segments + scenes aligned on time axis)
  3. Segment Detail Cards (transcript, keywords, matches, recommendations)
  4. Scene Detail Cards (keyframe, OCR, overlapping segments)
  5. Recommendations Panel (auto-generated actionable advice)
  6. Top 5 Priority Fixes (with visual evidence)
  7. Methodology Transparency (collapsible explainer)
  8. Delta_t Distribution Histogram
  9. OCR Evidence Trail (auditability)

Usage:
    from pipeline_v2_2_gpu.utils.viz_reports import generate_dashboard
    generate_dashboard(output_dir)
"""

import base64
import json
import logging
import os
import glob
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Zone colors
ZONE_COLORS = {
    "Optimal": "#27ae60",
    "Suboptimal": "#f39c12",
    "Disruptive": "#e67e22",
    "Unacceptable": "#e74c3c",
    "No Match": "#95a5a6",
}

CASE_LABELS = {
    "A": "Visual on screen (from scene start)",
    "B": "Visual appears later (forward search)",
    "C": "Visual expired (backward search)",
    "D": "Progressive reveal (mid-scene appearance)",
    "G": "SigLIP visual match (no OCR text)",
    "F": "No visual correlate",
}


def generate_dashboard(output_dir: str, cfg=None) -> str:
    """Generate HTML dashboard. Returns path to output file."""
    logger.info("Generating dashboard for %s", output_dir)

    # Load all data
    data = _load_all_data(output_dir)
    if data is None:
        logger.error("Cannot generate dashboard — missing data files")
        return None

    # Get video name
    video_name = os.path.basename(output_dir)

    # Build HTML
    html_parts = [
        _html_head(video_name),
        _section_executive_summary(data, video_name),
        _section_timeline(data),
        _section_segment_details(data, output_dir),
        _section_scene_details(data, output_dir),
        _section_recommendations(data),
        _section_top5_fixes(data, output_dir),
        _section_delta_histogram(data),
        _section_ocr_evidence(data),
        _section_methodology(),
        _html_footer(),
    ]

    html = "\n".join(html_parts)

    # Write
    out_path = os.path.join(output_dir, "report_dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Dashboard saved: %s", out_path)
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load_all_data(output_dir: str) -> dict:
    """Load all pipeline outputs into a dict."""
    data = {}

    def _load_csv(name):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            return pd.read_csv(path)
        return pd.DataFrame()

    def _load_json(name):
        path = os.path.join(output_dir, name)
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
        return {}

    data["results"] = _load_json("results.json")
    data["segments"] = _load_csv("transcript_segments_improved.csv")
    data["scenes"] = _load_csv("scenes.csv")
    data["keywords"] = _load_csv("keywords.csv")
    data["keyword_alignment"] = _load_csv("keyword_alignment.csv")
    data["keyword_scores"] = _load_csv("keyword_scores.csv")
    data["segment_scores"] = _load_csv("segment_scores.csv")
    data["segment_alignment"] = _load_csv("segment_alignment.csv")
    data["importance"] = _load_csv("pedagogical_importance.csv")
    data["ocr_per_frame"] = _load_csv("ocr_per_frame.csv")
    data["diag_scoring"] = _load_json(os.path.join("diagnostics", "stage7_scoring.json"))
    data["diag_alignment"] = _load_json(os.path.join("diagnostics", "stage4_alignment.json"))

    if data["results"] is None or not data["results"]:
        return None

    return data


def _load_image_base64(path: str) -> str:
    """Load image as base64 data URI. Returns empty string if not found."""
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(path)[1].lower()
        mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png"}.get(ext.strip("."), "jpeg")
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return ""


def _find_frame_at_time(output_dir: str, t: float) -> str:
    """Find the sampled frame closest to time t."""
    frames_dir = os.path.join(output_dir, "sampled_frames")
    if not os.path.isdir(frames_dir):
        return ""
    # Frame naming: frame_XXXXXXXXXX.XXXX.jpg
    best_path = ""
    best_dist = float("inf")
    for fname in os.listdir(frames_dir):
        if not fname.startswith("frame_") or not fname.endswith(".jpg"):
            continue
        try:
            time_str = fname[6:-4]  # strip "frame_" and ".jpg"
            frame_t = float(time_str)
            dist = abs(frame_t - t)
            if dist < best_dist:
                best_dist = dist
                best_path = os.path.join(frames_dir, fname)
        except ValueError:
            continue
    return best_path


def _fmt_time(seconds) -> str:
    """Format seconds as M:SS."""
    if seconds is None or pd.isna(seconds):
        return "N/A"
    m = int(float(seconds)) // 60
    s = int(float(seconds)) % 60
    return f"{m}:{s:02d}"


def _fmt_delta(dt) -> str:
    """Format delta_t with sign."""
    if dt is None or pd.isna(dt):
        return "N/A"
    dt = float(dt)
    if dt <= 0:
        return f"{dt:+.1f}s"
    return f"+{dt:.1f}s"


def _zone_for_delta(dt) -> str:
    if dt is None or pd.isna(dt):
        return "No Match"
    dt = float(dt)
    if dt <= 0:
        return "Optimal"
    if dt <= 1.0:
        return "Optimal"
    elif dt <= 3.0:
        return "Suboptimal"
    elif dt <= 5.0:
        return "Disruptive"
    return "Unacceptable"


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Head & CSS
# ═══════════════════════════════════════════════════════════════════════════════

def _html_head(video_name: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TC Analysis — {video_name}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 5px; }}
h2 {{ font-size: 1.4em; margin: 30px 0 15px; padding-bottom: 8px;
      border-bottom: 2px solid #3498db; color: #2c3e50; }}
h3 {{ font-size: 1.1em; margin: 15px 0 8px; color: #34495e; }}

/* Navigation */
.nav {{ position: sticky; top: 0; background: #2c3e50; padding: 10px 20px;
        z-index: 100; display: flex; gap: 15px; flex-wrap: wrap; }}
.nav a {{ color: #ecf0f1; text-decoration: none; font-size: 0.85em; padding: 4px 10px;
          border-radius: 4px; transition: background 0.2s; }}
.nav a:hover {{ background: #34495e; }}

/* Cards */
.card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 15px; }}

/* Summary cards */
.summary-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }}
.summary-card {{ background: white; border-radius: 8px; padding: 20px; text-align: center;
                 box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.summary-card .value {{ font-size: 2.2em; font-weight: 700; }}
.summary-card .label {{ font-size: 0.85em; color: #7f8c8d; margin-top: 5px; }}

/* Grade badges */
.grade {{ display: inline-block; padding: 4px 16px; border-radius: 20px; font-weight: 700;
          color: white; font-size: 0.9em; }}
.grade-excellent {{ background: #27ae60; }}
.grade-good {{ background: #2ecc71; }}
.grade-acceptable {{ background: #f39c12; }}
.grade-poor {{ background: #e67e22; }}
.grade-unacceptable {{ background: #e74c3c; }}
.grade-na {{ background: #95a5a6; }}

/* Zone badges */
.zone {{ display: inline-block; padding: 2px 10px; border-radius: 12px; font-size: 0.75em;
         font-weight: 600; color: white; }}
.zone-optimal {{ background: #27ae60; }}
.zone-suboptimal {{ background: #f39c12; }}
.zone-disruptive {{ background: #e67e22; }}
.zone-unacceptable {{ background: #e74c3c; }}
.zone-nomatch {{ background: #95a5a6; }}

/* Case badges */
.case-badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75em;
               font-weight: 600; color: white; margin-right: 4px; }}
.case-A {{ background: #27ae60; }}
.case-B {{ background: #e67e22; }}
.case-C {{ background: #e74c3c; }}
.case-D {{ background: #2ecc71; }}
.case-G {{ background: #9b59b6; }}
.case-F {{ background: #95a5a6; }}

/* Timeline */
.timeline-container {{ overflow-x: auto; background: white; border-radius: 8px;
                       padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
.timeline {{ position: relative; min-height: 200px; }}
.timeline-track {{ position: relative; height: 50px; margin-bottom: 10px; }}
.timeline-track-label {{ position: absolute; left: -80px; top: 15px; font-size: 0.8em;
                         font-weight: 600; color: #7f8c8d; width: 70px; text-align: right; }}
.timeline-bar {{ position: absolute; height: 36px; top: 7px; border-radius: 4px;
                 font-size: 0.65em; color: white; overflow: hidden; padding: 2px 4px;
                 cursor: pointer; transition: opacity 0.2s; min-width: 2px; }}
.timeline-bar:hover {{ opacity: 0.85; z-index: 10; box-shadow: 0 2px 8px rgba(0,0,0,0.3); }}
.timeline-axis {{ border-top: 1px solid #bdc3c7; padding-top: 5px; position: relative; height: 25px; }}
.timeline-tick {{ position: absolute; font-size: 0.7em; color: #7f8c8d; }}

/* Segment detail */
.seg-card {{ border-left: 4px solid #95a5a6; }}
.seg-card.matched {{ border-left-color: #27ae60; }}
.seg-card.partial {{ border-left-color: #f39c12; }}
.seg-card .transcript {{ background: #f8f9fa; padding: 10px; border-radius: 4px; margin: 8px 0;
                         font-size: 0.9em; line-height: 1.5; }}
.seg-card .kw-list {{ display: flex; flex-wrap: wrap; gap: 5px; margin: 8px 0; }}
.seg-card .kw-tag {{ padding: 2px 8px; border-radius: 12px; font-size: 0.75em; }}
.kw-matched {{ background: #d5f5e3; color: #1e8449; }}
.kw-unmatched {{ background: #fadbd8; color: #922b21; }}

/* Scene detail */
.scene-card {{ display: grid; grid-template-columns: 200px 1fr; gap: 15px; }}
.scene-card img {{ width: 200px; height: 140px; object-fit: cover; border-radius: 4px; }}
.scene-placeholder {{ width: 200px; height: 140px; background: #ecf0f1; border-radius: 4px;
                      display: flex; align-items: center; justify-content: center;
                      color: #95a5a6; font-size: 0.8em; }}

/* Recommendation */
.rec {{ padding: 12px 15px; border-left: 4px solid #3498db; background: #eaf2f8;
        border-radius: 0 4px 4px 0; margin-bottom: 10px; font-size: 0.9em; }}
.rec.high-priority {{ border-left-color: #e74c3c; background: #fdedec; }}
.rec.medium-priority {{ border-left-color: #f39c12; background: #fef5e7; }}
.rec .rec-label {{ font-weight: 600; font-size: 0.8em; text-transform: uppercase;
                   margin-bottom: 4px; }}

/* Histogram */
.hist-bar {{ display: flex; align-items: center; margin-bottom: 4px; }}
.hist-bar .bar-label {{ width: 180px; text-align: right; padding-right: 10px; font-size: 0.8em; }}
.hist-bar .bar {{ height: 22px; border-radius: 3px; min-width: 2px;
                  display: flex; align-items: center; padding-left: 6px;
                  font-size: 0.75em; color: white; font-weight: 600; }}

/* Collapsible */
.collapsible {{ cursor: pointer; user-select: none; }}
.collapsible::after {{ content: ' [+]'; font-size: 0.8em; color: #7f8c8d; }}
.collapsible.active::after {{ content: ' [-]'; }}
.collapsible-content {{ display: none; padding: 10px 0; }}
.collapsible-content.show {{ display: block; }}

/* Fix evidence */
.fix-card {{ display: grid; grid-template-columns: 1fr 1fr; gap: 15px; }}
.fix-frame {{ text-align: center; }}
.fix-frame img {{ max-width: 100%; height: 160px; object-fit: cover; border-radius: 4px; }}
.fix-arrow {{ display: flex; align-items: center; justify-content: center; font-size: 2em; color: #e74c3c; }}

/* Table */
table {{ width: 100%; border-collapse: collapse; font-size: 0.85em; }}
th, td {{ padding: 8px 10px; text-align: left; border-bottom: 1px solid #ecf0f1; }}
th {{ background: #f8f9fa; font-weight: 600; position: sticky; top: 0; }}
tr:hover {{ background: #f8f9fa; }}

/* OCR evidence */
.ocr-evidence {{ font-family: monospace; font-size: 0.8em; background: #f8f9fa;
                 padding: 8px; border-radius: 4px; margin-top: 5px; }}

/* Responsive */
@media (max-width: 900px) {{
    .summary-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .scene-card {{ grid-template-columns: 1fr; }}
    .fix-card {{ grid-template-columns: 1fr; }}
}}
@media print {{
    .nav {{ display: none; }}
    .card {{ break-inside: avoid; }}
}}
</style>
</head>
<body>
<div class="nav">
    <a href="#summary">Summary</a>
    <a href="#timeline">Timeline</a>
    <a href="#segments">Segments</a>
    <a href="#scenes">Scenes</a>
    <a href="#recommendations">Recommendations</a>
    <a href="#top5">Top 5 Fixes</a>
    <a href="#histogram">Distribution</a>
    <a href="#ocr-evidence">OCR Evidence</a>
    <a href="#methodology">Methodology</a>
</div>
<div class="container">
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: Executive Summary
# ═══════════════════════════════════════════════════════════════════════════════

def _section_executive_summary(data: dict, video_name: str) -> str:
    r = data["results"]
    kw = r.get("keyword_level", {})
    cov = r.get("coverage", {})
    score = r.get("overall_score", 0) or 0
    grade = r.get("overall_grade", "N/A")
    grade_cls = f"grade-{grade.lower()}" if grade != "N/A" else "grade-na"

    score_pw = r.get("overall_score_piecewise", 0) or 0
    coverage_rate = cov.get("coverage_rate", kw.get("match_rate", 0))
    cov_adjusted = cov.get("tc_score_coverage_adjusted", score * coverage_rate)

    n_matched = kw.get("n_matched", 0)
    n_total = kw.get("n_total", 0)

    # Zone percentages
    pct_opt = kw.get("pct_Optimal", 0)
    pct_sub = kw.get("pct_Suboptimal", 0)
    pct_dis = kw.get("pct_Disruptive", 0)
    pct_una = kw.get("pct_Unacceptable", 0)

    # Verdict
    if score >= 80 and coverage_rate > 0.5:
        verdict = "This video has excellent temporal alignment with good visual coverage."
    elif score >= 80:
        verdict = (f"Temporal alignment is excellent for visually-supported content, "
                   f"but {100*(1-coverage_rate):.0f}% of narrated concepts lack visual accompaniment.")
    elif score >= 60:
        verdict = "Good temporal alignment overall with some room for improvement."
    else:
        verdict = "Significant temporal alignment issues detected. See recommendations below."

    return f"""
<h1 id="summary">Temporal Contiguity Analysis: {video_name}</h1>
<p style="color: #7f8c8d; margin-bottom: 20px;">Pipeline v2.2 | Transcript-First Architecture | Asymmetric Scoring</p>

<div class="summary-grid">
    <div class="summary-card">
        <div class="value" style="color: {ZONE_COLORS.get(grade, '#333')}">{score:.1f}</div>
        <div class="label">TC Score (Gaussian)</div>
        <div><span class="{grade_cls} grade">{grade}</span></div>
    </div>
    <div class="summary-card">
        <div class="value">{score_pw:.1f}</div>
        <div class="label">TC Score (Piecewise)</div>
    </div>
    <div class="summary-card">
        <div class="value">{coverage_rate*100:.1f}%</div>
        <div class="label">Visual Coverage</div>
        <div style="font-size:0.8em;color:#7f8c8d">{n_matched}/{n_total} keywords</div>
    </div>
    <div class="summary-card">
        <div class="value">{cov_adjusted:.1f}</div>
        <div class="label">Coverage-Adjusted</div>
        <div style="font-size:0.75em;color:#7f8c8d">(TC x Coverage, composite)</div>
    </div>
</div>

<div class="card">
    <p><strong>Verdict:</strong> {verdict}</p>
    <div style="margin-top: 12px; display: flex; gap: 20px; flex-wrap: wrap;">
        <div><span class="zone zone-optimal">&nbsp;</span> Optimal: {pct_opt:.0f}%</div>
        <div><span class="zone zone-suboptimal">&nbsp;</span> Suboptimal: {pct_sub:.0f}%</div>
        <div><span class="zone zone-disruptive">&nbsp;</span> Disruptive: {pct_dis:.0f}%</div>
        <div><span class="zone zone-unacceptable">&nbsp;</span> Unacceptable: {pct_una:.0f}%</div>
    </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Timeline View
# ═══════════════════════════════════════════════════════════════════════════════

def _section_timeline(data: dict) -> str:
    scenes = data["scenes"]
    seg_scores = data["segment_scores"]
    segments = data["segments"]

    if scenes.empty or segments.empty:
        return '<h2 id="timeline">Timeline View</h2><div class="card"><p>No data available.</p></div>'

    # Find total duration
    total_dur = max(
        float(scenes["t_end"].max()) if not scenes.empty else 0,
        float(segments["end_time"].max()) if not segments.empty else 0,
    )
    if total_dur <= 0:
        total_dur = 1

    px_per_sec = 8  # pixels per second
    total_width = int(total_dur * px_per_sec) + 100

    # Build scene bars
    scene_bars = []
    for _, sc in scenes.iterrows():
        t0 = float(sc["t_start"])
        t1 = float(sc["t_end"])
        left = int(t0 * px_per_sec)
        width = max(int((t1 - t0) * px_per_sec), 3)
        sid = sc.get("scene_id", "")
        n_ocr = sc.get("n_ocr_words", 0)
        color = "#3498db" if n_ocr and int(n_ocr) > 0 else "#85c1e9"
        scene_bars.append(
            f'<div class="timeline-bar" style="left:{left}px;width:{width}px;background:{color};" '
            f'title="Scene {sid}: {_fmt_time(t0)}-{_fmt_time(t1)} ({t1-t0:.1f}s, {n_ocr} OCR words)">'
            f'S{sid}</div>'
        )

    # Build segment bars
    seg_bars = []
    for _, seg in segments.iterrows():
        sid = seg["segment_id"]
        t0 = float(seg["start_time"])
        t1 = float(seg["end_time"])
        left = int(t0 * px_per_sec)
        width = max(int((t1 - t0) * px_per_sec), 3)

        # Color by match rate from segment_scores
        match_rate = 0
        if not seg_scores.empty:
            ss = seg_scores[seg_scores["segment_id"] == sid]
            if not ss.empty:
                match_rate = float(ss.iloc[0].get("match_rate", 0))

        if match_rate >= 0.5:
            color = "#27ae60"
        elif match_rate > 0:
            color = "#f39c12"
        else:
            color = "#e74c3c"

        text_preview = str(seg["text"])[:20] + "..." if len(str(seg["text"])) > 20 else str(seg["text"])
        seg_bars.append(
            f'<div class="timeline-bar" style="left:{left}px;width:{width}px;background:{color};" '
            f'title="Seg {sid}: {_fmt_time(t0)}-{_fmt_time(t1)} | match={match_rate:.0%} | {text_preview}">'
            f'{sid}</div>'
        )

    # Time axis ticks
    ticks = []
    interval = 10 if total_dur < 120 else 30
    for t in range(0, int(total_dur) + 1, interval):
        left = int(t * px_per_sec)
        ticks.append(f'<span class="timeline-tick" style="left:{left}px">{_fmt_time(t)}</span>')

    return f"""
<h2 id="timeline">Timeline View</h2>
<div class="timeline-container" style="position: relative; padding-left: 90px;">
    <div class="timeline" style="width: {total_width}px;">
        <div class="timeline-track">
            <span class="timeline-track-label">Scenes</span>
            {"".join(scene_bars)}
        </div>
        <div class="timeline-track">
            <span class="timeline-track-label">Segments</span>
            {"".join(seg_bars)}
        </div>
        <div class="timeline-axis">
            {"".join(ticks)}
        </div>
    </div>
    <p style="font-size:0.75em; color:#95a5a6; margin-top:10px;">
        Segments: <span style="color:#27ae60">green</span>=50%+ matched,
        <span style="color:#f39c12">amber</span>=partial match,
        <span style="color:#e74c3c">red</span>=no matches.
        Scenes: <span style="color:#3498db">blue</span>=has OCR,
        <span style="color:#85c1e9">light blue</span>=no OCR.
        Hover for details.
    </p>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: Segment Details
# ═══════════════════════════════════════════════════════════════════════════════

def _section_segment_details(data: dict, output_dir: str) -> str:
    segments = data["segments"]
    seg_scores = data["segment_scores"]
    kw_scores = data["keyword_scores"]
    importance = data["importance"]

    if segments.empty:
        return '<h2 id="segments">Segment Details</h2><div class="card"><p>No segments.</p></div>'

    cards = []
    for _, seg in segments.iterrows():
        sid = seg["segment_id"]
        text = str(seg["text"])
        t0 = float(seg["start_time"])
        t1 = float(seg["end_time"])

        # Segment score data
        ss = seg_scores[seg_scores["segment_id"] == sid].iloc[0] if not seg_scores.empty and sid in seg_scores["segment_id"].values else None
        match_rate = float(ss["match_rate"]) if ss is not None else 0
        n_matched = int(ss["n_matched"]) if ss is not None else 0
        n_kw = int(ss["n_keywords"]) if ss is not None else 0
        mean_s = ss["mean_S_temporal"] if ss is not None and pd.notna(ss.get("mean_S_temporal")) else None

        # Importance
        imp_row = importance[importance["segment_id"] == sid].iloc[0] if not importance.empty and sid in importance["segment_id"].values else None
        imp_val = int(imp_row["importance"]) if imp_row is not None else 3

        # Keywords for this segment
        seg_kws = kw_scores[kw_scores["segment_id"] == sid] if not kw_scores.empty else pd.DataFrame()

        # Card class
        if match_rate >= 0.5:
            card_cls = "seg-card card matched"
        elif match_rate > 0:
            card_cls = "seg-card card partial"
        else:
            card_cls = "seg-card card"

        # Keyword tags
        kw_tags = []
        for _, kw in seg_kws.iterrows():
            kw_text = str(kw.get("keyword_text", ""))
            case = str(kw.get("match_case", "F"))
            dt = kw.get("delta_t")
            s_val = kw.get("S_temporal")

            if case != "F":
                zone = _zone_for_delta(dt)
                kw_tags.append(
                    f'<span class="kw-tag kw-matched" title="Case {case} | delta_t={_fmt_delta(dt)} | '
                    f'S={s_val:.0f}" >{kw_text} <span class="case-badge case-{case}">{case}</span></span>'
                )
            else:
                reason = str(kw.get("match_method", ""))
                kw_tags.append(
                    f'<span class="kw-tag kw-unmatched" title="No match: {reason}">{kw_text}</span>'
                )

        # Score display
        if mean_s is not None and not pd.isna(mean_s):
            score_display = f'<strong>Score: {mean_s:.1f}</strong>'
            zone = "Optimal" if mean_s >= 80 else "Suboptimal" if mean_s >= 60 else "Disruptive" if mean_s >= 40 else "Unacceptable"
            score_display += f' <span class="zone zone-{zone.lower()}">{zone}</span>'
        else:
            score_display = '<span style="color:#95a5a6">No score (no visual matches)</span>'

        cards.append(f"""
        <div class="{card_cls}">
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
                <h3>Segment {sid} <span style="font-weight:400;color:#7f8c8d;font-size:0.85em">
                    ({_fmt_time(t0)} - {_fmt_time(t1)})</span></h3>
                <div>{score_display}</div>
            </div>
            <div style="display:flex; gap:15px; font-size:0.8em; color:#7f8c8d; margin-bottom:8px;">
                <span>Keywords: {n_matched}/{n_kw} matched ({match_rate:.0%})</span>
                <span>Importance: {"*" * imp_val}{"." * (5-imp_val)} ({imp_val}/5)</span>
            </div>
            <div class="transcript">{text}</div>
            <div class="kw-list">{"".join(kw_tags)}</div>
        </div>
        """)

    return f"""
<h2 id="segments">Segment Details</h2>
{"".join(cards)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4: Scene Details
# ═══════════════════════════════════════════════════════════════════════════════

def _section_scene_details(data: dict, output_dir: str) -> str:
    scenes = data["scenes"]
    if scenes.empty:
        return '<h2 id="scenes">Scene Details</h2><div class="card"><p>No scenes.</p></div>'

    kw_align = data["keyword_alignment"]

    cards = []
    for _, sc in scenes.iterrows():
        sid = int(sc.get("scene_id", 0))
        t0 = float(sc["t_start"])
        t1 = float(sc["t_end"])
        dur = t1 - t0
        ocr_words = str(sc.get("ocr_words", ""))
        n_ocr = int(sc.get("n_ocr_words", 0)) if pd.notna(sc.get("n_ocr_words")) else 0
        kf_path = str(sc.get("keyframe_path", ""))

        # Load keyframe
        img_data = _load_image_base64(kf_path)
        if img_data:
            img_html = f'<img src="{img_data}" alt="Scene {sid} keyframe">'
        else:
            img_html = f'<div class="scene-placeholder">Scene {sid}<br>No keyframe</div>'

        # Keywords matched in this scene
        scene_kws = kw_align[kw_align["scene_id"] == sid] if not kw_align.empty and "scene_id" in kw_align.columns else pd.DataFrame()
        matched_kws = scene_kws[scene_kws["match_case"] != "F"] if not scene_kws.empty else pd.DataFrame()

        kw_info = ""
        if not matched_kws.empty:
            kw_items = []
            for _, kw in matched_kws.iterrows():
                kw_items.append(
                    f'<span class="case-badge case-{kw["match_case"]}">{kw["match_case"]}</span> '
                    f'{kw["keyword_text"]} ({_fmt_delta(kw.get("delta_t"))})'
                )
            kw_info = "<br>".join(kw_items)
        else:
            kw_info = '<span style="color:#95a5a6">No keyword matches in this scene</span>'

        cards.append(f"""
        <div class="card scene-card">
            {img_html}
            <div>
                <h3>Scene {sid} <span style="font-weight:400;color:#7f8c8d;font-size:0.85em">
                    ({_fmt_time(t0)} - {_fmt_time(t1)}, {dur:.1f}s)</span></h3>
                <p style="font-size:0.85em;margin:5px 0;">
                    <strong>OCR ({n_ocr} words):</strong>
                    <span style="color:#7f8c8d">{ocr_words[:200] if ocr_words != 'nan' else 'None'}</span>
                </p>
                <p style="font-size:0.85em;margin:5px 0;"><strong>Keyword matches:</strong></p>
                <div style="font-size:0.85em;">{kw_info}</div>
            </div>
        </div>
        """)

    return f"""
<h2 id="scenes">Scene Details</h2>
{"".join(cards)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 5: Recommendations
# ═══════════════════════════════════════════════════════════════════════════════

def _section_recommendations(data: dict) -> str:
    kw_scores = data["keyword_scores"]
    seg_scores = data["segment_scores"]
    results = data["results"]

    recs = []

    if kw_scores.empty:
        return '<h2 id="recommendations">Recommendations</h2><div class="card"><p>No data.</p></div>'

    # 1. Segments with high importance but 0% match rate
    if not seg_scores.empty:
        imp = data["importance"]
        for _, ss in seg_scores.iterrows():
            sid = ss["segment_id"]
            mr = float(ss.get("match_rate", 0))
            n_kw = int(ss.get("n_keywords", 0))
            if mr == 0 and n_kw > 0:
                imp_val = 3
                if not imp.empty and sid in imp["segment_id"].values:
                    imp_val = int(imp[imp["segment_id"] == sid].iloc[0]["importance"])
                if imp_val >= 4:
                    text = str(ss.get("text_preview", ""))
                    recs.append((
                        "high-priority",
                        "HIGH PRIORITY",
                        f'Segment {sid} ("{text}...") is pedagogically important (importance={imp_val}) '
                        f'but has <strong>zero visual support</strong> for {n_kw} keywords. '
                        f'Consider adding a diagram, slide, or annotation for this content.'
                    ))

    # 2. Case B with delta_t > 3s (late visuals)
    matched = kw_scores[kw_scores["match_case"].isin(["B", "C"])]
    for _, kw in matched.iterrows():
        dt = float(kw["delta_t"]) if pd.notna(kw.get("delta_t")) else 0
        if dt > 3.0:
            recs.append((
                "high-priority",
                "LATE VISUAL",
                f'"{kw["keyword_text"]}" (segment {kw["segment_id"]}): visual appears '
                f'<strong>{dt:.1f}s after</strong> narration. '
                f'Show the slide BEFORE or during the explanation, not after.'
            ))

    # 3. Case B with delta_t 1-3s (minor lag)
    for _, kw in matched.iterrows():
        dt = float(kw["delta_t"]) if pd.notna(kw.get("delta_t")) else 0
        if 1.0 < dt <= 3.0:
            recs.append((
                "medium-priority",
                "MINOR LAG",
                f'"{kw["keyword_text"]}" (segment {kw["segment_id"]}): visual appears '
                f'{dt:.1f}s after narration. Suboptimal but not severe.'
            ))

    # 4. Segments with no OCR nearby (abstract narration)
    f_reasons = results.get("keyword_level", {}).get("case_F_reasons", {})
    no_ocr = f_reasons.get("no_ocr_nearby", 0)
    vocab_mismatch = f_reasons.get("ocr_vocabulary_mismatch", 0)

    if no_ocr > 0:
        recs.append((
            "rec",
            "VISUAL COVERAGE",
            f'{no_ocr} keywords have <strong>no text on screen at all</strong> during narration. '
            f'These are likely talking-head segments. Consider adding visual aids.'
        ))

    if vocab_mismatch > 0:
        recs.append((
            "rec",
            "VOCABULARY MISMATCH",
            f'{vocab_mismatch} keywords use <strong>different terminology</strong> than what appears '
            f'on screen. Consider adding the spoken terms to slide text, or using consistent vocabulary.'
        ))

    # 5. Large visual lead time
    vlt = results.get("visual_lead_time", {})
    pct_over_10 = vlt.get("pct_over_10s", 0)
    if pct_over_10 > 20:
        recs.append((
            "rec",
            "PACING",
            f'{pct_over_10:.0f}% of matched visuals were on screen 10+ seconds before narration. '
            f'Consider restructuring slides to appear closer to when the narrator discusses them.'
        ))

    if not recs:
        recs.append(("rec", "NONE", "No specific improvements identified. Well done!"))

    rec_html = []
    for cls, label, text in recs:
        rec_html.append(f"""
        <div class="rec {cls}">
            <div class="rec-label">{label}</div>
            <p>{text}</p>
        </div>
        """)

    return f"""
<h2 id="recommendations">Recommendations for Video Improvement</h2>
{"".join(rec_html)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6b: Top 5 Fixes
# ═══════════════════════════════════════════════════════════════════════════════

def _section_top5_fixes(data: dict, output_dir: str) -> str:
    results = data["results"]
    fixes = results.get("top5_priority_fixes", [])

    if not fixes:
        return '<h2 id="top5">Top 5 Priority Fixes</h2><div class="card"><p>No fixes needed.</p></div>'

    cards = []
    for i, fix in enumerate(fixes):
        kw_text = fix.get("keyword_text", "")
        seg_id = fix.get("segment_id", 0)
        dt = fix.get("delta_t")
        s_temp = fix.get("S_temporal")
        priority = fix.get("priority", 0)

        # Find the segment text
        segments = data["segments"]
        seg_text = ""
        t_narr_val = None
        if not segments.empty:
            seg_row = segments[segments["segment_id"] == seg_id]
            if not seg_row.empty:
                seg_text = str(seg_row.iloc[0]["text"])[:100]

        # Find t_narr and t_vis from keyword_scores
        kw_scores = data["keyword_scores"]
        t_vis_val = None
        if not kw_scores.empty:
            kw_row = kw_scores[kw_scores["keyword_id"] == fix.get("keyword_id")]
            if not kw_row.empty:
                t_narr_val = kw_row.iloc[0].get("t_narr")
                t_vis_val = kw_row.iloc[0].get("t_vis")

        # Load frames
        frame_narr_html = '<div class="scene-placeholder">Frame at t_narr</div>'
        frame_vis_html = '<div class="scene-placeholder">Frame at t_vis</div>'

        if t_narr_val is not None and pd.notna(t_narr_val):
            narr_path = _find_frame_at_time(output_dir, float(t_narr_val))
            img = _load_image_base64(narr_path)
            if img:
                frame_narr_html = f'<img src="{img}" alt="Frame at t_narr">'

        if t_vis_val is not None and pd.notna(t_vis_val):
            vis_path = _find_frame_at_time(output_dir, float(t_vis_val))
            img = _load_image_base64(vis_path)
            if img:
                frame_vis_html = f'<img src="{img}" alt="Frame at t_vis">'

        zone = _zone_for_delta(dt) if dt else "No Match"
        zone_cls = zone.lower().replace(" ", "")

        dt_display = f"+{dt:.1f}s" if dt and dt > 0 else _fmt_delta(dt)

        cards.append(f"""
        <div class="card">
            <h3>#{i+1}: "{kw_text}" <span class="zone zone-{zone_cls}">{zone}</span></h3>
            <p style="font-size:0.85em;color:#7f8c8d;">Segment {seg_id} | delta_t = {dt_display} |
               Score = {f'{s_temp:.1f}' if s_temp else 'N/A'} | Priority = {priority:.1f}</p>
            <p style="font-size:0.85em;margin:8px 0;">"{seg_text}..."</p>
            <div class="fix-card" style="margin-top:10px;">
                <div class="fix-frame">
                    <p style="font-size:0.8em;font-weight:600;">At t_narr ({_fmt_time(t_narr_val)})</p>
                    <p style="font-size:0.75em;color:#7f8c8d;">What the student sees when concept is spoken</p>
                    {frame_narr_html}
                </div>
                <div class="fix-frame">
                    <p style="font-size:0.8em;font-weight:600;">At t_vis ({_fmt_time(t_vis_val)})</p>
                    <p style="font-size:0.75em;color:#7f8c8d;">When the visual actually appears</p>
                    {frame_vis_html}
                </div>
            </div>
            <div class="rec medium-priority" style="margin-top:10px;">
                <div class="rec-label">RECOMMENDATION</div>
                <p>Show the visual for "{kw_text}" <strong>before or during</strong> the narration
                   (currently appears {dt_display} {'late' if dt and dt > 0 else 'N/A'}).
                   Consider preparing the slide transition to coincide with the explanation.</p>
            </div>
        </div>
        """)

    return f"""
<h2 id="top5">Top 5 Priority Fixes</h2>
<p style="color:#7f8c8d;margin-bottom:15px;">Ranked by priority = (100 - S_temporal) x importance_weight x alpha</p>
{"".join(cards)}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6d: Delta_t Histogram
# ═══════════════════════════════════════════════════════════════════════════════

def _section_delta_histogram(data: dict) -> str:
    kw_scores = data["keyword_scores"]
    if kw_scores.empty:
        return '<h2 id="histogram">Delta_t Distribution</h2><div class="card"><p>No data.</p></div>'

    matched = kw_scores[kw_scores["match_case"] != "F"]
    if matched.empty:
        return '<h2 id="histogram">Delta_t Distribution</h2><div class="card"><p>No matched keywords.</p></div>'

    deltas = matched["delta_t"].dropna().values

    # Define bins
    bins = [
        ("<-10s (visual far ahead)", lambda d: d < -10, "#2c3e50"),
        ("-10s to -5s", lambda d: -10 <= d < -5, "#34495e"),
        ("-5s to -2s", lambda d: -5 <= d < -2, "#27ae60"),
        ("-2s to -1s", lambda d: -2 <= d < -1, "#2ecc71"),
        ("-1s to 0 (simultaneous)", lambda d: -1 <= d <= 0, "#27ae60"),
        ("0 to +1s (near-simultaneous)", lambda d: 0 < d <= 1, "#f1c40f"),
        ("+1s to +3s (suboptimal)", lambda d: 1 < d <= 3, "#f39c12"),
        ("+3s to +5s (disruptive)", lambda d: 3 < d <= 5, "#e67e22"),
        (">+5s (unacceptable)", lambda d: d > 5, "#e74c3c"),
    ]

    max_count = max(1, max(int(np.sum([fn(d) for d in deltas])) for _, fn, _ in bins))

    bars = []
    for label, fn, color in bins:
        count = int(np.sum([fn(d) for d in deltas]))
        width = max(int(300 * count / max_count), 0)
        bars.append(f"""
        <div class="hist-bar">
            <div class="bar-label">{label}</div>
            <div class="bar" style="width:{width}px;background:{color};">
                {count if count > 0 else ''}</div>
        </div>
        """)

    # Stats
    mean_dt = float(np.mean(deltas))
    median_dt = float(np.median(deltas))
    n_neg = int(np.sum(deltas <= 0))
    n_pos = int(np.sum(deltas > 0))

    return f"""
<h2 id="histogram">Delta_t Distribution</h2>
<div class="card">
    <p style="margin-bottom:15px;font-size:0.85em;color:#7f8c8d;">
        delta_t = t_vis - t_narr. Negative = visual was already on screen. Positive = visual appeared late.
    </p>
    {"".join(bars)}
    <div style="margin-top:15px;font-size:0.85em;display:flex;gap:20px;">
        <span>Mean: {mean_dt:+.1f}s</span>
        <span>Median: {median_dt:+.1f}s</span>
        <span>Visual ahead: {n_neg} ({100*n_neg/max(len(deltas),1):.0f}%)</span>
        <span>Visual late: {n_pos} ({100*n_pos/max(len(deltas),1):.0f}%)</span>
    </div>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6e: OCR Evidence
# ═══════════════════════════════════════════════════════════════════════════════

def _section_ocr_evidence(data: dict) -> str:
    kw_scores = data["keyword_scores"]
    if kw_scores.empty:
        return '<h2 id="ocr-evidence">OCR Evidence Trail</h2><div class="card"><p>No data.</p></div>'

    matched = kw_scores[kw_scores["match_case"] != "F"].head(30)  # limit to 30 for readability

    rows = []
    for _, kw in matched.iterrows():
        case = str(kw.get("match_case", ""))
        method = str(kw.get("match_method", ""))
        alpha = float(kw.get("alpha", 0))
        confidence = str(kw.get("confidence", ""))
        dt = kw.get("delta_t")
        s_val = kw.get("S_temporal")
        zone = str(kw.get("zone", ""))
        zone_cls = zone.lower().replace(" ", "")

        rows.append(f"""
        <tr>
            <td>{int(kw.get('keyword_id', 0))}</td>
            <td><strong>{kw.get('keyword_text', '')}</strong></td>
            <td>{int(kw.get('segment_id', 0))}</td>
            <td><span class="case-badge case-{case}">{case}</span></td>
            <td>{method}</td>
            <td>{_fmt_delta(dt)}</td>
            <td>{f'{s_val:.0f}' if pd.notna(s_val) else 'N/A'}</td>
            <td><span class="zone zone-{zone_cls}">{zone}</span></td>
            <td>{alpha:.1f}</td>
            <td>{confidence}</td>
        </tr>
        """)

    return f"""
<h2 id="ocr-evidence">OCR Evidence Trail</h2>
<div class="card" style="overflow-x:auto;">
    <p style="margin-bottom:10px;font-size:0.85em;color:#7f8c8d;">
        Showing matched keywords with their matching method, confidence, and evidence source.
        match_method indicates how the keyword was found (full phrase match vs partial word match).
    </p>
    <table>
        <thead>
            <tr>
                <th>ID</th><th>Keyword</th><th>Seg</th><th>Case</th><th>Method</th>
                <th>delta_t</th><th>Score</th><th>Zone</th><th>Alpha</th><th>Confidence</th>
            </tr>
        </thead>
        <tbody>
            {"".join(rows)}
        </tbody>
    </table>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Section 6c: Methodology
# ═══════════════════════════════════════════════════════════════════════════════

def _section_methodology() -> str:
    return """
<h2 id="methodology" class="collapsible" onclick="this.classList.toggle('active');
    this.nextElementSibling.classList.toggle('show');">Methodology</h2>
<div class="collapsible-content">
<div class="card">

<h3>What Temporal Contiguity (TC) Measures</h3>
<p>TC measures whether visual content is displayed <strong>simultaneously</strong> with the
corresponding narration. Based on Mayer & Moreno's Temporal Contiguity Principle:
"Students learn better when corresponding words and pictures are presented simultaneously
rather than successively."</p>

<h3 style="margin-top:15px;">What TC Does NOT Measure</h3>
<ul style="margin:8px 0 8px 20px; font-size:0.9em;">
    <li>Visual quality or clarity of slides</li>
    <li>Pedagogical effectiveness of the content itself</li>
    <li>Whether abstract narration SHOULD have visual support</li>
    <li>Student learning outcomes (TC is a design principle, not a learning measure)</li>
</ul>

<h3 style="margin-top:15px;">Why Coverage Rate != TC Score</h3>
<p>The <strong>TC Score</strong> (e.g., 92.35) measures temporal alignment quality among concepts that
DO have visual correlates. The <strong>Coverage Rate</strong> (e.g., 32.6%) measures what fraction of
narrated concepts have any visual support at all. These are different metrics:</p>
<ul style="margin:8px 0 8px 20px; font-size:0.9em;">
    <li>A video where the narrator says "imagine a foggy landscape" has no visual that SHOULD exist for this metaphor.
        Excluding it from TC is correct.</li>
    <li>A video where the narrator says "this algorithm" while a diagram is on screen IS a TC measurement.</li>
    <li>The Coverage-Adjusted score (TC x Coverage) is a composite metric, not a standard TC metric.</li>
</ul>

<h3 style="margin-top:15px;">Alignment Cases Explained</h3>
<table style="font-size:0.85em;">
    <tr><td><span class="case-badge case-A">A</span></td>
        <td><strong>Visual on screen:</strong> The keyword text is found in the current scene's OCR.
            The visual was already present when the narrator spoke. Score = 100 (no penalty).</td></tr>
    <tr><td><span class="case-badge case-D">D</span></td>
        <td><strong>Progressive reveal:</strong> The keyword appears mid-scene (not from the start).
            Common for animated slides. Still present at narration time. Score = 100.</td></tr>
    <tr><td><span class="case-badge case-B">B</span></td>
        <td><strong>Visual appears late:</strong> The keyword is not on screen at narration time but
            appears within +10 seconds. Score penalized by Gaussian/piecewise formula.</td></tr>
    <tr><td><span class="case-badge case-C">C</span></td>
        <td><strong>Visual expired:</strong> The keyword was shown earlier but has disappeared.
            Maximum penalty (delta_t capped at 10s).</td></tr>
    <tr><td><span class="case-badge case-G">G</span></td>
        <td><strong>SigLIP match:</strong> No text match, but visual similarity detected (animations,
            diagrams without labels). Reduced confidence (alpha &lt; 0.5).</td></tr>
    <tr><td><span class="case-badge case-F">F</span></td>
        <td><strong>No visual correlate:</strong> No matching visual found. Excluded from TC scoring.
            Sub-reasons: no_ocr_nearby (talking head), ocr_vocabulary_mismatch (different terms),
            low_groundability (abstract concept).</td></tr>
</table>

<h3 style="margin-top:15px;">Asymmetric Scoring</h3>
<p>delta_t = t_vis - t_narr. <strong>Only positive delta_t is penalized</strong> (visual appears AFTER
narration). Negative delta_t means the visual was already showing = no penalty. This is correct per TC
theory: having the visual prepared ahead of narration is good instructional design.</p>

<h3 style="margin-top:15px;">Confidence Tiers</h3>
<ul style="margin:8px 0 8px 20px; font-size:0.9em;">
    <li><strong>HIGH (alpha = 1.0):</strong> Full OCR phrase match in the scene</li>
    <li><strong>MEDIUM (alpha = 0.8):</strong> Partial word-level OCR match (at least one content word found)</li>
    <li><strong>LOW (alpha &lt; 0.5):</strong> SigLIP visual similarity match (no text evidence)</li>
</ul>

</div>
</div>

<script>
// Enable all collapsible sections
document.querySelectorAll('.collapsible').forEach(el => {
    el.style.cursor = 'pointer';
});
</script>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# HTML Footer
# ═══════════════════════════════════════════════════════════════════════════════

def _html_footer() -> str:
    return """
<div style="text-align:center; margin-top:40px; padding:20px; color:#95a5a6; font-size:0.8em;">
    <p>Generated by Pipeline v2.2 — Temporal Contiguity Analysis</p>
    <p>Transcript-First Architecture | Asymmetric Scoring | Scene OCR Union</p>
</div>
</div><!-- /container -->
</body>
</html>
"""
