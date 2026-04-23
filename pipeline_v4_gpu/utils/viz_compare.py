"""Comparative Dashboard — Multi-Video TC Analysis [v2.2].

Generates a self-contained HTML report comparing temporal contiguity
metrics across multiple videos side-by-side.

Usage:
    from pipeline_v2_2_gpu.utils.viz_compare import generate_comparison
    generate_comparison("outputs_v2_2", ["A0", "A1", "A3", "A5"])

    Or from CLI:
    python -m pipeline_v2_2_gpu.utils.viz_compare outputs_v2_2 A0 A1 A3 A5
"""

import json
import logging
import os
import sys

logger = logging.getLogger(__name__)

# ─── Colors ───────────────────────────────────────────────────────────────────
COLORS = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]
ZONE_COLORS = {
    "Optimal": "#27ae60",
    "Suboptimal": "#f39c12",
    "Disruptive": "#e67e22",
    "Unacceptable": "#e74c3c",
}
GRADE_COLORS = {
    "Excellent": "#27ae60",
    "Good": "#2ecc71",
    "Acceptable": "#f39c12",
    "Poor": "#e67e22",
    "Unacceptable": "#e74c3c",
}


def _load_results(output_root: str, video_id: str) -> dict:
    """Load results.json for a video."""
    path = os.path.join(output_root, video_id, "results.json")
    if not os.path.exists(path):
        logger.warning("results.json not found for %s", video_id)
        return None
    with open(path, "r") as f:
        return json.load(f)


def _safe_get(d, *keys, default=None):
    """Safely traverse nested dict."""
    for k in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(k, default)
    return d


def generate_comparison(output_root: str, video_ids: list, out_path: str = None) -> str:
    """Generate comparative HTML dashboard. Returns path to output file."""

    # Load all results
    results = {}
    for vid in video_ids:
        r = _load_results(output_root, vid)
        if r:
            results[vid] = r
        else:
            logger.warning("Skipping %s — no results.json", vid)

    if len(results) < 2:
        raise ValueError(f"Need at least 2 videos with results, got {len(results)}")

    vids = list(results.keys())
    n = len(vids)

    if out_path is None:
        out_path = os.path.join(output_root, f"comparison_{'_'.join(vids)}.html")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>TC Comparison — {', '.join(vids)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #f5f6fa; color: #2c3e50; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
h1 {{ font-size: 1.8em; margin-bottom: 5px; }}
h2 {{ font-size: 1.4em; margin: 30px 0 15px; padding-bottom: 8px;
      border-bottom: 2px solid #3498db; color: #2c3e50; }}
h3 {{ font-size: 1.1em; margin: 15px 0 8px; color: #34495e; }}
.subtitle {{ color: #7f8c8d; font-size: 0.9em; margin-bottom: 20px; }}

/* Navigation */
.nav {{ position: sticky; top: 0; background: #2c3e50; padding: 10px 20px;
        z-index: 100; display: flex; gap: 15px; flex-wrap: wrap; }}
.nav a {{ color: #ecf0f1; text-decoration: none; font-size: 0.85em; padding: 4px 10px;
          border-radius: 4px; transition: background 0.2s; }}
.nav a:hover {{ background: #34495e; }}

/* Cards */
.card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 15px;
         box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}

/* Table */
table {{ width: 100%; border-collapse: collapse; margin: 10px 0; }}
th {{ background: #2c3e50; color: white; padding: 10px 12px; text-align: left; font-size: 0.85em; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #ecf0f1; font-size: 0.85em; }}
tr:hover {{ background: #f8f9fa; }}
.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

/* Grade badges */
.grade {{ display: inline-block; padding: 3px 14px; border-radius: 16px; font-weight: 700;
          color: white; font-size: 0.8em; }}

/* Zone badges */
.zone {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.75em;
         font-weight: 600; color: white; }}
.zone-optimal {{ background: #27ae60; }}
.zone-suboptimal {{ background: #f39c12; }}
.zone-disruptive {{ background: #e67e22; }}
.zone-unacceptable {{ background: #e74c3c; }}

/* Bar chart */
.bar-chart {{ display: flex; flex-direction: column; gap: 8px; margin: 10px 0; }}
.bar-row {{ display: flex; align-items: center; gap: 10px; }}
.bar-label {{ width: 50px; font-size: 0.85em; font-weight: 600; text-align: right; }}
.bar-track {{ flex: 1; background: #ecf0f1; border-radius: 4px; height: 28px; position: relative; }}
.bar-fill {{ height: 100%; border-radius: 4px; display: flex; align-items: center;
             padding-left: 8px; color: white; font-size: 0.75em; font-weight: 600;
             transition: width 0.5s; min-width: 30px; }}
.bar-value {{ position: absolute; right: 8px; top: 4px; font-size: 0.8em; font-weight: 600; }}

/* Stacked bar */
.stacked-bar {{ display: flex; height: 28px; border-radius: 4px; overflow: hidden; }}
.stacked-segment {{ display: flex; align-items: center; justify-content: center;
                     color: white; font-size: 0.7em; font-weight: 600; min-width: 1px; }}

/* Radar-like summary */
.score-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px; }}
.score-card {{ background: white; border-radius: 8px; padding: 20px; text-align: center;
               box-shadow: 0 2px 8px rgba(0,0,0,0.08); border-top: 4px solid #3498db; }}
.score-card .video-name {{ font-size: 1.1em; font-weight: 700; margin-bottom: 10px; }}
.score-card .big-score {{ font-size: 2.5em; font-weight: 700; }}
.score-card .details {{ font-size: 0.8em; color: #7f8c8d; margin-top: 8px; line-height: 1.8; }}

/* Insight box */
.insight {{ padding: 15px; background: #eaf2f8; border-left: 4px solid #3498db;
            border-radius: 0 4px 4px 0; margin: 10px 0; font-size: 0.9em; }}
.insight.positive {{ border-left-color: #27ae60; background: #eafaf1; }}
.insight.negative {{ border-left-color: #e74c3c; background: #fdedec; }}
.insight.neutral {{ border-left-color: #f39c12; background: #fef9e7; }}
.insight strong {{ color: #2c3e50; }}

/* Legend */
.legend {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 10px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 0.8em; }}
.legend-swatch {{ width: 14px; height: 14px; border-radius: 3px; }}

/* Footer */
.footer {{ text-align: center; color: #95a5a6; font-size: 0.8em; padding: 30px 0 10px; }}
</style>
</head>
<body>

<div class="nav">
    <a href="#overview">Overview</a>
    <a href="#scoreboard">Scoreboard</a>
    <a href="#zones">Zone Distribution</a>
    <a href="#coverage">Coverage</a>
    <a href="#cases">Match Cases</a>
    <a href="#timing">Timing Analysis</a>
    <a href="#insights">Insights</a>
</div>

<div class="container">
<h1>Temporal Contiguity — Comparative Analysis</h1>
<p class="subtitle">Pipeline v2.2 | Videos: {', '.join(vids)} | Comparing {n} videos</p>

"""
    # ─── Section 1: Score Overview Cards ──────────────────────────────────────
    html += '<h2 id="overview">Score Overview</h2>\n<div class="score-grid">\n'

    for i, vid in enumerate(vids):
        r = results[vid]
        score = r.get("overall_score", 0) or 0
        grade = r.get("overall_grade", "N/A")
        grade_color = GRADE_COLORS.get(grade, "#95a5a6")
        score_pw = r.get("overall_score_piecewise", 0) or 0
        kl = r.get("keyword_level", {})
        n_matched = kl.get("n_matched", 0)
        n_total = kl.get("n_total", 0)
        cov = r.get("coverage", {})
        cov_rate = (cov.get("coverage_rate", 0) or 0) * 100
        cov_adj = cov.get("tc_score_coverage_adjusted", 0) or 0
        pct_opt = kl.get("pct_Optimal", 0) or 0

        html += f"""
    <div class="score-card" style="border-top-color: {COLORS[i % len(COLORS)]};">
        <div class="video-name">{vid}</div>
        <div class="big-score" style="color: {grade_color};">{score:.1f}</div>
        <div><span class="grade" style="background: {grade_color};">{grade}</span></div>
        <div class="details">
            Piecewise: {score_pw:.1f}<br>
            Keywords: {n_matched}/{n_total} matched<br>
            Coverage: {cov_rate:.1f}%<br>
            Coverage-adjusted: {cov_adj:.1f}<br>
            Optimal zone: {pct_opt:.1f}%
        </div>
    </div>
"""

    html += '</div>\n'

    # ─── Section 2: Scoreboard Table ──────────────────────────────────────────
    html += '<h2 id="scoreboard">Detailed Scoreboard</h2>\n<div class="card">\n<table>\n'
    html += '<tr><th>Metric</th>'
    for vid in vids:
        html += f'<th class="num">{vid}</th>'
    html += '</tr>\n'

    metrics = [
        ("Overall Score (Gaussian)", lambda r: r.get("overall_score", "")),
        ("Overall Score (Piecewise)", lambda r: r.get("overall_score_piecewise", "")),
        ("Grade", lambda r: r.get("overall_grade", "")),
        ("Total Keywords", lambda r: _safe_get(r, "keyword_level", "n_total", default="")),
        ("Matched Keywords", lambda r: _safe_get(r, "keyword_level", "n_matched", default="")),
        ("Match Rate", lambda r: f'{_safe_get(r, "keyword_level", "match_rate", default=0)*100:.1f}%'),
        ("Coverage Rate", lambda r: f'{(_safe_get(r, "coverage", "coverage_rate", default=0) or 0)*100:.1f}%'),
        ("Coverage-Adjusted Score", lambda r: _safe_get(r, "coverage", "tc_score_coverage_adjusted", default="")),
        ("Mean delta_t (s)", lambda r: f'{_safe_get(r, "keyword_level", "mean_delta_t", default=0):.2f}'),
        ("Median delta_t (s)", lambda r: f'{_safe_get(r, "keyword_level", "median_delta_t", default=0):.2f}'),
        ("Std delta_t (s)", lambda r: f'{_safe_get(r, "keyword_level", "std_delta_t", default=0):.2f}'),
        ("% Negative delta_t", lambda r: f'{_safe_get(r, "keyword_level", "pct_negative_delta_t", default=0):.1f}%'),
        ("% Optimal Zone", lambda r: f'{_safe_get(r, "keyword_level", "pct_Optimal", default=0):.1f}%'),
        ("% Suboptimal Zone", lambda r: f'{_safe_get(r, "keyword_level", "pct_Suboptimal", default=0):.1f}%'),
        ("% Disruptive Zone", lambda r: f'{_safe_get(r, "keyword_level", "pct_Disruptive", default=0):.1f}%'),
        ("% Unacceptable Zone", lambda r: f'{_safe_get(r, "keyword_level", "pct_Unacceptable", default=0):.1f}%'),
        ("Mean Alpha (confidence)", lambda r: f'{_safe_get(r, "keyword_level", "mean_alpha", default=0):.3f}'),
        ("Segments Total", lambda r: _safe_get(r, "segment_level", "n_total", default="")),
        ("Segments with Matches", lambda r: _safe_get(r, "segment_level", "n_with_matches", default="")),
        ("Positive delta_t (n)", lambda r: _safe_get(r, "positive_delta_t_only", "n", default="")),
        ("Positive delta_t mean (s)", lambda r: _safe_get(r, "positive_delta_t_only", "mean_delta_t", default="")),
        ("Positive delta_t mean S", lambda r: _safe_get(r, "positive_delta_t_only", "mean_S_temporal", default="")),
        ("Visual Lead Time mean (s)", lambda r: _safe_get(r, "visual_lead_time", "mean_lead_s", default="")),
        ("Visual Lead Time median (s)", lambda r: _safe_get(r, "visual_lead_time", "median_lead_s", default="")),
        ("% Lead within 2s", lambda r: f'{_safe_get(r, "visual_lead_time", "pct_within_2s", default=0):.1f}%'),
        ("% Lead over 10s", lambda r: f'{_safe_get(r, "visual_lead_time", "pct_over_10s", default=0):.1f}%'),
    ]

    for label, fn in metrics:
        html += f'<tr><td><strong>{label}</strong></td>'
        vals = []
        for vid in vids:
            v = fn(results[vid])
            vals.append(v)
            html += f'<td class="num">{v}</td>'
        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # ─── Section 3: Score Bar Chart ───────────────────────────────────────────
    html += '<h2 id="zones">Score Comparison</h2>\n<div class="card">\n'
    html += '<h3>Overall TC Score (Gaussian)</h3>\n<div class="bar-chart">\n'

    for i, vid in enumerate(vids):
        score = results[vid].get("overall_score", 0) or 0
        color = COLORS[i % len(COLORS)]
        html += f"""
    <div class="bar-row">
        <div class="bar-label">{vid}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width: {score}%; background: {color};">{score:.1f}</div>
        </div>
    </div>
"""

    html += '</div>\n'

    # Zone distribution stacked bars
    html += '<h3 style="margin-top:25px;">Zone Distribution</h3>\n'
    html += """<div class="legend">
        <div class="legend-item"><div class="legend-swatch" style="background:#27ae60;"></div>Optimal</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#f39c12;"></div>Suboptimal</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#e67e22;"></div>Disruptive</div>
        <div class="legend-item"><div class="legend-swatch" style="background:#e74c3c;"></div>Unacceptable</div>
    </div>\n"""
    html += '<div class="bar-chart">\n'

    for vid in vids:
        kl = results[vid].get("keyword_level", {})
        popt = kl.get("pct_Optimal", 0) or 0
        psub = kl.get("pct_Suboptimal", 0) or 0
        pdis = kl.get("pct_Disruptive", 0) or 0
        puna = kl.get("pct_Unacceptable", 0) or 0

        html += f"""
    <div class="bar-row">
        <div class="bar-label">{vid}</div>
        <div class="bar-track">
            <div class="stacked-bar" style="width:100%;">
                <div class="stacked-segment" style="width:{popt}%;background:#27ae60;">{popt:.0f}%</div>
                <div class="stacked-segment" style="width:{psub}%;background:#f39c12;">{psub:.0f}%</div>
                <div class="stacked-segment" style="width:{pdis}%;background:#e67e22;">{pdis:.0f}%</div>
                <div class="stacked-segment" style="width:{puna}%;background:#e74c3c;">{puna:.0f}%</div>
            </div>
        </div>
    </div>
"""

    html += '</div>\n</div>\n'

    # ─── Section 4: Coverage Comparison ───────────────────────────────────────
    html += '<h2 id="coverage">Coverage Analysis</h2>\n<div class="card">\n'
    html += '<h3>Coverage Rate vs TC Score</h3>\n'
    html += '<p style="color:#7f8c8d;font-size:0.85em;margin-bottom:10px;">'
    html += 'Coverage = fraction of groundable keywords with a visual match. '
    html += 'TC Score = alignment quality among matched keywords.</p>\n'

    html += '<div class="bar-chart">\n'
    for i, vid in enumerate(vids):
        cov = (results[vid].get("coverage", {}).get("coverage_rate", 0) or 0) * 100
        color = COLORS[i % len(COLORS)]
        html += f"""
    <div class="bar-row">
        <div class="bar-label">{vid}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width: {cov}%; background: {color};">{cov:.1f}%</div>
        </div>
    </div>
"""
    html += '</div>\n'

    html += '<h3 style="margin-top:20px;">Coverage-Adjusted Score</h3>\n'
    html += '<div class="bar-chart">\n'
    for i, vid in enumerate(vids):
        cov_adj = results[vid].get("coverage", {}).get("tc_score_coverage_adjusted", 0) or 0
        color = COLORS[i % len(COLORS)]
        html += f"""
    <div class="bar-row">
        <div class="bar-label">{vid}</div>
        <div class="bar-track">
            <div class="bar-fill" style="width: {cov_adj}%; background: {color};">{cov_adj:.1f}</div>
        </div>
    </div>
"""
    html += '</div>\n</div>\n'

    # ─── Section 5: Case Distribution ─────────────────────────────────────────
    html += '<h2 id="cases">Match Case Distribution</h2>\n<div class="card">\n'
    html += '<table>\n<tr><th>Video</th>'
    for case in ["A", "D", "B", "C", "G"]:
        html += f'<th class="num">Case {case}</th>'
    html += '<th class="num">Case F (unmatched)</th><th class="num">Total</th></tr>\n'

    for vid in vids:
        kl = results[vid].get("keyword_level", {})
        cd = kl.get("case_distribution", {})
        n_unmatched = kl.get("n_unmatched", 0)
        n_total = kl.get("n_total", 0)
        html += f'<tr><td><strong>{vid}</strong></td>'
        for case in ["A", "D", "B", "C", "G"]:
            v = cd.get(case, 0)
            html += f'<td class="num">{v}</td>'
        html += f'<td class="num">{n_unmatched}</td>'
        html += f'<td class="num">{n_total}</td>'
        html += '</tr>\n'

    html += '</table>\n'

    # Case F breakdown
    html += '<h3 style="margin-top:15px;">Case F Breakdown (Why No Match)</h3>\n'
    html += '<table>\n<tr><th>Video</th><th class="num">OCR Vocab Mismatch</th>'
    html += '<th class="num">No OCR Nearby</th><th class="num">Low Groundability</th></tr>\n'

    for vid in vids:
        reasons = _safe_get(results[vid], "keyword_level", "case_F_reasons", default={})
        html += f'<tr><td><strong>{vid}</strong></td>'
        html += f'<td class="num">{reasons.get("ocr_vocabulary_mismatch", 0)}</td>'
        html += f'<td class="num">{reasons.get("no_ocr_nearby", 0)}</td>'
        html += f'<td class="num">{reasons.get("low_groundability", 0)}</td>'
        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # ─── Section 6: Timing Analysis ───────────────────────────────────────────
    html += '<h2 id="timing">Timing Analysis</h2>\n<div class="card">\n'

    # Delta_t histograms side by side
    html += '<h3>Delta_t Distribution Comparison</h3>\n'
    html += '<table>\n<tr><th>Bin</th>'
    for vid in vids:
        html += f'<th class="num">{vid}</th>'
    html += '</tr>\n'

    # Collect all bin keys
    all_bins = set()
    for vid in vids:
        diag_path = os.path.join(output_root, vid, "diagnostics", "stage7_scoring.json")
        if os.path.exists(diag_path):
            with open(diag_path) as f:
                diag = json.load(f)
            for k in diag.get("delta_t_histogram", {}):
                all_bins.add(k)

    # Sort bins in a sensible order
    bin_order = [
        "<-10s (visual far ahead)", "-10s to -5s", "-5s to -2s", "-2s to -1s",
        "-1s to 0 (near-simultaneous)", "0 to 1s (near-simultaneous)",
        "1-3s (suboptimal)", "3-5s (disruptive)", ">5s (unacceptable)",
        # Fallback older bin names
        "\u22640 (visual ahead)", "0-1s", "1-3s", "3-5s", ">5s",
    ]
    sorted_bins = [b for b in bin_order if b in all_bins]
    # Add any remaining bins not in our predefined order
    for b in sorted(all_bins):
        if b not in sorted_bins:
            sorted_bins.append(b)

    for b in sorted_bins:
        html += f'<tr><td>{b}</td>'
        for vid in vids:
            diag_path = os.path.join(output_root, vid, "diagnostics", "stage7_scoring.json")
            val = 0
            if os.path.exists(diag_path):
                with open(diag_path) as f:
                    diag = json.load(f)
                val = diag.get("delta_t_histogram", {}).get(b, 0)
            html += f'<td class="num">{val}</td>'
        html += '</tr>\n'

    html += '</table>\n'

    # Visual lead time comparison
    html += '<h3 style="margin-top:20px;">Visual Lead Time (Case A/D)</h3>\n'
    html += '<table>\n<tr><th>Metric</th>'
    for vid in vids:
        html += f'<th class="num">{vid}</th>'
    html += '</tr>\n'

    lead_metrics = [
        ("N visual ahead", lambda r: _safe_get(r, "visual_lead_time", "n_visual_ahead", default=0)),
        ("Mean lead (s)", lambda r: _safe_get(r, "visual_lead_time", "mean_lead_s", default="")),
        ("Median lead (s)", lambda r: _safe_get(r, "visual_lead_time", "median_lead_s", default="")),
        ("Max lead (s)", lambda r: _safe_get(r, "visual_lead_time", "max_lead_s", default="")),
        ("% within 2s", lambda r: f'{_safe_get(r, "visual_lead_time", "pct_within_2s", default=0)}%'),
        ("% within 5s", lambda r: f'{_safe_get(r, "visual_lead_time", "pct_within_5s", default=0)}%'),
        ("% over 10s", lambda r: f'{_safe_get(r, "visual_lead_time", "pct_over_10s", default=0)}%'),
    ]

    for label, fn in lead_metrics:
        html += f'<tr><td><strong>{label}</strong></td>'
        for vid in vids:
            html += f'<td class="num">{fn(results[vid])}</td>'
        html += '</tr>\n'

    html += '</table>\n</div>\n'

    # ─── Section 7: Insights ──────────────────────────────────────────────────
    html += '<h2 id="insights">Key Insights</h2>\n'
    html += _generate_insights(vids, results)

    # ─── Footer ───────────────────────────────────────────────────────────────
    html += """
<div class="footer">
    Pipeline v2.2 — Temporal Contiguity Comparative Analysis<br>
    Transcript-first architecture | Asymmetric scoring (visual-before = no penalty)
</div>
</div>
</body>
</html>
"""

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    logger.info("Comparison dashboard saved: %s", out_path)
    return out_path


def _generate_insights(vids: list, results: dict) -> str:
    """Auto-generate comparative insights based on data patterns."""
    html = ""

    # 1. Best/worst overall score
    scores = {v: results[v].get("overall_score", 0) or 0 for v in vids}
    best_v = max(scores, key=scores.get)
    worst_v = min(scores, key=scores.get)
    spread = scores[best_v] - scores[worst_v]

    if spread < 5:
        html += f"""<div class="insight neutral">
            <strong>Similar TC scores across videos.</strong>
            All videos score within {spread:.1f} points ({worst_v}: {scores[worst_v]:.1f} to {best_v}: {scores[best_v]:.1f}).
            The instructional design is consistent across these videos in terms of visual-narration synchronization.
        </div>\n"""
    else:
        html += f"""<div class="insight {'positive' if spread > 20 else 'neutral'}">
            <strong>Score spread: {spread:.1f} points.</strong>
            Best: <strong>{best_v}</strong> ({scores[best_v]:.1f}) | Worst: <strong>{worst_v}</strong> ({scores[worst_v]:.1f}).
            {'Significant variation — the lower-scoring videos may benefit from timing adjustments.' if spread > 20 else 'Moderate variation in TC quality.'}
        </div>\n"""

    # 2. Coverage comparison
    coverages = {}
    for v in vids:
        cr = _safe_get(results[v], "coverage", "coverage_rate", default=0) or 0
        coverages[v] = cr * 100

    best_cov_v = max(coverages, key=coverages.get)
    worst_cov_v = min(coverages, key=coverages.get)
    cov_spread = coverages[best_cov_v] - coverages[worst_cov_v]

    if cov_spread > 10:
        html += f"""<div class="insight neutral">
            <strong>Coverage varies significantly ({cov_spread:.1f}pp spread).</strong>
            <strong>{best_cov_v}</strong> matches {coverages[best_cov_v]:.1f}% of keywords visually, while
            <strong>{worst_cov_v}</strong> only matches {coverages[worst_cov_v]:.1f}%.
            Lower coverage often indicates more talking-head sections or abstract narration without visual support.
        </div>\n"""

    # 3. Case A dominance
    case_a_pcts = {}
    for v in vids:
        cd = _safe_get(results[v], "keyword_level", "case_distribution", default={})
        n_matched = _safe_get(results[v], "keyword_level", "n_matched", default=1) or 1
        case_a_pcts[v] = (cd.get("A", 0) / n_matched * 100) if n_matched > 0 else 0

    high_a = {v: p for v, p in case_a_pcts.items() if p > 60}
    low_a = {v: p for v, p in case_a_pcts.items() if p < 40}

    if high_a:
        vlist = ", ".join(f"<strong>{v}</strong> ({p:.0f}%)" for v, p in high_a.items())
        html += f"""<div class="insight positive">
            <strong>Strong visual-present alignment (Case A dominance).</strong>
            {vlist} have most keywords already visible on screen when narrated.
            This is the ideal TC pattern — visuals prepared before narration.
        </div>\n"""

    if low_a:
        vlist = ", ".join(f"<strong>{v}</strong> ({p:.0f}%)" for v, p in low_a.items())
        html += f"""<div class="insight negative">
            <strong>Low Case A rates: {vlist}.</strong>
            Keywords are often not on screen when spoken. This may indicate either
            late slide transitions or narration covering concepts not shown in text form.
        </div>\n"""

    # 4. Positive delta_t (true TC violations) comparison
    pos_dt = {}
    for v in vids:
        pd_info = results[v].get("positive_delta_t_only", {})
        pos_dt[v] = pd_info.get("n", 0) or 0

    total_matched = {v: _safe_get(results[v], "keyword_level", "n_matched", default=0) or 0 for v in vids}
    pos_pcts = {v: (pos_dt[v] / total_matched[v] * 100) if total_matched[v] > 0 else 0 for v in vids}

    worst_pos = max(pos_pcts, key=pos_pcts.get)
    if pos_pcts[worst_pos] > 20:
        html += f"""<div class="insight negative">
            <strong>True TC violations (visual appears after narration).</strong>
            <strong>{worst_pos}</strong> has {pos_dt[worst_pos]} keywords ({pos_pcts[worst_pos]:.0f}% of matched)
            where the visual appears AFTER the narrator speaks the concept.
            These are the genuine temporal contiguity issues to address.
        </div>\n"""
    else:
        html += f"""<div class="insight positive">
            <strong>Low true TC violation rates across all videos.</strong>
            Most matched keywords have visuals already on screen (delta_t ≤ 0).
            The few positive-delta_t cases are the only true synchronization issues.
        </div>\n"""

    # 5. OCR vocabulary mismatch dominance
    ocr_mismatch_pcts = {}
    for v in vids:
        reasons = _safe_get(results[v], "keyword_level", "case_F_reasons", default={})
        n_unmatched = _safe_get(results[v], "keyword_level", "n_unmatched", default=0)
        if n_unmatched > 0:
            ocr_mismatch_pcts[v] = reasons.get("ocr_vocabulary_mismatch", 0) / n_unmatched * 100
        else:
            ocr_mismatch_pcts[v] = 0

    avg_mismatch = sum(ocr_mismatch_pcts.values()) / len(ocr_mismatch_pcts) if ocr_mismatch_pcts else 0
    if avg_mismatch > 60:
        html += f"""<div class="insight neutral">
            <strong>OCR vocabulary mismatch is the dominant reason for unmatched keywords ({avg_mismatch:.0f}% avg).</strong>
            The narrator uses different terminology than what appears on slides.
            This is a common pattern in educational videos — the instructor paraphrases slide content.
            This is NOT necessarily a TC problem; it may indicate good pedagogical practice (explaining in own words).
        </div>\n"""

    # 6. Visual lead time patterns
    lead_medians = {}
    for v in vids:
        lead_medians[v] = _safe_get(results[v], "visual_lead_time", "median_lead_s", default=0) or 0

    high_lead = {v: l for v, l in lead_medians.items() if l > 5}
    low_lead = {v: l for v, l in lead_medians.items() if l < 2}

    if high_lead and low_lead:
        h_list = ", ".join(f"<strong>{v}</strong> ({l:.1f}s)" for v, l in high_lead.items())
        l_list = ", ".join(f"<strong>{v}</strong> ({l:.1f}s)" for v, l in low_lead.items())
        html += f"""<div class="insight neutral">
            <strong>Pacing difference detected.</strong>
            Some videos show visuals far ahead: {h_list} (slides stay up longer before narration catches up).
            Others are tightly synchronized: {l_list} (slide transitions closer to narration timing).
            This reflects different instructional pacing strategies.
        </div>\n"""

    # 7. Segment coverage comparison
    seg_match_rates = {}
    for v in vids:
        sl = results[v].get("segment_level", {})
        n_t = sl.get("n_total", 1) or 1
        n_m = sl.get("n_with_matches", 0) or 0
        seg_match_rates[v] = n_m / n_t * 100

    best_seg = max(seg_match_rates, key=seg_match_rates.get)
    worst_seg = min(seg_match_rates, key=seg_match_rates.get)
    if seg_match_rates[best_seg] - seg_match_rates[worst_seg] > 15:
        html += f"""<div class="insight neutral">
            <strong>Segment coverage gap.</strong>
            <strong>{best_seg}</strong> has visual matches in {seg_match_rates[best_seg]:.0f}% of segments, while
            <strong>{worst_seg}</strong> only covers {seg_match_rates[worst_seg]:.0f}%.
            Videos with low segment coverage may have longer narration-only sections
            (introductions, verbal examples, transitions).
        </div>\n"""

    return html


# ─── CLI Entry Point ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if len(sys.argv) < 3:
        print("Usage: python -m pipeline_v2_2_gpu.utils.viz_compare <output_root> <video1> <video2> [video3] ...")
        print("Example: python -m pipeline_v2_2_gpu.utils.viz_compare outputs_v2_2 A0 A1 A3 A5")
        sys.exit(1)

    output_root = sys.argv[1]
    video_ids = sys.argv[2:]

    path = generate_comparison(output_root, video_ids)
    print(f"Comparison dashboard: {path}")
