"""Dashboard HTML + JSON generation (Chart.js)."""

import logging
import os
import json
import base64
import pandas as pd
import numpy as np

from ..config import Config
from ..utils.io_utils import load_json

logger = logging.getLogger(__name__)


def generate_dashboard(output_dir: str, video_name: str, cfg: Config) -> str:
    """Generate interactive HTML dashboard. Returns path to HTML file."""
    # Load all available data
    data = _load_all_data(output_dir)
    html = _build_html(data, video_name, cfg)
    path = os.path.join(output_dir, "report_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Dashboard saved: %s", path)
    return path


def _load_all_data(output_dir: str) -> dict:
    data = {}
    csv_files = [
        "scores_per_scene.csv", "keyword_alignment.csv",
        "segment_keyword_scores.csv", "pedagogical_importance.csv",
        "scenes.csv", "scene_concepts.csv", "scores_weighted.csv",
        "transcript_segments_improved.csv",
    ]
    for f in csv_files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            try:
                data[f] = pd.read_csv(path)
            except Exception:
                data[f] = pd.DataFrame()

    results_path = os.path.join(output_dir, "results.json")
    if os.path.exists(results_path):
        data["results"] = load_json(results_path)
    else:
        data["results"] = {}

    return data


def _build_html(data: dict, video_name: str, cfg: Config) -> str:
    results = data.get("results", {})
    scenes_df = data.get("scores_per_scene.csv", pd.DataFrame())
    kw_df = data.get("keyword_alignment.csv", pd.DataFrame())
    imp_df = data.get("pedagogical_importance.csv", pd.DataFrame())
    weighted_df = data.get("scores_weighted.csv", pd.DataFrame())
    concepts_df = data.get("scene_concepts.csv", pd.DataFrame())
    raw_scenes_df = data.get("scenes.csv", pd.DataFrame())

    # Merge keyframe_path from raw scenes into scores dataframe
    if not scenes_df.empty and not raw_scenes_df.empty and "keyframe_path" not in scenes_df.columns:
        if "scene_id" in raw_scenes_df.columns and "keyframe_path" in raw_scenes_df.columns:
            scenes_df = scenes_df.merge(
                raw_scenes_df[["scene_id", "keyframe_path", "t_start", "t_end"]],
                on="scene_id", how="left"
            )

    # Compute dashboard data
    scene_level = results.get("scene_level", {})
    kw_level = results.get("keyword_level", {})
    overall_grade = results.get("overall_grade", "N/A")
    overall_score = results.get("overall_score", 0)

    # Zone distribution
    zone_data = _zone_chart_data(scenes_df, kw_df)

    # Heatmap data
    heatmap = _heatmap_data(scenes_df, imp_df, weighted_df)

    # Scene table rows
    scene_rows = _scene_table_rows(scenes_df, concepts_df)

    # Keyword table rows
    kw_rows = _keyword_table_rows(kw_df)

    # Grounding method pie data
    grounding_pie = _grounding_pie_data(kw_df)

    # Groundability bar data
    groundability_bar = _groundability_bar_data(kw_df)

    # Top 5 segments to fix
    top5 = _top5_to_fix(weighted_df if not weighted_df.empty else scenes_df)

    # Unverified alignments
    unverified = _unverified_list(scenes_df)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Temporal Contiguity Report — {video_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ font-size: 1.6em; margin-bottom: 8px; }}
.header .subtitle {{ opacity: 0.8; font-size: 0.95em; }}
.grade-badge {{ display: inline-block; padding: 8px 20px; border-radius: 20px; font-weight: 700; font-size: 1.1em; margin-top: 12px; }}
.grade-Excellent {{ background: #10b981; color: white; }}
.grade-Good {{ background: #3b82f6; color: white; }}
.grade-Acceptable {{ background: #f59e0b; color: white; }}
.grade-Poor {{ background: #ef4444; color: white; }}
.grade-Unacceptable {{ background: #7f1d1d; color: white; }}
.grade-NA {{ background: #6b7280; color: white; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin-bottom: 24px; }}
.metric {{ background: white; padding: 20px; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; }}
.metric .value {{ font-size: 1.8em; font-weight: 700; color: #1a1a2e; }}
.metric .label {{ font-size: 0.85em; color: #6b7280; margin-top: 4px; }}
.panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; overflow: hidden; }}
.panel-header {{ padding: 16px 20px; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 1.05em; }}
.panel-body {{ padding: 20px; }}
.chart-container {{ position: relative; height: 300px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
th {{ background: #f8fafc; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; position: sticky; top: 0; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
tr:hover {{ background: #f8fafc; }}
.zone-optimal {{ color: #10b981; font-weight: 600; }}
.zone-suboptimal {{ color: #f59e0b; font-weight: 600; }}
.zone-disruptive {{ color: #ef4444; font-weight: 600; }}
.zone-unacceptable {{ color: #7f1d1d; font-weight: 600; }}
.alpha-high {{ background: #d1fae5; color: #065f46; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; }}
.alpha-medium {{ background: #fef3c7; color: #92400e; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; }}
.alpha-low {{ background: #fee2e2; color: #991b1b; padding: 2px 8px; border-radius: 10px; font-size: 0.85em; }}
.verify-badge {{ background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 4px; font-size: 0.8em; }}
.heatmap {{ display: flex; height: 40px; border-radius: 6px; overflow: hidden; margin: 10px 0; }}
.heatmap-cell {{ flex: 1; min-width: 2px; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
.scrollable {{ max-height: 500px; overflow-y: auto; }}
.thumb {{ width: 80px; height: 45px; object-fit: cover; border-radius: 4px; }}
</style>
</head>
<body>
<div class="container">

<!-- HEADER -->
<div class="header">
  <h1>Temporal Contiguity Analysis Report</h1>
  <div class="subtitle">Video: <strong>{video_name}</strong> | Pipeline v2.1</div>
  <div class="grade-badge grade-{overall_grade.replace(' ', '')}">
    {overall_grade} — Score: {overall_score:.1f}
  </div>
</div>

<!-- KEY METRICS -->
<div class="metrics">
  <div class="metric">
    <div class="value">{scene_level.get('n_scenes', 0)}</div>
    <div class="label">Scenes</div>
  </div>
  <div class="metric">
    <div class="value">{scene_level.get('n_matched', 0)}</div>
    <div class="label">Matched</div>
  </div>
  <div class="metric">
    <div class="value">{_fmt(scene_level.get('mean_S_temporal'))}</div>
    <div class="label">Mean S<sub>temporal</sub> (Scene)</div>
  </div>
  <div class="metric">
    <div class="value">{_fmt(scene_level.get('mean_alpha'), decimals=2)}</div>
    <div class="label">Mean &alpha; (Scene)</div>
  </div>
  <div class="metric">
    <div class="value">{kw_level.get('n_keywords_total', 0)}</div>
    <div class="label">Keywords</div>
  </div>
  <div class="metric">
    <div class="value">{kw_level.get('n_keywords_grounded', 0)}</div>
    <div class="label">Grounded</div>
  </div>
  <div class="metric">
    <div class="value">{_fmt(kw_level.get('mean_S_temporal'))}</div>
    <div class="label">Mean S<sub>temporal</sub> (Keyword)</div>
  </div>
  <div class="metric">
    <div class="value">{_fmt(scene_level.get('pct_high_confidence'), decimals=0)}%</div>
    <div class="label">High Confidence (&alpha;&ge;0.6)</div>
  </div>
</div>

<!-- PANEL 1: COGNITIVE LOAD HEAT MAP -->
<div class="panel">
  <div class="panel-header">1. Cognitive Load Heat Map (Timeline)</div>
  <div class="panel-body">
    <div style="font-size:0.85em;color:#6b7280;margin-bottom:8px;">
      <span style="background:#10b981;color:white;padding:2px 8px;border-radius:4px;">Optimal</span>
      <span style="background:#f59e0b;color:white;padding:2px 8px;border-radius:4px;margin-left:4px;">Suboptimal</span>
      <span style="background:#ef4444;color:white;padding:2px 8px;border-radius:4px;margin-left:4px;">Disruptive</span>
      <span style="background:#7f1d1d;color:white;padding:2px 8px;border-radius:4px;margin-left:4px;">Unacceptable</span>
      <span style="background:#d1d5db;color:#374151;padding:2px 8px;border-radius:4px;margin-left:4px;">No Match</span>
    </div>
    <div class="heatmap">{heatmap}</div>
  </div>
</div>

<!-- PANEL 2: ZONE DISTRIBUTION -->
<div class="two-col">
  <div class="panel">
    <div class="panel-header">2. Zone Distribution</div>
    <div class="panel-body">
      <div class="chart-container"><canvas id="zoneChart"></canvas></div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header">5. Grounding Method Breakdown</div>
    <div class="panel-body">
      <div class="chart-container"><canvas id="groundingPie"></canvas></div>
    </div>
  </div>
</div>

<!-- PANEL 3: SCENE TABLE -->
<div class="panel">
  <div class="panel-header">3. Scene-Level Alignment</div>
  <div class="panel-body scrollable">
    <table>
      <thead><tr>
        <th>Scene</th><th>Keyframe</th><th>Type</th><th>t<sub>vis</sub></th><th>t<sub>narr</sub></th><th>&Delta;t (s)</th>
        <th>S<sub>temporal</sub></th><th>&alpha;</th><th>Zone</th><th>Track</th><th>Concepts</th>
      </tr></thead>
      <tbody>{scene_rows}</tbody>
    </table>
  </div>
</div>

<!-- PANEL 4: KEYWORD TABLE -->
<div class="panel">
  <div class="panel-header">4. Keyword-Level Alignment</div>
  <div class="panel-body scrollable">
    <table>
      <thead><tr>
        <th>Keyword</th><th>Segment</th><th>t<sub>narr</sub></th><th>t<sub>vis</sub></th>
        <th>&Delta;t</th><th>Method</th><th>Confidence</th><th>Groundability</th><th>Zone</th>
      </tr></thead>
      <tbody>{kw_rows}</tbody>
    </table>
  </div>
</div>

<!-- PANEL 6: GROUNDABILITY DISTRIBUTION -->
<div class="two-col">
  <div class="panel">
    <div class="panel-header">6. Groundability Distribution</div>
    <div class="panel-body">
      <div class="chart-container"><canvas id="groundabilityBar"></canvas></div>
    </div>
  </div>
  <div class="panel">
    <div class="panel-header">10. Temporal Statistics</div>
    <div class="panel-body">
      <table>
        <tr><td>Mean &Delta;t</td><td><strong>{_fmt(scene_level.get('mean_delta_t'))}s</strong></td></tr>
        <tr><td>SD &Delta;t</td><td><strong>{_fmt(scene_level.get('sd_delta_t'))}s</strong></td></tr>
        <tr><td>Min &Delta;t</td><td><strong>{_fmt(scene_level.get('min_delta_t'))}s</strong></td></tr>
        <tr><td>Max &Delta;t</td><td><strong>{_fmt(scene_level.get('max_delta_t'))}s</strong></td></tr>
        <tr><td>% Optimal</td><td><strong>{_fmt(scene_level.get('pct_Optimal'), 1)}%</strong></td></tr>
        <tr><td>% Suboptimal</td><td><strong>{_fmt(scene_level.get('pct_Suboptimal'), 1)}%</strong></td></tr>
        <tr><td>% Disruptive</td><td><strong>{_fmt(scene_level.get('pct_Disruptive'), 1)}%</strong></td></tr>
        <tr><td>% Unacceptable</td><td><strong>{_fmt(scene_level.get('pct_Unacceptable'), 1)}%</strong></td></tr>
      </table>
    </div>
  </div>
</div>

<!-- PANEL 8: TOP 5 TO FIX -->
<div class="panel">
  <div class="panel-header">8. Top 5 Segments to Fix</div>
  <div class="panel-body">
    {top5}
  </div>
</div>

<!-- PANEL 9: UNVERIFIED -->
<div class="panel">
  <div class="panel-header">9. Unverified Alignments (&alpha; &lt; 0.6)</div>
  <div class="panel-body scrollable">
    {unverified}
  </div>
</div>

</div>

<script>
// Zone chart
new Chart(document.getElementById('zoneChart'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(zone_data['labels'])},
    datasets: [{{
      label: 'Scene-level',
      data: {json.dumps(zone_data['scene_counts'])},
      backgroundColor: ['#10b981', '#f59e0b', '#ef4444', '#7f1d1d'],
    }}, {{
      label: 'Keyword-level',
      data: {json.dumps(zone_data['kw_counts'])},
      backgroundColor: ['#6ee7b7', '#fcd34d', '#fca5a5', '#b91c1c'],
    }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});

// Grounding pie
new Chart(document.getElementById('groundingPie'), {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(grounding_pie['labels'])},
    datasets: [{{ data: {json.dumps(grounding_pie['counts'])},
      backgroundColor: ['#3b82f6','#10b981','#f59e0b','#8b5cf6','#6b7280'] }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});

// Groundability bar
new Chart(document.getElementById('groundabilityBar'), {{
  type: 'bar',
  data: {{
    labels: {json.dumps(groundability_bar['labels'])},
    datasets: [{{ label: 'Keywords',
      data: {json.dumps(groundability_bar['counts'])},
      backgroundColor: ['#10b981','#f59e0b','#ef4444'] }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }} }}
}});
</script>
</body>
</html>"""
    return html


def _fmt(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def _zone_chart_data(scenes_df, kw_df):
    labels = ["Optimal", "Suboptimal", "Disruptive", "Unacceptable"]
    scene_counts = [0, 0, 0, 0]
    kw_counts = [0, 0, 0, 0]

    if not scenes_df.empty and "zone" in scenes_df.columns:
        for z in scenes_df["zone"]:
            if z in labels:
                scene_counts[labels.index(z)] += 1

    if not kw_df.empty and "zone" in kw_df.columns:
        for z in kw_df["zone"]:
            if z in labels:
                kw_counts[labels.index(z)] += 1

    return {"labels": labels, "scene_counts": scene_counts, "kw_counts": kw_counts}


def _heatmap_data(scenes_df, imp_df, weighted_df):
    df = weighted_df if not weighted_df.empty else scenes_df
    if df.empty:
        return '<div style="color:#6b7280;">No data</div>'

    cells = []
    for _, row in df.iterrows():
        s = row.get("S_temporal", None)
        alpha = row.get("alpha", 0.5)
        if s is None or pd.isna(s):
            color = "#d1d5db"
        elif alpha < 0.3:
            color = "#d1d5db"  # gray = unverified
        elif s >= 80:
            color = "#10b981"
        elif s >= 40:
            color = "#f59e0b"
        elif s >= 20:
            color = "#ef4444"
        else:
            color = "#7f1d1d"
        cells.append(f'<div class="heatmap-cell" style="background:{color};" title="S={_fmt(s)} α={_fmt(alpha,2)}"></div>')

    return "\n".join(cells)


def _scene_table_rows(scenes_df, concepts_df):
    if scenes_df.empty:
        return "<tr><td colspan='9'>No scene data</td></tr>"

    # Merge keyframe_path from concepts_df (which has it from scenes.csv)
    if not concepts_df.empty and "keyframe_path" not in scenes_df.columns:
        # Try to get keyframe_path from the raw scenes.csv data
        pass

    rows = []
    for _, row in scenes_df.iterrows():
        sid = row.get("scene_id", "")
        s_temp = row.get("S_temporal", None)
        alpha = row.get("alpha", 0)
        dt = row.get("delta_t", None)
        zone = row.get("zone", "No Match")
        track = row.get("match_track", "")
        concept = str(row.get("concept_text", ""))[:80]

        # Keyframe thumbnail — try keyframe_path, or look up from scenes_raw
        kf = str(row.get("keyframe_path", ""))
        if kf == "nan" or not kf:
            # Try to find from concepts_df
            if not concepts_df.empty and "keyframe_path" in concepts_df.columns:
                match = concepts_df[concepts_df["scene_id"] == sid]
                if not match.empty:
                    kf = str(match.iloc[0].get("keyframe_path", ""))
        thumb = ""
        if kf and kf != "nan" and os.path.exists(kf):
            try:
                with open(kf, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode()
                thumb = f'<img class="thumb" src="data:image/jpeg;base64,{b64}">'
            except Exception:
                thumb = ""

        # Alpha styling
        if alpha >= 0.6:
            alpha_cls = "alpha-high"
        elif alpha >= 0.3:
            alpha_cls = "alpha-medium"
        else:
            alpha_cls = "alpha-low"

        zone_cls = f"zone-{zone.lower()}" if zone != "No Match" else ""

        t_vis = row.get("t_vis", row.get("t_start", None))
        t_narr = row.get("t_narr", None)

        rows.append(f"""<tr>
            <td>{sid}</td>
            <td>{thumb}</td>
            <td>{row.get('scene_type', row.get('frame_type', ''))}</td>
            <td>{_fmt(t_vis) if t_vis is not None and not pd.isna(t_vis) else 'N/A'}</td>
            <td>{_fmt(t_narr) if t_narr is not None and not pd.isna(t_narr) else 'N/A'}</td>
            <td>{_fmt(dt) if dt is not None and not pd.isna(dt) else 'N/A'}</td>
            <td>{_fmt(s_temp) if s_temp is not None and not pd.isna(s_temp) else 'N/A'}</td>
            <td><span class="{alpha_cls}">{_fmt(alpha, 2)}</span>
                {' <span class="verify-badge">⚠ verify</span>' if alpha < 0.6 else ''}</td>
            <td class="{zone_cls}">{zone}</td>
            <td>{track}</td>
            <td title="{concept}">{concept[:50]}{'...' if len(concept)>50 else ''}</td>
        </tr>""")

    return "\n".join(rows)


def _keyword_table_rows(kw_df):
    if kw_df.empty:
        return "<tr><td colspan='9'>No keyword data</td></tr>"

    from ..stages.scoring import zone_classification, gaussian_score

    rows = []
    for _, row in kw_df.iterrows():
        dt = row.get("delta_t", None)
        zone = zone_classification(dt) if dt is not None and not pd.isna(dt) else "N/A"
        zone_cls = f"zone-{zone.lower()}" if zone not in ("N/A", "No Match") else ""

        ts_warn = ""
        if row.get("timestamp_reliable") == False:
            ts_warn = " ⚠"

        rows.append(f"""<tr>
            <td>{row.get('keyword_text', '')}</td>
            <td>{row.get('segment_id', '')}</td>
            <td>{_fmt(row.get('t_narr'))}{ts_warn}</td>
            <td>{_fmt(row.get('t_vis')) if row.get('t_vis') is not None else 'N/A'}</td>
            <td>{_fmt(dt) if dt is not None and not pd.isna(dt) else 'N/A'}</td>
            <td>{row.get('method', '')}</td>
            <td>{row.get('confidence', '')}</td>
            <td>{row.get('groundability', '')}</td>
            <td class="{zone_cls}">{zone}</td>
        </tr>""")

    return "\n".join(rows[:200])  # Limit to 200 rows


def _grounding_pie_data(kw_df):
    if kw_df.empty or "method" not in kw_df.columns:
        return {"labels": ["No data"], "counts": [1]}

    counts = kw_df[kw_df["is_visual"] == True]["method"].value_counts() if "is_visual" in kw_df.columns else kw_df["method"].value_counts()
    return {"labels": counts.index.tolist(), "counts": counts.values.tolist()}


def _groundability_bar_data(kw_df):
    labels = ["HIGH", "MEDIUM", "LOW"]
    counts = [0, 0, 0]
    if not kw_df.empty and "groundability" in kw_df.columns:
        vc = kw_df["groundability"].value_counts()
        for i, l in enumerate(labels):
            counts[i] = int(vc.get(l, 0))
    return {"labels": labels, "counts": counts}


def _top5_to_fix(df):
    if df.empty or "priority" not in df.columns:
        if df.empty or "S_temporal" not in df.columns:
            return "<p>No data available</p>"
        # Fallback: sort by S_temporal ascending
        worst = df[df["S_temporal"].notna()].nsmallest(5, "S_temporal")
    else:
        worst = df.nlargest(5, "priority")

    if worst.empty:
        return "<p>All segments look good!</p>"

    rows = []
    for i, (_, row) in enumerate(worst.iterrows(), 1):
        s = row.get("S_temporal", 0)
        dt = row.get("delta_t", 0)
        concept = str(row.get("concept_text", ""))[:60]
        rows.append(f"""<div style="padding:8px 0;border-bottom:1px solid #f1f5f9;">
            <strong>#{i}</strong> Scene {row.get('scene_id','')} —
            S<sub>temporal</sub>={_fmt(s)}, &Delta;t={_fmt(dt)}s
            <br><span style="color:#6b7280;font-size:0.9em;">{concept}</span>
        </div>""")

    return "\n".join(rows)


def _unverified_list(scenes_df):
    if scenes_df.empty or "alpha" not in scenes_df.columns:
        return "<p>No data</p>"

    unverified = scenes_df[scenes_df["alpha"] < 0.6]
    if unverified.empty:
        return "<p>All alignments have high confidence!</p>"

    rows = []
    for _, row in unverified.iterrows():
        rows.append(f"""<div style="padding:6px 0;border-bottom:1px solid #f1f5f9;">
            Scene {row.get('scene_id','')} — &alpha;={_fmt(row['alpha'],2)},
            Track: {row.get('match_track','')},
            &Delta;t={_fmt(row.get('delta_t'))}s
        </div>""")

    return "\n".join(rows[:20])
