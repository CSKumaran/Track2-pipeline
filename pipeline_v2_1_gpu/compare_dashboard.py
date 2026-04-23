"""Generate cross-video comparison dashboard."""

import os
import json
import pandas as pd
import numpy as np


def generate_comparison(output_root: str) -> str:
    """Generate HTML comparison across all processed videos."""
    video_dirs = sorted([
        d for d in os.listdir(output_root)
        if os.path.isdir(os.path.join(output_root, d))
    ])

    all_data = []
    for vdir in video_dirs:
        results_path = os.path.join(output_root, vdir, "results.json")
        if not os.path.exists(results_path):
            continue
        with open(results_path) as f:
            results = json.load(f)

        scene_level = results.get("scene_level", {})
        kw_level = results.get("keyword_level", {})

        all_data.append({
            "video": vdir,
            "overall_grade": results.get("overall_grade", "N/A"),
            "overall_score": results.get("overall_score", 0),
            "n_scenes": scene_level.get("n_scenes", 0),
            "n_matched": scene_level.get("n_matched", 0),
            "mean_S_scene": scene_level.get("mean_S_temporal"),
            "median_S_scene": scene_level.get("median_S_temporal"),
            "mean_delta_t": scene_level.get("mean_delta_t"),
            "sd_delta_t": scene_level.get("sd_delta_t"),
            "pct_Optimal": scene_level.get("pct_Optimal", 0),
            "pct_Suboptimal": scene_level.get("pct_Suboptimal", 0),
            "pct_Disruptive": scene_level.get("pct_Disruptive", 0),
            "pct_Unacceptable": scene_level.get("pct_Unacceptable", 0),
            "n_keywords": kw_level.get("n_keywords_total", 0),
            "n_grounded": kw_level.get("n_keywords_grounded", 0),
            "mean_S_kw": kw_level.get("mean_S_temporal"),
        })

    if not all_data:
        return ""

    df = pd.DataFrame(all_data)

    # Build HTML
    videos_json = json.dumps(df["video"].tolist())
    scores_json = json.dumps([r.get("overall_score", 0) or 0 for r in all_data])
    grades_json = json.dumps(df["overall_grade"].tolist())

    # Zone data per video
    zone_optimal = json.dumps(df["pct_Optimal"].fillna(0).tolist())
    zone_suboptimal = json.dumps(df["pct_Suboptimal"].fillna(0).tolist())
    zone_disruptive = json.dumps(df["pct_Disruptive"].fillna(0).tolist())
    zone_unacceptable = json.dumps(df["pct_Unacceptable"].fillna(0).tolist())

    # Table rows
    table_rows = ""
    for _, row in df.iterrows():
        grade = row["overall_grade"]
        score = row["overall_score"] or 0
        grade_cls = f"grade-{grade.replace(' ', '')}" if grade != "N/A" else "grade-NA"

        table_rows += f"""<tr>
            <td><strong>{row['video']}</strong></td>
            <td><span class="{grade_cls}" style="padding:3px 10px;border-radius:10px;color:white;font-weight:600;">{grade}</span></td>
            <td><strong>{score:.1f}</strong></td>
            <td>{row['n_scenes']}</td>
            <td>{row['n_matched']}</td>
            <td>{_fmt(row['mean_S_scene'])}</td>
            <td>{_fmt(row['mean_delta_t'])}s</td>
            <td>{_fmt(row['sd_delta_t'])}s</td>
            <td>{row['pct_Optimal']:.0f}%</td>
            <td>{row['n_keywords']}</td>
            <td>{row['n_grounded']}</td>
            <td>{_fmt(row['mean_S_kw'])}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pipeline v2.1 — Cross-Video Comparison</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #f5f7fa; color: #1a1a2e; line-height: 1.6; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 20px; }}
.header {{ background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px; }}
.header h1 {{ font-size: 1.6em; margin-bottom: 8px; }}
.panel {{ background: white; border-radius: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px; overflow: hidden; }}
.panel-header {{ padding: 16px 20px; border-bottom: 1px solid #e5e7eb; font-weight: 600; font-size: 1.05em; }}
.panel-body {{ padding: 20px; }}
.chart-container {{ position: relative; height: 350px; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.88em; }}
th {{ background: #f8fafc; padding: 10px 12px; text-align: left; font-weight: 600; border-bottom: 2px solid #e5e7eb; }}
td {{ padding: 8px 12px; border-bottom: 1px solid #f1f5f9; }}
tr:hover {{ background: #f8fafc; }}
.grade-Excellent {{ background: #10b981; }}
.grade-Good {{ background: #3b82f6; }}
.grade-Acceptable {{ background: #f59e0b; }}
.grade-Poor {{ background: #ef4444; }}
.grade-Unacceptable {{ background: #7f1d1d; }}
.grade-NA {{ background: #6b7280; }}
.two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }}
@media (max-width: 768px) {{ .two-col {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<div class="container">
<div class="header">
  <h1>Pipeline v2.1 — Cross-Video Comparison</h1>
  <div style="opacity:0.8;">{len(all_data)} videos processed</div>
</div>

<div class="two-col">
  <div class="panel">
    <div class="panel-header">Overall Scores</div>
    <div class="panel-body"><div class="chart-container"><canvas id="scoreChart"></canvas></div></div>
  </div>
  <div class="panel">
    <div class="panel-header">Zone Distribution</div>
    <div class="panel-body"><div class="chart-container"><canvas id="zoneChart"></canvas></div></div>
  </div>
</div>

<div class="panel">
  <div class="panel-header">Detailed Comparison</div>
  <div class="panel-body" style="overflow-x:auto;">
    <table>
      <thead><tr>
        <th>Video</th><th>Grade</th><th>Score</th><th>Scenes</th><th>Matched</th>
        <th>Mean S<sub>t</sub></th><th>Mean &Delta;t</th><th>SD &Delta;t</th>
        <th>% Optimal</th><th>Keywords</th><th>Grounded</th><th>KW S<sub>t</sub></th>
      </tr></thead>
      <tbody>{table_rows}</tbody>
    </table>
  </div>
</div>
</div>

<script>
new Chart(document.getElementById('scoreChart'), {{
  type: 'bar',
  data: {{
    labels: {videos_json},
    datasets: [{{
      label: 'Overall Score',
      data: {scores_json},
      backgroundColor: {scores_json}.map(s => s >= 80 ? '#10b981' : s >= 60 ? '#3b82f6' : s >= 40 ? '#f59e0b' : '#ef4444'),
    }}]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, scales: {{ y: {{ min: 0, max: 100 }} }},
    plugins: {{ legend: {{ display: false }} }} }}
}});

new Chart(document.getElementById('zoneChart'), {{
  type: 'bar',
  data: {{
    labels: {videos_json},
    datasets: [
      {{ label: 'Optimal', data: {zone_optimal}, backgroundColor: '#10b981' }},
      {{ label: 'Suboptimal', data: {zone_suboptimal}, backgroundColor: '#f59e0b' }},
      {{ label: 'Disruptive', data: {zone_disruptive}, backgroundColor: '#ef4444' }},
      {{ label: 'Unacceptable', data: {zone_unacceptable}, backgroundColor: '#7f1d1d' }},
    ]
  }},
  options: {{ responsive: true, maintainAspectRatio: false, scales: {{ x: {{ stacked: true }}, y: {{ stacked: true, max: 100 }} }},
    plugins: {{ legend: {{ position: 'bottom' }} }} }}
}});
</script>
</body>
</html>"""

    path = os.path.join(output_root, "comparison_dashboard.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Comparison dashboard saved: {path}")
    return path


def _fmt(val, decimals=1):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs_v2_1"
    generate_comparison(root)
