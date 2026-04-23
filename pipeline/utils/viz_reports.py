"""HTML + JSON report generators for the temporal contiguity pipeline.

Produces polished dashboard-style reports with Chart.js visualizations.
"""

from __future__ import annotations

import json
import os
import logging
from datetime import datetime
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


# ── Grading logic ────────────────────────────────────────────────────

def _overall_grade(mean_score: float) -> str:
    """Map mean S_final to an overall grade label."""
    if mean_score >= 80:
        return "Excellent"
    elif mean_score >= 60:
        return "Good"
    elif mean_score >= 40:
        return "Acceptable"
    elif mean_score >= 20:
        return "Poor"
    else:
        return "Unacceptable"


# ── Concepts report (unchanged) ─────────────────────────────────────

def generate_concepts_html_report(
    scenes_df: pd.DataFrame,
    frames_dir: str,
    output_path: str,
    transcript_segments_df: Optional[pd.DataFrame] = None,
) -> str:
    """Write an HTML report showing scene frames + concept text."""
    lines: list[str] = [
        "<!DOCTYPE html>\n<html><head>"
        "<title>Scene Concepts Report</title>"
        "<style>"
        "body{font-family:sans-serif;margin:20px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}"
        "th{background:#f4f4f4}"
        "img{max-width:320px;height:auto}"
        "</style></head><body>\n"
    ]
    lines.append("<h1>Scene Concepts Report</h1>\n")
    lines.append(f"<p>Frames directory: <code>{frames_dir}</code></p>\n")
    lines.append("<table><tr><th>#</th><th>Frame</th>"
                 "<th>Concept</th><th>Nearby transcript</th></tr>\n")

    for _, row in scenes_df.iterrows():
        sid = int(row["scene_id"])
        concept = row.get("concept_text", "—")
        frame_path = os.path.join(frames_dir, f"scene_{sid}.jpg")
        rel_frame = os.path.relpath(frame_path, os.path.dirname(output_path))

        snippet = "—"
        if transcript_segments_df is not None and "t_seg" not in transcript_segments_df.columns:
            transcript_segments_df = transcript_segments_df.copy()
            transcript_segments_df["t_seg"] = 0.5 * (
                transcript_segments_df["start_time"] + transcript_segments_df["end_time"]
            )
        if transcript_segments_df is not None and len(transcript_segments_df):
            t_vis = row["t_vis"]
            diffs = (transcript_segments_df["t_seg"] - t_vis).abs()
            nearest_idx = diffs.idxmin()
            snippet = str(transcript_segments_df.loc[nearest_idx, "text"])

        lines.append(
            f"<tr><td>{sid}</td>"
            f'<td><img src="{rel_frame}"></td>'
            f"<td>{concept}</td>"
            f"<td>{snippet}</td></tr>\n"
        )

    lines.append("</table></body></html>\n")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Wrote concepts report → %s", output_path)
    return output_path


# ── Polished dashboard report ────────────────────────────────────────

def _fmt_time(seconds: Optional[float]) -> str:
    """Format seconds as mm:ss.s or '—'."""
    if seconds is None or pd.isna(seconds):
        return "—"
    m, s = divmod(abs(seconds), 60)
    return f"{int(m):02d}:{s:04.1f}"


def _fmt_delta(delta: Optional[float]) -> str:
    """Format Δt with sign, or '—'."""
    if delta is None or pd.isna(delta):
        return "N/A"
    sign = "+" if delta >= 0 else "−"
    return f"{sign}{abs(delta):.2f}s"


def _zone_css_class(zone: str) -> str:
    return "zone-" + zone.lower().replace(" ", "-")


def generate_dashboard_report(
    scores_df: pd.DataFrame,
    video_agg: dict,
    video_name: str,
    threshold: float,
    output_path: str,
    vlm_mode: str = "ollama",
    concept_texts: Optional[dict] = None,
    frames_dir: Optional[str] = None,
) -> str:
    """Generate a polished single-page HTML dashboard for one threshold.

    Parameters
    ----------
    scores_df : per-scene scores (for this threshold only)
    video_agg : dict with aggregate stats (mean_S_final, pct_Optimal, etc.)
    video_name : basename of the video
    threshold : the SSIM threshold used
    output_path : where to write the HTML
    vlm_mode : VLM backend name
    concept_texts : optional dict {scene_id: {"ocr_text": ..., "vlm_text": ..., "new_words": ...}}
    frames_dir : optional path to scene frames for thumbnails
    """
    n_scenes = int(video_agg.get("n_scenes", len(scores_df)))
    mean_sf = float(video_agg.get("mean_S_final", 0))
    std_sf = float(scores_df["S_final"].std()) if len(scores_df) > 1 else 0.0
    grade = _overall_grade(mean_sf)
    pct_opt = float(video_agg.get("pct_Optimal", 0))
    pct_sub = float(video_agg.get("pct_Suboptimal", 0))
    pct_dis = float(video_agg.get("pct_Disruptive", 0))
    pct_una = float(video_agg.get("pct_Unacceptable", 0))

    # Δt stats for matched rows
    matched = scores_df[scores_df["match_type"] == "matched"]
    if len(matched) > 0:
        mean_dt = float(matched["delta_t"].astype(float).mean())
        std_dt = float(matched["delta_t"].astype(float).std())
    else:
        mean_dt = 0.0
        std_dt = 0.0

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # Build per-scene data for charts and table
    scene_labels = []
    scene_scores = []
    table_rows_html = []

    for _, row in scores_df.iterrows():
        sid = int(row["scene_id"])
        match_type = str(row.get("match_type", ""))
        is_non_content = match_type == "non_content"

        # For charts, skip non-content scenes
        if not is_non_content:
            scene_labels.append(f"Scene {sid}")
            s_final_val = float(row["S_final"]) if row.get("S_final") is not None and not pd.isna(row.get("S_final", None)) else 0.0
            scene_scores.append(round(s_final_val, 2))

        new_words = "—"
        ocr_text = "—"
        vlm_text = "—"
        if concept_texts and sid in concept_texts:
            info = concept_texts[sid]
            if isinstance(info, dict):
                nw = info.get("new_words", "")
                new_words = nw if nw else "—"
                ot = info.get("ocr_text", "")
                ocr_text = ot if ot else "—"
                if len(ocr_text) > 120:
                    ocr_text = ocr_text[:117] + "…"
                vt = info.get("vlm_text", "")
                vlm_text = vt if vt and vt != "PLACEHOLDER - VLM not run" else "—"
                if len(vlm_text) > 150:
                    vlm_text = vlm_text[:147] + "…"
            else:
                # Legacy: plain string
                ocr_text = str(info)
                if len(ocr_text) > 120:
                    ocr_text = ocr_text[:117] + "…"

        zone = str(row.get("zone_label", ""))
        zone_cls = _zone_css_class(zone)

        # Scene type badge
        scene_type = str(row.get("scene_type", "")) if row.get("scene_type") and not pd.isna(row.get("scene_type", None)) else ""
        scene_type_conf = row.get("scene_type_conf")
        if scene_type:
            type_cls = f"type-{scene_type}"
            conf_str = f" ({float(scene_type_conf):.0%})" if scene_type_conf and not pd.isna(scene_type_conf) else ""
            type_badge = f'<span class="scene-type-badge {type_cls}">{scene_type.replace("_", " ")}{conf_str}</span>'
        else:
            type_badge = "—"

        # Match track badge
        match_track = str(row.get("match_track", "")) if row.get("match_track") and not pd.isna(row.get("match_track", None)) else ""
        track_labels = {"word_exact": "A (OCR)", "clip_vision": "B (CLIP)", "semantic": "C (Sem)"}
        if match_track:
            track_cls = f"track-{match_track}"
            track_label = track_labels.get(match_track, match_track)
            track_badge = f'<span class="track-badge {track_cls}">{track_label}</span>'
        else:
            track_badge = '<span class="track-badge track-none">—</span>'

        if is_non_content:
            dt_str = "—"
            t_vis_str = _fmt_time(row.get("t_vis"))
            t_narr_str = "—"
            score_str = "—"
            sim_str = "—"
            word_window = "—"
            tr_class = ' class="non-content"'
        else:
            dt_str = _fmt_delta(row.get("delta_t"))
            t_vis_str = _fmt_time(row.get("t_vis"))
            t_narr_str = _fmt_time(row.get("t_narr"))
            s_final_val = float(row["S_final"]) if row.get("S_final") is not None and not pd.isna(row.get("S_final", None)) else 0.0
            score_str = f"{s_final_val:.1f}"
            sim_seg = row.get("sim_segment")
            sim_w = row.get("sim_words")
            sim_str = f"{float(sim_w):.3f}" if sim_w is not None and not pd.isna(sim_w) else (
                f"{float(sim_seg):.3f}" if sim_seg is not None and not pd.isna(sim_seg) else "—"
            )
            word_window = row.get("best_word_window", "—")
            if word_window is None or (isinstance(word_window, float) and pd.isna(word_window)):
                word_window = "—"
            tr_class = ""

        # Frame thumbnail
        frame_img = ""
        if frames_dir:
            frame_path = os.path.join(frames_dir, f"scene_{sid}.jpg")
            if os.path.exists(frame_path):
                rel = os.path.relpath(frame_path, os.path.dirname(output_path))
                frame_img = f'<img src="{rel}" style="max-width:160px;height:auto;border-radius:4px">'

        table_rows_html.append(f"""        <tr{tr_class}>
          <td>{sid}</td>
          <td>{frame_img}</td>
          <td>{type_badge}</td>
          <td>{track_badge}</td>
          <td class="new-words-cell">{new_words}</td>
          <td class="ocr-cell">{ocr_text}</td>
          <td class="vlm-cell">{vlm_text}</td>
          <td>{t_vis_str}</td>
          <td>{t_narr_str}</td>
          <td>{dt_str}</td>
          <td class="score-cell">{score_str}</td>
          <td><span class="zone-badge {zone_cls}">{zone}</span></td>
          <td>{sim_str}</td>
          <td class="small-text">{word_window}</td>
        </tr>""")

    table_body = "\n".join(table_rows_html)

    # Zone data for doughnut
    zone_data = {
        "Optimal": pct_opt,
        "Suboptimal": pct_sub,
        "Disruptive": pct_dis,
        "Unacceptable": pct_una,
    }

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Temporal Contiguity Report — {video_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #f8f9fa; color: #333; padding: 2rem; font-size: 14px; }}
  h1 {{ font-size: 22px; font-weight: 600; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 16px; font-weight: 600; margin: 1.5rem 0 0.75rem; color: #444; }}
  .meta {{ font-size: 13px; color: #777; margin-bottom: 2rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                   gap: 12px; margin-bottom: 2rem; }}
  .card {{ background: white; border: 1px solid #e0e0e0; border-radius: 10px;
           padding: 1rem 1.25rem; }}
  .card-label {{ font-size: 12px; color: #888; margin-bottom: 4px; }}
  .card-value {{ font-size: 26px; font-weight: 600; }}
  .card-sub {{ font-size: 12px; color: #aaa; margin-top: 2px; }}
  .grade-Excellent {{ color: #1a7a1a; }}
  .grade-Good {{ color: #1a5ea6; }}
  .grade-Acceptable {{ color: #e07d00; }}
  .grade-Poor, .grade-Unacceptable {{ color: #c0392b; }}
  .charts {{ display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; margin-bottom: 2rem; }}
  .chart-card {{ background: white; border: 1px solid #e0e0e0; border-radius: 10px; padding: 1rem; }}
  .chart-card canvas {{ max-height: 220px; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; }}
  th {{ background: #f1f3f5; font-weight: 600; font-size: 12px; text-transform: uppercase;
        letter-spacing: 0.04em; padding: 10px 12px; text-align: left; white-space: nowrap; }}
  td {{ padding: 10px 12px; border-top: 1px solid #f0f0f0; vertical-align: top; }}
  tr:hover td {{ background: #fafbff; }}
  .score-cell {{ font-weight: 600; }}
  .zone-badge {{ display: inline-block; padding: 2px 9px; border-radius: 12px;
                 font-size: 11px; font-weight: 600; white-space: nowrap; }}
  .zone-optimal      {{ background: #d4edda; color: #155724; }}
  .zone-suboptimal   {{ background: #fff3cd; color: #856404; }}
  .zone-disruptive   {{ background: #fde8d8; color: #7d3c10; }}
  .zone-unacceptable {{ background: #f8d7da; color: #721c24; }}
  .zone-no-alignment {{ background: #e2e3e5; color: #383d41; }}
  .zone-non-content  {{ background: #e2e3e5; color: #6c757d; }}
  tr.non-content td  {{ opacity: 0.5; background: #f5f5f5; }}
  .reasoning {{ font-size: 12px; color: #555; max-width: 280px; line-height: 1.4; }}
  .new-words-cell {{ font-size: 13px; font-weight: 600; color: #1a5ea6; max-width: 120px; }}
  .ocr-cell {{ font-size: 11px; color: #555; max-width: 200px; line-height: 1.3; }}
  .vlm-cell {{ font-size: 11px; color: #777; max-width: 200px; line-height: 1.3; }}
  .small-text {{ font-size: 11px; color: #777; max-width: 150px; }}
  .scene-type-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                       font-size: 10px; font-weight: 600; white-space: nowrap; }}
  .type-text_slide  {{ background: #d1ecf1; color: #0c5460; }}
  .type-diagram     {{ background: #e8daef; color: #6c3483; }}
  .type-animation   {{ background: #fdebd0; color: #935116; }}
  .type-real_world  {{ background: #d4edda; color: #155724; }}
  .type-code        {{ background: #e2e3e5; color: #383d41; }}
  .type-unknown     {{ background: #f5f5f5; color: #999; }}
  .track-badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px;
                  font-size: 10px; font-weight: 600; white-space: nowrap; }}
  .track-word_exact  {{ background: #d4edda; color: #155724; }}
  .track-clip_vision {{ background: #d1ecf1; color: #0c5460; }}
  .track-semantic    {{ background: #fff3cd; color: #856404; }}
  .track-none        {{ background: #f8d7da; color: #721c24; }}
  .footer {{ margin-top: 2.5rem; font-size: 12px; color: #aaa; text-align: center; }}
  @media(max-width:900px) {{ .charts {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>

<h1>Temporal Contiguity Assessment Report</h1>
<p class="meta">
  Video: <strong>{video_name}</strong> &nbsp;·&nbsp;
  SSIM threshold: <strong>{threshold}</strong> &nbsp;·&nbsp;
  Scenes assessed: <strong>{n_scenes}</strong> &nbsp;·&nbsp;
  VLM: <strong>{vlm_mode}</strong> &nbsp;·&nbsp;
  Generated: {now}
</p>

<div class="summary-grid">
  <div class="card">
    <div class="card-label">Mean score</div>
    <div class="card-value">{mean_sf:.1f}<span style="font-size:16px;color:#aaa">/100</span></div>
    <div class="card-sub">SD = {std_sf:.1f}</div>
  </div>
  <div class="card">
    <div class="card-label">Overall grade</div>
    <div class="card-value grade-{grade}">{grade}</div>
  </div>
  <div class="card">
    <div class="card-label">Optimal (&le;1s)</div>
    <div class="card-value" style="color:#1a7a1a">{pct_opt:.0f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Unacceptable (&gt;5s)</div>
    <div class="card-value" style="color:#c0392b">{pct_una:.0f}%</div>
  </div>
  <div class="card">
    <div class="card-label">Mean &Delta;t</div>
    <div class="card-value">{_fmt_delta(mean_dt)}</div>
    <div class="card-sub">SD = {std_dt:.2f}s</div>
  </div>
  <div class="card">
    <div class="card-label">Content / Total</div>
    <div class="card-value">{int(video_agg.get('n_content', video_agg.get('n_matched', 0)))}/{n_scenes}</div>
    <div class="card-sub">{int(video_agg.get('n_non_content', 0))} non-content</div>
  </div>
</div>

<div class="charts">
  <div class="chart-card">
    <h2>Score per Scene</h2>
    <canvas id="scoreChart"></canvas>
  </div>
  <div class="chart-card">
    <h2>Zone Distribution</h2>
    <canvas id="zoneChart"></canvas>
  </div>
</div>

<h2>Per-Scene Detail</h2>
<div style="overflow-x:auto">
<table>
  <thead>
    <tr>
      <th>#</th>
      <th>Frame</th>
      <th>Type</th>
      <th>Track</th>
      <th>New Words</th>
      <th>OCR Text</th>
      <th>VLM Description</th>
      <th>t_vis</th>
      <th>t_narr</th>
      <th>&Delta;t</th>
      <th>Score</th>
      <th>Zone</th>
      <th>Sim</th>
      <th>Matched Words</th>
    </tr>
  </thead>
  <tbody>
{table_body}
  </tbody>
</table>
</div>

<p class="footer">
  Scoring: Optimal &le;1s (100pts) &middot; Suboptimal 1–3s (100&rarr;70) &middot;
  Disruptive 3–5s (70&rarr;0) &middot; Unacceptable &gt;5s (0pts)<br>
  S_final = S_raw &times; (0.5 + 0.5&times;&alpha;) where &alpha; depends on track: A=1.0, B=CLIP, C=semantic<br>
  Pipeline: Whisper ASR &rarr; SSIM scene detection &rarr; VLM/OCR concept labelling &rarr;
  3-Track alignment (A: OCR exact &rarr; B: CLIP vision &rarr; C: Semantic) &rarr; Temporal contiguity scoring
</p>

<script>
const scores = {json.dumps(scene_scores)};
const labels = {json.dumps(scene_labels)};
const zoneData = {json.dumps(zone_data)};

new Chart(document.getElementById("scoreChart"), {{
  type: "bar",
  data: {{
    labels: labels,
    datasets: [{{
      label: "Contiguity score",
      data: scores,
      backgroundColor: scores.map(s =>
        s >= 70 ? "#28a74540" : s >= 30 ? "#ffc10760" : "#dc354560"
      ),
      borderColor: scores.map(s =>
        s >= 70 ? "#28a745" : s >= 30 ? "#ffc107" : "#dc3545"
      ),
      borderWidth: 1.5,
      borderRadius: 4,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{ legend: {{ display: false }} }},
    scales: {{
      y: {{ min: 0, max: 100, ticks: {{ font: {{ size: 11 }} }} }},
      x: {{ ticks: {{ font: {{ size: 10 }}, maxRotation: 45 }} }}
    }}
  }}
}});

new Chart(document.getElementById("zoneChart"), {{
  type: "doughnut",
  data: {{
    labels: Object.keys(zoneData),
    datasets: [{{
      data: Object.values(zoneData),
      backgroundColor: ["#28a745", "#ffc107", "#fd7e14", "#dc3545"],
      borderWidth: 1,
    }}]
  }},
  options: {{
    responsive: true,
    maintainAspectRatio: false,
    plugins: {{
      legend: {{
        position: "bottom",
        labels: {{ font: {{ size: 11 }}, boxWidth: 12, padding: 8 }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote dashboard report → %s", output_path)
    return output_path


# ── JSON results output ──────────────────────────────────────────────

def generate_json_results(
    scores_df: pd.DataFrame,
    video_agg: dict,
    video_name: str,
    threshold: float,
    output_path: str,
    vlm_mode: str = "ollama",
    concept_texts: Optional[dict] = None,
) -> str:
    """Write a structured JSON results file for one threshold."""

    n_scenes = int(video_agg.get("n_scenes", len(scores_df)))
    mean_sf = float(video_agg.get("mean_S_final", 0))

    matched = scores_df[scores_df["match_type"] == "matched"]
    mean_dt = float(matched["delta_t"].astype(float).mean()) if len(matched) > 0 else None
    std_dt = float(matched["delta_t"].astype(float).std()) if len(matched) > 1 else None

    scene_records = []
    for _, row in scores_df.iterrows():
        sid = int(row["scene_id"])
        concept = ""
        if concept_texts and sid in concept_texts:
            info = concept_texts[sid]
            if isinstance(info, dict):
                concept = info.get("ocr_text", "") + "; " + info.get("vlm_text", "")
            else:
                concept = str(info)

        dt_val = row.get("delta_t")
        if dt_val is not None and not pd.isna(dt_val):
            dt_val = round(float(dt_val), 4)
        else:
            dt_val = None

        t_narr = row.get("t_narr")
        if t_narr is not None and not pd.isna(t_narr):
            t_narr = round(float(t_narr), 4)
        else:
            t_narr = None

        sim_w = row.get("sim_words")
        if sim_w is not None and not pd.isna(sim_w):
            sim_w = round(float(sim_w), 4)
        else:
            sim_w = None

        scene_records.append({
            "scene_id": sid,
            "concept": concept,
            "t_vis": round(float(row["t_vis"]), 4),
            "t_narr": t_narr,
            "delta_t": dt_val,
            "S_raw": float(row["S_raw"]) if row.get("S_raw") is not None and not pd.isna(row.get("S_raw")) else None,
            "S_final": float(row["S_final"]) if row.get("S_final") is not None and not pd.isna(row.get("S_final")) else None,
            "alpha": float(row["alpha"]) if row.get("alpha") is not None and not pd.isna(row.get("alpha")) else None,
            "zone": str(row.get("zone_label", "")),
            "match_type": str(row.get("match_type", "")),
            "sim_words": sim_w,
            "best_word_window": str(row.get("best_word_window", "")),
            "failure_code": str(row.get("failure_code", "")),
        })

    # Zone counts
    zone_counts = scores_df["zone_label"].value_counts().to_dict()

    result = {
        "video_name": video_name,
        "threshold": threshold,
        "vlm_backend": vlm_mode,
        "pipeline_version": "2.0-word-level",
        "generated": datetime.now().isoformat(),
        "n_scenes": n_scenes,
        "scene_scores": scene_records,
        "video_summary": {
            "n_scenes": n_scenes,
            "n_matched": int(video_agg.get("n_matched", 0)),
            "n_no_match": int(video_agg.get("n_no_match", 0)),
            "mean_S_final": round(mean_sf, 2),
            "std_S_final": round(float(scores_df["S_final"].std()), 2) if len(scores_df) > 1 else 0,
            "mean_delta_t": round(mean_dt, 3) if mean_dt is not None else None,
            "std_delta_t": round(std_dt, 3) if std_dt is not None else None,
            "pct_Optimal": float(video_agg.get("pct_Optimal", 0)),
            "pct_Suboptimal": float(video_agg.get("pct_Suboptimal", 0)),
            "pct_Disruptive": float(video_agg.get("pct_Disruptive", 0)),
            "pct_Unacceptable": float(video_agg.get("pct_Unacceptable", 0)),
            "zone_counts": {k: int(v) for k, v in zone_counts.items()},
            "overall_grade": _overall_grade(mean_sf),
        },
    }

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("Wrote JSON results → %s", output_path)
    return output_path


# ── Legacy Δt report (kept for backward compat) ─────────────────────

def generate_delta_t_report(
    scores_df: pd.DataFrame,
    output_path: str,
) -> str:
    """Write a simple HTML table of per-scene Δt and scores (legacy)."""
    lines: list[str] = [
        "<!DOCTYPE html>\n<html><head>"
        "<title>Δt / Scoring Report</title>"
        "<style>"
        "body{font-family:sans-serif;margin:20px}"
        "table{border-collapse:collapse;width:100%}"
        "th,td{border:1px solid #ccc;padding:6px 10px;text-align:left;vertical-align:top}"
        "th{background:#f4f4f4}"
        ".zone-Optimal{color:green}.zone-Suboptimal{color:orange}"
        ".zone-Disruptive{color:red}.zone-Unacceptable{color:darkred}"
        "</style></head><body>\n"
    ]
    lines.append("<h1>Δt &amp; Scoring Report</h1>\n")
    lines.append("<table><tr>"
                 "<th>scene_id</th><th>match_type</th><th>Δt</th>"
                 "<th>S_raw</th><th>α</th><th>S_final</th><th>Zone</th>"
                 "</tr>\n")

    for _, row in scores_df.iterrows():
        zone = row.get("zone_label", "")
        cls = f"zone-{zone}" if zone else ""
        dt = f'{row["delta_t"]:.2f}' if pd.notna(row.get("delta_t")) else "—"
        s_raw = f'{row["S_raw"]:.1f}' if row.get("S_raw") is not None and pd.notna(row.get("S_raw")) else "—"
        alpha = f'{row["alpha"]:.3f}' if row.get("alpha") is not None and pd.notna(row.get("alpha")) else "—"
        s_final = f'{row["S_final"]:.1f}' if row.get("S_final") is not None and pd.notna(row.get("S_final")) else "—"
        lines.append(
            f'<tr><td>{row["scene_id"]}</td>'
            f'<td>{row["match_type"]}</td>'
            f"<td>{dt}</td>"
            f'<td>{s_raw}</td>'
            f'<td>{alpha}</td>'
            f'<td>{s_final}</td>'
            f'<td class="{cls}">{zone}</td></tr>\n'
        )

    lines.append("</table></body></html>\n")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(lines)
    logger.info("Wrote Δt report → %s", output_path)
    return output_path
