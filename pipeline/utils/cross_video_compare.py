"""Cross-video scene alignment and delay-point detection.

Post-processing tool that takes pipeline outputs from multiple related videos
(e.g., A0/A1/A3/A5 delay experiment) and produces a unified comparison by
matching scenes across videos using OCR content fingerprints + visual SSIM.

Usage:
    python -m pipeline.utils.cross_video_compare \
        --reference A0 --compare A1 A3 A5 \
        --output-dir outputs/delay_comparison --threshold 0.95

    # With visual frame matching (requires video files):
    python -m pipeline.utils.cross_video_compare \
        --reference A0 --compare A1 A3 A5 \
        --video-dir . --threshold 0.95
"""
from __future__ import annotations

import argparse
import logging
import os
import re
from typing import Optional

import cv2
import pandas as pd
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────────────

def _normalize_words(text: str) -> set[str]:
    """Normalize OCR/text to a set of lowercase alphanumeric words (3+ chars)."""
    if not text or not isinstance(text, str) or text.strip() == "nan":
        return set()
    tokens = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
    return {w for w in tokens if len(w) >= 3}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


# ── visual frame matching ────────────────────────────────────────────────────

def _extract_frame(cap: cv2.VideoCapture, t: float,
                   size: tuple[int, int] = (320, 240)) -> np.ndarray | None:
    """Extract a single grayscale frame at time t (seconds)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size)


def find_visual_delay(
    ref_cap: cv2.VideoCapture,
    tgt_cap: cv2.VideoCapture,
    t_vis: float,
    search_window: float = 10.0,
    step: float = 0.1,
) -> dict:
    """Find when the reference keyframe at t_vis appears in the target video.

    Extracts the reference keyframe, then searches the target video in
    [t_vis - search_window, t_vis + search_window] for the SSIM peak.

    Returns dict with best_time, best_ssim, visual_delay.
    """
    ref_frame = _extract_frame(ref_cap, t_vis)
    if ref_frame is None:
        return {"best_time": t_vis, "best_ssim": 0.0, "visual_delay": 0.0}

    t_start = max(0, t_vis - search_window)
    t_end = t_vis + search_window
    search_times = np.arange(t_start, t_end, step)

    best_t = t_vis
    best_s = -1.0

    for t in search_times:
        frame = _extract_frame(tgt_cap, t)
        if frame is None:
            continue
        s = ssim(ref_frame, frame)
        if s > best_s:
            best_s = s
            best_t = t

    return {
        "best_time": round(best_t, 2),
        "best_ssim": round(best_s, 4),
        "visual_delay": round(best_t - t_vis, 2),
    }


def compute_visual_delays(
    comparison_df: pd.DataFrame,
    ref_name: str,
    comp_names: list[str],
    video_paths: dict[str, str],
    search_window: float = 10.0,
) -> pd.DataFrame:
    """Add visual delay columns to comparison table.

    For each matched scene, searches target videos for where the reference
    keyframe appears. Adds {video}_visual_delay and {video}_visual_ssim columns.
    """
    df = comparison_df.copy()
    all_videos = [ref_name] + comp_names

    # Open video captures
    caps = {}
    for vname, vpath in video_paths.items():
        caps[vname] = cv2.VideoCapture(vpath)
        if not caps[vname].isOpened():
            logger.warning("Could not open video: %s", vpath)

    ref_cap = caps.get(ref_name)
    if ref_cap is None or not ref_cap.isOpened():
        logger.error("Reference video not available for visual matching")
        return df

    # Initialize columns
    for cn in comp_names:
        df[f"{cn}_visual_delay"] = 0.0
        df[f"{cn}_visual_ssim"] = 0.0

    for idx, row in df.iterrows():
        ref_tvis = row.get(f"{ref_name}_tvis")
        if pd.isna(ref_tvis):
            continue

        for cn in comp_names:
            tgt_cap = caps.get(cn)
            if tgt_cap is None or not tgt_cap.isOpened():
                continue

            result = find_visual_delay(
                ref_cap, tgt_cap, ref_tvis,
                search_window=search_window,
            )
            df.at[idx, f"{cn}_visual_delay"] = result["visual_delay"]
            df.at[idx, f"{cn}_visual_ssim"] = result["best_ssim"]

        if idx % 5 == 0:
            logger.info("  Visual matching: %d/%d scenes done", idx + 1, len(df))

    # Compute visual delay spread (range across comparison videos)
    vd_cols = [f"{cn}_visual_delay" for cn in comp_names]
    if vd_cols:
        df["visual_delay_spread"] = df[vd_cols].max(axis=1) - df[vd_cols].min(axis=1)
    else:
        df["visual_delay_spread"] = 0.0

    # Release captures
    for cap in caps.values():
        cap.release()

    return df


# ── core matching ────────────────────────────────────────────────────────────

def load_video_data(video_name: str, outputs_root: str, threshold: float
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load scenes CSV and alignment CSV for a video."""
    vid_dir = os.path.join(outputs_root, video_name)
    t_str = f"{threshold:.2f}" if threshold != int(threshold) else str(int(threshold))

    scenes_path = os.path.join(vid_dir, f"scenes_threshold_{t_str}.csv")
    align_path = os.path.join(vid_dir, f"alignment_events_threshold_{t_str}.csv")

    if not os.path.exists(scenes_path):
        # Try alternate formatting
        for fname in os.listdir(vid_dir):
            if fname.startswith("scenes_threshold_") and fname.endswith(".csv"):
                scenes_path = os.path.join(vid_dir, fname)
                break
    if not os.path.exists(align_path):
        for fname in os.listdir(vid_dir):
            if fname.startswith("alignment_events_threshold_") and fname.endswith(".csv"):
                align_path = os.path.join(vid_dir, fname)
                break

    scenes_df = pd.read_csv(scenes_path)
    align_df = pd.read_csv(align_path)
    return scenes_df, align_df


def build_fingerprints(scenes_df: pd.DataFrame) -> list[dict]:
    """Build content fingerprint for each scene."""
    fingerprints = []
    for _, row in scenes_df.iterrows():
        ocr_text = str(row.get("ocr_text", ""))
        new_words = str(row.get("new_ocr_words", ""))
        # Combine both for fingerprint
        combined = f"{ocr_text} {new_words}"
        word_set = _normalize_words(combined)
        fingerprints.append({
            "scene_id": int(row["scene_id"]),
            "t_vis": float(row["t_vis"]),
            "word_set": word_set,
            "ocr_text": ocr_text[:80],
            "new_words": new_words[:80],
        })
    return fingerprints


def _match_score(scene_a: dict, scene_b: dict, max_time_gap: float = 20.0) -> float:
    """Score how well two scenes match, combining content + temporal proximity.

    Strategy: temporal proximity is PRIMARY, content similarity is SECONDARY.
    This prevents recurring text (e.g., a repeated paragraph) from matching
    scenes at wildly different timestamps.

    For scenes with text: Score = jaccard * temporal_weight
    For scenes without text: Score = temporal_weight * 0.5 (if within tight window)
    """
    dt = abs(scene_a["t_vis"] - scene_b["t_vis"])
    sigma = max_time_gap / 2.0
    temporal_w = np.exp(-0.5 * (dt / sigma) ** 2) if sigma > 0 else (1.0 if dt == 0 else 0.0)

    both_empty = (not scene_a["word_set"]) and (not scene_b["word_set"])
    jacc = _jaccard(scene_a["word_set"], scene_b["word_set"])

    if both_empty:
        # No text in either scene → match purely by temporal proximity
        # Use tight window (±5s) to avoid false matches
        if dt <= 5.0:
            return 0.5 * temporal_w  # lower score than text matches
        return 0.0

    if jacc < 0.1:
        return 0.0

    return jacc * temporal_w


def match_scenes_across_videos(
    ref_fps: list[dict],
    comp_fps_dict: dict[str, list[dict]],
    jaccard_threshold: float = 0.1,
    max_time_gap: float = 20.0,
) -> list[dict]:
    """Match scenes across videos using content + temporal proximity.

    Uses a Gaussian-weighted scoring: content similarity (Jaccard) is
    multiplied by temporal proximity weight. This ensures recurring text
    at different timestamps doesn't create false matches.

    Returns a list of content groups, each with scene IDs and t_vis per video.
    """
    # Build candidate score matrix for each (ref_scene, comp_video, comp_scene)
    # Then use greedy best-first matching
    used = set()  # (video_key, scene_id) pairs already assigned
    groups = []

    # Phase 1: Match reference scenes to comparison videos
    # Build all candidate pairs with scores
    candidates = []
    for ref_scene in ref_fps:
        for vname, fps in comp_fps_dict.items():
            for comp_scene in fps:
                score = _match_score(ref_scene, comp_scene, max_time_gap)
                if score > 0.05:
                    candidates.append({
                        "ref_sid": ref_scene["scene_id"],
                        "comp_video": vname,
                        "comp_sid": comp_scene["scene_id"],
                        "score": score,
                        "ref_scene": ref_scene,
                        "comp_scene": comp_scene,
                    })

    # Sort by score descending → greedy best-first
    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Track which ref scenes have groups
    ref_groups = {}  # ref_scene_id → group dict

    for cand in candidates:
        ref_sid = cand["ref_sid"]
        vname = cand["comp_video"]
        comp_sid = cand["comp_sid"]

        if (vname, comp_sid) in used:
            continue

        # Create or extend group for this ref scene
        if ref_sid not in ref_groups:
            ref_groups[ref_sid] = {"ref": cand["ref_scene"]}
            used.add(("ref", ref_sid))

        if vname not in ref_groups[ref_sid]:
            ref_groups[ref_sid][vname] = cand["comp_scene"]
            used.add((vname, comp_sid))

    groups = list(ref_groups.values())

    # Phase 2: Add unmatched ref scenes as singleton groups
    for ref_scene in ref_fps:
        if ref_scene["scene_id"] not in ref_groups:
            groups.append({"ref": ref_scene})
            used.add(("ref", ref_scene["scene_id"]))

    # Phase 3: Group unmatched comparison scenes among themselves
    for vname, fps in comp_fps_dict.items():
        for comp_scene in fps:
            if (vname, comp_scene["scene_id"]) in used:
                continue

            group = {vname: comp_scene}
            used.add((vname, comp_scene["scene_id"]))

            # Try matching with other comparison videos' unmatched scenes
            for other_vname, other_fps in comp_fps_dict.items():
                if other_vname == vname or other_vname in group:
                    continue
                best_match = None
                best_score = -1
                for other_scene in other_fps:
                    if (other_vname, other_scene["scene_id"]) in used:
                        continue
                    score = _match_score(comp_scene, other_scene, max_time_gap)
                    if score > best_score and score > 0.05:
                        best_score = score
                        best_match = other_scene
                if best_match is not None:
                    group[other_vname] = best_match
                    used.add((other_vname, best_match["scene_id"]))

            groups.append(group)

    return groups


def build_comparison_table(
    groups: list[dict],
    align_dfs: dict[str, pd.DataFrame],
    ref_name: str,
    comp_names: list[str],
) -> pd.DataFrame:
    """Build the unified comparison table from matched groups."""
    all_video_names = [ref_name] + comp_names
    rows = []

    for gid, group in enumerate(groups):
        row = {"content_id": gid}

        # Collect content description
        words_all = set()
        for vkey, scene in group.items():
            words_all |= scene.get("word_set", set())
        row["content_words"] = " ".join(sorted(words_all)[:8]) if words_all else "(no text)"

        # Per-video data
        tvis_values = []
        dt_values = []
        for vname in all_video_names:
            vkey = "ref" if vname == ref_name else vname
            if vkey in group:
                scene = group[vkey]
                sid = scene["scene_id"]
                t_vis = scene["t_vis"]
                row[f"{vname}_sid"] = sid
                row[f"{vname}_tvis"] = round(t_vis, 2)
                tvis_values.append(t_vis)

                # Get delta_t from alignment
                adf = align_dfs[vname]
                match = adf[adf["scene_id"] == sid]
                if not match.empty:
                    dt = float(match.iloc[0]["delta_t"])
                    row[f"{vname}_dt"] = round(dt, 2)
                    dt_values.append(dt)
                    row[f"{vname}_track"] = match.iloc[0].get("match_track", "?")
                else:
                    row[f"{vname}_dt"] = None
                    row[f"{vname}_track"] = "?"
            else:
                row[f"{vname}_sid"] = None
                row[f"{vname}_tvis"] = None
                row[f"{vname}_dt"] = None
                row[f"{vname}_track"] = "missing"

        # Compute spread metrics
        if len(tvis_values) >= 2:
            row["tvis_shift"] = round(max(tvis_values) - min(tvis_values), 2)
        else:
            row["tvis_shift"] = 0.0

        if len(dt_values) >= 2:
            row["dt_spread"] = round(max(dt_values) - min(dt_values), 2)
        else:
            row["dt_spread"] = 0.0

        # Present in how many videos
        row["n_videos"] = sum(1 for vn in all_video_names
                              if ("ref" if vn == ref_name else vn) in group)

        rows.append(row)

    df = pd.DataFrame(rows)
    # Sort by t_vis of reference, then by first available t_vis
    sort_col = f"{ref_name}_tvis"
    if sort_col in df.columns:
        df = df.sort_values(sort_col, na_position="last").reset_index(drop=True)
        df["content_id"] = range(len(df))
    return df


def identify_delay_points(
    comparison_df: pd.DataFrame,
    tvis_shift_threshold: float = 0.5,
    visual_delay_threshold: float = 0.5,
    top_n: int = 10,
) -> pd.DataFrame:
    """Identify likely delay insertion points by t_vis shift or visual delay.

    A scene is flagged as a delay point if either:
    - tvis_shift > tvis_shift_threshold (scene boundary moved), OR
    - visual_delay_spread > visual_delay_threshold (same boundary but content shifted)
    """
    df = comparison_df.copy()

    has_visual = "visual_delay_spread" in df.columns
    if has_visual:
        df["is_delay_point"] = (
            (df["tvis_shift"] > tvis_shift_threshold) |
            (df["visual_delay_spread"] > visual_delay_threshold)
        )
        sort_col = "visual_delay_spread"
    else:
        df["is_delay_point"] = df["tvis_shift"] > tvis_shift_threshold
        sort_col = "tvis_shift"

    delay_df = df[df["is_delay_point"]].sort_values(sort_col, ascending=False)
    if len(delay_df) > top_n:
        delay_df = delay_df.head(top_n)
    return delay_df


def generate_html_report(
    comparison_df: pd.DataFrame,
    delay_df: pd.DataFrame,
    ref_name: str,
    comp_names: list[str],
    output_path: str,
):
    """Generate an HTML comparison report."""
    all_videos = [ref_name] + comp_names

    # Summary stats
    n_total = len(comparison_df)
    n_delay = len(delay_df)
    n_stable = n_total - len(comparison_df[comparison_df["tvis_shift"] > 0.5])

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Cross-Video Delay Comparison</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #3f51b5; padding-bottom: 5px; }}
  .summary {{ background: white; padding: 15px; border-radius: 8px; margin: 15px 0;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .summary .stat {{ display: inline-block; margin: 0 25px; text-align: center; }}
  .stat .value {{ font-size: 2em; font-weight: bold; color: #1a237e; }}
  .stat .label {{ font-size: 0.9em; color: #666; }}
  table {{ border-collapse: collapse; width: 100%; background: white;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); border-radius: 8px; overflow: hidden; }}
  th {{ background: #1a237e; color: white; padding: 10px 8px; font-size: 0.85em; }}
  td {{ padding: 8px; border-bottom: 1px solid #e0e0e0; font-size: 0.85em; text-align: center; }}
  tr:hover {{ background: #e8eaf6; }}
  .delay-row {{ background: #fff3e0 !important; }}
  .delay-row:hover {{ background: #ffe0b2 !important; }}
  .stable {{ color: #2e7d32; font-weight: bold; }}
  .shifted {{ color: #e65100; font-weight: bold; }}
  .missing {{ color: #999; font-style: italic; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
            font-size: 0.8em; font-weight: bold; }}
  .badge-delay {{ background: #ff9800; color: white; }}
  .badge-stable {{ background: #4caf50; color: white; }}
</style>
</head><body>
<h1>Cross-Video Delay Comparison</h1>
<p>Reference: <strong>{ref_name}</strong> | Compared: <strong>{', '.join(comp_names)}</strong></p>

<div class="summary">
  <div class="stat"><div class="value">{n_total}</div><div class="label">Total Content Groups</div></div>
  <div class="stat"><div class="value">{n_delay}</div><div class="label">Delay Points Detected</div></div>
  <div class="stat"><div class="value">{n_stable}</div><div class="label">Stable Scenes</div></div>
</div>

<h2>Delay Points Detected (sorted by t_vis shift)</h2>
"""
    has_visual = "visual_delay_spread" in delay_df.columns if len(delay_df) > 0 else False

    if len(delay_df) > 0:
        html += "<table><tr><th>#</th><th>Content</th>"
        for v in all_videos:
            html += f"<th>{v} t_vis</th><th>{v} Δt</th>"
            if has_visual and v != ref_name:
                html += f"<th>{v} VisDelay</th>"
        html += "<th>t_vis Shift</th><th>Δt Spread</th>"
        if has_visual:
            html += "<th>Visual Spread</th>"
        html += "</tr>\n"
        for _, row in delay_df.iterrows():
            html += f'<tr class="delay-row"><td>{int(row["content_id"])}</td>'
            html += f'<td>{row["content_words"]}</td>'
            for v in all_videos:
                tvis = row.get(f"{v}_tvis")
                dt = row.get(f"{v}_dt")
                tvis_s = f"{tvis:.1f}" if pd.notna(tvis) else '<span class="missing">—</span>'
                dt_s = f"{dt:.2f}" if pd.notna(dt) else '<span class="missing">—</span>'
                html += f"<td>{tvis_s}</td><td>{dt_s}</td>"
                if has_visual and v != ref_name:
                    vd = row.get(f"{v}_visual_delay", 0)
                    vd_cls = "shifted" if abs(vd) > 0.5 else "stable"
                    html += f'<td class="{vd_cls}">{vd:+.1f}s</td>'
            html += f'<td class="shifted">{row["tvis_shift"]:.1f}s</td>'
            html += f'<td>{row["dt_spread"]:.1f}s</td>'
            if has_visual:
                vs = row.get("visual_delay_spread", 0)
                vs_cls = "shifted" if vs > 0.5 else "stable"
                html += f'<td class="{vs_cls}">{vs:.1f}s</td>'
            html += "</tr>\n"
        html += "</table>\n"
    else:
        html += "<p>No delay points detected.</p>\n"

    has_visual_full = "visual_delay_spread" in comparison_df.columns

    html += "<h2>Full Comparison Table</h2>\n"
    html += "<table><tr><th>#</th><th>Content</th><th>Videos</th>"
    for v in all_videos:
        html += f"<th>{v} t_vis</th><th>{v} Δt</th><th>{v} Track</th>"
        if has_visual_full and v != ref_name:
            html += f"<th>{v} VisDelay</th>"
    html += "<th>t_vis Shift</th><th>Δt Spread</th>"
    if has_visual_full:
        html += "<th>Vis Spread</th>"
    html += "<th>Status</th></tr>\n"

    delay_ids = set(delay_df["content_id"]) if len(delay_df) > 0 else set()
    for _, row in comparison_df.iterrows():
        cid = int(row["content_id"])
        is_delay = cid in delay_ids
        cls = ' class="delay-row"' if is_delay else ""
        html += f"<tr{cls}><td>{cid}</td>"
        html += f'<td style="text-align:left">{row["content_words"]}</td>'
        html += f'<td>{int(row["n_videos"])}/{len(all_videos)}</td>'
        for v in all_videos:
            tvis = row.get(f"{v}_tvis")
            dt = row.get(f"{v}_dt")
            track = row.get(f"{v}_track", "?")
            tvis_s = f"{tvis:.1f}" if pd.notna(tvis) else '<span class="missing">—</span>'
            dt_s = f"{dt:.2f}" if pd.notna(dt) else '<span class="missing">—</span>'
            track_s = track if track != "missing" else '<span class="missing">—</span>'
            html += f"<td>{tvis_s}</td><td>{dt_s}</td><td>{track_s}</td>"
            if has_visual_full and v != ref_name:
                vd = row.get(f"{v}_visual_delay", 0)
                vd_cls = "shifted" if abs(vd) > 0.5 else "stable"
                html += f'<td class="{vd_cls}">{vd:+.1f}s</td>'

        shift = row["tvis_shift"]
        spread = row["dt_spread"]
        shift_cls = "shifted" if shift > 0.5 else "stable"
        status = '<span class="badge badge-delay">DELAY</span>' if is_delay else '<span class="badge badge-stable">STABLE</span>'
        html += f'<td class="{shift_cls}">{shift:.1f}s</td>'
        html += f"<td>{spread:.1f}s</td>"
        if has_visual_full:
            vs = row.get("visual_delay_spread", 0)
            vs_cls = "shifted" if vs > 0.5 else "stable"
            html += f'<td class="{vs_cls}">{vs:.1f}s</td>'
        html += f"<td>{status}</td></tr>\n"

    html += "</table>\n"

    # Separate scoring
    stable_rows = comparison_df[~comparison_df["content_id"].isin(delay_ids)]
    if len(stable_rows) > 0:
        html += "<h2>Consistency Check: Stable Scenes</h2>\n"
        html += "<p>These scenes should have identical Δt across all videos (no delay introduced):</p>\n"
        all_video_names = [ref_name] + comp_names
        for v in all_video_names:
            col = f"{v}_dt"
            vals = stable_rows[col].dropna()
            if len(vals) > 0:
                html += f"<p><strong>{v}</strong>: mean Δt = {vals.mean():.2f}s, SD = {vals.std():.2f}s</p>\n"

    html += "</body></html>"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote comparison report -> %s", output_path)


# ── main ─────────────────────────────────────────────────────────────────────

def run_comparison(
    ref_name: str,
    comp_names: list[str],
    outputs_root: str = "outputs",
    output_dir: str = "outputs/delay_comparison",
    threshold: float = 0.95,
    video_dir: str | None = None,
):
    """Run full cross-video comparison.

    Args:
        video_dir: Directory containing video files (e.g., A0.mp4, A1.mp4).
                   If provided, enables visual frame matching via SSIM.
    """
    all_videos = [ref_name] + comp_names

    # Load data
    scenes_dfs = {}
    align_dfs = {}
    for vname in all_videos:
        s_df, a_df = load_video_data(vname, outputs_root, threshold)
        scenes_dfs[vname] = s_df
        align_dfs[vname] = a_df
        logger.info("Loaded %s: %d scenes, %d alignment events", vname, len(s_df), len(a_df))

    # Build fingerprints
    ref_fps = build_fingerprints(scenes_dfs[ref_name])
    comp_fps = {vn: build_fingerprints(scenes_dfs[vn]) for vn in comp_names}

    # Match scenes
    groups = match_scenes_across_videos(ref_fps, comp_fps)
    logger.info("Matched %d content groups across %d videos", len(groups), len(all_videos))

    # Build comparison table
    comparison_df = build_comparison_table(groups, align_dfs, ref_name, comp_names)

    # Visual frame matching (if video_dir provided)
    if video_dir:
        video_paths = {}
        for vname in all_videos:
            vpath = os.path.join(video_dir, f"{vname}.mp4")
            if os.path.exists(vpath):
                video_paths[vname] = vpath
            else:
                logger.warning("Video file not found: %s", vpath)

        if len(video_paths) >= 2 and ref_name in video_paths:
            logger.info("Running visual frame matching on %d videos...", len(video_paths))
            comparison_df = compute_visual_delays(
                comparison_df, ref_name, comp_names, video_paths,
            )
            logger.info("Visual matching complete.")
        else:
            logger.warning("Not enough video files for visual matching (need ref + at least 1 target)")

    # Identify delay points
    delay_df = identify_delay_points(comparison_df)
    logger.info("Identified %d delay points", len(delay_df))

    # Save CSVs
    os.makedirs(output_dir, exist_ok=True)
    comparison_df.to_csv(os.path.join(output_dir, "unified_comparison.csv"), index=False)
    if len(delay_df) > 0:
        delay_df.to_csv(os.path.join(output_dir, "delay_points.csv"), index=False)

    # Generate HTML report
    generate_html_report(
        comparison_df, delay_df, ref_name, comp_names,
        os.path.join(output_dir, "report_comparison.html"),
    )

    # Print summary to console
    print("\n" + "=" * 80)
    print("CROSS-VIDEO COMPARISON SUMMARY")
    print("=" * 80)
    print(f"Reference: {ref_name} | Compared: {', '.join(comp_names)}")
    print(f"Content groups: {len(comparison_df)}")
    print(f"Delay points detected: {len(delay_df)}")
    print()

    has_visual = "visual_delay_spread" in delay_df.columns if len(delay_df) > 0 else False

    if len(delay_df) > 0:
        print("DELAY POINTS:")
        for _, row in delay_df.iterrows():
            tvis_vals = {v: row.get(f"{v}_tvis") for v in all_videos}
            tvis_str = "  ".join(f"{v}={tvis_vals[v]:.1f}" if pd.notna(tvis_vals[v])
                                 else f"{v}=---" for v in all_videos)
            print(f"  #{int(row['content_id']):2d} [{row['content_words'][:40]}]")
            print(f"       t_vis: {tvis_str}  (shift={row['tvis_shift']:.1f}s)")
            dt_vals = {v: row.get(f"{v}_dt") for v in all_videos}
            dt_str = "  ".join(f"{v}={dt_vals[v]:.2f}" if pd.notna(dt_vals[v])
                               else f"{v}=---" for v in all_videos)
            print(f"       delta_t: {dt_str}  (spread={row['dt_spread']:.1f}s)")
            if has_visual:
                vd_str = "  ".join(
                    f"{v}={row.get(f'{v}_visual_delay', 0):+.1f}s"
                    for v in comp_names
                )
                vs = row.get("visual_delay_spread", 0)
                print(f"       visual_delay: {vd_str}  (spread={vs:.1f}s)")
        print()

    # Stable scene consistency
    delay_ids = set(delay_df["content_id"]) if len(delay_df) > 0 else set()
    stable = comparison_df[~comparison_df["content_id"].isin(delay_ids)]
    if len(stable) > 0:
        print("STABLE SCENES (should be consistent across videos):")
        for v in all_videos:
            vals = stable[f"{v}_dt"].dropna()
            if len(vals) > 0:
                print(f"  {v}: mean_dt={vals.mean():.2f}, SD={vals.std():.2f}, "
                      f"n={len(vals)} scenes")
    print("=" * 80)

    return comparison_df, delay_df


def main():
    parser = argparse.ArgumentParser(description="Cross-video scene comparison")
    parser.add_argument("--reference", required=True, help="Reference video name (e.g., A0)")
    parser.add_argument("--compare", nargs="+", required=True, help="Videos to compare (e.g., A1 A3 A5)")
    parser.add_argument("--outputs-root", default="outputs", help="Root outputs directory")
    parser.add_argument("--output-dir", default="outputs/delay_comparison", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.95, help="Scene threshold")
    parser.add_argument("--video-dir", default=None,
                        help="Directory containing video files for visual matching (e.g., . or videos/)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    run_comparison(args.reference, args.compare, args.outputs_root, args.output_dir,
                   args.threshold, video_dir=args.video_dir)


if __name__ == "__main__":
    main()
