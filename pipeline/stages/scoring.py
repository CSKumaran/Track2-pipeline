"""Stage 5 – Temporal contiguity scoring: S(Δt), α, S_final, aggregates."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import pandas as pd

from pipeline.utils.io_utils import safe_write_csv
from pipeline.utils.viz_reports import (
    generate_delta_t_report,
    generate_dashboard_report,
    generate_json_results,
)

if TYPE_CHECKING:
    from pipeline.config import Config

logger = logging.getLogger(__name__)


# ── Core functions ───────────────────────────────────────────────────

def compute_S_delta(delta_t: float) -> float:
    """Piecewise temporal contiguity score from |Δt| in seconds.

    |Δt| range     S
    ──────────     ───
    [0, 1]         100
    (1, 3]         100 − 15·(d−1)
    (3, 5]         70 − 35·(d−3)
    > 5            0
    """
    d = abs(delta_t)
    if d <= 1.0:
        return 100.0
    elif d <= 3.0:
        return 100.0 - 15.0 * (d - 1.0)
    elif d <= 5.0:
        return 70.0 - 35.0 * (d - 3.0)
    else:
        return 0.0


def zone_from_delta(delta_t: float) -> str:
    """Map |Δt| to a qualitative zone label."""
    d = abs(delta_t)
    if d <= 1.0:
        return "Optimal"
    elif d <= 3.0:
        return "Suboptimal"
    elif d <= 5.0:
        return "Disruptive"
    else:
        return "Unacceptable"


def alpha_from_similarity(
    sim: float,
    low: float = 0.30,
    high: float = 0.80,
) -> float:
    """Map cosine similarity to semantic weight α ∈ [0, 1].

    Linear mapping: sim ≤ low → 0, sim ≥ high → 1, linear in between.
    """
    if sim <= low:
        return 0.0
    if sim >= high:
        return 1.0
    return (sim - low) / (high - low)


def alpha_from_clip(
    sim: float,
    low: float = 0.20,
    high: float = 0.60,
) -> float:
    """Map CLIP similarity to α ∈ [0.5, 1.0].

    CLIP is more reliable than VLM semantic, so floor at 0.5.
    Linear mapping: sim ≤ low → 0.5, sim ≥ high → 1.0.
    """
    if sim <= low:
        return 0.5
    if sim >= high:
        return 1.0
    return 0.5 + 0.5 * (sim - low) / (high - low)


def rate_alignment_with_vlm(
    scene_frames: list[str],
    narration_text: str,
) -> float:
    """Placeholder: obtain α from a VLM on a 0–1 scale.

    Future implementation would pass the scene frames and the matched
    narration text to a VLM and ask it to rate how well the narration
    explains what is shown (0 = no relation, 1 = perfect match).
    """
    raise NotImplementedError(
        "VLM-based alignment rating is not yet implemented. "
        "Using similarity-based alpha_from_similarity() instead."
    )


# ── Main scoring function ───────────────────────────────────────────

def compute_scores_for_alignments(
    alignment_df: pd.DataFrame,
    config: "Config",
) -> pd.DataFrame:
    """Compute S_raw, α, S_final, zone_label, failure_code per scene.

    Updated to work with the new alignment format:
    - match_type = "matched" → score using delta_t and similarity
    - match_type = "no_match" → score = 0, flagged

    Parameters
    ----------
    alignment_df : pd.DataFrame
        Output of ``align_scenes_to_narration`` (revised).
        Must have: threshold, scene_id, t_vis, match_type, t_narr,
        delta_t, sim_segment, sim_words.
    config : Config
        Pipeline configuration (alpha mapping bounds).

    Returns
    -------
    pd.DataFrame
        Input columns plus: S_raw, alpha, S_final, zone_label, failure_code.
    """
    rows: list[dict] = []

    for _, r in alignment_df.iterrows():
        match_type = r["match_type"]
        delta_t = r.get("delta_t")

        match_track = r.get("match_track")

        if match_type == "non_content":
            # Non-instructional frame — excluded from scoring
            S_raw = None
            alpha = None
            S_final = None
            zone_label = "Non-content"
            failure_code = "NON_CONTENT"
        elif match_type == "matched" and delta_t is not None:
            S_raw = compute_S_delta(delta_t)

            # Alpha depends on match track
            if match_track == "word_exact":
                # Exact word match → high confidence
                alpha = 1.0
            elif match_track == "clip_vision":
                # CLIP vision match → alpha from CLIP similarity
                clip_sim = r.get("clip_sim", 0.0)
                if clip_sim is None or pd.isna(clip_sim):
                    clip_sim = 0.0
                alpha = alpha_from_clip(
                    float(clip_sim),
                    getattr(config, "CLIP_ALPHA_LOW", 0.20),
                    getattr(config, "CLIP_ALPHA_HIGH", 0.60),
                )
            else:
                # Semantic match → alpha from similarity as before
                sim_for_alpha = r.get("sim_words")
                if sim_for_alpha is None or pd.isna(sim_for_alpha):
                    sim_for_alpha = r.get("sim_segment", 0.0)
                alpha = alpha_from_similarity(
                    float(sim_for_alpha), config.ALPHA_SIM_LOW, config.ALPHA_SIM_HIGH
                )

            S_final = S_raw * (0.5 + 0.5 * alpha)
            zone_label = zone_from_delta(delta_t)
            failure_code = ""
        else:  # "no_match"
            S_raw = 0.0
            alpha = 0.0
            S_final = 0.0
            zone_label = "Unacceptable"
            failure_code = "NO_MATCH"

        rows.append({
            "threshold": r["threshold"],
            "scene_id": r["scene_id"],
            "t_vis": r["t_vis"],
            "match_type": match_type,
            "match_track": match_track,
            "best_segment_id": r.get("best_segment_id"),
            "best_word_window": r.get("best_word_window"),
            "t_narr": r.get("t_narr"),
            "delta_t": delta_t,
            "sim_segment": r.get("sim_segment"),
            "sim_words": r.get("sim_words"),
            "n_word_matches": r.get("n_word_matches"),
            "trackA_delta_t": r.get("trackA_delta_t"),
            "clip_sim": r.get("clip_sim"),
            "clip_word_window": r.get("clip_word_window"),
            "trackB_clip_delta_t": r.get("trackB_clip_delta_t"),
            "trackC_delta_t": r.get("trackC_delta_t"),
            "scene_type": r.get("scene_type"),
            "scene_type_conf": r.get("scene_type_conf"),
            "S_raw": round(S_raw, 2) if S_raw is not None else None,
            "alpha": round(alpha, 4) if alpha is not None else None,
            "S_final": round(S_final, 2) if S_final is not None else None,
            "zone_label": zone_label,
            "failure_code": failure_code,
        })

    return pd.DataFrame(rows)


# ── Per-video aggregates ─────────────────────────────────────────────

def compute_video_aggregates(
    scores_df: pd.DataFrame,
    video_id: str,
) -> pd.DataFrame:
    """Compute per-video, per-threshold aggregate statistics."""
    agg_rows: list[dict] = []

    for threshold, group in scores_df.groupby("threshold"):
        n_total = len(group)
        non_content = group[group["match_type"] == "non_content"]
        n_non_content = len(non_content)

        # Scorable = everything except non_content
        scorable = group[group["match_type"] != "non_content"]
        n_scorable = len(scorable)
        matched = group[group["match_type"] == "matched"]
        no_match = group[group["match_type"] == "no_match"]

        # Δt stats for matched scenes only
        dt_mean = dt_sd = dt_min = dt_max = None
        if len(matched) > 0:
            dt_vals = matched["delta_t"].astype(float)
            dt_mean = round(dt_vals.mean(), 2)
            dt_sd = round(dt_vals.std(), 2)
            dt_min = round(dt_vals.min(), 2)
            dt_max = round(dt_vals.max(), 2)

        # Mean/median only from scorable scenes (non-None S_final)
        scorable_scores = scorable["S_final"].dropna()
        mean_sf = round(scorable_scores.mean(), 2) if len(scorable_scores) > 0 else 0.0
        median_sf = round(scorable_scores.median(), 2) if len(scorable_scores) > 0 else 0.0

        # Zone percentages based on scorable scenes only
        n_denom = max(n_scorable, 1)  # avoid division by zero
        zone_counts = scorable["zone_label"].value_counts()

        agg_rows.append({
            "video_id": video_id,
            "threshold": threshold,
            "n_scenes": n_total,
            "n_content": n_scorable,
            "n_non_content": n_non_content,
            "n_matched": len(matched),
            "n_no_match": len(no_match),
            "mean_S_final": mean_sf,
            "median_S_final": median_sf,
            "mean_delta_t": dt_mean,
            "sd_delta_t": dt_sd,
            "min_delta_t": dt_min,
            "max_delta_t": dt_max,
            "pct_Optimal": round(100 * zone_counts.get("Optimal", 0) / n_denom, 1),
            "pct_Suboptimal": round(100 * zone_counts.get("Suboptimal", 0) / n_denom, 1),
            "pct_Disruptive": round(100 * zone_counts.get("Disruptive", 0) / n_denom, 1),
            "pct_Unacceptable": round(100 * zone_counts.get("Unacceptable", 0) / n_denom, 1),
            "pct_NO_MATCH": round(100 * len(no_match) / n_denom, 1),
        })

    return pd.DataFrame(agg_rows)


# ── Save helpers ─────────────────────────────────────────────────────

def save_scoring_results(
    scores_df: pd.DataFrame,
    video_agg_df: pd.DataFrame,
    output_dir: str,
    video_name: str = "",
    vlm_mode: str = "ollama",
    concept_texts: dict | None = None,
) -> None:
    """Write per-scene and per-video CSVs + dashboard HTML + JSON, per threshold."""
    if not video_name:
        video_name = os.path.basename(output_dir)

    for threshold, group in scores_df.groupby("threshold"):
        # CSV outputs
        safe_write_csv(
            group,
            os.path.join(output_dir, f"scores_per_scene_threshold_{threshold}.csv"),
        )

        # Get aggregate dict for this threshold
        agg_row = video_agg_df[video_agg_df["threshold"] == threshold]
        if len(agg_row) > 0:
            agg_dict = agg_row.iloc[0].to_dict()
        else:
            agg_dict = {"n_scenes": len(group), "mean_S_final": 0}

        # Determine frames directory
        frames_dir = os.path.join(output_dir, f"frames_threshold_{threshold}")

        # Dashboard HTML report
        generate_dashboard_report(
            scores_df=group,
            video_agg=agg_dict,
            video_name=video_name,
            threshold=threshold,
            output_path=os.path.join(output_dir, f"report_dashboard_threshold_{threshold}.html"),
            vlm_mode=vlm_mode,
            concept_texts=concept_texts,
            frames_dir=frames_dir,
        )

        # JSON results
        generate_json_results(
            scores_df=group,
            video_agg=agg_dict,
            video_name=video_name,
            threshold=threshold,
            output_path=os.path.join(output_dir, f"results_threshold_{threshold}.json"),
            vlm_mode=vlm_mode,
            concept_texts=concept_texts,
        )

        # Legacy simple HTML report (kept for backward compat)
        generate_delta_t_report(
            group,
            os.path.join(output_dir, f"report_scores_threshold_{threshold}.html"),
        )

    for threshold, group in video_agg_df.groupby("threshold"):
        safe_write_csv(
            group,
            os.path.join(output_dir, f"scores_per_video_threshold_{threshold}.csv"),
        )
