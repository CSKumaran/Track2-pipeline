"""Stage 7: Decoupled Scoring + Aggregates."""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io_utils import save_csv, save_json, cache_exists

logger = logging.getLogger(__name__)


def gaussian_score(delta_t: float, tau: float = 2.5) -> float:
    """S_temporal = 100 * exp(-0.5 * (d/τ)^2)."""
    if delta_t is None or pd.isna(delta_t):
        return None
    d = abs(delta_t)
    return 100.0 * np.exp(-0.5 * (d / tau) ** 2)


def zone_classification(delta_t: float) -> str:
    """Classify temporal gap into cognitive zones."""
    if delta_t is None or pd.isna(delta_t):
        return "No Match"
    d = abs(delta_t)
    if d <= 1.0:
        return "Optimal"
    elif d <= 3.0:
        return "Suboptimal"
    elif d <= 5.0:
        return "Disruptive"
    else:
        return "Unacceptable"


def overall_grade(mean_score: float) -> str:
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


def run_scoring(output_dir: str, cfg: Config) -> dict:
    """Compute scores for scenes and keywords. Returns paths to output files."""
    results = {}

    # Score scene-level alignment
    align_path = os.path.join(output_dir, "alignment_events.csv")
    if os.path.exists(align_path):
        align_df = pd.read_csv(align_path)
        if not align_df.empty:
            align_df["S_temporal"] = align_df["delta_t"].apply(
                lambda d: gaussian_score(d, cfg.SCORE_TAU)
            )
            align_df["zone"] = align_df["delta_t"].apply(zone_classification)
            save_csv(align_df, os.path.join(output_dir, "scores_per_scene.csv"))
            results["scores_per_scene.csv"] = os.path.join(output_dir, "scores_per_scene.csv")

    # Score keyword-level alignment
    kw_path = os.path.join(output_dir, "keyword_alignment.csv")
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        if not kw_df.empty:
            kw_df["S_temporal"] = kw_df["delta_t"].apply(
                lambda d: gaussian_score(d, cfg.SCORE_TAU)
            )
            kw_df["zone"] = kw_df["delta_t"].apply(zone_classification)
            save_csv(kw_df, os.path.join(output_dir, "keyword_alignment.csv"))

    # Importance-weighted scoring
    importance_path = os.path.join(output_dir, "pedagogical_importance.csv")
    if os.path.exists(importance_path) and os.path.exists(align_path):
        _compute_weighted_scores(output_dir, cfg)
        results["scores_weighted.csv"] = os.path.join(output_dir, "scores_weighted.csv")

    # Compute aggregates
    aggregates = _compute_aggregates(output_dir, cfg)
    save_json(aggregates, os.path.join(output_dir, "results.json"))
    results["results.json"] = os.path.join(output_dir, "results.json")

    return results


def _compute_weighted_scores(output_dir: str, cfg: Config):
    """Apply importance weights to scene scores."""
    align_df = pd.read_csv(os.path.join(output_dir, "scores_per_scene.csv"))
    imp_df = pd.read_csv(os.path.join(output_dir, "pedagogical_importance.csv"))

    # Merge on segment_id (scenes map to segments roughly by time)
    # Map each scene to closest segment by time
    if "segment_id" in imp_df.columns:
        seg_meta = pd.read_csv(os.path.join(output_dir, "segment_meta.csv"))
        imp_map = dict(zip(imp_df["segment_id"], imp_df["importance"]))

        # Find closest segment for each scene
        weights = []
        for _, scene in align_df.iterrows():
            t_kf = scene.get("t_keyframe", 0)
            if not seg_meta.empty:
                t_mids = pd.to_numeric(seg_meta["t_mid"], errors="coerce").values
                closest = int(np.argmin(np.abs(t_mids - t_kf)))
                seg_id = seg_meta.iloc[closest]["segment_id"]
                imp = imp_map.get(seg_id, 3)
            else:
                imp = 3
            weights.append(cfg.IMPORTANCE_WEIGHTS.get(imp, 1.0))

        align_df["importance_weight"] = weights
        align_df["S_weighted"] = align_df.apply(
            lambda r: r["S_temporal"] * r["importance_weight"]
            if pd.notna(r["S_temporal"]) else None, axis=1
        )

        # Priority for "top 5 to fix"
        align_df["priority"] = align_df.apply(
            lambda r: (100 - (r["S_temporal"] or 0)) * r["importance_weight"] * r["alpha"]
            if pd.notna(r.get("S_temporal")) else 0, axis=1
        )

        save_csv(align_df, os.path.join(output_dir, "scores_weighted.csv"))


def _compute_aggregates(output_dir: str, cfg: Config) -> dict:
    """Compute per-video aggregate metrics."""
    agg = {"pipeline_version": "2.1"}

    # Scene-level
    scene_scores_path = os.path.join(output_dir, "scores_per_scene.csv")
    if os.path.exists(scene_scores_path):
        df = pd.read_csv(scene_scores_path)
        matched = df[df["delta_t"].notna()]
        agg["scene_level"] = {
            "n_scenes": len(df),
            "n_content_scenes": int(df[df.get("match_type", pd.Series()) != "non_content"].shape[0])
                if "match_type" in df.columns else len(df),
            "n_matched": len(matched),
            "n_no_match": len(df) - len(matched),
            "mean_S_temporal": float(matched["S_temporal"].mean()) if len(matched) > 0 else None,
            "median_S_temporal": float(matched["S_temporal"].median()) if len(matched) > 0 else None,
            "mean_alpha": float(matched["alpha"].mean()) if "alpha" in matched.columns and len(matched) > 0 else None,
            "pct_high_confidence": float((matched["alpha"] >= 0.6).mean() * 100)
                if "alpha" in matched.columns and len(matched) > 0 else None,
            "mean_delta_t": float(matched["delta_t"].mean()) if len(matched) > 0 else None,
            "sd_delta_t": float(matched["delta_t"].std()) if len(matched) > 0 else None,
            "min_delta_t": float(matched["delta_t"].min()) if len(matched) > 0 else None,
            "max_delta_t": float(matched["delta_t"].max()) if len(matched) > 0 else None,
        }

        # Zone distribution
        if len(matched) > 0:
            zones = matched["zone"].value_counts(normalize=True) * 100
            agg["scene_level"]["pct_Optimal"] = float(zones.get("Optimal", 0))
            agg["scene_level"]["pct_Suboptimal"] = float(zones.get("Suboptimal", 0))
            agg["scene_level"]["pct_Disruptive"] = float(zones.get("Disruptive", 0))
            agg["scene_level"]["pct_Unacceptable"] = float(zones.get("Unacceptable", 0))

        # Overall grade
        if agg["scene_level"]["mean_S_temporal"] is not None:
            # Weighted by alpha
            if "alpha" in matched.columns:
                weighted_mean = float(
                    (matched["S_temporal"] * matched["alpha"]).sum() / matched["alpha"].sum()
                ) if matched["alpha"].sum() > 0 else agg["scene_level"]["mean_S_temporal"]
            else:
                weighted_mean = agg["scene_level"]["mean_S_temporal"]
            agg["overall_grade"] = overall_grade(weighted_mean)
            agg["overall_score"] = weighted_mean

    # Keyword-level
    kw_path = os.path.join(output_dir, "keyword_alignment.csv")
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        grounded = kw_df[kw_df["delta_t"].notna()] if "delta_t" in kw_df.columns else pd.DataFrame()
        agg["keyword_level"] = {
            "n_keywords_total": len(kw_df),
            "n_keywords_groundable": int(kw_df[kw_df.get("groundability", pd.Series()) != "LOW"].shape[0])
                if "groundability" in kw_df.columns else len(kw_df),
            "n_keywords_grounded": len(grounded),
            "n_not_visual": int(kw_df[kw_df.get("is_visual", pd.Series()) == False].shape[0])
                if "is_visual" in kw_df.columns else 0,
            "mean_S_temporal": float(grounded["S_temporal"].mean()) if len(grounded) > 0 and "S_temporal" in grounded.columns else None,
            "median_S_temporal": float(grounded["S_temporal"].median()) if len(grounded) > 0 and "S_temporal" in grounded.columns else None,
        }
        if "method" in kw_df.columns:
            method_dist = kw_df["method"].value_counts().to_dict()
            agg["keyword_level"]["grounding_method_distribution"] = method_dist

    return agg
