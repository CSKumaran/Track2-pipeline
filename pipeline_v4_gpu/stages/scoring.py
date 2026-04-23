"""Stage 7: Asymmetric Temporal Contiguity Scoring [v4.0].

[v4.0] Key changes from v2.2:
  - Visual-persistence-verified asymmetric scoring (reviewer-defensible)
  - `visual_on_screen` column in keyword_scores.csv for audit trail

[v2.2] Foundation:
  - ASYMMETRIC scoring: delta_t ≤ 0 → Score = 100 (visual already present, no penalty)
  - Dual scoring: Gaussian + piecewise (v2.0-style) computed side-by-side
  - Per-SEGMENT scoring (transcript-first), not per-scene
  - Floor formula: S_floored = S_raw * (floor + (1-floor) * alpha)
  - Two-tier confidence reporting: HIGH (alpha ≥ 0.8) vs MEDIUM (alpha < 0.8)
  - Compressed importance weights for heuristic backend (0.8-1.2 range)

Scoring formulas:
  Gaussian:  S = 100 * exp(-0.5 * (delta_t / tau)^2)   tau=2.5s
  Piecewise: |d|≤1 → 100, |d|∈(1,3] → 100-15(d-1), |d|∈(3,5] → 70-35(d-3), >5 → 0

Asymmetric rule — visual persistence verification:
  The asymmetric rule is NOT "if visual appeared before narration, assume it's fine."
  It is: "if the visual is VERIFIED ON SCREEN at the moment of narration (t_narr),
  the student experiences simultaneous visual-verbal presentation → score 100."

  Verification method per match case:
  - Case A: keyword found in OCR union of the CURRENT scene at t_narr.
            The scene has not ended. Visual confirmed present. Score = 100.
  - Case D: progressive reveal within current scene. Keyword appeared mid-scene
            and scene is still active at t_narr. Visual confirmed present. Score = 100.
  - Case E: [v4.0] Gemini concept match. If matched scene contains t_narr
            (scene.t_start ≤ t_narr ≤ scene.t_end), visual is on screen. Score = 100.
            If matched scene does NOT contain t_narr, penalty via |delta_t|.
  - Cases B/C/G with negative delta_t: Should not normally occur.
            Safety net: apply Gaussian penalty using |delta_t|.

  Theoretical basis (Mayer, 2009):
  - Temporal Contiguity Principle: students learn better when corresponding
    words and pictures are presented simultaneously rather than successively.
  - Pre-Training Principle: presenting visual material before verbal explanation
    can be beneficial (visual primes schema construction).
  - Key distinction: our score of 100 for delta_t ≤ 0 only applies when the
    visual is STILL ON SCREEN (verified). A visual that appeared and disappeared
    before narration gets Case C treatment (penalty), not score 100.

Zone classification:
  delta_t ≤ 0.0  → Optimal (visual verified on screen at t_narr)
  0 < delta_t ≤ 1.0 → Optimal (near-simultaneous)
  1.0 < delta_t ≤ 3.0 → Suboptimal
  3.0 < delta_t ≤ 5.0 → Disruptive
  delta_t > 5.0 → Unacceptable

Outputs:
    keyword_scores.csv — per-keyword: S_gaussian, S_piecewise, zone, importance_weight,
                          visual_on_screen (True/False — verification audit trail)
    segment_scores.csv — per-segment: aggregated scores, zone distribution
    results.json — overall video metrics, grade, zone percentages
    stage7_scoring.json — diagnostics
"""

import logging
import os
import numpy as np
import pandas as pd

from ..config import Config
from ..utils.io_utils import save_csv, save_json, cache_exists

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring Functions
# ═══════════════════════════════════════════════════════════════════════════════

def gaussian_score(delta_t: float, tau: float = 2.5,
                   match_case: str = None,
                   visual_on_screen: bool = None,
                   dwell_threshold: float = 15.0,
                   dwell_tau: float = 30.0) -> float:
    """Gaussian temporal score. Asymmetric with visual-persistence verification.

    [v4.0] Uses `visual_on_screen` flag from alignment stage for principled
    asymmetric scoring. The flag is True when the alignment stage has verified
    (via OCR or scene boundaries) that the visual IS STILL PRESENT at t_narr.

    [v4.1] Dwell Time Decay: If visual has been on screen > dwell_threshold
    seconds before narration, apply Gaussian decay. This penalizes "animation
    first" designs where visuals sit 30-60s before narration — the student's
    working memory for the visual degrades (Baddeley, 2000).

    delta_t ≤ 0 + visual_on_screen=True + dwell ≤ threshold → Score = 100
    delta_t ≤ 0 + visual_on_screen=True + dwell > threshold → Decayed score
    delta_t ≤ 0 + visual_on_screen=False → Gaussian penalty on |delta_t|

    Backward compatibility: if visual_on_screen is None, fall back to
    match_case heuristic (A/D → True, others → False).
    """
    if delta_t is None or pd.isna(delta_t):
        return None

    # Determine visual persistence
    if visual_on_screen is None:
        # Backward compatibility: infer from match_case
        visual_on_screen = match_case in ("A", "D", "E_on_screen", None)

    if delta_t <= 0:
        if visual_on_screen:
            # [v4.1] Dwell time decay — penalize long-dwelling visuals
            dwell_time = abs(delta_t)
            if dwell_time <= dwell_threshold:
                return 100.0  # Truly simultaneous — no penalty
            excess = dwell_time - dwell_threshold
            return 100.0 * np.exp(-0.5 * (excess / dwell_tau) ** 2)
        else:
            # Visual appeared but is NOT verified on screen — penalize
            d = abs(delta_t)
            return 100.0 * np.exp(-0.5 * (d / tau) ** 2)
    return 100.0 * np.exp(-0.5 * (delta_t / tau) ** 2)


def piecewise_score(delta_t: float, match_case: str = None,
                    visual_on_screen: bool = None,
                    dwell_threshold: float = 15.0,
                    dwell_tau: float = 15.0) -> float:
    """V2.0-style piecewise linear score. Asymmetric with verification.

    |d|≤1 → 100
    |d|∈(1,3] → 100 - 15*(d-1)  [linear: 100→70]
    |d|∈(3,5] → 70 - 35*(d-3)   [linear: 70→0]
    |d|>5 → 0

    [v4.1] Dwell time decay applied when visual_on_screen=True + dwell > threshold.
    delta_t ≤ 0 + visual_on_screen=True + dwell ≤ threshold → 100.
    delta_t ≤ 0 + visual_on_screen=True + dwell > threshold → Gaussian decay.
    delta_t ≤ 0 + visual_on_screen=False → piecewise on |delta_t|.
    """
    if delta_t is None or pd.isna(delta_t):
        return None

    # Determine visual persistence
    if visual_on_screen is None:
        visual_on_screen = match_case in ("A", "D", "E_on_screen", None)

    if delta_t <= 0:
        if visual_on_screen:
            # [v4.1] Dwell time decay
            dwell_time = abs(delta_t)
            if dwell_time <= dwell_threshold:
                return 100.0
            excess = dwell_time - dwell_threshold
            return 100.0 * np.exp(-0.5 * (excess / dwell_tau) ** 2)
        else:
            d = abs(delta_t)
            if d <= 1.0:
                return 100.0
            elif d <= 3.0:
                return 100.0 - 15.0 * (d - 1.0)
            elif d <= 5.0:
                return 70.0 - 35.0 * (d - 3.0)
            else:
                return 0.0
    d = delta_t
    if d <= 1.0:
        return 100.0
    elif d <= 3.0:
        return 100.0 - 15.0 * (d - 1.0)
    elif d <= 5.0:
        return 70.0 - 35.0 * (d - 3.0)
    else:
        return 0.0


def zone_classification(delta_t: float, match_case: str = None,
                        visual_on_screen: bool = None,
                        dwell_threshold: float = 15.0,
                        dwell_tau: float = 15.0) -> str:
    """Classify temporal gap into cognitive zones. Asymmetric with verification.

    [v4.1] Dwell-aware zones: long-dwelling visuals may shift from Optimal
    to Suboptimal/Disruptive based on dwell decay score.
    """
    if delta_t is None or pd.isna(delta_t):
        return "No Match"

    # Determine visual persistence
    if visual_on_screen is None:
        visual_on_screen = match_case in ("A", "D", "E_on_screen", None)

    if delta_t <= 0:
        if visual_on_screen:
            # [v4.1] Dwell-aware zone classification
            dwell_time = abs(delta_t)
            if dwell_time <= dwell_threshold:
                return "Optimal"
            # Compute decay score to determine zone
            excess = dwell_time - dwell_threshold
            score = 100.0 * np.exp(-0.5 * (excess / dwell_tau) ** 2)
            if score >= 80:
                return "Optimal"
            elif score >= 60:
                return "Suboptimal"
            elif score >= 40:
                return "Disruptive"
            else:
                return "Unacceptable"
        else:
            d = abs(delta_t)
            if d <= 1.0:
                return "Optimal"
            elif d <= 3.0:
                return "Suboptimal"
            elif d <= 5.0:
                return "Disruptive"
            else:
                return "Unacceptable"
    if delta_t <= 1.0:
        return "Optimal"
    elif delta_t <= 3.0:
        return "Suboptimal"
    elif delta_t <= 5.0:
        return "Disruptive"
    else:
        return "Unacceptable"


def overall_grade(mean_score: float) -> str:
    """Map mean score to letter grade."""
    if mean_score is None or pd.isna(mean_score):
        return "N/A"
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


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 7 Entry Point
# ═══════════════════════════════════════════════════════════════════════════════

def run_stage7(output_dir: str, cfg: Config, diag=None) -> dict:
    """Compute temporal contiguity scores for all keywords and segments.

    Requires:
        keyword_alignment.csv (Stage 4)
        pedagogical_importance.csv (Stage 6, optional)
        keywords.csv (Stage 5, for metadata)
    """
    cache_files = ["keyword_scores.csv", "segment_scores.csv", "results.json"]
    if cache_exists(output_dir, cache_files):
        logger.info("Stage 7: cache hit, skipping scoring")
        return {f: os.path.join(output_dir, f) for f in cache_files}

    # Load keyword alignment from Stage 4
    kw_align_path = os.path.join(output_dir, "keyword_alignment.csv")
    if not os.path.exists(kw_align_path):
        logger.error("keyword_alignment.csv not found — run Stage 4 first")
        return {}

    kw_df = pd.read_csv(kw_align_path)
    if kw_df.empty:
        logger.warning("No keyword alignment data for scoring")
        save_csv(pd.DataFrame(), os.path.join(output_dir, "keyword_scores.csv"))
        save_csv(pd.DataFrame(), os.path.join(output_dir, "segment_scores.csv"))
        save_json({"pipeline_version": "4.0", "error": "no_data"},
                  os.path.join(output_dir, "results.json"))
        return {f: os.path.join(output_dir, f) for f in cache_files}

    tau = cfg.SCORE_TAU
    scoring_mode = cfg.SCORING_MODE
    floor_factor = cfg.PIECEWISE_FLOOR_FACTOR
    dwell_threshold = getattr(cfg, 'DWELL_THRESHOLD', 15.0)
    dwell_tau = getattr(cfg, 'DWELL_TAU', 15.0)
    logger.info("Scoring params: tau=%.1f, dwell_threshold=%.1f, dwell_tau=%.1f",
                tau, dwell_threshold, dwell_tau)

    # ── Step 1: Per-keyword scoring ──────────────────────────────────────────
    kw_df = _score_keywords(kw_df, tau, scoring_mode, floor_factor,
                            dwell_threshold, dwell_tau)

    # ── Step 2: Apply importance weights ─────────────────────────────────────
    kw_df = _apply_importance_weights(kw_df, output_dir, cfg)

    # Save keyword scores
    save_csv(kw_df, os.path.join(output_dir, "keyword_scores.csv"))
    logger.info("Saved keyword_scores.csv (%d keywords)", len(kw_df))

    # ── Step 3: Per-segment aggregation ──────────────────────────────────────
    seg_df = _aggregate_segments(kw_df, output_dir, cfg)
    save_csv(seg_df, os.path.join(output_dir, "segment_scores.csv"))
    logger.info("Saved segment_scores.csv (%d segments)", len(seg_df))

    # ── Step 4: Overall aggregates ───────────────────────────────────────────
    results = _compute_overall_results(kw_df, seg_df, cfg)
    save_json(results, os.path.join(output_dir, "results.json"))
    logger.info("Saved results.json — Overall: %.1f (%s)",
                results.get("overall_score", 0), results.get("overall_grade", "N/A"))

    # ── Step 5: Diagnostics ──────────────────────────────────────────────────
    if diag:
        _write_diagnostics(diag, kw_df, seg_df, results, cfg)

    # Summary log
    _log_summary(results)

    return {f: os.path.join(output_dir, f) for f in cache_files}


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1: Per-Keyword Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def _score_keywords(kw_df: pd.DataFrame, tau: float, scoring_mode: str,
                    floor_factor: float,
                    dwell_threshold: float = 15.0,
                    dwell_tau: float = 15.0) -> pd.DataFrame:
    """Add S_gaussian, S_piecewise, S_temporal, zone, visual_on_screen to each keyword.

    [v4.0] Adds `visual_on_screen` column — True when the alignment stage has
    verified (via OCR scene union or scene boundaries) that the visual content
    is STILL PRESENT on screen at the moment of narration (t_narr).

    [v4.1] Dwell Time Decay: visuals on screen > dwell_threshold seconds
    before narration get a Gaussian decay penalty. Prevents animation-first
    designs from scoring 100 when visuals sit 30-60s before narration.

    Verification sources:
    - Case A: keyword in current scene's OCR union → scene active at t_narr → True
    - Case D: progressive reveal in current scene → scene active at t_narr → True
    - Case E: Gemini concept match → True ONLY if matched scene contains t_narr
    - Case B: visual appears AFTER narration → delta_t > 0 → flag irrelevant
    - Case C: visual expired (scene ended before t_narr) → False
    - Case G: SigLIP match → True ONLY if matched scene contains t_narr
    - Case F: no match → False
    """
    kw_df = kw_df.copy()

    # [v4.0] Compute visual_on_screen flag from alignment data
    def _is_visual_on_screen(row):
        mc = row.get("match_case")
        if mc in ("A", "D"):
            # Case A/D: keyword found in CURRENT scene at t_narr → visual verified
            return True
        elif mc == "E":
            # Case E: check if matched scene contains t_narr
            t_narr = row.get("t_narr")
            s_start = row.get("scene_t_start")
            s_end = row.get("scene_t_end")
            if pd.notna(t_narr) and pd.notna(s_start) and pd.notna(s_end):
                return float(s_start) <= float(t_narr) <= float(s_end)
            return False
        elif mc == "G":
            # Case G: SigLIP match — check if matched scene contains t_narr
            t_narr = row.get("t_narr")
            s_start = row.get("scene_t_start")
            s_end = row.get("scene_t_end")
            if pd.notna(t_narr) and pd.notna(s_start) and pd.notna(s_end):
                return float(s_start) <= float(t_narr) <= float(s_end)
            return False
        elif mc == "C":
            # Case C: visual expired — scene ended before t_narr
            return False
        elif mc == "F":
            return False
        else:
            # Case B or unknown: delta_t > 0, flag doesn't affect scoring
            return False

    kw_df["visual_on_screen"] = kw_df.apply(_is_visual_on_screen, axis=1)

    # Gaussian score — uses visual_on_screen + dwell decay for asymmetric scoring
    kw_df["S_gaussian"] = kw_df.apply(
        lambda r: gaussian_score(r["delta_t"], tau,
                                 match_case=r.get("match_case"),
                                 visual_on_screen=r.get("visual_on_screen"),
                                 dwell_threshold=dwell_threshold,
                                 dwell_tau=dwell_tau),
        axis=1
    )

    # Piecewise score (always computed for comparison)
    kw_df["S_piecewise"] = kw_df.apply(
        lambda r: piecewise_score(r["delta_t"],
                                  match_case=r.get("match_case"),
                                  visual_on_screen=r.get("visual_on_screen"),
                                  dwell_threshold=dwell_threshold,
                                  dwell_tau=dwell_tau),
        axis=1
    )

    # Primary temporal score based on mode
    if scoring_mode == "piecewise":
        kw_df["S_temporal"] = kw_df["S_piecewise"]
    else:
        # "gaussian" or "both" → use gaussian as primary
        kw_df["S_temporal"] = kw_df["S_gaussian"]

    # Zone classification
    kw_df["zone"] = kw_df.apply(
        lambda r: zone_classification(r["delta_t"],
                                      match_case=r.get("match_case"),
                                      visual_on_screen=r.get("visual_on_screen"),
                                      dwell_threshold=dwell_threshold,
                                      dwell_tau=dwell_tau),
        axis=1
    )

    # Floor formula: S_floored = S_raw * (floor + (1-floor) * alpha)
    # This prevents low-alpha matches from being fully zero
    alpha_col = kw_df["alpha"] if "alpha" in kw_df.columns else pd.Series(1.0, index=kw_df.index)
    kw_df["S_floored"] = kw_df["S_temporal"] * (
        floor_factor + (1 - floor_factor) * alpha_col
    )
    # Keep NaN for unmatched
    kw_df.loc[kw_df["S_temporal"].isna(), "S_floored"] = None

    return kw_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 2: Importance Weighting
# ═══════════════════════════════════════════════════════════════════════════════

def _apply_importance_weights(kw_df: pd.DataFrame, output_dir: str,
                              cfg: Config) -> pd.DataFrame:
    """Apply pedagogical importance weights to keyword scores."""
    kw_df = kw_df.copy()

    imp_path = os.path.join(output_dir, "pedagogical_importance.csv")
    if not os.path.exists(imp_path):
        logger.info("No pedagogical_importance.csv — using uniform weights")
        kw_df["importance"] = 3
        kw_df["importance_weight"] = 1.0
        kw_df["S_weighted"] = kw_df["S_temporal"]
        kw_df["priority"] = kw_df.apply(
            lambda r: (100 - (r["S_temporal"] or 0)) * r.get("alpha", 1.0)
            if pd.notna(r.get("S_temporal")) else 0, axis=1
        )
        return kw_df

    imp_df = pd.read_csv(imp_path)
    imp_map = dict(zip(imp_df["segment_id"], imp_df["importance"]))
    backend_map = dict(zip(imp_df["segment_id"], imp_df.get("backend", "unknown")))

    # Choose weight table based on backend
    def get_weight(seg_id):
        imp = imp_map.get(seg_id, 3)
        backend = backend_map.get(seg_id, "heuristic")
        if backend == "heuristic":
            return cfg.IMPORTANCE_HEURISTIC_WEIGHTS.get(imp, 1.0)
        else:
            return cfg.IMPORTANCE_WEIGHTS.get(imp, 1.0)

    kw_df["importance"] = kw_df["segment_id"].map(
        lambda sid: imp_map.get(sid, 3)
    )
    kw_df["importance_weight"] = kw_df["segment_id"].map(get_weight)

    # Weighted score
    kw_df["S_weighted"] = kw_df.apply(
        lambda r: r["S_temporal"] * r["importance_weight"]
        if pd.notna(r.get("S_temporal")) else None, axis=1
    )

    # Priority: higher = needs more attention
    # (100 - S) * importance_weight * alpha
    kw_df["priority"] = kw_df.apply(
        lambda r: (100 - (r["S_temporal"] or 0)) * r["importance_weight"] * r.get("alpha", 1.0)
        if pd.notna(r.get("S_temporal")) else 0, axis=1
    )

    return kw_df


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3: Per-Segment Aggregation
# ═══════════════════════════════════════════════════════════════════════════════

def _aggregate_segments(kw_df: pd.DataFrame, output_dir: str,
                        cfg: Config) -> pd.DataFrame:
    """Aggregate keyword scores into per-segment scores."""
    # Load segment text for preview
    seg_path = os.path.join(output_dir, "transcript_segments_improved.csv")
    seg_text = {}
    if os.path.exists(seg_path):
        seg_meta = pd.read_csv(seg_path)
        seg_text = dict(zip(seg_meta["segment_id"], seg_meta["text"]))

    rows = []
    for seg_id, grp in kw_df.groupby("segment_id"):
        matched = grp[grp["match_case"] != "F"]
        unmatched = grp[grp["match_case"] == "F"]

        text = str(seg_text.get(seg_id, ""))[:80]

        row = {
            "segment_id": seg_id,
            "text_preview": text,
            "n_keywords": len(grp),
            "n_matched": len(matched),
            "n_unmatched": len(unmatched),
            "match_rate": len(matched) / len(grp) if len(grp) > 0 else 0,
        }

        if len(matched) > 0:
            # Alpha-weighted mean temporal score
            alphas = matched["alpha"].fillna(1.0)
            s_vals = matched["S_temporal"].fillna(0)
            if alphas.sum() > 0:
                row["mean_S_temporal"] = float((s_vals * alphas).sum() / alphas.sum())
            else:
                row["mean_S_temporal"] = float(s_vals.mean())

            row["median_S_temporal"] = float(matched["S_temporal"].median())

            # Also compute piecewise aggregate
            s_pw = matched["S_piecewise"].fillna(0)
            if alphas.sum() > 0:
                row["mean_S_piecewise"] = float((s_pw * alphas).sum() / alphas.sum())
            else:
                row["mean_S_piecewise"] = float(s_pw.mean())

            # delta_t stats
            deltas = matched["delta_t"].dropna()
            if len(deltas) > 0:
                row["mean_delta_t"] = float(deltas.mean())
                row["median_delta_t"] = float(deltas.median())
                row["min_delta_t"] = float(deltas.min())
                row["max_delta_t"] = float(deltas.max())

            # Zone distribution
            zones = matched["zone"].value_counts()
            for z in ["Optimal", "Suboptimal", "Disruptive", "Unacceptable"]:
                row[f"n_{z}"] = int(zones.get(z, 0))

            # Confidence tiers
            high_conf = matched[matched["alpha"] >= 0.8]
            med_conf = matched[matched["alpha"] < 0.8]
            row["n_high_confidence"] = len(high_conf)
            row["n_medium_confidence"] = len(med_conf)

            if len(high_conf) > 0:
                row["mean_S_high_conf"] = float(high_conf["S_temporal"].mean())
            else:
                row["mean_S_high_conf"] = None

            # Importance
            if "importance" in matched.columns:
                row["importance"] = int(matched["importance"].mode().iloc[0])
            if "importance_weight" in matched.columns:
                row["importance_weight"] = float(matched["importance_weight"].mean())
            if "S_weighted" in matched.columns:
                row["mean_S_weighted"] = float(matched["S_weighted"].mean())

            # Priority (max across keywords — worst offender)
            if "priority" in matched.columns:
                row["max_priority"] = float(matched["priority"].max())
        else:
            row["mean_S_temporal"] = None
            row["median_S_temporal"] = None
            row["mean_S_piecewise"] = None
            row["mean_delta_t"] = None
            row["median_delta_t"] = None
            row["min_delta_t"] = None
            row["max_delta_t"] = None
            for z in ["Optimal", "Suboptimal", "Disruptive", "Unacceptable"]:
                row[f"n_{z}"] = 0
            row["n_high_confidence"] = 0
            row["n_medium_confidence"] = 0
            row["mean_S_high_conf"] = None
            row["importance"] = int(grp["importance"].mode().iloc[0]) if "importance" in grp.columns else 3
            row["importance_weight"] = float(grp["importance_weight"].mean()) if "importance_weight" in grp.columns else 1.0
            row["mean_S_weighted"] = None
            row["max_priority"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4: Overall Results
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_overall_results(kw_df: pd.DataFrame, seg_df: pd.DataFrame,
                             cfg: Config) -> dict:
    """Compute per-video aggregate metrics."""
    matched = kw_df[kw_df["match_case"] != "F"].copy()
    all_kw = kw_df.copy()

    results = {
        "pipeline_version": "4.1",
        "scoring_mode": cfg.SCORING_MODE,
        "tau": cfg.SCORE_TAU,
        "asymmetric": True,
        "floor_factor": cfg.PIECEWISE_FLOOR_FACTOR,
        "dwell_threshold": getattr(cfg, 'DWELL_THRESHOLD', 15.0),
        "dwell_tau": getattr(cfg, 'DWELL_TAU', 15.0),
    }

    # ── Keyword-Level Metrics ────────────────────────────────────────────────
    kw_metrics = {
        "n_total": len(all_kw),
        "n_groundable": int((all_kw["groundability"] != "LOW").sum()) if "groundability" in all_kw.columns else len(all_kw),
        "n_matched": len(matched),
        "n_unmatched": int((all_kw["match_case"] == "F").sum()),
        "match_rate": float(len(matched) / len(all_kw)) if len(all_kw) > 0 else 0,
    }

    # Case F sub-classification
    unmatched = all_kw[all_kw["match_case"] == "F"]
    if len(unmatched) > 0 and "match_method" in unmatched.columns:
        f_reasons = unmatched["match_method"].value_counts().to_dict()
        kw_metrics["case_F_reasons"] = {str(k): int(v) for k, v in f_reasons.items()}

    if len(matched) > 0:
        # Alpha-weighted mean (primary metric)
        alphas = matched["alpha"].fillna(1.0)
        s_gauss = matched["S_gaussian"].fillna(0)
        s_piece = matched["S_piecewise"].fillna(0)
        s_temp = matched["S_temporal"].fillna(0)

        if alphas.sum() > 0:
            kw_metrics["mean_S_gaussian_weighted"] = float((s_gauss * alphas).sum() / alphas.sum())
            kw_metrics["mean_S_piecewise_weighted"] = float((s_piece * alphas).sum() / alphas.sum())
            kw_metrics["mean_S_temporal_weighted"] = float((s_temp * alphas).sum() / alphas.sum())
        else:
            kw_metrics["mean_S_gaussian_weighted"] = float(s_gauss.mean())
            kw_metrics["mean_S_piecewise_weighted"] = float(s_piece.mean())
            kw_metrics["mean_S_temporal_weighted"] = float(s_temp.mean())

        # Unweighted means
        kw_metrics["mean_S_gaussian_unweighted"] = float(s_gauss.mean())
        kw_metrics["mean_S_piecewise_unweighted"] = float(s_piece.mean())
        kw_metrics["mean_S_temporal_unweighted"] = float(s_temp.mean())
        kw_metrics["median_S_temporal"] = float(matched["S_temporal"].median())

        # delta_t stats
        deltas = matched["delta_t"].dropna()
        if len(deltas) > 0:
            kw_metrics["mean_delta_t"] = float(deltas.mean())
            kw_metrics["median_delta_t"] = float(deltas.median())
            kw_metrics["std_delta_t"] = float(deltas.std())
            kw_metrics["min_delta_t"] = float(deltas.min())
            kw_metrics["max_delta_t"] = float(deltas.max())
            kw_metrics["pct_negative_delta_t"] = float((deltas <= 0).mean() * 100)
            kw_metrics["pct_within_1s"] = float((deltas.abs() <= 1.0).mean() * 100)

        # Alpha stats
        kw_metrics["mean_alpha"] = float(alphas.mean())
        kw_metrics["pct_high_confidence"] = float((alphas >= 0.8).mean() * 100)

        # Zone distribution
        zones = matched["zone"].value_counts(normalize=True) * 100
        for z in ["Optimal", "Suboptimal", "Disruptive", "Unacceptable"]:
            kw_metrics[f"pct_{z}"] = float(zones.get(z, 0))

        # Case distribution
        if "match_case" in matched.columns:
            cases = matched["match_case"].value_counts().to_dict()
            kw_metrics["case_distribution"] = {str(k): int(v) for k, v in cases.items()}

        # Two-tier confidence reporting
        high_conf = matched[matched["alpha"] >= 0.8]
        med_conf = matched[matched["alpha"] < 0.8]

        if len(high_conf) > 0:
            kw_metrics["high_confidence"] = {
                "n": len(high_conf),
                "mean_S_temporal": float(high_conf["S_temporal"].mean()),
                "mean_delta_t": float(high_conf["delta_t"].mean()) if high_conf["delta_t"].notna().any() else None,
            }
        if len(med_conf) > 0:
            kw_metrics["medium_confidence"] = {
                "n": len(med_conf),
                "mean_S_temporal": float(med_conf["S_temporal"].mean()),
                "mean_delta_t": float(med_conf["delta_t"].mean()) if med_conf["delta_t"].notna().any() else None,
            }

    results["keyword_level"] = kw_metrics

    # ── Segment-Level Metrics ────────────────────────────────────────────────
    seg_matched = seg_df[seg_df["mean_S_temporal"].notna()]

    seg_metrics = {
        "n_total": len(seg_df),
        "n_with_matches": len(seg_matched),
        "n_without_matches": len(seg_df) - len(seg_matched),
    }

    if len(seg_matched) > 0:
        seg_metrics["mean_S_temporal"] = float(seg_matched["mean_S_temporal"].mean())
        seg_metrics["median_S_temporal"] = float(seg_matched["mean_S_temporal"].median())
        if "mean_S_piecewise" in seg_matched.columns:
            seg_metrics["mean_S_piecewise"] = float(seg_matched["mean_S_piecewise"].mean())
        if "mean_S_weighted" in seg_matched.columns and seg_matched["mean_S_weighted"].notna().any():
            seg_metrics["mean_S_weighted"] = float(seg_matched["mean_S_weighted"].mean())

    results["segment_level"] = seg_metrics

    # ── Overall Score & Grade ────────────────────────────────────────────────
    # Combined weight = alpha (match confidence) × importance_weight (pedagogical)
    # This ensures both alignment confidence AND pedagogical importance affect
    # the final score: high-importance concepts with strong visual matches
    # contribute more to the overall temporal contiguity score.
    if len(matched) > 0:
        alphas = matched["alpha"].fillna(1.0)
        imp_w = matched["importance_weight"].fillna(1.0) if "importance_weight" in matched.columns else pd.Series(1.0, index=matched.index)
        combined_w = alphas * imp_w
        s_temp = matched["S_temporal"].fillna(0)
        if combined_w.sum() > 0:
            score = float((s_temp * combined_w).sum() / combined_w.sum())
        else:
            score = float(s_temp.mean())
        results["overall_score"] = round(score, 2)
        results["overall_grade"] = overall_grade(score)

        # Alpha-only score (no importance weighting) for comparison
        if alphas.sum() > 0:
            score_alpha_only = float((s_temp * alphas).sum() / alphas.sum())
        else:
            score_alpha_only = float(s_temp.mean())
        results["overall_score_alpha_only"] = round(score_alpha_only, 2)

        # Fully unweighted mean
        results["overall_score_unweighted"] = round(float(s_temp.mean()), 2)

        # Piecewise alternative (also uses combined weight)
        s_pw = matched["S_piecewise"].fillna(0)
        if combined_w.sum() > 0:
            pw_score = float((s_pw * combined_w).sum() / combined_w.sum())
        else:
            pw_score = float(s_pw.mean())
        results["overall_score_piecewise"] = round(pw_score, 2)
        results["overall_grade_piecewise"] = overall_grade(pw_score)
    else:
        results["overall_score"] = None
        results["overall_grade"] = "N/A"

    # ── Coverage-Adjusted Metrics ──────────────────────────────────────────────
    # TC score = "among concepts with visual correlates, how well-timed?"
    # Coverage = "what fraction of concepts have visual correlates?"
    # Both are needed for a complete picture. They measure different things.
    n_groundable = kw_metrics.get("n_groundable", len(all_kw))
    coverage = len(matched) / max(n_groundable, 1)

    results["coverage"] = {
        "n_groundable_keywords": n_groundable,
        "n_matched": len(matched),
        "coverage_rate": round(coverage, 4),
        "tc_score_matched_only": results.get("overall_score"),
        "tc_score_coverage_adjusted": round(
            (results.get("overall_score", 0) or 0) * coverage, 2
        ),
        "interpretation": (
            "tc_score = temporal alignment quality among matched keywords; "
            "coverage_rate = fraction of groundable keywords with visual correlate; "
            "coverage_adjusted = product (composite, not standard TC metric)"
        ),
    }

    # ── Visual Lead Time Analysis (Case A/D) ─────────────────────────────────
    # For Case A/D, delta_t is negative (visual precedes narration).
    # Report distribution of how far ahead the visual was.
    case_ad = matched[matched["match_case"].isin(["A", "D"])]
    if len(case_ad) > 0:
        lead_times = (-case_ad["delta_t"].dropna()).values  # flip to positive = lead time
        results["visual_lead_time"] = {
            "n_visual_ahead": len(case_ad),
            "mean_lead_s": round(float(lead_times.mean()), 2),
            "median_lead_s": round(float(np.median(lead_times)), 2),
            "max_lead_s": round(float(lead_times.max()), 2),
            "pct_within_2s": round(float((lead_times <= 2.0).mean() * 100), 1),
            "pct_within_5s": round(float((lead_times <= 5.0).mean() * 100), 1),
            "pct_over_10s": round(float((lead_times > 10.0).mean() * 100), 1),
            "distribution": {
                "0-1s (simultaneous)": int((lead_times <= 1.0).sum()),
                "1-3s (near-simultaneous)": int(((lead_times > 1) & (lead_times <= 3)).sum()),
                "3-5s (prepared ahead)": int(((lead_times > 3) & (lead_times <= 5)).sum()),
                "5-10s (slide already up)": int(((lead_times > 5) & (lead_times <= 10)).sum()),
                ">10s (long-showing slide)": int((lead_times > 10).sum()),
            },
        }

    # ── Positive delta_t Only (Case B/C — true TC violations) ────────────────
    case_bc = matched[matched["match_case"].isin(["B", "C"])]
    if len(case_bc) > 0:
        pos_deltas = case_bc["delta_t"].dropna()
        pos_scores = case_bc["S_temporal"].dropna()
        results["positive_delta_t_only"] = {
            "n": len(case_bc),
            "mean_delta_t": round(float(pos_deltas.mean()), 2),
            "mean_S_temporal": round(float(pos_scores.mean()), 2),
            "interpretation": "Cases B/C where visual appears AFTER narration (true TC lag)",
        }

    # ── Top 5 Priority Fixes ─────────────────────────────────────────────────
    if "priority" in kw_df.columns:
        top5 = kw_df.nlargest(5, "priority")
        results["top5_priority_fixes"] = [
            {
                "keyword_id": int(r["keyword_id"]),
                "keyword_text": str(r["keyword_text"]),
                "segment_id": int(r["segment_id"]),
                "delta_t": float(r["delta_t"]) if pd.notna(r.get("delta_t")) else None,
                "S_temporal": float(r["S_temporal"]) if pd.notna(r.get("S_temporal")) else None,
                "priority": float(r["priority"]),
            }
            for _, r in top5.iterrows()
        ]

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5: Diagnostics
# ═══════════════════════════════════════════════════════════════════════════════

def _write_diagnostics(diag, kw_df: pd.DataFrame, seg_df: pd.DataFrame,
                       results: dict, cfg: Config):
    """Write scoring diagnostics."""
    matched = kw_df[kw_df["match_case"] != "F"]

    diag_data = {
        "config": {
            "tau": cfg.SCORE_TAU,
            "scoring_mode": cfg.SCORING_MODE,
            "floor_factor": cfg.PIECEWISE_FLOOR_FACTOR,
            "asymmetric": True,
        },
        "overall": {
            "score_gaussian": results.get("overall_score"),
            "score_piecewise": results.get("overall_score_piecewise"),
            "grade_gaussian": results.get("overall_grade"),
            "grade_piecewise": results.get("overall_grade_piecewise"),
        },
        "n_keywords_scored": len(matched),
        "n_keywords_unmatched": int((kw_df["match_case"] == "F").sum()),
    }

    # Score distribution histogram (gaussian)
    if len(matched) > 0 and "S_gaussian" in matched.columns:
        s_vals = matched["S_gaussian"].dropna().values
        diag_data["gaussian_score_histogram"] = {
            "0-20": int((s_vals < 20).sum()),
            "20-40": int(((s_vals >= 20) & (s_vals < 40)).sum()),
            "40-60": int(((s_vals >= 40) & (s_vals < 60)).sum()),
            "60-80": int(((s_vals >= 60) & (s_vals < 80)).sum()),
            "80-100": int((s_vals >= 80).sum()),
        }

    # Piecewise score distribution
    if len(matched) > 0 and "S_piecewise" in matched.columns:
        s_vals = matched["S_piecewise"].dropna().values
        diag_data["piecewise_score_histogram"] = {
            "0-20": int((s_vals < 20).sum()),
            "20-40": int(((s_vals >= 20) & (s_vals < 40)).sum()),
            "40-60": int(((s_vals >= 40) & (s_vals < 60)).sum()),
            "60-80": int(((s_vals >= 60) & (s_vals < 80)).sum()),
            "80-100": int((s_vals >= 80).sum()),
        }

    # Zone distribution
    if len(matched) > 0:
        zones = matched["zone"].value_counts().to_dict()
        diag_data["zone_distribution"] = {str(k): int(v) for k, v in zones.items()}

    # delta_t distribution (finer bins for negative values)
    if len(matched) > 0:
        deltas = matched["delta_t"].dropna().values
        diag_data["delta_t_histogram"] = {
            "<-10s (visual far ahead)": int((deltas < -10).sum()),
            "-10s to -5s": int(((deltas >= -10) & (deltas < -5)).sum()),
            "-5s to -2s": int(((deltas >= -5) & (deltas < -2)).sum()),
            "-2s to -1s": int(((deltas >= -2) & (deltas < -1)).sum()),
            "-1s to 0 (near-simultaneous)": int(((deltas >= -1) & (deltas <= 0)).sum()),
            "0 to 1s (near-simultaneous)": int(((deltas > 0) & (deltas <= 1)).sum()),
            "1-3s (suboptimal)": int(((deltas > 1) & (deltas <= 3)).sum()),
            "3-5s (disruptive)": int(((deltas > 3) & (deltas <= 5)).sum()),
            ">5s (unacceptable)": int((deltas > 5).sum()),
        }

    # Importance weight impact
    if "S_weighted" in matched.columns and "S_temporal" in matched.columns:
        s_uw = matched["S_temporal"].mean()
        s_w = matched["S_weighted"].mean()
        if pd.notna(s_uw) and pd.notna(s_w):
            diag_data["importance_impact"] = {
                "mean_unweighted": float(s_uw),
                "mean_weighted": float(s_w),
                "delta": float(s_w - s_uw),
            }

    diag.write_json("stage7_scoring.json", diag_data)


# ═══════════════════════════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════════════════════════

def _log_summary(results: dict):
    """Log a human-readable summary."""
    logger.info("=" * 60)
    logger.info("TEMPORAL CONTIGUITY RESULTS (v2.2)")
    logger.info("=" * 60)

    score = results.get("overall_score")
    grade = results.get("overall_grade", "N/A")
    score_pw = results.get("overall_score_piecewise")
    grade_pw = results.get("overall_grade_piecewise", "N/A")

    if score is not None:
        logger.info("  Gaussian score: %.1f / 100 (%s)", score, grade)
    if score_pw is not None:
        logger.info("  Piecewise score: %.1f / 100 (%s)", score_pw, grade_pw)

    kw = results.get("keyword_level", {})
    logger.info("  Keywords: %d total, %d matched (%.1f%%)",
                kw.get("n_total", 0), kw.get("n_matched", 0),
                kw.get("match_rate", 0) * 100)

    if "mean_delta_t" in kw:
        logger.info("  delta_t: mean=%.2fs, median=%.2fs, std=%.2fs",
                    kw["mean_delta_t"], kw.get("median_delta_t", 0),
                    kw.get("std_delta_t", 0))

    for z in ["Optimal", "Suboptimal", "Disruptive", "Unacceptable"]:
        pct = kw.get(f"pct_{z}", 0)
        if pct > 0:
            logger.info("  Zone %s: %.1f%%", z, pct)

    # Two-tier confidence
    hc = kw.get("high_confidence", {})
    mc = kw.get("medium_confidence", {})
    if hc:
        logger.info("  HIGH confidence (%d): mean S=%.1f",
                    hc.get("n", 0), hc.get("mean_S_temporal", 0))
    if mc:
        logger.info("  MEDIUM confidence (%d): mean S=%.1f",
                    mc.get("n", 0), mc.get("mean_S_temporal", 0))

    # Coverage
    cov = results.get("coverage", {})
    if cov:
        logger.info("  Coverage: %d/%d groundable keywords matched (%.1f%%)",
                    cov.get("n_matched", 0), cov.get("n_groundable_keywords", 0),
                    cov.get("coverage_rate", 0) * 100)
        logger.info("  Coverage-adjusted score: %.1f",
                    cov.get("tc_score_coverage_adjusted", 0))

    # Visual lead time
    vlt = results.get("visual_lead_time", {})
    if vlt:
        logger.info("  Visual lead time (Case A/D): mean=%.1fs, median=%.1fs, "
                    "%.1f%% within 2s, %.1f%% over 10s",
                    vlt.get("mean_lead_s", 0), vlt.get("median_lead_s", 0),
                    vlt.get("pct_within_2s", 0), vlt.get("pct_over_10s", 0))

    # Positive delta_t only
    pos = results.get("positive_delta_t_only", {})
    if pos:
        logger.info("  Positive delta_t (Case B/C): n=%d, mean=%.1fs, mean S=%.1f",
                    pos.get("n", 0), pos.get("mean_delta_t", 0),
                    pos.get("mean_S_temporal", 0))

    seg = results.get("segment_level", {})
    logger.info("  Segments: %d total, %d with matches",
                seg.get("n_total", 0), seg.get("n_with_matches", 0))
    logger.info("=" * 60)
