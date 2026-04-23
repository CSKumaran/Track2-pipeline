"""Sensitivity analysis for heuristic importance weights and scoring parameters.

[v4.0] Provides a framework to:
1. Vary heuristic feature weights and measure impact on final scores
2. Vary scoring parameters (tau, importance multipliers) and measure stability
3. Generate a comparison table for inclusion in papers

Usage (standalone):
    python -m pipeline_v4_gpu.utils.sensitivity_analysis --output-dir outputs_v4/video_name

Usage (from code):
    from pipeline_v4_gpu.utils.sensitivity_analysis import run_sensitivity_analysis
    results = run_sensitivity_analysis(output_dir)

Theoretical context for weight choices:
    Each heuristic feature maps to an established pedagogical/cognitive principle.
    The weights reflect relative importance judgments, not empirical calibration.
    This analysis tests whether the final scores are ROBUST to weight perturbations.
    If scores are stable across ±50% weight changes, the specific values matter less
    than the feature set — a strong defense against "arbitrary weights" criticism.

References:
    - Keyword density → Cognitive Load Theory (Sweller, 1988)
    - Speech rate → Segmenting Principle (Mayer, 2009)
    - Word count → Element interactivity (Sweller & Chandler, 1994)
    - Instructional verbs → Bloom's Taxonomy (Anderson & Krathwohl, 2001)
    - Discourse signals → Signaling Principle (Mayer, 2009; Lemarie et al., 2008)
    - First-mention → Schema Theory / novelty processing (Piaget; Tulving, 1972)
"""

import itertools
import logging
import os

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# Default weight configuration (V4 baseline)
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_WEIGHTS = {
    "kw_density": 0.30,
    "speech_rate": 0.20,
    "word_count": 0.10,
    "instr_verb": 0.15,
    "discourse": 0.10,
    "first_mention": 0.15,
}

# Feature normalization constants
NORM_KW_DENSITY = 2.0      # keywords per second
NORM_SPEECH_RATE = 4.0     # words per second
NORM_WORD_COUNT = 25       # words per segment

# Importance multiplier maps
DEFAULT_IMP_WEIGHTS = {1: 0.3, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}
DEFAULT_HEUR_WEIGHTS = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}


# ═══════════════════════════════════════════════════════════════════════════════
# Weight perturbation configurations
# ═══════════════════════════════════════════════════════════════════════════════

def generate_weight_perturbations(base_weights: dict = None,
                                  perturbation_pcts: list = None) -> list:
    """Generate weight configurations by perturbing each feature ±N%.

    Returns a list of (config_name, weights_dict) tuples.
    Each configuration perturbs ONE feature at a time while keeping
    the rest proportionally re-normalized to sum to 1.0.
    """
    if base_weights is None:
        base_weights = DEFAULT_WEIGHTS.copy()
    if perturbation_pcts is None:
        perturbation_pcts = [-50, -25, 0, +25, +50]

    configs = [("baseline", base_weights.copy())]
    features = list(base_weights.keys())

    for feature in features:
        for pct in perturbation_pcts:
            if pct == 0:
                continue  # Already in baseline

            # Perturb one feature
            new_w = base_weights.copy()
            original_val = new_w[feature]
            new_val = original_val * (1 + pct / 100)
            new_val = max(0.01, new_val)  # Floor at 1%
            new_w[feature] = new_val

            # Re-normalize to sum to 1.0
            total = sum(new_w.values())
            new_w = {k: v / total for k, v in new_w.items()}

            name = f"{feature}_{pct:+d}pct"
            configs.append((name, new_w))

    return configs


def generate_ablation_configs(base_weights: dict = None) -> list:
    """Generate leave-one-out ablation configurations.

    For each feature, set its weight to 0 and re-normalize the rest.
    Also generates V3 config (only kw_density, speech_rate, word_count).
    """
    if base_weights is None:
        base_weights = DEFAULT_WEIGHTS.copy()

    configs = [("full_v4", base_weights.copy())]
    features = list(base_weights.keys())

    for feature in features:
        new_w = base_weights.copy()
        new_w[feature] = 0.0
        total = sum(new_w.values())
        if total > 0:
            new_w = {k: v / total for k, v in new_w.items()}
        configs.append((f"no_{feature}", new_w))

    # V3-equivalent: only original 3 features
    v3_w = {k: 0.0 for k in base_weights}
    v3_w["kw_density"] = 0.40  # V3 weights
    v3_w["speech_rate"] = 0.30
    v3_w["word_count"] = 0.20
    # tech_overlap was 0.10 in V3 but not in V4 features; redistribute
    v3_w["kw_density"] = 0.44
    v3_w["speech_rate"] = 0.33
    v3_w["word_count"] = 0.23
    configs.append(("v3_equivalent", v3_w))

    return configs


# ═══════════════════════════════════════════════════════════════════════════════
# Core: Compute heuristic scores with arbitrary weights
# ═══════════════════════════════════════════════════════════════════════════════

def compute_heuristic_features(output_dir: str) -> pd.DataFrame:
    """Extract raw heuristic features for all segments.

    Reads transcript_segments_improved.csv and keywords.csv to compute
    the 6 feature values (pre-weighting) for each segment.
    Returns a DataFrame with columns: segment_id, f_kw, f_rate, f_words,
    f_instr_verb, f_discourse, f_first_mention.
    """
    seg_path = os.path.join(output_dir, "transcript_segments_improved.csv")
    kw_path = os.path.join(output_dir, "keywords.csv")

    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Missing {seg_path}")

    segments_df = pd.read_csv(seg_path)

    # Keyword counts per segment
    kw_per_segment = {}
    first_mention_per_segment = {}
    if os.path.exists(kw_path):
        kw_df = pd.read_csv(kw_path)
        if not kw_df.empty and "segment_id" in kw_df.columns:
            kw_per_segment = kw_df.groupby("segment_id").size().to_dict()
            if "is_first_mention" in kw_df.columns:
                fm = kw_df[kw_df["is_first_mention"] == True]
                first_mention_per_segment = fm.groupby("segment_id").size().to_dict()

    # spaCy for instructional verbs
    nlp = None
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm", disable=["ner"])
    except Exception:
        pass

    # Import detection functions
    from ..stages.pedagogical_rating import _detect_instructional_verb, _detect_discourse_signals

    rows = []
    for _, seg in segments_df.iterrows():
        seg_id = seg["segment_id"]
        text = str(seg["text"])
        duration = float(seg["end_time"]) - float(seg["start_time"]) if pd.notna(seg["end_time"]) else 1
        n_words = len(text.split())
        n_kw = kw_per_segment.get(seg_id, 0)

        f_kw = min((n_kw / max(duration, 0.1)) / NORM_KW_DENSITY, 1.0)
        f_rate = min((n_words / max(duration, 0.1)) / NORM_SPEECH_RATE, 1.0)
        f_words = min(n_words / NORM_WORD_COUNT, 1.0)
        f_instr_verb = _detect_instructional_verb(text, nlp)
        f_discourse = _detect_discourse_signals(text)
        n_first = first_mention_per_segment.get(seg_id, 0)
        f_first_mention = min(n_first / max(n_kw, 1), 1.0) if n_kw > 0 else 0.0

        rows.append({
            "segment_id": seg_id,
            "f_kw": f_kw,
            "f_rate": f_rate,
            "f_words": f_words,
            "f_instr_verb": f_instr_verb,
            "f_discourse": f_discourse,
            "f_first_mention": f_first_mention,
        })

    return pd.DataFrame(rows)


def score_with_weights(features_df: pd.DataFrame, weights: dict) -> pd.Series:
    """Compute weighted heuristic score for each segment given a weight config.

    Returns a Series of raw scores (before percentile binning).
    """
    feature_cols = {
        "kw_density": "f_kw",
        "speech_rate": "f_rate",
        "word_count": "f_words",
        "instr_verb": "f_instr_verb",
        "discourse": "f_discourse",
        "first_mention": "f_first_mention",
    }

    score = pd.Series(0.0, index=features_df.index)
    for weight_name, col_name in feature_cols.items():
        w = weights.get(weight_name, 0.0)
        score += w * features_df[col_name]

    return score


def percentile_bin(scores: pd.Series) -> pd.Series:
    """Map continuous scores to 1-5 via percentile bins."""
    if scores.empty:
        return scores
    percentiles = np.percentile(scores.values, [20, 40, 60, 80])
    def _bin(s):
        if s <= percentiles[0]:
            return 1
        elif s <= percentiles[1]:
            return 2
        elif s <= percentiles[2]:
            return 3
        elif s <= percentiles[3]:
            return 4
        else:
            return 5
    return scores.apply(_bin)


# ═══════════════════════════════════════════════════════════════════════════════
# End-to-end sensitivity analysis
# ═══════════════════════════════════════════════════════════════════════════════

def run_sensitivity_analysis(output_dir: str, include_scoring: bool = True) -> dict:
    """Run full sensitivity analysis on a single video's outputs.

    Requires: Stages 1-5 outputs in output_dir (at minimum:
    transcript_segments_improved.csv, keywords.csv).
    Optionally uses keyword_alignment.csv for end-to-end scoring impact.

    Returns a dict with:
    - weight_perturbation: DataFrame of (config, mean_importance, std, rank_corr_vs_baseline)
    - ablation: DataFrame of (config, mean_importance, features_used)
    - scoring_impact: DataFrame of (config, overall_score, delta_vs_baseline)  [if include_scoring]
    - feature_correlations: pairwise correlation matrix of raw features
    """
    logger.info("Sensitivity analysis on: %s", output_dir)

    # Step 1: Extract raw features
    features_df = compute_heuristic_features(output_dir)
    n_seg = len(features_df)
    logger.info("  %d segments, extracting 6 features", n_seg)

    # Step 2: Feature correlations (useful for paper)
    feature_cols = ["f_kw", "f_rate", "f_words", "f_instr_verb", "f_discourse", "f_first_mention"]
    corr_matrix = features_df[feature_cols].corr()

    # Step 3: Weight perturbation analysis
    perturbation_configs = generate_weight_perturbations()
    baseline_scores = score_with_weights(features_df, DEFAULT_WEIGHTS)
    baseline_imp = percentile_bin(baseline_scores)

    perturbation_rows = []
    for name, weights in perturbation_configs:
        scores = score_with_weights(features_df, weights)
        imp = percentile_bin(scores)

        # Spearman rank correlation with baseline
        from scipy.stats import spearmanr
        if len(imp) > 2:
            rho, _ = spearmanr(baseline_imp, imp)
        else:
            rho = 1.0

        perturbation_rows.append({
            "config": name,
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "mean_importance": float(imp.mean()),
            "rank_corr_vs_baseline": float(rho),
            "pct_same_rating": float((imp == baseline_imp).mean() * 100),
            "max_rating_change": int((imp - baseline_imp).abs().max()),
        })

    perturbation_df = pd.DataFrame(perturbation_rows)

    # Step 4: Ablation analysis
    ablation_configs = generate_ablation_configs()
    ablation_rows = []
    for name, weights in ablation_configs:
        scores = score_with_weights(features_df, weights)
        imp = percentile_bin(scores)
        if len(imp) > 2:
            rho, _ = spearmanr(baseline_imp, imp)
        else:
            rho = 1.0

        active_features = [k for k, v in weights.items() if v > 0.01]
        ablation_rows.append({
            "config": name,
            "n_features": len(active_features),
            "features": ", ".join(active_features),
            "mean_importance": float(imp.mean()),
            "rank_corr_vs_full": float(rho),
            "pct_same_rating": float((imp == baseline_imp).mean() * 100),
        })

    ablation_df = pd.DataFrame(ablation_rows)

    # Step 5: End-to-end scoring impact (if alignment data available)
    scoring_df = None
    if include_scoring:
        align_path = os.path.join(output_dir, "keyword_alignment.csv")
        if os.path.exists(align_path):
            scoring_df = _scoring_sensitivity(output_dir, features_df,
                                              perturbation_configs + ablation_configs)

    results = {
        "output_dir": output_dir,
        "n_segments": n_seg,
        "feature_correlations": corr_matrix,
        "weight_perturbation": perturbation_df,
        "ablation": ablation_df,
        "scoring_impact": scoring_df,
    }

    # Save results
    _save_sensitivity_results(output_dir, results)

    return results


def _scoring_sensitivity(output_dir: str, features_df: pd.DataFrame,
                         configs: list) -> pd.DataFrame:
    """Measure end-to-end impact: weight change → importance → final video score.

    For each weight configuration:
    1. Compute heuristic importance ratings
    2. Apply importance weights to keyword scores
    3. Compute overall video score
    4. Compare with baseline
    """
    from ..stages.scoring import gaussian_score

    kw_path = os.path.join(output_dir, "keyword_alignment.csv")
    if not os.path.exists(kw_path):
        return None

    kw_df = pd.read_csv(kw_path)
    matched = kw_df[kw_df["match_case"] != "F"].copy()

    if matched.empty:
        return None

    # Compute baseline overall score (no importance weighting)
    matched["S_raw"] = matched.apply(
        lambda r: gaussian_score(r["delta_t"], 2.5, match_case=r.get("match_case")),
        axis=1
    )
    baseline_unweighted = float(matched["S_raw"].mean())

    rows = []
    for name, weights in configs:
        scores = score_with_weights(features_df, weights)
        imp = percentile_bin(scores)
        imp_map = dict(zip(features_df["segment_id"], imp))

        # Apply importance weights (heuristic backend → compressed range)
        def get_weight(seg_id):
            i = imp_map.get(seg_id, 3)
            return DEFAULT_HEUR_WEIGHTS.get(i, 1.0)

        matched_copy = matched.copy()
        matched_copy["imp_weight"] = matched_copy["segment_id"].map(get_weight)
        matched_copy["S_weighted"] = matched_copy["S_raw"] * matched_copy["imp_weight"]

        overall = float(matched_copy["S_weighted"].mean())

        rows.append({
            "config": name,
            "overall_score": round(overall, 2),
            "delta_vs_unweighted": round(overall - baseline_unweighted, 2),
        })

    scoring_df = pd.DataFrame(rows)

    # Add delta vs baseline (first config)
    if len(scoring_df) > 0:
        baseline_score = scoring_df.iloc[0]["overall_score"]
        scoring_df["delta_vs_baseline"] = scoring_df["overall_score"] - baseline_score

    return scoring_df


def _save_sensitivity_results(output_dir: str, results: dict):
    """Save sensitivity analysis results to output_dir/sensitivity/."""
    out_path = os.path.join(output_dir, "sensitivity")
    os.makedirs(out_path, exist_ok=True)

    # Feature correlations
    results["feature_correlations"].to_csv(
        os.path.join(out_path, "feature_correlations.csv"), float_format="%.3f"
    )

    # Weight perturbation
    results["weight_perturbation"].to_csv(
        os.path.join(out_path, "weight_perturbation.csv"), index=False, float_format="%.3f"
    )

    # Ablation
    results["ablation"].to_csv(
        os.path.join(out_path, "ablation.csv"), index=False, float_format="%.3f"
    )

    # Scoring impact
    if results["scoring_impact"] is not None:
        results["scoring_impact"].to_csv(
            os.path.join(out_path, "scoring_impact.csv"), index=False, float_format="%.3f"
        )

    # Summary JSON
    from ..utils.io_utils import save_json
    summary = {
        "n_segments": results["n_segments"],
        "n_perturbation_configs": len(results["weight_perturbation"]),
        "n_ablation_configs": len(results["ablation"]),
        "perturbation_stability": {
            "mean_rank_corr": float(results["weight_perturbation"]["rank_corr_vs_baseline"].mean()),
            "min_rank_corr": float(results["weight_perturbation"]["rank_corr_vs_baseline"].min()),
            "mean_pct_same": float(results["weight_perturbation"]["pct_same_rating"].mean()),
        },
        "ablation_stability": {
            "v3_vs_v4_rank_corr": float(
                results["ablation"][results["ablation"]["config"] == "v3_equivalent"]["rank_corr_vs_full"].iloc[0]
            ) if "v3_equivalent" in results["ablation"]["config"].values else None,
        },
    }

    if results["scoring_impact"] is not None:
        scores = results["scoring_impact"]["overall_score"]
        summary["scoring_stability"] = {
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
            "range": float(scores.max() - scores.min()),
            "cv_pct": float(scores.std() / scores.mean() * 100) if scores.mean() > 0 else 0,
        }

    save_json(summary, os.path.join(out_path, "sensitivity_summary.json"))
    logger.info("Sensitivity results saved to %s", out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    p = argparse.ArgumentParser(description="Sensitivity analysis for V4 heuristic weights")
    p.add_argument("--output-dir", required=True, help="Path to video output directory")
    p.add_argument("--no-scoring", action="store_true", help="Skip end-to-end scoring impact")
    args = p.parse_args()

    results = run_sensitivity_analysis(args.output_dir, include_scoring=not args.no_scoring)

    # Print summary
    print("\n" + "=" * 70)
    print("SENSITIVITY ANALYSIS SUMMARY")
    print("=" * 70)

    print(f"\nSegments: {results['n_segments']}")

    print("\n--- Weight Perturbation (±50%) ---")
    pdf = results["weight_perturbation"]
    print(f"  Rank correlation range: {pdf['rank_corr_vs_baseline'].min():.3f} — {pdf['rank_corr_vs_baseline'].max():.3f}")
    print(f"  Mean % same rating:     {pdf['pct_same_rating'].mean():.1f}%")
    print(f"  Max rating change:      {pdf['max_rating_change'].max()}")

    print("\n--- Feature Ablation ---")
    for _, row in results["ablation"].iterrows():
        print(f"  {row['config']:20s}  rank_corr={row['rank_corr_vs_full']:.3f}  "
              f"same={row['pct_same_rating']:.0f}%  features={row['n_features']}")

    if results["scoring_impact"] is not None:
        print("\n--- Scoring Impact ---")
        sdf = results["scoring_impact"]
        print(f"  Score range: {sdf['overall_score'].min():.1f} — {sdf['overall_score'].max():.1f}")
        print(f"  Score std:   {sdf['overall_score'].std():.2f}")
        cv = sdf['overall_score'].std() / sdf['overall_score'].mean() * 100
        print(f"  CV:          {cv:.1f}%")

    print(f"\nResults saved to: {os.path.join(args.output_dir, 'sensitivity')}")


if __name__ == "__main__":
    main()
