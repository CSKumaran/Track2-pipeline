"""Regression stub – placeholder for future learner-outcome modelling."""

from __future__ import annotations

import pandas as pd


def prepare_regression_dataset(
    scores_per_scene_path: str,
    learner_outcomes_path: str,
) -> pd.DataFrame:
    """Join scene-level scores with learner-level outcomes.

    Expected learner_outcomes schema (to be confirmed later):
        video_id, scene_id, learner_id, quiz_score, ecl, ...

    Steps to implement:
        1. Load scores CSV and learner outcomes CSV.
        2. Merge on (video_id, scene_id) or appropriate key.
        3. Return a flat DataFrame ready for regression.
    """
    raise NotImplementedError(
        "prepare_regression_dataset() is not yet implemented. "
        "Provide a learner_outcomes CSV and update this function."
    )


def fit_regression_models(df: pd.DataFrame) -> dict:
    """Fit OLS and/or XGBoost regression on alignment features.

    Candidate features:
        |Δt|, alpha, S_final, zone_label (one-hot), match_type (one-hot),
        threshold, n_scenes, ...

    Target(s):
        quiz_score, ecl, or other learner-level outcome.

    Steps to implement:
        1. Feature engineering (encode categoricals, handle missing).
        2. Train/test split or cross-validation.
        3. Fit statsmodels OLS for interpretability.
        4. Fit sklearn/XGBoost for predictive performance.
        5. Return dict of model summaries, R², RMSE, feature importances.
    """
    raise NotImplementedError(
        "fit_regression_models() is not yet implemented. "
        "Requires a prepared regression dataset from prepare_regression_dataset()."
    )
