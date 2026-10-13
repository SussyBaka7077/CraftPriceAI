"""Local SHAP explanations for the demand model.

This module is the single place that knows the internal structure of the fitted
scikit-learn pipeline (its named steps), keeping that coupling out of the UI.
"""
from __future__ import annotations

import pandas as pd

_PREPROCESSOR_STEP = "preprocessor"
_ESTIMATOR_STEP = "model"
_FEATURE_NAME_PREFIXES = ("numeric__", "categorical__")


def _strip_prefix(name: str) -> str:
    """Drop the ColumnTransformer prefix so the UI shows readable feature names."""
    for prefix in _FEATURE_NAME_PREFIXES:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def explain_recommendation(
    model, candidates_row: pd.DataFrame, top_n: int = 12
) -> pd.DataFrame:
    """Return the top local SHAP contributions for a single feature-complete row.

    Parameters
    ----------
    model:
        Fitted scikit-learn ``Pipeline`` whose steps are the preprocessor and estimator.
    candidates_row:
        A single-row ``DataFrame`` that is already feature-complete (e.g. one row of
        the candidate price frame).
    top_n:
        Maximum number of features to return, ranked by absolute SHAP value.

    Returns
    -------
    pandas.DataFrame
        Columns ``feature`` (readable name) and ``shap_value``, sorted by
        ``abs(shap_value)`` descending and truncated to ``top_n`` rows. Failures
        propagate naturally so the caller can surface a message.
    """
    import shap

    preprocessor = model.named_steps[_PREPROCESSOR_STEP]
    estimator = model.named_steps[_ESTIMATOR_STEP]

    transformed = preprocessor.transform(candidates_row)
    values = shap.TreeExplainer(estimator).shap_values(transformed)
    names = [_strip_prefix(str(name)) for name in preprocessor.get_feature_names_out()]

    explanation = (
        pd.DataFrame({"feature": names, "shap_value": values[0]})
        .sort_values("shap_value", key=abs, ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return explanation
