from __future__ import annotations
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from .config import CATEGORICAL_FEATURES, DATE_COLUMN, FEATURES, NUMERIC_FEATURES, TARGET
from .features import add_price_features


def chronological_split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by date so that no future rows are used to predict past demand."""
    ordered_dates = np.array(sorted(pd.to_datetime(frame[DATE_COLUMN]).unique()))
    train_end, validation_end = int(len(ordered_dates) * .70), int(len(ordered_dates) * .85)
    train_dates, validation_dates = ordered_dates[:train_end], ordered_dates[train_end:validation_end]
    dated = frame.copy()
    dated[DATE_COLUMN] = pd.to_datetime(dated[DATE_COLUMN])
    return (
        dated[dated[DATE_COLUMN].isin(train_dates)], 
        dated[dated[DATE_COLUMN].isin(validation_dates)],
        dated[~dated[DATE_COLUMN].isin(np.r_[train_dates, validation_dates])],
    )


def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median"))]), NUMERIC_FEATURES),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])


def evaluate(model: Pipeline, test: pd.DataFrame) -> dict[str, float]:
    predictions = np.clip(model.predict(test[FEATURES]), 0, None)
    actual = test[TARGET]
    return {
        "mae": round(float(mean_absolute_error(actual, predictions)), 3),
        "rmse": round(float(mean_squared_error(actual, predictions) ** .5), 3),
        "r2": round(float(r2_score(actual, predictions)), 3),
    }


MAD_TO_SIGMA = float(np.sqrt(np.pi / 2))
PRODUCT_ID_COLUMN = "product_id"


def make_xgb_pipeline() -> Pipeline:
    """Build the XGBoost pipeline. Single source of truth for the hyperparameters."""
    return Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(n_estimators=450, max_depth=5, learning_rate=.045,
                                subsample=.85, colsample_bytree=.9, reg_lambda=2,
                                objective="reg:squarederror", random_state=42, n_jobs=2)),
    ])


def _residual_scale(model: Pipeline, test: pd.DataFrame) -> dict[str, Any]:
    """Tier 2.2: a prediction-dependent interval width for the demand estimate.

    We fit a linear model of |holdout residual| on the holdout prediction itself
    (an estimate of the mean absolute deviation as a function of predicted demand),
    then convert that MAD into a standard deviation via sigma = MAD * sqrt(pi/2).
    """
    predictions = np.clip(model.predict(test[FEATURES]), 0, None)
    actual = test[TARGET].to_numpy()
    residuals = actual - predictions
    abs_residuals = np.abs(residuals)

    calibration = LinearRegression()
    calibration.fit(predictions.reshape(-1, 1), abs_residuals)
    intercept = float(calibration.intercept_)
    slope = float(calibration.coef_[0])

    sigma = np.clip(intercept + slope * predictions, 0, None) * MAD_TO_SIGMA
    nominal_z_80 = 1.2816
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(sigma > 0, abs_residuals / sigma, np.inf)
    z_measured_80 = float(np.quantile(ratio[np.isfinite(ratio)], .80))
    z_measured_50 = float(np.quantile(ratio[np.isfinite(ratio)], .50))

    nominal_coverage_80 = float(np.mean(abs_residuals <= nominal_z_80 * sigma))
    z_80 = nominal_z_80
    block: dict[str, Any] = {
        "intercept": round(intercept, 6),
        "slope": round(slope, 6),
        "mad_to_sigma": round(MAD_TO_SIGMA, 4),
        "z_80": nominal_z_80,
        "method": "linear model of |holdout residual| on prediction, MAD→σ via √(π/2)",
        "z_measured_80": round(z_measured_80, 4),
        "z_measured_50": round(z_measured_50, 4),
        "coverage_80": round(nominal_coverage_80, 4),
        "coverage_50": round(float(np.mean(abs_residuals <= .6745 * sigma)), 4),
    }
    if abs(nominal_coverage_80 - .80) > .03:
        # Nominal normal z misses the target by more than 3 points; use the
        # empirical quantile actually measured on this holdout instead.
        z_80 = z_measured_80
        block["z_80"] = round(z_80, 4)
        block["coverage_80"] = round(float(np.mean(abs_residuals <= z_80 * sigma)), 4)
        block["note"] = (
            f"nominal z_80=1.2816 covered {nominal_coverage_80:.3f}; "
            f"stored interval uses empirical z_80={z_80:.4f} measured on this holdout"
        )
    return block


def _per_season_metrics(model: Pipeline, test: pd.DataFrame, seasons: list[Any]) -> dict[str, Any]:
    """Tier 2.3: holdout error broken down by every season present in the data."""
    per_season: dict[str, Any] = {}
    for season in seasons:
        rows = test[test["season"] == season]
        if len(rows) == 0:
            per_season[str(season)] = {
                "status": "not_evaluated",
                "note": "no holdout rows for this season",
            }
        else:
            per_season[str(season)] = evaluate(model, rows)
    return per_season


def _naive_trailing_4_week_metrics(test: pd.DataFrame) -> dict[str, float]:
    """Tier 2.3: strongest naive baseline -- reuse the trailing 4-week mean as the forecast."""
    predictions = test["rolling_4_week_sales"].to_numpy()
    actual = test[TARGET].to_numpy()
    return {
        "mae": round(float(mean_absolute_error(actual, predictions)), 3),
        "rmse": round(float(mean_squared_error(actual, predictions) ** .5), 3),
        "r2": round(float(r2_score(actual, predictions)), 3),
    }


def _product_holdout_metrics(
    prepared: pd.DataFrame, train: pd.DataFrame, validation: pd.DataFrame
) -> dict[str, Any]:
    """Tier 2.3: unseen-product generalisation, the app's real use case.

    Whole products (the last 20% of sorted ids) are held out entirely; the fresh
    model sees train+validation rows of the remaining products only.
    """
    product_ids = sorted(prepared[PRODUCT_ID_COLUMN].dropna().unique().tolist())
    n_holdout = max(1, int(round(len(product_ids) * .20)))
    holdout_ids = product_ids[-n_holdout:]
    trainable_ids = set(product_ids) - set(holdout_ids)

    trainable = pd.concat([train, validation])
    train_rows = trainable[trainable[PRODUCT_ID_COLUMN].isin(list(trainable_ids))]
    holdout_rows = prepared[prepared[PRODUCT_ID_COLUMN].isin(holdout_ids)]

    model = make_xgb_pipeline()
    model.fit(train_rows[FEATURES], train_rows[TARGET])
    block: dict[str, Any] = evaluate(model, holdout_rows)
    block.update({
        "n_holdout_products": int(len(holdout_ids)),
        "n_holdout_rows": int(len(holdout_rows)),
        "note": "whole products held out; metrics on unseen-product rows",
    })
    return block


def train_models(frame: pd.DataFrame) -> tuple[Pipeline, dict[str, Any], pd.DataFrame]:
    prepared = add_price_features(frame)
    train, validation, test = chronological_split(prepared)
    train_and_validation = pd.concat([train, validation])
    mean_prediction = float(train[TARGET].mean())
    baseline_metrics = {
        "mae": round(float(mean_absolute_error(test[TARGET], np.full(len(test), mean_prediction))), 3),
        "rmse": round(float(mean_squared_error(test[TARGET], np.full(len(test), mean_prediction)) ** .5), 3),
        "r2": round(float(r2_score(test[TARGET], np.full(len(test), mean_prediction))), 3),
    }
    linear = Pipeline([("preprocessor", make_preprocessor()), ("model", LinearRegression())])
    linear.fit(train_and_validation[FEATURES], train_and_validation[TARGET])
    xgb = make_xgb_pipeline()
    xgb.fit(train_and_validation[FEATURES], train_and_validation[TARGET])
    seasons = sorted(prepared["season"].dropna().unique().tolist())
    metrics: dict[str, Any] = {
        "split": {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "historical_average": baseline_metrics,
        "linear_regression": evaluate(linear, test),
        "xgboost": evaluate(xgb, test),
        "residual_scale": _residual_scale(xgb, test),
        "per_season": _per_season_metrics(xgb, test, seasons),
        "naive_trailing_4_week": _naive_trailing_4_week_metrics(test),
        "product_holdout": _product_holdout_metrics(prepared, train, validation),
    }
    return xgb, metrics, test


def save_model(model: Pipeline, metrics: dict[str, Any], model_path, metrics_path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def load_model(model_path):
    return joblib.load(model_path)
