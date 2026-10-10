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
    xgb = Pipeline([
        ("preprocessor", make_preprocessor()),
        ("model", XGBRegressor(n_estimators=450, max_depth=5, learning_rate=.045,
                                subsample=.85, colsample_bytree=.9, reg_lambda=2,
                                objective="reg:squarederror", random_state=42, n_jobs=2)),
    ])
    xgb.fit(train_and_validation[FEATURES], train_and_validation[TARGET])
    metrics: dict[str, Any] = {
        "split": {"train_rows": len(train), "validation_rows": len(validation), "test_rows": len(test)},
        "historical_average": baseline_metrics,
        "linear_regression": evaluate(linear, test),
        "xgboost": evaluate(xgb, test),
    }
    return xgb, metrics, test


def save_model(model: Pipeline, metrics: dict[str, Any], model_path, metrics_path) -> None:
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2))


def load_model(model_path):
    return joblib.load(model_path)
