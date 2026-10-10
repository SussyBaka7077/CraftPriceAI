# CraftPrice AI — compact project understanding

## 1) What this project is
This repo is a small ML + pricing demo for handmade-product micro-sellers. It predicts weekly demand for a product at a candidate price, then simulates feasible prices and recommends the one with the highest expected contribution.

Core business formula:
- estimated contribution = (candidate price - unit cost) × predicted weekly demand
- The model is a demand predictor, not a causal pricing engine.
- The app is a decision-support tool with constraints such as minimum margin and competitor band.

## 2) Repository map
- [app.py](app.py): Streamlit dashboard for seller inputs and recommendation display.
- [scripts/train.py](scripts/train.py): generates synthetic data, trains model, saves artifacts.
- [src/craftprice/config.py](src/craftprice/config.py): all paths, feature lists, target names.
- [src/craftprice/data.py](src/craftprice/data.py): synthetic product-week data generator.
- [src/craftprice/features.py](src/craftprice/features.py): adds derived price features.
- [src/craftprice/modeling.py](src/craftprice/modeling.py): chronological train/validation/test split, model training, metrics, model load/save.
- [src/craftprice/pricing.py](src/craftprice/pricing.py): candidate price simulator and optimizer.
- [tests/test_pricing.py](tests/test_pricing.py): tests optimizer feasibility and best-price selection.
- [data/synthetic_product_weeks.csv](data/synthetic_product_weeks.csv): generated dataset.
- [artifacts/demand_model.joblib](artifacts/demand_model.joblib): trained model artifact.
- [artifacts/metrics.json](artifacts/metrics.json): evaluation metrics.

## 3) Data and model flow
1. [scripts/train.py](scripts/train.py) creates synthetic weekly product data.
2. [src/craftprice/data.py](src/craftprice/data.py) builds a product-week dataset with price, season, rating, competitor, and momentum effects.
3. [src/craftprice/modeling.py](src/craftprice/modeling.py) adds price features, splits chronologically, trains a baseline and XGBoost regressor.
4. [src/craftprice/pricing.py](src/craftprice/pricing.py) generates feasible candidate prices, predicts demand for each, and selects the highest contribution.
5. [app.py](app.py) exposes this to a user as an interactive pricing recommendation dashboard.

## 4) Important implementation details
- Feature set is centralized in [src/craftprice/config.py](src/craftprice/config.py): `FEATURES`, `NUMERIC_FEATURES`, `CATEGORICAL_FEATURES`, and target column `units_sold`.
- Training uses a time-based split instead of random split so the model is evaluated on future unseen data.
- Price simulation is discrete and constrained:
  - min margin percent
  - competitor-relative band
  - candidate price step
- `add_price_features()` in [src/craftprice/features.py](src/craftprice/features.py) must be called after any candidate-price substitution before predicting demand.

## 5) Core logic to respect when editing
- Keep feature names consistent with [src/craftprice/config.py](src/craftprice/config.py). If a new column is added to the data or pricing candidate, also update features/config.
- Do not change the optimization objective away from contribution-maximization unless the project goal changes.
- The repo is synthetic-data demo code: all predictions are estimates, not verified market truth.
- When modifying pricing logic, check both [src/craftprice/pricing.py](src/craftprice/pricing.py) and [tests/test_pricing.py](tests/test_pricing.py).

## 6) Fastest way to work in this repo
- Train model: `python scripts/train.py`
- Run app: `streamlit run app.py`
- Run tests: `pytest -q`

## 7) Recommended files to read for a specific task
- Want to understand the dataset: [src/craftprice/data.py](src/craftprice/data.py)
- Want to understand model training/evaluation: [src/craftprice/modeling.py](src/craftprice/modeling.py)
- Want to understand recommendation logic: [src/craftprice/pricing.py](src/craftprice/pricing.py)
- Want to understand UI flow: [app.py](app.py)
- Want to understand project assumptions: [README.md](README.md)

## 8) One-sentence summary for an AI agent
This project is a compact synthetic-demand forecasting and price-optimization demo where a seller enters product context, a trained model predicts weekly demand, and the app searches feasible prices to maximize estimated contribution under seller-defined constraints.
