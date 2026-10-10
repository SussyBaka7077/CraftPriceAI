from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
DATA_PATH = DATA_DIR / "synthetic_product_weeks.csv"
MODEL_PATH = ARTIFACT_DIR / "demand_model.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"

TARGET = "units_sold"
DATE_COLUMN = "week_start"

NUMERIC_FEATURES = [
    "material_cost", "labour_hours", "labour_cost", "packaging_cost", "unit_cost",
    "seller_experience_years", "seller_rating", "product_rating", "price",
    "competitor_avg_price", "competitor_count", "discount_pct", "category_popularity",
    "previous_week_sales", "rolling_4_week_sales", "recent_growth", "price_to_cost",
    "competitor_price_gap", "estimated_contribution_per_unit",
]
CATEGORICAL_FEATURES = ["category", "material_type", "season", "festival_flag"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
