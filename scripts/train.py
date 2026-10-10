"""Create the synthetic dataset and train/evaluate CraftPrice AI."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from craftprice.config import DATA_DIR, DATA_PATH, METRICS_PATH, MODEL_PATH
from craftprice.data import generate_synthetic_data
from craftprice.modeling import save_model, train_models


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    data = generate_synthetic_data()
    data.to_csv(DATA_PATH, index=False)
    model, metrics, _ = train_models(data)
    save_model(model, metrics, MODEL_PATH, METRICS_PATH)
    print(f"Wrote {len(data)} clearly-labelled synthetic product-week rows to {DATA_PATH}")
    print(f"Saved model to {MODEL_PATH}")
    for name, result in metrics.items():
        if isinstance(result, dict) and "mae" in result:
            print(f"{name}: MAE={result['mae']}, RMSE={result['rmse']}, R²={result['r2']}")


if __name__ == "__main__":
    main()
