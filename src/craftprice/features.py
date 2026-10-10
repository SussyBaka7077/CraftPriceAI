from __future__ import annotations
import pandas as pd


def add_price_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create price-derived features; call after every candidate-price substitution."""
    out = frame.copy()
    out["price_to_cost"] = out["price"] / out["unit_cost"].clip(lower=1)
    out["competitor_price_gap"] = out["price"] - out["competitor_avg_price"]
    out["estimated_contribution_per_unit"] = out["price"] - out["unit_cost"]
    return out
