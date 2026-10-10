"""Generate a transparent, domain-informed synthetic product-week dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORIES = {
    "Bracelet": (18, 260, 1.55), "Jewellery": (16, 480, 1.35),
    "Candle": (21, 340, 1.15), "Macrame": (11, 650, 1.00),
    "Customized Gift": (14, 520, 1.25), "Handmade Craft": (13, 430, 1.10),
}
MATERIALS = ["Beads", "Silver", "Soy Wax", "Cotton", "Resin", "Wood"]
FESTIVAL_MONTHS = {10, 11, 12}


def _season(month: int) -> str:
    if month in (3, 4, 5, 6):
        return "Summer"
    if month in (7, 8, 9):
        return "Monsoon"
    if month in (10, 11, 12):
        return "Festival"
    return "Winter"


def generate_synthetic_data(n_products: int = 180, n_weeks: int = 70, seed: int = 42) -> pd.DataFrame:
    """Return sequential product-week data with stated synthetic assumptions.

    Demand has a category baseline and responds to rating, season, price, relative
    competitor price, recent momentum, and random noise. Products vary in price
    sensitivity, preventing a single fixed demand equation from determining output.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n_weeks, freq="W-MON")
    records: list[dict] = []
    for index in range(n_products):
        category = rng.choice(list(CATEGORIES))
        base_demand, typical_price, base_elasticity = CATEGORIES[category]
        material_cost = float(rng.uniform(45, 190))
        labour_hours = float(rng.uniform(0.5, 5.5))
        labour_cost = labour_hours * float(rng.uniform(55, 95))
        packaging_cost = float(rng.uniform(8, 35))
        unit_cost = material_cost + labour_cost + packaging_cost
        product_rating = float(rng.uniform(3.5, 5.0))
        seller_rating = float(rng.uniform(3.6, 5.0))
        experience = float(rng.uniform(0.2, 9.0))
        popularity = float(rng.uniform(0.65, 1.35))
        elasticity = float(base_elasticity * rng.uniform(0.7, 1.35))
        prior_sales: list[float] = [max(1, base_demand * rng.uniform(0.6, 1.2))]
        for week, date in enumerate(dates):
            season = _season(date.month)
            festival = int(date.month in FESTIVAL_MONTHS)
            competitor = max(unit_cost * 1.12, typical_price * rng.normal(1, 0.10))
            discount = float(rng.choice([0, 0, 0, 0.05, 0.10]))
            listed_price = max(unit_cost * 1.07, competitor * rng.normal(1.02, 0.14))
            price = listed_price * (1 - discount)
            previous = prior_sales[-1]
            rolling = float(np.mean(prior_sales[-4:]))
            growth = 0.0 if len(prior_sales) < 2 else (previous - prior_sales[-2]) / max(prior_sales[-2], 1)
            season_lift = {"Festival": 1.32, "Winter": 1.08, "Summer": 0.96, "Monsoon": 0.91}[season]
            # Relative price is used so the same 10% increase has a comparable
            # directional effect across inexpensive and premium categories.
            price_effect = np.exp(-elasticity * (price / competitor - 1))
            rating_effect = 1 + 0.12 * (product_rating - 4.0) + 0.05 * (seller_rating - 4.0)
            momentum_effect = 1 + np.clip(growth, -0.25, 0.25) * 0.25
            expected = base_demand * popularity * season_lift * rating_effect * price_effect * momentum_effect
            units = int(max(0, round(expected + rng.normal(0, 2.8 + expected * 0.14))))
            records.append({
                "product_id": f"P{index + 1:03d}", "week_start": date, "category": category,
                "material_type": rng.choice(MATERIALS), "material_cost": round(material_cost, 2),
                "labour_hours": round(labour_hours, 2), "labour_cost": round(labour_cost, 2),
                "packaging_cost": round(packaging_cost, 2), "unit_cost": round(unit_cost, 2),
                "seller_experience_years": round(experience, 2), "seller_rating": round(seller_rating, 2),
                "product_rating": round(product_rating, 2), "price": round(price, 2),
                "competitor_avg_price": round(competitor, 2), "competitor_count": int(rng.integers(2, 16)),
                "discount_pct": discount, "season": season, "festival_flag": "Yes" if festival else "No",
                "category_popularity": round(popularity, 3), "previous_week_sales": round(previous, 2),
                "rolling_4_week_sales": round(rolling, 2), "recent_growth": round(growth, 3), "units_sold": units,
            })
            prior_sales.append(units)
    return pd.DataFrame(records)
