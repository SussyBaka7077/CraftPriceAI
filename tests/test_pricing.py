import pandas as pd
from craftprice.pricing import PricingConstraints, optimize_price


class DemandModel:
    def predict(self, frame):
        return 60 - .1 * frame["price"]


def scenario():
    return {"category": "Bracelet", "material_type": "Beads", "material_cost": 120,
        "labour_hours": 2, "labour_cost": 140, "packaging_cost": 20, "unit_cost": 280,
        "seller_experience_years": 2, "seller_rating": 4.5, "product_rating": 4.6,
        "price": 400, "competitor_avg_price": 400, "competitor_count": 5, "discount_pct": 0,
        "season": "Festival", "festival_flag": "Yes", "category_popularity": 1.1,
        "previous_week_sales": 20, "rolling_4_week_sales": 18, "recent_growth": .1}


def test_optimizer_only_returns_feasible_prices():
    table, best = optimize_price(DemandModel(), scenario(), PricingConstraints(min_margin_pct=.2, price_step=10))
    assert not table.empty
    assert (table["price"] >= 336).all()
    assert best["estimated_contribution"] == table["estimated_contribution"].max()

