import pytest
import pandas as pd
from craftprice.pricing import (PricingConstraints, optimize_price,
                                nearest_price_row, diagnose_recommendation)


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
    assert (table["price"] >= 280 / .8).all()
    assert best["estimated_contribution"] == table["estimated_contribution"].max()


def test_margin_floor_is_measured_on_price_for_all_margins():
    high_market = scenario()
    high_market["competitor_avg_price"] = 3000
    for margin in (.2, .5, .9):
        table, _ = optimize_price(DemandModel(), high_market,
                                  PricingConstraints(min_margin_pct=margin, price_step=10))
        assert not table.empty
        assert ((table["price"] - 280) / table["price"] >= margin - 1e-9).all()


def test_min_margin_at_or_above_100_percent_is_rejected():
    with pytest.raises(ValueError, match="must be below 100%"):
        optimize_price(DemandModel(), scenario(), PricingConstraints(min_margin_pct=1.0, price_step=10))
    with pytest.raises(ValueError, match="must be below 100%"):
        optimize_price(DemandModel(), scenario(), PricingConstraints(min_margin_pct=1.2, price_step=10))


def test_nearest_price_row_finds_closest_candidate():
    table, _ = optimize_price(DemandModel(), scenario(), PricingConstraints(min_margin_pct=.2, price_step=10))
    row = nearest_price_row(table, current_price=414.0)
    assert row["price"] == 410
    assert row["estimated_contribution"] == table.loc[table["price"] == 410, "estimated_contribution"].iloc[0]


def test_diagnose_at_band_edge_reports_the_constraint_as_source():
    s = scenario()
    s["unit_cost"] = 200
    s["competitor_avg_price"] = 300
    constraints = PricingConstraints(min_margin_pct=.2, competitor_band_pct=.15, price_step=10)
    table, best = optimize_price(DemandModel(), s, constraints)
    diag = diagnose_recommendation(table, best, constraints, competitor_price=300)
    assert diag["lower"] == pytest.approx(max(0, 200 / .8, 300 * .85))
    assert diag["upper"] == pytest.approx(min(10_000, 300 * 1.15))
    assert diag["at_upper_bound"] is True
    assert diag["at_lower_bound"] is False
    assert diag["binding_upper_source"] == "competitor_band"
    assert diag["note"] is not None
    assert "±15%" in diag["note"]


def test_diagnose_interior_optimum_has_no_note():
    s = scenario()
    s["unit_cost"] = 200
    s["competitor_avg_price"] = 300
    constraints = PricingConstraints(min_margin_pct=.2, competitor_band_pct=.6, price_step=10)
    table, best = optimize_price(DemandModel(), s, constraints)
    diag = diagnose_recommendation(table, best, constraints, competitor_price=300)
    assert diag["at_upper_bound"] is False
    assert diag["at_lower_bound"] is False
    assert diag["binding_upper_source"] == "competitor_band"
    assert diag["note"] is None


def test_optimizer_enforces_monotonic_demand_and_extrapolation_decay():
    class FlatTailTreeModel:
        def predict(self, frame):
            # Simulates a tree model that wiggles upward and flatlines at high prices
            return pd.Series(20.0, index=frame.index).where(frame["price"] < 460, 20.5)

    s = scenario()
    s["unit_cost"] = 320
    s["competitor_avg_price"] = 350
    constraints = PricingConstraints(min_margin_pct=.2, competitor_band_pct=.4, price_step=10)
    table, best = optimize_price(FlatTailTreeModel(), s, constraints)
    assert table["predicted_demand"].is_monotonic_decreasing
    assert best["price"] < table["price"].max()
