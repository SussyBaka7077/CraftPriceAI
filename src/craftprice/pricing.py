from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from craftprice.config import FEATURES
    from craftprice.features import add_price_features
else:
    from .config import FEATURES
    from .features import add_price_features


@dataclass(frozen=True)
class PricingConstraints:
    min_margin_pct: float = .20
    min_price: float = 0
    max_price: float = 10_000
    competitor_band_pct: float = .40
    price_step: float = 10


def _feasible_bounds(unit_cost: float, competitor_price: float,
                     constraints: PricingConstraints) -> tuple[float, float]:
    """Return the feasible price band ``[lower, upper]``.

    ``min_margin_pct`` is a margin *on price* -- the figure the app's "Minimum
    margin" slider shows, ``(price - unit_cost) / price``. The floor is therefore
    ``unit_cost / (1 - min_margin_pct)``, NOT the markup ``unit_cost * (1 + ...)``:
    the markup version would let the displayed margin-on-price fall below the
    slider value. A margin of 100% or more needs an infinite price, so reject it
    before the division.
    """
    if constraints.min_margin_pct >= 1:
        raise ValueError("Minimum margin must be below 100% (of price).")
    lower = max(
        constraints.min_price,
        unit_cost / (1 - constraints.min_margin_pct),
        competitor_price * (1 - constraints.competitor_band_pct),
    )
    upper = min(constraints.max_price, competitor_price * (1 + constraints.competitor_band_pct))
    return lower, upper


def optimize_price(model, scenario: dict, constraints: PricingConstraints) -> tuple[pd.DataFrame, pd.Series]:
    """Simulate valid discrete prices and maximise estimated contribution, not demand.

    Candidates are exactly the step-aligned prices (multiples of ``price_step``) that lie
    inside the feasible band ``[lower, upper]``. If no such price exists the function
    raises ``ValueError`` -- which app.py catches to show the seller a helpful message --
    instead of building an empty frame that would later blow up as ``KeyError: 'unit_cost'``.
    """
    step = float(constraints.price_step)
    if step <= 0:
        # A non-positive step can never define a price grid; guard the division below.
        raise ValueError("Candidate price step must be greater than zero.")

    unit_cost = float(scenario["unit_cost"])
    competitor_price = float(scenario["competitor_avg_price"])
    lower, upper = _feasible_bounds(unit_cost, competitor_price, constraints)

    # `lower`/`upper` are products of floats, so a bound that is mathematically an exact
    # step multiple can be stored a few ulps off (e.g. 350 * 1.4 == 489.99999999999994,
    # whose true value is 490). A 1e-9 tolerance *in step units* absorbs that: it is ~1e5x
    # larger than that round-off, yet ~1e9x smaller than any real price increment, so it
    # keeps a boundary-equal candidate (490.0, the true cap) without ever admitting a price
    # a seller would call outside the band.
    #
    # Every comparison below uses this same tolerance. The feasibility guard must NOT be a
    # bare `lower > upper`: the two bounds can coincide mathematically and still straddle
    # each other in float. For `unit_cost=34, min_margin=0.80, competitor=125, band=0.36,
    # step=10` the margin floor 34/0.2 and the band cap 125*1.36 are both mathematically
    # 170, yet float stores the floor as 170.00000000000003 and the cap as
    # 169.99999999999997 -- one ulp above and below the price the seller genuinely has.
    # A bare guard would reject that price. Guarding with the same tolerance lets the grid
    # below decide instead; a genuinely infeasible pair still fails there with the more
    # specific "no price in steps of N fits" message.
    tolerance = 1e-9  # in units of `step`
    if lower > upper + step * tolerance:
        raise ValueError("Constraints leave no feasible price. Widen the range or lower the minimum margin.")

    # Candidates are the step multiples k*step for integer k in [first_index, last_index].
    # Working in integer step indices (rather than ceil()-then-arange with a stop-pad) means
    # an empty grid is impossible to miss and no price can land above `upper`: every
    # k <= last_index satisfies k*step <= upper + step*tolerance.
    first_index = int(np.ceil(lower / step - tolerance))
    last_index = int(np.floor(upper / step + tolerance))
    if first_index > last_index:
        # No multiple of the step fits inside [lower, upper] (the band may be narrower than
        # the step, or rounding `lower` up overshot `upper`). Fail as ValueError so the
        # caller shows a helpful message instead of crashing on a malformed empty frame.
        raise ValueError(
            f"No price in steps of {step:g} fits inside the feasible range "
            f"[{lower:.2f}, {upper:.2f}]. Lower the candidate price step or widen the range."
        )

    prices = (np.arange(first_index, last_index + 1) * step).round(2)
    candidates = pd.DataFrame([scenario] * len(prices))
    candidates["price"] = prices
    candidates = add_price_features(candidates)
    candidates["predicted_demand"] = np.clip(model.predict(candidates[FEATURES]), 0, None)
    candidates["estimated_revenue"] = candidates["price"] * candidates["predicted_demand"]
    candidates["estimated_contribution"] = (candidates["price"] - unit_cost) * candidates["predicted_demand"]
    candidates["margin_pct"] = (candidates["price"] - unit_cost) / candidates["price"]
    recommendation = candidates.loc[candidates["estimated_contribution"].idxmax()]
    return candidates, recommendation


def nearest_price_row(candidates: pd.DataFrame, current_price: float) -> pd.Series:
    """Return the candidate row whose price is closest to ``current_price``."""
    nearest_index = (candidates["price"] - current_price).abs().argsort().iloc[0]
    return candidates.iloc[nearest_index]


def diagnose_recommendation(candidates: pd.DataFrame, recommendation: pd.Series,
                            constraints: PricingConstraints,
                            competitor_price: float) -> dict:
    """Say whether a constraint edge, not the demand model, picked the price.

    Returns a dict with the feasible bounds, flags for a recommendation sitting at
    (or within one ``price_step`` of) either end of the band, the source of the
    upper bound, and a plain-language ``note`` when an edge is binding.
    """
    unit_cost = float(recommendation["unit_cost"])
    lower, upper = _feasible_bounds(unit_cost, competitor_price, constraints)

    band_lower = competitor_price * (1 - constraints.competitor_band_pct)
    band_upper = competitor_price * (1 + constraints.competitor_band_pct)
    margin_floor = unit_cost / (1 - constraints.min_margin_pct)

    step = float(constraints.price_step)
    tolerance = 1e-9  # in units of `step`, same spirit as optimize_price
    within_step = step * (1 + tolerance)

    price = float(recommendation["price"])
    max_candidate = float(candidates["price"].max())
    min_candidate = float(candidates["price"].min())
    at_upper_bound = bool(price >= max_candidate - step * tolerance and upper - price <= within_step)
    at_lower_bound = bool(price <= min_candidate + step * tolerance and price - lower <= within_step)

    # Which constraint actually produced each bound? Dict iteration order mirrors the
    # max()/min() argument order in _feasible_bounds, so near-ties resolve identically.
    upper_sources = {"max_price": float(constraints.max_price), "competitor_band": band_upper}
    binding_upper_source = min(upper_sources, key=lambda source: upper_sources[source])
    lower_sources = {"min_price": float(constraints.min_price),
                     "margin": margin_floor, "competitor_band": band_lower}
    binding_lower_source = max(lower_sources, key=lambda source: lower_sources[source])

    band = constraints.competitor_band_pct
    margin = constraints.min_margin_pct
    note: str | None = None
    if at_upper_bound:
        if binding_upper_source == "max_price":
            note = (f"This is the highest price allowed by your cap of ₹{float(constraints.max_price):,.0f}. "
                    "The cap set this price, not the model — raise the cap to see whether "
                    "a higher price would earn more.")
        else:
            note = (f"This is the highest price inside your ±{band:.0%} competitor band. "
                    "The band set this price, not the model — widen the band to see whether "
                    "a higher price would earn more.")
    elif at_lower_bound:
        if binding_lower_source == "margin":
            note = (f"This is the lowest price that keeps your minimum margin of {margin:.0%} on price. "
                    "The margin floor set this price, not the model — lower the minimum margin to see "
                    "whether a lower price would earn more.")
        elif binding_lower_source == "min_price":
            note = (f"This is the lowest price allowed by your minimum price of "
                    f"₹{float(constraints.min_price):,.0f}. The floor set this price, not the model — "
                    "lower the minimum price to see whether a lower price would earn more.")
        else:
            note = (f"This is the lowest price inside your ±{band:.0%} competitor band. "
                    "The band set this price, not the model — widen the band to see whether "
                    "a lower price would earn more.")

    return {
        "lower": float(lower),
        "upper": float(upper),
        "at_upper_bound": at_upper_bound,
        "at_lower_bound": at_lower_bound,
        "binding_upper_source": binding_upper_source,
        "note": note,
    }


def main() -> None:
    """Run a simple example scenario from the saved model."""
    model_path = Path(__file__).resolve().parents[2] / "artifacts" / "demand_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError("Model artifacts are missing. Run `python scripts/train.py` first.")

    from craftprice.modeling import load_model

    model = load_model(model_path)
    scenario = {
        "category": "Bracelet",
        "material_type": "Beads",
        "material_cost": 120.0,
        "labour_hours": 2.5,
        "labour_cost": 180.0,
        "packaging_cost": 20.0,
        "unit_cost": 320.0,
        "seller_experience_years": 2.0,
        "seller_rating": 4.5,
        "product_rating": 4.6,
        "price": 399.0,
        "competitor_avg_price": 350.0,
        "competitor_count": 6,
        "discount_pct": 0.0,
        "season": "Festival",
        "festival_flag": "Yes",
        "category_popularity": 1.05,
        "previous_week_sales": 20.0,
        "rolling_4_week_sales": 18.0,
        "recent_growth": 0.08,
    }

    candidates, recommendation = optimize_price(model, scenario, PricingConstraints())
    print("\nTop candidate prices:")
    print(candidates[["price", "predicted_demand", "estimated_contribution"]].head(10).to_string(index=False))
    print(f"\nRecommended price: ₹{recommendation['price']:.0f}")
    print(f"Predicted demand: {recommendation['predicted_demand']:.1f} units")
    print(f"Estimated contribution: ₹{recommendation['estimated_contribution']:.0f}")


if __name__ == "__main__":
    main()
