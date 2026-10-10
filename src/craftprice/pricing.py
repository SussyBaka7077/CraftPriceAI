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
    lower = max(constraints.min_price, unit_cost * (1 + constraints.min_margin_pct),
                competitor_price * (1 - constraints.competitor_band_pct))
    upper = min(constraints.max_price, competitor_price * (1 + constraints.competitor_band_pct))

    # `lower`/`upper` are products of floats, so a bound that is mathematically an exact
    # step multiple can be stored a few ulps off (e.g. 350 * 1.4 == 489.99999999999994,
    # whose true value is 490). A 1e-9 tolerance *in step units* absorbs that: it is ~1e5x
    # larger than that round-off, yet ~1e9x smaller than any real price increment, so it
    # keeps a boundary-equal candidate (490.0, the true cap) without ever admitting a price
    # a seller would call outside the band.
    #
    # Every comparison below uses this same tolerance. The feasibility guard must NOT be a
    # bare `lower > upper`: the two bounds can coincide mathematically and still straddle
    # each other in float. For `unit_cost=100, min_margin=0.70, competitor=125,
    # band=0.36, step=10` both bounds describe the single price 170 (125 * 1.36 is exactly
    # 170 in decimal), yet `upper` stores as 169.99999999999997 -- one ulp below `lower`.
    # A bare guard would reject a price the seller genuinely has. Guarding with the same
    # tolerance lets the grid below decide instead; a genuinely infeasible pair still fails
    # there with the more specific "no price in steps of N fits" message.
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
    print(f"Predicted demand range: {candidates['predicted_demand'].min():.0f} - {candidates['predicted_demand'].max():.0f}")
    candidates["estimated_revenue"] = candidates["price"] * candidates["predicted_demand"]
    candidates["estimated_contribution"] = (candidates["price"] - unit_cost) * candidates["predicted_demand"]
    candidates["margin_pct"] = (candidates["price"] - unit_cost) / candidates["price"]
    recommendation = candidates.loc[candidates["estimated_contribution"].idxmax()]
    return candidates, recommendation


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
