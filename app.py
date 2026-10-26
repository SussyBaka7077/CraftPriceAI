"""Seller-facing dashboard for CraftPrice AI."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from craftprice.config import FEATURES, METRICS_PATH, MODEL_PATH
from craftprice.modeling import load_model
from craftprice.pricing import (
    PricingConstraints,
    diagnose_recommendation,
    nearest_price_row,
    optimize_price,
)

# -----------------------------------------------------------------------------
# Palette — validated CVD-safe set, module-level constants.
# -----------------------------------------------------------------------------
PALETTE = {
    "light": {
        "surface": "#fcfcfb",
        "series1": "#2a78d6",  # demand / primary
        "series2": "#eb6834",  # contribution / secondary
        "muted": "#898781",
        "gridline": "#e1e0d9",
        "axis": "#c3c2b7",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
    "dark": {
        "surface": "#1a1a19",
        "series1": "#3987e5",
        "series2": "#d95926",
        "muted": "#898781",
        "gridline": "#2c2c2a",
        "axis": "#383835",
        "good": "#0ca30c",
        "warning": "#fab219",
        "critical": "#d03b3b",
    },
}

try:
    _theme = st.config.get_option("theme.base")  # type: ignore[attr-defined]
except Exception:
    _theme = st.get_option("theme.base")
THEME = "dark" if _theme == "dark" else "light"
C = PALETTE[THEME]

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def fmt_inr(value) -> str:
    """Indian Rupee with thousands separators."""
    return f"₹{float(value):,.0f}"


def demand_interval(
    best_demand: float, residual_scale: dict
) -> tuple[float, float, float] | None:
    """Return (lower, upper, sigma) for a residual-based demand interval.

    Reads the z value from metrics.json so the same number is used for the
    label and the math.
    """
    if not residual_scale:
        return None
    required = {"intercept", "slope", "mad_to_sigma", "z_80"}
    if not required.issubset(residual_scale):
        return None
    intercept = float(residual_scale["intercept"])
    slope = float(residual_scale["slope"])
    mad_to_sigma = float(residual_scale["mad_to_sigma"])
    z = float(residual_scale["z_80"])
    sigma = max(0.0, (intercept + slope * best_demand) * mad_to_sigma)
    lower = max(0.0, best_demand - z * sigma)
    upper = best_demand + z * sigma
    return lower, upper, sigma


def make_demand_chart(
    candidates: pd.DataFrame,
    best: pd.Series,
    current_row: pd.Series,
    diag: dict,
) -> go.Figure:
    fig = go.Figure()

    # Feasible band as a subtle shaded region.
    y_max = float(candidates["predicted_demand"].max()) * 1.15
    fig.add_shape(
        type="rect",
        x0=diag["lower"],
        x1=diag["upper"],
        y0=0,
        y1=y_max if y_max > 0 else 1,
        fillcolor=C["series1"],
        opacity=0.06,
        layer="below",
        line_width=0,
    )

    # Demand curve.
    fig.add_trace(
        go.Scatter(
            x=candidates["price"],
            y=candidates["predicted_demand"],
            mode="lines+markers",
            name="Predicted demand",
            line=dict(color=C["series1"], width=2.5),
            marker=dict(size=9, color=C["series1"]),
            hovertemplate="Price %{x:,.0f}<br>Demand %{y:.1f} units<extra></extra>",
        )
    )

    # Current-price marker.
    fig.add_trace(
        go.Scatter(
            x=[current_row["price"]],
            y=[current_row["predicted_demand"]],
            mode="markers",
            name="Your current price",
            marker=dict(
                symbol="diamond",
                size=12,
                color=C["series2"],
                line=dict(color=C["surface"], width=1.5),
            ),
            hovertemplate="Current price %{x:,.0f}<br>Demand %{y:.1f} units<extra></extra>",
        )
    )

    # Direct labels.
    fig.add_annotation(
        x=best["price"],
        y=best["predicted_demand"],
        text=f"Recommended<br>{fmt_inr(best['price'])}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=C["series1"],
        ax=0,
        ay=-45,
        font=dict(color=C["series1"], size=11),
        bgcolor=C["surface"],
        bordercolor=C["series1"],
        borderwidth=1,
    )
    fig.add_annotation(
        x=current_row["price"],
        y=current_row["predicted_demand"],
        text=f"Your price<br>{fmt_inr(current_row['price'])}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=C["series2"],
        ax=45,
        ay=35,
        font=dict(color=C["series2"], size=11),
        bgcolor=C["surface"],
        bordercolor=C["series2"],
        borderwidth=1,
    )

    _apply_chart_layout(fig, "Predicted demand vs candidate price", ylabel="Demand (units)")
    return fig


def make_contribution_chart(
    candidates: pd.DataFrame,
    best: pd.Series,
    current_row: pd.Series,
    diag: dict,
) -> go.Figure:
    fig = go.Figure()

    # Feasible band shading.
    y_max = float(candidates["estimated_contribution"].max()) * 1.15
    fig.add_shape(
        type="rect",
        x0=diag["lower"],
        x1=diag["upper"],
        y0=0,
        y1=y_max if y_max > 0 else 1,
        fillcolor=C["series2"],
        opacity=0.06,
        layer="below",
        line_width=0,
    )

    # Contribution curve.
    fig.add_trace(
        go.Scatter(
            x=candidates["price"],
            y=candidates["estimated_contribution"],
            mode="lines+markers",
            name="Estimated contribution",
            line=dict(color=C["series2"], width=2.5),
            marker=dict(size=9, color=C["series2"]),
            hovertemplate="Price %{x:,.0f}<br>Contribution %{y:,.0f}<extra></extra>",
        )
    )

    # Current-price marker.
    fig.add_trace(
        go.Scatter(
            x=[current_row["price"]],
            y=[current_row["estimated_contribution"]],
            mode="markers",
            name="Your current price",
            marker=dict(
                symbol="diamond",
                size=12,
                color=C["series1"],
                line=dict(color=C["surface"], width=1.5),
            ),
            hovertemplate="Current price %{x:,.0f}<br>Contribution %{y:,.0f}<extra></extra>",
        )
    )

    # Direct label for the argmax.
    fig.add_annotation(
        x=best["price"],
        y=best["estimated_contribution"],
        text=f"Best contribution<br>{fmt_inr(best['estimated_contribution'])}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=C["series2"],
        ax=0,
        ay=-45,
        font=dict(color=C["series2"], size=11),
        bgcolor=C["surface"],
        bordercolor=C["series2"],
        borderwidth=1,
    )
    fig.add_annotation(
        x=current_row["price"],
        y=current_row["estimated_contribution"],
        text=f"Your price<br>{fmt_inr(current_row['estimated_contribution'])}",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=1.5,
        arrowcolor=C["series1"],
        ax=45,
        ay=35,
        font=dict(color=C["series1"], size=11),
        bgcolor=C["surface"],
        bordercolor=C["series1"],
        borderwidth=1,
    )

    _apply_chart_layout(
        fig, "Estimated contribution vs candidate price", ylabel="Contribution (₹)"
    )
    return fig


def _apply_chart_layout(fig: go.Figure, title: str, ylabel: str) -> None:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=C["axis"])),
        font=dict(family="system-ui, -apple-system, BlinkMacSystemFont, sans-serif", color=C["muted"]),
        paper_bgcolor=C["surface"],
        plot_bgcolor=C["surface"],
        xaxis=dict(
            title=dict(text="Candidate price (₹)", font=dict(color=C["muted"])),
            gridcolor=C["gridline"],
            linecolor=C["axis"],
            tickcolor=C["axis"],
            tickfont=dict(color=C["muted"]),
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=ylabel, font=dict(color=C["muted"])),
            gridcolor=C["gridline"],
            linecolor=C["axis"],
            tickcolor=C["axis"],
            tickfont=dict(color=C["muted"]),
            zeroline=False,
        ),
        hovermode="x unified",
        margin=dict(l=55, r=30, t=55, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color=C["muted"]),
            bgcolor="rgba(0,0,0,0)",
        ),
    )


# -----------------------------------------------------------------------------
# Page setup
# -----------------------------------------------------------------------------
st.set_page_config(page_title="CraftPrice AI", page_icon="🏷️", layout="wide")
st.title("CraftPrice AI")
st.caption("Demand prediction + constrained price simulation for handmade-product micro-sellers")

if not MODEL_PATH.exists():
    st.error("Model artifacts are missing. Run `python scripts/train.py` first.")
    st.stop()


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


with st.spinner("Loading demand model…"):
    model = get_model()

# -----------------------------------------------------------------------------
# Sidebar inputs
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Product & market input")
    category = st.selectbox("Category", ["Bracelet", "Jewellery", "Candle", "Macrame", "Customized Gift", "Handmade Craft"])
    material = st.selectbox("Material type", ["Beads", "Silver", "Soy Wax", "Cotton", "Resin", "Wood"])
    material_cost = st.number_input("Material cost (₹)", 0.0, 5000.0, 120.0, 10.0)
    labour_hours = st.number_input("Labour hours", 0.0, 50.0, 2.5, 0.25)
    labour_cost = st.number_input("Labour cost (₹)", 0.0, 5000.0, 180.0, 10.0)
    packaging_cost = st.number_input("Packaging cost (₹)", 0.0, 1000.0, 20.0, 5.0)
    current_price = st.number_input("Current price (₹)", 1.0, 10000.0, 399.0, 10.0)
    competitor_price = st.number_input("Competitor average price (₹)", 1.0, 10000.0, 350.0, 10.0)
    product_rating = st.slider("Product rating", 1.0, 5.0, 4.6, 0.1)
    seller_rating = st.slider("Seller rating", 1.0, 5.0, 4.5, 0.1)
    experience = st.number_input("Seller experience (years)", 0.0, 50.0, 2.0, 0.5)
    competitor_count = st.number_input("Competitor count", 0, 100, 6)
    season = st.selectbox("Season", ["Festival", "Winter", "Summer", "Monsoon"])
    popularity = st.slider("Category popularity index", 0.5, 1.5, 1.05, 0.01)
    previous_sales = st.number_input("Previous-week sales", 0.0, 10000.0, 20.0)
    rolling_sales = st.number_input("Rolling 4-week average sales", 0.0, 10000.0, 18.0)
    recent_growth = st.slider("Recent demand growth", -1.0, 1.0, 0.08, 0.01)

    st.divider()
    st.header("Optimisation constraints")
    discount_label = st.selectbox(
        "Promotional discount",
        ["No discount", "5% off", "10% off"],
        help="Discounts were present in training data but their measured effect on predicted demand is usually small.",
    )
    min_margin = st.slider("Minimum margin on price", 0, 90, 20) / 100
    band = st.slider("Competitor-relative band", 5, 100, 40) / 100
    price_step = st.number_input("Candidate price step (₹)", 1.0, 500.0, 10.0)

discount_map = {"No discount": 0.0, "5% off": 0.05, "10% off": 0.10}
discount_pct = discount_map[discount_label]

# -----------------------------------------------------------------------------
# Scenario and optimisation
# -----------------------------------------------------------------------------
unit_cost = material_cost + labour_cost + packaging_cost
scenario = {
    "category": category,
    "material_type": material,
    "material_cost": material_cost,
    "labour_hours": labour_hours,
    "labour_cost": labour_cost,
    "packaging_cost": packaging_cost,
    "unit_cost": unit_cost,
    "seller_experience_years": experience,
    "seller_rating": seller_rating,
    "product_rating": product_rating,
    "price": current_price,
    "competitor_avg_price": competitor_price,
    "competitor_count": competitor_count,
    "discount_pct": discount_pct,
    "season": season,
    "festival_flag": "Yes" if season == "Festival" else "No",
    "category_popularity": popularity,
    "previous_week_sales": previous_sales,
    "rolling_4_week_sales": rolling_sales,
    "recent_growth": recent_growth,
}

try:
    candidates, best = optimize_price(
        model,
        scenario,
        PricingConstraints(
            min_margin_pct=min_margin,
            competitor_band_pct=band,
            price_step=price_step,
        ),
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()

constraints = PricingConstraints(
    min_margin_pct=min_margin, competitor_band_pct=band, price_step=price_step
)
current_row = nearest_price_row(candidates, current_price)
diag = diagnose_recommendation(candidates, best, constraints, competitor_price)

# -----------------------------------------------------------------------------
# Metrics and attribution data
# -----------------------------------------------------------------------------
metrics: dict = {}
if METRICS_PATH.exists():
    metrics = json.loads(METRICS_PATH.read_text())

residual_scale = metrics.get("residual_scale", {})
interval = demand_interval(float(best["predicted_demand"]), residual_scale)

# Discount effect: compare chosen discount with no discount at the recommended row.
base_row = best.copy()
base_row["discount_pct"] = 0.0
base_demand = float(model.predict(pd.DataFrame([base_row])[FEATURES])[0])
selected_demand = float(best["predicted_demand"])
if base_demand > 0:
    discount_delta_pct = (selected_demand - base_demand) / base_demand * 100
else:
    discount_delta_pct = 0.0

# -----------------------------------------------------------------------------
# Hero KPI row
# -----------------------------------------------------------------------------
price_delta = float(best["price"]) - float(current_price)
if interval:
    lower, upper, _ = interval
    demand_text = f"{int(round(selected_demand))} units"
else:
    demand_text = f"{selected_demand:.1f} units"

rec_col, demand_col, contrib_col, cost_col = st.columns([1.25, 1.15, 1.0, 0.9])
rec_col.metric(
    "Recommended price",
    fmt_inr(best["price"]),
    f"₹{price_delta:+,.0f} vs current",
    delta_color="off",
)
demand_col.metric("Approx. weekly demand", demand_text)
if interval:
    demand_col.caption(f"80% interval: {int(lower)}–{int(upper)} units")
contrib_col.metric("Estimated contribution", fmt_inr(best["estimated_contribution"]))
cost_col.metric("Unit cost", fmt_inr(unit_cost))

if discount_pct > 0:
    st.caption(
        f"With the selected {discount_pct:.0%} discount, predicted demand is **{discount_delta_pct:+.2f}%** "
        "versus no discount. The measured effect of discounts is usually small."
    )

if diag.get("note"):
    st.info(diag["note"])

# -----------------------------------------------------------------------------
# Charts
# -----------------------------------------------------------------------------
if len(candidates) == 1:
    st.subheader("Single feasible price")
    st.write(
        f"Your constraints allow a single price: **{fmt_inr(candidates['price'].iloc[0])}**. "
        "Widen the competitor band or lower the minimum margin to see a grid of options."
    )
else:
    col_a, col_b = st.columns(2)
    with col_a:
        st.plotly_chart(make_demand_chart(candidates, best, current_row, diag), width="stretch")
    with col_b:
        st.plotly_chart(make_contribution_chart(candidates, best, current_row, diag), width="stretch")

# -----------------------------------------------------------------------------
# Candidate-price table
# -----------------------------------------------------------------------------
st.subheader("All feasible price scenarios")

if len(candidates) > 60:
    st.warning(
        f"Your grid has {len(candidates)} candidate prices, so the table below is long. "
        "Consider increasing the price step to see the shape more clearly."
    )

display = candidates.copy()
step = float(price_step)
marks = []
for _, row in display.iterrows():
    parts = []
    if row.name == best.name:
        parts.append("← recommended")
    if row.name == current_row.name:
        parts.append("≈ your current price")
    if abs(float(row["price"]) - diag["lower"]) <= step / 2 + 1e-9:
        parts.append("band floor")
    if abs(float(row["price"]) - diag["upper"]) <= step / 2 + 1e-9:
        parts.append("band ceiling")
    marks.append(", ".join(parts))
display["mark"] = marks
display["margin_pct"] = display["margin_pct"] * 100  # rendered with a %% format below

table_cols = ["price", "predicted_demand", "estimated_contribution", "margin_pct", "mark"]
st.dataframe(
    display[table_cols],
    column_config={
        "price": st.column_config.NumberColumn("Price", format="₹%,.0f", width="small"),
        "predicted_demand": st.column_config.NumberColumn("Demand", format="%.1f", width="small"),
        "estimated_contribution": st.column_config.NumberColumn("Contribution", format="₹%,.0f"),
        "margin_pct": st.column_config.NumberColumn("Margin (%)", format="%.1f%%", width="small"),
        "mark": st.column_config.TextColumn("Mark", width="medium"),
    },
    width="stretch",
    hide_index=True,
)
st.caption(
    f"Feasible band: {fmt_inr(diag['lower'])}–{fmt_inr(diag['upper'])}. "
    "Marks: ← recommended, ≈ your current price, band floor/ceiling. "
    "Contribution = (price − unit cost) × predicted demand."
)

# CSV download.
csv_bytes = candidates.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download candidate table as CSV",
    data=csv_bytes,
    file_name="craftprice_candidates.csv",
    mime="text/csv",
)

# -----------------------------------------------------------------------------
# Model evaluation panel
# -----------------------------------------------------------------------------
st.subheader("Model evaluation")

model_mtime = datetime.fromtimestamp(MODEL_PATH.stat().st_mtime)
metrics_mtime = datetime.fromtimestamp(METRICS_PATH.stat().st_mtime)
split_block = metrics.get("split", {})
split_text = (
    f"train={split_block.get('train_rows', '?')}, "
    f"validation={split_block.get('validation_rows', '?')}, "
    f"test={split_block.get('test_rows', '?')}"
)
attribution = (
    f"Split: chronological holdout ({split_text}). "
    f"Model file: {model_mtime:%Y-%m-%d %H:%M}. "
    f"Metrics file: {metrics_mtime:%Y-%m-%d %H:%M}."
)
if metrics_mtime < model_mtime:
    attribution += " **The metrics file is older than the model file — these numbers may be stale.**"
st.caption(attribution)

per_season = metrics.get("per_season", {})

# Build the honest evaluation table.
eval_rows = []
main = metrics.get("xgboost")
if main:
    eval_rows.append(
        {
            "Scope": "XGBoost (chronological holdout)",
            "MAE": main["mae"],
            "RMSE": main["rmse"],
            "R²": main["r2"],
            "Note": "",
        }
    )

for season_name in ["Festival", "Winter", "Summer", "Monsoon"]:
    block = per_season.get(season_name)
    if block is None:
        continue
    if block.get("status") == "not_evaluated":
        eval_rows.append(
            {
                "Scope": f"Season: {season_name} — not evaluated",
                "MAE": float("nan"),
                "RMSE": float("nan"),
                "R²": float("nan"),
                "Note": block.get("note", ""),
            }
        )
    else:
        eval_rows.append(
            {
                "Scope": f"Season: {season_name}",
                "MAE": block["mae"],
                "RMSE": block["rmse"],
                "R²": block["r2"],
                "Note": "",
            }
        )

naive = metrics.get("naive_trailing_4_week")
if naive:
    eval_rows.append(
        {
            "Scope": "Naive baseline: trailing 4-week mean",
            "MAE": naive["mae"],
            "RMSE": naive["rmse"],
            "R²": naive["r2"],
            "Note": "",
        }
    )

ph = metrics.get("product_holdout")
if ph:
    eval_rows.append(
        {
            "Scope": "Product holdout (unseen products)",
            "MAE": ph["mae"],
            "RMSE": ph["rmse"],
            "R²": ph["r2"],
            "Note": f"{ph.get('n_holdout_products', '?')} products, {ph.get('n_holdout_rows', '?')} rows",
        }
    )

eval_df = pd.DataFrame(eval_rows)
st.dataframe(
    eval_df,
    column_config={
        "Scope": st.column_config.TextColumn("Scope", width="large"),
        "MAE": st.column_config.NumberColumn("MAE", format="%.3f", width="small"),
        "RMSE": st.column_config.NumberColumn("RMSE", format="%.3f", width="small"),
        "R²": st.column_config.NumberColumn("R²", format="%.3f", width="small"),
        "Note": st.column_config.TextColumn("Note", width="medium"),
    },
    width="stretch",
    hide_index=True,
)
