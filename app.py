"""Seller-facing dashboard for CraftPrice AI."""
from __future__ import annotations
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from craftprice.config import METRICS_PATH, MODEL_PATH
from craftprice.modeling import load_model
from craftprice.pricing import PricingConstraints, optimize_price

st.set_page_config(page_title="CraftPrice AI", page_icon="🏷️", layout="wide")
st.title("CraftPrice AI")
st.caption("Demand prediction + constrained price simulation for handmade-product micro-sellers")

if not MODEL_PATH.exists():
    st.error("Model artifacts are missing. Run `python scripts/train.py` first.")
    st.stop()

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)

model = get_model()

with st.sidebar:
    st.header("Product & market input")
    category = st.selectbox("Category", ["Bracelet", "Jewellery", "Candle", "Macrame", "Customized Gift", "Handmade Craft"])
    material = st.selectbox("Material type", ["Beads", "Silver", "Soy Wax", "Cotton", "Resin", "Wood"])
    material_cost = st.number_input("Material cost (₹)", 0.0, 5000.0, 120.0, 10.0)
    labour_hours = st.number_input("Labour hours", 0.0, 50.0, 2.5, .25)
    labour_cost = st.number_input("Labour cost (₹)", 0.0, 5000.0, 180.0, 10.0)
    packaging_cost = st.number_input("Packaging cost (₹)", 0.0, 1000.0, 20.0, 5.0)
    current_price = st.number_input("Current price (₹)", 1.0, 10000.0, 399.0, 10.0)
    competitor_price = st.number_input("Competitor average price (₹)", 1.0, 10000.0, 350.0, 10.0)
    product_rating = st.slider("Product rating", 1.0, 5.0, 4.6, .1)
    seller_rating = st.slider("Seller rating", 1.0, 5.0, 4.5, .1)
    experience = st.number_input("Seller experience (years)", 0.0, 50.0, 2.0, .5)
    competitor_count = st.number_input("Competitor count", 0, 100, 6)
    season = st.selectbox("Season", ["Festival", "Winter", "Summer", "Monsoon"])
    popularity = st.slider("Category popularity index", .5, 1.5, 1.05, .01)
    previous_sales = st.number_input("Previous-week sales", 0.0, 10000.0, 20.0)
    rolling_sales = st.number_input("Rolling 4-week average sales", 0.0, 10000.0, 18.0)
    recent_growth = st.slider("Recent demand growth", -1.0, 1.0, .08, .01)

    st.divider()
    st.header("Optimisation constraints")
    min_margin = st.slider("Minimum margin", 0, 90, 20) / 100
    band = st.slider("Competitor-relative band", 5, 100, 40) / 100
    price_step = st.number_input("Candidate price step (₹)", 1.0, 500.0, 10.0)

unit_cost = material_cost + labour_cost + packaging_cost
scenario = {
    "category": category, "material_type": material, "material_cost": material_cost,
    "labour_hours": labour_hours, "labour_cost": labour_cost, "packaging_cost": packaging_cost,
    "unit_cost": unit_cost, "seller_experience_years": experience, "seller_rating": seller_rating,
    "product_rating": product_rating, "price": current_price, "competitor_avg_price": competitor_price,
    "competitor_count": competitor_count, "discount_pct": 0.0, "season": season,
    "festival_flag": "Yes" if season == "Festival" else "No", "category_popularity": popularity,
    "previous_week_sales": previous_sales, "rolling_4_week_sales": rolling_sales, "recent_growth": recent_growth,
}

try:
    candidates, best = optimize_price(model, scenario, PricingConstraints(
        min_margin_pct=min_margin, competitor_band_pct=band, price_step=price_step,
    ))
except ValueError as exc:
    st.error(str(exc))
    st.stop()

current = candidates.iloc[(candidates["price"] - current_price).abs().argsort().iloc[0]]
one, two, three, four = st.columns(4)
one.metric("Unit cost", f"₹{unit_cost:,.0f}")
two.metric("Recommended price", f"₹{best['price']:,.0f}", f"₹{best['price'] - current_price:+,.0f} vs current")
three.metric("Estimated weekly demand", f"{best['predicted_demand']:.1f} units")
four.metric("Estimated contribution", f"₹{best['estimated_contribution']:,.0f}")

st.info("Recommendation maximizes **estimated contribution** within your constraints. It is a model-based scenario estimate, not a guarantee of sales or a causal price effect.")

chart_data = candidates.melt(id_vars="price", value_vars=["predicted_demand", "estimated_revenue", "estimated_contribution"],
                             var_name="metric", value_name="value")
st.plotly_chart(px.line(chart_data, x="price", y="value", color="metric", markers=True,
                          title="Candidate-price simulation"), use_container_width=True)

left, right = st.columns([1.4, 1])
with left:
    st.subheader("All feasible price scenarios")
    st.dataframe(candidates[["price", "predicted_demand", "estimated_revenue", "estimated_contribution", "margin_pct"]]
                 .style.format({"price": "₹{:.0f}", "predicted_demand": "{:.1f}", "estimated_revenue": "₹{:,.0f}", "estimated_contribution": "₹{:,.0f}", "margin_pct": "{:.1%}"}),
                 use_container_width=True, hide_index=True)
with right:
    st.subheader("Model evaluation")
    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text())
        st.caption("Chronological holdout test metrics")
        st.dataframe(pd.DataFrame({k: v for k, v in metrics.items() if isinstance(v, dict) and "mae" in v}).T,
                     use_container_width=True)
    st.subheader("Why this demand estimate?")
    st.write("The dashboard’s prediction combines the candidate price with the entered product, seller, historical-demand, and market signals. SHAP is available below for a local model explanation.")

with st.expander("Local SHAP explanation (selected recommendation)"):
    try:
        import shap
        chosen = candidates.loc[[best.name]].copy()
        transformed = model.named_steps["preprocessor"].transform(chosen)
        estimator = model.named_steps["model"]
        names = model.named_steps["preprocessor"].get_feature_names_out()
        values = shap.TreeExplainer(estimator).shap_values(transformed)
        explanation = pd.DataFrame({"feature": names, "shap_value": values[0]}).sort_values("shap_value", key=abs, ascending=False).head(12)
        st.bar_chart(explanation.set_index("feature"))
        st.caption("Positive SHAP values push the estimate up; negative values push it down, relative to the model baseline.")
    except Exception as exc:
        st.warning(f"SHAP explanation could not be rendered: {exc}")

st.caption("Synthetic-data demonstration only. Historical association is not causal proof; new products and changing markets carry additional uncertainty.") 


