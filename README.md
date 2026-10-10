# CraftPrice AI

CraftPrice AI is a decision-support application for handmade-product micro-sellers. It predicts weekly demand at a **given** price, then simulates feasible candidate prices and recommends the one with the highest estimated contribution.

It does not learn a direct "correct price", establish price causality, or guarantee sales.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/train.py
streamlit run app.py
```

The training command creates a clearly labelled domain-informed synthetic product-week dataset in `data/` and model artifacts in `artifacts/`.

## Method

Every row is a product-week observation. Lagged/rolling sales features are generated only from prior weeks. Training, validation, and test data are split chronologically. The application evaluates candidate prices using:

`estimated contribution = (candidate price - unit cost) × predicted weekly demand`

The default optimiser imposes seller-configurable margin, price-range, and competitor-band constraints.

## Important limitations 

- The included dataset is synthetic and does not demonstrate real-market performance.
- Counterfactual candidate prices are model estimates, particularly uncertain outside historically observed prices.
- Historical correlations are not causal proof.
- Cold-start products and market changes reduce reliability.
