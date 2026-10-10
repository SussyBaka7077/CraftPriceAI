# CraftPrice AI methodology

## What is predicted

The demand model estimates `units_sold` for one product-week observation given the entered product, market, historical-demand, and candidate-price features. It does not directly predict a price.

## Optimisation

For each discrete feasible price, the application rebuilds price-derived features, predicts demand, and calculates:

`estimated contribution = (price - unit cost) × predicted demand`

The recommended price is the candidate with the highest estimated contribution. Feasibility requires the seller's minimum margin and a selected band around competitor price. The recommendation is conditional on those business assumptions.

## Synthetic dataset assumptions

`generate_synthetic_data()` produces sequential product-week histories. Demand incorporates category baseline/popularity, product and seller ratings, season/festival lift, recent momentum, price relative to competitor price, product-level price sensitivity, and noise. These are synthetic assumptions, not evidence about real customers.

Lagged sales and rolling sales are formed from sales before the current week. They do not include current or future targets.

## Evaluation

The first 70% of dates train the model, the next 15% provide a validation period, and the final 15% are an untouched chronological test period. A historical-average baseline and Linear Regression are reported alongside XGBoost using MAE, RMSE, and R².

## Limits

Historical associations do not identify causal price effects. Candidate prices that are rare or unseen in historical data are extrapolations. Results are decision support only and do not guarantee sales, revenue, or profit.
