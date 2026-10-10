# Demand Prediction & Pricing Architecture: In-Depth Technical Guide

> **Project:** CraftPrice AI (`/home/rupas/codes/aiml/microproject`)  
> **Domain:** Demand Forecasting & Price Optimization for Handmade-Product Micro-Sellers

---

## 1. Executive Summary & Core Philosophy

CraftPrice AI tackles a fundamental problem faced by small-scale craft sellers: **pricing products under uncertainty without sacrificing contribution margin**.

### Why Not Predict Price Directly?
A naive machine learning approach might treat pricing as a direct regression problem:
$$\text{Price} = f(\text{Product Features})$$
This is flawed because historical prices reflect past seller habits, competitor anchoring, or arbitrary pricing—not market optimality. Directly predicting price simply learns past behavior.

Instead, CraftPrice AI frames the problem in two distinct stages:

```
+-----------------------------------------------------------------------------+
| Stage 1: Demand Prediction                                                  |
| Given a candidate price P and contextual features X (costs, ratings,        |
| seasonality, competitor rates, momentum), predict weekly demand:            |
|                                                                             |
|            Weekly Demand (\hat{y}) = Model(P, X)                            |
+-----------------------------------------------------------------------------+
                                      |
                                      v
+-----------------------------------------------------------------------------+
| Stage 2: Constrained Optimization                                           |
| Sweep across discrete feasible candidate prices {P_1, P_2, ..., P_k}        |
| satisfying business constraints (minimum margin %, competitor band).        |
| Select candidate P* that maximizes estimated total contribution:            |
|                                                                             |
|      Estimated Contribution = (P - Unit Cost) * Predicted Weekly Demand     |
+-----------------------------------------------------------------------------+
```

---

## 2. Repository Architecture & File Breakdown

Below is the directory map and the exact responsibility of every file in the codebase:

```
microproject/
├── app.py                         # Interactive Streamlit dashboard & SHAP explainability
├── scripts/
│   └── train.py                   # End-to-end training orchestration script
├── src/
│   └── craftprice/
│       ├── __init__.py            # Package initialization
│       ├── config.py              # Central schema, paths, and feature definitions
│       ├── data.py                # Synthetic product-week history generator
│       ├── features.py            # Dynamic price-derived feature engineering
│       ├── modeling.py            # Chronological split, preprocessing, models & evaluation
│       └── pricing.py             # Feasible price simulation & contribution optimizer
├── tests/
│   └── test_pricing.py            # Unit tests for pricing bounds & optimization logic
├── artifacts/
│   ├── demand_model.joblib        # Serialized scikit-learn/XGBoost pipeline
│   └── metrics.json               # Model performance benchmark results
└── data/
    └── synthetic_product_weeks.csv# Generated dataset (180 products x 70 weeks)
```

### Detailed File Responsibilities

| File | Exact Role & Functions |
| :--- | :--- |
| [`src/craftprice/config.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/config.py) | **Central Configuration:**<br>• Defines paths (`DATA_PATH`, `MODEL_PATH`, `METRICS_PATH`).<br>• Defines `TARGET = "units_sold"` and `DATE_COLUMN = "week_start"`.<br>• Declares 19 numeric features and 4 categorical features. |
| [`src/craftprice/data.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/data.py) | **Data Generation:**<br>• `generate_synthetic_data(n_products=180, n_weeks=70, seed=42)` generates realistic sequential panel data.<br>• Implements economic rules: category base demand, product-level elasticity, cost structure, season/festival multipliers, and momentum. |
| [`src/craftprice/features.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/features.py) | **Dynamic Feature Engineering:**<br>• `add_price_features(frame)` computes interaction terms: `price_to_cost`, `competitor_price_gap`, and `estimated_contribution_per_unit`.<br>• **Crucial:** Must be called whenever a candidate price is substituted before running model inference. |
| [`src/craftprice/modeling.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/modeling.py) | **Model Pipeline & Training:**<br>• `chronological_split()`: Partitions time-series data without lookahead leakage.<br>• `make_preprocessor()`: Assembles imputers and one-hot encoders.<br>• `train_models()`: Trains historical mean baseline, Linear Regression, and XGBoost.<br>• `evaluate()`: Computes MAE, RMSE, and $R^2$ with non-negative prediction clipping.<br>• `save_model()` & `load_model()`: Handles joblib persistence. |
| [`src/craftprice/pricing.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/pricing.py) | **Optimization Engine:**<br>• `PricingConstraints`: Data class storing `min_margin_pct`, `min_price`, `max_price`, `competitor_band_pct`, and `price_step`.<br>• `optimize_price()`: Validates bounds, vectors out candidate prices, calls `model.predict()`, and chooses the argmax of contribution. |
| [`scripts/train.py`](file:///home/rupas/codes/aiml/microproject/scripts/train.py) | **Automation Runner:**<br>• Generates dataset, fits models, evaluates on holdout, dumps artifacts to disk, and prints terminal logs. |
| [`app.py`](file:///home/rupas/codes/aiml/microproject/app.py) | **User Interface (Streamlit):**<br>• Renders seller sidebar inputs.<br>• Runs the optimization loop interactively.<br>• Plots Plotly demand/revenue/contribution curves.<br>• Uses `shap.TreeExplainer` for local feature importance breakdown. |
| [`tests/test_pricing.py`](file:///home/rupas/codes/aiml/microproject/tests/test_pricing.py) | **Automated Testing:**<br>• Validates optimizer bounds, ensures candidate prices adhere to minimum margins and competitor limits, and tests error handling when no feasible price exists. |

---

## 3. Data Pipeline & Synthetic Data Modeling

The data generator [`src/craftprice/data.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/data.py) simulates 180 distinct products over 70 weekly intervals (12,600 total observations).

### Key Synthetic Economics:
1. **Base Demand & Elasticity per Category:**
   - Bracelet: base demand = 18, typical price = ₹260, base elasticity = 1.55
   - Jewellery: base demand = 16, typical price = ₹480, base elasticity = 1.35
   - Candle: base demand = 21, typical price = ₹340, base elasticity = 1.15
   - Macrame: base demand = 11, typical price = ₹650, base elasticity = 1.00
   - Customized Gift: base demand = 14, typical price = ₹520, base elasticity = 1.25
   - Handmade Craft: base demand = 13, typical price = ₹430, base elasticity = 1.10
2. **Cost Composition:**
   $$\text{unit\_cost} = \text{material\_cost} + (\text{labour\_hours} \times \text{hourly\_rate}) + \text{packaging\_cost}$$
3. **Price Effect (Relative Elasticity):**
   $$\text{Price Effect} = \exp\left(-\epsilon_i \cdot \left(\frac{\text{price}}{\text{competitor\_avg\_price}} - 1\right)\right)$$
   Using relative price ensures that a 10% price shift has comparable directional impact across both low-cost and high-ticket items.
4. **Multiplier Interactions:**
   $$\text{Expected Demand} = \text{Base} \times \text{Popularity} \times \text{Season Lift} \times \text{Rating Effect} \times \text{Price Effect} \times \text{Momentum}$$
   Noise is injected with heteroscedastic variance: $\mathcal{N}(0, 2.8 + 0.14 \times \text{Expected})$.

---

## 4. Feature Engineering & Preprocessing

### Dynamic Engineered Features ([`features.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/features.py))
Every candidate price alters three dynamic interaction features:
1. `price_to_cost` = $\frac{\text{price}}{\max(\text{unit\_cost}, 1)}$
2. `competitor_price_gap` = $\text{price} - \text{competitor\_avg\_price}$
3. `estimated_contribution_per_unit` = $\text{price} - \text{unit\_cost}$

### Chronological Train-Val-Test Split ([`modeling.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/modeling.py))
Random splitting (`train_test_split`) leaks future seasonal and trend information into past records. Hence, `chronological_split()` partitions ordered unique calendar weeks:
- **Train:** First 70% of chronological weeks (8,820 rows)
- **Validation:** Next 15% of weeks (1,800 rows)
- **Test:** Final 15% holdout weeks (1,980 rows)

### Pipeline Preprocessor ([`make_preprocessor()`](file:///home/rupas/codes/aiml/microproject/src/craftprice/modeling.py#L34-L41))
Constructed with scikit-learn's `ColumnTransformer`:
- **Numerical Pipeline:** `SimpleImputer(strategy="median")` across all 19 numeric features.
- **Categorical Pipeline:** `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")` across `category`, `material_type`, `season`, and `festival_flag`.

---

## 5. Model Implementations & Comparison

Three levels of models are trained and evaluated in [`modeling.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/modeling.py#L54-L78):

### 1. Historical Average Baseline
- **Formula:** $\hat{y} = \frac{1}{N_{\text{train}}} \sum_{i=1}^{N_{\text{train}}} y_i \approx \bar{y}_{\text{train}}$
- **Purpose:** Minimum benchmark. Any model that fails to beat this baseline has learned nothing.
- **Test Performance:**
  - $\text{MAE} = 5.763$
  - $\text{RMSE} = 7.253$
  - $R^2 = -0.010$

### 2. Multiple Linear Regression
- **Implementation:**
  ```python
  linear = Pipeline([
      ("preprocessor", make_preprocessor()),
      ("model", LinearRegression())
  ])
  linear.fit(train_and_validation[FEATURES], train_and_validation[TARGET])
  ```
- **Theory:** Fits an ordinary least squares hyperplane:
  $$\hat{y} = w_0 + \sum_{j=1}^{d} w_j x_j$$
- **Strengths:** High interpretability, closed-form solution.
- **Weaknesses:** Cannot naturally model exponential price-elasticity curves, saturation thresholds, or complex interaction terms (e.g. season $\times$ category) without explicit polynomial/interaction feature engineering.
- **Test Performance:**
  - $\text{MAE} = 4.260$ (26.1% improvement over baseline)
  - $\text{RMSE} = 5.375$
  - $R^2 = 0.445$

### 3. XGBoost Regressor (Production Model)
- **Implementation:**
  ```python
  xgb = Pipeline([
      ("preprocessor", make_preprocessor()),
      ("model", XGBRegressor(
          n_estimators=450,
          max_depth=5,
          learning_rate=0.045,
          subsample=0.85,
          colsample_bytree=0.9,
          reg_lambda=2,
          objective="reg:squarederror",
          random_state=42,
          n_jobs=2,
      )),
  ])
  xgb.fit(train_and_validation[FEATURES], train_and_validation[TARGET])
  ```
- **Test Performance:**
  - $\text{MAE} = 4.140$ (Best performance; beats linear regression by 0.12 units/week)
  - $\text{RMSE} = 5.266$
  - $R^2 = 0.468$

---

## 6. XGBoost Mechanics & Hyperparameter Rationale

XGBoost (*Extreme Gradient Boosting*) builds an ensemble of shallow decision trees sequentially:
$$\hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta f_t(\mathbf{x}_i)$$

### Objective Function & Regularization
At boosting round $t$, XGBoost minimizes:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n l\left(y_i, \hat{y}_i^{(t-1)} + f_t(\mathbf{x}_i)\right) + \Omega(f_t)$$
Where the tree complexity penalty is:
$$\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2$$
- $T$ is the number of leaves in the tree.
- $w_j$ is the continuous weight assigned to leaf $j$.
- $\lambda$ (`reg_lambda`) penalizes large leaf weights (L2 regularization).

### Detailed Hyperparameter Breakdown

| Hyperparameter | Value | Architectural Purpose |
| :--- | :--- | :--- |
| `n_estimators` | `450` | Total boosting iterations (trees). Coupled with the low learning rate, 450 trees allow smooth, gradual convergence. |
| `learning_rate` ($\eta$) | `0.045` | Shrinkage step size. Multiplies the output of each new tree by 0.045, preventing individual trees from dominating and mitigating overfitting. |
| `max_depth` | `5` | Maximum depth of each decision tree. Allows up to $2^5 = 32$ leaves and up to 5-way non-linear feature interactions (e.g., `price` $\times$ `competitor_avg_price` $\times$ `category` $\times$ `season` $\times$ `rating`). |
| `subsample` | `0.85` | Row subsampling: 85% of training samples are randomly sampled without replacement for each tree. Adds stochastic bagging behavior. |
| `colsample_bytree` | `0.90` | Feature column subsampling: 90% of features are randomly selected per tree, reducing collinearity dominance. |
| `reg_lambda` | `2.0` | $L_2$ regularization penalty on leaf weights. Encourages smoother predictions across leaf transitions. |
| `objective` | `"reg:squarederror"` | Standard squared error loss function ($L_2$ regression loss). |

### Non-Negative Prediction Post-Processing
In demand forecasting, selling negative quantities is physically impossible. In [`evaluate()`](file:///home/rupas/codes/aiml/microproject/src/craftprice/modeling.py#L45) and [`optimize_price()`](file:///home/rupas/codes/aiml/microproject/src/craftprice/pricing.py#L33):
$$\text{Demand}_{\text{final}} = \max(0, \hat{y})$$
This clipping guarantees all downstream financial calculations remain strictly non-negative.

---

## 7. Pricing Simulation & Constrained Optimizer

The optimization module [`src/craftprice/pricing.py`](file:///home/rupas/codes/aiml/microproject/src/craftprice/pricing.py) translates model predictions into business decisions.

### Feasibility Constraints
Given seller inputs:
- Unit Cost: $C$
- Competitor Average Price: $P_{\text{comp}}$
- Minimum Margin %: $M_{\min}$ (e.g., $20\%$)
- Competitor Band %: $B$ (e.g., $\pm 40\%$)

The lower and upper bounds for candidate prices are defined as:
$$\text{Lower Bound} = \max\left(P_{\min}, \; C \times (1 + M_{\min}), \; P_{\text{comp}} \times (1 - B)\right)$$
$$\text{Upper Bound} = \min\left(P_{\max}, \; P_{\text{comp}} \times (1 + B)\right)$$

If $\text{Lower Bound} > \text{Upper Bound}$, a `ValueError` is raised, alerting the seller that no price satisfies both their desired profit margin and competitor positioning.

### Discrete Simulation Sweep
1. A sequence of candidate prices is generated with step size `price_step` (e.g. ₹10 increments):
   $$P_{\text{candidates}} = [\text{Lower}, \text{Lower} + \Delta P, \dots, \text{Upper}]$$
2. A synthetic DataFrame is constructed with identical product attributes but varying prices.
3. [`add_price_features()`](file:///home/rupas/codes/aiml/microproject/src/craftprice/features.py) is executed to update `price_to_cost`, `competitor_price_gap`, and unit margin.
4. The trained XGBoost model predicts demand for all candidates simultaneously.
5. Total contribution is calculated:
   $$\text{Estimated Contribution}(P_k) = (P_k - C) \times \hat{y}(P_k)$$
6. The recommended price $P^*$ is selected via:
   $$P^* = \arg\max_{P_k} \text{Estimated Contribution}(P_k)$$

---

## 8. Interpretability with SHAP in Streamlit

In [`app.py`](file:///home/rupas/codes/aiml/microproject/app.py#L105-L117), the Streamlit dashboard provides model interpretability using Tree SHAP (SHapley Additive exPlanations):

```python
transformed = model.named_steps["preprocessor"].transform(chosen)
estimator = model.named_steps["model"]
names = model.named_steps["preprocessor"].get_feature_names_out()
values = shap.TreeExplainer(estimator).shap_values(transformed)
```

### How to Interpret the Output:
- **Baseline:** The expected value of the tree ensemble across the training distribution.
- **Positive SHAP values:** Features that push demand higher (e.g., high product rating, peak festival season, strong category popularity).
- **Negative SHAP values:** Features that suppress demand (e.g., price significantly higher than competitor average, low rolling sales momentum).

---

## 9. Key Limitations & Production Considerations

1. **Synthetic Data Realism:** The current data generator simulates clean relationships. Real-world sales exhibit stockouts, marketing campaign spikes, supply shocks, and multi-channel cannibalization.
2. **Correlation vs. Causality:** Machine learning models predict conditional expectations $E[Y \mid X = x, P = p]$. If higher-priced products in the dataset sell well because they have superior brand equity, the model may mistakenly associate higher prices with higher sales unless brand features are controlled.
3. **Extrapolation Uncertainty:** Predictions for candidate prices far outside the historical distribution $(\min P_{\text{hist}}, \max P_{\text{hist}})$ have high variance and lower reliability.
