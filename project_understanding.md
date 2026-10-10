# Regression → XGBoost — Study Notes

*Theme: demand & price prediction (the CraftPrice setting). Every example predicts weekly demand for a product from its price and other features.*

All numbers in the worked examples below were verified by running the code — they are not hand-waved.

---

## How to read these notes

Each concept follows the same ladder: **intuition → concrete example → the mechanism → the formula → when to use / avoid.** Skim the intuition, then slow down on the mechanism — that's where real understanding lives.

---

# PART 1 — REGRESSION

## 1. What is regression, really?

**The key idea:** regression is supervised learning where you predict a **continuous number**.

You have historical rows of `(inputs → known answer)`, and you want a function `f` that maps new inputs to a number you haven't seen yet.

- Predict **weekly demand** (e.g. 84 units) from price, season, ad spend → *regression*
- Predict **house price** (e.g. ₹62 lakh) from size, location → *regression*
- Predict whether a customer **churns** (yes / no) → *not regression, that's classification*

The single distinction to burn in:

| Question | Output type | Task |
|---|---|---|
| "How many units will sell?" | continuous number | **regression** |
| "Will this product sell out — yes or no?" | discrete category | **classification** |

**What / Why / How / When**

- **What:** learn `f(X) → y` where `y` is a real number.
- **Why:** so you can predict the number for new inputs *and* read off how each input pushes the output (e.g. "each ₹1 price rise costs ~4.5 units of demand").
- **How:** pick a form for `f` (a line, a curve, a tree ensemble…), define what "wrong" means (a loss function), then tune `f`'s parameters to minimize that loss on data.
- **When NOT:** when the target is a category (use classification) or when you have no labelled answers at all (use clustering / unsupervised methods).

---

## 2. Simple linear regression — the foundation

**Intuition:** assume the relationship is a straight line. One input `x`, one output `y`. Find the line that sits "closest" to all the points.

**The model:**

```
demand = w · price + b
```

- `w` (weight / slope) = how much demand changes per unit price. Sign and size tell the story.
- `b` (bias / intercept) = predicted demand at price 0 (often not physically meaningful, just anchors the line).

### Worked example (verified)

Tiny demand-vs-price table:

| price | 10 | 12 | 14 | 16 | 18 |
|---|---|---|---|---|---|
| demand | 100 | 90 | 84 | 72 | 64 |

The closed-form least-squares solution (derived in §3) gives:

```
w = Σ(xᵢ - x̄)(yᵢ - ȳ) / Σ(xᵢ - x̄)²   =  -180 / 40  = -4.5
b = ȳ - w·x̄                          =  82 - (-4.5)(14) = 145
```

So the fitted line is:

```
demand ≈ 145 − 4.5 × price
```

**Reading it:** every ₹1 increase in price is associated with ~4.5 fewer units sold. At price 14 it predicts `145 − 63 = 82` units (actual was 84 — off by 2).

Predictions vs actuals: `[100, 91, 82, 73, 64]` vs `[100, 90, 84, 72, 64]` → residuals `[0, −1, +2, −1, 0]`. Very tight fit (R² = 0.993).

### The same thing in scikit-learn

```python
import numpy as np
from sklearn.linear_model import LinearRegression

price  = np.array([10, 12, 14, 16, 18]).reshape(-1, 1)  # sklearn wants X as 2-D
demand = np.array([100, 90, 84, 72, 64])

model = LinearRegression().fit(price, demand)

print(model.coef_)        # [-4.5]   -> w
print(model.intercept_)   # 145.0    -> b
print(model.predict([[15]]))  # 145 - 4.5*15 = 77.5
```

> **Gotcha (common beginner error):** the feature matrix `X` must be 2-D, shape `(n_samples, n_features)`. A 1-D array throws an error. That's why we `.reshape(-1, 1)`.

---

## 3. How does regression actually *learn*? (the mechanism)

This is the part most tutorials skip. Two questions: **(a) what does "best line" mean**, and **(b) how do we find it**.

### (a) The loss function — defining "wrong"

The standard choice for regression is **Mean Squared Error (MSE)**:

```
MSE(w, b) = (1/n) Σ (yᵢ − (w·xᵢ + b))²
```

We square the errors for two reasons: positive and negative errors don't cancel, and big misses are punished disproportionately (a miss of 4 counts 4× worse than two misses of 2). The line that minimizes MSE is *the* least-squares line.

### (b) Two ways to minimize it

**Method 1 — the normal equation (closed form).** Because MSE is a smooth bowl (convex quadratic) in the parameters, calculus gives an exact answer in one shot. In matrix form, with `X` including a column of 1s for the intercept:

```
w* = (Xᵀ X)⁻¹ Xᵀ y
```

Exact, no iterations. But inverting `XᵀX` costs roughly `O(d³)` in the number of features `d`, so it becomes impractical when you have very many features.

**Method 2 — gradient descent (iterative).** Start with random `w, b`; repeatedly nudge them downhill along the slope (gradient) of the loss:

```
w ← w − η · ∂MSE/∂w
b ← b − η · ∂MSE/∂b
```

where `η` (eta) is the **learning rate** (step size). The gradients for MSE are:

```
∂MSE/∂w = (2/n) Σ (ŷᵢ − yᵢ)·xᵢ
∂MSE/∂b = (2/n) Σ (ŷᵢ − yᵢ)
```

I verified on the toy data that all three routes — the formula, the normal equation (`np.linalg.lstsq`), and 200k steps of gradient descent — land on exactly `w = −4.5, b = 145`. They *must* agree, because MSE has a single global minimum.

**Why you should care about the distinction:** linear regression can use the exact formula, but gradient descent is the workhorse for models where no closed form exists (neural nets, and — conceptually — the *gradient* in gradient **boosting**, which is where XGBoost comes from). Same downhill idea, different function being minimized.

---

## 4. Multiple linear regression — more than one input

Real demand depends on more than price. Add features and the line becomes a (hyper)plane:

```
demand = w₁·price + w₂·ad_spend + w₃·is_holiday + … + b
```

In vector form, `ŷ = Xw + b`. Nothing conceptually new — same MSE loss, same solving methods — just more weights.

```python
import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame({
    "price":     [10, 12, 14, 16, 18, 11, 13, 15],
    "ad_spend":  [50, 40, 60, 30, 20, 55, 45, 35],
    "is_holiday":[ 1,  0,  0,  1,  0,  1,  0,  0],
    "demand":    [120, 92, 96, 95, 60, 128, 90, 74],
})
X = df[["price", "ad_spend", "is_holiday"]]
y = df["demand"]

model = LinearRegression().fit(X, y)
for name, coef in zip(X.columns, model.coef_):
    print(f"{name:12s}: {coef:+.2f}")
print("intercept   :", round(model.intercept_, 2))
```

Each coefficient is the effect of that feature **holding the others fixed**. That "holding others fixed" clause is the source of a lot of misinterpretation — see multicollinearity in §7.

> **Feature scaling note:** plain `LinearRegression` (closed form) is unaffected by feature scale. But the moment you use gradient descent, regularization (Ridge/Lasso), or distance-based models, you should standardize features (mean 0, std 1) or the optimization and penalties get distorted.

---

## 5. Types of regression (the "…etc")

This is the map you asked for. They differ along two axes: **the shape of `f`** (linear vs curved vs flexible) and **what extra penalty/assumption they add**.

| Type | Model shape | What problem it solves | Watch out for |
|---|---|---|---|
| **Simple linear** | line, 1 feature | baseline, 1 driver | too simple for real data |
| **Multiple linear** | hyperplane, many features | several linear drivers | multicollinearity |
| **Polynomial** | curve (`x, x², x³…`) | curved relationships | overfits fast at high degree |
| **Ridge (L2)** | linear + penalty on Σw² | many correlated features; shrinks weights | needs scaling; keeps all features |
| **Lasso (L1)** | linear + penalty on Σ\|w\| | feature selection (drives weights to 0) | unstable with correlated features |
| **ElasticNet** | Ridge + Lasso mix | correlated features *and* selection | 2 knobs to tune |
| **Logistic** | S-curve → probability | **classification** (misnamed!) | output is a class, not a number |
| **Poisson** | log-link, counts ≥ 0 | count targets like demand | assumes mean ≈ variance |
| **Tree / ensemble** | piecewise, non-linear | complex interactions, no assumptions | less interpretable → *this is XGBoost's family* |

Three of these deserve a closer look.

### 5a. Polynomial regression — curves from a linear model

If demand bends (steep drop at high prices), fit `demand = w₁·price + w₂·price² + b`. The trick: it's **still linear in the weights** — you just invent new columns (`price²`) and feed them to ordinary linear regression.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression

# degree-2 model: adds price^2 (and interaction terms if multiple features)
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
model.fit(price, demand)
```

Higher degree = more wiggle = more risk of **overfitting** (fitting noise instead of signal). Degree 2–3 is usually plenty.

### 5b. Ridge & Lasso — regularization (the overfitting cure)

**The problem:** with many features (especially correlated ones), plain linear regression can learn huge, unstable weights that fit the training set beautifully and fail on new data.

**The fix:** add a penalty on weight size to the loss, so the model prefers *small, simple* weights unless a big one really earns its keep.

```
Ridge (L2):  minimize   MSE + α · Σ wⱼ²          → shrinks weights toward 0
Lasso (L1):  minimize   MSE + α · Σ |wⱼ|          → drives some weights exactly to 0
```

- `α` (alpha) is the **regularization strength**: 0 = ordinary regression; larger = simpler, more biased model. You tune it on a validation set.
- **Ridge vs Lasso in one line:** Ridge keeps all features but shrinks them (good when many features each matter a little); Lasso zeroes out the useless ones (built-in feature selection). **ElasticNet** blends both.

```python
from sklearn.linear_model import Ridge, Lasso
Ridge(alpha=1.0).fit(X, y)   # smooth shrinkage
Lasso(alpha=0.1).fit(X, y)   # some coefficients become exactly 0
```

This is your first taste of the **bias–variance trade-off**: regularization deliberately adds a little bias to cut variance. XGBoost has the *same* penalty built into its objective (§8) — that's a big reason it generalizes well.

### 5c. Logistic regression — the famous misnomer

Despite "regression" in the name, **logistic regression is a classification algorithm.** It passes a linear combination through a sigmoid to output a probability in [0, 1]:

```
p = 1 / (1 + e^-(w·x + b))
```

Mentioned here only so the name never confuses you in an exam: *linear/ridge/lasso/poisson = regression; logistic = classification.*

### 5d. Poisson regression — worth knowing for demand

Demand is a **count** (0, 1, 2, … units), never negative. Plain linear regression can happily predict −7 units, which is nonsense. Poisson regression models counts via a log link (`log(mean) = w·x + b`), guaranteeing non-negative predictions. `sklearn.linear_model.PoissonRegressor` — and XGBoost's `objective="count:poisson"` — both exist for exactly this. Relevant to CraftPrice, where the target is weekly unit counts.

---

## 6. Evaluating a regression model

A single number ("R² = 0.47") means little without knowing what each metric captures.

| Metric | Formula (per prediction) | Units | Intuition | Sensitive to outliers? |
|---|---|---|---|---|
| **MAE** | mean\|y − ŷ\| | same as y | "typical miss, in units" | robust |
| **MSE** | mean(y − ŷ)² | y² | penalizes big misses hard | very |
| **RMSE** | √MSE | same as y | like MAE but punishes big errors more | yes |
| **R²** | 1 − SS_res/SS_tot | unitless | fraction of variance explained (1 = perfect, 0 = no better than predicting the mean) | yes |
| **MAPE** | mean\|(y−ŷ)/y\| | % | error as a percentage | blows up when y≈0 |

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2   = r2_score(y_true, y_pred)
```

**How to choose:** report **MAE** when you want an interpretable "average units off"; use **RMSE** when large errors are especially costly; use **R²** to communicate explained variance. For CraftPrice, MAE is the natural headline (it's literally "units of demand we're typically wrong by").

### The train / validation / test split — the most important habit

You **cannot** judge a model on the data it learned from — it can memorize. Split your data three ways:

- **Train** (~60–70%) — the model fits its parameters here.
- **Validation** (~15–20%) — you tune hyperparameters (α, tree depth, learning rate) and pick the best model here. Used many times.
- **Test** (~15–20%) — touched **once**, at the very end, to get an honest estimate of real-world performance.

```python
from sklearn.model_selection import train_test_split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)
# -> 60% train / 20% val / 20% test
```

> **Time-series caveat (directly relevant to demand data):** if rows are ordered in time, do **not** shuffle. Train on the past, validate/test on the future — otherwise the model "sees" the future and you get **data leakage**: great scores in your notebook, disappointment in production. CraftPrice already does the right thing here with a chronological split. The gap in that project is different and important: **the validation split is carved out but never actually used** — the model is trained with hardcoded hyperparameters and no early stopping. §11 of Part 2 shows the exact fix, because a validation set only earns its keep if something (tuning or early stopping) *consumes* it.

---

## 7. Assumptions of linear regression (when it's the wrong tool)

Linear regression is only trustworthy when its assumptions roughly hold. Mnemonic **LINE**:

- **L**inearity — the true relationship is actually linear. If demand curves, a line will systematically miss.
- **I**ndependence — observations don't leak into each other (violated by autocorrelated time series).
- **N**ormality — residuals are roughly normal (mostly matters for confidence intervals).
- **E**qual variance (homoscedasticity) — the spread of errors is constant across the range, not fanning out.
- Plus **no severe multicollinearity** — features shouldn't be near-duplicates of each other, or coefficients become unstable and uninterpretable.

**How to check quickly:** plot residuals vs predictions. A shapeless cloud = good. A curve or a fanning cone = an assumption is violated → transform features, or switch to a flexible model like a tree ensemble. This is precisely the doorway to XGBoost.

---

## 8. Common regression mistakes (checklist)

1. **Evaluating on training data.** Always score on held-out data.
2. **Data leakage.** Fitting your scaler/encoder on the *whole* dataset before splitting, or including a feature that secretly encodes the target. Fit preprocessing on train only.
3. **Not scaling features** before Ridge/Lasso/gradient-descent models.
4. **Blindly trusting R².** A high R² on training + low on test = overfitting.
5. **Extrapolating** far outside the price range you trained on — the line keeps going, reality doesn't.
6. **Interpreting correlation as causation.** "Higher ad spend → higher demand" in a fitted coefficient is association, not proof of cause.

---
---

# PART 2 — XGBOOST

## 1. The bridge: why leave linear regression at all?

Linear regression assumes a straight-line world. Real demand rarely is:

- Price sensitivity **changes** across the range (customers barely react to ₹10→₹11, then fall off a cliff at ₹19→₹20).
- Features **interact** — a holiday matters *only* when ad spend is also high.
- Effects are **non-monotonic** — demand peaks at a mid price, not at the extremes.

You *can* bolt polynomial terms and hand-built interactions onto linear regression, but you're manually guessing the shape. Tree-based models **learn the shape for you**. XGBoost is the most battle-tested member of that family, and it's what your CraftPrice project already uses — so this section is you understanding your own tool from the ground up.

---

## 2. Building block: the decision tree for regression

**Intuition:** a regression tree plays "20 questions" with the features and predicts a number at the end.

```
                 price <= 15 ?
                /            \
              yes             no
         ad_spend<=40?      demand = 55
          /       \
        yes        no
    demand=70   demand=95
```

Each internal node splits on a feature threshold; each **leaf** outputs a constant prediction (the average demand of the training rows that land there). To predict, you drop a new row down the tree until it reaches a leaf.

**How a split is chosen:** the tree scans every feature and every threshold and picks the split that most reduces error (for regression, the squared-error / variance within the resulting groups). Then it recurses on each side.

```python
from sklearn.tree import DecisionTreeRegressor
tree = DecisionTreeRegressor(max_depth=3).fit(X_train, y_train)
```

- **Strength:** captures non-linearity and interactions automatically, needs no feature scaling, handles mixed feature types.
- **Weakness:** a single deep tree **overfits** wildly — grow it enough and every training row gets its own leaf (memorization). One tree is high-variance.

The whole idea of XGBoost is to fix that weakness by **combining many shallow trees**.

---

## 3. Two ways to combine trees: bagging vs boosting

| Aspect | **Bagging** (e.g. Random Forest) | **Boosting** (e.g. XGBoost) |
|---|---|---|
| Core idea | many trees in **parallel**, trained on random subsets, then **averaged** | many trees in **sequence**, each fixing the previous ones' mistakes |
| Attacks | **variance** (averaging cancels noise) | **bias** (each round reduces leftover error) |
| Trees are | deep, independent | shallow ("weak learners"), dependent |
| Risk | can plateau in accuracy | can overfit if run too long → needs regularization |
| Mental model | "wisdom of an independent crowd" | "a team where each member cleans up after the last" |

Both beat a single tree. Boosting usually reaches higher accuracy on tabular data — at the cost of more careful tuning. XGBoost is boosting done extremely well.

---

## 4. Gradient boosting — the core mechanism (this is the heart of it)

**The key idea in one sentence:** build trees one at a time, where each new tree is trained to predict the **errors (residuals)** left over by all the trees so far, and you add only a **shrunken slice** of each new tree to the running prediction.

Step by step, in demand terms:

1. **Start dumb.** Predict the same constant for everyone — the mean demand. Call it `F₀`.
2. **Look at what's left.** Compute residuals `r = actual − F₀` (who did we under/over-predict?).
3. **Fit a small tree to the residuals** — not to demand itself. This tree learns *the pattern in our mistakes*.
4. **Add a shrunken correction:** `F₁ = F₀ + η · tree₁`, where `η` (learning rate, e.g. 0.1) deliberately takes only a small step.
5. **Repeat.** New residuals, new tree, another small step. After `M` rounds:

```
F_M(x) = F₀ + η·tree₁(x) + η·tree₂(x) + … + η·tree_M(x)
```

Each tree is weak; their shrunken sum is strong. I verified this from scratch on curved demand data — MSE fell **107.6 → 63.3 → 39.4 → 13.7 → 5.6 → 2.7** over 20 rounds, each tree chipping away at the leftover error. That's the entire engine.

**Why "gradient"?** For MSE loss, the residual `(actual − predicted)` is exactly the negative gradient of the loss with respect to the prediction. So "fit the next tree to the residuals" is really "take a gradient-descent step, but in *function* space — add a whole tree instead of nudging a number." Same downhill idea from Part 1 §3, lifted to models. For other losses (Poisson, logistic), you fit to *that* loss's gradient instead — which is why boosting handles many objectives.

**Why the learning rate?** Small `η` means no single tree dominates, so the ensemble generalizes better — but you need more trees. `η` and `n_estimators` trade off against each other: halve the learning rate, roughly double the trees. This is the single most important tuning relationship in XGBoost.

---

## 5. What makes XGBoost special (vs plain gradient boosting)

Gradient boosting is the idea; **XGBoost** ("eXtreme Gradient Boosting") is a specific, highly engineered implementation. What it adds:

1. **Built-in regularization.** Its objective includes L1/L2 penalties on leaf weights *and* a penalty per leaf (`gamma`). Plain gradient boosting has none. This is the same overfitting cure as Ridge/Lasso (Part 1 §5b), baked into the boosting objective — a major reason XGBoost generalizes well.
2. **Second-order optimization.** It uses both the gradient *and* the curvature (2nd derivative, the Hessian) of the loss to choose splits — a smarter step than gradient-only boosting, so it converges in fewer, better trees.
3. **Smart split-finding.** A quantile-based approximate algorithm makes it fast on large data instead of scanning every possible threshold.
4. **Sparsity / missing-value awareness.** Each split learns a default direction for missing values — you don't have to impute them manually.
5. **Systems engineering.** Parallelized split-finding, cache-aware access, out-of-core support. This speed is why it dominated Kaggle for years.
6. **Early stopping.** It can watch validation error each round and stop when it stops improving — automatic control of `n_estimators`.

> Mental model: **gradient boosting is the algorithm; XGBoost is the Formula-1 car built around it** — same engine principle, but tuned, regularized, and engineered for speed and accuracy.

---

## 6. The objective function (the "why it's regularized" detail)

XGBoost minimizes, at each boosting round `t`:

```
Obj = Σ loss(yᵢ, ŷᵢ) + Σ Ω(tree_k)     where   Ω(tree) = γ·T + ½·λ·Σ wⱼ²
```

- First term: how wrong the predictions are (MSE for regression, or your chosen objective).
- Second term `Ω`: the **complexity penalty** — `T` = number of leaves (penalized by `γ`/gamma), `wⱼ` = leaf output values (penalized by `λ`/lambda, an L2 term).

The lesson: XGBoost doesn't just chase low training error, it *simultaneously* fights complexity. That built-in tension is why it resists overfitting better than a naive boosted-tree loop — and it's the same bias–variance trade-off you met with Ridge, now inside a tree ensemble.

---

## 7. Key hyperparameters (the ones you actually tune)

Grouped by what they control. These are the knobs; §8 shows them in code.

| Parameter | Controls | Typical range | Effect |
|---|---|---|---|
| `n_estimators` | number of boosting rounds (trees) | 100–2000 | more = higher capacity; pair with early stopping |
| `learning_rate` (η) | step size per tree | 0.01–0.3 | lower = better generalization, needs more trees |
| `max_depth` | tree depth (interaction complexity) | 3–8 | higher = more complex, overfits faster |
| `min_child_weight` | min "evidence" to make a leaf | 1–10 | higher = more conservative splits |
| `subsample` | row fraction per tree | 0.6–1.0 | <1 adds randomness → fights overfitting |
| `colsample_bytree` | feature fraction per tree | 0.6–1.0 | <1 decorrelates trees |
| `reg_lambda` (λ) | L2 penalty on leaf weights | 0–10 | higher = smoother, more regularized |
| `reg_alpha` (α) | L1 penalty on leaf weights | 0–10 | higher = sparser |
| `gamma` (γ) | min loss reduction to split | 0–5 | higher = fewer splits, simpler trees |

**Where beginners go wrong:** cranking `n_estimators` and `max_depth` up together. That's a fast route to overfitting. The healthier recipe: **low `learning_rate` + moderate `max_depth` (3–6) + many trees controlled by early stopping**, then add `subsample`/`colsample` and `reg_*` if you still overfit.

---

## 8. Full worked example — demand prediction with XGBoost

End-to-end and idiomatic for the current XGBoost 2.x API. This is essentially a cleaned-up version of the CraftPrice modeling step. *(Run locally — the notes' sandbox has no internet to install xgboost, but this is the correct modern API.)*

```python
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

# ---------- 1. Data (synthetic weekly demand; swap in your real frame) ----------
rng = np.random.default_rng(42)
n = 2000
df = pd.DataFrame({
    "price":       rng.uniform(8, 25, n),
    "ad_spend":    rng.uniform(0, 100, n),
    "is_holiday":  rng.integers(0, 2, n),
    "competitor_price": rng.uniform(8, 25, n),
    "week_of_year": rng.integers(1, 53, n),
})
# a deliberately non-linear, interaction-rich demand curve + noise
df["demand"] = (
    200
    - 6*df["price"] + 0.15*df["price"]**2          # curved price effect
    + 0.4*df["ad_spend"]                            # ads help
    + 15*df["is_holiday"]*(df["ad_spend"] > 50)     # holiday matters only w/ high ads (interaction)
    + 3*(df["competitor_price"] - df["price"])      # relative price vs competitor
    + rng.normal(0, 8, n)
).clip(lower=0)                                     # demand can't be negative

features = ["price", "ad_spend", "is_holiday", "competitor_price", "week_of_year"]
X, y = df[features], df["demand"]

# ---------- 2. Split: train / validation / test ----------
# (random split shown for simplicity; for TRUE time-series data, split by time — see note)
from sklearn.model_selection import train_test_split
X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42)

# ---------- 3. Model with early stopping ----------
model = xgb.XGBRegressor(
    n_estimators=2000,          # an UPPER BOUND; early stopping picks the real number
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    min_child_weight=3,
    objective="reg:squarederror",   # use "count:poisson" for pure count demand
    early_stopping_rounds=50,       # stop if val error doesn't improve for 50 rounds
    eval_metric="mae",
    random_state=42,
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],      # <-- THIS is what makes the validation split earn its keep
    verbose=False,
)
print("Best iteration (trees actually used):", model.best_iteration)

# ---------- 4. Evaluate ONCE on the untouched test set ----------
pred = model.predict(X_test)
print(f"Test MAE: {mean_absolute_error(y_test, pred):.3f}")
print(f"Test R² : {r2_score(y_test, pred):.3f}")

# ---------- 5. Which features drove predictions? ----------
importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print(importances)
```

**What each block teaches:**

- **Early stopping** turns `n_estimators` from a guess into an upper bound: XGBoost watches validation MAE and keeps the tree count that's genuinely best. This is the cleanest fix for the "validation set is never used" gap noted in Part 1 §6.
- `eval_set` is the mechanical link — without passing validation data here, the split you carved out does nothing.
- **Test set is touched exactly once**, at the end. That number is your honest performance estimate.
- **Feature importance** tells you *which* inputs mattered — but see the caveat in §10.

> **Time-series version (what CraftPrice needs):** replace the random `train_test_split` with a chronological cut — earliest weeks → train, middle → validation, latest → test — so you never validate on the past using the future. Everything else (early stopping via `eval_set`, single test evaluation) stays identical.

---

## 9. XGBoost vs Linear Regression — the honest comparison

| Aspect | Linear Regression | XGBoost |
|---|---|---|
| **Core idea** | one straight-line/plane fit | hundreds of small trees, each fixing prior errors |
| **Non-linearity** | manual (polynomial terms) | automatic |
| **Interactions** | manual (hand-built) | automatic (that's what tree depth *is*) |
| **Feature scaling** | needed for regularized/GD variants | not needed (splits are scale-invariant) |
| **Interpretability** | high — read the coefficients | lower — needs SHAP / importance plots |
| **Small data** | strong, stable | can overfit; may not beat linear |
| **Medium/large tabular** | often underfits | usually state-of-the-art |
| **Training cost** | trivial | higher; many hyperparameters |
| **Best use case** | few features, roughly linear, need explanation | complex tabular data, accuracy is priority |
| **When to avoid** | curved/interacting data | tiny datasets, need a simple auditable model, or images/text/audio (use deep learning) |

**Practical recommendation:** always fit linear regression (and a dumb baseline like "predict last week's demand") **first**. It's your yardstick. Reach for XGBoost when the data is non-linear/interacting and the linear baseline leaves accuracy on the table. CraftPrice does exactly this — XGBoost (MAE 4.14) beats linear (4.26) beats historical-average (5.76) — and reporting all three is the right instinct, because a fancy model that barely beats "guess the average" is telling you something.

---

## 10. Common XGBoost mistakes (checklist)

1. **No early stopping / no validation use.** Hardcoding `n_estimators` with a carved-but-unused validation split — the exact CraftPrice gap. Pass `eval_set` and let early stopping choose.
2. **Depth + trees both cranked high.** Fast overfitting. Prefer low learning rate + shallow trees + many rounds via early stopping.
3. **Shuffling time-series data.** Random splits on time-ordered demand leak the future → inflated scores that collapse in production.
4. **Trusting default feature importance blindly.** The default (`gain`/`weight`) can be biased toward high-cardinality features and splits importance among correlated ones. Prefer **SHAP values** for trustworthy attributions.
5. **Reading importance as causation.** "price is important to the model" ≠ "changing price causes demand to move by X." Same association-vs-causation caution as linear regression.
6. **Not setting `random_state`.** Subsampling makes runs non-reproducible unless you seed it.
7. **Using XGBoost when linear would do.** More accuracy isn't free — you pay in tuning time and lost interpretability. Justify it against the baseline.

---

## 11. Putting it together — the mental model to keep

- **Regression** = predict a number by minimizing a loss (MSE) over parameters.
- **Linear regression** solves that with a straight line — exactly (normal equation) or iteratively (gradient descent).
- **Regularization (Ridge/Lasso)** trades a little bias for less variance by penalizing big weights.
- **A single tree** captures non-linearity but overfits.
- **Boosting** chains shallow trees, each fitting the *residual* (the negative gradient) of the ones before, added in shrunken steps.
- **XGBoost** is boosting with regularization, second-order optimization, and heavy engineering — the go-to for tabular data.

The thread running through all of it: **define what "wrong" means, then reduce it while controlling complexity.** Linear regression, Ridge, and XGBoost are three points on that same spectrum — from one rigid line to a regularized ensemble of hundreds of trees.

---

## 12. Suggested next steps to cement this

1. Fit linear regression, then XGBoost, on the *same* demand data and compare MAE — feel the accuracy gain (and the tuning cost).
2. In CraftPrice, wire the existing validation split into `eval_set` + `early_stopping_rounds` and see how `best_iteration` compares to the hardcoded tree count. That single change closes the methodology gap.
3. Plot XGBoost's predicted demand vs price while holding other features fixed — watch it recover the curve linear regression can't.
4. Install SHAP and compare its feature attributions to `feature_importances_`.
5. Try `objective="count:poisson"` since demand is a count, and see if it helps at low-demand weeks.
