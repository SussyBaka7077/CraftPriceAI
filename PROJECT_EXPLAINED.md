# CraftPrice AI — The Complete Project Explained

A from-scratch explanation of every concept this project uses, written so that
someone who has never trained a model can follow it end to end, and so that the
author can answer any question about it in a viva.

Every claim about this project is checkable against the files in this repository.
Line references are to the revision at the time of writing.

---

## How to read this document

Each concept is explained in three layers, always in the same order:

> **In plain English** — the idea, no jargon, as you'd explain it to a friend.
> **Technically** — the correct definition, stated simply.
> **In this project** — exactly where it lives and what it does here.

If you only have ten minutes, read §0, §5.8, §7.5, and §12.
If you're preparing for a presentation, read straight through, then §11 and §12.
For the code map, jump to §14.

Your existing `2.md` covers the regression and boosting mathematics in depth.
This document is broader: it covers the data, the pipeline, the metrics, SHAP,
and the price optimizer as well.

---

## §0 — The project in one page

### The problem

A person makes things by hand — bracelets, candles, macramé — and sells them
online. They have one hard decision every week: **what should I charge?**

Charge too much and nobody buys. Charge too little and every sale earns almost
nothing. There is a best answer, and it changes with the product, the season,
the competition, and how popular the item is. Most small sellers guess.

### What this project does

It learns from 70 weeks of history how sales respond to price and everything
else, and then it *tries out every sensible price* and reports which one makes
the most money.

### The two stages

```
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 1 — LEARN                                                      │
│                                                                      │
│   12,600 past product-weeks  ──►  gradient-boosted trees             │
│   (price, cost, ratings,          (src/craftprice/modeling.py)       │
│    season, past sales, ...)                                          │
│                                                                      │
│   Result: a function that says "at this price, in these conditions,  │
│           expect roughly this many units"                            │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 2 — DECIDE                                                     │
│                                                                      │
│   Try price = 400, 410, 420, ... 490   (src/craftprice/pricing.py)   │
│   For each: ask Stage 1 for demand                                   │
│   For each: compute contribution = (price − unit_cost) × demand       │
│                                                                      │
│   Result: the price with the highest contribution, inside the        │
│           seller's constraints — plus a note saying whether the      │
│           model or a constraint actually decided it                  │
└──────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────────────┐
│ STAGE 3 — SHOW                                                       │
│   The Streamlit dashboard (app.py)                                   │
│   Recommended price · demand with an 80% interval · contribution      │
│   Two trade-off charts · candidate table · honest evaluation panel    │
│   Optional SHAP explanation (src/craftprice/explain.py)              │
└──────────────────────────────────────────────────────────────────────┘
```

### The one-sentence summary

**It is not a sales predictor. It is a decision tool.** The regression is the
engine; the price recommendation is the product. That distinction is the most
interesting thing about this project — say it early in any presentation.

### File map

| File | Role |
|---|---|
| `src/craftprice/config.py` | Declares the schema: the feature list, the target, the file paths |
| `src/craftprice/data.py` | Generates the synthetic dataset from a known formula |
| `src/craftprice/features.py` | Derives three extra price-based features |
| `src/craftprice/modeling.py` | Splits, preprocesses, trains, evaluates, reports |
| `src/craftprice/pricing.py` | The optimizer — turns the model into a price |
| `src/craftprice/explain.py` | Local SHAP explanations, kept out of the UI |
| `scripts/train.py` | Runs the whole training process in one command |
| `app.py` | The Streamlit dashboard |
| `tests/test_pricing.py` | Six tests, all on the optimizer |
| `artifacts/metrics.json` | The evaluation record — seven blocks of scores |
| `artifacts/demand_model.joblib` | The saved trained model |
| `data/synthetic_product_weeks.csv` | The dataset itself |

---

## §1 — The vocabulary of machine learning

Every term you'll need, in the order you'll meet them.

### 1.1 Feature (also called input, predictor, independent variable, or `X`)

> **In plain English** — The clues. The information you have going in. If a
> doctor diagnoses you, the symptoms are the features.
>
> **Technically** — An individual measurable property of the thing being
> predicted; one column of the design matrix `X`.
>
> **In this project** — 23 of them, declared in `src/craftprice/config.py:13-20`.
> Nineteen are numbers (price, costs, ratings, past sales) and four are
> categories (category, material type, season, festival flag).

### 1.2 Target (also called label, output, dependent variable, or `y`)

> **In plain English** — The thing you want to know. The answer key you study
> from.
>
> **Technically** — The variable being predicted; the response.
>
> **In this project** — `units_sold` (`config.py:10`) — how many units were sold
> in that product-week. Everything else is a feature.

### 1.3 Regression

> **In plain English** — Predicting a *number* ("how many?") rather than a
> *category* ("spam or not spam?"). Predicting a category is called
> classification.
>
> **Technically** — Supervised learning where the target is continuous or
> count-valued; the model estimates `E[y | X]`, the expected value of the target
> given the inputs.
>
> **In this project** — Predicting units sold is regression. It matters that
> the model targets the **mean** and not the median or the mode: the price
> optimizer maximizes *expected* money, and expected money depends on expected
> demand.

### 1.4 Training (also called fitting, or learning)

> **In plain English** — Showing the model thousands of past examples *with
> their answers*, over and over, until it finds the pattern.
>
> **Technically** — Adjusting the model's internal numbers to minimize a loss
> function on the training data.
>
> **In this project** — `modeling.py:185`, where XGBoost sees the historical
> rows (train + validation) and their actual sales.

### 1.5 Prediction (also called inference)

> **In plain English** — Asking the trained model "given this situation, what
> do you think?" Inference is the same word as when a detective infers.
>
> **Technically** — Evaluating the fitted function at inputs it may never have
> seen.
>
> **In this project** — Used in two completely different ways, which is worth
> noticing:
> 1. **Evaluation** (`modeling.py:45`): predict on the held-out test rows to
>    score the model.
> 2. **Simulation** (`pricing.py:105`): predict demand for a grid of *imaginary*
>    prices that never occurred, to see what *would* happen.

That second use is the whole trick of this project. The model isn't just
reporting the past — it's being used as a **simulator** of counterfactuals.

### 1.6 Model

> **In plain English** — The formula the computer found. It's called a "model"
> because, like a model aeroplane, it's a simplified stand-in for the real
> thing.
>
> **Technically** — A parameterized function mapping features to a predicted
> target, plus the learned values of its parameters.
>
> **In this project** — `artifacts/demand_model.joblib`, a serialized
> scikit-learn `Pipeline` containing both the preprocessing steps and the
> trained tree ensemble, saved to disk so the app can load it instantly.

### 1.7 Parameters vs hyperparameters

This distinction trips up almost everyone, so let's be precise.

> **Parameters** are what the model *learns*. You never set them; training
> finds them. In a linear regression they're the slope and intercept.
>
> **Hyperparameters** are what *you* set before training. The model can't learn
> them. They control how learning happens.
>
> **In this project** — the hyperparameters are the XGBoost settings in
> `modeling.py:62-64` (`n_estimators`, `max_depth`, `learning_rate`...), all
> explained in §3.8. The parameters are the thousands of split thresholds and
> leaf values inside the trees, which nobody ever sees.

---

## §2 — Regression, from scratch

### 2.1 The core idea

Regression is the search for a formula with adjustable numbers in it.

Suppose you believe sales depend on price. You write down a shape:

```
units = m × price + b
```

`m` and `b` are unknown. Regression is the process of finding the values of `m`
and `b` that make the formula's answers line up best with what actually
happened. That's it. Everything else — trees, boosting, neural networks — is a
more flexible version of "find the numbers in the formula."

### 2.2 A complete worked example, by hand

Five weeks of one product:

| Price (₹) | Units sold |
|---|---|
| 300 | 20 |
| 350 | 18 |
| 400 | 15 |
| 450 | 12 |
| 500 | 10 |

**Step 1 — means.** Mean price `x̄ = 400`. Mean units `ȳ = 15`.

**Step 2 — deviations from the mean.**

| x | x − x̄ | y | y − ȳ | (x − x̄)(y − ȳ) | (x − x̄)² |
|---|---|---|---|---|---|
| 300 | −100 | 20 | +5 | −500 | 10,000 |
| 350 | −50 | 18 | +3 | −150 | 2,500 |
| 400 | 0 | 15 | 0 | 0 | 0 |
| 450 | +50 | 12 | −3 | −150 | 2,500 |
| 500 | +100 | 10 | −5 | −500 | 10,000 |
| | | | | **Σ = −1,300** | **Σ = 25,000** |

**Step 3 — slope and intercept.**

```
m = Σ(x − x̄)(y − ȳ) / Σ(x − x̄)²  =  −1,300 / 25,000  =  −0.052
b = ȳ − m·x̄                       =  15 − (−0.052 × 400)  =  35.8
```

So the fitted formula is **`units ≈ 35.8 − 0.052 × price`**.

**Step 4 — predictions and errors.**

| Price | Actual | Predicted | Residual (actual − predicted) |
|---|---|---|---|
| 300 | 20 | 20.2 | −0.2 |
| 350 | 18 | 17.6 | +0.4 |
| 400 | 15 | 15.0 | 0.0 |
| 450 | 12 | 12.4 | −0.4 |
| 500 | 10 | 9.8 | +0.2 |

**Step 5 — the scores.**

```
MAE   = (0.2 + 0.4 + 0.0 + 0.4 + 0.2) / 5      = 0.24 units
RMSE  = √(0.40 / 5)                             = 0.283 units
SS_res = 0.2² + 0.4² + 0² + 0.4² + 0.2²         = 0.40
SS_tot = 5² + 3² + 0² + 3² + 5²                 = 68
R²    = 1 − 0.40 / 68                           = 0.994
```

An R² of 0.994 means price alone explained 99.4% of the week-to-week variation
in this product's sales. That's very high — real data is never this clean, and
in the full project the same metric lands at **0.468**.

**Step 6 — what the slope means, economically.**

`m = −0.052` says: every extra ₹1 of price is associated with 0.052 fewer units.
So ₹100 more → about 5.2 fewer units.

Turn that into an **elasticity** — the percentage change in demand for a 1%
change in price. At price 400 with sales 15, a 1% price rise is ₹4:

```
Δunits  = −0.052 × 4      = −0.208 units
Δ%      = −0.208 / 15     = −1.39%
elasticity ≈ 1.39
```

Read it as: **a 1% price increase costs about 1.4% of sales.** This single
number is the most important economic quantity in the whole project, because
it's what decides the optimal price (§8.7). The full model's implied
elasticity is about **1.06**, and the true average in the data generator is
**1.27**.

### 2.3 Why "least squares"?

> **In plain English** — When you can't be perfect, you minimize how wrong you
> are. The specific choice here is to minimize the *squared* misses, which
> punishes one big mistake more than several small ones.
>
> **Technically** — Ordinary Least Squares chooses the parameters minimizing
> `Σ(yᵢ − ŷᵢ)²`. Squaring has three useful properties: it makes errors positive
> (so they don't cancel), it's differentiable everywhere (so calculus can find
> the minimum exactly), and it has a closed-form solution.
>
> **In this project** — `LinearRegression()` at `modeling.py:182` does exactly
> this, extended to 23 features instead of 1.

### 2.4 Linear regression in this project

It's the "can a straight line do it?" control. Its score on the held-out test
period:

| Metric | Value |
|---|---|
| MAE | **4.260** units |
| RMSE | 5.375 units |
| R² | **0.445** |

Notice how close it comes to XGBoost. That is itself a finding, and §5.8
explains why.

### 2.5 Why linear regression isn't enough here

This is one of the best explanations you can give in a viva, because it shows
you understand your *data*, not just your model.

Open `src/craftprice/data.py` and look at line 67 — the formula that generates
demand:

```python
expected = base_demand * popularity * season_lift * rating_effect
           * price_effect * momentum_effect
```

Everything is **multiplied**. And multiplication creates **interactions**:

- A ₹50 price increase hurts a lot in a category nobody wants, and barely
  matters in a hot category.
- A 5-star rating is worth far more when the price is high than when it's a
  bargain.
- A Festival-season boost multiplies whatever the price effect already was.

A linear model can only **add**. It says "price contributes −3 units, rating
contributes +2 units, season contributes +5 units" and sums them. It cannot say
"price matters *twice as much* in Winter." Trees can, because every split
conditions on everything above it in the tree.

**Result:** XGBoost (4.140) beats linear regression (4.260) by about 3%. Small,
but in the predicted direction, and for a reason you can explain.

---

## §3 — XGBoost, from scratch

"XGBoost" is a dense name for a simple idea built in four layers. Let's build it.

### 3.1 Layer 1 — a decision tree

> **In plain English** — A flowchart of yes/no questions that ends in a guess.
> "Is the price above ₹400? → Yes. Is the rating above 4.2? → Yes. → Then expect
> about 12 sales."
>
> **Technically** — A recursive binary partition of the feature space. Each
> internal node tests one feature against one threshold; each leaf holds a
> constant prediction, which in regression is the mean of the training rows that
> landed there.
>
> **In this project** — Trees are never written by hand; XGBoost grows them. The
> relevant control is `max_depth=5` (`modeling.py:62`): each tree may ask at most
> five questions in a row before committing to an answer.

### 3.2 Layer 2 — why one tree isn't enough

> **In plain English** — One flowchart can be too rigid. Give it enough
> questions and it memorizes the exact cases it saw, then gets confused by a new
> one. Like a student who memorized practice answers instead of understanding.
>
> **Technically** — A single deep tree has high variance: near-zero training
> error, poor test error. It overfits.
>
> **In this project** — This is exactly why a single tree isn't used. Your repo
> even has a teaching figure for it: `viz/images/13_tree-depth-overfitting.png`.

### 3.3 Layer 3 — ensembles

> **In plain English** — Don't ask one person; ask a crowd and average their
> opinions. Crowds of mediocre-but-different guessers beat one confident
> guesser.
>
> **Technically** — Combining many weak learners into a strong one. The two
> main strategies are **bagging** (train many models independently on random
> subsets, then average — this is Random Forest) and **boosting** (train models
> sequentially, each fixing the last one's mistakes — this is XGBoost).
>
> **In this project** — XGBoost = boosting. Your repo has figures for the
> contrast: `viz/images/14_bagging-vs-boosting.png` and
> `viz/images/15_boosting-loop.png`.

### 3.4 Layer 4 — boosting: the relay race

> **In plain English** — Imagine 450 students taking the same exam one after
> another. Student 1 answers everything. Student 2 is handed *only the questions
> Student 1 got wrong*, plus how badly each one was missed, and studies those.
> Student 3 gets what's *still* wrong. And so on. At the end you add up
> everyone's corrections.
>
> **Technically** — Additive modeling. Start with a constant `F₀`, then:
>
> ```
> F_m(x) = F_{m-1}(x) + η · h_m(x)
> ```
>
> where `h_m` is a new tree fitted to the current errors and `η` is the
> **learning rate** — a fraction that shrinks each tree's contribution.
>
> **In this project** — `n_estimators=450` means 450 trees, and
> `learning_rate=0.045` means each tree's correction is scaled to 4.5%. Small
> steps, many of them: `450 × 0.045 = 20.25`, so the corrections accumulate
> gently into the final answer.

### 3.5 What the word "gradient" is doing in there

> **In plain English** — To tell the next student what to study, you need to
> know two things: *which direction* you were wrong, and *how much*. That
> direction is the slope of the error.
>
> **Technically** — Gradient boosting fits each new tree to the **negative
> gradient of the loss** with respect to the current predictions. With squared
> error the gradient is simply the residual, which is why the intuition above is
> exact rather than a metaphor.
>
> **In this project** — `objective="reg:squarederror"` (`modeling.py:64`)
> selects the loss. Choosing squared error is also *why* the model predicts
> conditional means, which is what the pricing math needs.

### 3.6 A tiny boosting example you can compute by hand

Four rows, learning rate set to 0.5 to keep the arithmetic readable (the real
project uses 0.045).

| Price | Units |
|---|---|
| 300 | 20 |
| 350 | 18 |
| 400 | 15 |
| 450 | 12 |

**Start with the average:** `F₀ = (20 + 18 + 15 + 12) / 4 = 16.25`

**Residuals** (what's left to explain):

| Price | Units | F₀ | Residual |
|---|---|---|---|
| 300 | 20 | 16.25 | **+3.75** |
| 350 | 18 | 16.25 | **+1.75** |
| 400 | 15 | 16.25 | **−1.25** |
| 450 | 12 | 16.25 | **−4.25** |

**Tree 1** — one split: `price ≤ 375?`
- Left leaf (300, 350): mean residual `(3.75 + 1.75) / 2 = +2.75`
- Right leaf (400, 450): mean residual `(−1.25 − 4.25) / 2 = −2.75`

Apply with `η = 0.5`:

| Price | F₁ = 16.25 + 0.5 × leaf | Residual after Tree 1 |
|---|---|---|
| 300 | 16.25 + 1.375 = **17.625** | +2.375 |
| 350 | **17.625** | +0.375 |
| 400 | 16.25 − 1.375 = **14.875** | +0.125 |
| 450 | **14.875** | −2.875 |

**Tree 2** — same split, fitted to the *new* residuals:
- Left leaf: `(2.375 + 0.375) / 2 = +1.375`
- Right leaf: `(0.125 − 2.875) / 2 = −1.375`

| Price | F₂ = F₁ + 0.5 × leaf | Residual after Tree 2 |
|---|---|---|
| 300 | **18.3125** | +1.6875 |
| 350 | **18.3125** | −0.3125 |
| 400 | **14.1875** | +0.8125 |
| 450 | **14.1875** | −2.1875 |

**The scoreboard:**

| | MAE |
|---|---|
| Before any tree (just the mean) | 2.750 |
| After Tree 1 | 1.438 |
| After Tree 2 | **1.250** |

Two trees cut the error by more than half. Now imagine this 450 times, with each
step a gentle 4.5% nudge instead of 50%. **That is what your model is.**

*Note the honest detail:* the row at 400 got worse after Tree 1 (−1.25 → +0.125,
overshooting past zero and then to +0.8125 after Tree 2). Boosting reduces
*overall* error; individual rows can wobble. A small learning rate is precisely
what keeps this from mattering.

### 3.7 XGBoost the library

> **In plain English** — Boosting is the idea. XGBoost is a very fast,
> well-engineered program that implements the idea, with extra guardrails to
> stop it overfitting.
>
> **Technically** — *eXtreme Gradient Boosting*: a regularized, sparsity-aware,
> second-order (Newton) implementation of gradient boosting, with row and column
> subsampling and an explicit complexity penalty on each tree. It is usually the
> strongest off-the-shelf method for tabular data, which is why it's the default
> choice for problems like this one.
>
> **In this project** — `from xgboost import XGBRegressor` at `modeling.py:14`,
> configured once in `make_xgb_pipeline()` at `modeling.py:58-65`.

### 3.8 Every hyperparameter in this project

These are the numbers at `modeling.py:62-64`. Know each one, because "why those
values?" is a guaranteed question.

| Setting | Value | What it does | What it prevents |
|---|---|---|---|
| `n_estimators` | **450** | How many trees to build | Too few = underfitting |
| `max_depth` | **5** | Max questions per tree | Deep trees memorize noise |
| `learning_rate` | **0.045** | How much each tree counts | Big steps overshoot |
| `subsample` | **0.85** | Each tree sees a random 85% of rows | Trees memorizing specific rows |
| `colsample_bytree` | **0.9** | Each tree sees 90% of the features | Trees all keying on the same feature |
| `reg_lambda` | **2** | Penalty on large leaf values | Confident extreme predictions |
| `objective` | `reg:squarederror` | The loss being minimized | — (defines what "best" means) |
| `random_state` | **42** | Seed for all randomness | Non-reproducible results |
| `n_jobs` | **2** | CPU threads | — (speed only) |

**The honest thing to say about these:** they are sensible, conventional values,
hand-picked rather than tuned. Tuning them — searching for better combinations —
is a standard next step, and on *this* dataset it would barely help, for the
reason explained in §5.8.

**Two real strengths to claim:**

- `random_state=42` is why `python scripts/train.py` produces *exactly* the same
  `metrics.json` every time. That reproducibility is verifiable, and it means
  every number in this document can be reproduced.
- `make_xgb_pipeline()` is the **single source of truth** for the
  hyperparameters. The training script and the product-holdout evaluation both
  call it, so there is no way for the two to drift apart.

---

## §4 — Preprocessing and the Pipeline

Models can't eat raw data. There's always a translation step.

### 4.1 Why raw data can't go straight in

Two problems in this dataset:

1. **Missing values.** Some rows have blanks.
2. **Text categories.** `category` holds the words `"Bracelet"`, `"Jewellery"`...
   Trees split on *numbers compared to thresholds*. `"Bracelet" < 400` is
   meaningless.

### 4.2 Missing values → imputation

> **In plain English** — If a number is missing, guess something sensible rather
> than throwing the row away.
>
> **Technically** — Imputation replaces missing values with a statistic. The
> **median** is preferred over the mean for numeric data because it isn't
> dragged around by extreme values.
>
> **In this project** — `SimpleImputer(strategy="median")` for numeric features
> and `SimpleImputer(strategy="most_frequent")` for categorical ones
> (`modeling.py:36-38`).

### 4.3 Text categories → one-hot encoding

> **In plain English** — Turn each category into its own yes/no column. Instead
> of one column saying `"Bracelet"`, make a column `is_Bracelet` holding 1 or 0.
>
> **Technically** — One-hot encoding creates one binary indicator column per
> distinct category value. It avoids imposing a false order (saying
> `Jewellery = 2, Bracelet = 1` would wrongly imply Jewellery is "twice"
> Bracelet).
>
> **In this project** — `OneHotEncoder(handle_unknown="ignore")`
> (`modeling.py:39`). The `handle_unknown="ignore"` part matters: if the app
> ever receives a category the model never saw during training, it encodes it as
> all-zeros instead of crashing.

### 4.4 The dummy variable trap — and a real quirk in this project

> **In plain English** — If you make a column for "is Winter" *and* columns for
> every other season, one of them is redundant — you can always work out the
> last from the others. That redundancy confuses models that add things up.
>
> **Technically** — With `k` categories and no dropped reference level, the `k`
> indicator columns sum to 1, which is collinear with the intercept. The design
> matrix becomes rank-deficient and coefficients are no longer uniquely
> determined (the solver returns an arbitrary minimum-norm solution).
>
> **In this project** — There is no `drop="first"`, so the trap is present, and
> it's worse than usual because the dataset also contains `festival_flag`, which
> is *exactly* determined by `season` (Festival ⟺ month ∈ {10,11,12}, see
> `data.py:14`), and `unit_cost`, which is *exactly* `material_cost +
> labour_cost + packaging_cost` (`data.py:44`). The encoded matrix has **37
> columns but rank 31** — six dimensions of pure redundancy.
>
> **What this means, honestly:** it doesn't hurt the tree model at all (trees
> pick whichever of two identical columns they like and ignore the rest). It
> *does* mean the linear regression's coefficients should not be interpreted
> individually. So don't present linear coefficients as insights.

### 4.5 The Pipeline — the good engineering, and why it prevents leakage

> **In plain English** — Bundle all the cleaning steps together with the model
> into one object, so the cleaning is done *identically* when you train and when
> you use it later.
>
> **Technically** — `sklearn.pipeline.Pipeline` chains transformers and a final
> estimator, applying them in order. Calling `.fit()` fits every step on the
> training data only; calling `.predict()` applies the already-fitted
> transformations to new data. Fitting an imputer on the *whole* dataset —
> including test rows — leaks information about the test set into training and
> inflates your scores.
>
> **In this project** — `make_preprocessor()` (`modeling.py:34-41`) returns a
> `ColumnTransformer` that routes numeric columns through median imputation and
> categorical columns through imputation + one-hot. The whole thing is wrapped
> in the `Pipeline` at `modeling.py:60-65` and pickled into
> `demand_model.joblib` as a single object.
>
> **Why this is a real strength to claim:** `app.py` never re-implements any
> preprocessing. It calls `model.predict(...)` and gets the same transformation
> the training used. **There is no train/serve skew.** That's an engineering
> guarantee most student projects don't have, and it's free here because of the
> `Pipeline`.

### 4.6 One full row, walked through

Take a Winter week for a bracelet:

```
INPUT ROW
  category = "Bracelet"     material_type = "Beads"
  season = "Winter"         festival_flag = "No"
  price = 399.0             unit_cost = 340.0
  competitor_avg_price = 350.0
  product_rating = 4.6      seller_rating = 4.5
  previous_week_sales = 20  rolling_4_week_sales = 18
  ... (23 feature values in total)

STEP 1 — DERIVE THREE FEATURES          (src/craftprice/features.py)
  price_to_cost                   = 399 / 340          = 1.1735
  competitor_price_gap            = 399 − 350          = 49.0
  estimated_contribution_per_unit = 399 − 340          = 59.0

STEP 2 — IMPUTE (nothing missing here, so values pass through)

STEP 3 — ONE-HOT THE CATEGORIES
  category_Bracelet = 1   category_Jewellery = 0   category_Candle = 0  ...
  material_type_Beads = 1 material_type_Silver = 0 ...
  season_Winter = 1       season_Festival = 0      season_Summer = 0    season_Monsoon = 0
  festival_flag_No = 1    festival_flag_Yes = 0

STEP 4 — CONCATENATE
  19 numeric columns + (6 + 6 + 4 + 2) = 18 categorical columns  =  37 columns

STEP 5 — THE MODEL PREDICTS
  Walk the row down 450 trees, sum all 450 leaf values,
  add the base score  →  e.g. 15.8 predicted units
```

That `37` is exactly the width reported when you inspect the transformed
matrix — so this walkthrough is literally what the code does.

---

## §5 — Judging the model: the metrics

The metrics answer one question: **how wrong is it, and compared to what?**

### 5.1 The residual

> **In plain English** — How far off one prediction was.
>
> **Technically** — `eᵢ = yᵢ − ŷᵢ`. Positive means you under-predicted.
>
> **In this project** — If a week sold 18 units and the model said 15, the
> residual is +3, and you under-predicted by 3.

### 5.2 MAE — Mean Absolute Error

> **In plain English** — "On average, how far off are we?" It ignores whether you
> guessed high or low — a miss is a miss.
>
> **Technically** — `MAE = (1/n) Σ |yᵢ − ŷᵢ|`. It's in the same units as the
> target, which makes it the most interpretable metric.
>
> **In this project** — **4.140 units.** In a typical product-week the demand
> estimate is off by about 4 units. If 16 were actually sold, the model might
> have said 12 or 20.
>
> **Why it's the headline number:** "off by 4 units" means something to a
> seller. "R² of 0.468" doesn't. Lead with MAE.
>
> **A detail worth knowing:** your code clips predictions at zero before scoring
> (`modeling.py:45`), because you can't sell −3 bracelets. Without the clip, some
> predictions would go negative and artificially inflate the error.

### 5.3 RMSE — Root Mean Squared Error

> **In plain English** — Also "how far off are we", but it *really* hates big
> mistakes. One catastrophic prediction counts far more than several small ones.
>
> **Technically** — `RMSE = √((1/n) Σ (yᵢ − ŷᵢ)²)`. Squaring before averaging
> weights large errors heavily; the square root brings it back to target units.
>
> **In this project** — **5.266 units.**

### 5.4 MAE vs RMSE — the ratio is a diagnostic

This is a genuinely useful trick that few students know.

If errors follow a bell curve, then mathematically `RMSE / MAE ≈ 1.253`.

**Your project: `5.266 / 4.140 = 1.272`.** Almost exactly the Gaussian value.

> **Interpretation:** your errors are well-behaved, with no wild outliers. If
> that ratio were 2.0, it would mean a few predictions were catastrophically
> wrong and dragging RMSE up — which is exactly the kind of thing you'd need to
> investigate.

So why report RMSE at all if MAE is easier to read? Because RMSE is more
sensitive to the failures that actually hurt. For a pricing decision, being off
by 30 units on one product costs far more than being off by 5 units on six
products. RMSE is the metric that notices.

*The same constant shows up again in §8.8, where the code uses it to convert a
measured mean-absolute-deviation into a standard deviation.*

### 5.5 R² — the coefficient of determination

> **In plain English** — Out of all the up-and-down movement in sales, what
> fraction did the model manage to explain? 1.0 is perfect. 0.0 means it's no
> better than always guessing the average. **Negative** means it's *worse* than
> guessing the average.
>
> **Technically** — `R² = 1 − SS_res / SS_tot`, where `SS_res = Σ(yᵢ − ŷᵢ)²` and
> `SS_tot = Σ(yᵢ − ȳ)²` is the total variance around the mean.
>
> **In this project** — **0.468**: the model explains about 47% of the
> week-to-week variation in sales.

**The trap to avoid in a viva:** never say "R² is only 0.47, so the model is
weak." That's wrong, and §5.8 explains exactly why.

### 5.6 Baselines — the comparison that makes metrics meaningful

> **In plain English** — A score means nothing on its own. You need something
> dumb to compare against. "I got 4.1" is meaningless; "I got 4.1 and the lazy
> guess gets 5.2" is a result.
>
> **Technically** — A baseline is a trivial predictor that establishes the
> performance floor. If your model can't beat it, you have no model.
>
> **In this project** — two baselines are reported. Here are all the models,
> scored on the same held-out test period:

| Predictor | MAE | RMSE | R² | Where it lives |
|---|---|---|---|---|
| Train-mean ("always guess the average") | 5.763 | 7.253 | −0.010 | `metrics.json` → `historical_average` |
| **Trailing 4-week mean** (reuse the seller's own recent sales) | **5.228** | 6.647 | 0.152 | `metrics.json` → `naive_trailing_4_week` |
| Linear regression | 4.260 | 5.375 | 0.445 | `metrics.json` → `linear_regression` |
| **XGBoost (the shipped model)** | **4.140** | **5.266** | **0.468** | `metrics.json` → `xgboost` |

The **trailing 4-week mean** is the interesting one. It's a real forecasting
rule a seller could apply with no model at all — "next week will be like my last
four weeks" — and it's the *strongest* naive baseline here. Beating it is
therefore a more meaningful claim than beating the train mean, and the project
reports both so you can say so honestly.

**The story in one line:** the model improves on the best naive rule by
`5.228 − 4.140 = 1.088` units of MAE, and on the dumbest rule by `1.623` units.
Quote the smaller, more honest number.

### 5.7 Reading `artifacts/metrics.json`

The file now has **seven blocks**. Read them top to bottom:

```json
{
  "split": { "train_rows": 8820, "validation_rows": 1800, "test_rows": 1980 },

  "historical_average":  { "mae": 5.763, "rmse": 7.253, "r2": -0.010 },
  "linear_regression":   { "mae": 4.260, "rmse": 5.375, "r2":  0.445 },
  "xgboost":             { "mae": 4.140, "rmse": 5.266, "r2":  0.468 },

  "residual_scale": {
    "intercept": 2.204654, "slope": 0.122982, "mad_to_sigma": 1.2533,
    "z_80": 1.2816, "z_measured_80": 1.299, "z_measured_50": 0.6723,
    "coverage_80": 0.7924, "coverage_50": 0.501,
    "method": "linear model of |holdout residual| on prediction, MAD→σ via √(π/2)"
  },

  "per_season": {
    "Festival": { "status": "not_evaluated", "note": "no holdout rows for this season" },
    "Monsoon":  { "status": "not_evaluated", "note": "no holdout rows for this season" },
    "Summer":   { "mae": 4.093, "rmse": 5.181, "r2": 0.458 },
    "Winter":   { "mae": 4.353, "rmse": 5.631, "r2": 0.468 }
  },

  "naive_trailing_4_week": { "mae": 5.228, "rmse": 6.647, "r2": 0.152 },

  "product_holdout": {
    "mae": 4.335, "rmse": 5.542, "r2": 0.530,
    "n_holdout_products": 36, "n_holdout_rows": 2520
  }
}
```

Block by block:

- **`split`** — how many rows went to each period. `8820 + 1800 + 1980 = 12600`,
  the whole dataset. It's 49 dates / 10 dates / 11 dates, times 180 products:
  `49×180 = 8820`, `10×180 = 1800`, `11×180 = 1980`.
- **`historical_average`** — the weak baseline. Predict "the average" always.
- **`linear_regression`**, **`xgboost`** — the two competing models.
- **`residual_scale`** — the calibration of the demand *interval* (explained in
  §8.8). This block is about uncertainty, not accuracy.
- **`per_season`** — the shipped model's error, broken down by season. Two
  seasons are honestly marked `not_evaluated` rather than being filled with a
  number they don't have.
- **`naive_trailing_4_week`** — the strong baseline.
- **`product_holdout`** — how the model does on **36 whole products it has never
  seen**, the app's real use case. 4.335 MAE, essentially identical to the
  time-based holdout — strong evidence the model generalizes rather than
  memorizes.

**One honest caveat to know.** The validation split is created and then folded
into training (`modeling.py:175`), and it is never *scored*. So the `450` trees
are a hand-picked value, not a data-selected one. If asked, say plainly: *"I
carved out a validation split but concatenated it into training without scoring
it, so my tree count is chosen rather than tuned. The fix is to fit on train only
and use the validation split for early stopping."* — and see §6.5.

### 5.8 The noise ceiling — the single most important section in this document

This is the part that turns a mediocre-looking R² into a strong result. Read it
twice.

**Open `src/craftprice/data.py` and look at line 68:**

```python
units = int(max(0, round(expected + rng.normal(0, 2.8 + expected * 0.14))))
```

That's how every sale in the dataset was created:

```
actual sales  =  the "true" formula  +  random noise  (then rounded, floored at 0)
```

The noise is `rng.normal(0, 2.8 + expected × 0.14)` — a bell curve centred on
zero, with a spread that grows with demand. On average across the test rows, that
spread is about **5 units**, on a mean demand of about 16.

**No model can ever predict that noise.** It is, by construction, unconnected to
any feature. It is what "randomness" means.

So I regenerated the data generator, recovered the hidden `expected` values
(the noise-free truth the model never sees), and computed what a **perfect
model** could score:

| | MAE | R² |
|---|---|---|
| Always guess the average | 5.763 | 0.000 |
| Linear regression | 4.260 | 0.445 |
| **XGBoost — the shipped model** | **4.140** | **0.468** |
| **ORACLE — knows the true formula exactly** | **3.882** | **0.566** |

The oracle is not a model you could build. It is the theoretical best any model
could possibly do on this dataset, because it already knows everything knowable.

**Now compute how good your model actually is:**

```
MAE gap to the best possible:   4.140 − 3.882 = 0.258 units
As a fraction:                  0.258 / 3.882 = 6.6% above the ceiling

Improvement over the mean baseline:
  achieved:  5.763 − 4.140 = 1.623 units
  maximum:   5.763 − 3.882 = 1.881 units
  captured:  1.623 / 1.881 = 86%

Improvement over the strongest naive baseline:
  5.228 − 4.140 = 1.088 units better than "repeat last 4 weeks"

R² as a share of the maximum achievable:
  0.468 / 0.566 = 83%
```

> **The sentence to say out loud:**
> *"My R² is 0.47, but the maximum achievable on this dataset is 0.57 — because
> the data generator injects irreducible random noise into every sale. So I'm
> capturing 83% of the explainable variation and my MAE is within 6.6% of an
> oracle that knows the true formula. The remaining error is the data's
> randomness, not the model's weakness. Adding model complexity here would buy
> nothing."*

This also explains why feature engineering doesn't help. I tested adding the
missing `price / competitor_avg_price` ratio — the exact quantity the generator
uses — and MAE moved from 4.140 to 4.139. Nothing. Switching to a Poisson
objective, appropriate for count data: 4.132. Nothing meaningful.

**And here's the flip side — the honest limitation:** because the ceiling is so
close, this dataset *cannot* demonstrate that better modelling would help. It's
excellent for learning the mechanics and the decision layer; it can't show you
what a real, harder forecasting problem looks like. Say that too.

### 5.9 Why there's no MAPE or "accuracy"

> **In plain English** — "Percentage error" breaks when the real answer is near
> zero: if you predicted 3 sales and 0 happened, the percentage error is
> infinite.
>
> **Technically** — MAPE is undefined at `y = 0` and unstable for small `y`;
> it's also asymmetric, penalizing over-predictions more than under-predictions.
>
> **In this project** — Correctly avoided. But be aware of the consequence: for
> low-volume products (actual sales ≤ 8), the *relative* error is about **100%**
> — the absolute error of ~5.7 units is as large as the sales themselves. That
> isn't a bug; it's the noise floor again. And it's precisely why a single
> confident number in the UI is misleading — which is why the app now shows an
> **interval** instead (§8.8).

And note: "accuracy" is a *classification* metric (fraction of correct
categories). It has no meaning for regression. If you're asked "what's the
accuracy?", the correct answer is *"accuracy doesn't apply to regression — I
report MAE, RMSE and R²; my MAE is 4.14 units."*

---

## §6 — Validation: how we know the model isn't cheating

### 6.1 The three sets

> **In plain English** — You can't test a student on the exact questions they
> studied. So you hold some data back entirely.
>
> **Technically** —
> - **Training set** — the model learns from this.
> - **Validation set** — used to make choices *while* building (which model,
>   how many trees). The model doesn't learn from it directly, but your decisions
>   are informed by it.
> - **Test set** — touched exactly once, at the very end, to report the final
>   number. Any tuning against it silently overfits it.
>
> **In this project** — `chronological_split` (`modeling.py:20-31`): 70% / 15% /
> 15% → 8,820 / 1,800 / 1,980 rows.

### 6.2 Chronological splitting — and why it's the right choice here

> **In plain English** — You can't use next month's sales to predict this
> month's. Split by *time*, not at random: train on the past, test on the future.
>
> **Technically** — A time-based holdout: sort by date, take the earliest rows
> for training and the latest for testing. A random split leaks future
> information into training whenever features are built from past values.
>
> **In this project** — This matters a lot, because two features —
> `previous_week_sales` and `rolling_4_week_sales` — are built from *earlier
> weeks of the same product*. A random split would scatter adjacent weeks of one
> product across train and test, letting the model effectively look up the
> answer. The split here is by unique `week_start` date, so **no future row is
> ever used to predict a past one.**

**Most student projects get this wrong.** Be ready to explain it — and be ready
for the follow-up question below, because it has a good answer.

### 6.3 Data leakage and target leakage

> **In plain English** — Accidentally letting the model see the answers, or
> letting it use information that wouldn't exist yet at prediction time.
>
> **Technically** — *Data leakage* is train/test contamination (e.g. fitting a
> scaler on all data). *Target leakage* is a feature that is a function of, or a
> consequence of, the target — so it can't be known at prediction time.
>
> **In this project** — Two things are handled correctly and one deserves an
> honest note:
> - **Correct:** no target leakage. Every feature is knowable before the week's
>   sales happen. No feature is derived from `units_sold`.
> - **Correct:** the preprocessing `Pipeline` is fitted on training data only
>   (§4.5).
> - **Worth admitting:** the synthetic data's price is drawn *independently* of
>   the week's demand shock (`data.py:56`), which means price is effectively
>   randomised. That's the only reason the model's price sensitivity comes out
>   unbiased. **On real data, price and demand move together, and a pooled price
>   coefficient would be biased by confounding** — you'd need within-product
>   (fixed-effects) estimation or an instrument. That's a genuine limitation of
>   the demonstration, and naming it is a strength, not a weakness.

### 6.4 Overfitting and underfitting

> **In plain English** — Overfitting: memorized the practice test, fails the real
> one. Underfitting: didn't study enough, fails both.
>
> **Technically** — Overfit models show a large gap between low training error
> and high test error (high variance). Underfit models show high error on both
> (high bias). The train/test gap is the diagnostic.
>
> **In this project** — Train MAE is about **3.27**, test MAE about **4.14**. A
> gap of 0.87 units on a base of 4 — a mildly optimistic model, which is normal
> and healthy. If train MAE were 1.0 against a test MAE of 4.1, the model would
> be memorizing.

**The strongest anti-overfitting evidence you have** is the product holdout:
**4.335 MAE across 36 products the model has never seen**, essentially identical
to the 4.140 on the time-based holdout. A model that had memorized products would
fall apart there. It doesn't.

### 6.5 What this project does and doesn't do

| Practice | Status here |
|---|---|
| Chronological split | ✅ Done, and correctly |
| No target leakage | ✅ Done |
| Preprocessing fitted on train only | ✅ Done (via `Pipeline`) |
| Test set touched once | ✅ Done |
| Reproducible (seeded) | ✅ Done — `metrics.json` reproduces exactly |
| Strongest naive baseline reported | ✅ Done — trailing 4-week mean, 5.228 |
| Unseen-**product** generalisation | ✅ Done — 36 products held out, 4.335 MAE |
| Per-season error breakdown | ✅ Done, with honest `not_evaluated` where data is absent |
| Prediction interval with measured coverage | ✅ Done — 80% interval covers 79.2% |
| Binding constraint surfaced to the user | ✅ Done — `diagnose_recommendation` |
| Validation split actually *scored* | ❌ Concatenated into training, never evaluated |
| Hyperparameter tuning | ❌ The 450 trees are hand-picked |
| Multiple test windows (rolling-origin) | ❌ One 10-week window |

Only the last three remain open, and each is a deliberate, explainable choice for
a microproject rather than an oversight.

**One composition fact worth knowing before anyone asks:** the test window covers
late February to late April, so it contains **Summer and Winter weeks only — zero
Festival and zero Monsoon weeks.** But "Festival" is the app's *default* season.
The project handles this honestly: `metrics.json` marks those seasons
`not_evaluated`, and the app shows a warning when you select a season that wasn't
in the holdout (`app.py:579-584`). **Volunteering this before you're asked is
worth a lot.**

---

## §7 — SHAP: explaining a black box

This is `src/craftprice/explain.py`, surfaced by the toggle in the dashboard
(`app.py:678-689`).

### 7.1 The problem it solves

> **In plain English** — The model is 450 flowcharts voting together. Nobody —
> including the person who built it — can read that and say *why* it predicted
> 15 units instead of 20. And a seller won't act on a number they can't
> interrogate.
>
> **Technically** — Tree ensembles are not directly interpretable. There is no
> single coefficient to read off, and no closed-form expression to inspect.
>
> **In this project** — Hence SHAP: a method for attributing a single prediction
> to its individual inputs.

### 7.2 The analogy: splitting the restaurant bill

> Imagine four friends eat together and the bill comes to ₹2,000. How much did
> each person contribute? If everyone ordered separately it's easy — but they
> shared starters, split a dessert, and one person's order is what made everyone
> else order more.
>
> SHAP answers exactly that question for a prediction. The "bill" is the
> predicted demand of 15.8 units. The "diners" are the 23 features. SHAP assigns
> each feature a share of the credit or blame.

### 7.3 Shapley values — where the math comes from

> **Technically** — SHAP values are **Shapley values** from cooperative game
> theory, for which Lloyd Shapley won the 2012 Nobel Prize in Economics. For a
> prediction, each feature's SHAP value is its **average marginal contribution
> across every possible ordering** of the features.
>
> The averaging over orderings is what makes it fair: if price only matters when
> season is Festival, that interaction gets split between them in a principled
> way rather than being arbitrarily assigned to whichever the model happened to
> use first.
>
> Two properties make it trustworthy:
> 1. **Local accuracy** — the SHAP values plus the base value (the model's
>    average prediction) sum *exactly* to the prediction. No leftovers.
> 2. **Consistency** — if a feature's true influence increases, its SHAP value
>    cannot decrease.
>
> Nothing else in explainability has both properties, which is why SHAP is the
> standard.

### 7.4 TreeSHAP

> **In plain English** — Computing Shapley values naively is astronomically slow.
> For trees there's a clever exact algorithm.
>
> **Technically** — `TreeExplainer` computes exact Shapley values for tree
> ensembles in polynomial time instead of exponential, by exploiting the tree
> structure.
>
> **In this project** — `explain.py:51`:
> ```python
> values = shap.TreeExplainer(estimator).shap_values(transformed)
> ```

### 7.5 Reading the chart in the app

`explain_recommendation` (`explain.py:23-59`) does five things; know each:

```python
preprocessor = model.named_steps["preprocessor"]     # the ColumnTransformer
estimator    = model.named_steps["model"]            # the 450 trees

transformed = preprocessor.transform(candidates_row)          # → 37 columns
values      = shap.TreeExplainer(estimator).shap_values(transformed)

names = [_strip_prefix(str(n)) for n in preprocessor.get_feature_names_out()]

explanation = (pd.DataFrame({"feature": names, "shap_value": values[0]})
                 .sort_values("shap_value", key=abs, ascending=False)
                 .head(top_n))                                 # top 12
```

What you see on screen:

- **One bar per feature, for one prediction.** This is a *local* explanation —
  it explains the recommended price *in this scenario only*. A different
  scenario gives different bars. **Do not call this "feature importance"** —
  global feature importance is a different thing.
- **Bar length = magnitude of influence.** Sorted by absolute value, so the top
  bar is the biggest driver either way.
- **Bars to the right push the estimate up; bars to the left push it down**,
  relative to the model's average prediction. The caption under the chart says
  exactly this.
- **The names are cleaned up.** `ColumnTransformer` produces names like
  `categorical__season_Festival` and `numeric__price`; `_strip_prefix`
  (`explain.py:15-20`) removes the `categorical__` / `numeric__` part so the
  chart reads `season_Festival` and `price`. If you're asked why the raw names
  have prefixes at all: they're how sklearn tracks *which* transformer produced
  each of the 37 columns.

### 7.6 Caveats you should volunteer

Knowing these, and saying them, is worth more than the chart itself:

1. **SHAP explains the model, not the world.** If the model learned something
   wrong, SHAP will explain that wrong belief very clearly and convincingly.
2. **Collinearity splits credit arbitrarily.** `season_Festival` and
   `festival_flag_Yes` are *exactly identical* columns (§4.4). SHAP will divide
   the credit between two duplicates in an arbitrary way. So don't over-read a
   single bar when its twin is also on screen.
3. **It's the most expensive thing in the app — and it's now handled well.**
   Measured at about **87 ms per call**, against about **12 ms** for the whole
   `optimize_price` search. So it runs **only when you turn the toggle on**
   (`app.py:678`) rather than on every Streamlit rerun, and the result is cached
   by `@st.cache_data` keyed on the scenario row (`app.py:309-313`, called at
   `app.py:683`). It's also wrapped in a `try/except` so an explainability
   failure can never take down the recommendation itself (`app.py:688-689`).

**The answer to "why did you add SHAP?"** — not "to look sophisticated." Say:
*"The output is a price recommendation, and a seller won't act on a number they
can't interrogate. SHAP lets me show which inputs drove this specific estimate,
so the recommendation is auditable. It explains this one prediction, not the
model globally. It's the expensive part of the app, so it's behind a toggle and
cached, and it lives in its own module so the UI doesn't need to know the
pipeline's internals."*

---

## §8 — Stage 2: from demand to a price

This is the part that makes the project interesting. Everything so far produces a
demand estimate; this section turns it into a decision.

### 8.1 Why not just maximize demand?

> **In plain English** — If you want to sell the most units, charge ₹1. You'll
> sell thousands and earn nothing. Selling more is not the goal — **earning more
> is the goal.**
>
> **Technically** — Demand is not the objective function. The objective is
> contribution (unit price minus unit cost, times units sold). Maximizing demand
> subject to no constraint has the degenerate solution `price → 0`.
>
> **In this project** — `pricing.py:107` computes contribution, and
> `pricing.py:109` takes the `argmax` of **contribution**, not of
> `predicted_demand`. That single line is the most important design decision in
> the codebase.

### 8.2 The contribution formula

```
contribution  =  (price − unit_cost)  ×  predicted_demand
                 └── profit per unit ─┘   └── how many units ──┘
```

Two things move in opposite directions as price rises: profit per unit goes up,
units sold goes down. Their *product* is a hill, and you're looking for the peak.

**The columns in the app's candidate table:**

| Column | Formula | Plain meaning |
|---|---|---|
| `price` | grid value | A price you could charge |
| `predicted_demand` | `model.predict(...)` | What the model thinks you'd sell |
| `estimated_revenue` | `price × demand` | Total money in |
| `estimated_contribution` | `(price − unit_cost) × demand` | **Money you actually keep** |
| `margin_pct` | `(price − unit_cost) / price` | Profit as a fraction of price |

**Contribution vs revenue, in one example.** Selling 10 units at ₹500 with a
₹400 cost: revenue ₹5,000, contribution ₹1,000. Selling 15 units at ₹450 with the
same cost: revenue ₹6,750 — *higher* — but contribution ₹750 — *lower*. Revenue
says "charge ₹450"; contribution correctly says "charge ₹500." **This is why the
project optimizes the right thing.**

### 8.3 Margin vs markup — the distinction the code gets right

This is the single most instructive piece of arithmetic in the project, and there
was once a real bug here, so it's worth understanding properly.

> **In plain English** — "I want 20% profit" is ambiguous. 20% *of what*? If you
> add 20% to your cost, you're taking 20% **markup**. If you want 20% of the
> final price to be profit, that's 20% **margin** — and it needs a *higher* price
> than 20% markup does. Confusing the two means you quietly earn less than you
> asked for.
>
> **Technically** —
> ```
> markup on cost:    price = unit_cost × (1 + m)          →  m = (price − cost) / cost
> margin on price:   price = unit_cost / (1 − m)          →  m = (price − cost) / price
>
> For the same price, margin = markup / (1 + markup).
> ```
> So a 20% markup is only a **16.67%** margin, and a 90% markup is only a
> **47.4%** margin. Margin is always the smaller of the two numbers.

**In this project** — `_feasible_bounds` (`pricing.py:27-46`) uses the
**margin-on-price** version:

```python
lower = max(
    constraints.min_price,
    unit_cost / (1 - constraints.min_margin_pct),   # margin on price
    competitor_price * (1 - constraints.competitor_band_pct),
)
```

That is the correct choice, because the app *displays* `(price − unit_cost) /
price` — a margin on price. The slider and the table now measure the same thing,
so the seller's requested margin is genuinely enforced. There's also an explicit
guard for the degenerate case (`pricing.py:38-39`): a margin of 100% or more
needs an infinite price, so it raises a `ValueError` that the app displays as a
readable message.

**The bug that used to be here, and why it's worth telling.** The original code
used the markup form `unit_cost × (1 + min_margin_pct)`. With a ₹100 cost and the
slider at 90%, that set the floor at ₹190 — which is a **47.4%** margin on price,
while the table displayed "47.4%" next to a slider reading "90%". Nothing
crashed; every number was individually correct; the two numbers just measured
different things. That is the most dangerous *kind* of bug.

It's now fixed, and — importantly — **it's fixed with a regression test** so it
can't silently come back:

```python
def test_margin_floor_is_measured_on_price_for_all_margins():
    for margin in (.2, .5, .9):
        table, _ = optimize_price(DemandModel(), high_market,
                                  PricingConstraints(min_margin_pct=margin, price_step=10))
        assert ((table["price"] - 280) / table["price"] >= margin - 1e-9).all()
```

**If you get one question about code quality, hope it's this one.** It shows you
understand that a bug can be a *unit mismatch* rather than a wrong number, and
that the durable fix is a test, not a patch.

### 8.4 The constraints — where the price is even allowed to be

A seller can't just pick any number. `PricingConstraints` (`pricing.py:18-24`)
sets a feasible band:

| Constraint | Default | Meaning |
|---|---|---|
| `min_margin_pct` | 0.20 | Don't sell below this profitability (as a margin on price) |
| `competitor_band_pct` | 0.40 | Stay within ±40% of the competitor's price |
| `min_price` / `max_price` | 0 / 10,000 | Absolute bounds |
| `price_step` | 10 | Only consider prices in multiples of ₹10 |

The feasible range is `lower = max(all the floors)` to `upper = min(all the
ceilings)` (`pricing.py:40-45`).

### 8.5 How the search works

The optimizer doesn't use calculus. It just **tries every candidate** — a brute
force grid search (`pricing.py:86-109`):

```
1. Compute the feasible band:        lower ──────────────── upper
2. List every multiple of the step in that band:
                                     400, 410, 420, ... 490
3. For each candidate price:
      a. write the price into the scenario
      b. recompute the derived price features     (add_price_features)
      c. ask the model for predicted demand       (model.predict)
      d. contribution = (price − unit_cost) × predicted_demand
4. Return the candidate with the largest contribution
```

**Why brute force is the right choice:** there are only a handful of candidates,
the objective isn't guaranteed to be smooth, and brute force has no failure
modes. Using a gradient-based optimizer here would be more clever and less
correct.

**The code is defensive in three specific ways**, each worth knowing:

- **Non-positive step** (`pricing.py:58-60`) → `ValueError`. A step of zero would
  divide by zero; a negative step defines no grid at all.
- **Infeasible band** (`pricing.py:83-84`) → `ValueError` telling the seller to
  widen the range or lower the margin.
- **Empty grid** (`pricing.py:92-99`) → `ValueError` naming the step size, for
  the case where the band is narrower than one step. The comment explains why
  this must be an error rather than an empty frame: an empty frame would later
  explode as `KeyError: 'unit_cost'` somewhere far from the cause.

All three are caught by the app (`app.py:410-412`) and shown to the user as plain
text instead of a stack trace.

### 8.6 A complete worked example (from the code's own demo)

Take the scenario in `pricing.py:198-219`: `unit_cost = 320`,
`competitor_avg_price = 350`, default constraints (20% margin, ±40% band,
₹10 step).

```
lower = max(0,  320 / (1 − 0.20),  350 × 0.60)
      = max(0,  400.00,            210.00)          = 400
upper = min(10000,  350 × 1.40)                      = 490

candidates:  400, 410, 420, 430, 440, 450, 460, 470, 480, 490   (10 prices)
```

Notice the **margin floor of ₹400 is what binds the bottom**, not the competitor
band's ₹210 — the seller's own profitability requirement is the stricter
constraint here. The model then predicts demand at each of the 10 prices, and the
contribution `(price − 320) × demand` is compared. The winner is the
recommendation.

**A detail that shows real care.** Look at `pricing.py:66-82`. The comment
explains that `350 × 1.4` in floating-point arithmetic equals
`489.99999999999994`, not `490`. A naive `floor()` would silently **drop the
₹490 candidate** — the very top of the band — for a reason no one would ever
notice. The code uses a `1e-9` tolerance in step units to keep it:

```python
tolerance = 1e-9          # in units of `step`
first_index = int(np.ceil(lower / step - tolerance))
last_index  = int(np.floor(upper / step + tolerance))
```

And the comment goes further, documenting a second case: with
`unit_cost=34, min_margin=0.80, competitor=125, band=0.36, step=10`, the margin
floor and the band cap are *both mathematically 170*, yet float stores one as
`170.00000000000003` and the other as `169.99999999999997`. A naive
`if lower > upper: raise` would reject a price the seller genuinely has. The
feasibility guard (`pricing.py:83`) uses the same tolerance for exactly that
reason.

**That is a genuine, subtle, correctly-diagnosed numerical issue — found, fixed,
explained in a comment, and covered by a test.**

### 8.7 What the optimizer is really doing: elasticity

> **In plain English** — Elasticity measures how twitchy customers are about
> price. Elasticity 2 means a 1% price rise loses 2% of sales — very price
> sensitive. Elasticity 0.5 means the same rise loses only 0.5% — customers
> barely care, so you should charge more.
>
> **Technically** — `ε = − (∂Q/∂p) × (p/Q)`. In this project's data generator
> (`data.py:64`), demand follows:
>
> ```
> price_effect = exp(−ε × (price / competitor − 1))
> ```
>
> a log-linear demand curve relative to the competitor's price. With that shape,
> the contribution-maximizing price has a clean closed form:
>
> ```
> p* = unit_cost + competitor_avg_price / ε
> ```
>
> **Read that formula and you can predict the whole app's behaviour:** more
> elastic products (high ε) should be priced *closer to cost*; less elastic ones
> (low ε) should be priced *well above* it. And the competitor's price matters
> more than yours — a higher competitor price lets you charge more.
>
> **A useful exercise with it.** With `unit_cost = 320` and `competitor = 350`,
> `p*` rises past the +40% band cap (₹490) whenever `320 + 350/ε > 490`, i.e.
> whenever **ε < 2.06**. Since the true elasticities in this dataset run from
> **0.71 to 1.99**, the unconstrained optimum lies above the band for essentially
> every product. That's *why* the recommendation so often sits on the ceiling —
> it isn't the model misbehaving, it's the constraint binding. Measured directly:
> the recommendation lands on the band's **top edge in about 60% of scenarios**
> (about 74% if you narrow the band to ±15%), and on the bottom edge in **0%** of
> them.

### 8.8 The demand interval — and why the app no longer shows one number

> **In plain English** — "About 16 units" is overconfident. "Somewhere between 9
> and 23" is honest, because we know from the test period how far off this model
> typically is.
>
> **Technically** — The app builds an 80% prediction interval from the
> **holdout residuals**. `_residual_scale` (`modeling.py:68-115`) fits a small
> linear model of `|residual|` against the predicted value — so the interval
> widens for high-demand products, which is exactly how the data generator's
> noise behaves — then converts the estimated mean absolute deviation into a
> standard deviation using the Gaussian constant `MAD = σ√(2/π)`, i.e.
> `σ = MAD × 1.2533`.
>
> **In this project** — the measured calibration is genuinely good:

| Quantity | Nominal | Measured on the holdout |
|---|---|---|
| 80% interval coverage | 80% | **79.2%** |
| 50% interval coverage | 50% | **50.1%** |
| z for 80% | 1.2816 | 1.299 |
| z for 50% | 0.6745 | 0.6723 |

**The code checks its own honesty.** `modeling.py:105-110` compares the nominal
coverage against what actually happened on the holdout, and if they differ by
more than 3 percentage points it *discards the textbook z-value* and stores the
empirical one instead, with a note explaining why. Here 79.2% is within tolerance
of 80%, so the nominal z was kept — and the file records both, plus the method,
so the choice is auditable. The app then labels the interval with the same z it
used for the arithmetic (`app.py:68-88`, `app.py:663-673`).

**A lovely cross-check worth mentioning.** The recovered noise law is
`σ ≈ 2.76 + 0.154 × demand`. The generator's actual noise is
`2.8 + 0.14 × expected` (`data.py:68`). The model independently rediscovered the
structure of the randomness it was trained on — evidence that the residuals are
behaving exactly as the theory says they should.

### 8.9 Known weaknesses of the decision layer

Say these before you're asked. The first three are now *surfaced* by the app
rather than being weaknesses it hides:

1. **The constraint usually decides — and the app now says so.**
   `diagnose_recommendation` (`pricing.py:119-186`) works out whether the
   recommended price sits at (or within one step of) either end of the band, and
   which constraint produced that edge, then returns a plain-language note. The
   app shows it (`app.py:472-473`) — *"This is the highest price inside your ±40%
   competitor band. The band set this price, not the model — widen the band to
   see whether a higher price would earn more."* There's also a "What would
   change this answer?" expander (`app.py:694-726`) that re-runs the optimizer at
   ±10% band width and tables the answers. **This is the highest-value honesty
   feature in the app.**
2. **The surface is flat, so the winner is fragile.** The top 5 candidates
   differ by only about 8% of the best contribution. Refining the price step from
   ₹10 to ₹1 moves the recommendation by more than 1% of the competitor's price
   in about **49% of cases**. So the exact number is less meaningful than the
   range around it — which is why the candidate table and the band shading matter
   as much as the headline.
3. **The demand curve is a step function, not a smooth line.** Because trees make
   piecewise-constant predictions, predicted demand is strictly non-increasing in
   price in only about **2.7%** of candidate grids — the curve wiggles slightly.
   (The *sign* is always right: the implied elasticity is negative in 100% of
   scenarios, median about −1.06.) The app detects this per-grid and says so in
   the caption under the charts (`app.py:485-498`) instead of pretending the
   curve is smooth.
4. **One elasticity for all products.** The model can only learn an *average*
   price response, but true elasticities here vary from 0.71 to 1.99. Measured
   consequence: a single pooled elasticity rule captures only about **47%** of
   the contribution that a perfect per-product pricing rule would achieve. That
   is the single largest source of value left in the project, and it's a genuinely
   interesting limitation to discuss.

---

## §9 — The dashboard, panel by panel

`app.py` is now ~735 lines. Top to bottom, with what each part is for:

| Section | Code | What it does |
|---|---|---|
| **Palette** | `app.py:27-57` | A validated colour-blind-safe palette, with light and dark variants chosen from the Streamlit theme |
| **Formatting helpers** | `app.py:63-65` | `fmt_inr` — ₹ with thousands separators |
| **`demand_interval`** | `app.py:68-88` | Turns `residual_scale` into a ±80% interval for the demand estimate |
| **`make_demand_chart`** | `app.py:91-178` | Demand vs price, with the feasible band shaded, and the recommended and current prices both marked and directly labelled |
| **`make_contribution_chart`** | `app.py:181-270` | The same for contribution — **a separate chart, not a second y-axis** |
| **`_apply_chart_layout`** | `app.py:273-306` | Shared axis/legend/grid styling for both charts |
| **`cached_explain`** | `app.py:309-313` | SHAP, cached on the scenario row so it computes once |
| **Page setup** | `app.py:319-334` | Title, and an early exit with a clear message if the model file is missing |
| **Sidebar: product & market** | `app.py:339-357` | 16 inputs — cost, price, competitor, ratings, season, past sales |
| **Sidebar: constraints** | `app.py:360-368` | Promotional discount, minimum margin **on price**, competitor band, price step |
| **Scenario assembly** | `app.py:376-398` | Builds the exact dict the model expects; derives `unit_cost` and `festival_flag` |
| **Optimize** | `app.py:400-412` | Runs Stage 2; catches `ValueError` and shows the message instead of crashing |
| **Current row & diagnosis** | `app.py:414-418` | Finds the candidate nearest the seller's current price, and works out which constraint bound |
| **Metrics, interval, discount effect** | `app.py:423-438` | Loads `metrics.json`; computes the interval; measures the *actual* discount effect rather than asserting one |
| **Hero KPI row** | `app.py:443-459` | Recommended price (+ delta vs current), demand **with its 80% interval**, contribution, unit cost |
| **Disclaimers** | `app.py:467-473` | Contribution-not-revenue note, plus the binding-constraint note when there is one |
| **Charts** | `app.py:478-498` | Two side-by-side charts; handles the single-feasible-price case; captions the monotonicity honestly |
| **Candidate table** | `app.py:503-554` | Every feasible price, marked `← recommended` / `≈ your current price` / `band floor` / `band ceiling`; warns when the grid is very long; CSV download |
| **Evaluation panel** | `app.py:559-673` | Model/metrics file timestamps with a **staleness warning**, the per-season `not_evaluated` warning, the honest multi-scope score table, and the interval's measured coverage |
| **SHAP** | `app.py:678-689` | Behind a toggle, cached, failure-isolated |
| **Sensitivity** | `app.py:694-726` | "What would change this answer?" — re-runs the optimizer at ±10% band |
| **Footer** | `app.py:731-734` | Synthetic data; association is not causation |

**Things in here that are genuinely good practice**, if you need to point at
something:

1. **No dual-axis chart.** Demand (units) and contribution (₹) are different
   measures on different scales, so they get two charts sharing an x-axis
   (`app.py:493-497`). Overlaying them on one plot with two y-axes would invite
   the reader to compare curve *crossings* that mean nothing.
2. **The evaluation panel is self-auditing.** It compares the metrics file's
   modification time against the model file's and warns if the metrics are older
   (`app.py:561-576`) — because stale metrics displayed as current is a silent,
   common failure.
3. **It shows what it cannot claim.** Seasons absent from the holdout appear in
   the table as `not evaluated` with a NaN score and a reason, not as a blank or a
   fabricated number (`app.py:600-623`), and selecting such a season triggers a
   visible warning.
4. **The optimiser's failure modes are user-facing text.** All three `ValueError`
   paths surface as readable sentences (`app.py:410-412`).
5. **The discount effect is measured, not asserted.** `app.py:430-438` re-predicts
   the recommended row with `discount_pct = 0` and reports the actual difference,
   with a help note saying the measured effect is usually small.
6. **`@st.cache_resource`** (`app.py:328-330`) loads the model from disk once,
   not on every interaction.

---

## §10 — The full story, in order

Tell it as a pipeline and you can't get lost:

1. **`data.py`** invents 12,600 product-weeks: 180 products × 70 weeks. Demand is
   generated from a known multiplicative formula plus random noise.
2. **`config.py`** declares the contract: 23 features, target `units_sold`.
3. **`features.py`** derives three extra price-based features
   (`price_to_cost`, `competitor_price_gap`, `estimated_contribution_per_unit`).
4. **`modeling.py`** splits *by date* (70/15/15), wraps preprocessing in a
   `Pipeline`, trains **linear regression** and **XGBoost**, and scores them on
   the held-out test period — alongside a mean baseline, the trailing-4-week
   baseline, a per-season breakdown, a 36-product holdout, and a calibration
   study of the residual spread.
5. **`metrics.json`** records all of it. XGBoost wins: MAE 4.140, R² 0.468 — and
   it's within 6.6% of the theoretical ceiling imposed by the data's noise.
6. **`pricing.py`** uses that model as a *simulator*: "at ₹400 demand would be X,
   at ₹410 it would be Y..." — then takes the price with the highest contribution
   inside the seller's constraints, and works out whether the model or a
   constraint actually decided it.
7. **`explain.py`** can attribute that one prediction to its inputs with SHAP.
8. **`app.py`** collects the seller's situation, runs steps 6–7, and shows the
   recommended price, a demand interval, two trade-off charts, the full candidate
   list, a sensitivity check, and an honest evaluation panel.

---

## §11 — Numbers to remember

Memorize this table and you can answer almost any factual question.

| Quantity | Value | Where it comes from |
|---|---|---|
| Dataset size | 12,600 rows = 180 products × 70 weeks | `data.py:27` |
| Features | 23 (19 numeric + 4 categorical) | `config.py:13-21` |
| Encoded width | 37 columns (rank 31) | after one-hot encoding |
| Split | 8,820 train / 1,800 val / 1,980 test | `metrics.json` |
| Trees | 450 (`n_estimators`) | `modeling.py:62` |
| Tree depth | 5 (`max_depth`) | `modeling.py:62` |
| Learning rate | 0.045 | `modeling.py:62` |
| Random seed | 42 | `modeling.py:64` |
| Baseline MAE (train mean) | 5.763 | `metrics.json` |
| Baseline MAE (trailing 4-week) | 5.228 | `metrics.json` |
| Linear regression MAE / R² | 4.260 / 0.445 | `metrics.json` |
| **XGBoost MAE / RMSE / R²** | **4.140 / 5.266 / 0.468** | `metrics.json` |
| Oracle (best possible) MAE / R² | **3.882 / 0.566** | regenerated DGP |
| Share of achievable improvement captured | **86%** | (5.763−4.140)/(5.763−3.882) |
| RMSE/MAE ratio | 1.272 (Gaussian would be 1.253) | diagnostic |
| Test error by season | Summer 4.093 · Winter 4.353 · Festival/Monsoon **not evaluated** | `per_season` |
| Product holdout (36 unseen products, 2,520 rows) | MAE **4.335**, R² 0.530 | `product_holdout` |
| Demand interval calibration | 80% interval → **79.2%** coverage; 50% → **50.1%** | `residual_scale` |
| Train MAE (overfitting gap) | ~3.27 vs 4.14 test | measured |
| True elasticities in the data | mean 1.27, range 0.71–1.99 | latent DGP |
| Model's implied elasticity | about −1.06 | fitted from predictions |
| Recommendations on the band's top edge | ~60% (±40% band) | measured |
| Relative error, low-volume products | ~100% | measured |

---

## §12 — Viva questions and answers

**Q: What does your project do?**
*"It's a two-stage decision tool for handmade-product sellers. First a
gradient-boosted regression predicts weekly demand from product, seller, market
and price signals. Then an optimizer simulates every price the seller is allowed
to charge and picks the one that maximizes contribution — profit per unit times
units sold. It's not a sales predictor; the recommendation is the product."*

**Q: Why XGBoost instead of linear regression?**
*"The data-generating process is multiplicative, so the features interact — a
price increase matters more in a low-demand category. A linear model can only add
contributions; trees can represent interactions. XGBoost wins: MAE 4.14 vs 4.26.
But the margin is small because the dataset's error is dominated by noise, not by
model structure."*

**Q: Why not use a neural network?**
*"For tabular data of this size — about 10,000 training rows — gradient-boosted
trees are usually the strongest option, and they train in seconds. And because the
model is already within 6.6% of the theoretical ceiling, more capacity would buy
nothing measurable."*

**Q: R² is only 0.47. Isn't that weak?**
*"No — the ceiling on this dataset is 0.566. I regenerated the data generator and
recovered the hidden noise-free demand, so I know what a perfect model would
score. My MAE is 4.14 against an oracle's 3.88 — within 6.6%. I capture 86% of
the improvement available over the naive baseline and 83% of the explainable
variation. The rest is irreducible noise the generator adds to every sale."*

**Q: How do you know it isn't overfitting?**
*"Four reasons. The split is chronological, so no future information leaks into
training. Train MAE is 3.27 against test MAE 4.14 — a small gap. I report the
strongest naive baseline, not the weakest, and I beat it by 1.09 units. And when
I hold out 36 entire products the model has never seen, MAE is 4.335 —
essentially the same as on the time-based holdout."*

**Q: What is data leakage, and how did you avoid it?**
*"Leakage is letting information into training that you wouldn't have at
prediction time. My features include `previous_week_sales` and a 4-week rolling
mean, both built from earlier weeks of the same product. A random train/test split
would scatter adjacent weeks of one product across both sets, so the model could
effectively look up the answer. Splitting by date prevents that. I also keep all
preprocessing inside a scikit-learn Pipeline, so the imputer and encoder are
fitted on training data only."*

**Q: Why optimize contribution instead of revenue or units?**
*"Maximizing units has a degenerate answer — charge ₹1. Revenue is better but
still wrong: 15 units at ₹450 earns ₹6,750 of revenue but only ₹750 of
contribution, while 10 units at ₹500 earns less revenue but ₹1,000 of
contribution. Contribution is what the seller keeps, so that's what I maximize."*

**Q: In the demo, the recommended price is the highest one allowed. Is the model
even doing anything?**
*"Yes — and that's exactly the right question. The app checks for this. It
compares the recommendation against the feasible band and tells you when a
constraint, not the model, set the price. In about 60% of scenarios the ±40%
competitor band is binding, so the honest answer is 'this is the most you're
allowed to charge' rather than 'the model thinks this is optimal.' There's also a
sensitivity panel that re-runs the optimizer with a wider band, so you can see
whether the constraint is actually costing you money."*

**Q: What does the slider 'minimum margin on price' actually enforce?**
*"A margin measured on the price, not a markup on cost. So 20% means
(price − cost) / price ≥ 0.20, which is a floor of cost / 0.80. Those are
different numbers — a 20% markup is only a 16.7% margin — and getting them
confused was a real bug in an earlier version of this code, which is why there's
now a regression test checking the displayed margin never falls below the
requested one, for margins of 20%, 50% and 90%."*

**Q: What is SHAP and why did you use it?**
*"SHAP assigns each feature a share of the credit or blame for one specific
prediction, using Shapley values from cooperative game theory. I used it because
the output is a price recommendation and a seller won't act on a number they can't
interrogate. It's a local explanation — it explains this one estimate, not the
model globally. It's the slowest part of the app, so it's behind a toggle and
cached, and it lives in its own module so the UI doesn't need to know the fitted
pipeline's internals."*

**Q: What are the limitations?**
*"The data is synthetic, so the model sits at a noise ceiling and this dataset
can't demonstrate that better modelling would help. The simulator assigns price
randomly, so my price sensitivity is unbiased here but would need causal care on
real data. The test window contained no Festival or Monsoon weeks, even though
Festival is the app's default season — the code marks those seasons 'not
evaluated' rather than inventing a number. And for low-volume products the
relative error is about 100%, which is why the demand estimate is shown as an
interval, not a single number."*

**Q: What would you do next?**
*"Three things. Score the validation split — right now it's concatenated into
training without being used, so my 450 trees are hand-picked rather than tuned;
fitting on train only and early-stopping on validation would justify the tree
count with data. Add rolling-origin backtesting instead of a single 10-week test
window, so the score isn't one window's luck. And segment the price response —
the model learns one average elasticity, but true elasticities here range from
0.71 to 1.99, and I measured that a pooled rule captures only about 47% of the
contribution a per-product rule would achieve."*

**Q: What's the biggest thing you'd change if you started over?**
*"I'd segment the price response from the start. One pooled elasticity is the
single largest source of value left in the project — it captures under half of
what a per-product pricing rule would. Everything else is polish."*

---

## §13 — Glossary

| Term | Meaning |
|---|---|
| **Baseline** | A trivial predictor a real model must beat |
| **Binding constraint** | A limit that is active at the optimum, i.e. it decided the answer |
| **Boosting** | Training models sequentially, each fixing the previous one's errors |
| **Contribution** | `(price − unit_cost) × units` — profit actually kept |
| **Cross-validation** | Repeatedly splitting data into different train/validation folds |
| **Design matrix** | The table of all feature values fed to a model, often written `X` |
| **Elasticity (ε)** | % change in demand per 1% change in price |
| **Feature** | An input variable |
| **Gradient** | The slope of the loss; the direction to move to reduce error |
| **Hyperparameter** | A setting you choose before training, e.g. `max_depth` |
| **Imputation** | Filling in missing values with a sensible substitute |
| **Inference / prediction** | Using a trained model on new inputs |
| **Leakage** | Information reaching training that wouldn't exist at prediction time |
| **Learning rate** | How much each boosting step counts toward the final answer |
| **MAE** | Mean Absolute Error — average size of the miss |
| **MAD** | Mean Absolute Deviation — the average residual size, ignoring sign |
| **Margin (on price)** | `(price − cost) / price` |
| **Markup (on cost)** | `(price − cost) / cost` — always larger than the equivalent margin |
| **One-hot encoding** | Turning categories into 0/1 indicator columns |
| **Oracle** | A hypothetical perfect model, used to find the performance ceiling |
| **Overfitting** | Memorizing training data, failing on new data |
| **Parameter** | A number the model learns during training |
| **Pipeline** | A bundled sequence of preprocessing steps and a model |
| **Prediction interval** | A range that should contain the true value a stated % of the time |
| **R²** | Fraction of the target's variation explained by the model |
| **Regression** | Predicting a numeric value |
| **Residual** | One prediction's error: `actual − predicted` |
| **RMSE** | Root Mean Squared Error — like MAE but punishing big misses |
| **Rolling-origin backtesting** | Repeatedly re-training and testing on successive time windows |
| **Shapley value** | A fair way to split credit among contributors (game theory) |
| **SHAP** | An explainability method applying Shapley values to a prediction |
| **Target** | The variable being predicted, written `y` |
| **Test set** | Data held back and used once, to report final performance |
| **Train/serve skew** | When preprocessing differs between training and deployment |
| **Underfitting** | Model too simple to capture the pattern; bad on all data |
| **Validation set** | Data used to make choices while building the model |
| **XGBoost** | A fast, regularized library implementing gradient boosting |

---

## §14 — Where to look in the code

| If you're asked about... | Open |
|---|---|
| The feature list or target | `src/craftprice/config.py:10-21` |
| How the data is generated, and the noise | `src/craftprice/data.py:27-82` (line 67 = the formula, 68 = the noise) |
| The three derived features | `src/craftprice/features.py:5-11` |
| The chronological split | `src/craftprice/modeling.py:20-31` |
| Imputation and one-hot encoding | `src/craftprice/modeling.py:34-41` |
| How MAE/RMSE/R² are computed | `src/craftprice/modeling.py:44-51` |
| Model hyperparameters (single source of truth) | `src/craftprice/modeling.py:58-65` |
| The mean baseline | `src/craftprice/modeling.py:176-181` |
| The interval calibration and its self-check | `src/craftprice/modeling.py:68-115` |
| Per-season metrics | `src/craftprice/modeling.py:118-130` |
| The trailing-4-week baseline | `src/craftprice/modeling.py:133-141` |
| The 36-product holdout | `src/craftprice/modeling.py:144-169` |
| The margin-vs-markup floor | `src/craftprice/pricing.py:27-46` |
| The floating-point tolerance story | `src/craftprice/pricing.py:66-91` |
| The contribution formula and `argmax` | `src/craftprice/pricing.py:105-109` |
| The binding-constraint diagnosis | `src/craftprice/pricing.py:119-186` |
| The SHAP explanation | `src/craftprice/explain.py:23-59` |
| The dashboard inputs | `app.py:339-368` |
| The two trade-off charts | `app.py:91-270` |
| The KPI row and the interval label | `app.py:443-459` |
| The candidate table and its marks | `app.py:503-554` |
| The evaluation panel | `app.py:559-673` |
| The SHAP toggle | `app.py:678-689` |
| The sensitivity check | `app.py:694-726` |
| The saved scores | `artifacts/metrics.json` |
| The optimizer's tests | `tests/test_pricing.py` |
| What to improve next | `improvement.md` |

---

*Companion documents in this repository: `2.md` for the regression and boosting
mathematics in depth; `1.md` for the architecture; `3.md` for a compact project
map; `improvement.md` for the improvement brief that this revision implements.*
