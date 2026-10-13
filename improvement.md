# CraftPrice AI — Dashboard improvement brief

**Audience: a coding agent.** This file is a work order for improving the Streamlit
dashboard in this repository. Read the actual files before changing anything. If any
statement here contradicts the code, trust the code and say so in your report.

---

## 0. How to use this file

- The repository root is the working directory. The UI is a single file: `app.py`.
- Work through **Tier 1 → Tier 2 → Tier 3**. Tier 1 items are correctness bugs; do not
  skip them for cosmetic work.
- Use the **acceptance criteria** on each item as the definition of done.
- Section 3 contains measured numbers. They are **context to keep your UI copy honest**.
  They are **not** values to hardcode. Anything displayed must be computed at runtime
  from `artifacts/` or `data/`.
- Line numbers below refer to the current revision. As you edit, re-locate by content.
- Do not invent a number, a chart series, or a metric. If a value you want is not
  available, either compute it honestly or leave it out and say so.

---

## 1. What this project is

A two-stage decision system for handmade-product micro-sellers:

1. **Demand model** — an `XGBRegressor` inside a scikit-learn `Pipeline` predicting weekly
   `units_sold` from 19 numeric + 4 categorical features (`src/craftprice/config.py`).
2. **Price optimizer** — `optimize_price` (`src/craftprice/pricing.py`) enumerates
   step-aligned candidate prices inside a feasible band, predicts demand for each, and
   returns the `argmax` of `(price − unit_cost) × predicted_demand`.

The Streamlit app is the only user interface. It is a **college microproject using
synthetic data**. It is not deployed, and it is not intended to be.

**Stack already installed — do not add dependencies:** streamlit, plotly, pandas,
numpy, scikit-learn, xgboost, shap, joblib, pytest.

---

## 2. Ground rules

**Do not:**

- Retrain, replace, tune, or "improve" the model. `artifacts/demand_model.joblib` stays
  as it is. (One exception: Tier 2.3 requires *adding* derived metrics to the training
  function — that is a reporting change, not a model change.)
- Add any dependency, or introduce CI, Docker, a lockfile, an API layer, a database,
  authentication, or deployment configuration.
- Rewrite the UI in another framework. It stays Streamlit.
- Remove or soften the synthetic-data and correlation-is-not-causation disclaimers
  (`app.py:82`, `app.py:119`). You may reword them; you may not delete them.
- Hardcode any number from Section 3 into the interface.
- Fabricate a metric, an interval, a coverage figure, or any other quantity.
- Touch `src/craftprice/data.py`, the model hyperparameters, the train/test split, or
  the feature list.

**Do:**

- Keep business logic in `src/craftprice/` and presentation in `app.py`. Where the UI
  currently reaches into the model's internals (`app.py:109-111`, `model.named_steps[...]`),
  move that behind a library function.
- Keep the existing candidate-prices table. It is the accessible, exact-value view of
  the charts and must survive any restyling.
- Preserve the existing handling of infeasible constraints (`app.py:67-73`) and the
  cached model load (`app.py:24-28`).
- Currency is INR, displayed as `₹` with thousands separators. Demand is a count of units.

---

## 3. Measured facts — context only, never display these literals

These were produced by a read-only audit of this repository on 2026-10-12. They are here
so your UI copy does not overstate what the model does. Re-derive anything you display.

| Fact | Value |
|---|---|
| Model, chronological holdout (n=1980) | MAE **4.140**, RMSE **5.266**, R² **0.468** |
| Train-mean baseline (the only one currently reported) | MAE 5.763, R² −0.010 |
| **Strongest** naive baseline (trailing 4-week mean) | MAE **5.228**, R² 0.152 |
| Naive persistence (`previous_week_sales`) | R² −0.210 |
| **Oracle** — the true conditional mean of the simulator | MAE **3.882**, R² **0.566** |
| Composition of the holdout window | Summer 1620 · Winter 360 · **Festival 0 · Monsoon 0** |
| Model error by season (holdout) | Summer MAE 4.093 · Winter MAE 4.353 |
| Irreducible noise | sd ≈ `2.8 + 0.14 × expected demand`; mean ≈ 5.0 units |
| Relative error, low-volume products (actual ≤ 8 units) | **≈100%** (n=278) |
| Recommendation sits on the **upper** constraint bound | ~60% at ±40% band, ~74% at ±15% |
| Recommendation sits on the **lower** constraint bound | **0%** |
| Recommendations above the seller's own current price | **97%** (mean ratio 1.37) |
| Predicted demand strictly non-increasing in price | only **2.7%** of candidate grids |
| Implied price elasticity of the model | median **−1.06** (always negative) |
| `discount_pct` at inference | hardcoded `0.0`; training has 0.05/0.10 in 40% of rows |

**What this means for your copy — in plain terms:**

- The model is within **6.6% MAE of an oracle that knows the true data-generating
  process.** It is at the noise floor of this dataset. Never write that the model could
  be made substantially more accurate, and never imply more precision than a ±5-unit
  scatter on a ~16-unit mean supports.
- The holdout window contains **no Festival and no Monsoon weeks**, while Festival is the
  app's default season. Any per-season display must make that absence visible rather
  than imply the model was validated across all seasons.
- The headline recommendation is **usually the constraint talking, not the model**.
  Saying so is the single most valuable honesty feature in this brief (Tier 2.1).

---

## 4. Current state of the interface

All in `app.py`:

| Lines | What it does |
|---|---|
| 16-18 | `set_page_config`, title, caption |
| 20-22 | Guard for a missing model artifact |
| 24-28 | `@st.cache_resource` model load |
| 30-54 | Sidebar: 16 product/market inputs + 3 optimisation constraints |
| 56-65 | Builds the scenario dict (note: `unit_cost` re-derived at line 56) |
| 67-73 | `optimize_price` call; `ValueError` → `st.error` + `st.stop` |
| **75** | **Dead code:** `current` is computed and never used |
| 76-80 | KPI row: 4 `st.metric` tiles |
| 82 | Disclaimer banner |
| 84-87 | One `px.line` chart, three series melted together |
| 89-94 | Candidate-price table (the accessible table view — keep it) |
| 95-103 | "Model evaluation" panel reading `artifacts/metrics.json` |
| 105-117 | SHAP expander |
| 119 | Footer disclaimer |

---

## 5. Tier 1 — Correctness (the UI currently shows wrong or stale information)

### T1.1 — The "Minimum margin" slider enforces a *markup*, then the table shows a *margin*

**Severity: high — this is a wrong-money bug in the primary output.**

`src/craftprice/pricing.py:42-43` computes the lower bound as
`unit_cost * (1 + min_margin_pct)`. That is **markup on cost**. The app labels the control
"Minimum margin" (`app.py:52`) and then displays `margin_pct = (price − unit_cost) / price`
(`pricing.py:88`, formatted at `app.py:93`) — **margin on price**. The two differ by
`margin = markup / (1 + markup)`.

Observed through the real code path, with `unit_cost = 100`:

| Slider | Lowest feasible price | Margin displayed at that price |
|---|---|---|
| 0.20 | 120.00 | **0.1667** |
| 0.50 | 150.00 | **0.3333** |
| 0.90 | 190.00 | **0.4737** |

A seller who sets a 90% floor is shown a recommended margin of 47.4%, with no warning.

**Do this — pick one and be consistent:**

- **Preferred (fix the math).** Change the bound to `unit_cost / (1 - min_margin_pct)` so
  the slider means what it says, and guard `min_margin_pct >= 1` with a `ValueError`
  (the slider caps at 0.90, but `optimize_price` is public).
- **Alternative (zero risk).** Keep the math and relabel the control
  "Minimum markup on cost", adding a `markup_pct` column beside `margin_pct`.

Either way, update `docs/METHODOLOGY.md`, which currently says "Feasibility requires the
seller's minimum margin" — true only under the preferred fix.

**Acceptance criteria:**

- No displayed margin is ever below the value the slider enforces.
- A test asserts that for every returned candidate, `(price − unit_cost) / price >= m`.
- You report the measured change in recommended prices (the margin floor is not the
  binding constraint in most scenarios — the competitor band usually is — so expect little
  change, but measure it rather than assuming).
- Confirmed the change does **not** affect `artifacts/metrics.json`: those metrics come
  from the demand model, not from the optimizer.

### T1.2 — `discount_pct` is hardcoded at inference

`app.py:62` sends `"discount_pct": 0.0` always, but 40% of training rows carry a 0.05 or
0.10 discount. The app therefore cannot answer "should I discount?".

**Do this:** either expose a discount input in the sidebar and pass it through, or leave
the value fixed and add a visible caption stating that discounts are not modelled.
Silently ignoring a real seller decision is the one option that is not acceptable.

**Honesty requirement:** the measured effect is **small** — mean |% change| in predicted
demand is 1.34% at 0.05 and 1.68% at 0.10. Do not present this control as a major lever.

### T1.3 — Dead code that should become a feature

`app.py:75` computes `current` (the candidate nearest the seller's current price) and never
uses it, at the cost of an O(n) `argsort` on every rerun.

**Do this:** use it. Mark the seller's current price on the demand chart and highlight its
row in the candidate table, so the recommendation can be read as a *move* from where the
seller is now. Deleting the line is the minimum acceptable fix; using it is the improvement.

### T1.4 — The evaluation panel is not attributable to the loaded model

`app.py:95-103` renders whatever `artifacts/metrics.json` contains, labelled "Model
evaluation", for whichever `demand_model.joblib` was loaded. Replace the artifact and the
displayed metrics silently describe a different model.

**Do this:** add a caption identifying the artifact the numbers came from (the file's
modification time, or a short hash of it if you add one), and state which split the
metrics are from ("chronological holdout"). If the metrics file is older than the model
file, say so explicitly rather than presenting the numbers as current.

---

## 6. Tier 2 — Honesty and information (the highest-value polish)

These change the app from "a number" into "a number with its caveats attached" — which is
what the project is actually about.

### T2.1 — Tell the user when the constraint, not the model, set the price

In roughly 60% of scenarios (74% at a ±15% band) the recommendation is simply the highest
price the band allows, and in **0%** of scenarios is it the lowest. The app presents that
as a model insight.

**Do this:** determine whether `best["price"]` sits at the upper or lower edge of the
feasible band, and if so display a plain-language note, e.g.:

> *This is the highest price inside your ±40% competitor band. The band set this price,
> not the model — widen the band to see whether a higher price would earn more.*

You can compute this in `app.py` from the returned candidate frame plus the
`PricingConstraints` you passed, or return explicit flags from `optimize_price`. Either is
acceptable; a library function is cleaner.

**Acceptance criteria:** with the band slider at 5%, the note appears on a scenario whose
optimum lies outside the band; with the band wide (e.g. 100%) and an interior optimum, it
does not.

### T2.2 — Put an interval on the demand estimate

The app currently prints `Estimated weekly demand: 15.8 units` to one decimal place. The
scatter in this data is ±5 units on a ~16-unit mean, and for low-volume products the
relative error is ~100%. That single decimal is the least defensible claim in the UI.

**Do this:**

1. In `train_models` (`src/craftprice/modeling.py`), compute the holdout residual scale as
   a function of predicted demand. The noise here is heteroscedastic and close to Gaussian
   (measured RMSE/MAE = 1.272, against 1.253 for a perfect Gaussian), so a linear model of
   the residual spread against the prediction is appropriate — e.g. regress `|residual|`
   on the prediction, multiply by `sqrt(π/2) ≈ 1.2533` to convert mean absolute deviation
   to a standard deviation, or bin the predictions and store a per-bin residual sd.
2. Write it to `artifacts/` (a new `residual_scale.json`, or a new block in
   `metrics.json`).
3. In the app, display an interval alongside the point estimate, e.g.
   `16 units (80% interval: 9–23)`.
4. Label the method precisely: *"interval from holdout residuals (±1.28σ); it reflects
   historical scatter, not a guarantee."* This is a residual-based interval, **not** a
   confidence interval on a true mean, and it must not be described as one.

**Required verification — do not skip:** measure the empirical coverage of the interval on
the chronological holdout. Report the number you measured. If it is not close to 80%,
either fix the scale model or label the interval with its true coverage. Claiming 80%
without having measured it is a fabrication.

### T2.3 — Report how the model performs where it actually matters

Currently `metrics.json` holds one test-set number, and the holdout window contains **zero
Festival and zero Monsoon rows** — while Festival is the app's default season. The reported
metric does not cover the app's main use case, and the interface does not reveal that.

**Do this:**

1. Extend `train_models` to also compute and store:
   - **per-season** MAE/RMSE/R² on the holdout, including rows *for seasons with no
     holdout data* — represent those explicitly as "not evaluated" rather than omitting
     them;
   - the **strongest naive baseline** (trailing 4-week mean, i.e. `rolling_4_week_sales`),
     not only the train mean. Reporting the weakest available baseline overstates the
     model's achievement;
   - a **product-level holdout** metric: hold out whole `product_id`s and report the score.
     This is the app's real use case — a product the model has not seen. (Measured at
     audit time: MAE 4.324, R² 0.468.)
2. Display these in the evaluation panel as a small table, with a caption naming the split
   and noting the seasons that were not evaluated.

Retraining is deterministic and seeded, so `python scripts/train.py` reproduces the existing
metrics exactly; you are adding keys, not changing values.

### T2.4 — Make the current price and the feasible band visible in the numbers

In the candidate table, mark: the recommended row, the row nearest the seller's current
price (from T1.3), and the feasible lower/upper bounds. Nothing in the current table tells
the reader which row is the answer or where the band ends.

---

## 7. Tier 3 — Visual design and performance

### T3.1 — The main chart mixes units on one axis

`app.py:84-87` melts `predicted_demand` (units, ~0-30), `estimated_revenue` (₹, ~0-15,000)
and `estimated_contribution` (₹, ~0-5,000) into one line chart on a shared y-axis. The
demand series is squashed flat against the axis while the rupee series dominate.

**Never put two different measures on one axis, and never add a second y-axis.** Split it:

- **Chart A — demand curve:** predicted demand (units) against candidate price, with the
  recommended price direct-labelled and the feasible band shown as a shaded region.
- **Chart B — the objective:** estimated contribution (₹) against candidate price, with
  the argmax direct-labelled.

Revenue is `price × demand` and contribution is `(price − cost) × demand`, so the two
rupee series are near-duplicates of each other; contribution is the one the app optimises.
Drop revenue from the chart (it can stay in the table), or give it its own chart if you
want it.

**Also note in a caption, honestly:** predicted demand is strictly non-increasing in price
in only 2.7% of candidate grids, because a tree ensemble is a step function. The curve
will look slightly wiggly. Do not smooth it or hide it — caption it.

### T3.2 — Design direction

Define the palette once as module-level constants rather than using Plotly defaults, and
set **explicit** colors on every trace so the charts render correctly in both Streamlit
themes (the default template assumes a white surface and can render ink-on-dark
unreadable).

Use these values — they are a validated set (adjacent-pair CVD separation and contrast
checked), not a guess:

| Role | Light | Dark |
|---|---|---|
| Chart surface | `#fcfcfb` | `#1a1a19` |
| Series 1 (primary / demand) | `#2a78d6` | `#3987e5` |
| Series 2 (secondary / contribution) | `#eb6834` | `#d95926` |
| Muted axis & tick labels | `#898781` | `#898781` |
| Gridline, hairline | `#e1e0d9` | `#2c2c2a` |
| Axis / baseline | `#c3c2b7` | `#383835` |
| Status: good | `#0ca30c` | `#0ca30c` |
| Status: warning | `#fab219` | `#fab219` |
| Status: critical | `#d03b3b` | `#d03b3b` |

Rules to hold to:

- **One axis per chart.** No dual y-axes, ever.
- Assign categorical hues in a **fixed order** and never cycle or generate new ones. Two
  series is the target here; you do not need more.
- A legend is present whenever there are two or more series. Direct-label selectively
  (the recommendation, the current price) — never a number on every point.
- Gridlines and axis rules are **solid hairlines**, one shade off the surface. Never dashed.
- Thin marks: ~2px lines, ≥8px markers.
- **Never encode meaning in color alone.** The candidate table is the exact-value twin of
  the charts — keep them consistent.
- Do not use a status color for a non-status series.
- **Do not color the price delta green/red.** `app.py:78` renders
  `₹{best["price"] - current_price:+,.0f} vs current`, and Streamlit's delta styling implies
  good/bad. A higher price is not inherently good — it depends on the demand response. Use
  a neutral delta or a plain text label.
- Everything, including large numbers, in the system sans. No display or serif face.
- Reserve `tabular-nums` for columns that align vertically (table rows, axis ticks), not
  for large standalone stat-tile values.

### T3.3 — Layout and states

- The KPI row (`app.py:76-80`) is the hero. Make the recommended price the visually
  dominant figure; unit cost, demand (with its interval), and contribution support it.
- The candidate table should not require horizontal scrolling on a phone-width window.
  Check the layout at ~375px: 16px side gutter, no horizontal page scroll.
- Handle these states explicitly, since the current code does not:
  - **one candidate only** (a band narrower than the step) — the table and charts look
    broken; say "your constraints allow a single price" instead;
  - **a very large grid** (`price_step` = 1) — warn that the table is long;
  - **model load in progress** — the `st.cache_resource` path should not flash an empty
    page.
- Check for Streamlit deprecation warnings (e.g. `use_container_width` is deprecated in
  recent versions) and clear them.

### T3.4 — SHAP cost

The SHAP block (`app.py:105-117`) sits inside `st.expander`, but Streamlit executes a
collapsed expander's body on every rerun. Measured: **87 ms per rerun**, against 12 ms for
`optimize_price` itself.

**Do this:** compute the explanation only when the user asks for it (a `st.toggle` or
session-state gate), and cache it against the scenario. While you are there, move the
`model.named_steps[...]` access out of `app.py:109-111` into a function in
`src/craftprice/` (e.g. `explain.py`) so the UI is not coupled to the estimator's internal
structure.

### T3.5 — Optional, only if the above is done

- A CSV download button for the candidate table.
- A short "what would change this answer" panel: recompute the recommendation at a band
  ±10 percentage points and show how much the price moves. This makes the fragility of the
  recommendation visible instead of hiding it — which is the same honesty theme as T2.1.

---

## 8. Verification — required before you report done

1. `streamlit run app.py` starts with **no exceptions and no warnings** on load, and the
   full flow (change inputs → recommendation → charts → table → SHAP) works.
2. `python3 -m pytest tests -q` **currently fails** with
   `ModuleNotFoundError: No module named 'craftprice'` — the `src/` layout is not on the
   path. Fix it in four lines by adding a `pyproject.toml` containing:
   ```toml
   [tool.pytest.ini_options]
   pythonpath = ["src"]
   ```
   Then leave the suite passing, and add the margin-invariant test from T1.1.
3. For each Tier 1 item, state the before/after behaviour you observed — not just that you
   changed the code.
4. Review both the light and the dark theme.
5. Report the **measured** interval coverage from T2.2 and the measured change in
   recommended prices from T1.1.
6. Search the diff for any literal from Section 3 and confirm none is hardcoded.
7. Confirm the diff is limited to the files you intended to touch.

---

## 9. Out of scope — do not do these

- Do not add dependencies, CI, Docker, packaging beyond the `pyproject.toml` above,
  lockfiles, an API layer, a database, authentication, or deployment configuration.
- Do not retrain, retune, or replace the model; do not change features, the split, or
  `data.py`.
- Do not rewrite the model to be "more accurate". It is within 6.6% of the theoretical
  ceiling of this dataset; there is nothing to gain and any claim otherwise would be false.
- Do not remove the disclaimers.
- Do not add decorative animation, illustration, or a hero image.
- Do not pad the page with charts that carry no information. Every added chart must answer
  a question the user actually has.
- Do not present a model estimate as a guarantee, a prediction, or a causal effect.

---

## 10. Report format

When you are done, report:

1. **What changed** — file and location, and the reason, per item.
2. **What you verified** — the exact commands and observations, and their results.
3. **What you could not verify** — anything you could not run or measure, stated plainly.
4. **Anything in this brief that the code contradicts** — if you found a statement here
   that is wrong, say so rather than working around it.
5. **Numbers you measured** — interval coverage (T2.2), change in recommended prices
   (T1.1), and anything else you had to measure to complete the work.
