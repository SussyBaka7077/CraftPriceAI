"""Concept figures: how linear regression and gradient boosting actually work.

Every model here is implemented from scratch in toy_models.py (numpy only), so
each picture is drawn from the real mechanism rather than from a library's
output. Where possible the numbers reproduce the worked examples already in
project_understanding.md, so the images and the notes agree line for line.

Run:
    python viz/concept_figures.py            # all figures
    python viz/concept_figures.py 07         # just one, while iterating
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle

from _style import (ACCENT, ALT, ERROR, GOOD, GRID, INK, MODEL, MUTED, PANEL,
                    Registry, annotate_box, titled, write_manifest)
from toy_models import (TOY_DEMAND, TOY_PRICE, BaggedTrees, GradientBoostedTrees,
                        RegressionTree, curved_demand, gradient_descent, lasso_fit,
                        loss_surface, mse, ols_closed_form, ols_normal_equation,
                        r2_score, ridge_fit, split_scan, standardize, true_curve)
from tree_draw import draw_tree, from_regression_tree

reg = Registry()
REGRESSION = "1 · Regression"
BOOSTING = "2 · Trees & gradient boosting"

OUT_DIR = Path(__file__).resolve().parent / "images"
MANIFEST = Path(__file__).resolve().parent / "manifest.json"


# ==========================================================================
# PART 1 — REGRESSION
# ==========================================================================
@reg.register(
    "01", "least-squares-line", "The least-squares line",
    "The five toy rows from the notes. The dashed red sticks are the residuals — "
    "the vertical distance between what happened and what the model says. Least squares "
    "picks the one line out of infinitely many that makes the sum of those sticks, squared, "
    "as small as possible. It is a definition, not a guess.",
    REGRESSION)
def fig_least_squares():
    x, y = TOY_PRICE, TOY_DEMAND
    w, b = ols_closed_form(x, y)
    pred = w * x + b

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    grid = np.linspace(8.6, 19.4, 200)
    ax.plot(grid, w * grid + b, color=MODEL, lw=2.6, zorder=3,
            label=f"fitted line:  demand = {b:.0f} − {abs(w)} × price")

    for xi, yi, pi in zip(x, y, pred):
        ax.plot([xi, xi], [yi, pi], color=ERROR, lw=2.0, ls=(0, (3, 2.4)), zorder=2)
        if abs(yi - pi) > 1e-9:
            ax.annotate(f"{yi - pi:+.0f}", xy=(xi, (yi + pi) / 2), xytext=(7, 0),
                        textcoords="offset points", color=ERROR, fontsize=10,
                        fontweight="bold", va="center")

    ax.scatter(x, y, s=110, color=INK, zorder=5, label="observed (actual demand)")
    ax.scatter(x, pred, s=70, facecolor="white", edgecolor=MODEL, lw=2,
               zorder=5, label="predicted by the line")

    ax.scatter([15], [w * 15 + b], s=150, marker="*", color=ACCENT,
               edgecolor="#8A5A00", lw=0.8, zorder=6)
    ax.annotate("a price never observed:\n145 − 4.5×15 = 77.5",
                xy=(15, w * 15 + b), xytext=(15.3, 92), fontsize=9.5, color="#8A5A00",
                arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.4))

    annotate_box(ax, 0.025, 0.28,
                 "w = Σ(xᵢ−x̄)(yᵢ−ȳ) / Σ(xᵢ−x̄)²\n"
                 f"  = −180 / 40 = {w}\n"
                 f"b = ȳ − w·x̄ = 82 − ({w})(14) = {b:.0f}\n\n"
                 "residuals: [0, −1, +2, −1, 0]\n"
                 f"R² = {r2_score(y, pred):.3f}")

    ax.set_xlabel("price"); ax.set_ylabel("weekly demand (units)")
    ax.legend(loc="upper right")
    titled(fig, "1 · The least-squares line",
           "Every ₹1 of price is associated with ~4.5 fewer units. The slope is the story; the intercept just anchors the line.")
    return fig


@reg.register(
    "02", "why-squared-error", "Why we square the errors",
    "Each residual is drawn as a literal square, so its area is the squared error. "
    "Least squares minimises the total shaded area. Squaring stops positive and negative "
    "misses from cancelling, and punishes one big miss far more than several small ones — "
    "which is exactly why the fitted line refuses to ignore an outlier.",
    REGRESSION)
def fig_squared_error():
    x, y = TOY_PRICE, TOY_DEMAND
    w_best, b_best = ols_closed_form(x, y)
    candidates = [(-3.0, 130.0, "a plausible-looking line"), (w_best, b_best, "the least-squares line")]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 6.4))
    for ax, (w, b, name) in zip(axes, candidates):
        pred = w * x + b
        resid = y - pred
        total = float(np.sum(resid ** 2))

        ax.set_xlim(7.0, 22.5)
        ax.set_ylim(52, 108)
        ax.set_box_aspect(1)  # makes a square in data units look square on screen
        x_span = 22.5 - 7.0
        y_span = 108 - 52

        grid = np.linspace(7.5, 22.0, 100)
        ax.plot(grid, w * grid + b, color=MODEL if name.startswith("the") else MUTED,
                lw=2.4, zorder=3)
        for xi, yi, pi, ri in zip(x, y, pred, resid):
            side_y = abs(ri)
            side_x = side_y * (x_span / y_span)  # equal visual side length
            ax.add_patch(Rectangle((xi, min(yi, pi)), side_x, side_y,
                                   facecolor=ERROR, alpha=0.22,
                                   edgecolor=ERROR, lw=1.2, zorder=1))
            ax.plot([xi, xi], [yi, pi], color=ERROR, lw=1.4, zorder=2)
        ax.scatter(x, y, s=90, color=INK, zorder=5)
        ax.set_title(f"{name}\ndemand = {b:.0f} − {abs(w):.1f} × price", fontsize=11.5)
        annotate_box(ax, 0.04, 0.20,
                     f"Σ residual²  =  {total:,.0f}" +
                     ("\n(the smallest value possible)" if name.startswith("the") else
                      f"\n{total / np.sum((y - (w_best * x + b_best)) ** 2):.0f}× worse"),
                     color=INK if name.startswith("the") else "#8A3B3B")
        ax.set_xlabel("price")
    axes[0].set_ylabel("weekly demand (units)")

    titled(fig, "2 · Why the error is squared",
           "Area = squared error. Minimising total area is what 'best fit' means — and why big misses dominate the fit.")
    return fig


@reg.register(
    "03", "loss-surface", "The loss surface is a bowl",
    "Fix the data, vary the two parameters, and plot the resulting MSE. The surface is a "
    "convex bowl with exactly one bottom. That single fact is why linear regression has a "
    "closed-form answer and why gradient descent on it can never get stuck in a bad local "
    "minimum — there are no other minima to get stuck in.",
    REGRESSION)
def fig_loss_surface():
    x, y = TOY_PRICE, TOY_DEMAND
    w_best, b_best = ols_closed_form(x, y)
    w_grid = np.linspace(w_best - 3, w_best + 3, 150)
    b_grid = np.linspace(b_best - 40, b_best + 40, 150)
    Z = loss_surface(x, y, w_grid, b_grid)
    W, B = np.meshgrid(w_grid, b_grid)

    fig = plt.figure(figsize=(12.6, 5.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax3d.plot_surface(W, B, Z, cmap="YlGnBu_r", alpha=0.95, linewidth=0,
                      antialiased=True, rstride=2, cstride=2)
    ax3d.scatter([w_best], [b_best], [Z.min()], color=MODEL, s=70, depthshade=False)
    ax3d.set_xlabel("w  (slope)", fontsize=9.5, labelpad=2)
    ax3d.set_ylabel("b  (intercept)", fontsize=9.5, labelpad=4)
    ax3d.set_zlabel("MSE", fontsize=9.5, labelpad=2)
    ax3d.view_init(elev=26, azim=-60)
    ax3d.set_title("MSE(w, b) — a convex bowl", fontsize=11.5, pad=0)
    ax3d.tick_params(labelsize=8)

    ax = fig.add_subplot(1, 2, 2)
    levels = np.geomspace(max(Z.min(), 1e-2) + 0.5, Z.max(), 40)
    cs = ax.contourf(W, B, Z, levels=levels, cmap="YlGnBu_r", alpha=0.85)
    fig.colorbar(cs, ax=ax, label="MSE", shrink=0.88)
    ax.scatter([w_best], [b_best], marker="*", s=300, color=MODEL,
               edgecolor="white", lw=1.3, zorder=5)
    ax.annotate(f"global minimum\nw = {w_best}, b = {b_best:.0f}",
                xy=(w_best, b_best), xytext=(w_best - 2.7, b_best + 26),
                fontsize=10, color="white", fontweight="semibold",
                arrowprops=dict(arrowstyle="->", color="white", lw=1.4))
    ax.set_xlabel("w  (slope)"); ax.set_ylabel("b  (intercept)")
    ax.set_title("Same surface from above", fontsize=11.5)
    ax.grid(False)

    titled(fig, "3 · What 'minimising the loss' actually means",
           "The bowl is stretched into a long diagonal valley because w and b are correlated — remember that for figure 6.")
    return fig


@reg.register(
    "04", "gradient-descent", "Gradient descent: rolling downhill",
    "Instead of solving for the answer, start anywhere and repeatedly step against the "
    "slope of the loss. Each arrow is one update of w and b. It converges to the exact same "
    "point the closed-form formula gives — it must, because the bowl has one bottom. "
    "This matters because most models have no closed form, and this loop is all they have.",
    REGRESSION)
def fig_gradient_descent():
    x_raw, y = TOY_PRICE, TOY_DEMAND
    x = (x_raw - x_raw.mean()) / x_raw.std()  # standardised: see figure 6 for why
    run = gradient_descent(x, y, lr=0.12, steps=60, w0=6.0, b0=40.0)
    w_star, b_star = ols_closed_form(x, y)

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.8))
    ax = axes[0]
    w_grid = np.linspace(-20, 12, 160)
    b_grid = np.linspace(30, 120, 160)
    W, B = np.meshgrid(w_grid, b_grid)
    Z = loss_surface(x, y, w_grid, b_grid)
    levels = np.geomspace(Z.min() + 1, Z.max(), 20)
    ax.contour(W, B, Z, levels=levels, colors=[MUTED], linewidths=0.7, alpha=0.7)
    ax.plot(run["w"], run["b"], color=ALT, lw=1.6, alpha=0.85, zorder=3)
    step = 3
    ax.quiver(run["w"][:-1:step], run["b"][:-1:step],
              np.diff(run["w"])[::step], np.diff(run["b"])[::step],
              angles="xy", scale_units="xy", scale=1, color=ALT,
              width=0.006, zorder=4)
    ax.scatter([run["w"][0]], [run["b"][0]], s=90, color=ACCENT,
               edgecolor="white", lw=1.2, zorder=6, label="random start")
    ax.scatter([w_star], [b_star], marker="*", s=280, color=MODEL,
               edgecolor="white", lw=1.2, zorder=6, label="closed-form answer")
    ax.set_xlabel("w  (slope, standardised price)"); ax.set_ylabel("b  (intercept)")
    ax.set_title("The path taken across the loss surface", fontsize=11.5)
    ax.legend(loc="lower left")
    ax.grid(False)

    ax = axes[1]
    ax.plot(run["loss"], color=ALT, lw=2.4)
    ax.axhline(mse(y, w_star * x + b_star), color=MODEL, ls="--", lw=1.6,
               label="loss at the closed-form solution")
    ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("MSE (log scale)")
    ax.set_title("Loss falls fast, then crawls", fontsize=11.5)
    ax.legend(loc="upper right")
    scale = TOY_PRICE.std()
    annotate_box(ax, 0.40, 0.62,
                 f"after {len(run['loss']) - 1} steps\n"
                 f"gradient descent:  w = {run['w'][-1]:.3f}\n"
                 f"closed form:       w = {w_star:.3f}\n\n"
                 f"back on the original price scale:\n"
                 f"{run['w'][-1]:.3f} / {scale:.3f} = {run['w'][-1] / scale:.2f}  ✓")

    titled(fig, "4 · How the parameters are actually found",
           "w ← w − η·∂MSE/∂w   and   b ← b − η·∂MSE/∂b, repeated. Big steps while the slope is steep, tiny ones near the bottom.")
    return fig


@reg.register(
    "05", "learning-rate", "The learning rate decides everything",
    "One knob, three completely different outcomes. Too small and you burn compute crawling "
    "toward the answer. Too large and each step overshoots the valley, bouncing to ever-worse "
    "parameters until the loss blows up. The useful range is wide but not unlimited — and this "
    "same knob reappears in XGBoost as the shrinkage applied to every tree.",
    REGRESSION)
def fig_learning_rate():
    x_raw, y = TOY_PRICE, TOY_DEMAND
    x = (x_raw - x_raw.mean()) / x_raw.std()
    settings = [(0.004, "too small (η = 0.004)", "still nowhere near after 60 steps"),
                (0.12, "just right (η = 0.12)", "converges in a few dozen steps"),
                (1.02, "too large (η = 1.02)", "overshoots, then diverges")]

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.6))
    w_grid = np.linspace(-24, 16, 150)
    b_grid = np.linspace(20, 130, 150)
    W, B = np.meshgrid(w_grid, b_grid)
    Z = loss_surface(x, y, w_grid, b_grid)
    levels = np.geomspace(Z.min() + 1, Z.max(), 18)
    w_star, b_star = ols_closed_form(x, y)

    runs = []
    for ax, (lr, name, note) in zip(axes, settings):
        run = gradient_descent(x, y, lr=lr, steps=60, w0=6.0, b0=40.0)
        runs.append((lr, name, run))
        ax.contour(W, B, Z, levels=levels, colors=[MUTED], linewidths=0.6, alpha=0.65)
        inside = (np.abs(run["w"]) < 60) & (np.abs(run["b"]) < 400)
        ax.plot(run["w"][inside], run["b"][inside], "-o", color=ALT, lw=1.5,
                ms=3.0, alpha=0.9, zorder=3)
        ax.scatter([run["w"][0]], [run["b"][0]], s=80, color=ACCENT,
                   edgecolor="white", lw=1.1, zorder=5)
        ax.scatter([w_star], [b_star], marker="*", s=210, color=MODEL,
                   edgecolor="white", lw=1.0, zorder=5)
        ax.set_xlim(w_grid.min(), w_grid.max())
        ax.set_ylim(b_grid.min(), b_grid.max())
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("w")
        ax.text(0.5, -0.19, note, transform=ax.transAxes, ha="center",
                fontsize=9.2, color="#63707E")
        ax.grid(False)
    axes[0].set_ylabel("b")

    ax = axes[3]
    for (lr, name, run), colour in zip(runs, [MUTED, GOOD, MODEL]):
        finite = np.isfinite(run["loss"]) & (run["loss"] < 1e12)
        ax.plot(np.arange(len(run["loss"]))[finite], run["loss"][finite],
                color=colour, lw=2.2, label=f"η = {lr}")
    ax.set_yscale("log")
    ax.set_xlabel("iteration"); ax.set_ylabel("MSE (log scale)")
    ax.set_title("The same three runs, as loss curves", fontsize=11)
    ax.legend(loc="center")

    titled(fig, "5 · The learning rate η — the one knob that breaks everything",
           "A rising loss curve is the unmistakable signature of a learning rate that is too large.")
    return fig


@reg.register(
    "06", "feature-scaling", "Why features get standardised",
    "Left: on raw prices the loss valley is a long thin ravine, because w and b are measured "
    "in wildly different units. Gradient descent zigzags across the ravine instead of running "
    "along it, and you must use a tiny step size to avoid diverging. Right: after centring and "
    "scaling, the valley is nearly circular and the path goes almost straight to the bottom.",
    REGRESSION)
def fig_feature_scaling():
    y = TOY_DEMAND
    raw = TOY_PRICE
    scaled = (TOY_PRICE - TOY_PRICE.mean()) / TOY_PRICE.std()

    def curvature(x: np.ndarray) -> float:
        """Condition number of the MSE Hessian = how stretched the valley is.

        H = (2/n) [[Sum x^2, Sum x], [Sum x, n]]. The ratio of its eigenvalues
        controls both the maximum safe step size and the amount of zigzagging.
        """
        n = len(x)
        H = (2.0 / n) * np.array([[np.sum(x ** 2), np.sum(x)], [np.sum(x), n]])
        eig = np.linalg.eigvalsh(H)
        return float(eig[-1] / eig[0])

    variants = [
        (raw, 0.0047, 60, "raw price (10 … 18)", (-14, 4), (100, 190), (-1.0, 175.0), "#8A3B3B"),
        (scaled, 0.12, 60, "standardised price (mean 0, sd 1)", (-24, 12), (30, 130),
         (6.0, 40.0), "#1F6F63"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.8),
                             gridspec_kw={"height_ratios": [1.35, 1.0]})
    for col, (x, lr, steps, name, wlim, blim, start, colour) in enumerate(variants):
        run = gradient_descent(x, y, lr=lr, steps=steps, w0=start[0], b0=start[1])
        w_star, b_star = ols_closed_form(x, y)

        ax = axes[0, col]
        w_grid = np.linspace(*wlim, 170)
        b_grid = np.linspace(*blim, 170)
        W, B = np.meshgrid(w_grid, b_grid)
        Z = loss_surface(x, y, w_grid, b_grid)
        ax.contour(W, B, Z, levels=np.geomspace(Z.min() + 1, Z.max(), 24),
                   colors=[MUTED], linewidths=0.65, alpha=0.7)
        ax.plot(run["w"], run["b"], "-o", color=ALT, lw=1.4, ms=3.2, alpha=0.9, zorder=3)
        ax.scatter([run["w"][0]], [run["b"][0]], s=85, color=ACCENT,
                   edgecolor="white", lw=1.1, zorder=5)
        ax.scatter([w_star], [b_star], marker="*", s=260, color=MODEL,
                   edgecolor="white", lw=1.1, zorder=5)
        ax.set_xlim(*wlim); ax.set_ylim(*blim)
        ax.set_title(f"{name}\nvalley stretch (condition number) = {curvature(x):,.0f}",
                     fontsize=11)
        ax.set_xlabel("w"); ax.set_ylabel("b")
        ax.grid(False)

        # The oscillation above retraces the same line, so it hides in the contour
        # plot. Plotting the parameter against the iteration number exposes it.
        ax = axes[1, col]
        ax.plot(run["w"], "-o", color=ALT, lw=1.5, ms=3.2)
        ax.axhline(w_star, color=MODEL, ls="--", lw=1.6,
                   label=f"true optimum w* = {w_star:.2f}")
        ax.set_xlabel("iteration"); ax.set_ylabel("w")
        ax.set_title(f"largest safe η ≈ {lr}  →  MSE after {steps} steps = {run['loss'][-1]:,.1f}",
                     fontsize=10.5, color=colour)
        ax.legend(loc="lower right" if col == 0 else "upper right", fontsize=9)

    annotate_box(axes[0, 0], 0.035, 0.44,
                 "60 steps in, still stranded up there:\nthe run bounces sideways across the\nravine and barely moves along it",
                 color="#8A3B3B")
    annotate_box(axes[0, 1], 0.035, 0.97,
                 "almost circular valley —\nevery step points at the minimum",
                 color="#1F6F63")
    annotate_box(axes[1, 0], 0.30, 0.40,
                 "the zigzag, made visible: w overshoots every step.\nIt settles near −6.5, not −4.5 — that is the best w\ncan do while b is still stuck at 174 instead of 145.",
                 color="#8A3B3B", fontsize=9.6)
    annotate_box(axes[1, 1], 0.36, 0.72,
                 "smooth, monotone approach —\nand it reaches the true optimum",
                 color="#1F6F63", fontsize=9.6)

    titled(fig, "6 · Feature scaling is an optimisation fix, not a statistics ritual",
           "Identical data, identical loss, identical answer — only the units changed. Scaling reshapes the valley the optimiser has to walk down.")
    return fig

    titled(fig, "6 · Feature scaling is an optimisation fix, not a statistics ritual",
           "Identical data, identical loss, identical answer — only the units changed. Scaling reshapes the valley the optimiser has to walk down.")
    return fig


@reg.register(
    "07", "polynomial-fit", "Underfitting and overfitting",
    "The same data fitted with a straight line, a gentle curve and a wildly flexible curve. "
    "Degree 1 is too rigid to follow the real shape (high bias). Degree 15 threads every single "
    "training point, including the noise, and goes haywire between them (high variance). Degree 4 "
    "recovers the underlying truth. Note that the *training* error keeps falling the whole time — "
    "which is precisely why training error must never be your judge.",
    REGRESSION)
def fig_polynomial():
    price, demand, truth = curved_demand(n=26, noise=7.5, seed=3)
    u = (price - 25) / 15.0  # rescale to [-1, 1] so x^15 stays numerically sane
    grid = np.linspace(price.min(), price.max(), 400)
    grid_u = (grid - 25) / 15.0

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharey=True)
    for ax, degree, verdict in zip(axes, [1, 4, 15],
                                   ["underfitting — too rigid",
                                    "about right",
                                    "overfitting — memorising noise"]):
        design = np.column_stack([u ** d for d in range(1, degree + 1)])
        coef = ols_normal_equation(design, demand)
        grid_design = np.column_stack([grid_u ** d for d in range(1, degree + 1)])
        fitted = coef[0] + grid_design @ coef[1:]
        train_pred = coef[0] + design @ coef[1:]

        ax.plot(grid, true_curve(grid), color=MUTED, lw=2.0, ls="--",
                label="true underlying curve", zorder=2)
        ax.plot(grid, fitted, color=MODEL, lw=2.6, label=f"degree {degree} fit", zorder=3)
        ax.scatter(price, demand, s=48, color=INK, alpha=0.85, zorder=4, label="training rows")
        ax.set_ylim(-15, 130)
        ax.set_xlabel("price")
        ax.set_title(f"degree {degree} — {verdict}\ntrain MSE = {mse(demand, train_pred):.1f}",
                     fontsize=11)
    axes[0].set_ylabel("weekly demand")
    axes[0].legend(loc="lower left", fontsize=9)

    titled(fig, "7 · The same data, three levels of flexibility",
           "Polynomial regression is still linear regression — you just invent price², price³ … columns and fit as usual.")
    return fig


@reg.register(
    "08", "bias-variance", "The bias–variance trade-off",
    "Sweep model flexibility and track two errors. Training error falls forever — more "
    "flexibility can always fit the data you already have. Test error falls, bottoms out, then "
    "climbs as the model starts memorising noise it will never see again. The gap between the "
    "two curves *is* overfitting, and the bottom of the test curve is the model you want.",
    REGRESSION)
def fig_bias_variance():
    rng = np.random.default_rng(11)
    price, demand, _ = curved_demand(n=60, noise=8.0, seed=5)
    u = (price - 25) / 15.0
    degrees = np.arange(1, 16)

    # A single train/test split is noisy enough to hide the U-shape, and reading
    # a trend off one noisy split is exactly the mistake this figure warns about.
    # So average over many random splits: this is the *expected* error curve.
    n_repeats = 60
    train_err = np.zeros((n_repeats, len(degrees)))
    test_err = np.zeros((n_repeats, len(degrees)))
    for r in range(n_repeats):
        order = rng.permutation(len(price))
        train_idx, test_idx = order[:34], order[34:]
        for c, degree in enumerate(degrees):
            design = np.column_stack([u ** d for d in range(1, degree + 1)])
            coef = ols_normal_equation(design[train_idx], demand[train_idx])
            pred = coef[0] + design @ coef[1:]
            train_err[r, c] = mse(demand[train_idx], pred[train_idx])
            test_err[r, c] = mse(demand[test_idx], pred[test_idx])
    train_mean, test_mean = train_err.mean(axis=0), test_err.mean(axis=0)
    best = int(degrees[np.argmin(test_mean)])

    fig, ax = plt.subplots(figsize=(11.0, 6.2))
    ax.fill_between(degrees, train_mean, test_mean, color=ERROR, alpha=0.10,
                    label="the overfitting gap")
    ax.plot(degrees, train_mean, "-o", color=ALT, lw=2.4, ms=5, label="training error")
    ax.plot(degrees, test_mean, "-o", color=MODEL, lw=2.4, ms=5,
            label="test error (unseen rows)")
    ax.axvline(best, color=GOOD, ls="--", lw=1.8)
    ax.scatter([best], [test_mean[best - 1]], s=220, marker="*", color=GOOD,
               edgecolor="white", lw=1.0, zorder=6)
    ax.annotate(f"sweet spot: degree {best}", xy=(best, test_mean[best - 1]),
                xytext=(best + 0.6, test_mean[best - 1] * 2.6), color="#1F6F63",
                fontweight="semibold", fontsize=10.5,
                arrowprops=dict(arrowstyle="->", color=GOOD, lw=1.5))
    ax.set_yscale("log")
    ax.set_xlabel("model flexibility  (polynomial degree)")
    ax.set_ylabel(f"MSE (log scale), averaged over {n_repeats} random splits")
    ax.set_xticks(degrees)
    ax.legend(loc="lower left")

    ax.annotate("too rigid\n(high bias)", xy=(1.6, train_mean[0] * 0.75),
                fontsize=10, color="#63707E", ha="left")
    ax.annotate("too flexible\n(high variance)", xy=(13.2, test_mean[-1] * 1.45),
                fontsize=10, color="#63707E", ha="center")

    titled(fig, "8 · Why you cannot judge a model on the data it learned from",
           "Training error is a measure of memory. Test error is a measure of understanding. Only one of them keeps falling forever.")
    return fig


@reg.register(
    "09", "ridge-vs-lasso", "Ridge shrinks, Lasso selects",
    "Six features, only two of which genuinely drive demand; the rest are noise. As the penalty "
    "λ grows, Ridge pulls every coefficient smoothly toward zero but never quite lets go of any of "
    "them. Lasso drives the useless ones to exactly zero, one after another, and keeps the real "
    "signal longest — that is automatic feature selection falling out of the penalty's shape.",
    REGRESSION)
def fig_ridge_lasso():
    rng = np.random.default_rng(4)
    n = 90
    price = rng.uniform(10, 40, n)
    rating = rng.uniform(3.5, 5.0, n)
    noise_features = rng.normal(size=(n, 3))
    X = np.column_stack([price, rating, noise_features])
    names = ["price (real)", "rating (real)", "noise A", "noise B", "noise C"]
    y = 120 - 2.4 * price + 14 * rating + rng.normal(0, 4, n)

    lambdas = np.geomspace(0.02, 400, 45)
    ridge_path = np.array([ridge_fit(X, y, lam) for lam in lambdas])
    lasso_lams = np.geomspace(0.01, 30, 45)
    lasso_path = np.array([lasso_fit(X, y, lam) for lam in lasso_lams])

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.9), sharey=True)
    for ax, path, lams, name in [
        (axes[0], ridge_path, lambdas, "Ridge  (L2 penalty on Σw²)"),
        (axes[1], lasso_path, lasso_lams, "Lasso  (L1 penalty on Σ|w|)"),
    ]:
        for j, label in enumerate(names):
            real = "real" in label
            ax.plot(lams, path[:, j], lw=2.6 if real else 1.8,
                    color=[INK, GOOD, "#8A939C", "#A8B0B8", "#C2C9CF"][j],
                    ls="-" if real else (0, (4, 2)), label=label, zorder=4 if real else 2)
        ax.axhline(0, color="#5A6672", lw=1.1)
        ax.set_xscale("log")
        ax.set_xlabel("penalty strength λ  (log scale)")
        ax.set_title(name, fontsize=11.5)

    axes[0].set_ylabel("coefficient value (standardised features)")
    axes[0].legend(loc="lower right", fontsize=9)
    annotate_box(axes[0], 0.04, 0.30,
                 "every coefficient shrinks toward 0,\nbut none ever arrives:\nthe noise features are still in the model",
                 color="#22314E")

    noise_dead = np.argmax(np.all(np.isclose(lasso_path[:, 2:], 0), axis=1))
    annotate_box(axes[1], 0.04, 0.30,
                 f"all three noise coefficients are\nexactly 0 by λ ≈ {lasso_lams[noise_dead]:.2f},\n"
                 "while the two real signals survive\n→ the model selected its own features",
                 color="#1F6F63")

    titled(fig, "9 · Regularisation — buying a little bias to kill a lot of variance",
           "Both add a penalty on coefficient size to the loss. The shape of the penalty decides whether weights shrink or vanish.")
    return fig


@reg.register(
    "10", "l1-l2-geometry", "Why L1 produces exact zeros",
    "The ellipses are contours of squared error, centred on the unpenalised solution. The "
    "penalty confines the answer to the shaded region, so the fit lands where the smallest "
    "affordable ellipse first touches that region. L2's region is a circle — contact almost "
    "always happens off-axis, so both weights stay non-zero. L1's region is a diamond with "
    "corners *on* the axes, and corners are what an expanding ellipse hits first.",
    REGRESSION)
def fig_l1_l2_geometry():
    # Two strongly correlated features, so the error contours form a long
    # diagonal ellipse - the situation where the two penalties disagree most.
    rng = np.random.default_rng(2)
    n = 60
    x1 = rng.normal(0, 1, n)
    x2 = 0.9 * x1 + rng.normal(0, np.sqrt(1 - 0.81), n)
    X = np.column_stack([x1, x2])
    y = 2.5 * x1 + 1.2 * x2 + rng.normal(0, 0.8, n)

    Xs, _, _ = standardize(X)
    yc = y - y.mean()
    ols = np.linalg.lstsq(Xs, yc, rcond=None)[0]

    # Find the smallest penalty that kills w2 outright, then use the resulting
    # |w1| + |w2| as the budget for BOTH panels, so the only difference between
    # them is the shape of the region - not how tight it is.
    solution = None
    for lam in np.linspace(0.1, 6.0, 400):
        candidate = lasso_fit(X, y, lam, iters=400)
        if abs(candidate[1]) < 1e-9:
            solution, used_lambda = candidate, lam
            break
    budget = float(np.abs(solution).sum())

    # grid deliberately built to contain exactly 0.0, otherwise an "exact zero"
    # could never actually be found
    grid = np.round(np.arange(-0.6, 3.001, 0.01), 4)
    G1, G2 = np.meshgrid(grid, grid)
    resid = yc[:, None, None] - (Xs[:, 0][:, None, None] * G1 + Xs[:, 1][:, None, None] * G2)
    rss = np.sum(resid ** 2, axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.4))
    for ax, kind in zip(axes, ["L2 (Ridge)", "L1 (Lasso)"]):
        if kind.startswith("L2"):
            feasible = (G1 ** 2 + G2 ** 2) <= budget ** 2
            ax.add_patch(Circle((0, 0), budget, facecolor=ALT, alpha=0.16,
                                edgecolor=ALT, lw=2.2, zorder=2))
        else:
            feasible = (np.abs(G1) + np.abs(G2)) <= budget
            ax.add_patch(Polygon([(budget, 0), (0, budget), (-budget, 0), (0, -budget)],
                                 facecolor=GOOD, alpha=0.16, edgecolor=GOOD, lw=2.2, zorder=2))

        flat = int(np.argmin(np.where(feasible, rss, np.inf)))
        best = (float(G1.ravel()[flat]), float(G2.ravel()[flat]))

        ax.contour(G1, G2, rss, levels=np.geomspace(rss.min() + 0.5, rss.max(), 18),
                   colors=[MUTED], linewidths=0.85, alpha=0.8, zorder=1)
        ax.scatter(*ols, s=95, color=MODEL, zorder=5)
        ax.annotate("unpenalised\nleast squares", xy=ols, xytext=(ols[0] - 0.5, ols[1] + 0.55),
                    fontsize=9.5, color=MODEL, ha="center",
                    arrowprops=dict(arrowstyle="->", color=MODEL, lw=1.2))
        ax.scatter(*best, s=260, marker="*", color=ACCENT, edgecolor="#8A5A00",
                   lw=0.9, zorder=6)
        note = "w₂ = 0 exactly → feature dropped" if abs(best[1]) < 1e-9 else "both weights survive"
        ax.annotate(f"w = ({best[0]:.2f}, {best[1]:.2f})\n{note}",
                    xy=best, xytext=(best[0] + 0.30, best[1] - 0.95), fontsize=9.5,
                    color="#8A5A00", fontweight="semibold",
                    arrowprops=dict(arrowstyle="->", color=ACCENT, lw=1.3))
        ax.axhline(0, color="#8A939C", lw=1.1, zorder=3)
        ax.axvline(0, color="#8A939C", lw=1.1, zorder=3)
        ax.set_xlim(-0.6, 3.0); ax.set_ylim(-1.6, 2.4)
        ax.set_box_aspect(1)
        ax.set_xlabel("w₁"); ax.set_ylabel("w₂")
        ax.set_title(f"{kind} — same budget |w| ≤ {budget:.2f}", fontsize=11.5)
        ax.grid(False)

    titled(fig, "10 · The geometry behind 'Lasso does feature selection'",
           "The fit lands where the smallest affordable error contour first touches the region. A diamond has corners on the axes; a circle does not.")
    return fig


# ==========================================================================
# PART 2 — TREES AND GRADIENT BOOSTING
# ==========================================================================
# One shared dataset for the whole section, so every figure is talking about
# the same 70 weeks. The shape is deliberately non-linear: flat while cheap,
# a sharp fall through the middle, then a floor of loyal buyers. A straight
# line cannot express it; a sum of shallow trees can.
BOOST_X, BOOST_Y, BOOST_TRUTH = curved_demand(n=70, noise=5.0, seed=3)
PRICE_GRID = np.linspace(BOOST_X.min(), BOOST_X.max(), 400)


def _thresholds_of(node, found=None):
    """Every split point in a tree, so the step edges can be drawn on the data."""
    found = [] if found is None else found
    if not node.is_leaf:
        found.append(node.threshold)
        _thresholds_of(node.left, found)
        _thresholds_of(node.right, found)
    return found


@reg.register(
    "11", "single-regression-tree", "What a single regression tree actually is",
    "A tree is a set of yes/no questions ending in a number. Each leaf predicts the mean of "
    "the training rows that land in it, so the model is a step function: it can bend anywhere, "
    "but it is flat everywhere between bends. Note it never extrapolates — outside the "
    "training range it just repeats its edge leaf.", BOOSTING)
def fig_single_tree():
    tree = RegressionTree(max_depth=2, min_samples=5).fit(BOOST_X, BOOST_Y)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.9),
                             gridspec_kw={"width_ratios": [1.1, 1.0]})

    draw_tree(axes[0], from_regression_tree(tree.root, ["price"], value_fmt="{:.1f}"),
              fontsize=9.8)
    axes[0].set_title("the rules it learned", fontsize=11.5)

    ax = axes[1]
    ax.scatter(BOOST_X, BOOST_Y, s=28, color=INK, alpha=0.5, label="observed weeks")
    ax.plot(PRICE_GRID, tree.predict(PRICE_GRID), color=MODEL, lw=2.6,
            label="tree prediction")
    for t in _thresholds_of(tree.root):
        ax.axvline(t, color=ACCENT, ls="--", lw=1.3, alpha=0.9)
    ax.set_xlabel("price"); ax.set_ylabel("units sold")
    ax.set_title("the same tree, drawn as a prediction", fontsize=11.5)
    ax.legend(loc="upper right")
    annotate_box(ax, 0.04, 0.30,
                 "amber lines = the split points.\nBetween them the prediction cannot\nchange at all — that flatness is the\nsingle weakness boosting fixes.")

    titled(fig, "11 · A regression tree is a step function",
           f"Depth 2 → {tree.n_leaves()} leaves. Every leaf value below is literally the average demand of the weeks that reach it.")
    return fig


@reg.register(
    "12", "how-a-split-is-chosen", "How the tree picks a split",
    "No cleverness, just search. For every candidate cut point the tree computes the squared "
    "error left over if it split there, and keeps the cheapest. The bottom curve *is* that "
    "search; its minimum is the split the tree actually made.", BOOSTING)
def fig_split_choice():
    thresholds, scores = split_scan(BOOST_X, BOOST_Y)
    best = int(np.argmin(scores))
    cut = thresholds[best]
    parent_sse = float(np.sum((BOOST_Y - BOOST_Y.mean()) ** 2))

    left, right = BOOST_X <= cut, BOOST_X > cut
    fig, axes = plt.subplots(2, 1, figsize=(11.4, 8.0), sharex=True,
                             gridspec_kw={"height_ratios": [1.0, 1.0]})

    ax = axes[0]
    ax.scatter(BOOST_X[left], BOOST_Y[left], s=30, color=ALT, alpha=0.75, label="goes left")
    ax.scatter(BOOST_X[right], BOOST_Y[right], s=30, color=MODEL, alpha=0.75, label="goes right")
    ax.hlines(BOOST_Y[left].mean(), BOOST_X.min(), cut, color=ALT, lw=3)
    ax.hlines(BOOST_Y[right].mean(), cut, BOOST_X.max(), color=MODEL, lw=3)
    ax.axhline(BOOST_Y.mean(), color=MUTED, ls=":", lw=1.8, label="one mean for everyone")
    ax.axvline(cut, color=ACCENT, ls="--", lw=1.8)
    ax.set_ylabel("units sold")
    ax.set_title(f"the winning split: price ≤ {cut:.2f}", fontsize=11.5)
    ax.legend(loc="upper right", fontsize=9.5)

    ax = axes[1]
    ax.plot(thresholds, scores, color=INK, lw=2.0)
    ax.axhline(parent_sse, color=MUTED, ls=":", lw=1.8,
               label=f"no split at all: SSE = {parent_sse:,.0f}")
    ax.scatter([cut], [scores[best]], s=170, color=ACCENT, edgecolor="white",
               lw=1.3, zorder=5)
    ax.set_xlabel("candidate split point (price)")
    ax.set_ylabel("SSE left after splitting here")
    ax.legend(loc="upper right", fontsize=9.5)
    drop = 100 * (1 - scores[best] / parent_sse)
    annotate_box(ax, 0.28, 0.93,
                 f"best cut removes {drop:.0f}% of the squared error.\n"
                 f"SSE = Σ(y − leaf mean)² summed over both sides —\n"
                 f"so 'best split' means 'the two groups that are\nmost internally consistent'.")

    titled(fig, "12 · A split is chosen by brute-force search",
           "Try every cut point, score each one by the error it leaves behind, take the minimum. Repeat inside each half. That is the entire tree-growing algorithm.")
    return fig


@reg.register(
    "13", "tree-depth-overfitting", "Depth is the tree's capacity dial",
    "One deep tree can drive training error to almost zero by carving out a leaf for nearly "
    "every point — and that is exactly when test error starts climbing. The gap between the "
    "two numbers under each panel is the memorisation.", BOOSTING)
def fig_tree_depth():
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(BOOST_X))
    tr, te = idx[:45], idx[45:]

    fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4), sharey=True)
    for ax, depth in zip(axes, (1, 3, 10)):
        tree = RegressionTree(max_depth=depth, min_samples=1).fit(BOOST_X[tr], BOOST_Y[tr])
        train_mse = mse(BOOST_Y[tr], tree.predict(BOOST_X[tr]))
        test_mse = mse(BOOST_Y[te], tree.predict(BOOST_X[te]))

        ax.scatter(BOOST_X[tr], BOOST_Y[tr], s=26, color=INK, alpha=0.5, label="train")
        ax.scatter(BOOST_X[te], BOOST_Y[te], s=34, marker="^", color=ACCENT,
                   edgecolor="#8A6420", lw=0.5, label="held-out test")
        ax.plot(PRICE_GRID, true_curve(PRICE_GRID), color=MUTED, ls=":", lw=1.8,
                label="the truth")
        ax.plot(PRICE_GRID, tree.predict(PRICE_GRID), color=MODEL, lw=2.3, label="tree")
        ax.set_xlabel("price")
        ax.set_title(f"max_depth = {depth}  ({tree.n_leaves()} leaves)", fontsize=11.5)
        ax.text(0.5, -0.185,
                f"train MSE {train_mse:5.1f}   test MSE {test_mse:5.1f}",
                transform=ax.transAxes, ha="center", fontsize=10,
                color=MODEL if test_mse > 40 else GOOD, fontweight="semibold")
    axes[0].set_ylabel("units sold")
    axes[0].legend(loc="lower left", fontsize=8.8)
    annotate_box(axes[0], 0.30, 0.98, "too coarse:\nunderfits", color="#8A3B3B", fontsize=9.5)
    annotate_box(axes[1], 0.30, 0.98, "about right", color="#1F6F63", fontsize=9.5)
    annotate_box(axes[2], 0.22, 0.98, "chasing the noise:\noverfits", color="#8A3B3B", fontsize=9.5)

    titled(fig, "13 · Deeper is not better — it is just more memorised",
           "Same 45 training weeks in all three panels. Only the depth limit changes.")
    return fig


@reg.register(
    "14", "bagging-vs-boosting", "Bagging vs boosting",
    "Both build many trees, but for opposite reasons. Bagging grows deep, independent trees "
    "and averages away their variance. Boosting grows shallow, dependent trees, each one "
    "correcting the last, and grinds down bias. Bagging can be run in parallel; boosting "
    "cannot, because tree m needs tree m−1's mistakes.", BOOSTING)
def fig_bagging_vs_boosting():
    rng = np.random.default_rng(11)
    idx = rng.permutation(len(BOOST_X))
    tr, te = idx[:45], idx[45:]
    Xtr, ytr, Xte, yte = BOOST_X[tr], BOOST_Y[tr], BOOST_X[te], BOOST_Y[te]

    bag = BaggedTrees(n_trees=25, max_depth=6, seed=4).fit(Xtr, ytr)
    boost = GradientBoostedTrees(n_rounds=25, learning_rate=0.35, max_depth=2).fit(Xtr, ytr)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 5.4))

    ax = axes[0]
    for tree in bag.trees[:8]:
        ax.plot(PRICE_GRID, tree.predict(PRICE_GRID), color=ALT, lw=0.9, alpha=0.35)
    ax.scatter(Xtr, ytr, s=24, color=INK, alpha=0.45)
    ax.plot(PRICE_GRID, bag.predict(PRICE_GRID), color=MODEL, lw=2.6)
    ax.set_title("bagging: 25 deep trees, averaged", fontsize=11.5)
    ax.set_xlabel("price"); ax.set_ylabel("units sold")
    annotate_box(ax, 0.03, 0.30,
                 "thin blue = individual trees.\nEach is wild on its own; the average\n(red) is calm. Variance reduction.",
                 fontsize=9.3)

    ax = axes[1]
    for m, alpha in ((1, 0.30), (3, 0.45), (8, 0.6), (25, 1.0)):
        ax.plot(PRICE_GRID, boost.predict(PRICE_GRID, n_rounds=m),
                color=MODEL if m == 25 else ALT,
                lw=2.6 if m == 25 else 1.2, alpha=alpha,
                label=f"after {m} tree{'s' if m > 1 else ''}")
    ax.scatter(Xtr, ytr, s=24, color=INK, alpha=0.45)
    ax.set_title("boosting: 25 shallow trees, added up", fontsize=11.5)
    ax.set_xlabel("price")
    ax.legend(loc="upper right", fontsize=8.6)
    annotate_box(ax, 0.03, 0.30,
                 "each tree is nearly useless alone\n(depth 2). The curve creeps toward\nthe data. Bias reduction.",
                 fontsize=9.3)

    ax = axes[2]
    bag_curve = [mse(yte, np.mean([t.predict(Xte) for t in bag.trees[:k]], axis=0))
                 for k in range(1, 26)]
    boost_curve = [mse(yte, boost.predict(Xte, n_rounds=k)) for k in range(1, 26)]
    ax.plot(range(1, 26), bag_curve, color=ALT, lw=2.2, marker="o", ms=3.4, label="bagging")
    ax.plot(range(1, 26), boost_curve, color=MODEL, lw=2.2, marker="o", ms=3.4, label="boosting")
    ax.set_xlabel("number of trees"); ax.set_ylabel("test MSE")
    ax.set_title("held-out error as trees are added", fontsize=11.5)
    ax.legend(loc="upper right")
    annotate_box(ax, 0.30, 0.62,
                 "bagging is already near its floor\nafter a handful of trees.\nBoosting starts worse and keeps\nimproving — that is the trade.",
                 fontsize=9.3)

    titled(fig, "14 · Two different jobs: averaging noise out vs correcting error down",
           "Random Forest is the bagging idea; XGBoost is the boosting idea. Knowing which error they attack tells you which knobs matter.")
    return fig


@reg.register(
    "15", "boosting-loop", "The boosting loop, round by round",
    "Read left to right, then down. Column 1 is what the model currently believes; column 2 "
    "is what it is still getting wrong; column 3 is the little tree fitted to *those mistakes*, "
    "which then gets added (scaled by the learning rate) to produce the next row's column 1. "
    "That is the whole algorithm — there is nothing else in it.", BOOSTING)
def fig_boosting_loop():
    lr, rounds = 0.5, 4
    model = GradientBoostedTrees(n_rounds=rounds, learning_rate=lr, max_depth=2,
                                 min_samples=5).fit(BOOST_X, BOOST_Y)
    hist = model.history

    fig, axes = plt.subplots(rounds, 3, figsize=(13.4, 13.6), sharex=True)
    for m in range(rounds):
        current = model.predict(PRICE_GRID, n_rounds=m)
        residual = hist.residuals[m]
        tree = hist.trees[m]

        ax = axes[m, 0]
        ax.scatter(BOOST_X, BOOST_Y, s=20, color=INK, alpha=0.42)
        ax.plot(PRICE_GRID, current, color=MODEL, lw=2.4)
        ax.set_ylabel(f"round {m + 1}\n\nunits sold", fontsize=10.5)
        label = "F₀ = mean(y)" if m == 0 else f"F{'₁₂₃'[m - 1]} so far"
        ax.set_title(f"model before this round  ·  {label}\nMSE = {hist.train_mse[m]:,.1f}",
                     fontsize=10.2)

        ax = axes[m, 1]
        ax.axhline(0, color=MUTED, lw=1.4)
        ax.vlines(BOOST_X, 0, residual, color=ERROR, lw=1.0, alpha=0.55)
        ax.scatter(BOOST_X, residual, s=20, color=ERROR, alpha=0.85)
        ax.set_title(f"what is still wrong  ·  r = y − F{'₀₁₂₃'[m]}\n"
                     f"(for squared error this is the negative gradient)", fontsize=10.2)

        ax = axes[m, 2]
        ax.axhline(0, color=MUTED, lw=1.4)
        ax.scatter(BOOST_X, residual, s=18, color=ERROR, alpha=0.4)
        ax.plot(PRICE_GRID, tree.predict(PRICE_GRID), color=ALT, lw=2.2,
                label="tree fitted to r")
        ax.plot(PRICE_GRID, lr * tree.predict(PRICE_GRID), color=GOOD, lw=2.4, ls="--",
                label=f"× η={lr} → what gets added")
        ax.set_title(f"the new tree  ·  F{'₁₂₃₄'[m]} = F{'₀₁₂₃'[m]} + η·tree\n"
                     f"MSE drops to {hist.train_mse[m + 1]:,.1f}", fontsize=10.2)
        if m == 0:
            ax.legend(loc="lower left", fontsize=8.6)

    for ax in axes[-1]:
        ax.set_xlabel("price")

    titled(fig, "15 · Gradient boosting is a loop over your own mistakes",
           "F₀ = mean(y);  repeat: r = y − F, fit a shallow tree to r, F ← F + η·tree. "
           "Each row below is one turn of that loop, on real numbers.", top=0.945)
    return fig


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else None
    print(f"Writing concept figures to {OUT_DIR}")
    entries = reg.run(OUT_DIR, only=only)
    if not only:
        write_manifest(MANIFEST, entries)
    print(f"Done: {len(entries)} figure(s).")
