"""Minimal from-scratch implementations of every model the figures illustrate.

Deliberately written with numpy only - no scikit-learn, no xgboost. Two reasons:

1. The figures must show the *mechanism*. If a library hides the boosting loop,
   you cannot draw the boosting loop.
2. It keeps this half of the visualisation runnable anywhere numpy + matplotlib
   exist.

These are teaching implementations: correct and readable, not optimised. The
real project uses scikit-learn and XGBoost (see project_figures.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# --------------------------------------------------------------------------
# the toy dataset from the study notes (project_understanding.md, Part 1 s2)
# closed-form answer: w = -4.5, b = 145, R^2 = 0.993
# --------------------------------------------------------------------------
TOY_PRICE = np.array([10.0, 12.0, 14.0, 16.0, 18.0])
TOY_DEMAND = np.array([100.0, 90.0, 84.0, 72.0, 64.0])


def mse(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Mean squared error - the loss every model on this page minimises."""
    return float(np.mean((y - y_hat) ** 2))


def r2_score(y: np.ndarray, y_hat: np.ndarray) -> float:
    """Fraction of variance explained. 1.0 = perfect, 0.0 = no better than the mean."""
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot


# --------------------------------------------------------------------------
# linear regression
# --------------------------------------------------------------------------
def ols_closed_form(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least squares for one feature, via the covariance formula.

        w = sum((x - xbar)(y - ybar)) / sum((x - xbar)^2)
        b = ybar - w * xbar

    This is the same answer the normal equation gives; for a single feature the
    matrix algebra collapses to these two lines.
    """
    x_c, y_c = x - x.mean(), y - y.mean()
    w = float(np.sum(x_c * y_c) / np.sum(x_c ** 2))
    return w, float(y.mean() - w * x.mean())


def ols_normal_equation(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """w* = (X^T X)^-1 X^T y, with a bias column prepended.

    Uses lstsq rather than an explicit inverse: numerically safer and it still
    solves the same normal equations.
    """
    design = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return coef  # coef[0] = intercept, coef[1:] = weights


def gradient_descent(x: np.ndarray, y: np.ndarray, *, lr: float, steps: int,
                     w0: float = 0.0, b0: float = 0.0) -> dict:
    """Fit y = w*x + b by repeatedly stepping downhill on the MSE surface.

    Gradients of MSE = mean((w*x + b - y)^2):
        dMSE/dw = (2/n) * sum((y_hat - y) * x)
        dMSE/db = (2/n) * sum(y_hat - y)

    Returns the whole trajectory so the figures can draw the path, not just the
    destination.
    """
    n = len(x)
    w, b = w0, b0
    path_w, path_b, losses = [w], [b], [mse(y, w * x + b)]
    for _ in range(steps):
        y_hat = w * x + b
        grad_w = (2.0 / n) * np.sum((y_hat - y) * x)
        grad_b = (2.0 / n) * np.sum(y_hat - y)
        w -= lr * grad_w
        b -= lr * grad_b
        path_w.append(w)
        path_b.append(b)
        losses.append(mse(y, w * x + b))
        if not np.isfinite(w) or abs(w) > 1e12:  # diverged: stop early
            break
    return {"w": np.array(path_w), "b": np.array(path_b), "loss": np.array(losses)}


def loss_surface(x: np.ndarray, y: np.ndarray, w_grid: np.ndarray,
                 b_grid: np.ndarray) -> np.ndarray:
    """MSE evaluated on a (w, b) mesh - the bowl that gradient descent rolls down."""
    W, B = np.meshgrid(w_grid, b_grid)
    preds = W[..., None] * x[None, None, :] + B[..., None]
    return np.mean((preds - y[None, None, :]) ** 2, axis=-1)


def standardize(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centre and scale columns. Required before any penalised fit, because a
    penalty on raw weights would punish features measured in small units."""
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    return (X - mu) / sigma, mu, sigma


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """L2-penalised least squares, closed form: w = (X^T X + lam*I)^-1 X^T y.

    The identity term inflates the diagonal, which shrinks every weight toward
    zero - but never exactly to zero.
    """
    Xs, _, _ = standardize(X)
    yc = y - y.mean()
    p = Xs.shape[1]
    return np.linalg.solve(Xs.T @ Xs + lam * np.eye(p), Xs.T @ yc)


def lasso_fit(X: np.ndarray, y: np.ndarray, lam: float, iters: int = 800) -> np.ndarray:
    """L1-penalised least squares by coordinate descent.

    One weight at a time: compute what it would be without a penalty, then apply
    the soft-threshold  sign(z) * max(|z| - lam, 0).  That max() is the whole
    story - once a weight's evidence is weaker than lam, it is clipped to
    exactly zero. This is why Lasso selects features and Ridge does not.
    """
    Xs, _, _ = standardize(X)
    yc = y - y.mean()
    n, p = Xs.shape
    w = np.zeros(p)
    norms = np.sum(Xs ** 2, axis=0)
    for _ in range(iters):
        for j in range(p):
            residual_wo_j = yc - Xs @ w + Xs[:, j] * w[j]
            z = Xs[:, j] @ residual_wo_j
            w[j] = np.sign(z) * max(abs(z) - lam * n, 0.0) / norms[j]
    return w


def polynomial_design(x: np.ndarray, degree: int) -> np.ndarray:
    """Columns [x, x^2, ..., x^degree]. Polynomial regression is just linear
    regression on invented columns - still linear *in the weights*."""
    return np.column_stack([x ** d for d in range(1, degree + 1)])


def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> np.ndarray:
    return ols_normal_equation(polynomial_design(x, degree), y)


def predict_polynomial(coef: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coef[0] + polynomial_design(x, len(coef) - 1) @ coef[1:]


# --------------------------------------------------------------------------
# regression trees
# --------------------------------------------------------------------------
@dataclass
class Node:
    """One node of a regression tree. Leaves have feature is None."""
    value: float                      # prediction if this node is a leaf
    n_samples: int
    sse: float
    feature: int | None = None        # which column we split on
    threshold: float | None = None    # split point: goes left if x <= threshold
    left: "Node | None" = None
    right: "Node | None" = None
    depth: int = 0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _sse(y: np.ndarray) -> float:
    """Sum of squared deviations from the group mean - a node's 'impurity'."""
    return float(np.sum((y - y.mean()) ** 2)) if len(y) else 0.0


def best_split(X: np.ndarray, y: np.ndarray, min_samples: int) -> tuple[int, float, float] | None:
    """Scan every feature and every candidate threshold; keep the split with the
    lowest combined SSE. Brute force on purpose - it mirrors the definition."""
    best = None
    parent = _sse(y)
    for feature in range(X.shape[1]):
        values = np.unique(X[:, feature])
        if len(values) < 2:
            continue
        thresholds = (values[:-1] + values[1:]) / 2.0  # midpoints between observed values
        for threshold in thresholds:
            mask = X[:, feature] <= threshold
            if mask.sum() < min_samples or (~mask).sum() < min_samples:
                continue
            score = _sse(y[mask]) + _sse(y[~mask])
            if best is None or score < best[2]:
                best = (feature, float(threshold), score)
    if best is None or best[2] >= parent:
        return None
    return best


def split_scan(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Every candidate threshold on a single feature and the SSE it would leave.

    Used by the 'how is a split chosen?' figure - the curve's minimum *is* the
    split the tree picks.
    """
    values = np.unique(x)
    thresholds = (values[:-1] + values[1:]) / 2.0
    scores = np.array([_sse(y[x <= t]) + _sse(y[x > t]) for t in thresholds])
    return thresholds, scores


class RegressionTree:
    """CART for regression: recursively split to minimise squared error.

    Every leaf predicts the mean of the training rows that reach it, so the
    model is a step function - it can bend anywhere but is flat everywhere.
    """

    def __init__(self, max_depth: int = 3, min_samples: int = 2):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.root: Node | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegressionTree":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        self.root = self._grow(X, np.asarray(y, dtype=float), depth=0)
        return self

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        node = Node(value=float(y.mean()), n_samples=len(y), sse=_sse(y), depth=depth)
        if depth >= self.max_depth or len(y) < 2 * self.min_samples:
            return node
        found = best_split(X, y, self.min_samples)
        if found is None:
            return node
        feature, threshold, _ = found
        mask = X[:, feature] <= threshold
        node.feature, node.threshold = feature, threshold
        node.left = self._grow(X[mask], y[mask], depth + 1)
        node.right = self._grow(X[~mask], y[~mask], depth + 1)
        return node

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        return np.array([self._descend(self.root, row) for row in X])

    def _descend(self, node: Node, row: np.ndarray) -> float:
        while not node.is_leaf:
            node = node.left if row[node.feature] <= node.threshold else node.right
        return node.value

    def n_leaves(self) -> int:
        def count(n: Node) -> int:
            return 1 if n.is_leaf else count(n.left) + count(n.right)
        return count(self.root)

    def leaves(self) -> list[Node]:
        """All leaf nodes, left to right. Used to show how XGBoost's lambda
        shrinks leaf outputs (small leaves get shrunk hardest)."""
        found: list[Node] = []

        def walk(n: Node) -> None:
            if n.is_leaf:
                found.append(n)
            else:
                walk(n.left)
                walk(n.right)

        walk(self.root)
        return found


# --------------------------------------------------------------------------
# gradient boosting
# --------------------------------------------------------------------------
@dataclass
class BoostingHistory:
    """Everything needed to *draw* the boosting process, round by round."""
    f0: float
    trees: list[RegressionTree] = field(default_factory=list)
    stage_predictions: list[np.ndarray] = field(default_factory=list)  # F_m on train
    residuals: list[np.ndarray] = field(default_factory=list)          # target of tree m
    train_mse: list[float] = field(default_factory=list)
    valid_mse: list[float] = field(default_factory=list)


class GradientBoostedTrees:
    """Gradient boosting with squared-error loss, written out in full.

        F_0(x) = mean(y)
        for m in 1..M:
            r      = y - F_{m-1}(x)          # residual = -gradient of MSE
            tree_m = fit(X, r)               # learn the pattern in our mistakes
            F_m    = F_{m-1} + lr * tree_m   # take a deliberately small step

    The residual step is the whole reason it is called *gradient* boosting: for
    squared error, (y - y_hat) is exactly the negative gradient of the loss with
    respect to the prediction. Boosting is gradient descent in function space -
    each step adds a whole tree instead of nudging a number.
    """

    def __init__(self, n_rounds: int = 30, learning_rate: float = 0.3,
                 max_depth: int = 2, min_samples: int = 2):
        self.n_rounds = n_rounds
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.history = BoostingHistory(f0=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray,
            eval_set: tuple[np.ndarray, np.ndarray] | None = None) -> "GradientBoostedTrees":
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        y = np.asarray(y, dtype=float)

        f0 = float(y.mean())                     # step 1: start dumb
        current = np.full(len(y), f0)
        self.history = BoostingHistory(f0=f0)
        self.history.stage_predictions.append(current.copy())
        self.history.train_mse.append(mse(y, current))

        if eval_set is not None:
            Xv, yv = eval_set
            Xv = np.asarray(Xv, dtype=float)
            if Xv.ndim == 1:
                Xv = Xv[:, None]
            current_valid = np.full(len(yv), f0)
            self.history.valid_mse.append(mse(yv, current_valid))

        for _ in range(self.n_rounds):
            residual = y - current                                   # step 2
            tree = RegressionTree(self.max_depth, self.min_samples).fit(X, residual)  # step 3
            current = current + self.learning_rate * tree.predict(X)  # step 4
            self.history.trees.append(tree)
            self.history.residuals.append(residual)
            self.history.stage_predictions.append(current.copy())
            self.history.train_mse.append(mse(y, current))
            if eval_set is not None:
                current_valid = current_valid + self.learning_rate * tree.predict(Xv)
                self.history.valid_mse.append(mse(yv, current_valid))
        return self

    def predict(self, X: np.ndarray, n_rounds: int | None = None) -> np.ndarray:
        """Predict using the first `n_rounds` trees (None = all of them)."""
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        limit = len(self.history.trees) if n_rounds is None else n_rounds
        out = np.full(len(X), self.history.f0)
        for tree in self.history.trees[:limit]:
            out = out + self.learning_rate * tree.predict(X)
        return out


class BaggedTrees:
    """Random-forest-style bagging, for the bagging-vs-boosting contrast.

    Many deep trees on bootstrap samples, averaged. Attacks variance; the trees
    are independent, so they can be trained in any order (or in parallel) -
    the exact opposite of boosting's strict sequence.
    """

    def __init__(self, n_trees: int = 25, max_depth: int = 6, seed: int = 0):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.seed = seed
        self.trees: list[RegressionTree] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaggedTrees":
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X[:, None]
        n = len(y)
        for _ in range(self.n_trees):
            idx = rng.integers(0, n, n)  # bootstrap sample: draw n rows with replacement
            self.trees.append(RegressionTree(self.max_depth, 2).fit(X[idx], y[idx]))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.mean([t.predict(X) for t in self.trees], axis=0)


# --------------------------------------------------------------------------
# synthetic curves used by the figures
# --------------------------------------------------------------------------
def curved_demand(n: int = 90, noise: float = 6.0, seed: int = 7) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A deliberately non-linear price->demand curve.

    Flat-ish while cheap, a sharp fall through the middle of the range, then a
    floor of loyal buyers. A straight line cannot express this shape; a sum of
    shallow trees can. Returns (price, observed demand, noiseless truth).
    """
    rng = np.random.default_rng(seed)
    price = np.sort(rng.uniform(10, 40, n))
    truth = 20 + 80 / (1 + np.exp((price - 25) / 2.6)) + 6 * np.sin(price / 3.0)
    return price, truth + rng.normal(0, noise, n), truth


def true_curve(price: np.ndarray) -> np.ndarray:
    """The noiseless generating function behind curved_demand - the target any
    model is really trying to recover."""
    return 20 + 80 / (1 + np.exp((price - 25) / 2.6)) + 6 * np.sin(price / 3.0)
