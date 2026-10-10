"""Draw a decision tree as a real tree diagram, using only matplotlib.

Why not graphviz / xgboost.plot_tree? Those need the graphviz *binary* installed
system-wide, which is a common source of "it worked on my machine". This module
has no dependency beyond matplotlib, and it renders both the from-scratch trees
in toy_models.py and a real tree pulled out of a trained XGBoost booster.

Layout is the classic tidy-tree rule:
    leaves are placed left to right in the order they are visited,
    an internal node sits at the midpoint of its two children,
    depth sets the vertical position.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from matplotlib.patches import FancyBboxPatch

from _style import ACCENT, ALT, GOOD, INK, MODEL


@dataclass
class DrawNode:
    """Renderer-agnostic node: whatever produced it, this is what gets drawn."""
    label: str                       # main line inside the box
    sub: str = ""                    # small second line (sample count, etc.)
    children: list["DrawNode"] = field(default_factory=list)
    edge_labels: tuple[str, str] = ("yes", "no")

    @property
    def is_leaf(self) -> bool:
        return not self.children


def from_regression_tree(node, feature_names: list[str], *, value_fmt: str = "{:.1f}",
                         show_n: bool = True) -> DrawNode:
    """Convert a toy_models.Node into a DrawNode."""
    if node.is_leaf:
        return DrawNode(label=value_fmt.format(node.value),
                        sub=f"n = {node.n_samples}" if show_n else "")
    name = feature_names[node.feature] if node.feature < len(feature_names) else f"f{node.feature}"
    return DrawNode(
        label=f"{name} ≤ {node.threshold:.2f}",
        sub=f"n = {node.n_samples}" if show_n else "",
        children=[from_regression_tree(node.left, feature_names, value_fmt=value_fmt, show_n=show_n),
                  from_regression_tree(node.right, feature_names, value_fmt=value_fmt, show_n=show_n)],
    )


def _layout(node: DrawNode, depth: int, cursor: list[float],
            positions: dict[int, tuple[float, float]]) -> float:
    """Post-order walk assigning an x to every node; returns this node's x."""
    if node.is_leaf:
        x = cursor[0]
        cursor[0] += 1.0
    else:
        kids = [_layout(child, depth + 1, cursor, positions) for child in node.children]
        x = sum(kids) / len(kids)
    positions[id(node)] = (x, -float(depth))
    return x


def _depth_of(node: DrawNode) -> int:
    return 0 if node.is_leaf else 1 + max(_depth_of(c) for c in node.children)


def draw_tree(ax, root: DrawNode, *, box_w: float = 0.86, box_h: float = 0.44,
              fontsize: float = 9.5, leaf_color: str = GOOD,
              split_color: str = INK, highlight_path: list[int] | None = None) -> None:
    """Render `root` onto `ax`. highlight_path is a list of child indices
    (0 = left, 1 = right) marking a route to trace in amber."""
    positions: dict[int, tuple[float, float]] = {}
    _layout(root, 0, [0.0], positions)
    depth = _depth_of(root)

    highlighted: set[int] = set()
    if highlight_path is not None:
        walker = root
        highlighted.add(id(walker))
        for step in highlight_path:
            if walker.is_leaf:
                break
            walker = walker.children[step]
            highlighted.add(id(walker))

    def walk(node: DrawNode) -> None:
        x, y = positions[id(node)]
        for index, child in enumerate(node.children):
            cx, cy = positions[id(child)]
            on_path = id(child) in highlighted and id(node) in highlighted
            ax.plot([x, cx], [y - box_h / 2, cy + box_h / 2],
                    color=ACCENT if on_path else "#B9C1CA",
                    lw=2.6 if on_path else 1.5, zorder=1,
                    solid_capstyle="round")
            label = node.edge_labels[index] if index < len(node.edge_labels) else ""
            ax.text((x + cx) / 2 + (0.06 if index else -0.06), (y + cy) / 2 + 0.02, label,
                    ha="right" if index == 0 else "left", va="center",
                    fontsize=fontsize - 1.2,
                    color=ACCENT if on_path else "#7A848E",
                    fontweight="bold" if on_path else "normal", zorder=2)
            walk(child)

        face = "#F2FAF8" if node.is_leaf else "#F4F6F9"
        edge = leaf_color if node.is_leaf else split_color
        if id(node) in highlighted:
            face, edge = "#FDF4E3", ACCENT
        ax.add_patch(FancyBboxPatch(
            (x - box_w / 2, y - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.035,rounding_size=0.09",
            facecolor=face, edgecolor=edge, linewidth=1.7, zorder=3))
        offset = 0.07 if node.sub else 0.0
        ax.text(x, y + offset, node.label, ha="center", va="center",
                fontsize=fontsize, color=INK,
                fontweight="bold" if node.is_leaf else "semibold", zorder=4)
        if node.sub:
            ax.text(x, y - 0.115, node.sub, ha="center", va="center",
                    fontsize=fontsize - 2.1, color="#7A848E", zorder=4)

    walk(root)
    n_leaves = sum(1 for _ in _iter_leaves(root))
    ax.set_xlim(-0.9, max(n_leaves - 1, 1) + 0.9)
    ax.set_ylim(-depth - 0.6, 0.6)
    ax.axis("off")


def _iter_leaves(node: DrawNode):
    if node.is_leaf:
        yield node
    for child in node.children:
        yield from _iter_leaves(child)


def from_xgboost_frame(frame, tree_index: int = 0, *, max_depth: int | None = None) -> DrawNode:
    """Build a DrawNode from `booster.trees_to_dataframe()`.

    The dataframe has one row per node with columns ID, Feature, Split, Yes, No,
    Gain, Cover. Feature == 'Leaf' marks a leaf, whose output sits in Gain.
    """
    tree = frame[frame["Tree"] == tree_index].set_index("ID")

    def build(node_id: str, depth: int) -> DrawNode:
        row = tree.loc[node_id]
        is_leaf = row["Feature"] == "Leaf"
        if is_leaf or (max_depth is not None and depth >= max_depth):
            value = row["Gain"] if is_leaf else float("nan")
            label = f"{value:+.2f}" if is_leaf else "(pruned)"
            return DrawNode(label=label, sub=f"cover {row['Cover']:.0f}")
        name = str(row["Feature"])
        if len(name) > 20:
            name = name[:19] + "…"
        return DrawNode(
            label=f"{name} < {row['Split']:.2f}",
            sub=f"gain {row['Gain']:.0f}",
            children=[build(row["Yes"], depth + 1), build(row["No"], depth + 1)],
        )

    root_id = f"{tree_index}-0"
    return build(root_id, 0)
