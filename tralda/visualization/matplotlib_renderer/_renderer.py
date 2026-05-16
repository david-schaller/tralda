"""Matplotlib renderer for tree layouts.

This module provides :class:`MatplotlibRenderer`, which consumes a
:class:`~tralda.visualization.layout.TreeLayout` and draws the tree on a matplotlib
:class:`~matplotlib.axes.Axes`.

**Separation of concerns**

All geometric computation lives in :class:`~tralda.visualization.layout.TreeLayout`.  This module
is only responsible for translating that geometry into matplotlib drawing calls.

**Coordinate normalization**

Depths are optionally normalised to ``[0, 1]`` before rendering (see *rescale_depth*).  This keeps
symbol and font sizes visually consistent regardless of the unit system used for edge lengths (e.g.
millions of years, substitutions per site, integer rank).

**Node styling**

Every node is styled by calling :meth:`~tralda.visualization.style.TreeStyle.resolve` on the
renderer's :class:`~tralda.visualization.style.TreeStyle`.  The resolved
:class:`~tralda.visualization.style.NodeStyle` provides the symbol name, colors, and edge style.
See :mod:`~tralda.visualization.style` for the full styling API.
"""

from __future__ import annotations

import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tralda.datastructures.tree import TreeNode
from tralda.visualization.layout import LayoutMode
from tralda.visualization.layout import TreeLayout
from tralda.visualization.style import NodeStyle
from tralda.visualization.style import TreeStyle
from tralda.visualization.matplotlib_renderer._symbols import SYMBOL_REGISTRY


# --------------------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------------------


class MatplotlibRenderer:
    """Render a :class:`~tralda.visualization.layout.TreeLayout` with matplotlib.

    All node symbols are drawn at a size specified in *points*, so they remain visually consistent
    regardless of the data-coordinate scale.  Depth coordinates are optionally normalised to
    ``[0, 1]`` before rendering.

    Pass a :class:`~tralda.visualization.style.TreeStyle` to control the appearance of every node.
    The style is resolved once per node during :meth:`render` via
    :meth:`~tralda.visualization.style.TreeStyle.resolve`, which merges the default style, an
    optional per-node style function, and optional direct node overrides in that order.

    Example usage:

        from tralda.visualization.layout import TreeLayout
        from tralda.visualization.matplotlib_renderer import MatplotlibRenderer
        from tralda.visualization.style import TreeStyle

        layout = TreeLayout(tree, edge_length_mode="attr")
        style = TreeStyle.from_maps(
            symbol_map={"S": "circle", "D": "square", "H": "triangle"},
            color_map={"a": "steelblue", "b": "tomato"},
        )
        renderer = MatplotlibRenderer(layout, tree_style=style)
        fig, ax = renderer.render()
    """

    def __init__(
        self,
        layout: TreeLayout,
        ax: Axes | None = None,
        *,
        tree_style: TreeStyle | None = None,
        rescale_depth: bool = True,
        label_pad: float = 4.0,
        show_labels: bool = True,
        show_ghost_segments: bool = True,
        figsize: tuple[float, float] | None = None,
    ) -> None:
        """Construct a renderer.

        Args:
            layout: Pre-computed tree layout to render.
            ax: Existing :class:`~matplotlib.axes.Axes` to draw on.  When ``None`` (the default)
                a new figure and axes are created automatically when :meth:`render` is called.
            tree_style: Styling configuration for nodes, edges, and labels.  When ``None`` the
                default :class:`~tralda.visualization.style.TreeStyle` is used.
            rescale_depth: Normalise the depth axis to ``[0, 1]`` before rendering.  Recommended
                when edge lengths are in absolute units (e.g. millions of years).  Default ``True``.
            label_pad: Gap between the symbol edge and the start of the label text, in points.
                Default ``4``.
            show_labels: Draw leaf labels.  Default ``True``.
            show_ghost_segments: Extend short leaves to the maximum depth with a dashed line.  Only
                visible when leaves sit at different depths (``ATTR`` / ``UNIFORM`` edge-length
                modes).  Default ``True``.
            figsize: Figure size ``(width, height)`` in inches.  When ``None`` the size is derived
                automatically from the leaf count and layout mode.
        """
        self.layout = layout
        self._user_ax = ax

        self.tree_style: TreeStyle = tree_style if tree_style is not None else TreeStyle()
        self.rescale_depth = rescale_depth
        self.label_pad = label_pad
        self.show_labels = show_labels
        self.show_ghost_segments = show_ghost_segments
        self.figsize = figsize

    # ----------------------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------------------

    def render(self) -> tuple[Figure, Axes]:
        """Draw the tree and return the figure and axes.

        When no *ax* was supplied to the constructor a new figure is created with an automatically
        derived size.  ``fig.tight_layout()`` is called automatically for new figures; when drawing
        on a caller-supplied axes the caller is responsible for layout management.

        Returns:
            ``(fig, ax)`` — the :class:`~matplotlib.figure.Figure` and
            :class:`~matplotlib.axes.Axes` the tree was drawn on.
        """
        new_fig = False
        if self._user_ax is not None:
            ax = self._user_ax
            fig = ax.get_figure()
        else:
            fig, ax = plt.subplots(figsize=self._auto_figsize())
            new_fig = True

        mode = self.layout.layout_mode
        positions = self._build_positions()

        # Resolve per-node styles once, reused by edge and node drawing passes.
        node_styles: dict[TreeNode, NodeStyle] = {
            v: self.tree_style.resolve(v, mode) for v in self.layout.tree.preorder()
        }

        self._draw_edges(ax, positions, node_styles)
        self._draw_nodes(ax, positions, node_styles, mode)
        if self.show_labels:
            self._draw_labels(ax, positions, node_styles)
        self._style_axes(ax)

        if new_fig:
            fig.tight_layout()

        return fig, ax

    # ----------------------------------------------------------------------------------------------
    # Coordinate preparation
    # ----------------------------------------------------------------------------------------------

    def _build_positions(self) -> dict[TreeNode, tuple[float, float]]:
        """Return layout positions, optionally normalising the depth axis to [0, 1].

        Returns:
            Mapping from tree nodes to (x, y) positions in data coordinates.
        """
        positions = dict(self.layout.positions)

        if not self.rescale_depth or self.layout.max_depth == 0.0:
            return positions

        scale = 1.0 / self.layout.max_depth
        mode = self.layout.layout_mode

        if mode is LayoutMode.HORIZONTAL:
            return {v: (x * scale, y) for v, (x, y) in positions.items()}
        elif mode is LayoutMode.VERTICAL:
            return {v: (x, y * scale) for v, (x, y) in positions.items()}
        else:
            # CIRCULAR: depth is encoded in the radius — scale uniformly.
            return {v: (x * scale, y * scale) for v, (x, y) in positions.items()}

    # ----------------------------------------------------------------------------------------------
    # Edge drawing
    # ----------------------------------------------------------------------------------------------

    def _draw_edges(
        self,
        ax: Axes,
        positions: dict[TreeNode, tuple[float, float]],
        node_styles: dict[TreeNode, NodeStyle],
    ) -> None:
        """Draw all tree edges and optional ghost segments."""
        layout = self.layout
        mode = layout.layout_mode
        ts = self.tree_style

        for v in layout.tree.preorder():
            vx, vy = positions[v]
            ns = node_styles[v]

            # ── parent edge ────────────────────────────────────────────────────────────────────
            if v.parent is not None:
                px, py = positions[v.parent]

                if mode is LayoutMode.HORIZONTAL:
                    # Horizontal run from parent x to child x at the child's y.
                    ax.plot(
                        [px, vx],
                        [vy, vy],
                        color=ns.edge_color,
                        lw=ns.edge_lw,
                        ls=ns.edge_ls,
                        solid_capstyle="round",
                    )
                elif mode is LayoutMode.VERTICAL:
                    # Vertical run from parent y to child y at the child's x.
                    ax.plot(
                        [vx, vx],
                        [py, vy],
                        color=ns.edge_color,
                        lw=ns.edge_lw,
                        ls=ns.edge_ls,
                        solid_capstyle="round",
                    )
                else:
                    # CIRCULAR: radial segment at the child's angle.
                    r_parent = math.hypot(px, py)
                    if r_parent > 1e-12:
                        theta_v = math.atan2(vy, vx)
                        start_x = r_parent * math.cos(theta_v)
                        start_y = r_parent * math.sin(theta_v)
                    else:
                        start_x, start_y = px, py
                    ax.plot(
                        [start_x, vx],
                        [start_y, vy],
                        color=ns.edge_color,
                        lw=ns.edge_lw,
                        ls=ns.edge_ls,
                        solid_capstyle="round",
                    )

            # ── child connector ────────────────────────────────────────────────────────────────
            if v.children:
                children = list(v.children)
                first_pos = positions[children[0]]
                last_pos = positions[children[-1]]
                consensus_style = ts.consensus_style(children, mode)

                if mode is LayoutMode.HORIZONTAL:
                    ax.plot(
                        [vx, vx],
                        [first_pos[1], last_pos[1]],
                        color=consensus_style.edge_color,
                        lw=consensus_style.edge_lw,
                        ls=consensus_style.edge_ls,
                        solid_capstyle="round",
                    )
                elif mode is LayoutMode.VERTICAL:
                    ax.plot(
                        [first_pos[0], last_pos[0]],
                        [vy, vy],
                        color=consensus_style.edge_color,
                        lw=consensus_style.edge_lw,
                        ls=consensus_style.edge_ls,
                        solid_capstyle="round",
                    )
                else:
                    # CIRCULAR: arc at the node's radius spanning the outermost children's angles.
                    self._draw_arc(ax, vx, vy, first_pos, last_pos, consensus_style)

        # ── ghost segments ─────────────────────────────────────────────────────────────────────
        if self.show_ghost_segments:
            for v in layout.tree.leaves():
                seg = layout.ghost_segment(v)
                if seg is None:
                    continue
                (x0, y0), (x1, y1) = seg
                if self.rescale_depth and layout.max_depth > 0:
                    scale = 1.0 / layout.max_depth
                    if mode is LayoutMode.HORIZONTAL:
                        x0, x1 = x0 * scale, x1 * scale
                    elif mode is LayoutMode.VERTICAL:
                        y0, y1 = y0 * scale, y1 * scale
                    else:
                        x0, y0 = x0 * scale, y0 * scale
                        x1, y1 = x1 * scale, y1 * scale
                ax.plot(
                    [x0, x1],
                    [y0, y1],
                    color=ts.ghost_color,
                    lw=ts.ghost_lw,
                    ls=ts.ghost_ls,
                )

    def _draw_arc(
        self,
        ax: Axes,
        px: float,
        py: float,
        first_pos: tuple[float, float],
        last_pos: tuple[float, float],
        ns: NodeStyle,
    ) -> None:
        """Draw a circular arc at radius ``hypot(px, py)``.

        The arc spans from the angular projection of *first_pos* to that of *last_pos*, sweeping
        the shorter way around.
        """
        r = math.hypot(px, py)
        if r < 1e-12:
            return

        # atan2 returns values in (-π, π].  Restore to [0, 2π) so that the angles match the
        # layout's counterclockwise assignment (rank 0 → 0, rank n-1 → just below 2π).
        # Because the first child always has a smaller rank than the last, theta1 ≤ theta2
        # after this normalisation, and linspace sweeps the correct arc without any heuristic.
        theta1 = math.atan2(first_pos[1], first_pos[0])
        theta2 = math.atan2(last_pos[1], last_pos[0])
        if theta1 < 0.0:
            theta1 += 2.0 * math.pi
        if theta2 < 0.0:
            theta2 += 2.0 * math.pi
        if theta2 < theta1:
            theta2 += 2.0 * math.pi

        n_pts = max(3, int((theta2 - theta1) * 30) + 2)
        thetas = np.linspace(theta1, theta2, n_pts)
        ax.plot(
            r * np.cos(thetas),
            r * np.sin(thetas),
            color=ns.edge_color,
            lw=ns.edge_lw,
            ls=ns.edge_ls,
            solid_capstyle="round",
        )

    # ----------------------------------------------------------------------------------------------
    # Node drawing
    # ----------------------------------------------------------------------------------------------

    def _draw_nodes(
        self,
        ax: Axes,
        positions: dict[TreeNode, tuple[float, float]],
        node_styles: dict[TreeNode, NodeStyle],
        mode: LayoutMode,
    ) -> None:
        """Draw a symbol at every node."""
        for v in self.layout.tree.preorder():
            ns = node_styles[v]
            drawer = SYMBOL_REGISTRY[ns.symbol]
            x, y = positions[v]

            if mode is LayoutMode.HORIZONTAL:
                angle = 0.0
            elif mode is LayoutMode.VERTICAL:
                angle = -90.0
            else:  # CIRCULAR: radial direction from the origin
                angle = math.degrees(math.atan2(y, x))

            drawer(ax, x, y, ns, angle=angle, layout_mode=mode)

    # ----------------------------------------------------------------------------------------------
    # Label drawing
    # ----------------------------------------------------------------------------------------------

    def _draw_labels(
        self,
        ax: Axes,
        positions: dict[TreeNode, tuple[float, float]],
        node_styles: dict[TreeNode, NodeStyle],
    ) -> None:
        """Draw a label next to every leaf that has a ``label`` attribute."""
        layout = self.layout
        mode = layout.layout_mode

        for v in layout.tree.leaves():
            label = getattr(v, "label", None)
            if label is None:
                continue

            ns = node_styles[v]
            offset_pts = ns.symbol_size / 2.0 + self.label_pad
            x, y = positions[v]

            # If a ghost segment extends this leaf to max_depth, anchor the label at the far end
            # so all leaf labels are visually aligned regardless of actual branch length.
            seg = layout.ghost_segment(v)
            if seg is not None:
                (_, _), (x1_raw, y1_raw) = seg
                scale = (
                    (1.0 / layout.max_depth)
                    if (self.rescale_depth and layout.max_depth > 0)
                    else 1.0
                )
                if mode is LayoutMode.HORIZONTAL:
                    x = x1_raw * scale
                elif mode is LayoutMode.VERTICAL:
                    y = y1_raw * scale
                else:  # CIRCULAR
                    x = x1_raw * scale
                    y = y1_raw * scale

            ha = layout.label_ha.get(v, "left")
            va = layout.label_va.get(v, "center")
            angle = layout.label_angle.get(v, 0.0)

            if mode is LayoutMode.HORIZONTAL:
                xytext = (offset_pts, 0.0)
            elif mode is LayoutMode.VERTICAL:
                # y-axis is inverted; use a negative offset to place text below the leaf symbol.
                xytext = (0.0, -offset_pts)
            else:
                # CIRCULAR: offset radially outward from the node.
                angle_rad = math.radians(angle)
                xytext = (math.cos(angle_rad) * offset_pts, math.sin(angle_rad) * offset_pts)
                # Flip text rotation on the left half so it reads away from the root.
                # Simple ±180 flip; ha is already set to "right" for left-half nodes by the layout.
                if not (-90.0 <= angle <= 90.0):
                    angle = angle - 180.0 if angle >= 0.0 else angle + 180.0

            ax.annotate(
                str(label),
                xy=(x, y),
                xytext=xytext,
                textcoords="offset points",
                fontsize=ns.label_fontsize,
                color=ns.label_color,
                fontweight=ns.label_fontweight,
                fontstyle=ns.label_fontstyle,
                ha=ha,
                va=va,
                rotation=angle,
                rotation_mode="anchor",
            )

    # ----------------------------------------------------------------------------------------------
    # Axis styling
    # ----------------------------------------------------------------------------------------------

    def _style_axes(self, ax: Axes) -> None:
        """Remove decorations and orient the axes for the chosen layout mode."""
        mode = self.layout.layout_mode
        n = self.layout.leaf_count

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if mode is LayoutMode.HORIZONTAL:
            # Leaf rank 0 at the top, increasing downward.  Setting ylim explicitly avoids
            # matplotlib's default 5 % margin, which wastes significant vertical space for
            # large trees.  Half a rank unit of padding keeps the outermost symbols unclipped.
            ax.set_ylim(n - 0.5, -0.5)
        elif mode is LayoutMode.VERTICAL:
            # Depth 0 (root) at the top, increasing downward.
            ax.invert_yaxis()
            # Tighten the leaf-rank (x) axis for the same reason as the horizontal case.
            ax.set_xlim(-0.5, n - 0.5)
        else:
            # CIRCULAR: equal aspect ratio so the tree is not distorted.
            ax.set_aspect("equal")

    # ----------------------------------------------------------------------------------------------
    # Figure sizing
    # ----------------------------------------------------------------------------------------------

    def _auto_figsize(self) -> tuple[float, float]:
        """Compute a sensible default figure size from the leaf count."""
        if self.figsize is not None:
            return self.figsize

        n = max(1, self.layout.leaf_count)
        mode = self.layout.layout_mode
        default_symbol_size = self.tree_style.default.symbol_size

        if mode is LayoutMode.HORIZONTAL:
            return (8.0, max(3.0, n * 0.28))
        elif mode is LayoutMode.VERTICAL:
            return (max(3.0, n * 0.28), 7.0)
        else:
            # CIRCULAR: scale the figure so that adjacent leaf symbols have adequate arc spacing.
            # The effective plot circumference ≈ π × size_inches × 72 pts/inch × 0.8 (tight-layout
            # margin factor).  Setting arc-per-leaf ≥ 1.5 × symbol_size and solving for size:
            #   size = n × symbol_size × 1.5 / (π × 72 × 0.8)
            min_size = n * default_symbol_size * 1.5 / (math.pi * 72.0 * 0.8)
            return (max(5.0, min_size), max(5.0, min_size))
