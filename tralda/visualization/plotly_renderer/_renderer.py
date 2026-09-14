"""Plotly renderer for tree layouts.

This module provides :class:`PlotlyRenderer`, which consumes a
:class:`~tralda.visualization.layout.TreeLayout` and builds an interactive
:class:`plotly.graph_objects.Figure`.

**Separation of concerns**

All geometric computation lives in :class:`~tralda.visualization.layout.TreeLayout`. This module
is only responsible for translating that geometry into Plotly traces.

**Edge batching**

To keep the figure lightweight, edges that share the same color, width, and dash style are batched
into a single :class:`~plotly.graph_objects.Scatter` trace using ``None`` gaps to separate
individual segments.  Circular arcs are emitted as polylines (no intra-arc gaps).

**Coordinate normalisation**

Depths are optionally normalised to ``[0, 1]`` before rendering (see *rescale_depth*), matching
the behaviour of the matplotlib renderer.

**Node styling**

Every node is styled by calling :meth:`~tralda.visualization.style.TreeStyle.resolve`. Symbol
names are mapped to Plotly marker symbols. See :mod:`~tralda.visualization.plotly_renderer._symbols`
for details.

**Interactive hover**

Each node marker carries a hover tooltip.  By default it shows the node's ``label`` attribute and
its depth.  Pass additional attribute names via *hover_attrs* to include more information.

**Label placement and rotation**

All labels are placed via individual :func:`~plotly.graph_objects.Figure.add_annotation` entries
so that each label can receive a pixel-space offset (``xshift``/``yshift``) and a per-label
``textangle``.  Horizontal labels are offset to the right of the node; vertical labels are offset
downward in screen space and rotated 90° clockwise so they read top-to-bottom without overlap;
circular labels are offset radially outward using the node's original polar angle (before the
left-half readability flip) and rotated to align with the radial direction.
"""

from __future__ import annotations

import html
import math
from typing import Any

import plotly.graph_objects as go

from tralda.datastructures.tree import TreeNode
from tralda.visualization._base_renderer import BaseRenderer
from tralda.visualization._base_renderer import arc_theta_range
from tralda.visualization._symbol_defs import resolve_linestyle
from tralda.visualization.layout import LayoutMode
from tralda.visualization.layout import TreeLayout
from tralda.visualization.plotly_renderer._symbols import get_node_traces
from tralda.visualization.plotly_renderer._utils import to_plotly_color
from tralda.visualization.plotly_renderer._utils import to_plotly_font_size
from tralda.visualization.style import NodeStyle
from tralda.visualization.style import TreeStyle


# --------------------------------------------------------------------------------------------------
# Type aliases
# --------------------------------------------------------------------------------------------------


EdgeKey = tuple[str, float, str]  # (color, width, dash)


# --------------------------------------------------------------------------------------------------
# Renderer
# --------------------------------------------------------------------------------------------------


class PlotlyRenderer(BaseRenderer):
    """Render a :class:`~tralda.visualization.layout.TreeLayout` as an interactive Plotly figure.

    All tree geometry is read from a pre-computed :class:`~tralda.visualization.layout.TreeLayout`.
    Node appearance is controlled by a :class:`~tralda.visualization.style.TreeStyle`.

    Example usage::

        from tralda.visualization.layout import TreeLayout
        from tralda.visualization.plotly_renderer import PlotlyRenderer
        from tralda.visualization.style import TreeStyle

        layout = TreeLayout(tree, edge_length_mode="attr")
        style = TreeStyle(leaf_symbol="circle")
        renderer = PlotlyRenderer(layout, tree_style=style, hover_attrs=["dist", "support"])
        fig = renderer.render()
        fig.show()                      # interactive browser / Jupyter
        fig.write_html("tree.html")     # self-contained HTML file
        fig.write_image("tree.png")     # static image, requires the 'kaleido' package
                                        # (included in the 'plotly' extra:
                                        # pip install tralda[plotly])
    """

    def __init__(
        self,
        layout: TreeLayout,
        *,
        tree_style: TreeStyle | None = None,
        rescale_depth: bool = True,
        show_labels: bool = True,
        show_internal_labels: bool = False,
        show_ghost_segments: bool = True,
        width: int | None = None,
        height: int | None = None,
        hover_attrs: list[str] | str | None = None,
    ) -> None:
        """Construct a renderer.

        Args:
            layout: Pre-computed tree layout to render.
            tree_style: Styling configuration for nodes, edges, and labels. When ``None`` the
                default :class:`~tralda.visualization.style.TreeStyle` is used.
            rescale_depth: Normalise the depth axis to ``[0, 1]`` before rendering. Default
                ``True``.
            show_labels: Draw leaf labels. Default ``True``.
            show_internal_labels: Also draw labels for internal nodes (including the root). Only
                nodes that have a ``label`` attribute are labelled. Default ``False``.
            show_ghost_segments: Extend short leaves to the maximum depth with a dashed line. Only
                visible when leaves sit at different depths (``ATTR`` / ``UNIFORM`` edge-length
                modes). Default ``True``.
            width: Figure width in pixels. When ``None`` an automatic width is derived from the
                leaf count and layout mode.
            height: Figure height in pixels. When ``None`` an automatic height is derived from
                the leaf count and layout mode.
            hover_attrs: One or more node attribute names to include in hover tooltips in addition
                to ``label`` and ``depth``. Accepts a single string or a list of strings.
        """
        super().__init__(
            layout,
            tree_style=tree_style,
            rescale_depth=rescale_depth,
            show_labels=show_labels,
            show_internal_labels=show_internal_labels,
            show_ghost_segments=show_ghost_segments,
        )
        self.width = width
        self.height = height
        self.hover_attrs: list[str] = (
            [hover_attrs] if isinstance(hover_attrs, str) else (hover_attrs or [])
        )

    # ----------------------------------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------------------------------

    def render(self) -> go.Figure:
        """Build and return the Plotly figure.

        Returns:
            A :class:`plotly.graph_objects.Figure` containing all tree traces.
        """
        mode = self.layout.layout_mode
        positions = self._build_positions()

        node_styles: dict[TreeNode, NodeStyle] = {
            v: self.tree_style.resolve(v, mode) for v in self.layout.tree.preorder()
        }

        traces: list[Any] = []
        traces.extend(self._edge_traces(positions, node_styles))
        if self.show_ghost_segments:
            ghost = self._ghost_trace()
            if ghost is not None:
                traces.append(ghost)
        traces.extend(get_node_traces(self.layout, positions, node_styles, self.hover_attrs))
        fig = go.Figure(data=traces)
        self._configure_layout(fig)
        self._add_label_annotations(fig, positions, node_styles)
        return fig

    # ----------------------------------------------------------------------------------------------
    # Edge traces
    # ----------------------------------------------------------------------------------------------

    def _edge_traces(
        self,
        positions: dict[TreeNode, tuple[float, float]],
        node_styles: dict[TreeNode, NodeStyle],
    ) -> list[go.Scatter]:
        """Build batched edge traces, one per unique (color, width, dash) combination.

        Edges that share a style are grouped into a single :class:`~plotly.graph_objects.Scatter`
        trace using ``None`` gaps to break the line between separate segments.  Circular arcs are
        added as continuous polylines (no intra-arc None gaps).
        """
        layout = self.layout
        mode = layout.layout_mode
        ts = self.tree_style

        # Accumulator: (color, lw, dash) → (x_list, y_list) with None gaps between segments.
        groups: dict[EdgeKey, tuple[list[float | None], list[float | None]]] = {}

        def _key(ns: NodeStyle) -> EdgeKey:
            return (
                to_plotly_color(ns.edge_color),
                float(ns.edge_lw),
                resolve_linestyle(str(ns.edge_ls)),
            )

        def _add_segment(k: EdgeKey, x0: float, y0: float, x1: float, y1: float) -> None:
            if k not in groups:
                groups[k] = ([], [])
            xs, ys = groups[k]
            xs += [x0, x1, None]
            ys += [y0, y1, None]

        def _add_polyline(k: EdgeKey, pts: list[tuple[float, float]]) -> None:
            """Append a connected polyline (no intra-segment None gaps)."""
            if not pts:
                return
            if k not in groups:
                groups[k] = ([], [])
            xs, ys = groups[k]
            for px, py in pts:
                xs.append(px)
                ys.append(py)
            xs.append(None)
            ys.append(None)

        for v in layout.tree.preorder():
            vx, vy = positions[v]
            ns = node_styles[v]
            k = _key(ns)

            # ── parent edge ───────────────────────────────────────────────────────────────────
            if v.parent is not None:
                px, py = positions[v.parent]
                x0, y0, x1, y1 = self._parent_edge_segment(vx, vy, px, py, mode)
                _add_segment(k, x0, y0, x1, y1)

            # ── child connector ───────────────────────────────────────────────────────────────
            if v.children:
                children, first_pos, last_pos = self._child_connector_range(v, positions)
                cs = ts.consensus_style(children, mode)
                ck = _key(cs)

                if mode is LayoutMode.HORIZONTAL:
                    _add_segment(ck, vx, first_pos[1], vx, last_pos[1])
                elif mode is LayoutMode.VERTICAL:
                    _add_segment(ck, first_pos[0], vy, last_pos[0], vy)
                else:  # CIRCULAR — arc at the node's radius
                    arc_pts = _arc_polyline(vx, vy, first_pos, last_pos)
                    _add_polyline(ck, arc_pts)

        traces = []
        for (color, lw, dash), (xs, ys) in groups.items():
            traces.append(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line=dict(color=color, width=lw, dash=dash),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        return traces

    # ----------------------------------------------------------------------------------------------
    # Ghost trace
    # ----------------------------------------------------------------------------------------------

    def _ghost_trace(self) -> go.Scatter | None:
        """Build a single batched ghost-segment trace for all leaves."""
        layout = self.layout
        mode = layout.layout_mode
        ts = self.tree_style

        xs: list[float | None] = []
        ys: list[float | None] = []

        for v in layout.tree.leaves():
            seg = layout.ghost_segment(v)
            if seg is None:
                continue
            (x0, y0), (x1, y1) = seg
            x0, y0, x1, y1 = self._rescale_seg(x0, y0, x1, y1, mode)
            xs += [x0, x1, None]
            ys += [y0, y1, None]

        if not xs:
            return None

        return go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            line=dict(
                color=to_plotly_color(ts.ghost_color),
                width=ts.ghost_lw,
                dash=resolve_linestyle(ts.ghost_ls),
            ),
            hoverinfo="skip",
            showlegend=False,
        )

    # ----------------------------------------------------------------------------------------------
    # Label trace
    # ----------------------------------------------------------------------------------------------

    def _add_label_annotations(
        self,
        fig: go.Figure,
        positions: dict[TreeNode, tuple[float, float]],
        node_styles: dict[TreeNode, NodeStyle],
    ) -> None:
        """Add label annotations to *fig* for all layout modes.

        :func:`plotly.graph_objects.Figure.add_annotation` supports per-label ``textangle`` and
        pixel-space ``xshift``/``yshift``, which allows correct padding and rotation for every
        layout mode:

        * **Horizontal** — labels placed to the right of each node with a radial pixel gap;
          text is horizontal (``textangle = 0``).
        * **Vertical** — labels placed below each node (in screen space) with a pixel gap;
          text is rotated 90° clockwise so that labels read top-to-bottom and do not overlap.
        * **Circular** — each label is offset radially outward using the node's polar angle;
          text on the left half of the tree is flipped by 180° so it always reads outward from
          the root.  The radial offset uses the *original* (pre-flip) angle so that the offset
          vector points outward regardless of the flip.
        """
        layout = self.layout
        mode = layout.layout_mode
        _PAD = 4.0  # gap between symbol edge and label start, in pixels

        # Accumulator for all annotations; added in a single batch at the end for performance.
        annotations: list[go.layout.Annotation] = []

        for v in layout.tree.preorder():
            if v.is_leaf() and not self.show_labels:
                continue
            if not v.is_leaf() and not self.show_internal_labels:
                continue

            label = getattr(v, "label", None)
            if label is None:
                continue

            ns = node_styles[v]
            x, y = positions[v]
            x, y = self._ghost_label_anchor(v, x, y, mode)

            offset = ns.symbol_size / 2.0 + _PAD
            angle = layout.label_angle.get(v, 0.0)
            ha = layout.label_ha.get(v, "left")
            font_size = to_plotly_font_size(ns.label_fontsize)

            if mode is LayoutMode.HORIZONTAL:
                # Leaves are at the rightmost position; push label further right.
                xshift, yshift = offset, 0.0
                textangle = 0.0
                xanchor, yanchor = "left", "middle"
            elif mode is LayoutMode.VERTICAL:
                # Leaves are at the screen bottom (y-axis reversed); push label downward.
                # Rotate 90° CW so labels read top-to-bottom and do not overlap.
                xshift, yshift = 0.0, -offset
                textangle = 90.0
                xanchor, yanchor = "center", "top"
            else:  # CIRCULAR
                # Compute the offset vector from the ORIGINAL angle so it always points outward,
                # regardless of the 180° readability flip applied to the text rotation.
                display_angle = angle
                if not (-90.0 <= angle <= 90.0):
                    display_angle = angle - 180.0 if angle >= 0.0 else angle + 180.0
                # Plotly textangle is clockwise-from-horizontal; matplotlib uses CCW.
                textangle = -display_angle

                # Determine the anchor and shift direction based on the original angle.  Plotly uses
                # a different convention for the relationship between textangle and x/yanchor than
                # matplotlib. The following logic produces a similar visual result to matplotlib.
                # Using yanchor="bottom"/"top" shifts the anchor by half the line-box height.
                # Plotly uses a CSS line-height of 1.5, so the line box is 1.5 * fontsize tall and
                # half of that is 0.75 * fontsize. The correction nudges the anchor back to the text
                # centre along the text-perpendicular direction.
                xanchor = "left" if ha == "left" else "right"
                yanchor = "bottom" if 0.0 <= angle <= 180.0 else "top"
                orig_rad = math.radians(angle)
                xshift = math.cos(orig_rad) * offset
                yshift = math.sin(orig_rad) * offset
                correction_factor = font_size * 0.75
                if 0.0 <= angle <= 90.0 or -180.0 <= angle < -90.0:
                    xshift -= correction_factor * math.sin(orig_rad)
                    yshift -= correction_factor * math.cos(orig_rad)
                else:  # -90.0 <= angle < 0.0 or 90.0 < angle <= 180.0
                    xshift += correction_factor * math.sin(orig_rad)
                    yshift += correction_factor * math.cos(orig_rad)

            # Apply font weight and style via HTML tags.
            text = html.escape(str(label))
            if ns.label_fontstyle == "italic":
                text = f"<i>{text}</i>"
            if ns.label_fontweight == "bold":
                text = f"<b>{text}</b>"

            annotations.append(
                go.layout.Annotation(
                    x=x,
                    y=y,
                    text=text,
                    textangle=textangle,
                    showarrow=False,
                    font=dict(
                        color=to_plotly_color(ns.label_color),
                        size=font_size,
                    ),
                    xanchor=xanchor,
                    yanchor=yanchor,
                    xshift=xshift,
                    yshift=yshift,
                    xref="x",
                    yref="y",
                )
            )

        if annotations:
            fig.update_layout(annotations=annotations)

    # ----------------------------------------------------------------------------------------------
    # Figure layout configuration
    # ----------------------------------------------------------------------------------------------

    def _configure_layout(self, fig: go.Figure) -> None:
        """Remove decorations and orient axes to match the chosen layout mode.

        Args:
            fig: The figure to configure.
        """
        mode = self.layout.layout_mode
        n = self.layout.rank_span
        w, h = self._auto_size()

        clean_axis: dict[str, Any] = dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
            showline=False,
            ticks="",
        )

        if mode is LayoutMode.HORIZONTAL:
            fig.update_layout(
                width=w,
                height=h,
                xaxis=dict(**clean_axis),
                yaxis=dict(**clean_axis, range=[n - 0.5, -0.5]),
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
            )
        elif mode is LayoutMode.VERTICAL:
            fig.update_layout(
                width=w,
                height=h,
                xaxis=dict(**clean_axis, range=[-0.5, n - 0.5]),
                yaxis=dict(**clean_axis, autorange="reversed"),
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
            )
        else:  # CIRCULAR — equal aspect ratio so the tree is not distorted
            fig.update_layout(
                width=w,
                height=h,
                xaxis=dict(**clean_axis, scaleanchor="y", scaleratio=1),
                yaxis=dict(**clean_axis),
                plot_bgcolor="white",
                margin=dict(l=10, r=10, t=10, b=10),
            )

    def _auto_size(self) -> tuple[int, int]:
        """Return a sensible default figure size in pixels from the leaf count and symbol size."""
        n = max(1, self.layout.rank_span)
        mode = self.layout.layout_mode
        symbol_size = self.tree_style.default.symbol_size

        if mode is LayoutMode.HORIZONTAL:
            return self.width or 1000, self.height or max(300, int(n * symbol_size * 1.7))
        elif mode is LayoutMode.VERTICAL:
            return self.width or max(300, int(n * symbol_size * 1.7)), self.height or 800
        else:  # CIRCULAR
            # Scale so adjacent leaf symbols have adequate arc spacing.
            # Effective circumference ≈ π × figure_px; setting arc-per-leaf ≥ 2 × symbol_size
            # and solving: figure_px ≥ n × symbol_size × 2 / π.
            px = max(500, int(n * symbol_size * 2 / math.pi))
            return self.width or px, self.height or px


# --------------------------------------------------------------------------------------------------
# Module-level helpers
# --------------------------------------------------------------------------------------------------


def _arc_polyline(
    px: float,
    py: float,
    first_pos: tuple[float, float],
    last_pos: tuple[float, float],
) -> list[tuple[float, float]]:
    """Return a list of (x, y) points approximating a circular arc at radius ``hypot(px, py)``.

    The arc spans from the angular projection of *first_pos* to that of *last_pos*, sweeping
    counterclockwise (increasing angle).

    Args:
        px: X-coordinate of the parent node (defines the arc radius).
        py: Y-coordinate of the parent node.
        first_pos: (x, y) of the first (smallest-rank) child.
        last_pos: (x, y) of the last (largest-rank) child.

    Returns:
        List of ``(x, y)`` points.  Empty if the radius is negligibly small.
    """
    result = arc_theta_range(px, py, first_pos, last_pos)
    if result is None:
        return []

    theta1, theta2, r, n_pts = result
    step = (theta2 - theta1) / (n_pts - 1)

    return [
        (r * math.cos(theta1 + i * step), r * math.sin(theta1 + i * step)) for i in range(n_pts)
    ]
