"""Plotly marker symbol mapping and trace construction for tree nodes.

**Node styling**

Every node is styled by calling :meth:`~tralda.visualization.style.TreeStyle.resolve`. Symbol
names are mapped to Plotly marker symbols via
:data:`~tralda.visualization._symbol_defs.SYMBOL_DEF_BY_NAME`. Symbols that require orientation
(those with ``directional=True`` in the definition) are rotated via ``marker.angle``.
"""

from __future__ import annotations

import html
import math
import plotly.graph_objects as go

from tralda.datastructures.tree import TreeNode
from tralda.visualization._symbol_defs import SymbolKind, get_symbol_def
from tralda.visualization.layout import LayoutMode
from tralda.visualization.layout import TreeLayout
from tralda.visualization.plotly_renderer._utils import to_plotly_color
from tralda.visualization.style import NodeStyle


# ----------------------------------------------------------------------------------------------
# Node trace
# ----------------------------------------------------------------------------------------------


def get_node_traces(
    layout: TreeLayout,
    positions: dict[TreeNode, tuple[float, float]],
    node_styles: dict[TreeNode, NodeStyle],
    hover_attrs: list[str],
) -> list[go.Scatter]:
    """Return Scatter traces for the tree nodes.

    Handles symbol mapping, color conversion, and hover text construction.  All nodes are included
    in the main trace; 'circle-inner-ring' nodes additionally receive a secondary trace that
    overlays the inner ring on top.

    Args:
        layout: The tree layout containing the nodes.
        positions: Mapping from tree nodes to their (x, y) positions in data coordinates.
        node_styles: Mapping from tree nodes to their resolved NodeStyle objects.
        hover_attrs: List of additional node attributes to include in the hover text.

    Returns:
        A list of Scatter traces for the nodes (main trace and optional secondary trace).
    """
    mode = layout.layout_mode

    node_x: list[float] = []
    node_y: list[float] = []
    symbols: list[str] = []
    colors: list[str] = []
    sizes: list[float] = []
    edge_colors: list[str] = []
    edge_widths: list[float] = []
    angles: list[float] = []
    hover_texts: list[str] = []

    for v in layout.tree.preorder():
        ns = node_styles[v]
        x, y = positions[v]

        plotly_sym, eff_color, eff_size, eff_edge_color, eff_edge_width = _resolve_marker(ns)
        angle = _marker_angle(ns.symbol, x, y, mode)

        # Build hover text; escape user-controlled values, keep the intentional <b> wrapper.
        label = getattr(v, "label", None)
        parts: list[str] = []
        if label is not None:
            parts.append(f"<b>{html.escape(str(label))}</b>")
        parts.append(f"depth: {layout.depths[v]:.4g}")
        for attr in hover_attrs:
            val = getattr(v, attr, None)
            if val is not None:
                parts.append(f"{html.escape(str(attr))}: {html.escape(str(val))}")

        node_x.append(x)
        node_y.append(y)
        symbols.append(plotly_sym)
        colors.append(eff_color)
        sizes.append(eff_size)
        edge_colors.append(eff_edge_color)
        edge_widths.append(eff_edge_width)
        angles.append(angle)
        hover_texts.append("<br>".join(parts))

    main_node_traces = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            symbol=symbols,
            color=colors,
            size=sizes,
            line=dict(color=edge_colors, width=edge_widths),
            angle=angles,
        ),
        hovertext=hover_texts,
        hoverinfo="text",
        showlegend=False,
    )

    traces = [main_node_traces]

    secondary_node_traces = _secondary_node_traces(layout, positions, node_styles)
    if secondary_node_traces is not None:
        traces.append(secondary_node_traces)

    return traces


def _secondary_node_traces(
    layout: TreeLayout,
    positions: dict[TreeNode, tuple[float, float]],
    node_styles: dict[TreeNode, NodeStyle],
) -> go.Scatter | None:
    """Build a secondary scatter trace that draws the inner ring for 'circle-inner-ring' nodes.

    Plotly has no native composite symbol, so the inner ring is overlaid as a second ``circle-open``
    scatter trace at a smaller size.

    Args:
        layout: The tree layout containing the nodes.
        positions: Mapping from tree nodes to their (x, y) positions in data coordinates.
        node_styles: Mapping from tree nodes to their resolved NodeStyle objects.

    Returns:
        A Scatter trace for the inner rings, or None if there are no 'circle-inner-ring' nodes.
    """
    node_x: list[float] = []
    node_y: list[float] = []
    sizes: list[float] = []
    colors: list[str] = []
    lwidths: list[float] = []

    for v in layout.tree.preorder():
        ns = node_styles[v]
        if get_symbol_def(ns.symbol).name != "circle-inner-ring":
            continue
        x, y = positions[v]
        node_x.append(x)
        node_y.append(y)
        sizes.append(ns.symbol_size * 0.45)
        colors.append(to_plotly_color(ns.symbol_edge_color))
        lwidths.append(max(ns.symbol_lw * 0.8, 1.0))

    if not node_x:
        return None

    return go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        marker=dict(
            symbol="circle-open",
            color=colors,
            size=sizes,
            line=dict(color=colors, width=lwidths),
        ),
        hoverinfo="skip",
        showlegend=False,
    )


def _resolve_marker(ns: NodeStyle) -> tuple[str, str, float, str, float]:
    """Return ``(symbol, fill_color, size, edge_color, edge_width)`` for a Plotly marker.

    Handles the special ``"dot"``/``"cap"`` (edge-colored) and ``"none"`` / unknown
    (invisible zero-size marker) cases.

    Args:
        ns: Resolved :class:`~tralda.visualization.style.NodeStyle` for the node.

    Returns:
        A 5-tuple ``(plotly_symbol, fill_color, size, edge_color, edge_width)``.
    """
    defn = get_symbol_def(ns.symbol)

    if defn.kind is SymbolKind.INVISIBLE or defn.plotly_marker is None:
        invisible = "rgba(0,0,0,0)"
        return "circle", invisible, 0.0, invisible, 0.0

    plotly_sym = defn.plotly_marker

    if defn.kind is SymbolKind.EDGE_STYLE:
        c = to_plotly_color(ns.edge_color)
        if defn.name == "dot":
            # Small, edge-colored, no marker outline.
            return plotly_sym, c, ns.symbol_size * 0.35, c, 0.0
        # cap: full size, edge-colored; line weight gives the bar its stroke.
        return plotly_sym, c, ns.symbol_size, c, max(ns.edge_lw, 1.0)

    return (
        plotly_sym,
        to_plotly_color(ns.symbol_color),
        ns.symbol_size * defn.size_factor,
        to_plotly_color(ns.symbol_edge_color),
        ns.symbol_lw,
    )


def _marker_angle(symbol: str | None, x: float, y: float, mode: LayoutMode) -> float:
    """Return the Plotly ``marker.angle`` (degrees, clockwise) for a node symbol.

    Only symbols with ``directional=True`` in :data:`~tralda.visualization._symbol_defs.SYMBOL_DEFS`
    are rotated; all others return ``0``.

    Plotly's ``marker.angle`` convention: ``0`` is the symbol's default orientation, positive
    values rotate clockwise.  For ``"line-ns"`` (the cap), ``0`` is a vertical bar.

    Args:
        symbol: The tralda symbol name.
        x: Node x position in data coordinates.
        y: Node y position in data coordinates.
        mode: Current layout mode.

    Returns:
        Rotation angle in degrees (clockwise, Plotly convention).
    """
    if symbol is None:
        return 0.0
    defn = get_symbol_def(symbol)
    if not defn.directional:
        return 0.0

    if mode is LayoutMode.HORIZONTAL:
        # "line-ns" at 0° is already a vertical bar (perpendicular to horizontal edges).
        return 0.0

    if mode is LayoutMode.VERTICAL:
        # Edges run top-to-bottom; cap should be horizontal → rotate 90°.
        if defn.name == "cap":
            return 90.0
        return 0.0

    # CIRCULAR — rotate to align with the radial direction from the origin.
    # Math angle θ (CCW from east) converted to Plotly angle (CW from north); and additional 180°
    # rotation is required such that e.g. "triangle-up" points toward the root (inward).
    theta = math.degrees(math.atan2(y, x))
    plotly_angle = 270.0 - theta  # = (90° − θ) + 180°

    # Cap is perpendicular to the radial; as 0° is at north, rotate by an additional 90°.
    if defn.name == "cap":
        return plotly_angle + 90.0

    return plotly_angle
