"""Node symbols for matplotlib tree rendering.

All built-in symbols are sized in *points* so they scale consistently across DPI settings and axis
extents.  Custom symbols can be added globally with :func:`register_symbol`.

Built-in symbol names:

+------------------------------+----------------------------------------------------------------+
| Name                         | Appearance                                                     |
+==============================+================================================================+
| ``"none"``                   | Nothing drawn                                                  |
+------------------------------+----------------------------------------------------------------+
| ``"dot"``                    | Small solid circle; color and width follow the incoming edge   |
+------------------------------+----------------------------------------------------------------+
| ``"cap"``                    | Bar perpendicular to the edge; color and width follow edge     |
+------------------------------+----------------------------------------------------------------+
| ``"circle"``                 | Circle                                                         |
+------------------------------+----------------------------------------------------------------+
| ``"circle_with_dot"``        | Circle with centre dot                                         |
+------------------------------+----------------------------------------------------------------+
| ``"circle_with_inner_ring"`` | Circle with inner ring                                         |
+------------------------------+----------------------------------------------------------------+
| ``"square"``                 | Square, rotated in circular mode                               |
+------------------------------+----------------------------------------------------------------+
| ``"triangle"``               | Upward-pointing triangle, rotated in circular mode             |
+------------------------------+----------------------------------------------------------------+
| ``"triangle_down"``          | Downward-pointing triangle, rotated in circular mode           |
+------------------------------+----------------------------------------------------------------+
| ``"star"``                   | Star, rotated in circular mode                                 |
+------------------------------+----------------------------------------------------------------+
"""

from __future__ import annotations

from typing import Any, Callable
import warnings

from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from tralda.visualization.layout import LayoutMode
from tralda.visualization.style import NodeStyle

# --------------------------------------------------------------------------------------------------
# Type alias
# --------------------------------------------------------------------------------------------------

#: Type alias for symbol-drawer callables.
#:
#: A drawer must accept ``(ax, x, y, ns)`` as positional arguments, where *ns* is the fully
#: resolved :class:`~tralda.visualization.style.NodeStyle` for the node.  It must also accept
#: the keyword arguments ``angle`` and ``layout_mode``.  Any extra keyword arguments should be
#: accepted and silently ignored via ``**kwargs``.  *ns.symbol_size* is in points (like
#: ``markersize``).  *angle* is the edge-direction angle in degrees (0 = right, CCW positive);
#: drawers that are invariant under rotation may ignore it.
#:
#: ``dot`` and ``cap`` are *edge-style* symbols: they intentionally use ``ns.edge_color`` and
#: ``ns.edge_lw`` instead of the ``ns.symbol_*`` fields so they visually blend into the edge line.
SymbolDrawer = Callable[..., None]


# --------------------------------------------------------------------------------------------------
# Built-in symbol drawers
# --------------------------------------------------------------------------------------------------
# Signature: (ax, x, y, ns, *, angle, layout_mode, **kwargs) -> None

# --------------------------------------------------------------------------------------------------
# Structural fallbacks
# --------------------------------------------------------------------------------------------------


def _draw_none(ax: Axes, x: float, y: float, ns: NodeStyle, **_: Any) -> None:
    """Draw nothing."""


# --------------------------------------------------------------------------------------------------
# Edge-style symbols (color and size follow the incoming edge, not the symbol style)
# --------------------------------------------------------------------------------------------------


def _draw_dot(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Small solid circle that follows the incoming-edge color and width."""
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size * 0.35,
        mfc=ns.edge_color,
        mec="none",
        ls="none",
        zorder=ns.symbol_zorder,
    )


def _draw_cap(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Bar perpendicular to the edge direction, colored and sized like the incoming edge."""
    marker = MarkerStyle("|", transform=Affine2D().rotate_deg(angle))
    ax.plot(
        x,
        y,
        marker=marker,
        ms=ns.symbol_size,
        mec=ns.edge_color,
        mew=max(ns.edge_lw, 1.0),  # ensure visibility even for very thin edges
        ls="none",
        zorder=ns.symbol_zorder,
    )


# --------------------------------------------------------------------------------------------------
# Circles
# --------------------------------------------------------------------------------------------------


def _draw_circle(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Solid filled circle."""
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )


def _draw_circle_with_dot(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Circle with a small centre dot."""
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size / 3.5,
        mfc=ns.symbol_edge_color,
        mec="none",
        ls="none",
        zorder=ns.symbol_zorder + 0.1,
    )


def _draw_circle_with_inner_ring(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Circle with an inner ring."""
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )
    ax.plot(
        x,
        y,
        "o",
        ms=ns.symbol_size * 0.45,
        mfc="none",
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw * 0.8,
        ls="none",
        zorder=ns.symbol_zorder + 0.1,
    )


# --------------------------------------------------------------------------------------------------
# Squares
# --------------------------------------------------------------------------------------------------


def _draw_square(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Empty square with a border, rotated to align with the edge direction."""
    marker = MarkerStyle("s", transform=Affine2D().rotate_deg(angle))
    ax.plot(
        x,
        y,
        marker=marker,
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )


# --------------------------------------------------------------------------------------------------
# Other shapes
# --------------------------------------------------------------------------------------------------


def _draw_triangle(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Upward-pointing triangle, rotated to align with the edge direction in circular mode."""
    if layout_mode is LayoutMode.CIRCULAR:
        marker = MarkerStyle("<", transform=Affine2D().rotate_deg(angle))
    else:
        marker = "^"
    ax.plot(
        x,
        y,
        marker=marker,
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )


def _draw_triangle_down(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Downward-pointing triangle, rotated to align with the edge direction in circular mode."""
    if layout_mode is LayoutMode.CIRCULAR:
        marker = MarkerStyle(">", transform=Affine2D().rotate_deg(angle))
    else:
        marker = "v"
    ax.plot(
        x,
        y,
        marker=marker,
        ms=ns.symbol_size,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )


def _draw_star(
    ax: Axes,
    x: float,
    y: float,
    ns: NodeStyle,
    *,
    angle: float = 0.0,
    layout_mode: LayoutMode | None = None,
    **_: Any,
) -> None:
    """Star, rotated to align with the edge direction in circular mode."""
    if layout_mode is LayoutMode.CIRCULAR:
        marker = MarkerStyle("*", transform=Affine2D().rotate_deg(angle + 90.0))
    else:
        marker = "*"
    ax.plot(
        x,
        y,
        marker=marker,
        ms=ns.symbol_size * 1.3,
        mfc=ns.symbol_color,
        mec=ns.symbol_edge_color,
        mew=ns.symbol_lw,
        ls="none",
        zorder=ns.symbol_zorder,
    )


# --------------------------------------------------------------------------------------------------
# Module-level symbol registry
# --------------------------------------------------------------------------------------------------


class _SymbolRegistry(dict[str, SymbolDrawer]):
    """Module-level registry for symbol drawers.

    This is used by :func:`register_symbol` to add custom symbols and by :class:`MatplotlibRenderer`
    to look up symbol drawers by name.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in symbols."""
        super().__init__()

        # add built-in symbols to the registry
        self.update(
            {
                "none": _draw_none,
                "dot": _draw_dot,
                "cap": _draw_cap,
                "circle": _draw_circle,
                "circle_with_dot": _draw_circle_with_dot,
                "circle_with_inner_ring": _draw_circle_with_inner_ring,
                "square": _draw_square,
                "triangle": _draw_triangle,
                "triangle_down": _draw_triangle_down,
                "star": _draw_star,
            }
        )

    def __getitem__(self, key: str) -> SymbolDrawer:
        """Look up a symbol drawer by name, with a warning for missing symbols.

        If the requested symbol name is not found in the registry, a warning is issued and the
        'none' drawer is returned, which results in no symbol being drawn.

        Args:
            key: Symbol name to look up.

        Returns:
            The corresponding symbol drawer if found; otherwise, the 'none' drawer.
        """
        if key not in self:
            warnings.warn(f"Symbol '{key}' is not registered; using 'none' instead", stacklevel=2)
            return self["none"]

        return super().__getitem__(key)

    def __setitem__(self, key: str, value: SymbolDrawer) -> None:
        """Register a new symbol drawer, with a check for duplicate names.

        Args:
            key: Symbol name under which to register the drawer.
            value: The symbol drawer to register.

        Raises:
            KeyError: If a symbol with the given name is already registered.
        """
        if key in self:
            raise KeyError(f"Symbol '{key}' is already registered")

        super().__setitem__(key, value)


SYMBOL_REGISTRY = _SymbolRegistry()


def register_symbol(name: str, drawer: SymbolDrawer) -> None:
    """Register a custom symbol drawer in the module-level registry.

    Once registered, the symbol is immediately available to all :class:`MatplotlibRenderer`
    instances by name.

    Args:
        name: Key under which the drawer is registered.
        drawer: Callable with the signature
            ``(ax, x, y, ns, *, angle, layout_mode, **kwargs) -> None``, where *ns* is the
            fully-resolved :class:`~tralda.visualization.style.NodeStyle` for the node.
            *ns.symbol_size* is in points (like ``markersize``).  *angle* is the edge-direction
            angle in degrees (0 = right, CCW positive); drawers invariant under rotation may
            ignore it.
            *layout_mode* is the current :class:`~tralda.visualization.layout.LayoutMode`.
    """
    SYMBOL_REGISTRY[name] = drawer
