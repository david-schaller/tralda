"""Node symbols for matplotlib tree rendering.

All built-in symbols are sized in *points* so they scale consistently across DPI settings and axis
extents.  Custom symbols can be added globally with :func:`register_symbol`.

The canonical list of built-in symbol names and their cross-renderer properties is maintained in
:mod:`~tralda.visualization._symbol_defs`.  Simple symbols (``SymbolKind.SIMPLE``) are drawn by
auto-generated drawers produced by :func:`_make_simple_drawer`; no per-symbol boilerplate is
needed.  Composite and edge-style symbols have explicit drawers defined below.
"""

from __future__ import annotations

import warnings
from typing import Any, Callable

from matplotlib.axes import Axes
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D

from tralda.visualization._symbol_defs import SYMBOL_DEFS, SymbolDef, SymbolKind, resolve_symbol
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
# Edge-style and composite drawers (explicit per-renderer logic)
# --------------------------------------------------------------------------------------------------
# Signature: (ax, x, y, ns, *, angle, layout_mode, **kwargs) -> None


def _draw_none(ax: Axes, x: float, y: float, ns: NodeStyle, **_: Any) -> None:
    """Draw nothing."""


# Edge-style symbols: color and size follow the incoming edge, not the symbol_* fields.


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


# Composite symbols: two ax.plot calls to achieve the full visual.


def _draw_circle_dot(
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


def _draw_circle_inner_ring(
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
# Explicit drawers for INVISIBLE / EDGE_STYLE / COMPOSITE symbols
# --------------------------------------------------------------------------------------------------

_SPECIAL_DRAWERS: dict[str, SymbolDrawer] = {
    "none": _draw_none,
    "dot": _draw_dot,
    "cap": _draw_cap,
    "circle-dot": _draw_circle_dot,
    "circle-inner-ring": _draw_circle_inner_ring,
}


# --------------------------------------------------------------------------------------------------
# Factory for SIMPLE symbols
# --------------------------------------------------------------------------------------------------


def _make_simple_drawer(defn: SymbolDef) -> SymbolDrawer:
    """Return a :data:`SymbolDrawer` for a ``SIMPLE`` symbol definition.

    The generated drawer uses a single ``ax.plot`` call.  In CIRCULAR mode the marker is wrapped
    in a :class:`~matplotlib.markers.MarkerStyle` transform to rotate it toward the radial
    direction; *mpl_circ_marker* and *mpl_circ_angle_offset* from the definition are applied.

    Args:
        defn: A :class:`~tralda.visualization._symbol_defs.SymbolDef` with
            ``kind == SymbolKind.SIMPLE``.

    Returns:
        A callable matching the :data:`SymbolDrawer` protocol.
    """

    def draw(
        ax: Axes,
        x: float,
        y: float,
        ns: NodeStyle,
        *,
        angle: float = 0.0,
        layout_mode: LayoutMode | None = None,
        **_: Any,
    ) -> None:
        size = ns.symbol_size * defn.size_factor
        if defn.directional and layout_mode is LayoutMode.CIRCULAR:
            # An extra 90° rotates the marker's natural upward orientation to face
            # radially inward, which is the correct convention for all directional symbols.
            marker: Any = MarkerStyle(
                defn.mpl_marker, transform=Affine2D().rotate_deg(angle + 90.0)
            )
        else:
            marker = defn.mpl_marker
        ax.plot(
            x,
            y,
            marker=marker,
            ms=size,
            mfc=ns.symbol_color,
            mec=ns.symbol_edge_color,
            mew=ns.symbol_lw,
            ls="none",
            zorder=ns.symbol_zorder,
        )

    return draw


# --------------------------------------------------------------------------------------------------
# Module-level symbol registry
# --------------------------------------------------------------------------------------------------


class _SymbolRegistry(dict[str, SymbolDrawer]):
    """Module-level registry for symbol drawers.

    This is used by :func:`register_symbol` to add custom symbols and by :class:`MatplotlibRenderer`
    to look up symbol drawers by name.
    """

    def __init__(self) -> None:
        """Initialize the registry with built-in symbols derived from :data:`SYMBOL_DEFS`."""
        super().__init__()
        for defn in SYMBOL_DEFS:
            if defn.kind is SymbolKind.SIMPLE:
                self[defn.name] = _make_simple_drawer(defn)
            elif defn.name in _SPECIAL_DRAWERS:
                self[defn.name] = _SPECIAL_DRAWERS[defn.name]

    def __getitem__(self, key: str) -> SymbolDrawer:
        """Look up a symbol drawer by name, with a warning for missing symbols.

        Accepts tralda names, matplotlib marker strings (e.g. ``"^"``), and Plotly marker
        names (e.g. ``"line-ns"``); all are resolved to the tralda canonical name first via
        :func:`~tralda.visualization._symbol_defs.resolve_symbol`.

        If the resolved name is not found in the registry, a warning is issued and the
        'none' drawer is returned, which results in no symbol being drawn.

        Args:
            key: Symbol name to look up.

        Returns:
            The corresponding symbol drawer if found; otherwise, the 'none' drawer.
        """
        key = resolve_symbol(key)
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
