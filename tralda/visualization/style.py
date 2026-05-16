"""Node and tree styling for tree rendering.

Styling is separated from rendering so that the same :class:`TreeStyle` can be applied to
different layouts or reused across figures.

**Architecture**

:class:`NodeStyle` is a plain dataclass where every field defaults to ``None``, meaning "inherit
from the layer below".  This makes it a *partial* override rather than a full specification.

:class:`TreeStyle` holds:

* **Default style** — a fully resolved :class:`NodeStyle` (no ``None`` fields) that applies to
  every node when nothing overrides it.
* **Style function** — an optional ``Callable[[TreeNode, LayoutMode], NodeStyle | None]`` called
  once per node.  Return ``None`` or a partially-filled :class:`NodeStyle`; only non-``None``
  fields override the default.
* **Node overrides** — an optional ``dict[TreeNode, NodeStyle]`` for surgical per-node tweaks,
  applied last (highest priority).

Resolution order (later wins, ``None`` fields are skipped):

    default  →  style_fn(node)  →  node_overrides[node]

Convenience class methods on :class:`TreeStyle` accept ``symbol_map`` / ``color_map`` dicts and
translate them into a style function.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Callable, Iterable

from tralda.datastructures.tree import TreeNode
from tralda.visualization.layout import LayoutMode


# --------------------------------------------------------------------------------------------------
# NodeStyle
# --------------------------------------------------------------------------------------------------


@dataclass
class NodeStyle:
    """Partial style specification for a single tree node.

    Every field defaults to ``None``, which means "inherit from the layer below".  Pass a
    :class:`NodeStyle` from a style function or node-override dict to override only the fields you
    care about.

    **Symbol fields**

    symbol
        Name of the symbol to draw (must be registered in :data:`~._symbols.SYMBOL_REGISTRY`).
    symbol_size
        Diameter of the symbol in points.
    symbol_color
        Fill color of the symbol (passed as *color* to the drawer).
    symbol_edge_color
        Border color of the symbol (passed as *edge_color* to the drawer).
    symbol_lw
        Border line width of the symbol in points.
    symbol_zorder
        Z-order of the symbol layer.

    **Incoming-edge fields (edge from parent to this node)**

    edge_color
        Color of the edge from this node's parent.
    edge_lw
        Line width of the incoming edge in points.
    edge_ls
        Line style of the incoming edge (e.g. ``"-"``, ``"--"``, ``":"``, ``"-.\"``).

    **Label fields**

    label_color
        Color of the leaf label text.
    label_fontsize
        Font size in points, or a named size string (e.g. ``"small"``).
    label_fontweight
        Font weight: ``"normal"`` or ``"bold"``.
    label_fontstyle
        Font style: ``"normal"`` or ``"italic"``.
    """

    # symbol
    symbol: str | None = None
    symbol_size: float | None = None
    symbol_color: Any = None
    symbol_edge_color: str | None = None
    symbol_lw: float | None = None
    symbol_zorder: int | None = None
    # incoming edge
    edge_color: Any | None = None
    edge_lw: float | None = None
    edge_ls: str | None = None
    # label
    label_color: Any | None = None
    label_fontsize: float | str | None = None
    label_fontweight: str | None = None
    label_fontstyle: str | None = None

    def overlay(self, other: NodeStyle) -> NodeStyle:
        """Return a new :class:`NodeStyle` with *other*'s non-``None`` fields applied over self.

        Args:
            other: The higher-priority partial style.

        Returns:
            A new :class:`NodeStyle` merging both layers.
        """
        merged = NodeStyle(**{f.name: getattr(self, f.name) for f in fields(self)})
        for f in fields(other):
            v = getattr(other, f.name)
            if v is not None:
                setattr(merged, f.name, v)
        return merged


#: Baseline :class:`NodeStyle` used when no *default* is passed to :class:`TreeStyle`.
#: Every field is fully resolved (no ``None`` values).  Use this as a reference when constructing
#: a partial :class:`NodeStyle` to pass as *default*.
DEFAULT_NODE_STYLE = NodeStyle(
    symbol=None,  # resolved per-node via TreeStyle.root_symbol / leaf_symbol / internal_symbol
    symbol_size=9.0,
    symbol_color="white",
    symbol_edge_color="black",
    symbol_lw=1.0,
    symbol_zorder=3,
    edge_color="black",
    edge_lw=1.0,
    edge_ls="-",
    label_color="black",
    label_fontsize=9,
    label_fontweight="normal",
    label_fontstyle="normal",
)


# --------------------------------------------------------------------------------------------------
# TreeStyle
# --------------------------------------------------------------------------------------------------

#: Type alias for style functions.
StyleFn = Callable[[TreeNode, LayoutMode], "NodeStyle | None"]


@dataclass
class TreeStyle:
    """Resolved styling configuration for an entire tree.

    Attributes:
        default: Default :class:`NodeStyle` applied to every node before any per-node overrides.
            Pass a *partial* :class:`NodeStyle` (only the fields you want to change) and it will
            be overlaid onto :data:`DEFAULT_NODE_STYLE` in ``__post_init__``, so omitted fields
            keep their standard values.  After construction, ``default`` is always fully resolved
            (no ``None`` fields).
        root_symbol: Symbol name used for the root node when the style function and node-override
            dict do not specify one.  Default ``"circle_with_inner_ring"``.
        leaf_symbol: Symbol name used for leaf nodes when not otherwise specified.
            Default ``"circle_with_dot"``.
        internal_symbol: Symbol name used for internal (non-root) nodes when not otherwise
            specified.  Default ``"dot"``.
        ghost_color: Color of ghost (leaf-extension) segments.  Default ``"grey"``.
        ghost_lw: Line width of ghost segments.  Default ``0.6``.
        ghost_ls: Line style of ghost segments.  Default ``"--"``.
        style_fn: Optional callable ``(node, layout_mode) -> NodeStyle | None`` called once per
            node.  Return ``None`` or a partial :class:`NodeStyle`; only non-``None`` fields
            override the default.
        node_overrides: Optional per-node overrides applied after *style_fn* (highest priority).
    """

    # ── tree-wide defaults ─────────────────────────────────────────────────────────────────────
    default: NodeStyle = field(default=None)  # type: ignore[assignment]  resolved in __post_init__

    # ── structural symbol fallbacks ────────────────────────────────────────────────────────────
    root_symbol: str = "none"
    leaf_symbol: str = "none"
    internal_symbol: str = "none"

    # ── ghost segment styling ──────────────────────────────────────────────────────────────────
    ghost_color: str = "grey"
    ghost_lw: float = 0.6
    ghost_ls: str = "--"

    # ── override layers ────────────────────────────────────────────────────────────────────────
    style_fn: StyleFn | None = None
    node_overrides: dict[TreeNode, NodeStyle] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default is None:
            self.default = DEFAULT_NODE_STYLE.overlay(NodeStyle())  # fresh copy
        else:
            self.default = DEFAULT_NODE_STYLE.overlay(self.default)

    # ------------------------------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------------------------------

    def resolve(self, node: TreeNode, layout_mode: LayoutMode) -> NodeStyle:
        """Return the fully-resolved :class:`NodeStyle` for *node*.

        Applies the default, then the style function, then any direct node override.  The
        ``symbol`` field is filled from the structural fallbacks when it is still ``None`` after all
        layers have been merged.

        Args:
            node: The node to resolve styling for.
            layout_mode: Current layout mode, passed to :attr:`style_fn`.

        Returns:
            A :class:`NodeStyle` with no ``None`` fields.
        """
        style = self.default

        # Apply style function overrides.
        if self.style_fn is not None:
            override = self.style_fn(node, layout_mode)
            if override is not None:
                style = style.overlay(override)

        # Apply direct node overrides.
        if node in self.node_overrides:
            style = style.overlay(self.node_overrides[node])

        # Fill in structural symbol fallback if still unresolved.
        if style.symbol is None:
            if node.parent is None:
                sym = self.root_symbol
            elif not node.children:
                sym = self.leaf_symbol
            else:
                sym = self.internal_symbol
            style = style.overlay(NodeStyle(symbol=sym))

        return style

    def consensus_style(self, nodes: Iterable[TreeNode], layout_mode: LayoutMode) -> NodeStyle:
        """Return a consensus style for a set of nodes (with fallback to default).

        If all nodes agree on a particular field (e.g. all have the same symbol or symbol color),
        the consensus style will use that value.  If there's any disagreement, the consensus style
        falls back to the default for that field.  This is useful for styling internal nodes or
        edges based on the styles of their children (e.g. for styling the child connector line / arc
        based on the styles of the child nodes).

        Args:
            nodes: The nodes to find a consensus style for.
            layout_mode: Current layout mode, passed to :attr:`style_fn`.

        Returns:
            A :class:`NodeStyle` representing the consensus style.
        """
        consensus = self.default.overlay(NodeStyle())  # start with a fresh copy of the default

        styles = [self.resolve(n, layout_mode) for n in nodes]
        if not styles:
            return consensus

        # For each field, check if all styles agree on the same non-None value.  If so, use it in
        # the consensus; otherwise leave the default.
        for f in fields(NodeStyle):
            values = [getattr(s, f.name) for s in styles]
            if values[0] is not None and all(_safe_eq(v, values[0]) for v in values):
                setattr(consensus, f.name, values[0])

        return consensus

    # ------------------------------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------------------------------

    @classmethod
    def from_maps(
        cls,
        *,
        style_attr: str = "style",
        style_map: dict[str, NodeStyle] | None = None,
        symbol_attr: str = "symbol",
        symbol_map: dict[str, str] | None = None,
        node_color_attr: str = "color",
        node_color_map: dict[Any, Any] | None = None,
        edge_color_attr: str = "edge_color",
        edge_color_map: dict[Any, Any] | None = None,
        **kwargs: Any,
    ) -> TreeStyle:
        """Construct a :class:`TreeStyle` from attribute lookup maps.

        This is a convenience constructor for the common case where styling is determined by
        looking up node attributes in provided maps.  It translates the maps into a style function
        that applies the specified lookups and falls back to the default styling when no match is
        found.  The kwargs are forwarded to the :class:`TreeStyle` constructor, so you can still
        set defaults and other options.

        If provided, *style_map* is applied first (lowest priority). The other maps may override
        symbols and colors specified in *style_map*.

        Args:
            style_attr: Node attribute whose value is looked up in *style_map* to get a full or
                partial :class:`NodeStyle` for the node.  Default ``"style"``.
            style_map: Mapping from *style_attr* values to :class:`NodeStyle` instances.
            symbol_attr: Node attribute whose value is looked up in *symbol_map*.
                Default ``"symbol"``.
            symbol_map: Mapping from *symbol_attr* values to symbol names.  Nodes not in the map
                use the structural fallbacks.
            node_color_attr: Node attribute whose value is looked up in *node_color_map*.
                Default ``"color"``.
            node_color_map: Mapping from *node_color_attr* values to fill colors.
            edge_color_attr: Node attribute whose value is looked up in *edge_color_map*.
                Default ``"edge_color"``.
            edge_color_map: Mapping from *edge_color_attr* values to edge colors.
            **kwargs: Forwarded to the :class:`TreeStyle` constructor (e.g. ``root_symbol``,
                ``ghost_color``, ``default`` for a custom base :class:`NodeStyle`, etc.).

        Returns:
            A new :class:`TreeStyle` with a style function encoding the provided maps.
        """
        _style_map: dict[str, NodeStyle] = style_map or {}
        _symbol_map: dict[str, str] = symbol_map or {}
        _node_col_map: dict[Any, Any] = node_color_map or {}
        _edge_col_map: dict[Any, Any] = edge_color_map or {}

        def _style_fn(node: TreeNode, _mode: LayoutMode) -> NodeStyle | None:
            ns = NodeStyle()

            # Style from style attribute.
            val = getattr(node, style_attr, None)
            if val is not None and val in _style_map:
                ns = ns.overlay(_style_map[val])

            # Symbol from symbol attribute.
            ev = getattr(node, symbol_attr, None)
            if ev is not None and ev in _symbol_map:
                ns.symbol = _symbol_map[ev]

            # Fill color from color attribute.
            key = getattr(node, node_color_attr, None)
            if key is not None and key in _node_col_map:
                ns.symbol_color = _node_col_map[key]

            # Edge color from edge color attribute.
            key = getattr(node, edge_color_attr, None)
            if key is not None and key in _edge_col_map:
                ns.edge_color = _edge_col_map[key]

            # Return None if nothing was customised (avoids a no-op overlay).
            if ns.symbol is None and ns.symbol_color is None and ns.edge_color is None:
                return None

            return ns

        return cls(style_fn=_style_fn, **kwargs)


# --------------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------------


def _safe_eq(a: object, b: object) -> bool:
    """Return True if *a* and *b* are equal, handling array-valued colours safely.

    Plain ``a == b`` raises ``ValueError`` when either operand is a numpy array (or similar
    sequence), because the result is an array of bools rather than a scalar.  This helper
    iterates the result when it is iterable, falling back to ``False`` on any error.
    """
    try:
        result = a == b
        if hasattr(result, "__iter__"):
            return all(result)
        return bool(result)
    except (TypeError, ValueError):
        return False
