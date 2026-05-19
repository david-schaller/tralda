"""Shared symbol definitions for all tralda tree renderers.

This module provides the canonical cross-renderer symbol table.  Every built-in symbol is
described by a :class:`SymbolDef` entry that carries its tralda name, matplotlib marker
string(s), Plotly marker string, and rendering metadata.

**Extending the symbol set**

To add a new *simple* symbol (single marker, standard styling) for all existing renderers,
append one :class:`SymbolDef` row with ``kind=SymbolKind.SIMPLE`` to :data:`SYMBOL_DEFS`.
Both the matplotlib and Plotly renderers pick it up automatically — no further code changes
are needed.

Composite or edge-style symbols require additional per-renderer handling.  Use the existing
``COMPOSITE`` and ``EDGE_STYLE`` entries as a guide and add the corresponding special logic
in each renderer's ``_symbols.py``.

**Symbol name resolution**

The :func:`resolve_symbol` function normalises any of the following notations to the tralda
canonical name, so users can write whichever they are most familiar with:

* **Tralda names** — ``"triangle-up"``, ``"star"``, …  (always accepted)
* **Matplotlib marker strings** — ``"^"``, ``"*"``, ``"|"``, …
* **Plotly marker names** — ``"line-ns"``, ``"triangle-up"``, …  (most already match tralda)

For ambiguous matplotlib markers (``"o"`` is used by ``"circle"``, ``"dot"``,
``"circle-dot"``, and ``"circle-inner-ring"``) the ``SIMPLE`` symbol takes priority.

**Symbol kinds**

+-------------------+-------------------------------------------------------------------+
| Kind              | Description                                                       |
+===================+===================================================================+
| ``INVISIBLE``     | Nothing is drawn (``"none"``).                                    |
+-------------------+-------------------------------------------------------------------+
| ``EDGE_STYLE``    | Color and width follow the incoming edge rather than the          |
|                   | ``symbol_*`` style fields (``"dot"``, ``"cap"``).                 |
+-------------------+-------------------------------------------------------------------+
| ``SIMPLE``        | Single marker drawn with standard ``symbol_*`` style fields.      |
|                   | Both renderers generate drawing code automatically from the table.|
+-------------------+-------------------------------------------------------------------+
| ``COMPOSITE``     | Requires an additional draw call / trace for the full visual      |
|                   | effect (``"circle-dot"``, ``"circle-inner-ring"``).  The primary  |
|                   | marker in the table describes the *outer* shape; the overlay is   |
|                   | handled per-renderer.                                             |
+-------------------+-------------------------------------------------------------------+

**Built-in symbol table**

+------------------------------+--------+-----------+------------------+------+-----+
| Name                         | Kind   | mpl       | plotly           |  sf  | dir |
+==============================+========+===========+==================+======+=====+
| ``"none"``                   | INV    | —         | —                | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"dot"``                    | EDGE   | ``"o"``   | ``"circle"``     | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"cap"``                    | EDGE   | ``"|"``   | ``"line-ns"``    | 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"circle"``                 | SIMPLE | ``"o"``   | ``"circle"``     | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"square"``                 | SIMPLE | ``"s"``   | ``"square"``     | 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"triangle-up"``            | SIMPLE | ``"^"``   | ``"triangle-up"``| 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"triangle-down"``          | SIMPLE | ``"v"``   | ``"triangle-dn"``| 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"triangle-left"``          | SIMPLE | ``"<"``   | ``"triangle-lft"``| 1.0 | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"triangle-right"``         | SIMPLE | ``">"``   | ``"triangle-rt"``| 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"star"``                   | SIMPLE | ``"*"``   | ``"star"``       | 1.3  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"diamond"``                | SIMPLE | ``"D"``   | ``"diamond"``    | 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"pentagon"``               | SIMPLE | ``"p"``   | ``"pentagon"``   | 1.0  | Yes |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"hexagon"``                | SIMPLE | ``"h"``   | ``"hexagon"``    | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"cross"``                  | SIMPLE | ``"P"``   | ``"cross"``      | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"x-mark"``                 | SIMPLE | ``"X"``   | ``"x"``          | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"circle-dot"``             | COMP   | ``"o"``   | ``"circle-dot"`` | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+
| ``"circle-inner-ring"``      | COMP   | ``"o"``   | ``"circle"``     | 1.0  | No  |
+------------------------------+--------+-----------+------------------+------+-----+

*(sf = size_factor, dir = directional; directional symbols are rotated by an additional 90°
in CIRCULAR mode so that their "natural up" orientation faces radially inward)*
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto


# --------------------------------------------------------------------------------------------------
# SymbolKind
# --------------------------------------------------------------------------------------------------


class SymbolKind(Enum):
    """Rendering category that determines how each renderer draws the symbol."""

    INVISIBLE = auto()  #: Nothing is drawn.
    EDGE_STYLE = auto()  #: Color/width follow the incoming edge (``"dot"``, ``"cap"``).
    SIMPLE = auto()  #: Single marker; standard ``symbol_*`` style fields.
    COMPOSITE = auto()  #: Extra draw call/trace needed (``"circle-dot"``, ``"circle-inner-ring"``).


# --------------------------------------------------------------------------------------------------
# SymbolDef
# --------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SymbolDef:
    """Cross-renderer definition for a tree-node symbol.

    Attributes:
        name: Tralda canonical symbol name.
        kind: Rendering category; determines how each renderer draws the symbol.
        mpl_marker: Matplotlib marker string used in all layout modes.
        plotly_marker: Plotly marker symbol name, or ``None`` for invisible markers.
        size_factor: Multiplier applied to ``ns.symbol_size`` before drawing.
        directional: If ``True``, the symbol is rotated to face the radial direction in CIRCULAR
            mode.  Matplotlib drawers apply an additional 90° offset so that the marker's natural
            upward orientation points radially inward.
    """

    name: str
    kind: SymbolKind
    mpl_marker: str | None = None
    plotly_marker: str | None = None
    size_factor: float = 1.0
    directional: bool = False


# --------------------------------------------------------------------------------------------------
# Built-in symbol table
# --------------------------------------------------------------------------------------------------


#: Ordered list of all built-in symbol definitions.
#:
#: Add a new row here to register a new *simple* symbol globally.  Composite and edge-style
#: symbols also require additional per-renderer handling in each ``_symbols.py``.
SYMBOL_DEFS: list[SymbolDef] = [
    # ── invisible ──────────────────────────────────────────────────────────────────────────────
    SymbolDef(name="none", kind=SymbolKind.INVISIBLE),
    # ── edge-style ─────────────────────────────────────────────────────────────────────────────
    SymbolDef(name="dot", kind=SymbolKind.EDGE_STYLE, mpl_marker="o", plotly_marker="circle"),
    SymbolDef(
        name="cap",
        kind=SymbolKind.EDGE_STYLE,
        mpl_marker="|",
        plotly_marker="line-ns",
        directional=True,
    ),
    # ── simple ─────────────────────────────────────────────────────────────────────────────────
    SymbolDef(name="circle", kind=SymbolKind.SIMPLE, mpl_marker="o", plotly_marker="circle"),
    SymbolDef(
        name="square",
        kind=SymbolKind.SIMPLE,
        mpl_marker="s",
        plotly_marker="square",
        directional=True,
    ),
    SymbolDef(
        name="triangle-up",
        kind=SymbolKind.SIMPLE,
        mpl_marker="^",
        plotly_marker="triangle-up",
        directional=True,
    ),
    SymbolDef(
        name="triangle-down",
        kind=SymbolKind.SIMPLE,
        mpl_marker="v",
        plotly_marker="triangle-down",
        directional=True,
    ),
    SymbolDef(
        name="triangle-left",
        kind=SymbolKind.SIMPLE,
        mpl_marker="<",
        plotly_marker="triangle-left",
        directional=True,
    ),
    SymbolDef(
        name="triangle-right",
        kind=SymbolKind.SIMPLE,
        mpl_marker=">",
        plotly_marker="triangle-right",
        directional=True,
    ),
    SymbolDef(
        name="star",
        kind=SymbolKind.SIMPLE,
        mpl_marker="*",
        plotly_marker="star",
        size_factor=1.3,
        directional=True,
    ),
    SymbolDef(
        name="diamond",
        kind=SymbolKind.SIMPLE,
        mpl_marker="D",
        plotly_marker="diamond",
        directional=True,
    ),
    SymbolDef(
        name="pentagon",
        kind=SymbolKind.SIMPLE,
        mpl_marker="p",
        plotly_marker="pentagon",
        directional=True,
    ),
    SymbolDef(name="hexagon", kind=SymbolKind.SIMPLE, mpl_marker="h", plotly_marker="hexagon"),
    SymbolDef(name="cross", kind=SymbolKind.SIMPLE, mpl_marker="P", plotly_marker="cross"),
    SymbolDef(name="x-mark", kind=SymbolKind.SIMPLE, mpl_marker="X", plotly_marker="x"),
    # ── composite ──────────────────────────────────────────────────────────────────────────────
    SymbolDef(
        name="circle-dot", kind=SymbolKind.COMPOSITE, mpl_marker="o", plotly_marker="circle-dot"
    ),
    SymbolDef(
        name="circle-inner-ring", kind=SymbolKind.COMPOSITE, mpl_marker="o", plotly_marker="circle"
    ),
]


#: Mapping from symbol name to :class:`SymbolDef`, for O(1) lookup.
SYMBOL_DEF_BY_NAME: dict[str, SymbolDef] = {d.name: d for d in SYMBOL_DEFS}


# --------------------------------------------------------------------------------------------------
# Reverse lookup dicts and symbol resolution
# --------------------------------------------------------------------------------------------------
# For ambiguous marker strings that are shared by multiple tralda symbols (e.g. mpl "o" is used
# by "circle", "dot", "circle-dot", and "circle-inner-ring"), SIMPLE takes priority over
# EDGE_STYLE, which takes priority over COMPOSITE.  Lower-priority entries are inserted first
# so that higher-priority entries overwrite them.
_KIND_PRIORITY: dict[SymbolKind, int] = {
    SymbolKind.INVISIBLE: 0,
    SymbolKind.COMPOSITE: 1,
    SymbolKind.EDGE_STYLE: 2,
    SymbolKind.SIMPLE: 3,  # highest priority — wins all collisions
}

#: Reverse lookup: matplotlib marker string → tralda canonical symbol name.
#: For ambiguous strings (e.g. ``"o"``), the ``SIMPLE`` symbol (``"circle"``) takes priority.
MPL_TO_TRALDA: dict[str, str] = {
    d.mpl_marker: d.name
    for d in sorted(SYMBOL_DEFS, key=lambda d: _KIND_PRIORITY[d.kind])
    if d.mpl_marker is not None
}

#: Reverse lookup: Plotly marker name → tralda canonical symbol name.
#: For collisions, the ``SIMPLE`` symbol takes priority.
PLOTLY_TO_TRALDA: dict[str, str] = {
    d.plotly_marker: d.name
    for d in sorted(SYMBOL_DEFS, key=lambda d: _KIND_PRIORITY[d.kind])
    if d.plotly_marker is not None
}


def resolve_symbol(name: str) -> str:
    """Resolve any renderer-specific marker notation to the tralda canonical symbol name.

    Accepts tralda names (e.g. ``"triangle-up"``), matplotlib marker strings
    (e.g. ``"^"``), and Plotly marker names (e.g. ``"line-ns"``).  Lookup order:

    1. If *name* is already a tralda canonical name, return it unchanged.
    2. If *name* is a matplotlib marker string, return the corresponding tralda name.
    3. If *name* is a Plotly marker name, return the corresponding tralda name.
    4. Return *name* unchanged (unknown symbols are handled by each renderer's
       warning system).

    Args:
        name: A symbol identifier in any supported notation.

    Returns:
        The tralda canonical symbol name, or *name* if no match is found.
    """
    if name in SYMBOL_DEF_BY_NAME:
        return name
    return MPL_TO_TRALDA.get(name) or PLOTLY_TO_TRALDA.get(name) or name


def get_symbol_def(name: str | None) -> SymbolDef:
    """Resolve *name* and return the corresponding :class:`SymbolDef`.

    Combines :func:`resolve_symbol` and the :data:`SYMBOL_DEF_BY_NAME` lookup into a
    single call.  Accepts ``None`` and unknown names, both of which fall back to the
    ``"none"`` entry (invisible symbol) rather than returning ``None``.

    Args:
        name: A symbol identifier in any supported notation, or ``None``.

    Returns:
        The :class:`SymbolDef` for the resolved tralda name, or the ``"none"``
        :class:`SymbolDef` if *name* is ``None`` or does not match any registered symbol.
    """
    _none = SYMBOL_DEF_BY_NAME["none"]
    if not name:
        return _none
    return SYMBOL_DEF_BY_NAME.get(resolve_symbol(name), _none)


# --------------------------------------------------------------------------------------------------
# Linestyle resolution
# --------------------------------------------------------------------------------------------------

#: Mapping from matplotlib linestyle shorthand to the canonical Plotly dash name.
#:
#: +-----------+-------------+
#: | mpl       | Plotly      |
#: +===========+=============+
#: | ``"-"``   | ``"solid"`` |
#: +-----------+-------------+
#: | ``"--"``  | ``"dash"``  |
#: +-----------+-------------+
#: | ``":"``   | ``"dot"``   |
#: +-----------+-------------+
#: | ``"-."``  |``"dashdot"``|
#: +-----------+-------------+
MPL_TO_DASH: dict[str, str] = {
    "-": "solid",
    "--": "dash",
    ":": "dot",
    "-.": "dashdot",
}

#: Reverse mapping: Plotly canonical dash name → matplotlib linestyle shorthand.
DASH_TO_MPL: dict[str, str] = {v: k for k, v in MPL_TO_DASH.items()}


def resolve_linestyle(name: str, *, target: str = "plotly") -> str:
    """Resolve any linestyle notation to the requested target format.

    Accepts matplotlib shorthand (``"-"``, ``"--"``, ``":"``, ``"-."``) and Plotly
    names (``"solid"``, ``"dash"``, ``"dot"``, ``"dashdot"``).  Any other string is
    returned unchanged so renderers can pass it through to the backend.

    Args:
        name: A linestyle string in any supported notation.
        target: Output format — ``"plotly"`` (default) for the canonical Plotly dash
            name, or ``"mpl"`` for the matplotlib shorthand string.

    Returns:
        The resolved linestyle string in the requested format, or *name* unchanged if
        no match is found in either direction.
    """
    canonical = name if name in DASH_TO_MPL else MPL_TO_DASH.get(name, name)
    if target == "mpl":
        return DASH_TO_MPL.get(canonical, canonical)
    return canonical
