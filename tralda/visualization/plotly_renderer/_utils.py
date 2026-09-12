"""Utility functions for Plotly rendering."""

from typing import Any


# --------------------------------------------------------------------------------------------------
# Mappings
# --------------------------------------------------------------------------------------------------

#: Single-letter matplotlib color shortcuts to CSS color names.
_MPL_SHORT_COLORS: dict[str, str] = {
    "b": "blue",
    "g": "green",
    "r": "red",
    "c": "cyan",
    "m": "magenta",
    "y": "yellow",
    "k": "black",
    "w": "white",
}

#: Matplotlib named font sizes mapped to points, using matplotlib's default font-scaling factors
#: (``matplotlib.font_manager.font_scalings``) applied to its default base size of 10pt.  Plotly's
#: font size must be numeric, unlike matplotlib which accepts these names directly.
_NAMED_FONT_SIZES: dict[str, float] = {
    "xx-small": 5.79,
    "x-small": 6.94,
    "smaller": 8.33,
    "small": 8.33,
    "medium": 10.0,
    "large": 12.0,
    "larger": 12.0,
    "x-large": 14.4,
    "xx-large": 17.28,
}


# ---------------------------------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------------------------------


def to_plotly_color(color: Any) -> str:
    """Convert a color value to a Plotly-compatible string.

    Handles named colors, hex strings, and float RGB/RGBA tuples (values in ``[0, 1]``).
    Single-letter matplotlib shortcuts (``"k"``, ``"r"``, …) are expanded to full color names.

    Args:
        color: A color in any format accepted by matplotlib or Plotly.

    Returns:
        A color string accepted by Plotly (named, hex, or ``"rgba(r,g,b,a)"``).
    """
    if color is None:
        return "black"
    if isinstance(color, str):
        if color.lower() in ("none", "transparent"):
            return "rgba(0,0,0,0)"
        return _MPL_SHORT_COLORS.get(color, color)
    # Handle any sequence (list, tuple, numpy array, …) of 3 or 4 float components in [0, 1].
    try:
        if len(color) in (3, 4):
            r, g, b = (int(round(float(color[i]) * 255)) for i in range(3))
            a = float(color[3]) if len(color) == 4 else 1.0
            return f"rgba({r},{g},{b},{a})"
    except (TypeError, ValueError):
        pass

    return str(color)


def to_plotly_font_size(size: float | str) -> float:
    """Convert a font size to a Plotly-compatible number.

    ``NodeStyle.label_fontsize`` also accepts matplotlib named sizes (e.g. ``"small"``), but
    Plotly's ``font.size`` property must be numeric. Named sizes are resolved via
    :data:`_NAMED_FONT_SIZES`; unrecognised names fall back to the ``"medium"`` size (10pt).

    Args:
        size: A font size in points, or a matplotlib named size string.

    Returns:
        A numeric font size in points.
    """
    if isinstance(size, str):
        return _NAMED_FONT_SIZES.get(size, _NAMED_FONT_SIZES["medium"])

    return float(size)
