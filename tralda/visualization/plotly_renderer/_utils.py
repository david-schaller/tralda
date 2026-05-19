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
