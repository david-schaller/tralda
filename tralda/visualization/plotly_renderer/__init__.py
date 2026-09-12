"""Plotly renderer for tree layouts."""

from __future__ import annotations

try:
    from tralda.visualization.plotly_renderer._renderer import PlotlyRenderer as PlotlyRenderer
except ImportError as exc:
    raise ImportError(
        "PlotlyRenderer requires the optional 'plotly' dependency. "
        "Install it with: pip install tralda[plotly]"
    ) from exc
