"""Tree visualization for tralda.

:class:`~tralda.visualization.matplotlib_renderer.MatplotlibRenderer` and
:class:`~tralda.visualization.plotly_renderer.PlotlyRenderer` depend on the optional ``matplotlib``
and ``plotly`` packages respectively (install with ``pip install tralda[matplotlib]``,
``pip install tralda[plotly]``, or ``pip install tralda[viz]`` for both). They are imported lazily
here so that importing :mod:`tralda.visualization` does not require both backends to be installed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from tralda.visualization._convenience import plot_tree as plot_tree
from tralda.visualization.layout import EdgeLengthMode as EdgeLengthMode
from tralda.visualization.layout import LayoutMode as LayoutMode
from tralda.visualization.layout import NodeRankMode as NodeRankMode
from tralda.visualization.layout import TreeLayout as TreeLayout
from tralda.visualization.style import DEFAULT_NODE_STYLE as DEFAULT_NODE_STYLE
from tralda.visualization.style import NodeStyle as NodeStyle
from tralda.visualization.style import TreeStyle as TreeStyle

if TYPE_CHECKING:
    from tralda.visualization.matplotlib_renderer import MatplotlibRenderer as MatplotlibRenderer
    from tralda.visualization.matplotlib_renderer import register_symbol as register_symbol
    from tralda.visualization.plotly_renderer import PlotlyRenderer as PlotlyRenderer

# name -> submodule providing it, imported on first access via __getattr__ (PEP 562)
_LAZY_SUBMODULES = {
    "MatplotlibRenderer": "tralda.visualization.matplotlib_renderer",
    "register_symbol": "tralda.visualization.matplotlib_renderer",
    "PlotlyRenderer": "tralda.visualization.plotly_renderer",
}


def __getattr__(name: str) -> Any:
    try:
        module_name = _LAZY_SUBMODULES[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_SUBMODULES))
