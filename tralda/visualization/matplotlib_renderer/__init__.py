"""Matplotlib renderer for tree layouts."""

from __future__ import annotations

try:
    from tralda.visualization.matplotlib_renderer._renderer import (
        MatplotlibRenderer as MatplotlibRenderer,
    )
    from tralda.visualization.matplotlib_renderer._symbols import (
        register_symbol as register_symbol,
    )
except ImportError as exc:
    raise ImportError(
        "MatplotlibRenderer requires the optional 'matplotlib' dependency. "
        "Install it with: pip install tralda[matplotlib]"
    ) from exc
