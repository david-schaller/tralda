"""One-call convenience wrapper around :class:`TreeLayout` and the renderer backends.

The layout/style/renderer separation used elsewhere in this package is flexible, but it means the
common case — "just show me the tree" — requires constructing three objects by hand.
:func:`plot_tree` collapses that into a single call.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, Literal

from tralda.visualization.layout import EdgeLengthMode, LayoutMode, NodeRankMode, TreeLayout
from tralda.visualization.style import TreeStyle

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from plotly.graph_objects import Figure as PlotlyFigure

    from tralda.datastructures.tree import Tree


def plot_tree(
    tree: Tree,
    *,
    backend: Literal["matplotlib", "plotly"] = "matplotlib",
    path: str | os.PathLike | None = None,
    show: bool = False,
    edge_length_mode: EdgeLengthMode
    | Literal["attr", "uniform", "even", "rank"] = EdgeLengthMode.ATTR,
    edge_length_attr: str = "dist",
    layout_mode: LayoutMode | Literal["horizontal", "vertical", "circular"] = LayoutMode.HORIZONTAL,
    node_rank_mode: NodeRankMode | Literal["mean", "first", "last", "node"] = NodeRankMode.MEAN,
    tree_style: TreeStyle | None = None,
    **renderer_kwargs: Any,
) -> tuple[Figure, Axes] | PlotlyFigure:
    """Lay out and render *tree* in a single call.

    Convenience wrapper around :class:`~tralda.visualization.layout.TreeLayout` and
    :class:`~tralda.visualization.matplotlib_renderer.MatplotlibRenderer` /
    :class:`~tralda.visualization.plotly_renderer.PlotlyRenderer`. Construct those objects directly
    for more control (e.g. drawing onto an existing matplotlib ``Axes``, or reusing one layout with
    several styles).

    Args:
        tree: The tree to render.
        backend: ``"matplotlib"`` (default) or ``"plotly"``.
        path: If given, save the resulting figure to this path. For the matplotlib backend the
            format is inferred from the file extension by
            :meth:`~matplotlib.figure.Figure.savefig` (e.g. ``.png``, ``.pdf``, ``.svg``). For the
            plotly backend, a ``.html`` extension writes a self-contained interactive file via
            :meth:`~plotly.graph_objects.Figure.write_html`; any other extension writes a static
            image via :meth:`~plotly.graph_objects.Figure.write_image`, which requires the
            ``kaleido`` package (included in the ``plotly`` extra: ``pip install tralda[plotly]``).
        show: If ``True``, display the figure (``fig.show()``) after rendering.
        edge_length_mode: How edge lengths are determined. See
            :class:`~tralda.visualization.layout.EdgeLengthMode`.
        edge_length_attr: Node attribute read as edge length in ``"attr"`` mode.
        layout_mode: Tree orientation. See :class:`~tralda.visualization.layout.LayoutMode`.
        node_rank_mode: How the rank of internal nodes is derived from their children. See
            :class:`~tralda.visualization.layout.NodeRankMode`.
        tree_style: Styling configuration. When ``None`` the default style is used.
        **renderer_kwargs: Additional keyword arguments forwarded to the renderer constructor
            (e.g. ``figsize`` / ``ax`` for matplotlib, ``width`` / ``height`` / ``hover_attrs`` for
            plotly).

    Returns:
        ``(fig, ax)`` for the matplotlib backend, or a Plotly ``Figure`` for the plotly backend.

    Raises:
        ValueError: If *backend* is not ``"matplotlib"`` or ``"plotly"``.
        ImportError: If the optional dependency required by *backend* is not installed.
    """
    layout = TreeLayout(
        tree,
        edge_length_mode=edge_length_mode,
        edge_length_attr=edge_length_attr,
        layout_mode=layout_mode,
        node_rank_mode=node_rank_mode,
    )

    if backend == "matplotlib":
        from tralda.visualization.matplotlib_renderer import MatplotlibRenderer

        fig, ax = MatplotlibRenderer(layout, tree_style=tree_style, **renderer_kwargs).render()
        if path is not None:
            fig.savefig(path, bbox_inches="tight")
        if show:
            fig.show()
        return fig, ax

    elif backend == "plotly":
        from tralda.visualization.plotly_renderer import PlotlyRenderer

        fig = PlotlyRenderer(layout, tree_style=tree_style, **renderer_kwargs).render()
        if path is not None:
            if os.path.splitext(str(path))[1].lower() == ".html":
                fig.write_html(path)
            else:
                try:
                    fig.write_image(path)
                except (ImportError, ValueError) as exc:
                    if "kaleido" not in str(exc).lower():
                        raise
                    raise ImportError(
                        "Static image export for the plotly backend requires the 'kaleido' "
                        "package. Install it with: pip install tralda[plotly]"
                    ) from exc
        if show:
            fig.show()
        return fig

    else:
        raise ValueError(f"unknown backend {backend!r}; expected 'matplotlib' or 'plotly'")
