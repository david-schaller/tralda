"""Tests for tralda.visualization.matplotlib_renderer (MatplotlibRenderer, register_symbol)."""

from __future__ import annotations

import warnings

import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")  # headless backend, must be set before pyplot is used anywhere

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from tralda.visualization.layout import TreeLayout
from tralda.visualization.matplotlib_renderer import (
    MatplotlibRenderer,
    register_symbol,
)
from tralda.visualization.matplotlib_renderer._symbols import SYMBOL_REGISTRY
from tralda.visualization.style import NodeStyle, TreeStyle


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after every test to avoid resource warnings."""
    yield
    plt.close("all")


# ===========================================================================
# Basic rendering
# ===========================================================================


class TestRenderBasics:
    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    def test_render_returns_fig_and_ax(self, small_tree, mode):
        layout = TreeLayout(small_tree, layout_mode=mode)
        fig, ax = MatplotlibRenderer(layout).render()
        assert isinstance(fig, Figure)
        assert isinstance(ax, Axes)

    def test_render_on_existing_axes_reuses_it(self, small_tree):
        fig_in, ax_in = plt.subplots()
        layout = TreeLayout(small_tree)
        fig_out, ax_out = MatplotlibRenderer(layout, ax=ax_in).render()
        assert ax_out is ax_in
        assert fig_out is fig_in

    def test_figsize_is_used_when_given(self, small_tree):
        layout = TreeLayout(small_tree)
        fig, _ = MatplotlibRenderer(layout, figsize=(4.0, 5.0)).render()
        assert tuple(fig.get_size_inches()) == (4.0, 5.0)

    def test_auto_figsize_used_when_not_given(self, small_tree):
        layout = TreeLayout(small_tree)
        fig, _ = MatplotlibRenderer(layout).render()
        width, height = fig.get_size_inches()
        assert width > 0 and height > 0


# ===========================================================================
# Labels
# ===========================================================================


class TestLabels:
    def test_leaf_labels_drawn_by_default(self, small_tree):
        layout = TreeLayout(small_tree)
        _, ax = MatplotlibRenderer(layout).render()
        assert len(ax.texts) == sum(1 for _ in small_tree.leaves())

    def test_show_labels_false_draws_no_leaf_labels(self, small_tree):
        layout = TreeLayout(small_tree)
        _, ax = MatplotlibRenderer(layout, show_labels=False).render()
        assert len(ax.texts) == 0

    def test_show_internal_labels_true_adds_internal_labels(self, small_tree):
        layout = TreeLayout(small_tree)
        _, ax = MatplotlibRenderer(layout, show_internal_labels=True).render()
        assert len(ax.texts) == len(small_tree)  # every node has a label in small_tree

    def test_nodes_without_label_attr_are_skipped(self):
        from tralda.datastructures.tree import Tree, TreeNode

        root = TreeNode()  # no 'label' attribute set
        root.dist = 0.0
        layout = TreeLayout(Tree(root))
        _, ax = MatplotlibRenderer(layout, show_internal_labels=True).render()
        assert len(ax.texts) == 0


# ===========================================================================
# Ghost segments
# ===========================================================================


class TestGhostSegments:
    def test_ghost_segments_drawn_when_leaves_differ_in_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        _, ax_with = plt.subplots()
        MatplotlibRenderer(layout, ax=ax_with, show_ghost_segments=True).render()
        _, ax_without = plt.subplots()
        MatplotlibRenderer(layout, ax=ax_without, show_ghost_segments=False).render()
        assert len(ax_with.lines) > len(ax_without.lines)

    def test_no_ghost_segments_when_all_leaves_aligned(self, small_tree):
        # edge_length_mode="even" forces all leaves to the same depth -> no ghost segments.
        layout = TreeLayout(small_tree, edge_length_mode="even")
        _, ax_with = plt.subplots()
        MatplotlibRenderer(layout, ax=ax_with, show_ghost_segments=True).render()
        _, ax_without = plt.subplots()
        MatplotlibRenderer(layout, ax=ax_without, show_ghost_segments=False).render()
        assert len(ax_with.lines) == len(ax_without.lines)


# ===========================================================================
# Styling
# ===========================================================================


class TestStyling:
    def test_from_maps_with_node_color_map_renders_without_error(self, small_tree):
        for v in small_tree.preorder():
            v.color = v.label
        style = TreeStyle.from_maps(symbol_map={"a": "square"}, node_color_map={"c": "steelblue"})
        layout = TreeLayout(small_tree)
        fig, _ax = MatplotlibRenderer(layout, tree_style=style).render()
        assert isinstance(fig, Figure)

    def test_unknown_symbol_warns_and_falls_back(self, small_tree):
        style = TreeStyle(node_symbol="not-a-real-symbol")
        layout = TreeLayout(small_tree)
        with pytest.warns(UserWarning, match="not-a-real-symbol"):
            MatplotlibRenderer(layout, tree_style=style).render()

    @pytest.mark.parametrize("fontsize", [8, "small", "large"])
    def test_named_and_numeric_fontsizes_both_work(self, small_tree, fontsize):
        style = TreeStyle(default=NodeStyle(label_fontsize=fontsize))
        layout = TreeLayout(small_tree)
        fig, _ = MatplotlibRenderer(layout, tree_style=style).render()
        assert isinstance(fig, Figure)


# ===========================================================================
# register_symbol
# ===========================================================================


class TestRegisterSymbol:
    def test_register_and_use_custom_symbol(self, small_tree):
        name = "test-custom-symbol-abc123"
        calls = []

        def _drawer(ax, x, y, ns, **kwargs):
            calls.append((x, y))

        register_symbol(name, _drawer)
        try:
            style = TreeStyle(node_symbol=name)
            layout = TreeLayout(small_tree)
            MatplotlibRenderer(layout, tree_style=style).render()
            assert len(calls) == len(small_tree)
        finally:
            del SYMBOL_REGISTRY[name]

    def test_registering_duplicate_name_raises(self):
        name = "test-custom-symbol-dup"
        register_symbol(name, lambda ax, x, y, ns, **kwargs: None)
        try:
            with pytest.raises(KeyError):
                register_symbol(name, lambda ax, x, y, ns, **kwargs: None)
        finally:
            del SYMBOL_REGISTRY[name]


# ===========================================================================
# Empty / degenerate trees
# ===========================================================================


class TestEmptyAndSingleNodeTree:
    def test_empty_tree_renders_without_warning(self, empty_tree):
        layout = TreeLayout(empty_tree)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fig, ax = MatplotlibRenderer(layout).render()
        assert isinstance(fig, Figure)
        assert len(ax.texts) == 0

    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    def test_single_node_tree_renders(self, single_node_tree, mode):
        layout = TreeLayout(single_node_tree, layout_mode=mode)
        fig, ax = MatplotlibRenderer(layout).render()
        assert isinstance(fig, Figure)
        assert len(ax.texts) == 1  # the single leaf/root has a label
