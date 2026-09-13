"""Tests for tralda.visualization.plotly_renderer (PlotlyRenderer)."""

from __future__ import annotations

import pytest

go = pytest.importorskip("plotly.graph_objects")

from tralda.visualization.layout import TreeLayout
from tralda.visualization.plotly_renderer import PlotlyRenderer
from tralda.visualization.style import NodeStyle, TreeStyle

# ===========================================================================
# Basic rendering
# ===========================================================================


class TestRenderBasics:
    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    def test_render_returns_go_figure(self, small_tree, mode):
        layout = TreeLayout(small_tree, layout_mode=mode)
        fig = PlotlyRenderer(layout).render()
        assert isinstance(fig, go.Figure)

    def test_render_has_at_least_one_trace(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout).render()
        assert len(fig.data) > 0

    def test_width_and_height_are_used_when_given(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, width=640, height=480).render()
        assert fig.layout.width == 640
        assert fig.layout.height == 480

    def test_auto_size_used_when_not_given(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout).render()
        assert fig.layout.width is not None
        assert fig.layout.height is not None


# ===========================================================================
# Hover text
# ===========================================================================


class TestHoverAttrs:
    def test_hover_attrs_default_includes_label_and_depth(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout).render()
        node_trace = next(t for t in fig.data if t.mode == "markers")
        assert any("depth:" in text for text in node_trace.hovertext)

    def test_extra_hover_attr_included(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, hover_attrs=["dist"]).render()
        node_trace = next(t for t in fig.data if t.mode == "markers")
        assert any("dist:" in text for text in node_trace.hovertext)

    def test_hover_attrs_accepts_single_string(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, hover_attrs="dist").render()
        node_trace = next(t for t in fig.data if t.mode == "markers")
        assert any("dist:" in text for text in node_trace.hovertext)


# ===========================================================================
# Labels
# ===========================================================================


class TestLabels:
    def test_labels_produce_annotations(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout).render()
        assert len(fig.layout.annotations) == sum(1 for _ in small_tree.leaves())

    def test_show_labels_false_produces_no_annotations(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, show_labels=False).render()
        assert len(fig.layout.annotations) == 0

    def test_show_internal_labels_true_adds_internal_annotations(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, show_internal_labels=True).render()
        assert len(fig.layout.annotations) == len(small_tree)

    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    @pytest.mark.parametrize("fontsize", [8, "small", "large"])
    def test_named_and_numeric_fontsizes_both_work(self, small_tree, mode, fontsize):
        """Regression test: string label_fontsize used to crash the Plotly backend."""
        style = TreeStyle(default=NodeStyle(label_fontsize=fontsize))
        layout = TreeLayout(small_tree, layout_mode=mode)
        fig = PlotlyRenderer(layout, tree_style=style, show_internal_labels=True).render()
        assert isinstance(fig, go.Figure)
        for ann in fig.layout.annotations:
            assert isinstance(ann.font.size, (int, float))


# ===========================================================================
# Composite symbols
# ===========================================================================


class TestCompositeSymbols:
    def test_circle_inner_ring_adds_secondary_trace(self, small_tree):
        style = TreeStyle(node_symbol="circle-inner-ring")
        layout = TreeLayout(small_tree)
        fig_plain = PlotlyRenderer(layout).render()
        fig_ring = PlotlyRenderer(layout, tree_style=style).render()
        assert len(fig_ring.data) == len(fig_plain.data) + 1

    def test_no_secondary_trace_without_inner_ring_symbol(self, small_tree):
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout).render()
        # Only edge trace(s) + a single node marker trace expected.
        marker_traces = [t for t in fig.data if t.mode == "markers"]
        assert len(marker_traces) == 1


# ===========================================================================
# Edge batching
# ===========================================================================


class TestEdgeBatching:
    def test_single_style_tree_has_one_edge_trace(self, small_tree):
        # Disable ghost segments so only the (uniformly styled) parent/connector edges remain.
        layout = TreeLayout(small_tree)
        fig = PlotlyRenderer(layout, show_ghost_segments=False).render()
        line_traces = [t for t in fig.data if t.mode == "lines"]
        assert len(line_traces) == 1

    def test_ghost_segments_add_separate_trace(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        fig_with = PlotlyRenderer(layout, show_ghost_segments=True).render()
        fig_without = PlotlyRenderer(layout, show_ghost_segments=False).render()
        assert len(fig_with.data) == len(fig_without.data) + 1


# ===========================================================================
# Empty / degenerate trees
# ===========================================================================


class TestEmptyAndSingleNodeTree:
    def test_empty_tree_renders_without_error(self, empty_tree):
        layout = TreeLayout(empty_tree)
        fig = PlotlyRenderer(layout).render()
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 0

    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    def test_single_node_tree_renders(self, single_node_tree, mode):
        layout = TreeLayout(single_node_tree, layout_mode=mode)
        fig = PlotlyRenderer(layout).render()
        assert isinstance(fig, go.Figure)
        assert len(fig.layout.annotations) == 1
