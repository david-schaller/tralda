"""Tests for tralda.visualization.layout (TreeLayout, EdgeLengthMode, NodeRankMode, LayoutMode)."""

from __future__ import annotations

import math

import pytest

from tralda.datastructures.tree import Tree, TreeNode
from tralda.visualization.layout import EdgeLengthMode, LayoutMode, TreeLayout

# ===========================================================================
# EdgeLengthMode — depth computation
# ===========================================================================


class TestEdgeLengthModeAttr:
    def test_root_depth_is_zero(self, small_tree):
        layout = TreeLayout(small_tree, edge_length_mode="attr")
        assert layout.depths[small_tree.root] == 0.0

    def test_depths_match_cumulative_dist(self, small_tree):
        layout = TreeLayout(small_tree, edge_length_mode=EdgeLengthMode.ATTR)
        expected = {"r": 0.0, "a": 1.0, "b": 2.0, "c": 2.0, "d": 2.5, "e": 2.5, "f": 3.0}
        for v in small_tree.preorder():
            assert layout.depths[v] == pytest.approx(expected[v.label])

    def test_max_depth_is_deepest_leaf(self, small_tree):
        layout = TreeLayout(small_tree, edge_length_mode="attr")
        assert layout.max_depth == pytest.approx(3.0)

    def test_missing_dist_attribute_defaults_to_zero(self):
        root = TreeNode(label="r")
        child = TreeNode(label="c")  # no 'dist' attribute set
        root.add_child(child)
        layout = TreeLayout(Tree(root), edge_length_mode="attr")
        assert layout.depths[child] == 0.0

    def test_custom_edge_length_attr(self):
        root = TreeNode(label="r")
        root.weight = 0.0
        child = TreeNode(label="c")
        child.weight = 2.5
        root.add_child(child)
        layout = TreeLayout(Tree(root), edge_length_mode="attr", edge_length_attr="weight")
        assert layout.depths[child] == pytest.approx(2.5)


class TestEdgeLengthModeUniform:
    def test_depth_equals_topological_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        by_label = {v.label: layout.depths[v] for v in unbalanced_tree.preorder()}
        assert by_label == {"r": 0.0, "x": 1.0, "y": 1.0, "z": 2.0, "w": 2.0}

    def test_leaves_may_differ_in_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        leaf_depths = {v.label: layout.depths[v] for v in unbalanced_tree.leaves()}
        assert leaf_depths == {"x": 1.0, "z": 2.0, "w": 2.0}

    def test_max_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        assert layout.max_depth == pytest.approx(2.0)


class TestEdgeLengthModeEven:
    def test_all_leaves_reach_max_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="even")
        for leaf in unbalanced_tree.leaves():
            assert layout.depths[leaf] == pytest.approx(layout.max_depth)

    def test_internal_node_depth_between_root_and_leaves(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="even")
        y = next(v for v in unbalanced_tree.preorder() if v.label == "y")
        assert 0.0 < layout.depths[y] < layout.max_depth

    def test_max_depth_is_one(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="even")
        assert layout.max_depth == pytest.approx(1.0)

    def test_single_node_tree(self, single_node_tree):
        layout = TreeLayout(single_node_tree, edge_length_mode="even")
        assert layout.depths[single_node_tree.root] == 0.0
        assert layout.max_depth == 0.0


class TestEdgeLengthModeRank:
    def test_leaves_extended_to_topological_max(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="rank")
        leaf_depths = {v.label: layout.depths[v] for v in unbalanced_tree.leaves()}
        assert leaf_depths == {"x": 2.0, "z": 2.0, "w": 2.0}

    def test_internal_nodes_at_topological_depth(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="rank")
        y = next(v for v in unbalanced_tree.preorder() if v.label == "y")
        assert layout.depths[y] == pytest.approx(1.0)

    def test_max_depth_equals_topological_height(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="rank")
        assert layout.max_depth == pytest.approx(2.0)


class TestEdgeLengthModeGeneral:
    @pytest.mark.parametrize("mode", ["attr", "uniform", "even", "rank"])
    def test_string_and_enum_modes_are_equivalent(self, small_tree, mode):
        by_string = TreeLayout(small_tree, edge_length_mode=mode)
        by_enum = TreeLayout(small_tree, edge_length_mode=EdgeLengthMode(mode))
        assert by_string.depths == by_enum.depths

    def test_invalid_mode_string_raises(self, small_tree):
        with pytest.raises(ValueError):
            TreeLayout(small_tree, edge_length_mode="bogus")

    @pytest.mark.parametrize("mode", ["attr", "uniform", "even", "rank"])
    def test_empty_tree_no_crash(self, empty_tree, mode):
        layout = TreeLayout(empty_tree, edge_length_mode=mode)
        assert layout.depths == {}
        assert layout.max_depth == 0.0

    @pytest.mark.parametrize("mode", ["attr", "uniform", "even", "rank"])
    def test_single_node_tree_depth_zero(self, single_node_tree, mode):
        layout = TreeLayout(single_node_tree, edge_length_mode=mode)
        assert layout.depths[single_node_tree.root] == 0.0


# ===========================================================================
# NodeRankMode — leaf/internal rank computation
# ===========================================================================


class TestNodeRankMode:
    def test_leaves_get_consecutive_postorder_ranks(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="mean")
        ranks = {v.label: layout.leaf_ranks[v] for v in small_tree.leaves()}
        assert ranks == {"c": 0.0, "d": 1.0, "e": 2.0, "f": 3.0}

    def test_mean_mode_internal_ranks(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="mean")
        by_label = {v.label: layout.leaf_ranks[v] for v in small_tree.preorder()}
        assert by_label["a"] == pytest.approx(0.5)
        assert by_label["b"] == pytest.approx(2.5)
        assert by_label["r"] == pytest.approx(1.5)

    def test_first_mode_internal_ranks(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="first")
        by_label = {v.label: layout.leaf_ranks[v] for v in small_tree.preorder()}
        assert by_label["a"] == pytest.approx(0.0)
        assert by_label["b"] == pytest.approx(2.0)
        assert by_label["r"] == pytest.approx(0.0)

    def test_last_mode_internal_ranks(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="last")
        by_label = {v.label: layout.leaf_ranks[v] for v in small_tree.preorder()}
        assert by_label["a"] == pytest.approx(1.0)
        assert by_label["b"] == pytest.approx(3.0)
        assert by_label["r"] == pytest.approx(3.0)

    def test_node_mode_gives_every_node_a_unique_rank(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="node")
        ranks = list(layout.leaf_ranks.values())
        assert len(ranks) == len(set(ranks)) == len(small_tree)

    def test_node_mode_rank_span_equals_node_count(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="node")
        assert layout.rank_span == len(small_tree)

    def test_node_mode_leaf_count_still_counts_only_leaves(self, small_tree):
        layout = TreeLayout(small_tree, node_rank_mode="node")
        assert layout.leaf_count == sum(1 for _ in small_tree.leaves())

    @pytest.mark.parametrize("mode", ["mean", "first", "last"])
    def test_rank_span_equals_leaf_count_for_non_node_modes(self, small_tree, mode):
        layout = TreeLayout(small_tree, node_rank_mode=mode)
        assert layout.rank_span == layout.leaf_count == 4

    def test_invalid_mode_string_raises(self, small_tree):
        with pytest.raises(ValueError):
            TreeLayout(small_tree, node_rank_mode="bogus")

    def test_star_tree_mean_rank_is_centered(self, star_tree):
        layout = TreeLayout(star_tree, node_rank_mode="mean")
        # 3 leaves ranked 0, 1, 2 -> mean of first (0) and last (2) is 1.
        assert layout.leaf_ranks[star_tree.root] == pytest.approx(1.0)


# ===========================================================================
# LayoutMode — screen-space positions
# ===========================================================================


class TestPositionsHorizontal:
    def test_position_is_depth_rank_pair(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="horizontal")
        for v in small_tree.preorder():
            assert layout.positions[v] == (layout.depths[v], layout.leaf_ranks[v])

    def test_label_metadata(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="horizontal")
        v = small_tree.root
        assert layout.label_angle[v] == 0.0
        assert layout.label_ha[v] == "left"
        assert layout.label_va[v] == "center"


class TestPositionsVertical:
    def test_position_is_rank_depth_pair(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="vertical")
        for v in small_tree.preorder():
            assert layout.positions[v] == (layout.leaf_ranks[v], layout.depths[v])

    def test_label_angle_is_vertical(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="vertical")
        assert layout.label_angle[small_tree.root] == -90.0


class TestPositionsCircular:
    def test_root_is_at_origin(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="circular", edge_length_mode="attr")
        x, y = layout.positions[small_tree.root]
        assert x == pytest.approx(0.0, abs=1e-9)
        assert y == pytest.approx(0.0, abs=1e-9)

    def test_position_matches_polar_conversion(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="circular")
        for v in small_tree.preorder():
            r = layout.depths[v]
            theta = 2.0 * math.pi * layout.leaf_ranks[v] / layout.rank_span
            expected = (r * math.cos(theta), r * math.sin(theta))
            got = layout.positions[v]
            assert got[0] == pytest.approx(expected[0], abs=1e-9)
            assert got[1] == pytest.approx(expected[1], abs=1e-9)

    def test_label_angle_within_range(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="circular")
        for angle in layout.label_angle.values():
            assert -180.0 <= angle < 180.0

    def test_label_ha_right_half_is_left_aligned(self):
        # A single leaf at rank 0 sits at theta = 0 (angle 0deg) -> right half -> "left" ha.
        root = TreeNode(label="r")
        root.dist = 0.0
        leaf = TreeNode(label="leaf")
        leaf.dist = 1.0
        root.add_child(leaf)
        layout = TreeLayout(Tree(root), layout_mode="circular")
        assert layout.label_ha[leaf] == "left"


class TestComputePositionsSwitching:
    def test_switching_mode_does_not_recompute_depths(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="horizontal")
        depths_before = dict(layout.depths)
        layout.compute_positions("vertical")
        assert layout.depths == depths_before

    def test_switching_mode_updates_positions(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="horizontal")
        horizontal_positions = dict(layout.positions)
        layout.compute_positions("vertical")
        assert layout.positions != horizontal_positions
        assert layout.layout_mode == LayoutMode.VERTICAL

    def test_switching_to_none_keeps_current_mode(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="vertical")
        layout.compute_positions(None)
        assert layout.layout_mode == LayoutMode.VERTICAL


# ===========================================================================
# parent_edge / children_connector / ghost_segment
# ===========================================================================


class TestParentEdge:
    def test_root_has_no_parent_edge(self, small_tree):
        layout = TreeLayout(small_tree)
        assert layout.parent_edge(small_tree.root) is None

    def test_non_root_returns_parent_and_node_positions(self, small_tree):
        layout = TreeLayout(small_tree)
        a = next(v for v in small_tree.preorder() if v.label == "a")
        parent_pos, node_pos = layout.parent_edge(a)
        assert parent_pos == layout.positions[small_tree.root]
        assert node_pos == layout.positions[a]


class TestChildrenConnector:
    def test_leaf_returns_none(self, small_tree):
        layout = TreeLayout(small_tree)
        c = next(v for v in small_tree.preorder() if v.label == "c")
        assert layout.children_connector(c) is None

    def test_horizontal_connector_spans_children_ranks(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="horizontal")
        a = next(v for v in small_tree.preorder() if v.label == "a")
        start, end = layout.children_connector(a)
        c = next(v for v in small_tree.preorder() if v.label == "c")
        d = next(v for v in small_tree.preorder() if v.label == "d")
        assert start == (layout.positions[a][0], layout.positions[c][1])
        assert end == (layout.positions[a][0], layout.positions[d][1])

    def test_circular_connector_matches_manual_polar_calc(self, small_tree):
        layout = TreeLayout(small_tree, layout_mode="circular")
        a = next(v for v in small_tree.preorder() if v.label == "a")
        c = next(v for v in small_tree.preorder() if v.label == "c")
        d = next(v for v in small_tree.preorder() if v.label == "d")
        start, end = layout.children_connector(a)
        r = layout.depths[a]
        theta_c = 2.0 * math.pi * layout.leaf_ranks[c] / layout.rank_span
        theta_d = 2.0 * math.pi * layout.leaf_ranks[d] / layout.rank_span
        assert start == pytest.approx((r * math.cos(theta_c), r * math.sin(theta_c)))
        assert end == pytest.approx((r * math.cos(theta_d), r * math.sin(theta_d)))


class TestGhostSegment:
    def test_leaf_at_max_depth_has_no_ghost(self, small_tree):
        layout = TreeLayout(small_tree, edge_length_mode="attr")
        f = next(v for v in small_tree.preorder() if v.label == "f")  # deepest leaf
        assert layout.ghost_segment(f) is None

    def test_shallower_leaf_has_ghost_segment(self, unbalanced_tree):
        layout = TreeLayout(unbalanced_tree, edge_length_mode="uniform")
        x = next(v for v in unbalanced_tree.preorder() if v.label == "x")
        seg = layout.ghost_segment(x)
        assert seg is not None
        (x0, y0), (x1, y1) = seg
        assert x0 == pytest.approx(layout.depths[x])
        assert x1 == pytest.approx(layout.max_depth)
        assert y0 == y1 == pytest.approx(layout.leaf_ranks[x])

    def test_internal_node_has_no_ghost(self, small_tree):
        layout = TreeLayout(small_tree)
        a = next(v for v in small_tree.preorder() if v.label == "a")
        assert layout.ghost_segment(a) is None


# ===========================================================================
# Empty / degenerate trees
# ===========================================================================


class TestEmptyAndSingleNodeTree:
    def test_empty_tree_all_dicts_empty(self, empty_tree):
        layout = TreeLayout(empty_tree)
        assert layout.depths == {}
        assert layout.leaf_ranks == {}
        assert layout.positions == {}
        assert layout.max_depth == 0.0
        assert layout.leaf_count == 0
        assert layout.rank_span == 0

    def test_single_node_tree_leaf_count_and_rank_span(self, single_node_tree):
        layout = TreeLayout(single_node_tree)
        assert layout.leaf_count == 1
        assert layout.rank_span == 1

    @pytest.mark.parametrize("mode", ["horizontal", "vertical", "circular"])
    def test_single_node_tree_position_at_origin(self, single_node_tree, mode):
        layout = TreeLayout(single_node_tree, layout_mode=mode)
        assert layout.positions[single_node_tree.root] == (0.0, 0.0)
