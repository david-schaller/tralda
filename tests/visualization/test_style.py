"""Tests for tralda.visualization.style (NodeStyle, TreeStyle)."""

from __future__ import annotations

import pytest

from tralda.datastructures.tree import Tree, TreeNode
from tralda.visualization.layout import LayoutMode
from tralda.visualization.style import DEFAULT_NODE_STYLE, NodeStyle, TreeStyle, _safe_eq

# ===========================================================================
# NodeStyle.overlay
# ===========================================================================


class TestNodeStyleOverlay:
    def test_non_none_fields_of_other_win(self):
        base = NodeStyle(symbol="circle", symbol_color="red")
        other = NodeStyle(symbol="square")
        merged = base.overlay(other)
        assert merged.symbol == "square"
        assert merged.symbol_color == "red"

    def test_none_fields_of_other_do_not_override(self):
        base = NodeStyle(symbol="circle", edge_lw=2.0)
        other = NodeStyle(symbol=None, edge_lw=None)
        merged = base.overlay(other)
        assert merged.symbol == "circle"
        assert merged.edge_lw == 2.0

    def test_overlay_does_not_mutate_operands(self):
        base = NodeStyle(symbol="circle")
        other = NodeStyle(symbol="square")
        base.overlay(other)
        assert base.symbol == "circle"
        assert other.symbol == "square"

    def test_overlay_returns_new_instance(self):
        base = NodeStyle()
        merged = base.overlay(NodeStyle())
        assert merged is not base


class TestDefaultNodeStyle:
    def test_all_fields_are_resolved(self):
        from dataclasses import fields

        for f in fields(DEFAULT_NODE_STYLE):
            if f.name == "symbol":
                continue  # symbol is intentionally left to structural fallbacks
            assert getattr(DEFAULT_NODE_STYLE, f.name) is not None


# ===========================================================================
# TreeStyle construction
# ===========================================================================


class TestTreeStyleConstruction:
    def test_default_none_uses_default_node_style_values(self):
        style = TreeStyle()
        assert style.default.symbol_size == DEFAULT_NODE_STYLE.symbol_size
        assert style.default.symbol_color == DEFAULT_NODE_STYLE.symbol_color

    def test_partial_default_overrides_only_given_fields(self):
        style = TreeStyle(default=NodeStyle(symbol_color="tomato"))
        assert style.default.symbol_color == "tomato"
        assert style.default.symbol_size == DEFAULT_NODE_STYLE.symbol_size

    def test_default_has_no_none_fields_after_init(self):
        from dataclasses import fields

        style = TreeStyle(default=NodeStyle(symbol_color="tomato"))
        for f in fields(style.default):
            if f.name == "symbol":
                continue
            assert getattr(style.default, f.name) is not None

    def test_node_overrides_default_is_empty_dict(self):
        style = TreeStyle()
        assert style.node_overrides == {}

    def test_separate_instances_do_not_share_node_overrides(self):
        s1, s2 = TreeStyle(), TreeStyle()
        s1.node_overrides[TreeNode()] = NodeStyle()
        assert s2.node_overrides == {}


# ===========================================================================
# TreeStyle.resolve
# ===========================================================================


class TestResolve:
    def test_default_only_matches_tree_style_default_fields(self):
        style = TreeStyle(default=NodeStyle(symbol_color="steelblue"))
        node = TreeNode()
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol_color == "steelblue"

    def test_style_fn_overrides_default(self):
        style = TreeStyle(
            default=NodeStyle(symbol_color="steelblue"),
            style_fn=lambda n, m: NodeStyle(symbol_color="tomato"),
        )
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        assert resolved.symbol_color == "tomato"

    def test_style_fn_returning_none_keeps_default(self):
        style = TreeStyle(
            default=NodeStyle(symbol_color="steelblue"),
            style_fn=lambda n, m: None,
        )
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        assert resolved.symbol_color == "steelblue"

    def test_node_override_has_highest_priority(self):
        node = TreeNode()
        style = TreeStyle(
            default=NodeStyle(symbol_color="steelblue"),
            style_fn=lambda n, m: NodeStyle(symbol_color="tomato"),
            node_overrides={node: NodeStyle(symbol_color="gold")},
        )
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol_color == "gold"

    def test_node_override_only_applies_to_that_node(self):
        node, other = TreeNode(), TreeNode()
        style = TreeStyle(node_overrides={node: NodeStyle(symbol_color="gold")})
        resolved_other = style.resolve(other, LayoutMode.HORIZONTAL)
        assert resolved_other.symbol_color == DEFAULT_NODE_STYLE.symbol_color

    @pytest.mark.parametrize(
        "attr,fallback_kwarg,expected",
        [
            ("root_symbol", "root_symbol", "star"),
            ("leaf_symbol", "leaf_symbol", "square"),
            ("internal_symbol", "internal_symbol", "diamond"),
        ],
    )
    def test_structural_symbol_fallback(self, attr, fallback_kwarg, expected):
        root = TreeNode(label="r")
        mid = TreeNode(label="m")
        leaf = TreeNode(label="l")
        root.add_child(mid)
        mid.add_child(leaf)
        style = TreeStyle(**{fallback_kwarg: expected})
        node = {"root_symbol": root, "leaf_symbol": leaf, "internal_symbol": mid}[attr]
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol == expected

    def test_node_symbol_used_when_no_structural_fallback_set(self):
        style = TreeStyle(node_symbol="hexagon")
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        assert resolved.symbol == "hexagon"

    def test_explicit_symbol_from_style_fn_skips_structural_fallback(self):
        root = TreeNode(label="r")
        style = TreeStyle(root_symbol="star", style_fn=lambda n, m: NodeStyle(symbol="circle"))
        resolved = style.resolve(root, LayoutMode.HORIZONTAL)
        assert resolved.symbol == "circle"

    def test_resolve_result_has_no_none_fields(self):
        from dataclasses import fields

        style = TreeStyle()
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        for f in fields(resolved):
            assert getattr(resolved, f.name) is not None


# ===========================================================================
# TreeStyle.consensus_style
# ===========================================================================


class TestConsensusStyle:
    def test_agreement_on_field_is_used(self):
        n1, n2 = TreeNode(), TreeNode()
        style = TreeStyle(
            node_overrides={n1: NodeStyle(edge_color="red"), n2: NodeStyle(edge_color="red")}
        )
        consensus = style.consensus_style([n1, n2], LayoutMode.HORIZONTAL)
        assert consensus.edge_color == "red"

    def test_disagreement_falls_back_to_default(self):
        n1, n2 = TreeNode(), TreeNode()
        style = TreeStyle(
            default=NodeStyle(edge_color="black"),
            node_overrides={n1: NodeStyle(edge_color="red"), n2: NodeStyle(edge_color="blue")},
        )
        consensus = style.consensus_style([n1, n2], LayoutMode.HORIZONTAL)
        assert consensus.edge_color == "black"

    def test_empty_node_list_returns_default(self):
        style = TreeStyle(default=NodeStyle(edge_color="black"))
        consensus = style.consensus_style([], LayoutMode.HORIZONTAL)
        assert consensus.edge_color == "black"

    def test_consensus_with_array_valued_colors(self):
        n1, n2 = TreeNode(), TreeNode()
        rgba = (0.1, 0.2, 0.3, 1.0)
        style = TreeStyle(
            node_overrides={
                n1: NodeStyle(symbol_color=rgba),
                n2: NodeStyle(symbol_color=rgba),
            }
        )
        consensus = style.consensus_style([n1, n2], LayoutMode.HORIZONTAL)
        assert consensus.symbol_color == rgba


# ===========================================================================
# _safe_eq helper
# ===========================================================================


class TestSafeEq:
    def test_equal_scalars(self):
        assert _safe_eq(1, 1) is True

    def test_unequal_scalars(self):
        assert _safe_eq(1, 2) is False

    def test_equal_tuples(self):
        assert _safe_eq((1, 2, 3), (1, 2, 3)) is True

    def test_array_like_equality_via_iteration(self):
        numpy = pytest.importorskip("numpy")
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([1.0, 2.0, 3.0])
        assert _safe_eq(a, b) is True

    def test_array_like_inequality(self):
        numpy = pytest.importorskip("numpy")
        a = numpy.array([1.0, 2.0, 3.0])
        b = numpy.array([1.0, 2.0, 4.0])
        assert _safe_eq(a, b) is False

    def test_incomparable_types_return_false(self):
        class Uncomparable:
            def __eq__(self, other):
                raise TypeError("cannot compare")

        assert _safe_eq(Uncomparable(), Uncomparable()) is False


# ===========================================================================
# TreeStyle.from_maps
# ===========================================================================


class TestFromMaps:
    def test_symbol_map_applied(self):
        node = TreeNode(symbol="S")
        style = TreeStyle.from_maps(symbol_map={"S": "circle", "D": "square"})
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol == "circle"

    def test_node_color_map_applied(self):
        """Regression test: the constructor keyword is `node_color_map`, not `color_map`."""
        node = TreeNode(color="a")
        style = TreeStyle.from_maps(node_color_map={"a": "steelblue", "b": "tomato"})
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol_color == "steelblue"

    def test_edge_color_map_applied(self):
        node = TreeNode(edge_color="x")
        style = TreeStyle.from_maps(edge_color_map={"x": "grey"})
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.edge_color == "grey"

    def test_custom_attribute_names(self):
        node = TreeNode(event="D")
        style = TreeStyle.from_maps(symbol_attr="event", symbol_map={"D": "square"})
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol == "square"

    def test_style_map_is_lower_priority_than_symbol_map(self):
        node = TreeNode(style="grp", symbol="S")
        style = TreeStyle.from_maps(
            style_map={"grp": NodeStyle(symbol="triangle-up")},
            symbol_map={"S": "circle"},
        )
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol == "circle"

    def test_unmatched_attribute_falls_back_to_structural_default(self):
        node = TreeNode(symbol="unknown-key")
        style = TreeStyle.from_maps(symbol_map={"S": "circle"}, node_symbol="hexagon")
        resolved = style.resolve(node, LayoutMode.HORIZONTAL)
        assert resolved.symbol == "hexagon"

    def test_kwargs_forwarded_to_tree_style_constructor(self):
        style = TreeStyle.from_maps(symbol_map={}, root_symbol="star")
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        assert resolved.symbol == "star"

    def test_from_maps_with_no_maps_behaves_like_plain_tree_style(self):
        style = TreeStyle.from_maps()
        resolved = style.resolve(TreeNode(), LayoutMode.HORIZONTAL)
        assert resolved.symbol == "none"


def test_empty_tree_style_resolution_smoke(empty_tree):
    """Resolving styles over an empty tree's (empty) node set must not raise."""
    style = TreeStyle()
    assert list(empty_tree.preorder()) == []
    # Sanity: resolve() still works for an arbitrary node not in the tree.
    assert style.resolve(TreeNode(), LayoutMode.HORIZONTAL).symbol == "none"


def test_empty_tree_fixture_is_actually_empty(empty_tree):
    assert isinstance(empty_tree, Tree)
    assert empty_tree.root is None
