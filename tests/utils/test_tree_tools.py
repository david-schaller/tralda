"""Tests for tralda.utils.tree_tools."""

from __future__ import annotations

import pytest

from tralda.datastructures import Tree
from tralda.utils.tree_tools import assert_leaf_sets_equal


# ===========================================================================
# Helpers
# ===========================================================================


def _tree(newick: str) -> Tree:
    return Tree.parse_newick(newick)


# ===========================================================================
# assert_leaf_sets_equal
# ===========================================================================


class TestAssertLeafSetsEqual:
    # ── Happy-path: returns the shared leaf set ──────────────────────────────

    @pytest.mark.parametrize(
        "newicks",
        [
            ["(a,b,c);"],
            ["(a,(b,c));", "(b,(a,c));"],
            ["((a,b),(c,d));", "((a,d),(b,c));", "(a,b,c,d);"],
        ],
    )
    def test_equal_leaf_sets_returns_set(self, newicks):
        trees = [_tree(n) for n in newicks]
        result = assert_leaf_sets_equal(trees)
        assert isinstance(result, set)
        # Each tree's leaves should match the returned set
        for t in trees:
            assert {v.label for v in t.leaves()} == result

    def test_single_tree_returns_its_leaves(self):
        T = _tree("(a,b,c);")
        result = assert_leaf_sets_equal([T])
        assert result == {"a", "b", "c"}

    # ── Returns None when leaf sets differ ──────────────────────────────────

    @pytest.mark.parametrize(
        "newicks",
        [
            ["(a,b,c);", "(a,b,d);"],
            ["(a,b);", "(a,b,c);"],
            ["(a,b,c);", "(b,(a,d));", "(a,b,c);"],
        ],
    )
    def test_unequal_leaf_sets_returns_none(self, newicks):
        trees = [_tree(n) for n in newicks]
        assert assert_leaf_sets_equal(trees) is None

    # ── Raises on empty input ────────────────────────────────────────────────

    def test_raises_on_empty_collection(self):
        with pytest.raises(ValueError, match="empty"):
            assert_leaf_sets_equal([])

    # ── Raises on non-Tree elements ──────────────────────────────────────────

    def test_raises_on_non_tree(self):
        with pytest.raises(TypeError):
            assert_leaf_sets_equal(["not a tree"])

    def test_raises_on_non_tree_mixed(self):
        T = _tree("(a,b);")
        with pytest.raises(TypeError):
            assert_leaf_sets_equal([T, "not a tree"])

    # ── Raises on empty trees ────────────────────────────────────────────────

    def test_raises_on_first_tree_empty(self):
        # A tree with root=None has no leaves.
        T_empty = Tree(None)
        with pytest.raises(ValueError, match="empty"):
            assert_leaf_sets_equal([T_empty])

    def test_raises_on_second_tree_empty(self):
        # Empty tree must be detected for all positions, not just the first.
        T_normal = _tree("(a,b,c);")
        T_empty = Tree(None)
        with pytest.raises(ValueError, match="empty"):
            assert_leaf_sets_equal([T_normal, T_empty])

    # ── Raises on duplicate leaf labels ─────────────────────────────────────

    def test_raises_on_duplicate_labels_first_tree(self):
        # Manually build a tree with two leaves sharing the same label.
        from tralda.datastructures.tree import TreeNode

        root = TreeNode()
        c1 = TreeNode(label="a")
        c2 = TreeNode(label="a")
        root.add_child(c1)
        root.add_child(c2)
        T = Tree(root)
        with pytest.raises(ValueError, match="not unique"):
            assert_leaf_sets_equal([T])

    def test_raises_on_duplicate_labels_second_tree(self):
        from tralda.datastructures.tree import TreeNode

        T1 = _tree("(a,b);")
        root2 = TreeNode()
        c1 = TreeNode(label="x")
        c2 = TreeNode(label="x")
        root2.add_child(c1)
        root2.add_child(c2)
        T2 = Tree(root2)
        with pytest.raises(ValueError, match="not unique"):
            assert_leaf_sets_equal([T1, T2])
