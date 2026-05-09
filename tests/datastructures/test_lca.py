"""Tests for tralda.datastructures.last_common_ancestor (LCA)."""

from __future__ import annotations

import itertools

import pytest

from tralda.datastructures import LCA, Tree, TreeNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def naive_lca(a: TreeNode, b: TreeNode) -> TreeNode:
    """Brute-force LCA: walk parent pointers upward.

    This is a naïve reference implementation used in the tests to validate the O(1)-query LCA
    structure.
    """
    ancestors_a: set[TreeNode] = set()
    node = a
    while node is not None:
        ancestors_a.add(node)
        node = node.parent

    node = b
    while node is not None:
        if node in ancestors_a:
            return node
        node = node.parent

    raise ValueError("Nodes belong to different trees")


def all_leaf_pairs_with_expected_lca(tree: Tree) -> list[tuple[TreeNode, TreeNode, TreeNode]]:
    """Naïve enumeration of leaf pairs with their expected LCA, for testing purposes.

    For every inner node n, enumerate (leaf-a, leaf-b, n) triples where a is in a different child
    subtree than b, so the expected LCA is exactly n.
    """
    leaf_dict = tree.leaf_dict()
    result: list[tuple[TreeNode, TreeNode, TreeNode]] = []

    for node in tree.inner_nodes():
        children = list(node.children)
        for c1, c2 in itertools.combinations(children, 2):
            for a, b in itertools.product(leaf_dict[c1], leaf_dict[c2]):
                result.append((a, b, node))

    return result


# ===========================================================================
# Construction
# ===========================================================================


class TestLCAConstruction:
    def test_init_with_valid_tree(self, example_tree):
        lca = LCA(example_tree)
        assert lca is not None

    def test_init_raises_for_non_tree(self):
        with pytest.raises(TypeError):
            LCA("not a tree")  # type: ignore[arg-type]

    def test_init_raises_for_none(self):
        with pytest.raises(TypeError):
            LCA(None)  # type: ignore[arg-type]

    def test_init_raises_for_integer(self):
        with pytest.raises(TypeError):
            LCA(42)  # type: ignore[arg-type]

    def test_init_raises_for_empty_tree(self):
        with pytest.raises(ValueError, match="non-empty"):
            LCA(Tree(None))

    def test_init_raises_for_duplicate_labels(self):
        root = TreeNode(label=0)
        child_a = TreeNode(label=1)
        child_b = TreeNode(label=1)  # duplicate label
        root.add_child(child_a)
        root.add_child(child_b)
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            LCA(Tree(root))

    def test_init_single_node_tree(self):
        tree = Tree(TreeNode(label=0))
        lca = LCA(tree)
        assert lca is not None

    def test_init_two_node_tree(self):
        root = TreeNode(label=0)
        child = TreeNode(label=1)
        root.add_child(child)
        tree = Tree(root)
        lca = LCA(tree)
        assert lca is not None

    def test_init_random_tree(self, random_tree_20):
        lca = LCA(random_tree_20)
        assert lca is not None


# ===========================================================================
# LCA correctness – exhaustive comparison against naïve reference
# ===========================================================================


class TestLCACorrectness:
    def test_all_cross_child_pairs_example_tree(self, example_tree):
        """Test that all leaf pairs in the example tree have the expected LCA.

        Every (leaf-a, leaf-b) pair from different child subtrees must have the expected LCA as
        determined by the inner node they split at.
        """
        lca = LCA(example_tree)
        for a, b, expected in all_leaf_pairs_with_expected_lca(example_tree):
            assert lca(a, b) is expected, (
                f"lca({a}, {b}) returned {lca(a, b)!r}, expected {expected!r}"
            )

    def test_all_pairs_random_tree(self, random_tree_20):
        """Test that all leaf pairs have the same LCA as the naïve reference implementation.

        Compare LCA query results against the naïve implementation for all leaf pairs in a random
        20-leaf tree.
        """
        lca = LCA(random_tree_20)
        leaves = list(random_tree_20.leaves())
        for a, b in itertools.combinations(leaves, 2):
            assert lca(a, b) is naive_lca(a, b)

    def test_all_node_pairs_random_tree(self, random_tree_20):
        """Compare against naïve for all node pairs (not just leaves)."""
        lca = LCA(random_tree_20)
        nodes = list(random_tree_20.preorder())
        for a, b in itertools.combinations(nodes, 2):
            assert lca(a, b) is naive_lca(a, b)

    def test_get_agrees_with_call(self, example_tree):
        """``get`` and ``__call__`` must return identical results."""
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        for a, b in itertools.islice(itertools.combinations(leaves, 2), 50):
            assert lca.get(a, b) is lca(a, b)


# ===========================================================================
# Self and root special cases
# ===========================================================================


class TestLCASpecialCases:
    def test_lca_node_with_itself(self, example_tree):
        """lca(v, v) must be v for every node v."""
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert lca(v, v) is v

    def test_lca_root_with_any_node(self, example_tree):
        """lca(root, x) must equal the root for every node x."""
        lca = LCA(example_tree)
        root = example_tree.root
        for v in example_tree.preorder():
            assert lca(root, v) is root
            assert lca(v, root) is root

    def test_lca_parent_with_child(self, example_tree):
        """lca(parent, child) must equal the parent."""
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca(parent, child) is parent
            assert lca(child, parent) is parent

    def test_lca_symmetry(self, random_tree_20):
        """lca(a, b) == lca(b, a) for all pairs."""
        lca = LCA(random_tree_20)
        leaves = list(random_tree_20.leaves())
        for a, b in itertools.islice(itertools.combinations(leaves, 2), 50):
            assert lca(a, b) is lca(b, a)

    def test_lca_result_is_ancestor_of_both(self, random_tree_20):
        """The result of lca(a, b) must be an ancestor-or-equal of both a and b."""
        lca = LCA(random_tree_20)
        leaves = list(random_tree_20.leaves())
        for a, b in itertools.islice(itertools.combinations(leaves, 2), 40):
            ancestor = lca(a, b)
            assert lca(ancestor, a) is ancestor
            assert lca(ancestor, b) is ancestor

    def test_lca_result_is_tree_node(self, random_tree_20):
        """LCA query must always return a TreeNode instance."""
        lca = LCA(random_tree_20)
        for a, b in itertools.islice(itertools.combinations(random_tree_20.preorder(), 2), 30):
            assert isinstance(lca(a, b), TreeNode)

    def test_single_node_tree_lca_with_itself(self):
        root = TreeNode(label=0)
        tree = Tree(root)
        lca = LCA(tree)
        assert lca(root, root) is root


# ===========================================================================
# Label-based interface
# ===========================================================================


class TestLCALabelInterface:
    def test_query_by_label_agrees_with_node_query(self, example_tree):
        """Test that querying by label returns the same result as querying by TreeNode instance."""
        lca = LCA(example_tree)
        for a, b, expected in itertools.islice(all_leaf_pairs_with_expected_lca(example_tree), 50):
            result_by_node = lca(a, b)
            result_by_label = lca(a.label, b.label)
            assert result_by_node is result_by_label
            assert result_by_label is expected

    def test_query_mixed_node_and_label(self, example_tree):
        """Mix of TreeNode and label arguments must work."""
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        a, b = leaves[0], leaves[1]
        assert lca(a, b.label) is lca(a, b)
        assert lca(a.label, b) is lca(a, b)

    def test_query_by_label_returns_tree_node(self, example_tree):
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        result = lca(leaves[0].label, leaves[1].label)
        assert isinstance(result, TreeNode)


# ===========================================================================
# displays_triple
# ===========================================================================


class TestDisplaysTriple:
    def _build_triples_from_tree(
        self, tree: Tree
    ) -> tuple[
        list[tuple[TreeNode, TreeNode, TreeNode]], list[tuple[TreeNode, TreeNode, TreeNode]]
    ]:
        """Return (displayed_triples, non_displayed_triples) for the tree."""
        lca_obj = LCA(tree)
        displayed: list[tuple[TreeNode, TreeNode, TreeNode]] = []
        not_displayed: list[tuple[TreeNode, TreeNode, TreeNode]] = []

        leaves = list(tree.leaves())

        for a, b, c in itertools.combinations(leaves, 3):
            lca_ab = lca_obj(a, b)
            lca_abc = lca_obj(lca_ab, c)
            if lca_ab is not lca_abc:
                displayed.append((a, b, c))
                not_displayed.append((a, c, b))
                not_displayed.append((b, c, a))
            elif lca_obj(a, c) is not lca_obj(lca_obj(a, c), b):
                displayed.append((a, c, b))
                not_displayed.append((a, b, c))
                not_displayed.append((b, c, a))
            elif lca_obj(b, c) is not lca_obj(lca_obj(b, c), a):
                displayed.append((b, c, a))
                not_displayed.append((a, b, c))
                not_displayed.append((a, c, b))
            else:
                not_displayed.append((a, b, c))
                not_displayed.append((a, c, b))
                not_displayed.append((b, c, a))

        return displayed, not_displayed

    def test_true_for_displayed_triple(self, example_tree):
        """displays_triple must return True for triples that the tree actually displays."""
        lca = LCA(example_tree)
        leaf_dict = example_tree.leaf_dict()

        # For every inner node n with at least two children,
        # pick two leaves a, b from different child subtrees and any other leaf c
        # that is NOT in the subtree of lca(a,b). Then ab|c is a displayed triple.
        for node in example_tree.inner_nodes():
            children = list(node.children)
            if len(children) < 2:
                continue
            for c1, c2 in itertools.islice(itertools.combinations(children, 2), 3):
                leaves_c1 = leaf_dict[c1]
                leaves_c2 = leaf_dict[c2]
                if not leaves_c1 or not leaves_c2:
                    continue
                a = leaves_c1[0]
                b = leaves_c2[0]
                # c must be outside the subtree rooted at `node`
                outside = [v for v in example_tree.leaves() if v not in set(leaf_dict[node])]
                if not outside:
                    continue
                c = outside[0]
                assert lca.displays_triple(a, b, c), (
                    f"Expected tree to display triple ({a}, {b} | {c})"
                )

    def test_false_when_c_equals_lca(self, example_tree):
        """If lca(a,b) == lca(lca(a,b), c), the triple is not displayed."""
        lca = LCA(example_tree)
        # a and b are siblings; c is in the same subtree -> lca(a,b) == lca(a,c)
        # Use leaves 14 and 15 (both children of node 10) – lca is 10.
        # c = 27 is also under 10, so lca(14,15)=10, lca(10,27)=10 -> not displayed 14,15|27
        node_by_label = {v.label: v for v in example_tree.preorder()}
        a = node_by_label[14]
        b = node_by_label[15]
        c = node_by_label[27]
        assert not lca.displays_triple(a, b, c)

    def test_false_when_a_equals_b(self, example_tree):
        """Degenerate triple with a == b must return False."""
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        a = leaves[0]
        b = leaves[0]
        c = leaves[1]
        assert not lca.displays_triple(a, b, c)

    def test_false_for_unknown_label(self, example_tree):
        """Unknown label must not raise – must return False."""
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        assert lca.displays_triple(leaves[0], leaves[1], 9999) is False

    def test_by_label(self, example_tree):
        """displays_triple must accept integer labels."""
        lca = LCA(example_tree)
        # 14 and 15 share parent 10; 19 is outside that subtree.
        assert lca.displays_triple(14, 15, 19)

    def test_triple_ab_c_equals_ba_c(self, example_tree):
        """ab|c and ba|c must be equivalent."""
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        for a, b, c in itertools.islice(itertools.combinations(leaves, 3), 15):
            assert lca.displays_triple(a, b, c) == lca.displays_triple(b, a, c)


# ===========================================================================
# ancestor_or_equal / ancestor_not_equal
# ===========================================================================


class TestAncestorOrEqual:
    def test_node_is_ancestor_or_equal_of_itself(self, example_tree):
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert lca.ancestor_or_equal(v, v)

    def test_root_is_ancestor_of_all(self, example_tree):
        lca = LCA(example_tree)
        root = example_tree.root
        for v in example_tree.preorder():
            assert lca.ancestor_or_equal(root, v)

    def test_parent_is_ancestor_of_child(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca.ancestor_or_equal(parent, child)

    def test_child_is_not_ancestor_of_parent(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            if child is not parent:
                assert not lca.ancestor_or_equal(child, parent)

    def test_leaf_is_not_ancestor_of_non_descendant(self, example_tree):
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        assert not lca.ancestor_or_equal(leaves[0], leaves[1])

    def test_ancestor_not_equal_false_for_self(self, example_tree):
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert not lca.ancestor_not_equal(v, v)

    def test_ancestor_not_equal_true_for_parent_child(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca.ancestor_not_equal(parent, child)

    def test_ancestor_not_equal_false_for_child_parent(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert not lca.ancestor_not_equal(child, parent)

    def test_root_is_strict_ancestor_of_non_root(self, example_tree):
        lca = LCA(example_tree)
        root = example_tree.root
        for v in example_tree.preorder():
            if v is not root:
                assert lca.ancestor_not_equal(root, v)


# ===========================================================================
# descendant_or_equal / descendant_not_equal
# ===========================================================================


class TestDescendantOrEqual:
    def test_node_is_descendant_or_equal_of_itself(self, example_tree):
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert lca.descendant_or_equal(v, v)

    def test_child_is_descendant_of_parent(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca.descendant_or_equal(child, parent)

    def test_parent_is_not_descendant_of_child(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert not lca.descendant_or_equal(parent, child)

    def test_leaf_is_descendant_of_root(self, example_tree):
        lca = LCA(example_tree)
        root = example_tree.root
        for leaf in example_tree.leaves():
            assert lca.descendant_or_equal(leaf, root)

    def test_descendant_not_equal_false_for_self(self, example_tree):
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert not lca.descendant_not_equal(v, v)

    def test_descendant_not_equal_true_for_child_parent(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca.descendant_not_equal(child, parent)

    def test_ancestor_or_equal_and_descendant_or_equal_are_symmetric(self, random_tree_20):
        """ancestor_or_equal(u, v) iff descendant_or_equal(v, u)."""
        lca = LCA(random_tree_20)
        nodes = list(random_tree_20.preorder())
        for u, v in itertools.islice(itertools.combinations(nodes, 2), 40):
            assert lca.ancestor_or_equal(u, v) == lca.descendant_or_equal(v, u)


# ===========================================================================
# are_comparable
# ===========================================================================


class TestAreComparable:
    def test_node_is_comparable_with_itself(self, example_tree):
        lca = LCA(example_tree)
        for v in example_tree.preorder():
            assert lca.are_comparable(v, v)

    def test_parent_and_child_are_comparable(self, example_tree):
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            assert lca.are_comparable(parent, child)
            assert lca.are_comparable(child, parent)

    def test_siblings_are_not_comparable(self, example_tree):
        """Two leaves from different child subtrees of the same inner node
        must not be comparable."""
        lca = LCA(example_tree)
        leaf_dict = example_tree.leaf_dict()
        for node in example_tree.inner_nodes():
            children = list(node.children)
            if len(children) < 2:
                continue
            c1, c2 = children[0], children[1]
            for a in leaf_dict[c1][:2]:
                for b in leaf_dict[c2][:2]:
                    assert not lca.are_comparable(a, b)

    def test_root_is_comparable_to_all(self, example_tree):
        lca = LCA(example_tree)
        root = example_tree.root
        for v in example_tree.preorder():
            assert lca.are_comparable(root, v)

    def test_comparability_is_symmetric(self, random_tree_20):
        lca = LCA(random_tree_20)
        nodes = list(random_tree_20.preorder())
        for u, v in itertools.islice(itertools.combinations(nodes, 2), 40):
            assert lca.are_comparable(u, v) == lca.are_comparable(v, u)


# ===========================================================================
# Edge-based interface (ancestor_or_equal / are_comparable with edges)
# ===========================================================================


class TestEdgeInterface:
    """Methods are_comparable and ancestor_or_equal accept edge tuples in addition to nodes."""

    def test_edge_ancestor_of_descendant_node(self, example_tree):
        """Edge (u, v) should be an ancestor-or-equal of any node in the subtree of v."""
        lca = LCA(example_tree)
        for parent, child in example_tree.inner_edges():
            # child is an inner node; pick any leaf in its subtree
            leaf = next(example_tree.traverse_subtree(child))
            if not leaf.is_leaf():
                continue
            assert lca.ancestor_or_equal((parent, child), leaf)

    def test_node_ancestor_of_edge(self, example_tree):
        """A node u should be ancestor-or-equal of an edge (u, v) or (ancestor, u)."""
        lca = LCA(example_tree)
        for parent, child in example_tree.edges():
            # parent is ancestor_or_equal of edge (parent, child)
            assert lca.ancestor_or_equal(parent, (parent, child))

    def test_edge_comparable_to_ancestor_edge(self, example_tree):
        """Two edges on the same root-to-leaf path are comparable."""
        lca = LCA(example_tree)
        # Build a root-to-leaf path
        leaf = next(example_tree.leaves())
        path: list[TreeNode] = []
        node = leaf
        while node is not None:
            path.append(node)
            node = node.parent
        path.reverse()  # root first

        # Each consecutive pair of edges on the path must be comparable
        for i in range(len(path) - 2):
            e1 = (path[i], path[i + 1])
            e2 = (path[i + 1], path[i + 2])
            assert lca.are_comparable(e1, e2)


# ===========================================================================
# consistent_triples / consistent_triple_generator
# ===========================================================================


class TestConsistentTriples:
    def test_empty_input_returns_empty_list(self, example_tree):
        lca = LCA(example_tree)
        assert lca.consistent_triples([]) == []

    def test_all_displayed_triples_are_kept(self, example_tree):
        """consistent_triples must retain all triples that the tree displays."""
        lca = LCA(example_tree)
        displayed = [
            t for t in example_tree.get_triples(label_only=False) if lca.displays_triple(*t)
        ]
        result = lca.consistent_triples(displayed)
        assert set(result) == set(displayed)

    def test_non_displayed_triples_are_removed(self, example_tree):
        lca = LCA(example_tree)
        leaves = list(example_tree.leaves())
        # Build triples where a == b (never displayed)
        bad_triples = [(leaves[0], leaves[0], leaves[i]) for i in range(1, 4)]
        assert lca.consistent_triples(bad_triples) == []

    def test_consistent_triples_returns_list(self, example_tree):
        lca = LCA(example_tree)
        result = lca.consistent_triples([])
        assert isinstance(result, list)

    def test_generator_yields_same_as_list(self, example_tree):
        """consistent_triple_generator must yield the same items as consistent_triples."""
        lca = LCA(example_tree)
        triples = example_tree.get_triples(label_only=False)
        from_list = lca.consistent_triples(triples)
        from_gen = list(lca.consistent_triple_generator(triples))
        assert from_list == from_gen

    def test_generator_is_lazy(self, example_tree):
        """consistent_triple_generator must return an iterator, not a list."""
        import types

        lca = LCA(example_tree)
        gen = lca.consistent_triple_generator([])
        assert isinstance(gen, types.GeneratorType)
