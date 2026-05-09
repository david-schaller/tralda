"""Tests for tralda.datastructures.tree (TreeNode and Tree)."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree, TreeNode


# ===========================================================================
# TreeNode
# ===========================================================================


class TestTreeNode:
    """Unit tests for the TreeNode class."""

    # ── String representation ────────────────────────────────────────────────

    def test_str_with_label(self):
        node = TreeNode(label="foo")
        assert str(node) == "foo"

    def test_str_without_label(self):
        node = TreeNode()
        assert str(node) == ""

    def test_repr_with_label(self):
        node = TreeNode(label=42)
        assert "42" in repr(node)

    def test_repr_without_label(self):
        node = TreeNode()
        assert str(id(node)) in repr(node)

    # ── attributes() ────────────────────────────────────────────────────────

    def test_attributes_yields_user_attrs(self):
        node = TreeNode(label=1, weight=0.5)
        attrs = dict(node.attributes())
        assert attrs["label"] == 1
        assert attrs["weight"] == 0.5

    def test_attributes_excludes_internal_fields(self):
        node = TreeNode(label=1)
        keys = {k for k, _ in node.attributes()}
        assert "parent" not in keys
        assert "children" not in keys
        assert "_par_dll_node" not in keys

    # ── add_child ───────────────────────────────────────────────────────────

    def test_add_child_sets_parent(self):
        parent = TreeNode()
        child = TreeNode()
        parent.add_child(child)
        assert child.parent is parent

    def test_add_child_appears_in_children(self):
        parent = TreeNode()
        child = TreeNode()
        parent.add_child(child)
        assert child in list(parent.children)

    def test_add_child_idempotent(self):
        parent = TreeNode()
        child = TreeNode()
        parent.add_child(child)
        parent.add_child(child)  # duplicate – should be a no-op
        assert len(list(parent.children)) == 1

    def test_add_child_reparents_node(self):
        p1, p2, child = TreeNode(), TreeNode(), TreeNode()
        p1.add_child(child)
        p2.add_child(child)
        assert child.parent is p2
        assert child not in list(p1.children)

    def test_add_multiple_children_preserves_order(self):
        parent = TreeNode()
        children = [TreeNode(label=i) for i in range(4)]
        for c in children:
            parent.add_child(c)
        assert list(parent.children) == children

    # ── add_child_right_of ──────────────────────────────────────────────────

    def test_add_child_right_of_inserts_at_correct_position(self):
        parent = TreeNode()
        c1, c2, c3 = TreeNode(label=1), TreeNode(label=2), TreeNode(label=3)
        parent.add_child(c1)
        parent.add_child(c3)
        parent.add_child_right_of(c2, c1)
        children = list(parent.children)
        assert children.index(c2) == children.index(c1) + 1

    def test_add_child_right_of_raises_for_non_child_anchor(self):
        parent, other, child = TreeNode(), TreeNode(), TreeNode()
        with pytest.raises(KeyError):
            parent.add_child_right_of(child, other)

    # ── remove_child ────────────────────────────────────────────────────────

    def test_remove_child_clears_parent(self):
        parent, child = TreeNode(), TreeNode()
        parent.add_child(child)
        parent.remove_child(child)
        assert child.parent is None

    def test_remove_child_removes_from_children(self):
        parent, child = TreeNode(), TreeNode()
        parent.add_child(child)
        parent.remove_child(child)
        assert child not in list(parent.children)

    def test_remove_child_raises_for_non_child(self):
        parent, other = TreeNode(), TreeNode()
        with pytest.raises(KeyError):
            parent.remove_child(other)

    # ── detach ──────────────────────────────────────────────────────────────

    def test_detach_removes_from_parent(self):
        parent, child = TreeNode(), TreeNode()
        parent.add_child(child)
        child.detach()
        assert child.parent is None
        assert child not in list(parent.children)

    def test_detach_on_root_is_noop(self):
        root = TreeNode()
        root.detach()  # must not raise
        assert root.parent is None

    # ── is_leaf ─────────────────────────────────────────────────────────────

    def test_is_leaf_without_children(self):
        node = TreeNode()
        assert node.is_leaf()

    def test_is_leaf_with_child(self):
        parent, child = TreeNode(), TreeNode()
        parent.add_child(child)
        assert not parent.is_leaf()
        assert child.is_leaf()

    # ── child_subsequence ───────────────────────────────────────────────────

    def test_child_subsequence_returns_correct_range(self):
        parent = TreeNode()
        children = [TreeNode(label=i) for i in range(5)]
        for c in children:
            parent.add_child(c)
        sub = parent.child_subsequence(children[1], children[3])
        assert sub == children[1:4]

    def test_child_subsequence_raises_for_non_child_left(self):
        parent, c1, other = TreeNode(), TreeNode(), TreeNode()
        parent.add_child(c1)
        with pytest.raises(KeyError):
            parent.child_subsequence(other, c1)

    def test_child_subsequence_raises_for_non_child_right(self):
        parent, c1, other = TreeNode(), TreeNode(), TreeNode()
        parent.add_child(c1)
        with pytest.raises(KeyError):
            parent.child_subsequence(c1, other)


# ===========================================================================
# Tree – construction & basic properties
# ===========================================================================


class TestTreeConstruction:
    def test_init_with_tree_node(self):
        root = TreeNode(label="r")
        tree = Tree(root)
        assert tree.root is root

    def test_init_with_none(self):
        tree = Tree(None)
        assert tree.root is None

    def test_init_with_newick_string(self, example_newick):
        tree = Tree(example_newick)
        assert tree.root is not None

    def test_init_invalid_type_raises(self):
        with pytest.raises(TypeError):
            Tree(123)

    def test_len(self, example_tree):
        assert len(example_tree) == 31

    def test_height(self, example_tree):
        assert example_tree.height() == 6

    def test_height_empty_tree(self):
        assert Tree(None).height() == -1

    def test_len_single_node(self):
        assert len(Tree(TreeNode())) == 1


# ===========================================================================
# Tree – traversals
# ===========================================================================


class TestTraversals:
    def test_preorder_visits_all_nodes(self, example_tree):
        assert len(list(example_tree.preorder())) == len(example_tree)

    def test_postorder_visits_all_nodes(self, example_tree):
        assert len(list(example_tree.postorder())) == len(example_tree)

    def test_pre_and_postorder_same_node_set(self, random_tree_20):
        assert set(random_tree_20.preorder()) == set(random_tree_20.postorder())

    def test_preorder_root_is_first(self, example_tree):
        assert next(example_tree.preorder()) is example_tree.root

    def test_postorder_root_is_last(self, example_tree):
        assert list(example_tree.postorder())[-1] is example_tree.root

    def test_postorder_leaves_before_parents(self, random_tree_20):
        """Every node must appear after all of its descendants in postorder."""
        index = {v: i for i, v in enumerate(random_tree_20.postorder())}
        for v in random_tree_20.preorder():
            for child in v.children:
                assert index[child] < index[v]

    def test_preorder_and_level_root_at_level_0(self, example_tree):
        root_node, level = next(example_tree.preorder_and_level())
        assert root_node is example_tree.root
        assert level == 0

    def test_preorder_and_level_count(self, example_tree):
        assert len(list(example_tree.preorder_and_level())) == len(example_tree)

    def test_preorder_and_level_values(self, random_tree_20):
        """Level must equal the depth of the node from the root."""
        for node, level in random_tree_20.preorder_and_level():
            depth = 0
            current = node
            while current.parent is not None:
                depth += 1
                current = current.parent
            assert level == depth

    def test_leaves_are_subset_of_preorder(self, random_tree_20):
        assert set(random_tree_20.leaves()).issubset(set(random_tree_20.preorder()))

    def test_all_leaves_have_no_children(self, random_tree_20):
        for leaf in random_tree_20.leaves():
            assert leaf.is_leaf()

    def test_inner_nodes_have_children(self, example_tree):
        for v in example_tree.inner_nodes():
            assert not v.is_leaf()

    def test_leaves_and_inner_nodes_partition_all_nodes(self, random_tree_20):
        leaves = set(random_tree_20.leaves())
        inner = set(random_tree_20.inner_nodes())
        all_nodes = set(random_tree_20.preorder())
        assert leaves | inner == all_nodes
        assert leaves & inner == set()

    def test_edges_count(self, random_tree_20):
        assert len(list(random_tree_20.edges())) == len(random_tree_20) - 1

    def test_edges_parent_child_relationship(self, example_tree):
        for u, v in example_tree.edges():
            assert v.parent is u

    def test_inner_edges_are_subset_of_all_edges(self, example_tree):
        assert set(example_tree.inner_edges()).issubset(set(example_tree.edges()))

    def test_inner_edges_child_is_not_leaf(self, example_tree):
        for _, v in example_tree.inner_edges():
            assert not v.is_leaf()

    def test_edges_sibling_order_indices(self, example_tree):
        for u, v, idx in example_tree.edges_sibling_order():
            children = list(u.children)
            assert children[idx] is v

    def test_traverse_subtree_subset_of_all(self, example_tree):
        inner = next(v for v in example_tree.inner_nodes() if v is not example_tree.root)
        sub = set(example_tree.traverse_subtree(inner))
        assert sub.issubset(set(example_tree.preorder()))
        assert inner in sub

    def test_traverse_subtree_contains_inner_node(self, example_tree):
        inner = next(example_tree.inner_nodes())
        assert inner in set(example_tree.traverse_subtree(inner))

    def test_euler_generator_length(self, random_tree_20):
        # Euler tour of a tree with n nodes has length 2n - 1
        n = len(random_tree_20)
        assert len(list(random_tree_20.euler_generator())) == 2 * n - 1

    def test_euler_generator_root_first(self, example_tree):
        assert next(example_tree.euler_generator()) is example_tree.root

    def test_euler_and_level_root_appears_first(self, example_tree):
        node, level = next(example_tree.euler_and_level())
        assert node is example_tree.root
        assert level == 0

    def test_empty_tree_all_traversals_empty(self):
        empty = Tree(None)
        assert list(empty.preorder()) == []
        assert list(empty.postorder()) == []
        assert list(empty.leaves()) == []
        assert list(empty.inner_nodes()) == []
        assert list(empty.edges()) == []
        assert list(empty.euler_generator()) == []
        assert list(empty.preorder_and_level()) == []
        assert list(empty.euler_and_level()) == []


# ===========================================================================
# Tree – structural queries
# ===========================================================================


class TestStructuralQueries:
    def test_leaf_dict_leaf_maps_to_itself(self, example_tree):
        leaf_dict = example_tree.leaf_dict()
        for leaf in example_tree.leaves():
            assert leaf_dict[leaf] == [leaf]

    def test_leaf_dict_root_contains_all_leaves(self, example_tree):
        leaf_dict = example_tree.leaf_dict()
        all_leaves = set(example_tree.leaves())
        assert set(leaf_dict[example_tree.root]) == all_leaves

    def test_leaf_dict_subtree_leaves_match(self, example_tree):
        leaf_dict = example_tree.leaf_dict()
        for v in example_tree.inner_nodes():
            expected = set(example_tree.traverse_subtree(v)) & set(example_tree.leaves())
            assert set(leaf_dict[v]) == expected

    def test_is_binary_on_binary_tree(self):
        tree = Tree.random_tree(15, binary=True)
        assert tree.is_binary()

    def test_is_binary_false_on_non_binary(self):
        # Build a simple star tree (root with 3 leaves) – not binary
        root = TreeNode()
        for i in range(3):
            root.add_child(TreeNode(label=i))
        tree = Tree(root)
        assert not tree.is_binary()

    def test_is_phylogenetic_random_tree(self, random_tree_20):
        assert random_tree_20.is_phylogenetic()

    def test_is_phylogenetic_false_with_unary_node(self):
        root = TreeNode()
        only_child = TreeNode()
        leaf = TreeNode()
        root.add_child(only_child)
        only_child.add_child(leaf)
        tree = Tree(root)
        assert not tree.is_phylogenetic()

    def test_get_hierarchy_size_equals_node_count(self, example_tree):
        # Each node defines a unique cluster so |hierarchy| == |nodes|
        assert len(example_tree.get_hierarchy()) == len(example_tree)

    def test_equal_topology_with_self(self, random_tree_20):
        assert random_tree_20.equal_topology(random_tree_20)

    def test_equal_topology_with_copy(self, random_tree_20):
        assert random_tree_20.equal_topology(random_tree_20.copy())

    def test_equal_topology_returns_bool(self):
        t1 = Tree.random_tree(8, binary=True)
        t2 = Tree.random_tree(8, binary=True)
        assert isinstance(t1.equal_topology(t2), bool)

    def test_is_refinement_of_self(self, random_tree_20):
        assert random_tree_20.is_refinement(random_tree_20)

    def test_refinement_direction(self, example_tree):
        """Contracting an inner edge yields a coarser tree.

        The original must be a refinement of the coarser tree, but not vice versa.
        """
        inner_edges = list(example_tree.inner_edges())
        if not inner_edges:
            pytest.skip("no inner edges in fixture tree")
        coarser, mapping = example_tree.copy(mapping=True)
        u_orig, v_orig = inner_edges[0]
        coarser.contract([(mapping[u_orig], mapping[v_orig])])

        assert example_tree.is_refinement(coarser)
        assert not coarser.is_refinement(example_tree)

    def test_get_triples_returns_list(self, example_tree):
        assert isinstance(example_tree.get_triples(), list)

    def test_get_triples_node_version_contains_tree_nodes(self, example_tree):
        for triple in example_tree.get_triples(label_only=False):
            assert all(isinstance(x, TreeNode) for x in triple)

    def test_get_triples_label_only_contains_primitives(self, example_tree):
        for triple in example_tree.get_triples(label_only=True):
            assert all(isinstance(x, (int, str)) for x in triple)


# ===========================================================================
# Tree – modification operations
# ===========================================================================


class TestTreeModification:
    def test_delete_and_reconnect_attaches_grandchildren_to_grandparent(self, example_tree):
        node = next(v for v in example_tree.inner_nodes() if v is not example_tree.root)
        parent = node.parent
        grandchildren = list(node.children)
        result = example_tree.delete_and_reconnect(node)
        assert result is parent
        for gc in grandchildren:
            assert gc.parent is parent

    def test_delete_and_reconnect_root_returns_none(self, example_tree):
        assert example_tree.delete_and_reconnect(example_tree.root) is None

    def test_contract_inplace_reduces_size(self):
        tree = Tree.random_tree(15)
        inner_edges = list(tree.inner_edges())
        if not inner_edges:
            pytest.skip("no inner edges")
        original_size = len(tree)
        tree.contract([inner_edges[0]])
        assert len(tree) == original_size - 1

    def test_contract_inplace_returns_same_object(self):
        tree = Tree.random_tree(15)
        inner_edges = list(tree.inner_edges())
        if not inner_edges:
            pytest.skip("no inner edges")
        result = tree.contract([inner_edges[0]])
        assert result is tree

    def test_contract_not_inplace_preserves_original(self):
        tree = Tree.random_tree(15)
        inner_edges = list(tree.inner_edges())
        if not inner_edges:
            pytest.skip("no inner edges")
        original_size = len(tree)
        contracted = tree.contract([inner_edges[0]], inplace=False)
        assert len(tree) == original_size
        assert len(contracted) == original_size - 1

    def test_contract_not_inplace_returns_distinct_object(self):
        tree = Tree.random_tree(15)
        inner_edges = list(tree.inner_edges())
        if not inner_edges:
            pytest.skip("no inner edges")
        contracted = tree.contract([inner_edges[0]], inplace=False)
        assert contracted is not tree

    def test_random_leaves_count(self, random_tree_20):
        n_leaves = sum(1 for _ in random_tree_20.leaves())
        sample = random_tree_20.random_leaves(0.5)
        assert len(sample) == round(0.5 * n_leaves)

    def test_random_leaves_are_leaves(self, random_tree_20):
        for leaf in random_tree_20.random_leaves(0.6):
            assert leaf.is_leaf()

    def test_random_leaves_proportion_zero(self, random_tree_20):
        assert random_tree_20.random_leaves(0.0) == []

    def test_random_leaves_invalid_proportion_raises(self, random_tree_20):
        with pytest.raises(ValueError):
            random_tree_20.random_leaves(1.5)
        with pytest.raises(ValueError):
            random_tree_20.random_leaves(-0.1)


# ===========================================================================
# Tree – copy
# ===========================================================================


class TestCopy:
    def test_copy_equal_topology(self, random_tree_20):
        assert random_tree_20.equal_topology(random_tree_20.copy())

    def test_copy_nodes_are_distinct_objects(self, random_tree_20):
        orig_nodes = set(random_tree_20.preorder())
        copy_nodes = set(random_tree_20.copy().preorder())
        assert orig_nodes.isdisjoint(copy_nodes)

    def test_copy_preserves_labels(self, example_tree):
        for orig, dup in zip(example_tree.preorder(), example_tree.copy().preorder()):
            assert orig.label == dup.label

    def test_copy_with_mapping_node_count(self, random_tree_20):
        _, mapping = random_tree_20.copy(mapping=True)
        assert len(mapping) == len(random_tree_20)

    def test_copy_with_mapping_distinct_nodes(self, random_tree_20):
        _, mapping = random_tree_20.copy(mapping=True)
        for orig, new in mapping.items():
            assert orig is not new

    def test_copy_empty_tree(self):
        copy = Tree(None).copy()
        assert copy.root is None

    def test_copy_integrity(self, random_tree_20):
        assert random_tree_20.copy()._assert_integrity()


# ===========================================================================
# Tree – Newick I/O
# ===========================================================================


class TestNewick:
    def test_parse_newick_node_count(self, example_tree):
        assert len(example_tree) == 31

    def test_round_trip_topology(self, random_tree_20):
        reloaded = Tree.parse_newick(random_tree_20.to_newick())
        assert random_tree_20.equal_topology(reloaded)

    def test_round_trip_labels(self, example_tree):
        reloaded = Tree.parse_newick(example_tree.to_newick())
        for orig, dup in zip(example_tree.preorder(), reloaded.preorder()):
            assert orig.label == dup.label

    def test_round_trip_distances(self):
        tree = Tree.random_tree(10)
        for v in tree.preorder():
            v.dist = round(random.random(), 6)
        reloaded = Tree.parse_newick(tree.to_newick())
        for orig, dup in zip(tree.preorder(), reloaded.preorder()):
            assert abs(orig.dist - dup.dist) <= 1e-6

    def test_subtree_newick_ends_with_semicolon(self, example_tree):
        inner = next(example_tree.inner_nodes())
        assert example_tree.to_newick(node=inner).endswith(";")

    def test_empty_tree_newick(self):
        assert Tree(None).to_newick() == ";"

    def test_parse_newick_invalid_type_raises(self):
        with pytest.raises(TypeError):
            Tree._parse_newick_and_return_root(42)

    def test_parse_newick_unbalanced_parens_raises(self):
        with pytest.raises(ValueError):
            Tree.parse_newick("((a,b)")

    def test_parse_newick_integer_labels(self, example_tree):
        for v in example_tree.preorder():
            assert isinstance(v.label, int)


# ===========================================================================
# Tree – NetworkX I/O
# ===========================================================================


class TestNetworkX:
    def test_round_trip_topology(self, random_tree_20):
        graph, root_id = random_tree_20.to_nx()
        reloaded = Tree.parse_nx(graph, root_id)
        assert random_tree_20.equal_topology(reloaded)

    def test_to_nx_node_count(self, random_tree_20):
        graph, _ = random_tree_20.to_nx()
        assert graph.number_of_nodes() == len(random_tree_20)

    def test_to_nx_edge_count(self, random_tree_20):
        graph, _ = random_tree_20.to_nx()
        assert graph.number_of_edges() == len(random_tree_20) - 1

    def test_parse_nx_none_root_returns_empty_tree(self):
        import networkx as nx

        result = Tree.parse_nx(nx.DiGraph(), None)
        assert result.root is None


# ===========================================================================
# Tree – dict serialization
# ===========================================================================


class TestDictSerialization:
    def test_round_trip_topology(self, random_tree_20):
        reloaded = Tree.parse_dict(random_tree_20.to_dict())
        assert random_tree_20.equal_topology(reloaded)

    def test_to_dict_empty_tree_raises(self):
        with pytest.raises(RuntimeError):
            Tree(None).to_dict()


# ===========================================================================
# Tree – file serialization
# ===========================================================================


class TestFileSerialization:
    def test_pickle_round_trip(self, random_tree_20, tmp_path):
        path = str(tmp_path / "tree.pickle")
        random_tree_20.serialize(path)
        assert random_tree_20.equal_topology(Tree.load(path))

    def test_json_round_trip(self, random_tree_20, tmp_path):
        path = str(tmp_path / "tree.json")
        random_tree_20.serialize(path)
        assert random_tree_20.equal_topology(Tree.load(path))

    def test_serialize_unknown_extension_raises(self, random_tree_20, tmp_path):
        with pytest.raises(ValueError):
            random_tree_20.serialize(str(tmp_path / "tree.xyz"))

    def test_load_unknown_extension_raises(self, tmp_path):
        with pytest.raises(ValueError):
            Tree.load(str(tmp_path / "tree.xyz"))

    def test_serialize_unknown_mode_raises(self, random_tree_20, tmp_path):
        with pytest.raises(ValueError):
            random_tree_20.serialize(str(tmp_path / "tree.pickle"), mode="csv")


# ===========================================================================
# Tree – print_tree / _lines_for_print_tree
# ===========================================================================


class TestPrintTree:
    _EXPECTED_LINES = [
        "0",
        "├───1",
        "│   ├───7",
        "│   │   ├───10",
        "│   │   │   ├───14",
        "│   │   │   ├───15",
        "│   │   │   └───18",
        "│   │   │       ├───27",
        "│   │   │       └───28",
        "│   │   │           ├───29",
        "│   │   │           └───30",
        "│   │   ├───11",
        "│   │   │   ├───19",
        "│   │   │   └───20",
        "│   │   └───16",
        "│   ├───8",
        "│   ├───9",
        "│   └───17",
        "├───2",
        "│   ├───4",
        "│   └───5",
        "├───3",
        "├───6",
        "│   ├───12",
        "│   │   ├───23",
        "│   │   └───24",
        "│   ├───13",
        "│   └───22",
        "└───21",
        "    ├───25",
        "    └───26",
    ]

    def test_lines_for_print_tree(self, example_tree):
        assert example_tree._lines_for_print_tree(4) == self._EXPECTED_LINES

    def test_print_tree_invalid_indentation_raises(self, example_tree):
        with pytest.raises(ValueError):
            example_tree.print_tree(0)
        with pytest.raises(ValueError):
            example_tree.print_tree(-1)

    def test_print_tree_produces_output(self, example_tree, capsys):
        example_tree.print_tree(3)
        assert len(capsys.readouterr().out) > 0

    def test_print_tree_line_count(self, example_tree, capsys):
        example_tree.print_tree(3)
        lines = capsys.readouterr().out.strip().splitlines()
        assert len(lines) == len(example_tree)


# ===========================================================================
# Tree – random_tree factory
# ===========================================================================


class TestRandomTree:
    @pytest.mark.parametrize("n", [1, 5, 10, 50])
    def test_leaf_count(self, n):
        tree = Tree.random_tree(n)
        assert sum(1 for _ in tree.leaves()) == n

    def test_is_phylogenetic(self):
        assert Tree.random_tree(20).is_phylogenetic()

    def test_binary_flag(self):
        assert Tree.random_tree(20, binary=True).is_binary()

    @pytest.mark.parametrize("bad_value", [3.5, 0, -1, "five"])
    def test_invalid_argument_raises(self, bad_value):
        with pytest.raises(TypeError):
            Tree.random_tree(bad_value)


# ===========================================================================
# Tree – integrity
# ===========================================================================


class TestIntegrity:
    def test_random_tree_passes_integrity(self, random_tree_20):
        assert random_tree_20._assert_integrity()

    def test_example_tree_passes_integrity(self, example_tree):
        assert example_tree._assert_integrity()

    def test_copy_passes_integrity(self, random_tree_20):
        assert random_tree_20.copy()._assert_integrity()
