"""Tests for tralda.datastructures.hdtgraph (HDTGraph and ETTree)."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from tralda.datastructures import HDTGraph, Tree
from tralda.datastructures.hdtgraph.et_tree import ETTree, ETTreeNode, EdgeOccurrences


# ===========================================================================
# Helpers
# ===========================================================================


def _nx_graph(edges: list[tuple]) -> nx.Graph:
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def _assert_connectivity_matches(hdt: HDTGraph, ref: nx.Graph, nodes: list) -> None:
    """Assert that HDTGraph.connected matches NetworkX for all pairs in *nodes*."""
    for u in nodes:
        for v in nodes:
            if not ref.has_node(u) or not ref.has_node(v):
                continue
            if not hdt.has_node(u) or not hdt.has_node(v):
                continue
            assert hdt.connected(u, v) == nx.has_path(ref, u, v), f"connected({u}, {v}) mismatch"


def _build_interleaved(seed: int, n_nodes: int = 12, n_ops: int = 300):
    """Return an (HDTGraph, nx.Graph) pair built by interleaved random inserts/deletes."""
    rng = random.Random(seed)
    G_hdt = HDTGraph()
    G_ref = nx.Graph()
    for _ in range(n_ops):
        op = rng.choice(["insert", "delete"])
        a = rng.randint(0, n_nodes - 1)
        b = rng.randint(0, n_nodes - 1)
        if a == b:
            continue
        if op == "insert":
            G_hdt.insert_edge(a, b)
            G_ref.add_edge(a, b)
        else:
            G_hdt.delete_edge(a, b)
            if G_ref.has_edge(a, b):
                G_ref.remove_edge(a, b)
    return G_hdt, G_ref


def _check_ett_integrity(hdt: HDTGraph) -> None:
    """Assert ETTree integrity invariants at every level."""
    for level in hdt._levels:
        for ett in level.forest:
            assert ett.check_integrity(
                verbose=True,
                check_ett_properties=True,
                edge2occurrences=level.edge2occurrences,
            ), f"ETTree integrity check failed at level {level.index}"


# ===========================================================================
# HDTGraph – construction and basic node/edge API
# ===========================================================================


class TestHDTGraphConstruction:
    def test_empty_graph_has_no_nodes(self):
        G = HDTGraph()
        assert list(G.get_nodes()) == []

    def test_empty_graph_has_no_edges(self):
        G = HDTGraph()
        assert list(G.get_edges()) == []

    def test_insert_node_adds_node(self):
        G = HDTGraph()
        G.insert_node(1)
        assert G.has_node(1)

    def test_insert_node_duplicate_raises(self):
        G = HDTGraph()
        G.insert_node(42)
        with pytest.raises(KeyError):
            G.insert_node(42)

    def test_insert_edge_creates_both_nodes(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        assert G.has_node(1)
        assert G.has_node(2)

    def test_insert_edge_idempotent(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(1, 2)  # duplicate – must be a no-op
        assert len(list(G.get_edges())) == 1

    def test_insert_edge_reverse_idempotent(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(2, 1)  # reversed – same undirected edge
        assert len(list(G.get_edges())) == 1

    @pytest.mark.parametrize("node", [0, "a", (1, 2), 3.14])
    def test_insert_node_various_hashable_types(self, node):
        G = HDTGraph()
        G.insert_node(node)
        assert G.has_node(node)

    def test_has_node_false_for_missing(self):
        G = HDTGraph()
        assert not G.has_node(99)

    def test_has_edge_true_both_directions(self):
        G = HDTGraph()
        G.insert_edge("x", "y")
        assert G.has_edge("x", "y")
        assert G.has_edge("y", "x")

    def test_has_edge_false_for_missing(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        assert not G.has_edge(1, 3)
        assert not G.has_edge(99, 100)

    def test_get_nodes_returns_all_nodes(self):
        G = HDTGraph()
        for u, v in [(1, 2), (2, 3), (4, 5)]:
            G.insert_edge(u, v)
        assert set(G.get_nodes()) == {1, 2, 3, 4, 5}

    def test_get_edges_returns_all_edges(self):
        G = HDTGraph()
        expected = {(1, 2), (2, 3), (3, 4)}
        for u, v in expected:
            G.insert_edge(u, v)
        # edges are sorted by sort_edge, so normalise
        actual = {tuple(sorted(e)) for e in G.get_edges()}
        assert actual == expected


# ===========================================================================
# HDTGraph – delete_edge
# ===========================================================================


class TestDeleteEdge:
    def test_delete_nonexistent_edge_is_silent(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.delete_edge(1, 99)  # should not raise

    def test_delete_edge_removes_from_get_edges(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(2, 3)
        G.delete_edge(1, 2)
        edges = {tuple(sorted(e)) for e in G.get_edges()}
        assert (1, 2) not in edges
        assert (2, 3) in edges

    def test_delete_nontree_edge(self):
        # insert a cycle so one edge is a non-tree edge
        G = HDTGraph()
        for u, v in [(1, 2), (2, 3), (3, 1)]:
            G.insert_edge(u, v)
        G.delete_edge(3, 1)
        assert not G.has_edge(3, 1)
        # remaining nodes still connected
        assert G.connected(1, 3)

    def test_delete_tree_edge_without_replacement(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(2, 3)
        G.delete_edge(1, 2)
        assert not G.connected(1, 3)
        assert G.connected(2, 3)

    def test_delete_tree_edge_with_replacement(self):
        G = HDTGraph()
        for u, v in [(1, 2), (2, 3), (3, 4), (4, 1)]:
            G.insert_edge(u, v)
        G.delete_edge(1, 2)
        # All nodes remain connected via the cycle's alternative path
        for u in range(1, 5):
            for v in range(1, 5):
                assert G.connected(u, v)

    @pytest.mark.parametrize("seed", [0, 7, 42, 137, 999])
    def test_delete_matches_networkx(self, seed):
        G_hdt, G_ref = _build_interleaved(seed)
        nodes = list(range(12))
        _assert_connectivity_matches(G_hdt, G_ref, nodes)


# ===========================================================================
# HDTGraph – connected
# ===========================================================================


class TestConnected:
    def test_connected_reflexive_for_present_node(self):
        G = HDTGraph()
        G.insert_node(5)
        assert G.connected(5, 5)

    def test_connected_false_for_absent_node(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        assert not G.connected(1, 99)
        assert not G.connected(99, 1)
        assert not G.connected(99, 99)

    def test_connected_true_after_insert_edge(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        assert G.connected(1, 2)
        assert G.connected(2, 1)

    def test_connected_false_for_isolated_nodes(self):
        G = HDTGraph()
        G.insert_node(1)
        G.insert_node(2)
        assert not G.connected(1, 2)

    def test_connected_via_path(self):
        G = HDTGraph()
        for u, v in [(1, 2), (2, 3), (3, 4)]:
            G.insert_edge(u, v)
        assert G.connected(1, 4)
        assert G.connected(4, 1)

    def test_not_connected_after_bridge_deletion(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(2, 3)
        G.delete_edge(2, 3)
        assert not G.connected(1, 3)

    @pytest.mark.parametrize("seed", [1, 17, 271, 314, 1000, 2718, 9999])
    def test_interleaved_ops_match_networkx(self, seed):
        G_hdt, G_ref = _build_interleaved(seed, n_nodes=10, n_ops=400)
        nodes = list(range(10))
        _assert_connectivity_matches(G_hdt, G_ref, nodes)


# ===========================================================================
# HDTGraph – is_connected
# ===========================================================================


class TestIsConnected:
    def test_empty_graph_not_connected(self):
        assert not HDTGraph().is_connected()

    def test_single_node_is_connected(self):
        G = HDTGraph()
        G.insert_node(1)
        assert G.is_connected()

    def test_two_isolated_nodes_not_connected(self):
        G = HDTGraph()
        G.insert_node(1)
        G.insert_node(2)
        assert not G.is_connected()

    def test_connected_after_linking_isolated_nodes(self):
        G = HDTGraph()
        G.insert_node(1)
        G.insert_node(2)
        G.insert_edge(1, 2)
        assert G.is_connected()

    def test_not_connected_after_bridge_removal(self):
        G = HDTGraph()
        for u, v in [(1, 2), (2, 3), (3, 4)]:
            G.insert_edge(u, v)
        assert G.is_connected()
        G.delete_edge(2, 3)
        assert not G.is_connected()

    @pytest.mark.parametrize(
        "edges, expected",
        [
            ([(1, 2), (2, 3), (3, 1)], True),
            ([(1, 2), (3, 4)], False),
            ([(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)], True),
        ],
    )
    def test_is_connected_static_graphs(self, edges, expected):
        G = HDTGraph()
        for u, v in edges:
            G.insert_edge(u, v)
        assert G.is_connected() == expected


# ===========================================================================
# HDTGraph – component_iterator
# ===========================================================================


class TestComponentIterator:
    def test_isolated_node_component(self):
        G = HDTGraph()
        G.insert_node(7)
        assert list(G.component_iterator(7)) == [7]

    def test_missing_node_raises(self):
        G = HDTGraph()
        with pytest.raises(KeyError):
            list(G.component_iterator(99))

    def test_component_matches_networkx(self):
        G_hdt = HDTGraph()
        G_ref = nx.Graph()
        for u, v in [(1, 2), (2, 3), (5, 6)]:
            G_hdt.insert_edge(u, v)
            G_ref.add_edge(u, v)
        for node in G_hdt.get_nodes():
            assert set(G_hdt.component_iterator(node)) == nx.node_connected_component(G_ref, node)

    @pytest.mark.parametrize("seed", [5, 50, 500, 5000])
    def test_component_iterator_matches_networkx_random(self, seed):
        G_hdt, G_ref = _build_interleaved(seed, n_nodes=8, n_ops=150)
        for node in list(G_hdt.get_nodes()):
            expected = nx.node_connected_component(G_ref, node)
            assert set(G_hdt.component_iterator(node)) == expected

    def test_all_components_partition_nodes(self):
        G = HDTGraph()
        for u, v in [(1, 2), (3, 4), (5, 6), (2, 3)]:
            G.insert_edge(u, v)
        seen = set()
        for node in G.get_nodes():
            comp = frozenset(G.component_iterator(node))
            seen.add(comp)
        # union of all component sets equals full node set
        assert set().union(*seen) == set(G.get_nodes())


# ===========================================================================
# HDTGraph – get_component
# ===========================================================================


class TestGetComponent:
    def test_get_component_missing_node_raises(self):
        G = HDTGraph()
        with pytest.raises(KeyError):
            G.get_component(42)

    def test_get_component_returns_ett(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        ett = G.get_component(1)
        assert isinstance(ett, ETTree)

    def test_same_component_for_connected_nodes(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_edge(2, 3)
        assert G.get_component(1) is G.get_component(3)

    def test_different_components_for_disconnected_nodes(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.insert_node(3)
        assert G.get_component(1) is not G.get_component(3)


# ===========================================================================
# HDTGraph – add_loose_tree
# ===========================================================================


class TestAddLooseTree:
    def test_add_loose_tree_wrong_type_raises(self):
        G = HDTGraph()
        with pytest.raises(TypeError):
            G.add_loose_tree("not a tree")

    def test_add_loose_tree_single_node(self):
        T = Tree.parse_newick("a;")
        G = HDTGraph()
        G.add_loose_tree(T)
        assert G.has_node(T.root)
        assert G.is_connected()

    def test_add_loose_tree_connectivity(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        G = HDTGraph()
        G.add_loose_tree(T)
        # all tree nodes must be in the same component
        nodes = list(T.preorder())
        for u in nodes:
            for v in nodes:
                assert G.connected(u, v), f"nodes {u} and {v} should be connected"

    def test_add_loose_tree_edge_count(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        G = HDTGraph()
        G.add_loose_tree(T)
        tree_nodes = list(T.preorder())
        n = len(tree_nodes)
        # a tree on n nodes has n-1 edges
        assert len(list(G.get_edges())) == n - 1

    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
    def test_add_loose_tree_random_tree(self, seed):
        random.seed(seed)
        T = Tree.random_tree(10)
        G = HDTGraph()
        G.add_loose_tree(T)
        nodes = list(T.preorder())
        # all nodes connected
        root = T.root
        for node in nodes:
            assert G.connected(root, node)
        # edge count = n - 1
        assert len(list(G.get_edges())) == len(nodes) - 1

    def test_add_loose_tree_ett_integrity(self):
        T = Tree.parse_newick("(((a,b),c),(d,(e,f)));")
        G = HDTGraph()
        G.add_loose_tree(T)
        _check_ett_integrity(G)


# ===========================================================================
# HDTGraph – print_ett_forest
# ===========================================================================


class TestPrintEttForest:
    def test_invalid_level_type_raises_value_error(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        with pytest.raises(ValueError):
            G.print_ett_forest(level=1.5)

    def test_invalid_level_string_raises_value_error(self):
        G = HDTGraph()
        G.insert_edge(1, 2)
        with pytest.raises(ValueError):
            G.print_ett_forest(level="bad")

    def test_valid_all_does_not_raise(self, capsys):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.print_ett_forest(level="all")  # should not raise

    def test_valid_integer_level_does_not_raise(self, capsys):
        G = HDTGraph()
        G.insert_edge(1, 2)
        G.print_ett_forest(level=0)  # should not raise


# ===========================================================================
# ETTree – basic properties and integrity
# ===========================================================================


class TestETTreeBasic:
    def _build_linear_ett(self, n: int) -> tuple[ETTree, list[ETTreeNode], dict]:
        """Build an ETTree as a linear sequence of n nodes."""
        ett = ETTree()
        edge2occ: dict = {}
        nodes = []
        prev = None
        for i in range(n):
            node = ETTreeNode(i, active=True)
            nodes.append(node)
            ett.add_right_child_and_rebalance(prev, node, edge2occ)
            prev = node
        return ett

    def test_num_active_occurrences_single_node(self):
        ett = ETTree()
        node = ETTreeNode(0, active=True)
        ett.add_right_child_and_rebalance(None, node, {})
        assert ett.num_active_occurrences == 1

    def test_num_active_occurrences_sequence(self):
        ett = self._build_linear_ett(5)
        assert ett.num_active_occurrences == 5

    def test_num_active_occurrences_inactive(self):
        ett = ETTree()
        edge2occ = {}
        prev = None
        for i in range(4):
            node = ETTreeNode(i, active=(i % 2 == 0))  # only even indices active
            ett.add_right_child_and_rebalance(prev, node, edge2occ)
            prev = node
        assert ett.num_active_occurrences == 2

    def test_empty_ett_num_active(self):
        ett = ETTree()
        assert ett.num_active_occurrences == 0

    def test_integrity_linear_sequence(self):
        ett = self._build_linear_ett(8)
        assert ett.check_integrity(verbose=True, check_ett_properties=True)

    @pytest.mark.parametrize("n", [1, 2, 5, 10, 20])
    def test_integrity_various_sizes(self, n):
        ett = self._build_linear_ett(n)
        assert ett.check_integrity(verbose=True, check_ett_properties=True)


# ===========================================================================
# ETTreeNode – is_smaller
# ===========================================================================


class TestETTreeNodeIsSmaller:
    def _build_sequence(self, n: int) -> tuple[ETTree, list[ETTreeNode]]:
        ett = ETTree()
        nodes = []
        prev = None
        for i in range(n):
            node = ETTreeNode(i, active=True)
            nodes.append(node)
            ett.add_right_child_and_rebalance(prev, node, {})
            prev = node
        return ett, nodes

    def test_node_not_smaller_than_itself(self):
        _, nodes = self._build_sequence(3)
        assert not nodes[1].is_smaller(nodes[1])

    def test_earlier_node_is_smaller(self):
        _, nodes = self._build_sequence(5)
        for i in range(5):
            for j in range(i + 1, 5):
                assert nodes[i].is_smaller(nodes[j])

    def test_later_node_not_smaller(self):
        _, nodes = self._build_sequence(5)
        for i in range(5):
            for j in range(i):
                assert not nodes[i].is_smaller(nodes[j])

    def test_nodes_from_different_trees_raise(self):
        _, nodes_a = self._build_sequence(3)
        _, nodes_b = self._build_sequence(3)
        with pytest.raises(ValueError):
            nodes_a[0].is_smaller(nodes_b[0])

    @pytest.mark.parametrize("n", [2, 4, 8, 16])
    def test_is_smaller_consistent_for_various_sizes(self, n):
        _, nodes = self._build_sequence(n)
        for i in range(n):
            for j in range(n):
                assert nodes[i].is_smaller(nodes[j]) == (i < j)


# ===========================================================================
# ETTreeNode – get_root / path helpers
# ===========================================================================


class TestETTreeNodeGetRoot:
    def test_get_root_single_node(self):
        ett = ETTree()
        node = ETTreeNode(1, active=True)
        ett.add_right_child_and_rebalance(None, node, {})
        assert node.get_root() is ett.root

    def test_all_nodes_share_same_root(self):
        ett = ETTree()
        prev = None
        nodes = []
        for i in range(6):
            n = ETTreeNode(i, active=True)
            nodes.append(n)
            ett.add_right_child_and_rebalance(prev, n, {})
            prev = n
        root = ett.root
        for n in nodes:
            assert n.get_root() is root


# ===========================================================================
# ETTreeNode – find_inorder_predecessor / find_inorder_successor
# ===========================================================================


class TestInorderNeighbours:
    def _build_sequence(self, n: int):
        ett = ETTree()
        nodes = []
        prev = None
        for i in range(n):
            node = ETTreeNode(i, active=True)
            nodes.append(node)
            ett.add_right_child_and_rebalance(prev, node, {})
            prev = node
        return ett, nodes

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_predecessor_chain(self, n):
        _, nodes = self._build_sequence(n)
        # The first node has no predecessor
        assert nodes[0].find_inorder_predecessor() is None
        # Every other node's predecessor has a smaller insertion index
        for i in range(1, n):
            pred = nodes[i].find_inorder_predecessor()
            assert pred is not None
            assert pred.key < nodes[i].key

    @pytest.mark.parametrize("n", [3, 5, 8])
    def test_successor_chain(self, n):
        _, nodes = self._build_sequence(n)
        # The last node has no successor
        assert nodes[-1].find_inorder_successor() is None
        # Every other node's successor has a larger insertion index
        for i in range(n - 1):
            succ = nodes[i].find_inorder_successor()
            assert succ is not None
            assert succ.key > nodes[i].key


# ===========================================================================
# EdgeOccurrences
# ===========================================================================


class TestEdgeOccurrences:
    def test_update_dispatches_a_first_pair(self):
        """update(a_node, b_node) stores into _a1/_b1."""
        node_a = ETTreeNode("a", active=True)
        node_b = ETTreeNode("b", active=True)
        occ = EdgeOccurrences("a", "b")
        occ.update(node_a, node_b)
        assert occ._a1 is node_a
        assert occ._b1 is node_b
        # second pair not yet set
        assert occ._a2 is None
        assert occ._b2 is None

    def test_update_dispatches_b_first_pair(self):
        """update(b_node, a_node) stores into _b2/_a2."""
        node_b = ETTreeNode("b", active=True)
        node_a = ETTreeNode("a", active=True)
        occ = EdgeOccurrences("a", "b")
        occ.update(node_b, node_a)
        assert occ._b2 is node_b
        assert occ._a2 is node_a
        # first pair not yet set
        assert occ._a1 is None
        assert occ._b1 is None

    def test_update_both_pairs(self):
        """Two update() calls populate all four fields."""
        node_a1 = ETTreeNode("a", active=True)
        node_b1 = ETTreeNode("b", active=True)
        node_b2 = ETTreeNode("b", active=False)
        node_a2 = ETTreeNode("a", active=False)
        occ = EdgeOccurrences("a", "b")
        occ.update(node_a1, node_b1)
        occ.update(node_b2, node_a2)
        assert occ._a1 is node_a1
        assert occ._b1 is node_b1
        assert occ._b2 is node_b2
        assert occ._a2 is node_a2


# ===========================================================================
# Stress tests – ETTree integrity under random HDT operations
# ===========================================================================


class TestStressIntegrity:
    @pytest.mark.parametrize("seed", [0, 1, 2, 3, 42, 100, 200, 314, 999, 2718])
    def test_ett_integrity_after_random_ops(self, seed):
        G_hdt, _ = _build_interleaved(seed, n_nodes=10, n_ops=200)
        _check_ett_integrity(G_hdt)

    @pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
    def test_num_active_occurrences_matches_actual(self, seed):
        G_hdt, _ = _build_interleaved(seed, n_nodes=8, n_ops=150)
        level0 = G_hdt._levels[0]
        for ett in level0.forest:
            expected = sum(1 for occ in ett if occ.active)
            assert ett.num_active_occurrences == expected

    @pytest.mark.parametrize("seed", [7, 14, 28, 56, 112])
    def test_connectivity_matches_networkx_large(self, seed):
        G_hdt, G_ref = _build_interleaved(seed, n_nodes=20, n_ops=600)
        _assert_connectivity_matches(G_hdt, G_ref, list(range(20)))

    @pytest.mark.parametrize("seed", [3, 31, 314])
    def test_component_sizes_match_networkx(self, seed):
        G_hdt, G_ref = _build_interleaved(seed, n_nodes=12, n_ops=250)
        for node in G_hdt.get_nodes():
            hdt_size = G_hdt.get_component(node).num_active_occurrences
            ref_size = len(nx.node_connected_component(G_ref, node))
            assert hdt_size == ref_size, (
                f"seed={seed}, node={node}: component size {hdt_size} != {ref_size}"
            )


# ===========================================================================
# Large-graph randomised tests using _nx_graph as the reference oracle
# ===========================================================================


def _random_edges(rng: random.Random, n_nodes: int, n_edges: int) -> list[tuple]:
    """Generate a list of distinct random edges (no self-loops)."""
    edges = set()
    while len(edges) < n_edges:
        a = rng.randint(0, n_nodes - 1)
        b = rng.randint(0, n_nodes - 1)
        if a != b:
            edges.add((min(a, b), max(a, b)))
    return list(edges)


def _hdt_from_edges(edges: list[tuple]) -> HDTGraph:
    G = HDTGraph()
    for u, v in edges:
        G.insert_edge(u, v)
    return G


class TestLargeGraphsVsNetworkX:
    """Randomised tests on larger graphs that compare every HDTGraph query against
    the corresponding NetworkX function."""

    # ------------------------------------------------------------------
    # Static graphs (insert-only) – 50 nodes, ~150 edges
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seed", [10, 20, 30, 40, 50])
    def test_is_connected_large_static(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=50, n_edges=150)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        assert G_hdt.is_connected() == nx.is_connected(G_nx)

    @pytest.mark.parametrize("seed", [11, 22, 33, 44, 55])
    def test_number_of_components_large_static(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=60, n_edges=60)  # sparse → multiple components
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        assert len(G_hdt._levels[0].forest) == nx.number_connected_components(G_nx)

    @pytest.mark.parametrize("seed", [100, 200, 300, 400, 500])
    def test_all_component_memberships_large_static(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=50, n_edges=100)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        for node in G_hdt.get_nodes():
            assert set(G_hdt.component_iterator(node)) == nx.node_connected_component(G_nx, node), (
                f"seed={seed}, node={node}: component mismatch"
            )

    @pytest.mark.parametrize("seed", [7, 77, 777])
    def test_sampled_connectivity_large_static(self, seed):
        """Check connected(u, v) against nx.has_path for 200 random pairs."""
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=80, n_edges=200)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        nodes = list(G_hdt.get_nodes())
        for _ in range(200):
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            assert G_hdt.connected(u, v) == nx.has_path(G_nx, u, v), (
                f"seed={seed}: connected({u}, {v}) mismatch"
            )

    # ------------------------------------------------------------------
    # After bulk deletion – 50 nodes, remove half the edges
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seed", [13, 26, 39, 52, 65])
    def test_is_connected_after_deletions_large(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=50, n_edges=150)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        to_delete = rng.sample(edges, len(edges) // 2)
        for u, v in to_delete:
            G_hdt.delete_edge(u, v)
            G_nx.remove_edge(u, v)
        assert G_hdt.is_connected() == nx.is_connected(G_nx)

    @pytest.mark.parametrize("seed", [17, 34, 51, 68, 85])
    def test_number_of_components_after_deletions_large(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=50, n_edges=120)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        to_delete = rng.sample(edges, len(edges) // 2)
        for u, v in to_delete:
            G_hdt.delete_edge(u, v)
            G_nx.remove_edge(u, v)
        assert len(G_hdt._levels[0].forest) == nx.number_connected_components(G_nx)

    @pytest.mark.parametrize("seed", [19, 38, 57])
    def test_all_component_memberships_after_deletions_large(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=40, n_edges=100)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        to_delete = rng.sample(edges, len(edges) // 2)
        for u, v in to_delete:
            G_hdt.delete_edge(u, v)
            G_nx.remove_edge(u, v)
        for node in G_hdt.get_nodes():
            assert set(G_hdt.component_iterator(node)) == nx.node_connected_component(G_nx, node), (
                f"seed={seed}, node={node}: component mismatch after deletions"
            )

    @pytest.mark.parametrize("seed", [23, 46, 69])
    def test_sampled_connectivity_after_deletions_large(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=60, n_edges=180)
        G_nx = _nx_graph(edges)
        G_hdt = _hdt_from_edges(edges)
        to_delete = rng.sample(edges, len(edges) // 2)
        for u, v in to_delete:
            G_hdt.delete_edge(u, v)
            G_nx.remove_edge(u, v)
        nodes = list(G_hdt.get_nodes())
        for _ in range(200):
            u = rng.choice(nodes)
            v = rng.choice(nodes)
            assert G_hdt.connected(u, v) == nx.has_path(G_nx, u, v), (
                f"seed={seed}: connected({u}, {v}) mismatch after deletions"
            )

    # ------------------------------------------------------------------
    # ETTree integrity on large graphs
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("seed", [101, 202, 303, 404, 505])
    def test_ett_integrity_large_graph(self, seed):
        rng = random.Random(seed)
        edges = _random_edges(rng, n_nodes=60, n_edges=150)
        G_hdt = _hdt_from_edges(edges)
        to_delete = rng.sample(edges, len(edges) // 3)
        for u, v in to_delete:
            G_hdt.delete_edge(u, v)
        _check_ett_integrity(G_hdt)
