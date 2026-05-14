"""Tests for tralda.utils.graph_tools."""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from tralda.utils.graph_tools import (
    build_adjacency_matrix,
    complete_multipartite_graph_from_sets,
    contingency_table,
    copy_node_attributes,
    disturb_graph,
    false_edges,
    graphs_equal,
    independent_sets,
    is_properly_colored,
    is_subgraph,
    performance,
    random_graph,
    sort_by_colors,
    sort_edge,
    symmetric_diff,
)


# ===========================================================================
# sort_edge
# ===========================================================================


class TestSortEdge:
    @pytest.mark.parametrize(
        "u, v, expected",
        [
            (1, 2, (1, 2)),
            (2, 1, (1, 2)),
            ("a", "b", ("a", "b")),
            ("z", "a", ("a", "z")),
            (1, 1, (1, 1)),
        ],
    )
    def test_comparable(self, u, v, expected):
        assert sort_edge(u, v) == expected

    def test_incomparable_consistent(self):
        # Objects with no natural order: result must be consistent
        a, b = object(), object()
        assert sort_edge(a, b) == sort_edge(a, b)
        assert sort_edge(a, b) == sort_edge(b, a)

    def test_incomparable_uses_id_order(self):
        a, b = object(), object()
        result = sort_edge(a, b)
        if id(a) < id(b):
            assert result == (a, b)
        else:
            assert result == (b, a)


# ===========================================================================
# build_adjacency_matrix
# ===========================================================================


class TestBuildAdjacencyMatrix:
    def test_returns_tuple(self):
        G = nx.path_graph(3)
        result = build_adjacency_matrix(G)
        assert isinstance(result, tuple) and len(result) == 2

    def test_matrix_shape(self):
        G = nx.path_graph(4)
        matrix, index = build_adjacency_matrix(G)
        assert matrix.shape == (4, 4)
        assert len(index) == 4

    def test_index_covers_all_nodes(self):
        G = nx.cycle_graph(5)
        _, index = build_adjacency_matrix(G)
        assert set(index.keys()) == set(G.nodes())

    @pytest.mark.parametrize("seed", [0, 1, 42])
    def test_edges_reflected(self, seed):
        G = nx.gnp_random_graph(6, 0.5, seed=seed)
        matrix, index = build_adjacency_matrix(G)
        for u, v in G.edges():
            assert matrix[index[u], index[v]] == 1
            assert matrix[index[v], index[u]] == 1

    def test_non_edges_are_zero(self):
        G = nx.path_graph(4)
        matrix, index = build_adjacency_matrix(G)
        for u in G.nodes():
            for v in G.nodes():
                if not G.has_edge(u, v):
                    assert matrix[index[u], index[v]] == 0

    def test_empty_graph(self):
        G = nx.Graph()
        matrix, index = build_adjacency_matrix(G)
        assert matrix.shape == (0, 0)
        assert index == {}

    def test_directed_graph(self):
        G = nx.DiGraph([(0, 1), (1, 2)])
        matrix, index = build_adjacency_matrix(G)
        assert matrix[index[0], index[1]] == 1
        assert matrix[index[1], index[0]] == 0  # no back-edge


# ===========================================================================
# graphs_equal
# ===========================================================================


class TestGraphsEqual:
    def test_equal_graphs(self):
        G1 = nx.path_graph(4)
        G2 = nx.path_graph(4)
        assert graphs_equal(G1, G2)

    def test_different_edges(self):
        G1 = nx.path_graph(4)
        G2 = nx.cycle_graph(4)
        assert not graphs_equal(G1, G2)

    def test_different_node_count(self):
        G1 = nx.path_graph(3)
        G2 = nx.path_graph(4)
        assert not graphs_equal(G1, G2)

    def test_different_node_labels(self):
        G1 = nx.Graph([(1, 2)])
        G2 = nx.Graph([(3, 4)])
        assert not graphs_equal(G1, G2)

    @pytest.mark.parametrize("seed", [7, 13, 99])
    def test_random_equal_copy(self, seed):
        G = nx.gnp_random_graph(8, 0.4, seed=seed)
        assert graphs_equal(G, G.copy())


# ===========================================================================
# is_subgraph
# ===========================================================================


class TestIsSubgraph:
    def test_same_graph(self):
        G = nx.path_graph(4)
        assert is_subgraph(G, G.copy())

    def test_proper_subgraph(self):
        G_sub = nx.path_graph(3)
        G = nx.cycle_graph(4)
        assert is_subgraph(G_sub, G)

    def test_not_subgraph(self):
        G1 = nx.Graph([(1, 3)])
        G2 = nx.Graph([(1, 2), (2, 3)])
        assert not is_subgraph(G1, G2)

    def test_directed_vs_undirected(self):
        G1 = nx.DiGraph([(0, 1)])
        G2 = nx.Graph([(0, 1)])
        assert not is_subgraph(G1, G2)

    def test_empty_is_subgraph_of_anything(self):
        G_empty = nx.Graph()
        G = nx.path_graph(4)
        assert is_subgraph(G_empty, G)


# ===========================================================================
# symmetric_diff
# ===========================================================================


class TestSymmetricDiff:
    def test_identical_graphs(self):
        G = nx.path_graph(4)
        assert symmetric_diff(G, G.copy()) == 0

    def test_single_extra_edge(self):
        G1 = nx.path_graph(4)
        G2 = G1.copy()
        G2.add_edge(0, 3)
        assert symmetric_diff(G1, G2) == 1
        assert symmetric_diff(G2, G1) == 1

    def test_disjoint_edge_sets(self):
        G1 = nx.Graph()
        G1.add_nodes_from([0, 1, 2])
        G1.add_edge(0, 1)
        G2 = nx.Graph()
        G2.add_nodes_from([0, 1, 2])
        G2.add_edge(1, 2)
        assert symmetric_diff(G1, G2) == 2

    def test_raises_on_different_vertex_sets(self):
        G1 = nx.path_graph(3)
        G2 = nx.path_graph(4)
        with pytest.raises(RuntimeError):
            symmetric_diff(G1, G2)

    @pytest.mark.parametrize("seed", [3, 17, 55])
    def test_symmetric_diff_directed(self, seed):
        rng = random.Random(seed)
        nodes = list(range(5))
        G1 = nx.DiGraph()
        G1.add_nodes_from(nodes)
        G2 = nx.DiGraph()
        G2.add_nodes_from(nodes)
        for u in nodes:
            for v in nodes:
                if u != v:
                    if rng.random() < 0.4:
                        G1.add_edge(u, v)
                    if rng.random() < 0.4:
                        G2.add_edge(u, v)
        result = symmetric_diff(G1, G2)
        assert result >= 0


# ===========================================================================
# contingency_table
# ===========================================================================


class TestContingencyTable:
    def _make_pair(self):
        true_g = nx.cycle_graph(5)
        pred_g = nx.path_graph(5)
        return true_g, pred_g

    def test_returns_dict_by_default(self):
        tg, pg = self._make_pair()
        result = contingency_table(tg, pg)
        assert isinstance(result, dict)
        assert set(result.keys()) == {"tp", "tn", "fp", "fn"}

    def test_returns_tuple_when_asked(self):
        tg, pg = self._make_pair()
        result = contingency_table(tg, pg, as_dict=False)
        assert isinstance(result, tuple) and len(result) == 4

    def test_raises_on_different_vertices(self):
        G1 = nx.path_graph(3)
        G2 = nx.path_graph(4)
        with pytest.raises(ValueError):
            contingency_table(G1, G2)

    def test_identical_graphs(self):
        G = nx.cycle_graph(5)
        ct = contingency_table(G, G.copy())
        assert ct["tp"] == G.size()
        assert ct["fp"] == 0
        assert ct["fn"] == 0

    def test_empty_pred_graph(self):
        G_true = nx.cycle_graph(4)
        G_empty = nx.Graph()
        G_empty.add_nodes_from(G_true.nodes())
        ct = contingency_table(G_true, G_empty)
        assert ct["tp"] == 0
        assert ct["fp"] == 0
        assert ct["fn"] == G_true.size()

    @pytest.mark.parametrize("seed", [2, 8, 21])
    def test_tp_fp_fn_sum_to_possible_edges(self, seed):
        G_true = nx.gnp_random_graph(6, 0.5, seed=seed)
        G_pred = nx.gnp_random_graph(6, 0.5, seed=seed + 1)
        # Ensure same node set
        G_pred.add_nodes_from(G_true.nodes())
        G_true.add_nodes_from(G_pred.nodes())
        ct = contingency_table(G_true, G_pred)
        n = G_true.order()
        possible = n * (n - 1) // 2
        assert ct["tp"] + ct["tn"] + ct["fp"] + ct["fn"] == possible


# ===========================================================================
# performance
# ===========================================================================


class TestPerformance:
    def test_perfect_prediction(self):
        G = nx.cycle_graph(5)
        result = performance(G, G.copy())
        order, size, tp, tn, fp, fn, acc, prec, rec = result
        assert order == G.order()
        assert size == G.size()
        assert tp == G.size()
        assert fp == 0
        assert fn == 0
        assert tn == (order * (order - 1) // 2) - size
        assert acc == pytest.approx(1.0)
        assert prec == pytest.approx(1.0)
        assert rec == pytest.approx(1.0)

    def test_empty_prediction(self):
        G_true = nx.cycle_graph(4)
        G_empty = nx.Graph()
        G_empty.add_nodes_from(G_true.nodes())
        _, _, tp, _, fp, _, _, prec, rec = performance(G_true, G_empty)
        assert tp == 0
        assert fp == 0
        import math

        assert math.isnan(prec)
        assert rec == pytest.approx(0.0)


# ===========================================================================
# false_edges
# ===========================================================================


class TestFalseEdges:
    def test_no_false_edges_when_equal(self):
        G = nx.path_graph(4)
        fn_g, fp_g = false_edges(G, G.copy())
        assert fn_g.size() == 0
        assert fp_g.size() == 0

    def test_false_negatives(self):
        G_true = nx.path_graph(4)
        G_pred = nx.Graph()
        G_pred.add_nodes_from(G_true.nodes())
        fn_g, fp_g = false_edges(G_true, G_pred)
        assert fn_g.size() == G_true.size()
        assert fp_g.size() == 0

    def test_false_positives(self):
        G_true = nx.path_graph(4)
        G_pred = G_true.copy()
        G_pred.add_edge(0, 3)
        fn_g, fp_g = false_edges(G_true, G_pred)
        assert fn_g.size() == 0
        assert fp_g.size() == 1
        assert fp_g.has_edge(0, 3)

    def test_directed_type_preserved(self):
        G_true = nx.DiGraph([(0, 1), (1, 2)])
        G_pred = nx.DiGraph([(0, 1)])
        G_pred.add_nodes_from(G_true.nodes())
        fn_g, fp_g = false_edges(G_true, G_pred)
        assert fn_g.is_directed()
        assert fp_g.is_directed()


# ===========================================================================
# is_properly_colored
# ===========================================================================


class TestIsProperlyColored:
    def _colored_graph(self, edges, colors):
        G = nx.Graph()
        for node, color in colors.items():
            G.add_node(node, color=color)
        G.add_edges_from(edges)
        return G

    def test_properly_colored(self):
        G = self._colored_graph([(0, 1), (1, 2)], {0: "red", 1: "blue", 2: "red"})
        assert is_properly_colored(G)

    def test_improperly_colored(self):
        G = self._colored_graph([(0, 1)], {0: "red", 1: "red"})
        assert not is_properly_colored(G)

    def test_no_edges(self):
        G = nx.Graph()
        G.add_node(0, color="red")
        assert is_properly_colored(G)

    def test_raises_on_missing_attribute(self):
        G = nx.Graph([(0, 1)])
        with pytest.raises(KeyError):
            is_properly_colored(G)


# ===========================================================================
# sort_by_colors
# ===========================================================================


class TestSortByColors:
    def test_groups_correctly(self):
        G = nx.Graph()
        G.add_node(0, color="red")
        G.add_node(1, color="blue")
        G.add_node(2, color="red")
        result = sort_by_colors(G)
        assert set(result["red"]) == {0, 2}
        assert set(result["blue"]) == {1}

    def test_single_color(self):
        G = nx.Graph()
        for i in range(4):
            G.add_node(i, color="green")
        result = sort_by_colors(G)
        assert set(result["green"]) == {0, 1, 2, 3}

    def test_raises_on_missing_attribute(self):
        G = nx.Graph()
        G.add_node(0)
        with pytest.raises(KeyError):
            sort_by_colors(G)


# ===========================================================================
# copy_node_attributes
# ===========================================================================


class TestCopyNodeAttributes:
    def test_copies_label(self):
        G_src = nx.Graph()
        G_src.add_node(1, label="x")
        G_dst = nx.Graph()
        G_dst.add_node(1)
        copy_node_attributes(G_src, G_dst, attributes="label")
        assert G_dst.nodes[1]["label"] == "x"

    def test_copies_multiple_attributes(self):
        G_src = nx.Graph()
        G_src.add_node(1, label="x", color="red")
        G_dst = nx.Graph()
        G_dst.add_node(1)
        copy_node_attributes(G_src, G_dst, attributes=["label", "color"])
        assert G_dst.nodes[1]["label"] == "x"
        assert G_dst.nodes[1]["color"] == "red"

    def test_skips_absent_nodes(self):
        G_src = nx.Graph()
        G_src.add_node(1, label="x")
        G_src.add_node(2, label="y")
        G_dst = nx.Graph()
        G_dst.add_node(1)
        copy_node_attributes(G_src, G_dst, attributes="label")
        assert not G_dst.has_node(2)


# ===========================================================================
# random_graph
# ===========================================================================


class TestRandomGraph:
    @pytest.mark.parametrize("n", [5, 10, 20])
    def test_node_count(self, n):
        G = random_graph(n)
        assert G.order() == n

    def test_nodes_labeled_from_one(self):
        G = random_graph(5)
        assert set(G.nodes()) == {1, 2, 3, 4, 5}

    @pytest.mark.parametrize("p", [0.0, 1.0])
    def test_extreme_probabilities(self, p):
        G = random_graph(6, p=p)
        if p == 0.0:
            assert G.size() == 0
        else:
            import math

            assert G.size() == math.comb(6, 2)


# ===========================================================================
# disturb_graph
# ===========================================================================


class TestDisturbGraph:
    @pytest.mark.parametrize("seed", [1, 42, 100])
    def test_inplace_false_does_not_modify_original(self, seed):
        random.seed(seed)
        G = random_graph(8, p=0.5)
        original_edges = set(G.edges())
        disturb_graph(G, insertion_prob=0.3, deletion_prob=0.3, inplace=False)
        assert set(G.edges()) == original_edges

    def test_inplace_true_modifies_original(self):
        random.seed(7)
        G = nx.complete_graph(6)
        original_size = G.size()
        disturb_graph(G, insertion_prob=0.0, deletion_prob=1.0, inplace=True)
        assert G.size() < original_size

    def test_no_deletion_preserves_all_edges(self):
        random.seed(5)
        G = nx.path_graph(5)
        original_edges = set(G.edges())
        G2 = disturb_graph(G, insertion_prob=0.0, deletion_prob=0.0, inplace=False)
        assert set(G2.edges()) == original_edges


# ===========================================================================
# independent_sets
# ===========================================================================


class TestIndependentSets:
    def test_complete_bipartite(self):
        G = nx.complete_bipartite_graph(2, 3)
        result = independent_sets(G)
        assert result is not None
        sizes = sorted(len(s) for s in result)
        assert sizes == [2, 3]

    def test_complete_multipartite(self):
        partition = [{0, 1}, {2, 3}, {4}]
        G = complete_multipartite_graph_from_sets(partition)
        result = independent_sets(G)
        assert result is not None
        sizes = sorted(len(s) for s in result)
        assert sizes == [1, 2, 2]

    def test_path_graph_not_complete_multipartite(self):
        # A path of length >= 3 is not complete multipartite (e.g., P4: 0-1-2-3 → no edge 0-2)
        G = nx.path_graph(4)
        result = independent_sets(G)
        assert result is None

    def test_empty_graph_single_independent_set(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        result = independent_sets(G)
        assert result is not None
        assert len(result) == 1
        assert set(result[0]) == {0, 1, 2}


# ===========================================================================
# complete_multipartite_graph_from_sets
# ===========================================================================


class TestCompleteMultipartiteGraphFromSets:
    @pytest.mark.parametrize(
        "partition",
        [
            [{0, 1}, {2, 3}],
            [{0}, {1}, {2}],
            [{0, 1, 2}, {3, 4}],
        ],
    )
    def test_all_cross_edges_present(self, partition):
        G = complete_multipartite_graph_from_sets(partition)
        for i, part_i in enumerate(partition):
            for j, part_j in enumerate(partition):
                if i == j:
                    continue
                for u in part_i:
                    for v in part_j:
                        assert G.has_edge(u, v)

    @pytest.mark.parametrize(
        "partition",
        [
            [{0, 1}, {2, 3}],
            [{0}, {1}, {2}],
        ],
    )
    def test_no_intra_partition_edges(self, partition):
        G = complete_multipartite_graph_from_sets(partition)
        for part in partition:
            for u, v in itertools.combinations(part, 2):
                assert not G.has_edge(u, v)

    def test_roundtrip_independent_sets(self):
        partition = [{0, 1}, {2, 3}, {4}]
        G = complete_multipartite_graph_from_sets(partition)
        result = independent_sets(G)
        assert result is not None
        recovered = sorted(sorted(s) for s in result)
        expected = sorted(sorted(s) for s in partition)
        assert recovered == expected
