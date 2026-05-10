"""Tests for cograph detection: ``to_cotree`` and ``LinearCographDetector``."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from tralda.cograph import random_cotree, to_cograph, to_cotree
from tralda.cograph.detection import LinearCographDetector

from .conftest import (
    graphs_equal,
    is_discriminating,
    is_valid_cotree_structure,
    leaf_labels,
)


# ===========================================================================
# Known cographs — to_cotree should return a non-None Tree
# ===========================================================================


class TestKnownCographs:
    """``to_cotree`` must accept all well-known cograph families."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_complete_graph_kn(self, n):
        G = nx.complete_graph(n)
        assert to_cotree(G) is not None

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_n_isolated_vertices(self, n):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        assert to_cotree(G) is not None

    @pytest.mark.parametrize("left,right", [(1, 1), (1, 3), (2, 2), (2, 3), (3, 4)])
    def test_complete_bipartite(self, left, right):
        # K_{l,r} = join(l isolated, r isolated) — always a cograph
        G = nx.complete_bipartite_graph(left, right)
        assert to_cotree(G) is not None

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_random_cotree_produces_cograph(self, seed):
        random.seed(seed)
        cotree = random_cotree(20)
        G = to_cograph(cotree)
        assert to_cotree(G) is not None

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_random_connected_cotree_produces_cograph(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        assert to_cotree(G) is not None

    def test_single_vertex(self):
        G = nx.Graph()
        G.add_node("x")
        assert to_cotree(G) is not None

    def test_k2(self):
        G = nx.Graph()
        G.add_edge("a", "b")
        assert to_cotree(G) is not None

    def test_two_isolated_string_vertices(self):
        G = nx.Graph()
        G.add_nodes_from(["a", "b"])
        assert to_cotree(G) is not None

    def test_c4_is_cograph(self):
        # C_4 ≅ K_{2,2} complement of 2*K_2, which is a cograph
        G = nx.cycle_graph(4)
        assert to_cotree(G) is not None


# ===========================================================================
# Known non-cographs — to_cotree should return None
# ===========================================================================


class TestKnownNonCographs:
    """``to_cotree`` must return None for graphs that contain an induced P_4."""

    def test_p4(self):
        assert to_cotree(nx.path_graph(4)) is None

    def test_p5(self):
        assert to_cotree(nx.path_graph(5)) is None

    @pytest.mark.parametrize("n", [5, 6, 7])
    def test_cycle_cn(self, n):
        # C_n for n >= 5 contains an induced P_4
        assert to_cotree(nx.cycle_graph(n)) is None

    def test_petersen_graph(self):
        assert to_cotree(nx.petersen_graph()) is None

    def test_bull_graph(self):
        # Bull graph contains a P_4
        G = nx.bull_graph()
        assert to_cotree(G) is None

    def test_explicit_p4_with_string_labels(self):
        G = nx.Graph()
        G.add_edges_from([("a", "b"), ("b", "c"), ("c", "d")])
        assert to_cotree(G) is None


# ===========================================================================
# Cotree structure validation
# ===========================================================================


class TestCotreeStructure:
    """The returned cotree must have correct structure."""

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_inner_nodes_have_series_or_parallel_label(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        result = to_cotree(G)
        assert is_valid_cotree_structure(result)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_leaf_set_matches_vertex_set(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        result = to_cotree(G)
        assert leaf_labels(result) == set(G.nodes())

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_returned_cotree_is_discriminating(self, seed):
        """to_cotree always returns the discriminating (canonical) cotree."""
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        result = to_cotree(G)
        assert is_discriminating(result)

    def test_single_vertex_cotree_root_is_leaf(self):
        G = nx.Graph()
        G.add_node(0)
        cotree = to_cotree(G)
        assert cotree.root.is_leaf()
        assert cotree.root.label == 0

    def test_k2_cotree_root_is_series(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        cotree = to_cotree(G)
        assert cotree.root.label == "series"
        leaves = list(cotree.leaves())
        assert {leaf.label for leaf in leaves} == {0, 1}

    def test_two_isolated_cotree_root_is_parallel(self):
        G = nx.Graph()
        G.add_nodes_from([0, 1])
        cotree = to_cotree(G)
        assert cotree.root.label == "parallel"


# ===========================================================================
# Round-trip: to_cograph(to_cotree(G)) ≅ G
# ===========================================================================


class TestRoundTrip:
    """Converting a cograph to its cotree and back must reproduce the original graph."""

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_complete_graph_round_trip(self, n):
        G = nx.complete_graph(n)
        cotree = to_cotree(G)
        G2 = to_cograph(cotree)
        assert graphs_equal(G, G2)

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_isolated_vertices_round_trip(self, n):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        cotree = to_cotree(G)
        G2 = to_cograph(cotree)
        assert graphs_equal(G, G2)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_random_cograph_round_trip(self, seed):
        random.seed(seed)
        cotree = random_cotree(20)
        G = to_cograph(cotree)
        cotree2 = to_cotree(G)
        G2 = to_cograph(cotree2)
        assert graphs_equal(G, G2)

    @pytest.mark.parametrize("left,right", [(1, 2), (2, 3), (3, 3)])
    def test_complete_bipartite_round_trip(self, left, right):
        G = nx.complete_bipartite_graph(left, right)
        cotree = to_cotree(G)
        G2 = to_cograph(cotree)
        assert graphs_equal(G, G2)


# ===========================================================================
# Edge cases and error handling
# ===========================================================================


class TestEdgeCases:
    """Edge cases and documented exceptional behaviour."""

    def test_empty_graph_returns_none(self):
        assert to_cotree(nx.Graph()) is None

    def test_invalid_graph_type_raises(self):
        with pytest.raises(TypeError):
            LinearCographDetector("not-a-graph")

    def test_invalid_graph_object_raises(self):
        class BadGraph:
            pass

        with pytest.raises(TypeError):
            LinearCographDetector(BadGraph())
