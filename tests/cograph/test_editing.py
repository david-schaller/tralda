"""Tests for cograph editing: ``CographEditor`` and ``edit_to_cograph``."""

from __future__ import annotations

import random

import networkx as nx
import pytest

from tralda.cograph import (
    CographEditor,
    edit_to_cograph,
    random_cotree,
    to_cograph,
    to_cotree,
)
import tralda.utils.graph_tools as gt

from .conftest import graphs_equal


# ===========================================================================
# CographEditor — initialisation
# ===========================================================================


class TestCographEditorInit:
    def test_empty_graph_raises_value_error(self):
        with pytest.raises(ValueError, match="empty graph"):
            CographEditor(nx.Graph())

    def test_non_graph_raises_type_error(self):
        with pytest.raises(TypeError):
            CographEditor("not-a-graph")

    def test_non_graph_object_raises_type_error(self):
        class Stub:
            pass

        with pytest.raises(TypeError):
            CographEditor(Stub())

    def test_single_vertex_graph_accepted(self):
        G = nx.Graph()
        G.add_node("v")
        ce = CographEditor(G)
        assert ce.best_cost == float("inf")
        assert ce.cotrees == []
        assert ce.costs == []


# ===========================================================================
# CographEditor — already-a-cograph inputs
# ===========================================================================


class TestCographEditorOnCographs:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_complete_graph_cost_zero(self, n):
        G = nx.complete_graph(n)
        ce = CographEditor(G)
        ce.cograph_edit(run_number=5)
        assert ce.best_cost == 0

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_cograph_cost_zero(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        ce = CographEditor(G)
        ce.cograph_edit(run_number=5)
        assert ce.best_cost == 0

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_cograph_result_equals_input(self, seed):
        """When the input is already a cograph the edit must leave it unchanged."""
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        ce = CographEditor(G)
        best_cotree = ce.cograph_edit(run_number=5)
        G_result = to_cograph(best_cotree)
        assert graphs_equal(G, G_result)

    def test_cograph_stops_after_first_run(self):
        """When cost reaches 0 after the first run no further runs are performed."""
        random.seed(42)
        G = to_cograph(random_cotree(10))
        ce = CographEditor(G)
        ce.cograph_edit(run_number=20)
        assert len(ce.costs) == 1

    def test_single_vertex_cost_zero(self):
        G = nx.Graph()
        G.add_node(0)
        ce = CographEditor(G)
        ce.cograph_edit(run_number=3)
        assert ce.best_cost == 0

    def test_k2_cost_zero(self):
        G = nx.Graph()
        G.add_edge(0, 1)
        ce = CographEditor(G)
        ce.cograph_edit(run_number=3)
        assert ce.best_cost == 0


# ===========================================================================
# CographEditor — non-cograph inputs
# ===========================================================================


class TestCographEditorOnNonCographs:
    def test_p4_positive_cost(self):
        ce = CographEditor(nx.path_graph(4))
        ce.cograph_edit(run_number=10)
        assert ce.best_cost > 0

    def test_p4_result_is_cograph(self):
        ce = CographEditor(nx.path_graph(4))
        best_cotree = ce.cograph_edit(run_number=10)
        result = to_cograph(best_cotree)
        assert to_cotree(result) is not None

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_random_graph_result_is_cograph(self, seed):
        random.seed(seed)
        G = gt.random_graph(25, p=0.4)
        ce = CographEditor(G)
        best_cotree = ce.cograph_edit(run_number=15)
        result = to_cograph(best_cotree)
        assert to_cotree(result) is not None

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_result_has_same_vertex_set(self, seed):
        random.seed(seed)
        G = gt.random_graph(20, p=0.5)
        ce = CographEditor(G)
        best_cotree = ce.cograph_edit(run_number=10)
        result = to_cograph(best_cotree)
        assert set(result.nodes()) == set(G.nodes())

    def test_petersen_positive_cost(self):
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=10)
        assert ce.best_cost > 0

    def test_petersen_result_is_cograph(self):
        ce = CographEditor(nx.petersen_graph())
        best_cotree = ce.cograph_edit(run_number=10)
        result = to_cograph(best_cotree)
        assert to_cotree(result) is not None


# ===========================================================================
# CographEditor — run bookkeeping
# ===========================================================================


class TestCographEditorBookkeeping:
    def test_costs_length_equals_run_number_for_non_cograph(self):
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=7)
        assert len(ce.costs) == 7

    def test_cotrees_length_equals_run_number_for_non_cograph(self):
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=7)
        assert len(ce.cotrees) == 7

    def test_best_cost_is_minimum_of_costs(self):
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=10)
        assert ce.best_cost == min(ce.costs)

    def test_multiple_calls_append_to_costs(self):
        """Successive calls to cograph_edit accumulate results."""
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=3)
        ce.cograph_edit(run_number=4)
        assert len(ce.costs) == 7

    def test_multiple_calls_best_cost_non_increasing(self):
        """best_cost after a second call is ≤ best_cost after the first call."""
        random.seed(42)
        G = gt.random_graph(20, p=0.45)
        ce = CographEditor(G)
        ce.cograph_edit(run_number=5)
        cost_after_first = ce.best_cost
        ce.cograph_edit(run_number=10)
        assert ce.best_cost <= cost_after_first

    @pytest.mark.parametrize("run_number", [1, 5, 10])
    def test_run_number_parameter(self, run_number):
        ce = CographEditor(nx.petersen_graph())
        ce.cograph_edit(run_number=run_number)
        assert len(ce.costs) == run_number


# ===========================================================================
# edit_to_cograph
# ===========================================================================


class TestEditToCograph:
    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_result_is_cograph(self, seed):
        random.seed(seed)
        G = gt.random_graph(25, p=0.4)
        H = edit_to_cograph(G, run_number=10)
        assert to_cotree(H) is not None

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_result_has_same_vertex_set(self, seed):
        random.seed(seed)
        G = gt.random_graph(20, p=0.5)
        H = edit_to_cograph(G, run_number=10)
        assert set(H.nodes()) == set(G.nodes())

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_cograph_input_is_returned_unchanged(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        H = edit_to_cograph(G, run_number=5)
        assert graphs_equal(G, H)

    def test_run_number_one(self):
        G = nx.petersen_graph()
        H = edit_to_cograph(G, run_number=1)
        assert to_cotree(H) is not None

    def test_result_is_networkx_graph(self):
        G = nx.path_graph(5)
        H = edit_to_cograph(G)
        assert isinstance(H, nx.Graph)
