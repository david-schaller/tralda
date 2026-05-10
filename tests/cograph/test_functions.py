"""Tests for tralda.cograph.functions:
``random_cotree``, ``to_cograph``, ``complement_cograph``,
``cluster_deletion``, ``complete_multipartite_completion``, and
``paths_of_length_2``.
"""

from __future__ import annotations

import itertools
import random

import networkx as nx
import pytest

from tralda.cograph import (
    cluster_deletion,
    complete_multipartite_completion,
    random_cotree,
    to_cograph,
    to_cotree,
)
from tralda.cograph.functions import complement_cograph, paths_of_length_2
from tralda.datastructures.tree import Tree, TreeNode

from .conftest import (
    graphs_equal,
    is_clique,
    is_complete_multipartite,
    is_discriminating,
    is_independent_set,
    is_partition_of,
    is_subgraph_of,
    is_valid_cotree_structure,
    leaf_labels,
)


# ===========================================================================
# random_cotree
# ===========================================================================


class TestRandomCotree:
    @pytest.mark.parametrize("n", [1, 2, 3, 5, 10, 20])
    def test_leaf_count(self, n):
        random.seed(42)
        cotree = random_cotree(n)
        assert sum(1 for _ in cotree.leaves()) == n

    @pytest.mark.parametrize("n", [2, 3, 5, 10, 20])
    def test_inner_nodes_have_valid_labels(self, n):
        random.seed(42)
        cotree = random_cotree(n)
        assert is_valid_cotree_structure(cotree)

    @pytest.mark.parametrize("n", [2, 3, 5, 10])
    def test_adjacent_inner_nodes_alternate(self, n):
        """Adjacent inner nodes always carry opposite labels."""
        random.seed(0)
        cotree = random_cotree(n)
        assert is_discriminating(cotree)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_force_series_root(self, seed):
        random.seed(seed)
        cotree = random_cotree(10, force_series_root=True)
        assert cotree.root.label == "series"

    @pytest.mark.parametrize("seed", [7, 42, 99, 137])
    def test_to_cograph_produces_cograph(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        assert to_cotree(G) is not None

    def test_reproducible_with_seed(self):
        random.seed(123)
        ct1 = random_cotree(10)
        random.seed(123)
        ct2 = random_cotree(10)
        # Both cotrees must produce identical cographs
        assert graphs_equal(to_cograph(ct1), to_cograph(ct2))

    def test_single_leaf_cotree(self):
        random.seed(0)
        cotree = random_cotree(1)
        # A single-leaf cotree has no inner nodes
        assert sum(1 for _ in cotree.inner_nodes()) == 0
        assert cotree.root.is_leaf()


# ===========================================================================
# to_cograph
# ===========================================================================


class TestToCograph:
    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_vertex_set_matches_cotree_leaves(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        assert set(G.nodes()) == leaf_labels(cotree)

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_series_root_with_n_leaves_produces_Kn(self, n):
        """A series node with n leaf children encodes K_n."""
        root = TreeNode(label="series")
        cotree = Tree(root)
        for i in range(n):
            root.add_child(TreeNode(label=i))
        G = to_cograph(cotree)
        assert set(G.nodes()) == set(range(n))
        # K_n has exactly n*(n-1)//2 edges
        assert G.number_of_edges() == n * (n - 1) // 2

    @pytest.mark.parametrize("n", [2, 3, 5, 8])
    def test_parallel_root_with_n_leaves_produces_empty_graph(self, n):
        """A parallel node with n leaf children encodes n isolated vertices."""
        root = TreeNode(label="parallel")
        cotree = Tree(root)
        for i in range(n):
            root.add_child(TreeNode(label=i))
        G = to_cograph(cotree)
        assert set(G.nodes()) == set(range(n))
        assert G.number_of_edges() == 0

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_round_trip_to_cotree(self, seed):
        """to_cograph(to_cotree(G)) must equal G for any cograph G."""
        random.seed(seed)
        cotree = random_cotree(20)
        G = to_cograph(cotree)
        G2 = to_cograph(to_cotree(G))
        assert graphs_equal(G, G2)

    def test_single_leaf_cotree(self):
        random.seed(0)
        ct = random_cotree(1)
        G = to_cograph(ct)
        assert G.number_of_nodes() == 1
        assert G.number_of_edges() == 0

    def test_complete_bipartite_k23(self):
        """K_{2,3}: series root → parallel({0,1}), parallel({2,3,4})."""
        p1 = TreeNode(label="parallel")
        p1.add_child(TreeNode(label=0))
        p1.add_child(TreeNode(label=1))
        p2 = TreeNode(label="parallel")
        for i in range(2, 5):
            p2.add_child(TreeNode(label=i))
        root = TreeNode(label="series")
        root.add_child(p1)
        root.add_child(p2)
        cotree = Tree(root)
        G = to_cograph(cotree)
        expected = nx.complete_bipartite_graph(2, 3)
        # same number of edges and both should be recognised as cographs
        assert G.number_of_edges() == expected.number_of_edges()
        assert to_cotree(G) is not None


# ===========================================================================
# complement_cograph
# ===========================================================================


class TestComplementCograph:
    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_double_complement_is_identity(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G = to_cograph(cotree)
        compl = complement_cograph(cotree, inplace=False)
        double_compl = complement_cograph(compl, inplace=False)
        G_dc = to_cograph(double_compl)
        assert graphs_equal(G, G_dc)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_inplace_false_does_not_modify_original(self, seed):
        random.seed(seed)
        cotree = random_cotree(15)
        G_before = to_cograph(cotree)
        complement_cograph(cotree, inplace=False)
        G_after = to_cograph(cotree)
        assert graphs_equal(G_before, G_after)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137])
    def test_inplace_true_modifies_original_and_returns_it(self, seed):
        random.seed(seed)
        cotree = random_cotree(10)
        result = complement_cograph(cotree, inplace=True)
        assert result is cotree

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_complement_of_kn_is_empty_graph(self, n):
        """Complement of K_n (all-series cotree) is the empty graph."""
        root = TreeNode(label="series")
        cotree = Tree(root)
        for i in range(n):
            root.add_child(TreeNode(label=i))
        compl_cotree = complement_cograph(cotree, inplace=False)
        G_compl = to_cograph(compl_cotree)
        assert G_compl.number_of_edges() == 0
        assert G_compl.number_of_nodes() == n

    @pytest.mark.parametrize("n", [2, 3, 5])
    def test_complement_of_empty_graph_is_kn(self, n):
        """Complement of n isolated vertices is K_n."""
        root = TreeNode(label="parallel")
        cotree = Tree(root)
        for i in range(n):
            root.add_child(TreeNode(label=i))
        compl_cotree = complement_cograph(cotree, inplace=False)
        G_compl = to_cograph(compl_cotree)
        assert G_compl.number_of_edges() == n * (n - 1) // 2

    def test_invalid_inner_label_raises(self):
        root = TreeNode(label="invalid_label")
        root.add_child(TreeNode(label=0))
        root.add_child(TreeNode(label=1))
        cotree = Tree(root)
        with pytest.raises(ValueError):
            complement_cograph(cotree, inplace=False)

    @pytest.mark.parametrize("seed", [7, 42, 99])
    def test_complement_is_cograph(self, seed):
        """The complement of a cograph is also a cograph."""
        random.seed(seed)
        cotree = random_cotree(15)
        compl_cotree = complement_cograph(cotree, inplace=False)
        compl_G = to_cograph(compl_cotree)
        assert to_cotree(compl_G) is not None


# ===========================================================================
# cluster_deletion
# ===========================================================================


class TestClusterDeletion:
    # ── Structural correctness ─────────────────────────────────────────────

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_result_is_partition_of_vertex_set(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition = cluster_deletion(G)
        assert is_partition_of(partition, set(G.nodes()))

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_each_part_is_clique(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition = cluster_deletion(G)
        for part in partition:
            assert is_clique(G, part), f"Part {part} is not a clique"

    # ── Known results ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
    def test_kn_gives_single_clique(self, n):
        G = nx.complete_graph(n)
        partition = cluster_deletion(G)
        assert len(partition) == 1
        assert set(partition[0]) == set(G.nodes())

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_isolated_vertices_give_n_singletons(self, n):
        G = nx.Graph()
        G.add_nodes_from(range(n))
        partition = cluster_deletion(G)
        assert len(partition) == n
        for part in partition:
            assert len(part) == 1

    def test_k12_star_optimal_partition(self):
        """K_{1,2}: optimal cluster deletion removes one edge (cost 1)."""
        # 0 = centre, 1 and 2 = leaves; edges (0,1) and (0,2)
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2)])
        partition = cluster_deletion(G)
        # Optimal: one clique has the centre + one leaf; one singleton remains
        assert is_partition_of(partition, {0, 1, 2})
        for part in partition:
            assert is_clique(G, part)
        # The edge that is not inside a clique is the deleted edge → cost 1
        kept_edges = {
            (min(u, v), max(u, v)) for part in partition for u, v in itertools.combinations(part, 2)
        }
        assert len(kept_edges) == 1  # exactly one edge kept

    def test_k22_cluster_deletion(self):
        """K_{2,2}: optimal cost is 2 (two cliques of size 2)."""
        G = nx.complete_bipartite_graph(2, 2)
        partition = cluster_deletion(G)
        assert is_partition_of(partition, set(G.nodes()))
        for part in partition:
            assert is_clique(G, part)
        kept = sum(len(part) * (len(part) - 1) // 2 for part in partition)
        deleted = G.number_of_edges() - kept
        assert deleted == 2

    # ── Accepts both graph and cotree inputs ──────────────────────────────

    @pytest.mark.parametrize("seed", [7, 42, 99])
    def test_graph_and_cotree_inputs_are_both_valid(self, seed):
        """Both graph and cotree inputs must yield valid partitions into cliques.

        Different equivalent cotrees may produce different (but equally optimal)
        solutions, so we only verify structural validity rather than identity.
        """
        random.seed(seed)
        cotree = random_cotree(15, force_series_root=True)
        G = to_cograph(cotree)
        p_graph = cluster_deletion(G)
        p_tree = cluster_deletion(cotree)
        vertices = set(G.nodes())
        for p in (p_graph, p_tree):
            assert is_partition_of(p, vertices)
            for part in p:
                assert is_clique(G, part)

    # ── Error handling ─────────────────────────────────────────────────────

    def test_non_cograph_raises_value_error(self):
        with pytest.raises(ValueError):
            cluster_deletion(nx.path_graph(4))

    def test_empty_graph_raises(self):
        # Empty graph triggers the RuntimeError from LinearCographDetector
        with pytest.raises(Exception):
            cluster_deletion(nx.Graph())


# ===========================================================================
# complete_multipartite_completion
# ===========================================================================


class TestCompleteMultipartiteCompletion:
    # ── Structural correctness ─────────────────────────────────────────────

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_result_is_partition_of_vertex_set(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition = complete_multipartite_completion(G)
        assert is_partition_of(partition, set(G.nodes()))

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_completed_graph_is_supergraph(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert is_subgraph_of(G, H)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_each_part_is_independent_set_in_completion(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        for part in partition:
            assert is_independent_set(H, part), f"Part {part} is not an independent set"

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_completed_graph_is_complete_multipartite(self, seed):
        random.seed(seed)
        cotree = random_cotree(20, force_series_root=True)
        G = to_cograph(cotree)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert is_complete_multipartite(H, partition)

    # ── supply_graph parameter ─────────────────────────────────────────────

    def test_supply_graph_false_returns_list(self):
        random.seed(42)
        G = to_cograph(random_cotree(10, force_series_root=True))
        result = complete_multipartite_completion(G, supply_graph=False)
        assert isinstance(result, list)

    def test_supply_graph_true_returns_tuple(self):
        random.seed(42)
        G = to_cograph(random_cotree(10, force_series_root=True))
        result = complete_multipartite_completion(G, supply_graph=True)
        assert isinstance(result, tuple) and len(result) == 2

    # ── Known results ──────────────────────────────────────────────────────

    @pytest.mark.parametrize("n", [1, 2, 3, 5])
    def test_kn_already_complete_multipartite(self, n):
        """K_n is already K_{1,...,1} — completion adds no edges."""
        G = nx.complete_graph(n)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert graphs_equal(G, H)
        # Each part must be a singleton
        for part in partition:
            assert len(part) == 1

    def test_k22_already_complete_bipartite(self):
        """K_{2,2} is already complete bipartite — completion adds no edges."""
        G = nx.complete_bipartite_graph(2, 2)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert graphs_equal(G, H)
        assert len(partition) == 2
        for part in partition:
            assert len(part) == 2

    def test_k12_completion(self):
        """K_{1,2} is already complete bipartite (= K_{1,2} itself).

        Its parts are {0} (centre) and {1, 2} (leaves); all cross edges already
        exist, so no edges need to be added.
        """
        G = nx.Graph()
        G.add_edges_from([(0, 1), (0, 2)])  # 0 = centre
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert is_partition_of(partition, {0, 1, 2})
        assert is_subgraph_of(G, H)
        assert is_complete_multipartite(H, partition)
        # K_{1,2} is already complete 2-partite — zero edges need to be added
        added = H.number_of_edges() - G.number_of_edges()
        assert added == 0

    def test_one_edge_graph_completion(self):
        """Graph with 3 vertices and 1 edge needs 1 edge to become complete bipartite."""
        # G: vertices {0,1,2}, single edge (0,1)
        G = nx.Graph()
        G.add_nodes_from([0, 1, 2])
        G.add_edge(0, 1)
        partition, H = complete_multipartite_completion(G, supply_graph=True)
        assert is_partition_of(partition, {0, 1, 2})
        assert is_subgraph_of(G, H)
        assert is_complete_multipartite(H, partition)
        # Optimal: {0, 2} (or {1, 2}) as one independent set, other vertex alone
        # — exactly 1 edge is added
        added = H.number_of_edges() - G.number_of_edges()
        assert added == 1

    # ── Accepts both graph and cotree inputs ──────────────────────────────

    @pytest.mark.parametrize("seed", [7, 42, 99])
    def test_graph_and_cotree_inputs_agree(self, seed):
        random.seed(seed)
        cotree = random_cotree(15, force_series_root=True)
        G = to_cograph(cotree)
        p_graph = complete_multipartite_completion(G)
        p_tree = complete_multipartite_completion(cotree)
        assert sorted(sorted(p) for p in p_graph) == sorted(sorted(p) for p in p_tree)

    # ── Error handling ─────────────────────────────────────────────────────

    def test_non_cograph_raises_value_error(self):
        with pytest.raises(ValueError):
            complete_multipartite_completion(nx.path_graph(4))


# ===========================================================================
# paths_of_length_2
# ===========================================================================


class TestPathsOfLength2:
    """``paths_of_length_2`` is an internal helper in ``tralda.cograph.functions``.

    Internally, ``LCA`` is constructed with ``strict_labels=False`` so that duplicate inner-node
    labels (``"series"`` / ``"parallel"``) do not raise.  All LCA queries in the function use
    ``TreeNode`` objects directly, so label-based lookup is never invoked.
    """

    def test_works_for_two_inner_nodes(self):
        """Minimal cotree: series -> { parallel -> {0, 1}, 2 }."""
        root = TreeNode(label="series")
        par = TreeNode(label="parallel")
        par.add_child(TreeNode(label=0))
        par.add_child(TreeNode(label=1))
        root.add_child(par)
        root.add_child(TreeNode(label=2))
        cotree = Tree(root)
        paths = list(paths_of_length_2(cotree))
        assert isinstance(paths, list)
        assert len(paths) == 1

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_yields_tuples_of_three_tree_nodes(self, seed):
        random.seed(seed)
        cotree = random_cotree(10, force_series_root=True)
        for path in paths_of_length_2(cotree):
            assert len(path) == 3
            assert all(isinstance(n, TreeNode) for n in path)

    @pytest.mark.parametrize("seed", [7, 42, 99, 137, 256])
    def test_middle_node_adjacent_to_outer_nodes(self, seed):
        """Check the LCA condition: the middle node is adjacent to both outer nodes in the cograph.

        For every yielded (t1, t3, t2), t3 must be adjacent to both t1 and t2 in the cograph
        (t3's LCA with t1 and t2 must be a series node).
        """
        from tralda.datastructures.last_common_ancestor import LCA

        random.seed(seed)
        cotree = random_cotree(10, force_series_root=True)
        lca = LCA(cotree, strict_labels=False)
        for t1, t3, t2 in paths_of_length_2(cotree):
            assert lca(t1, t3).label == "series"
            assert lca(t2, t3).label == "series"

    def test_parallel_only_cotree_yields_nothing(self):
        """A parallel root with leaf children encodes an edgeless graph: no paths."""
        root = TreeNode(label="parallel")
        for i in range(4):
            root.add_child(TreeNode(label=i))
        cotree = Tree(root)
        assert list(paths_of_length_2(cotree)) == []

    def test_invalid_inner_label_raises(self):
        root = TreeNode(label="bad_label")
        root.add_child(TreeNode(label=0))
        root.add_child(TreeNode(label=1))
        cotree = Tree(root)
        with pytest.raises(ValueError):
            list(paths_of_length_2(cotree))
