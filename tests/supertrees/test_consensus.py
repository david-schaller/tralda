"""Tests for tralda.supertree.consensus (merge_trees, one_way_compatible,
merge_all, LooseConsensusTree, loose_consensus_tree)."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree
from tralda.supertree.consensus import (
    LooseConsensusTree,
    loose_consensus_tree,
    merge_all,
    merge_trees,
    one_way_compatible,
)
from tralda.supertree.common_refinement import linear_common_refinement
from tests.supertrees.helpers import SEEDS, make_partial_trees


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _cluster_set(tree: Tree) -> set[frozenset]:
    return {frozenset(c) for c in tree.get_hierarchy()}


# ══════════════════════════════════════════════════════════════════════════════
# merge_trees
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeTrees:
    @pytest.mark.parametrize(
        "newick1,newick2,expected_cr",
        [
            ("((a,b),c,d);", "((a,b,c),d);", "(((a,b),c),d);"),
            ("(a,b,c,d);", "((a,b),c,d);", "((a,b),c,d);"),
            ("((a,b),(c,d));", "(a,b,c,d);", "((a,b),(c,d));"),
        ],
    )
    def test_known_compatible_pair(self, newick1, newick2, expected_cr):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = merge_trees(T1, T2)
        assert result is not None
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    def test_merge_contains_all_clusters_from_both(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        result = merge_trees(T1, T2)
        clusters_t1 = _cluster_set(T1)
        clusters_t2 = _cluster_set(T2)
        clusters_result = _cluster_set(result)
        assert clusters_t1 <= clusters_result
        assert clusters_t2 <= clusters_result

    def test_merge_does_not_modify_inputs(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        before_t1 = _cluster_set(T1)
        merge_trees(T1, T2)
        assert _cluster_set(T1) == before_t1

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_compatible_pair_contains_both_cluster_sets(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=2, contraction_prob=0.6)
        T1, T2 = partial[0], partial[1]
        result = merge_trees(T1, T2)
        assert _cluster_set(T1) <= _cluster_set(result)
        assert _cluster_set(T2) <= _cluster_set(result)


# ══════════════════════════════════════════════════════════════════════════════
# one_way_compatible
# ══════════════════════════════════════════════════════════════════════════════


class TestOneWayCompatible:
    def test_compatible_pair_no_contractions(self):
        # T1 is a refinement of T2, so all clusters of T1 are compatible with T2
        T1 = Tree.parse_newick("(((a,b),c),d);")
        T2 = Tree.parse_newick("((a,b),c,d);")
        result, n_contracted = one_way_compatible(T1, T2, return_no_of_contractions=True)
        assert n_contracted == 0
        assert result.equal_topology(T1)

    def test_incompatible_cluster_is_removed(self):
        # T1 has cluster {a,b}, T2 has cluster {a,c} — they overlap
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,c),b,d);")
        result, n_contracted = one_way_compatible(T1, T2, return_no_of_contractions=True)
        # The incompatible cluster {a,b} in T1 wrt T2 should be removed
        assert n_contracted > 0
        # The removed cluster must not appear in the result
        result_clusters = _cluster_set(result)
        assert frozenset({"a", "b"}) not in result_clusters

    def test_all_clusters_of_result_are_compatible_with_t2(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,c),b,d);")
        result = one_way_compatible(T1, T2)
        result_clusters = _cluster_set(result)
        t2_clusters = _cluster_set(T2)
        # Every cluster in result must be compatible with every cluster in T2
        for c1 in result_clusters:
            for c2 in t2_clusters:
                intersection = c1 & c2
                assert not (intersection and not c1 <= c2 and not c2 <= c1), (
                    f"Clusters {c1} and {c2} overlap in result"
                )

    def test_returns_without_contraction_count(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        result = one_way_compatible(T1, T2)
        assert isinstance(result, Tree)


# ══════════════════════════════════════════════════════════════════════════════
# merge_all
# ══════════════════════════════════════════════════════════════════════════════


class TestMergeAll:
    @pytest.mark.parametrize(
        "newicks,expected_cr",
        [
            (["((a,b),c,d);", "((a,b,c),d);"], "(((a,b),c),d);"),
            (["(a,b,c,d);", "((a,b),c,d);", "((a,b,c),d);"], "(((a,b),c),d);"),
        ],
    )
    def test_known_compatible_list(self, newicks, expected_cr):
        trees = [Tree.parse_newick(nw) for nw in newicks]
        result = merge_all(trees)
        assert result is not None
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    def test_single_tree_returns_copy(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = merge_all([T])
        assert result is not None
        assert result.equal_topology(T)
        assert result is not T  # it is a copy, not the same object

    def test_result_contains_all_clusters(self):
        trees = [
            Tree.parse_newick("((a,b),c,d);"),
            Tree.parse_newick("((a,b,c),d);"),
        ]
        result = merge_all(trees)
        assert result is not None
        result_clusters = _cluster_set(result)
        for T_i in trees:
            assert _cluster_set(T_i) <= result_clusters

    def test_raises_on_different_leaf_sets(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        with pytest.raises((ValueError, RuntimeError)):
            merge_all([T1, T2])

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_compatible_trees_contain_all_clusters(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5, contraction_prob=0.6)
        result = merge_all(partial)
        assert result is not None
        result_clusters = _cluster_set(result)
        for T_i in partial:
            assert _cluster_set(T_i) <= result_clusters


# ══════════════════════════════════════════════════════════════════════════════
# loose_consensus_tree / LooseConsensusTree
# ══════════════════════════════════════════════════════════════════════════════


class TestLooseConsensusTree:
    # ── Compatible inputs ─────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "newicks,expected_cr",
        [
            (["((a,b),c,d);", "((a,b,c),d);"], "(((a,b),c),d);"),
            (["(a,b,c,d);", "((a,b),c,d);"], "((a,b),c,d);"),
        ],
    )
    def test_compatible_equals_common_refinement(self, newicks, expected_cr):
        trees = [Tree.parse_newick(nw) for nw in newicks]
        result = loose_consensus_tree(trees)
        assert result is not None
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    def test_single_tree_returns_copy(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = loose_consensus_tree([T])
        assert result is not None
        assert result.equal_topology(T)

    # ── Incompatible inputs ───────────────────────────────────────────────────

    def test_incompatible_pair_removes_conflicting_clusters(self):
        # T1 has {a,b}, T2 has {a,c} — these conflict; the result should
        # contain neither of them.
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,c),b,d);")
        result = loose_consensus_tree([T1, T2])
        assert result is not None
        result_clusters = _cluster_set(result)
        assert frozenset({"a", "b"}) not in result_clusters
        assert frozenset({"a", "c"}) not in result_clusters

    def test_incompatible_result_clusters_are_mutually_compatible(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((b,c),a,d);")
        result = loose_consensus_tree([T1, T2])
        assert result is not None
        clusters = list(_cluster_set(result))
        for i, c1 in enumerate(clusters):
            for c2 in clusters[i + 1 :]:
                intersection = c1 & c2
                assert not (intersection and not c1 <= c2 and not c2 <= c1), (
                    f"Result contains overlapping clusters: {c1} and {c2}"
                )

    def test_raises_on_different_leaf_sets(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        with pytest.raises(ValueError):
            loose_consensus_tree([T1, T2])

    # ── Random tests ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("seed", SEEDS)
    def test_compatible_equals_linear_common_refinement(self, seed):
        random.seed(seed)
        base = Tree.random_tree(25, binary=True)
        partial = make_partial_trees(base, n=6, contraction_prob=0.7)
        lct = loose_consensus_tree(partial)
        cr = linear_common_refinement(partial)
        assert lct is not None and cr is not None
        assert lct.equal_topology(cr), (
            "loose_consensus_tree and linear_common_refinement disagree on compatible trees"
        )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_leaf_set(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result = loose_consensus_tree(partial)
        assert result is not None
        expected = {v.label for v in base.leaves()}
        assert {v.label for v in result.leaves()} == expected

    # ── LooseConsensusTree class ──────────────────────────────────────────────

    def test_class_and_function_agree(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        result_fn = loose_consensus_tree([T1, T2])
        result_cls = LooseConsensusTree([T1, T2]).run()
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_class_and_function_agree_on_random_trees(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result_fn = loose_consensus_tree(partial)
        result_cls = LooseConsensusTree(partial).run()
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)
