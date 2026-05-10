"""Tests for tralda.supertree.build_st (BuildST algorithm)."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree
from tralda.supertree.build_st import BuildST, build_st
from tralda.supertree.build import tree_profile_to_triples
from tests.supertrees.helpers import (
    SEEDS,
    displays_all_triples,
    make_partial_trees,
    restrict_to_leaves,
)


# ══════════════════════════════════════════════════════════════════════════════
# build_st — convenience function
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildST_CompatibleSameLeafSet:
    """build_st on compatible trees that share a common leaf set."""

    @pytest.mark.parametrize(
        "newick1,newick2,expected_cr",
        [
            ("((a,b),c,d);", "((a,b,c),d);", "(((a,b),c),d);"),
            ("(a,b,c,d);", "((a,b),c,d);", "((a,b),c,d);"),
        ],
    )
    def test_known_compatible_pair(self, newick1, newick2, expected_cr):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = build_st([T1, T2])
        assert result is not None
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_trees_return_nonnone(self, seed):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base, n=8)
        result = build_st(partial)
        assert result is not None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_trees_is_refinement_of_each_input(self, seed):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base, n=8)
        result = build_st(partial)
        assert result is not None
        for T_i in partial:
            assert result.is_refinement(T_i), (
                "build_st result is not a refinement of one of the input trees"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_is_phylogenetic(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result = build_st(partial)
        assert result is not None
        assert result.is_phylogenetic()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_leaf_set(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result = build_st(partial)
        assert result is not None
        expected = {v.label for v in base.leaves()}
        assert {v.label for v in result.leaves()} == expected


class TestBuildST_Incompatible:
    """build_st returns None when the input trees are incompatible."""

    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            # Clusters {a,b} and {a,c} overlap ─ incompatible
            ("((a,b),c,d);", "((a,c),b,d);"),
            # Clusters {a,b} and {b,c} overlap
            ("((a,b),c,d);", "((b,c),a,d);"),
        ],
    )
    def test_incompatible_returns_none(self, newick1, newick2):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        assert build_st([T1, T2]) is None


class TestBuildST_DifferentLeafSets:
    """build_st on trees with partially overlapping (but connected) leaf sets."""

    def test_known_example_from_docs(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        result = build_st([T1, T2])
        assert result is not None
        assert {v.label for v in result.leaves()} == {"a", "b", "c", "d"}

    def test_displays_all_triples_different_leaf_sets(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        T3 = Tree.parse_newick("(d,(a,c));")
        result = build_st([T1, T2, T3])
        if result is not None:
            _, triples = tree_profile_to_triples([T1, T2, T3])
            assert displays_all_triples(result, triples)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_leaf_set_supertree(self, seed):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        labels = [v.label for v in base.leaves()]

        # Two overlapping leaf subsets: first 20 and last 20 (10-leaf overlap).
        T_A = restrict_to_leaves(base, set(labels[:20]))
        T_B = restrict_to_leaves(base, set(labels[10:]))
        result = build_st([T_A, T_B])
        assert result is not None
        assert {v.label for v in result.leaves()} == set(labels)


class TestBuildST_EdgeCases:
    def test_single_tree(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = build_st([T])
        assert result is not None
        assert result.equal_topology(T)

    def test_two_identical_trees(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = build_st([T, T.copy()])
        assert result is not None
        assert result.equal_topology(T)


# ══════════════════════════════════════════════════════════════════════════════
# BuildST class interface
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSTClass:
    def test_run_returns_same_as_convenience_function(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        result_fn = build_st([T1, T2])
        result_cls = BuildST([T1, T2]).run()
        # Both should agree on topology (or both be None)
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_class_and_function_agree_on_random_trees(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result_fn = build_st(partial)
        result_cls = BuildST(partial).run()
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)
