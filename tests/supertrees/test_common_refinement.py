"""Tests for tralda.supertree.common_refinement (LinCR algorithm)."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree
from tralda.supertree.common_refinement import LinCR, linear_common_refinement
from tests.supertrees.helpers import SEEDS, make_partial_trees


# ══════════════════════════════════════════════════════════════════════════════
# linear_common_refinement — convenience function
# ══════════════════════════════════════════════════════════════════════════════


class TestLinCR_KnownCompatible:
    """LinCR with hand-crafted compatible tree pairs."""

    @pytest.mark.parametrize(
        "newick1,newick2,expected_cr",
        [
            # ((a,b),c,d) + ((a,b,c),d) → (((a,b),c),d)
            ("((a,b),c,d);", "((a,b,c),d);", "(((a,b),c),d);"),
            # Star + resolved
            ("(a,b,c,d);", "((a,b),c,d);", "((a,b),c,d);"),
            # One tree is already fully resolved and compatible with the other
            ("(((a,b),c),d);", "((a,b),c,d);", "(((a,b),c),d);"),
            ("((a,b),(c,d));", "(a,b,c,d);", "((a,b),(c,d));"),
        ],
    )
    def test_known_result(self, newick1, newick2, expected_cr):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = linear_common_refinement([T1, T2])
        assert result is not None
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    @pytest.mark.parametrize(
        "newick1,newick2,expected_cr",
        [
            ("((a,b),c,d);", "((a,b,c),d);", "(((a,b),c),d);"),
        ],
    )
    def test_result_is_refinement_of_both_inputs(self, newick1, newick2, expected_cr):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = linear_common_refinement([T1, T2])
        assert result is not None
        assert result.is_refinement(T1)
        assert result.is_refinement(T2)
        assert result.equal_topology(Tree.parse_newick(expected_cr))

    def test_single_tree_returns_same_topology(self):
        T = Tree.parse_newick("(((a,b),c),d);")
        result = linear_common_refinement([T])
        assert result is not None
        assert result.equal_topology(T)

    def test_duplicate_tree_returns_same_topology(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = linear_common_refinement([T, T.copy()])
        assert result is not None
        assert result.equal_topology(T)


class TestLinCR_KnownIncompatible:
    """LinCR returns None for incompatible trees."""

    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            # {a,b} and {a,c} overlap
            ("((a,b),c,d);", "((a,c),b,d);"),
            # {a,b} and {b,c} overlap
            ("((a,b),c,d);", "((b,c),a,d);"),
            # {a,b,c} and {b,c,d} overlap
            ("((a,b,c),d);", "((b,c,d),a);"),
        ],
    )
    def test_incompatible_returns_none(self, newick1, newick2):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        assert linear_common_refinement([T1, T2]) is None


class TestLinCR_RandomCompatible:
    """LinCR on randomly generated partial trees (always compatible by construction)."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_trees_return_nonnone(self, seed):
        random.seed(seed)
        base = Tree.random_tree(40, binary=True)
        partial = make_partial_trees(base, n=10, contraction_prob=0.8)
        result = linear_common_refinement(partial)
        assert result is not None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_is_refinement_of_each_input(self, seed):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base, n=10, contraction_prob=0.9)
        result = linear_common_refinement(partial)
        assert result is not None
        for T_i in partial:
            assert result.is_refinement(T_i), (
                "LinCR result is not a refinement of one of the input trees"
            )

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_is_phylogenetic(self, seed):
        random.seed(seed)
        base = Tree.random_tree(25, binary=True)
        partial = make_partial_trees(base, n=8)
        result = linear_common_refinement(partial)
        assert result is not None
        assert result.is_phylogenetic()

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_leaf_set_equals_base(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=6)
        result = linear_common_refinement(partial)
        assert result is not None
        expected = {v.label for v in base.leaves()}
        assert {v.label for v in result.leaves()} == expected

    @pytest.mark.parametrize(
        "seed,contraction_prob",
        [(s, p) for s in [0, 42] for p in [0.5, 0.7, 0.9]],
    )
    def test_various_contraction_rates(self, seed, contraction_prob):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base, n=8, contraction_prob=contraction_prob)
        result = linear_common_refinement(partial)
        assert result is not None
        for T_i in partial:
            assert result.is_refinement(T_i)


class TestLinCR_DifferentLeafSets:
    """LinCR raises when trees do not share the same leaf set."""

    def test_raises_on_different_leaf_sets(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        with pytest.raises(ValueError):
            linear_common_refinement([T1, T2])


class TestLinCR_RunOnceConstraint:
    """LinCR.run() may only be called once per instance."""

    def test_second_run_raises(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        cr = LinCR([T1, T2])
        cr.run()
        with pytest.raises(ValueError):
            cr.run()


class TestLinCRClass:
    """LinCR class interface mirrors the convenience function."""

    def test_class_and_function_agree(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,b,c),d);")
        result_fn = linear_common_refinement([T1, T2])
        result_cls = LinCR([T1, T2]).run()
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_class_and_function_agree_on_random_trees(self, seed):
        random.seed(seed)
        base = Tree.random_tree(25, binary=True)
        partial = make_partial_trees(base, n=6)
        result_fn = linear_common_refinement(partial)
        result_cls = LinCR(partial).run()
        assert (result_fn is None) == (result_cls is None)
        if result_fn is not None:
            assert result_fn.equal_topology(result_cls)
