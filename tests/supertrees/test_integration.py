"""Integration tests comparing all supertree algorithms.

For trees on the same leaf set that are compatible (derived from the same base
tree by edge contraction), all three methods —

    * ``linear_common_refinement``
    * ``build_supertree``
    * ``build_st``

— must agree on topology and each result must be a refinement of every input
tree.
"""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree
from tralda.supertree import (
    build_st,
    build_supertree,
    linear_common_refinement,
)
from tests.supertrees.helpers import SEEDS, make_partial_trees


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════


def _all_refinements_of_inputs(result: Tree, partial_trees: list[Tree]) -> bool:
    return all(result.is_refinement(T_i) for T_i in partial_trees)


# ══════════════════════════════════════════════════════════════════════════════
# All algorithms on hand-crafted examples
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegrationKnownExamples:
    """Validate all three algorithms on explicit Newick inputs."""

    @pytest.mark.parametrize(
        "newicks,expected_cr",
        [
            (["((a,b),c,d);", "((a,b,c),d);"], "(((a,b),c),d);"),
            (["(a,b,c,d);", "((a,b),c,d);"], "((a,b),c,d);"),
            (
                ["((a,b),c,d,e);", "((a,b,c),d,e);", "((a,b,c,d),e);"],
                "((((a,b),c),d),e);",
            ),
        ],
    )
    def test_all_methods_agree_on_compatible_trees(self, newicks, expected_cr):
        trees = [Tree.parse_newick(nw) for nw in newicks]
        cr_expected = Tree.parse_newick(expected_cr)

        cr = linear_common_refinement(trees)
        bs = build_supertree(trees)
        bst = build_st(trees)

        assert cr is not None, "linear_common_refinement returned None"
        assert bs is not None, "build_supertree returned None"
        assert bst is not None, "build_st returned None"

        assert cr.equal_topology(cr_expected), "LinCR topology mismatch"
        assert bs.equal_topology(cr_expected), "build_supertree topology mismatch"
        assert bst.equal_topology(cr_expected), "build_st topology mismatch"

    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            ("((a,b),c,d);", "((a,c),b,d);"),
            ("((a,b),c,d);", "((b,c),a,d);"),
        ],
    )
    def test_all_methods_return_none_for_incompatible(self, newick1, newick2):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        assert linear_common_refinement([T1, T2]) is None
        assert build_supertree([T1, T2]) is None
        assert build_st([T1, T2]) is None


# ══════════════════════════════════════════════════════════════════════════════
# All algorithms on randomly generated partial trees
# ══════════════════════════════════════════════════════════════════════════════


class TestIntegrationRandomTrees:
    """Compare LinCR, build_supertree, and build_st on random partial trees."""

    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_methods_agree_on_topology(self, seed):
        random.seed(seed)
        base = Tree.random_tree(50, binary=True)
        partial = make_partial_trees(base, n=10, contraction_prob=0.9)

        cr = linear_common_refinement(partial)
        bs = build_supertree(partial)
        bst = build_st(partial)

        # Partial trees are always compatible (derived from the same base) so all
        # three must return a non-None result.
        assert cr is not None, "linear_common_refinement returned None on compatible trees"
        assert bs is not None, "build_supertree returned None on compatible trees"
        assert bst is not None, "build_st returned None on compatible trees"

        assert bs.equal_topology(bst), "build_supertree and build_st disagree"
        assert bs.equal_topology(cr), "build_supertree and linear_common_refinement disagree"

    @pytest.mark.parametrize("seed", SEEDS)
    def test_all_methods_refine_each_input(self, seed):
        random.seed(seed)
        base = Tree.random_tree(40, binary=True)
        partial = make_partial_trees(base, n=8, contraction_prob=0.8)

        for result, name in [
            (linear_common_refinement(partial), "LinCR"),
            (build_supertree(partial), "build_supertree"),
            (build_st(partial), "build_st"),
        ]:
            assert result is not None, f"{name} returned None"
            assert _all_refinements_of_inputs(result, partial), (
                f"{name} result is not a refinement of all input trees"
            )

    @pytest.mark.parametrize(
        "seed,n_leaves,n_partial",
        [
            (0, 20, 5),
            (42, 30, 8),
            (137, 40, 10),
            (271, 25, 6),
            (999, 35, 7),
        ],
    )
    def test_varying_sizes(self, seed, n_leaves, n_partial):
        random.seed(seed)
        base = Tree.random_tree(n_leaves, binary=True)
        partial = make_partial_trees(base, n=n_partial, contraction_prob=0.7)

        cr = linear_common_refinement(partial)
        bs = build_supertree(partial)
        bst = build_st(partial)

        assert cr is not None
        assert bs is not None
        assert bst is not None
        assert bs.equal_topology(cr)
        assert bs.equal_topology(bst)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_base_tree_is_refinement_of_result(self, seed):
        """The original base tree must be a refinement of the computed supertree.

        The supertree is the minimal common refinement of the partial trees, which is coarser than
        or equal to the base tree.
        """
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base, n=8, contraction_prob=0.8)

        cr = linear_common_refinement(partial)
        assert cr is not None
        assert base.is_refinement(cr), (
            "The base tree is not a refinement of its own common refinement"
        )
