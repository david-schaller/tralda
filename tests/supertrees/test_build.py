"""Tests for tralda.supertree.build (BUILD algorithm and helpers)."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures import Tree
from tralda.supertree.build import (
    Build,
    aho_graph,
    best_pair_merge_first,
    build_supertree,
    greedy_build,
    minimal_identifying_triple_set,
    tree_profile_to_triples,
)
from tests.supertrees.helpers import SEEDS, displays_all_triples, make_partial_trees


# ══════════════════════════════════════════════════════════════════════════════
# aho_graph
# ══════════════════════════════════════════════════════════════════════════════


class TestAhoGraph:
    def test_all_leaves_present_as_nodes(self):
        L = {"a", "b", "c"}
        G = aho_graph([("a", "b", "c")], L)
        assert set(G.nodes) == L

    def test_triple_creates_edge(self):
        G = aho_graph([("a", "b", "c")], {"a", "b", "c"})
        assert G.has_edge("a", "b")

    def test_triple_does_not_create_other_edges(self):
        G = aho_graph([("a", "b", "c")], {"a", "b", "c"})
        assert not G.has_edge("a", "c")
        assert not G.has_edge("b", "c")

    def test_empty_triple_set_no_edges(self):
        G = aho_graph([], {"a", "b", "c"})
        assert G.number_of_edges() == 0

    def test_multiple_triples_merge_edges(self):
        triples = [("a", "b", "c"), ("a", "b", "d")]
        G = aho_graph(triples, {"a", "b", "c", "d"}, weighted=True)
        assert G.has_edge("a", "b")
        assert G["a"]["b"]["weight"] == pytest.approx(2.0)

    def test_weighted_single_triple(self):
        triples = [("a", "b", "c")]
        G = aho_graph(
            triples,
            {"a", "b", "c"},
            weighted=True,
            triple_weights={("a", "b", "c"): 3.0},
        )
        assert G["a"]["b"]["weight"] == pytest.approx(3.0)


# ══════════════════════════════════════════════════════════════════════════════
# minimal_identifying_triple_set
# ══════════════════════════════════════════════════════════════════════════════


class TestMinimalIdentifyingTripleSet:
    def test_star_tree_yields_no_triples(self):
        T = Tree.parse_newick("(a,b,c,d);")
        triples = list(minimal_identifying_triple_set(T))
        assert triples == []

    def test_binary_tree_rebuild(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        triples = list(minimal_identifying_triple_set(T))
        leaves = {node.label for node in T.leaves()}
        rebuilt = Build(triples, leaves).build_tree()
        assert rebuilt is not None
        assert rebuilt.equal_topology(T)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_binary_tree_rebuild(self, seed):
        random.seed(seed)
        T = Tree.random_tree(20, binary=True)
        triples = list(minimal_identifying_triple_set(T))
        leaves = {node.label for node in T.leaves()}
        rebuilt = Build(triples, leaves).build_tree()
        assert rebuilt is not None
        assert rebuilt.equal_topology(T)


# ══════════════════════════════════════════════════════════════════════════════
# tree_profile_to_triples
# ══════════════════════════════════════════════════════════════════════════════


class TestTreeProfileToTriples:
    def test_leaf_set_is_union(self):
        T1 = Tree.parse_newick("((a,b),c);")
        T2 = Tree.parse_newick("((b,c),d);")
        leaves, _ = tree_profile_to_triples([T1, T2])
        assert leaves == {"a", "b", "c", "d"}

    def test_triples_are_displayed_by_input(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        _, triples = tree_profile_to_triples([T])
        assert displays_all_triples(T, triples)

    def test_single_tree_triples_rebuild(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        rebuilt = Build(triples, leaves).build_tree()
        assert rebuilt is not None
        assert rebuilt.equal_topology(T)

    def test_star_tree_empty_triples(self):
        T = Tree.parse_newick("(a,b,c,d);")
        _, triples = tree_profile_to_triples([T])
        assert len(triples) == 0

    def test_each_tree_displays_its_own_triples(self):
        # Each input tree must display its OWN minimal identifying triples.
        # It need not display triples extracted from a *different* tree.
        for nw in ["((a,b),c,d);", "((a,b,c),d);", "((a,b),(c,d));"]:
            T = Tree.parse_newick(nw)
            _, triples = tree_profile_to_triples([T])
            assert displays_all_triples(T, triples)


# ══════════════════════════════════════════════════════════════════════════════
# Build
# ══════════════════════════════════════════════════════════════════════════════


class TestBuild:
    # ── Edge cases ────────────────────────────────────────────────────────────

    def test_single_leaf(self):
        T = Build([], {"a"}).build_tree()
        assert T is not None
        leaves = [v.label for v in T.leaves()]
        assert leaves == ["a"]

    def test_two_leaves(self):
        T = Build([], {"a", "b"}).build_tree()
        assert T is not None
        assert {v.label for v in T.leaves()} == {"a", "b"}

    def test_empty_triples_gives_star(self):
        L = {"a", "b", "c", "d"}
        T = Build([], L).build_tree()
        assert T is not None
        assert T.is_phylogenetic()
        assert {v.label for v in T.leaves()} == L

    # ── Consistent triple sets ────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "newick",
        [
            "((a,b),c);",
            "((a,b),(c,d));",
            "(((a,b),c),d);",
            "((a,b,c),d);",
        ],
    )
    def test_consistent_triples_return_tree(self, newick):
        T = Tree.parse_newick(newick)
        leaves, triples = tree_profile_to_triples([T])
        result = Build(triples, leaves).build_tree()
        assert result is not None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_consistent_triples_return_correct_tree(self, seed):
        random.seed(seed)
        T = Tree.random_tree(25, binary=True)
        leaves, triples = tree_profile_to_triples([T])
        result = Build(triples, leaves).build_tree()
        assert result is not None
        assert result.equal_topology(T)

    # ── Inconsistent triple sets ──────────────────────────────────────────────

    @pytest.mark.parametrize(
        "triples,leaves",
        [
            # Circular dependency: ab|c, bc|a, ac|b
            ([("a", "b", "c"), ("b", "c", "a"), ("a", "c", "b")], {"a", "b", "c"}),
            # On 4 leaves: ab|c and ac|b imply overlapping clusters {a,b} and {a,c}
            ([("a", "b", "c"), ("a", "c", "b")], {"a", "b", "c", "d"}),
        ],
    )
    def test_inconsistent_triples_return_none(self, triples, leaves):
        result = Build(triples, leaves, mincut=False).build_tree()
        assert result is None

    # ── MinCut mode ───────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "triples,leaves",
        [
            ([("a", "b", "c"), ("b", "c", "a"), ("a", "c", "b")], {"a", "b", "c"}),
        ],
    )
    def test_mincut_always_returns_tree(self, triples, leaves):
        result = Build(triples, leaves, mincut=True).build_tree()
        assert result is not None
        assert result.is_phylogenetic()
        assert {v.label for v in result.leaves()} == leaves

    # ── return_root flag ──────────────────────────────────────────────────────

    def test_return_root_gives_treenode(self):
        from tralda.datastructures.tree import TreeNode

        T_ref = Tree.parse_newick("((a,b),c);")
        leaves, triples = tree_profile_to_triples([T_ref])
        root = Build(triples, leaves).build_tree(return_root=True)
        assert isinstance(root, TreeNode)

    def test_return_none_on_inconsistent_with_return_root(self):
        triples = [("a", "b", "c"), ("b", "c", "a"), ("a", "c", "b")]
        result = Build(triples, {"a", "b", "c"}, mincut=False).build_tree(return_root=True)
        assert result is None


# ══════════════════════════════════════════════════════════════════════════════
# build_supertree
# ══════════════════════════════════════════════════════════════════════════════


class TestBuildSupertree:
    # ── Known examples ────────────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            ("((a,b),c);", "((b,c),d);"),
            ("((a,b),c,d);", "((a,b,c),d);"),
            ("((a,b,c),d);", "(a,b,c,d);"),
        ],
    )
    def test_compatible_pair_returns_tree(self, newick1, newick2):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = build_supertree([T1, T2])
        assert result is not None
        assert result.is_phylogenetic()

    @pytest.mark.parametrize(
        "newick1,newick2",
        [
            ("((a,b),c);", "((b,c),d);"),
            ("((a,b),c,d);", "((a,b,c),d);"),
        ],
    )
    def test_compatible_pair_displays_all_triples(self, newick1, newick2):
        T1, T2 = Tree.parse_newick(newick1), Tree.parse_newick(newick2)
        result = build_supertree([T1, T2])
        assert result is not None
        _, triples = tree_profile_to_triples([T1, T2])
        assert displays_all_triples(result, triples)

    def test_single_tree_returns_equivalent_topology(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        result = build_supertree([T])
        assert result is not None
        assert result.equal_topology(T)

    def test_incompatible_returns_none(self):
        T1 = Tree.parse_newick("((a,b),c,d);")
        T2 = Tree.parse_newick("((a,c),b,d);")
        # Triple ab|c conflicts with ac|b – inconsistent
        result = build_supertree([T1, T2])
        assert result is None

    # ── Random tests ──────────────────────────────────────────────────────────

    @pytest.mark.parametrize("seed", SEEDS)
    def test_partial_trees_display_triples(self, seed):
        random.seed(seed)
        base = Tree.random_tree(30, binary=True)
        partial = make_partial_trees(base)
        result = build_supertree(partial)
        if result is not None:
            _, triples = tree_profile_to_triples(partial)
            assert displays_all_triples(result, triples)

    @pytest.mark.parametrize("seed", SEEDS)
    def test_leaf_set_equals_union(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base, n=5)
        result = build_supertree(partial)
        if result is not None:
            expected_leaves = {v.label for t in partial for v in t.leaves()}
            actual_leaves = {v.label for v in result.leaves()}
            assert actual_leaves == expected_leaves

    @pytest.mark.parametrize("seed", SEEDS)
    def test_result_is_phylogenetic(self, seed):
        random.seed(seed)
        base = Tree.random_tree(20, binary=True)
        partial = make_partial_trees(base)
        result = build_supertree(partial)
        if result is not None:
            assert result.is_phylogenetic()


# ══════════════════════════════════════════════════════════════════════════════
# greedy_build
# ══════════════════════════════════════════════════════════════════════════════


class TestGreedyBuild:
    def test_consistent_triples_returns_tree(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        result = greedy_build(list(triples), leaves)
        assert result is not None
        assert result.is_phylogenetic()
        assert {v.label for v in result.leaves()} == leaves

    def test_inconsistent_triples_returns_tree(self):
        triples = [("a", "b", "c"), ("b", "c", "a"), ("a", "c", "b")]
        leaves = {"a", "b", "c"}
        result = greedy_build(triples, leaves)
        assert result is not None
        assert {v.label for v in result.leaves()} == leaves

    def test_consistent_recovers_original_topology(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        result = greedy_build(list(triples), leaves)
        assert result is not None
        assert result.equal_topology(T)

    def test_with_weights_accepts_uniform_weights(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        triples = list(triples)
        weights = {t: 1.0 for t in triples}
        result = greedy_build(triples, leaves, triple_weights=weights)
        assert result is not None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_consistent_returns_tree_with_correct_leaves(self, seed):
        random.seed(seed)
        T = Tree.random_tree(15, binary=True)
        leaves, triples = tree_profile_to_triples([T])
        result = greedy_build(list(triples), leaves)
        assert result is not None
        assert {v.label for v in result.leaves()} == leaves


# ══════════════════════════════════════════════════════════════════════════════
# best_pair_merge_first
# ══════════════════════════════════════════════════════════════════════════════


class TestBestPairMergeFirst:
    def test_consistent_triples_returns_tree(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        result = best_pair_merge_first(list(triples), leaves)
        assert result is not None
        assert result.is_phylogenetic()
        assert {v.label for v in result.leaves()} == leaves

    def test_inconsistent_triples_returns_tree(self):
        triples = [("a", "b", "c"), ("b", "c", "a"), ("a", "c", "b")]
        leaves = {"a", "b", "c"}
        result = best_pair_merge_first(triples, leaves)
        assert result is not None
        assert {v.label for v in result.leaves()} == leaves

    def test_with_uniform_weights_returns_tree(self):
        T = Tree.parse_newick("((a,b),(c,d));")
        leaves, triples = tree_profile_to_triples([T])
        triples = list(triples)
        weights = {t: 1.0 for t in triples}
        result = best_pair_merge_first(triples, leaves, triple_weights=weights)
        assert result is not None

    @pytest.mark.parametrize("seed", SEEDS)
    def test_random_returns_tree_with_correct_leaves(self, seed):
        random.seed(seed)
        T = Tree.random_tree(15, binary=True)
        leaves, triples = tree_profile_to_triples([T])
        result = best_pair_merge_first(list(triples), leaves)
        assert result is not None
        assert {v.label for v in result.leaves()} == leaves
