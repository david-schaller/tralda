"""Shared fixtures and helpers for tralda.cograph tests."""

from __future__ import annotations

import itertools
import random
from typing import Any

import networkx as nx
import pytest

from tralda.cograph import random_cotree
from tralda.datastructures.tree import Tree


# ---------------------------------------------------------------------------
# Helper functions (module-level, usable in all test files)
# ---------------------------------------------------------------------------


def is_valid_cotree_structure(cotree: Tree) -> bool:
    """Return True if every inner node has label 'series' or 'parallel'."""
    for v in cotree.inner_nodes():
        if v.label not in ("series", "parallel"):
            return False

    return True


def is_discriminating(cotree: Tree) -> bool:
    """Return True if no two adjacent inner nodes share the same label."""
    for v in cotree.inner_nodes():
        if v.parent is not None and v.parent.label == v.label:
            return False

    return True


def leaf_labels(cotree: Tree) -> set:
    """Return the set of leaf labels (= vertex identifiers) in a cotree."""
    return {v.label for v in cotree.leaves()}


def is_partition_of(partition: list[list[Any]], vertex_set: set) -> bool:
    """Return True if *partition* is a valid partition of *vertex_set*."""
    covered = [v for part in partition for v in part]
    return set(covered) == vertex_set and len(covered) == len(vertex_set)


def is_clique(G: nx.Graph, vertices: list) -> bool:
    """Return True if every pair in *vertices* is connected by an edge in G."""
    for u, v in itertools.combinations(vertices, 2):
        if not G.has_edge(u, v):
            return False

    return True


def is_independent_set(G: nx.Graph, vertices: list) -> bool:
    """Return True if no two vertices in *vertices* are connected in G."""
    for u, v in itertools.combinations(vertices, 2):
        if G.has_edge(u, v):
            return False

    return True


def is_subgraph_of(sub: nx.Graph, sup: nx.Graph) -> bool:
    """Return True if every edge of *sub* is also present in *sup*."""
    if not set(sub.nodes()) <= set(sup.nodes()):
        return False

    return all(sup.has_edge(u, v) for u, v in sub.edges())


def is_complete_multipartite(G: nx.Graph, partition: list[list]) -> bool:
    """Return True if G contains every cross-part edge implied by *partition*."""
    for part1, part2 in itertools.combinations(partition, 2):
        for u in part1:
            for v in part2:
                if not G.has_edge(u, v):
                    return False

    return True


def graphs_equal(G: nx.Graph, H: nx.Graph) -> bool:
    """Return True if G and H have exactly the same vertices and edges."""
    return (
        set(G.nodes()) == set(H.nodes())
        and all(H.has_edge(u, v) for u, v in G.edges())
        and all(G.has_edge(u, v) for u, v in H.edges())
    )


# ---------------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def p4() -> nx.Graph:
    """The path P_4 — the minimal forbidden induced subgraph of cographs."""
    return nx.path_graph(4)


@pytest.fixture
def k5() -> nx.Graph:
    """The complete graph K_5."""
    return nx.complete_graph(5)


@pytest.fixture
def k23() -> nx.Graph:
    """The complete bipartite graph K_{2,3}."""
    return nx.complete_bipartite_graph(2, 3)


@pytest.fixture(params=[7, 42, 99, 137, 256])
def seeded_cotree_10(request) -> Tree:
    """A random cotree on 10 leaves for each of five fixed seeds."""
    random.seed(request.param)
    return random_cotree(10)


@pytest.fixture(params=[7, 42, 99, 137, 256])
def seeded_cotree_20_connected(request) -> Tree:
    """A random connected cotree on 20 leaves (series root) for each of five fixed seeds."""
    random.seed(request.param)
    return random_cotree(20, force_series_root=True)
