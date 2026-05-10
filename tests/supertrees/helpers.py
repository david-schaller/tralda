"""Shared helpers and constants for tralda.supertree tests."""

from __future__ import annotations

import random
from typing import Any

from tralda.datastructures import Tree

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

# Random seeds used to parameterise stochastic tests
SEEDS = [0, 42, 137, 271, 999]


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_partial_trees(
    tree: Tree,
    n: int = 10,
    contraction_prob: float = 0.9,
) -> list[Tree]:
    """Return *n* partial trees derived from *tree* by randomly contracting inner edges.

    The global ``random`` state is used so that the caller is responsible for seeding ``random``
    before calling this function to achieve reproducibility.

    Args:
        tree: The base tree from which partial trees are derived.
        n: Number of partial trees to generate.
        contraction_prob: Probability that any given inner edge is contracted.

    Returns:
        A list of *n* partial trees all sharing the same leaf set as *tree*.
    """
    partial = []
    for _ in range(n):
        T_i = tree.copy()
        edges = [(u, v) for u, v in T_i.inner_edges() if random.random() < contraction_prob]
        T_i.contract(edges)
        partial.append(T_i)

    return partial


def restrict_to_leaves(tree: Tree, keep_labels: set) -> Tree:
    """Return a copy of *tree* restricted to leaves whose labels are in *keep_labels*.

    Leaves outside *keep_labels* are removed; degree-1 inner nodes that result from leaf
    removals are then suppressed. If the root ends up with a single child it is replaced by
    that child.

    Args:
        tree: The source tree (not modified).
        keep_labels: Set of leaf labels to retain.

    Returns:
        A new ``Tree`` whose leaf set equals *keep_labels* ∩ (leaf labels of *tree*).
    """
    restricted = tree.copy()

    # Repeatedly remove leaves outside keep_labels until none remain.
    # Each removal can expose a new leaf (an inner node whose last child just left).
    changed = True
    while changed:
        to_remove = [v for v in restricted.leaves() if v.label not in keep_labels]
        changed = bool(to_remove)
        for v in to_remove:
            restricted.delete_and_reconnect(v)

    # Suppress non-root degree-1 inner nodes.
    changed = True
    while changed:
        changed = False
        for v in list(restricted.inner_nodes()):
            if len(v.children) == 1 and v is not restricted.root:
                restricted.delete_and_reconnect(v)
                changed = True

    # Promote the root's sole child to root when the root itself became degree-1.
    while restricted.root is not None and len(restricted.root.children) == 1:
        new_root = restricted.root.children[0]
        new_root.parent = None
        restricted.root = new_root

    return restricted


def displays_all_triples(supertree: Tree, triples: set[tuple[Any, Any, Any]]) -> bool:
    """Return True iff *supertree* displays every triple in *triples*.

    A triple ``(a, b, c)`` represents ``ab|c``.  Because ``ab|c = ba|c`` both orderings of the
    first two leaves are accepted.

    Args:
        supertree: The tree whose triple set is checked.
        triples: The triples that must be displayed.

    Returns:
        True if every triple is displayed by *supertree*.
    """
    st_triples: set[tuple[Any, Any, Any]] = set(supertree.get_triples(label_only=True))
    for a, b, c in triples:
        if (a, b, c) not in st_triples and (b, a, c) not in st_triples:
            return False

    return True
