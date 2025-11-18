"""Collection of functions for tree analysis and comparison."""

from __future__ import annotations

from collections.abc import Collection
from typing import Any

from tralda.datastructures.tree import Tree


def assert_leaf_sets_equal(trees: Collection[Tree]) -> set[Any] | None:
    """Checks whether trees have the same set of unique leaf labels.

    Args:
        trees: Collection of Tree instances.

    Returns:
        The common set of leaf labels; or None if the trees do not share the same set of leaf
        labels.

    Raises:
        ValueError: If the input collection is empty, contains empty trees, or the leaf labels are
            not unique in one tree.
        TypeError: If any of the input elements is not of type Tree.
    """
    if not trees:
        raise ValueError("empty list of trees")

    leaves = None

    for T_i in trees:
        if not isinstance(T_i, Tree):
            raise TypeError("not a 'Tree' instance")

        leaves2 = set()
        for v in T_i.leaves():
            if v.label in leaves2:
                raise ValueError("leaf labels not unique")
            else:
                leaves2.add(v.label)

        if not leaves:
            leaves = leaves2
            if len(leaves) == 0:
                raise ValueError("empty tree in tree list")
        elif leaves != leaves2:
            return

    return leaves
