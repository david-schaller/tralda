"""Shared fixtures and helpers for tralda.visualization tests."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures.tree import Tree, TreeNode


def _node(label: str, dist: float) -> TreeNode:
    node = TreeNode(label=label)
    node.dist = dist
    return node


@pytest.fixture
def small_tree() -> Tree:
    """A small, deterministic, perfectly balanced binary tree.

    Structure (label: dist)::

                       r
                    /     \\
                a(1.0)     b(2.0)
                /   \\       /   \\
            c(1.0) d(1.5) e(0.5) f(1.0)

    Resulting ATTR depths: r=0, a=1.0, b=2.0, c=2.0, d=2.5, e=2.5, f=3.0.
    Postorder leaf ranks: c=0, d=1, e=2, f=3.
    """
    root = _node("r", 0.0)
    a, b = _node("a", 1.0), _node("b", 2.0)
    c, d = _node("c", 1.0), _node("d", 1.5)
    e, f = _node("e", 0.5), _node("f", 1.0)
    root.add_child(a)
    root.add_child(b)
    a.add_child(c)
    a.add_child(d)
    b.add_child(e)
    b.add_child(f)

    return Tree(root)


@pytest.fixture
def unbalanced_tree() -> Tree:
    """A small tree with leaves at different topological depths.

    Structure (label: dist)::

                    r
                 /     \\
             x(1.0)     y
                       /     \\
                   z(1.0)   w(1.0)

    ``x`` is a leaf at topological depth 1; ``z`` and ``w`` are leaves at depth 2.  Useful for
    distinguishing ``UNIFORM`` (leaves may end at different depths) from ``RANK`` (leaves are
    extended to the maximum topological depth) edge-length modes.
    """
    root = _node("r", 0.0)
    x = _node("x", 1.0)
    y = _node("y", 1.0)
    z = _node("z", 1.0)
    w = _node("w", 1.0)
    root.add_child(x)
    root.add_child(y)
    y.add_child(z)
    y.add_child(w)

    return Tree(root)


@pytest.fixture
def star_tree() -> Tree:
    """A root with three leaf children (a multifurcation, no grandchildren)."""
    root = _node("r", 0.0)
    for lbl in ("a", "b", "c"):
        root.add_child(_node(lbl, 1.0))

    return Tree(root)


@pytest.fixture
def single_node_tree() -> Tree:
    """A tree consisting of a single root node with no children."""
    return Tree(_node("only", 0.0))


@pytest.fixture
def empty_tree() -> Tree:
    """A tree with no root at all."""
    return Tree(None)


@pytest.fixture
def random_labeled_tree():
    """Factory fixture: a random binary tree with ``label``/``dist`` attributes for a given seed."""

    def _make(n_leaves: int = 8, seed: int = 0) -> Tree:
        random.seed(seed)
        tree = Tree.random_tree(n_leaves, binary=True)
        for i, v in enumerate(tree.preorder()):
            v.label = f"n{i}"
            v.dist = 0.5 + (i % 5) * 0.25
        return tree

    return _make
