"""Shared fixtures and helpers used across the tralda test suite."""

from __future__ import annotations

import pytest

from tralda.datastructures import Tree

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: A fixed Newick string used throughout the tree tests.  The tree has 31
#: nodes (labels 0–30) and height 6.
EXAMPLE_NEWICK = (
    "((((14,15,(27,(29,30)28)18)10,(19,20)11,16)7,8,9,17)1,(4,5)2,3,((23,24)12,13,22)6,(25,26)21)0;"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def example_newick() -> str:
    """Return the fixed example Newick string."""
    return EXAMPLE_NEWICK


@pytest.fixture
def example_tree() -> Tree:
    """Return a freshly parsed copy of the fixed example tree."""
    return Tree.parse_newick(EXAMPLE_NEWICK)


@pytest.fixture
def random_tree_20() -> Tree:
    """Return a random phylogenetic tree with 20 leaves."""
    return Tree.random_tree(20)
