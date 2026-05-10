"""Shared pytest fixtures for tralda.supertree tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def compatible_newick_pair() -> tuple[str, str]:
    """A small pair of compatible trees used in multiple tests."""
    return "((a,b),c,d);", "((a,b,c),d);"


@pytest.fixture
def incompatible_newick_pair() -> tuple[str, str]:
    """A small pair of *incompatible* trees (clusters {a,b} and {a,c} overlap)."""
    return "((a,b),c,d);", "((a,c),b,d);"
