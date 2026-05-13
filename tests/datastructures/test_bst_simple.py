"""Tests for tralda.datastructures.bst.simple.BinarySearchTree."""

from __future__ import annotations

import pytest
import numpy as np

from tralda.datastructures.bst.simple import BinarySearchTree


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sorted_keys(tree: BinarySearchTree) -> list:
    return list(tree)


def build_tree(keys) -> BinarySearchTree:
    t = BinarySearchTree()
    for k in keys:
        t.insert(k)
    return t


# ---------------------------------------------------------------------------
# Construction / emptiness
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_tree_has_len_zero(self):
        assert len(BinarySearchTree()) == 0

    def test_empty_tree_is_falsy(self):
        assert not BinarySearchTree()

    def test_non_empty_tree_is_truthy(self):
        t = build_tree([1])
        assert t

    def test_empty_tree_does_not_contain_any_key(self):
        t = BinarySearchTree()
        assert 0 not in t
        assert "x" not in t


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------


class TestInsert:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_len_after_insert(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = BinarySearchTree()
        for i, k in enumerate(keys):
            t.insert(k)
            assert len(t) == i + 1

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_insert_raises_key_error_for_duplicate(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(50).tolist()
        t = build_tree(keys)
        for k in keys[:10]:
            with pytest.raises(KeyError):
                t.insert(k)

    def test_add_silently_ignores_duplicate(self):
        t = BinarySearchTree()
        t.add(1)
        t.add(1)
        assert len(t) == 1

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_insert_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(300).tolist()
        t = build_tree(keys)
        assert t.check_integrity(verbose=True)


# ---------------------------------------------------------------------------
# Iteration / sorted order
# ---------------------------------------------------------------------------


class TestIteration:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_iteration_yields_sorted_order(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(300).tolist()
        t = build_tree(keys)
        assert sorted_keys(t) == sorted(keys)

    def test_empty_tree_iteration_yields_nothing(self):
        assert sorted_keys(BinarySearchTree()) == []

    def test_iter_does_not_return_self(self):
        t = build_tree([1, 2, 3])
        assert iter(t) is not t

    def test_next_on_tree_raises_type_error(self):
        t = build_tree([1])
        with pytest.raises(TypeError):
            next(t)


# ---------------------------------------------------------------------------
# Membership
# ---------------------------------------------------------------------------


class TestContains:
    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_contains_inserted_keys(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(100).tolist()
        t = build_tree(keys)
        for k in keys:
            assert k in t

    @pytest.mark.parametrize("seed", [0, 42])
    def test_does_not_contain_absent_keys(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(50).tolist()
        t = build_tree(keys)
        for k in range(50, 100):
            assert k not in t


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class TestRemove:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_remove_decrements_len(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        to_remove = keys[:20]
        t = build_tree(keys)
        for i, k in enumerate(to_remove):
            t.remove(k)
            assert len(t) == len(keys) - i - 1

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_remove_key_no_longer_present(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(100).tolist()
        to_remove = keys[:10]
        t = build_tree(keys)
        for k in to_remove:
            t.remove(k)
            assert k not in t

    @pytest.mark.parametrize("seed", [0, 42])
    def test_remove_raises_key_error_for_absent_key(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(50).tolist()
        t = build_tree(keys)
        with pytest.raises(KeyError):
            t.remove(9999)

    def test_discard_absent_key_is_silent(self):
        t = build_tree([1, 2, 3])
        t.discard(9999)  # must not raise
        assert len(t) == 3

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_remove_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(300).tolist()
        to_remove = keys[:50]
        t = build_tree(keys)
        for k in to_remove:
            t.remove(k)
        assert t.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_sorted_order_after_removal(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        to_remove = set(keys[:30])
        t = build_tree(keys)
        for k in to_remove:
            t.remove(k)
        expected = sorted(set(keys) - to_remove)
        assert sorted_keys(t) == expected


# ---------------------------------------------------------------------------
# Index-based access
# ---------------------------------------------------------------------------


class TestIndexAccess:
    @pytest.mark.parametrize("idx", [0, 1, 5, -1, -5, -10])
    def test_key_at_index(self, idx):
        keys = list(range(10))
        t = build_tree(keys)
        assert t.key_at_index(idx) == sorted(keys)[idx]

    def test_getitem_matches_key_at_index(self):
        keys = list(range(10))
        t = build_tree(keys)
        for i in range(10):
            assert t[i] == t.key_at_index(i)

    @pytest.mark.parametrize("idx", [10, 100, -11, -100])
    def test_key_at_index_out_of_bounds_raises_index_error(self, idx):
        keys = list(range(10))
        t = build_tree(keys)
        with pytest.raises(IndexError):
            t.key_at_index(idx)

    def test_key_at_index_empty_tree_raises_index_error(self):
        with pytest.raises(IndexError):
            BinarySearchTree().key_at_index(0)


# ---------------------------------------------------------------------------
# pop / pop_at_index
# ---------------------------------------------------------------------------


class TestPop:
    def test_pop_returns_greatest_key(self):
        keys = list(range(10))
        t = build_tree(keys)
        assert t.pop() == 9

    def test_pop_removes_key(self):
        keys = list(range(10))
        t = build_tree(keys)
        t.pop()
        assert 9 not in t
        assert len(t) == 9

    def test_pop_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            BinarySearchTree().pop()

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_pop_at_index(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(100).tolist()
        t = build_tree(keys)
        sorted_ref = sorted(keys)

        # pop every third element from the front
        to_pop = list(range(0, 30, 3))
        expected = [sorted_ref[i] for i in to_pop]

        # adjust indices as elements are removed
        for offset, i in enumerate(to_pop):
            actual_idx = i - offset
            val = t.pop_at_index(actual_idx)
            assert val == expected[offset]

        assert t.check_integrity(verbose=True)

    def test_pop_at_index_negative(self):
        t = build_tree(range(10))
        assert t.pop_at_index(-1) == 9
        assert t.pop_at_index(-1) == 8

    def test_pop_at_index_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            BinarySearchTree().pop_at_index(0)


# ---------------------------------------------------------------------------
# clear / copy
# ---------------------------------------------------------------------------


class TestClearAndCopy:
    def test_clear_empties_tree(self):
        t = build_tree(range(20))
        t.clear()
        assert len(t) == 0
        assert not t

    def test_copy_has_same_elements(self):
        keys = list(range(20))
        t = build_tree(keys)
        t_copy = t.copy()
        assert sorted_keys(t_copy) == sorted(keys)

    def test_copy_is_independent(self):
        keys = list(range(20))
        t = build_tree(keys)
        t_copy = t.copy()
        t_copy.insert(9999)
        assert 9999 not in t

    def test_copy_check_integrity(self):
        t = build_tree(range(50))
        t_copy = t.copy()
        assert t_copy.check_integrity(verbose=True)


# ---------------------------------------------------------------------------
# Large random sequences
# ---------------------------------------------------------------------------


class TestLargeRandom:
    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_insert_and_remove_large(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(5000).tolist()
        to_remove = keys[:500]
        t = build_tree(keys)
        for k in to_remove:
            t.remove(k)
        assert len(t) == len(keys) - len(to_remove)
        assert sorted_keys(t) == sorted(set(keys) - set(to_remove))
        assert t.check_integrity(verbose=True)
