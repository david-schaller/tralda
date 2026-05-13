"""Tests for tralda.datastructures.bst.avl (TreeSet and TreeDict)."""

from __future__ import annotations

import pytest
import numpy as np

from tralda.datastructures.bst.avl import TreeSet, TreeDict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sorted_keys(tree: TreeSet) -> list:
    return list(tree)


def build_tree(keys) -> TreeSet:
    t = TreeSet()
    for k in keys:
        t.insert(k)
    return t


# ---------------------------------------------------------------------------
# Construction / emptiness
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_empty_tree_has_len_zero(self):
        assert len(TreeSet()) == 0

    def test_empty_tree_is_falsy(self):
        assert not TreeSet()

    def test_non_empty_tree_is_truthy(self):
        assert build_tree([1])

    def test_empty_tree_does_not_contain_any_key(self):
        t = TreeSet()
        assert 0 not in t
        assert "x" not in t


# ---------------------------------------------------------------------------
# Insertion
# ---------------------------------------------------------------------------


class TestInsert:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_len_after_insert(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(300).tolist()
        t = TreeSet()
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
        t = TreeSet()
        t.add(1)
        t.add(1)
        assert len(t) == 1

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_insert_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        t = build_tree(keys)
        assert t.check_integrity(verbose=True)


# ---------------------------------------------------------------------------
# AVL balance property
# ---------------------------------------------------------------------------


class TestAVLBalance:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_balance_factor_within_bounds_after_insert(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        t = build_tree(keys)
        for node in t._inorder_traversal():
            assert abs(node.balance()) <= 1, f"node {node} is unbalanced"

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_balance_factor_within_bounds_after_removal(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        t = build_tree(keys)
        for k in keys[:100]:
            t.remove(k)
        for node in t._inorder_traversal():
            assert abs(node.balance()) <= 1, f"node {node} is unbalanced"

    @pytest.mark.parametrize(
        "keys",
        [
            list(range(100)),  # ascending insertion (worst case for BST)
            list(range(99, -1, -1)),  # descending insertion
            [i for i in range(0, 200, 2)],  # even keys only
        ],
    )
    def test_balance_after_sequential_insertion(self, keys):
        t = build_tree(keys)
        assert t.check_integrity(verbose=True)
        for node in t._inorder_traversal():
            assert abs(node.balance()) <= 1


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
        assert sorted_keys(TreeSet()) == []

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
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        for k in keys:
            assert k in t

    @pytest.mark.parametrize("seed", [0, 42])
    def test_does_not_contain_absent_keys(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(100).tolist()
        t = build_tree(keys)
        for k in range(100, 200):
            assert k not in t


# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------


class TestRemove:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_remove_decrements_len(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(300).tolist()
        to_remove = keys[:30]
        t = build_tree(keys)
        for i, k in enumerate(to_remove):
            t.remove(k)
            assert len(t) == len(keys) - i - 1

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_remove_key_no_longer_present(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        to_remove = keys[:20]
        t = build_tree(keys)
        for k in to_remove:
            t.remove(k)
            assert k not in t

    def test_remove_raises_key_error_for_absent_key(self):
        t = build_tree([1, 2, 3])
        with pytest.raises(KeyError):
            t.remove(9999)

    def test_discard_absent_key_is_silent(self):
        t = build_tree([1, 2, 3])
        t.discard(9999)
        assert len(t) == 3

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_remove_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        to_remove = keys[:100]
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
        assert sorted_keys(t) == sorted(set(keys) - to_remove)


# ---------------------------------------------------------------------------
# check_integrity propagation
# ---------------------------------------------------------------------------


class TestCheckIntegrity:
    def test_check_integrity_returns_false_on_corrupted_height(self):
        t = build_tree(range(10))
        t.root.height = 9999  # corrupt height
        assert t.check_integrity() is False

    def test_check_integrity_returns_false_on_corrupted_size(self):
        t = build_tree(range(10))
        t.root.size = 9999
        assert t.check_integrity() is False

    def test_check_integrity_returns_false_on_unbalanced_node(self):
        """AVL-specific: node with |balance| > 1 must fail check_integrity."""
        t = build_tree(range(10))
        # temporarily corrupt a balance by adjusting a subtree height
        if t.root.left:
            original_height = t.root.left.height
            t.root.left.height = 0  # force imbalance
            assert t.check_integrity() is False
            t.root.left.height = original_height  # restore

    def test_check_integrity_passes_fresh_tree(self):
        rng = np.random.default_rng(0)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        assert t.check_integrity(verbose=True) is True


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
        t = build_tree(range(10))
        with pytest.raises(IndexError):
            t.key_at_index(idx)

    def test_key_at_index_empty_tree_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().key_at_index(0)


# ---------------------------------------------------------------------------
# pop / pop_at_index
# ---------------------------------------------------------------------------


class TestPop:
    def test_pop_returns_greatest_key(self):
        t = build_tree(range(10))
        assert t.pop() == 9

    def test_pop_removes_key(self):
        t = build_tree(range(10))
        t.pop()
        assert 9 not in t
        assert len(t) == 9

    def test_pop_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().pop()

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_pop_at_index_returns_correct_key(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        sorted_ref = sorted(keys)
        # pop from index 0 repeatedly
        for expected in sorted_ref:
            assert t.pop_at_index(0) == expected

    @pytest.mark.parametrize("seed", [0, 42])
    def test_pop_at_index_negative(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(50).tolist()
        t = build_tree(keys)
        sorted_ref = sorted(keys)
        for expected in reversed(sorted_ref):
            assert t.pop_at_index(-1) == expected

    def test_pop_at_index_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().pop_at_index(0)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_pop_at_index_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        for _ in range(50):
            t.pop_at_index(len(t) // 2)
        assert t.check_integrity(verbose=True)


# ---------------------------------------------------------------------------
# clear / copy
# ---------------------------------------------------------------------------


class TestClearAndCopy:
    def test_clear_empties_tree(self):
        t = build_tree(range(20))
        t.clear()
        assert len(t) == 0

    def test_copy_has_same_elements(self):
        keys = list(range(30))
        t = build_tree(keys)
        assert sorted_keys(t.copy()) == sorted(keys)

    def test_copy_is_independent(self):
        t = build_tree(range(20))
        t_copy = t.copy()
        t_copy.insert(9999)
        assert 9999 not in t

    def test_copy_check_integrity(self):
        t = build_tree(range(100))
        assert t.copy().check_integrity(verbose=True)


# ---------------------------------------------------------------------------
# TreeDict
# ---------------------------------------------------------------------------


class TestTreeDict:
    def test_insert_and_get(self):
        d = TreeDict()
        d.insert(1, "a")
        d.insert(2, "b")
        assert d[1] == "a"
        assert d[2] == "b"

    def test_get_with_default(self):
        d = TreeDict()
        d.insert(1, "a")
        assert d.get(1) == "a"
        assert d.get(999) is None
        assert d.get(999, "missing") == "missing"

    def test_get_absent_key_raises_key_error(self):
        d = TreeDict()
        with pytest.raises(KeyError):
            _ = d[999]

    def test_insert_duplicate_raises_key_error(self):
        d = TreeDict()
        d.insert(1, "a")
        with pytest.raises(KeyError):
            d.insert(1, "b")

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_keys_iteration(self, seed):
        rng = np.random.default_rng(seed)
        raw_keys = rng.permutation(100).tolist()
        d = TreeDict()
        for k in raw_keys:
            d.insert(k, k * 10)
        assert list(d.keys()) == sorted(raw_keys)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_values_iteration(self, seed):
        rng = np.random.default_rng(seed)
        raw_keys = rng.permutation(100).tolist()
        d = TreeDict()
        for k in raw_keys:
            d.insert(k, k * 10)
        assert list(d.values()) == [k * 10 for k in sorted(raw_keys)]

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_items_iteration(self, seed):
        rng = np.random.default_rng(seed)
        raw_keys = rng.permutation(50).tolist()
        d = TreeDict()
        for k in raw_keys:
            d.insert(k, k * 2)
        assert list(d.items()) == [(k, k * 2) for k in sorted(raw_keys)]

    def test_value_at_index(self):
        d = TreeDict()
        for k in range(10):
            d.insert(k, k * 100)
        for i in range(10):
            assert d.value_at_index(i) == i * 100

    def test_key_and_value_at_index(self):
        d = TreeDict()
        for k in range(10):
            d.insert(k, str(k))
        for i in range(10):
            assert d.key_and_value_at_index(i) == (i, str(i))

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_pop_at_index_returns_key_value_pair(self, seed):
        rng = np.random.default_rng(seed)
        raw_keys = rng.permutation(50).tolist()
        d = TreeDict()
        for k in raw_keys:
            d.insert(k, k * 3)
        result = d.pop_at_index(0)
        assert result == (min(raw_keys), min(raw_keys) * 3)

    def test_check_integrity(self):
        d = TreeDict()
        rng = np.random.default_rng(0)
        for k in rng.permutation(200).tolist():
            d.insert(k, None)
        assert d.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_sorted_order_after_removal(self, seed):
        rng = np.random.default_rng(seed)
        raw_keys = rng.permutation(100).tolist()
        to_remove = raw_keys[:10]
        d = TreeDict()
        for k in raw_keys:
            d.insert(k, None)
        for k in to_remove:
            d.remove(k)
        assert list(d.keys()) == sorted(set(raw_keys) - set(to_remove))


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
