"""Tests for tralda.datastructures.bst.red_black.TreeSet."""

from __future__ import annotations

import pytest
import numpy as np

from tralda.datastructures.bst.red_black import TreeSet


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


def build_disjoint_pair(seed: int) -> tuple[TreeSet, TreeSet, list, list]:
    """Build two disjoint sorted trees from a seeded RNG."""
    rng = np.random.default_rng(seed)
    all_keys = rng.permutation(1000).tolist()
    mid = len(all_keys) // 2
    keys_left = sorted(all_keys[:mid])
    keys_right = sorted(all_keys[mid:])
    # make sure they are strictly disjoint and ordered
    max_left = max(keys_left)
    keys_right = [k for k in keys_right if k > max_left]
    rng.shuffle(keys_left)
    rng.shuffle(keys_right)
    return build_tree(keys_left), build_tree(keys_right), keys_left, keys_right


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
# Red-black property
# ---------------------------------------------------------------------------


class TestRedBlackProperty:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_rb_property_after_insert(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        t = build_tree(keys)
        assert t.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_rb_property_after_removal(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(500).tolist()
        t = build_tree(keys)
        for k in keys[:100]:
            t.remove(k)
        assert t.check_integrity(verbose=True)

    def test_root_is_black(self):
        rng = np.random.default_rng(0)
        keys = rng.permutation(100).tolist()
        t = build_tree(keys)
        assert t.root.is_black

    def test_no_adjacent_red_nodes_after_insert(self):
        rng = np.random.default_rng(0)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        for node in t._postorder_traversal():
            if node.is_red and node.parent:
                assert node.parent.is_black, f"Red node {node} has red parent {node.parent}"

    def test_uniform_black_height(self):
        rng = np.random.default_rng(0)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        # check_integrity already validates this; here we also verify directly
        for node in t._postorder_traversal():
            bh_left = node.left.black_height if node.left else 1
            bh_right = node.right.black_height if node.right else 1
            assert bh_left == bh_right, (
                f"Node {node} has unequal black heights: left={bh_left}, right={bh_right}"
            )

    @pytest.mark.parametrize(
        "keys",
        [
            list(range(100)),  # ascending
            list(range(99, -1, -1)),  # descending
        ],
    )
    def test_rb_property_sequential_insertion(self, keys):
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
        assert sorted_keys(TreeSet()) == []

    def test_iter_does_not_return_self(self):
        t = build_tree([1, 2, 3])
        assert iter(t) is not t

    def test_next_on_tree_raises_type_error(self):
        t = build_tree([1])
        with pytest.raises(TypeError):
            next(t)


# ---------------------------------------------------------------------------
# Membership / removal
# ---------------------------------------------------------------------------


class TestMembership:
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

    def test_remove_raises_key_error_for_absent_key(self):
        t = build_tree([1, 2, 3])
        with pytest.raises(KeyError):
            t.remove(9999)

    def test_discard_absent_key_is_silent(self):
        t = build_tree([1, 2, 3])
        t.discard(9999)
        assert len(t) == 3

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
# Index-based access / pop
# ---------------------------------------------------------------------------


class TestIndexAccess:
    @pytest.mark.parametrize("idx", [0, 1, 5, -1, -5, -10])
    def test_key_at_index(self, idx):
        keys = list(range(10))
        t = build_tree(keys)
        assert t.key_at_index(idx) == sorted(keys)[idx]

    @pytest.mark.parametrize("idx", [10, 100, -11])
    def test_key_at_index_out_of_bounds_raises_index_error(self, idx):
        t = build_tree(range(10))
        with pytest.raises(IndexError):
            t.key_at_index(idx)

    def test_key_at_index_empty_tree_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().key_at_index(0)

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_pop_at_index_returns_correct_key(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        sorted_ref = sorted(keys)
        for expected in sorted_ref:
            assert t.pop_at_index(0) == expected

    @pytest.mark.parametrize("seed", [0, 42])
    def test_pop_at_index_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        for _ in range(50):
            t.pop_at_index(len(t) // 2)
        assert t.check_integrity(verbose=True)

    def test_pop_at_index_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().pop_at_index(0)

    def test_pop_returns_greatest(self):
        t = build_tree(range(10))
        assert t.pop() == 9

    def test_pop_empty_raises_index_error(self):
        with pytest.raises(IndexError):
            TreeSet().pop()


# ---------------------------------------------------------------------------
# copy
# ---------------------------------------------------------------------------


class TestCopy:
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
# join
# ---------------------------------------------------------------------------


class TestJoin:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_join_contains_all_keys(self, seed):
        t_left, t_right, keys_left, keys_right = build_disjoint_pair(seed)
        middle = max(keys_left) + 1  # guaranteed to be between the two sets
        # exclude middle from both sets
        if middle in t_right:
            t_right.remove(middle)
            keys_right = [k for k in keys_right if k != middle]
        joined = TreeSet.join(t_left, t_right, key=middle)
        expected = sorted(set(keys_left) | set(keys_right) | {middle})
        assert sorted_keys(joined) == expected

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_join_check_integrity(self, seed):
        t_left, t_right, keys_left, _ = build_disjoint_pair(seed)
        middle = max(keys_left) + 1
        if middle in t_right:
            t_right.remove(middle)
        joined = TreeSet.join(t_left, t_right, key=middle)
        assert joined.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_join_without_key(self, seed):
        """join with key=None inserts and removes a dummy node."""
        t_left, t_right, keys_left, keys_right = build_disjoint_pair(seed)
        all_keys = sorted(set(keys_left) | set(keys_right))
        joined = TreeSet.join(t_left, t_right)
        assert sorted_keys(joined) == all_keys
        assert joined.check_integrity(verbose=True)

    def test_join_left_empty(self):
        t_left = TreeSet()
        t_right = build_tree(range(10, 20))
        joined = TreeSet.join(t_left, t_right, key=5)
        assert sorted_keys(joined) == [5] + list(range(10, 20))
        assert joined.check_integrity(verbose=True)

    def test_join_right_empty(self):
        t_left = build_tree(range(10))
        t_right = TreeSet()
        joined = TreeSet.join(t_left, t_right, key=15)
        assert sorted_keys(joined) == list(range(10)) + [15]
        assert joined.check_integrity(verbose=True)

    def test_join_both_empty(self):
        joined = TreeSet.join(TreeSet(), TreeSet(), key=42)
        assert sorted_keys(joined) == [42]
        assert joined.check_integrity(verbose=True)

    def test_join_wrong_type_raises_type_error(self):
        from tralda.datastructures.bst.avl import TreeSet as AVLTreeSet

        avl = AVLTreeSet()
        rb = build_tree([1])
        with pytest.raises(TypeError):
            TreeSet.join(avl, rb, key=0)  # type: ignore

    @pytest.mark.parametrize("seed", [0, 42])
    def test_join_sorted_order(self, seed):
        t_left, t_right, keys_left, keys_right = build_disjoint_pair(seed)
        middle = max(keys_left) + 1
        if middle in t_right:
            t_right.remove(middle)
            keys_right = [k for k in keys_right if k != middle]
        joined = TreeSet.join(t_left, t_right, key=middle)
        result = sorted_keys(joined)
        assert result == sorted(result), "join result is not in sorted order"


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------


class TestSplit:
    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_split_key_count(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_node = t._node_at_index(len(t) // 2)
        split_key = split_node.key
        left, right = t.split_at_node(split_node)
        # default: split_key is in neither
        assert split_key not in sorted_keys(left)
        assert split_key not in sorted_keys(right)
        assert len(left) + len(right) == len(keys) - 1

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_split_all_keys_preserved(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_idx = len(t) // 2
        split_node = t._node_at_index(split_idx)
        split_key = split_node.key
        left, right = t.split_at_node(split_node)
        recovered = set(sorted_keys(left)) | set(sorted_keys(right))
        assert recovered == set(keys) - {split_key}

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_split_left_keys_smaller(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_node = t._node_at_index(len(t) // 2)
        split_key = split_node.key
        left, right = t.split_at_node(split_node)
        for k in sorted_keys(left):
            assert k < split_key
        for k in sorted_keys(right):
            assert k > split_key

    @pytest.mark.parametrize("seed", [0, 1, 42, 123, 999])
    def test_split_check_integrity(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_node = t._node_at_index(len(t) // 2)
        left, right = t.split_at_node(split_node)
        assert left.check_integrity(verbose=True)
        assert right.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_split_keep_node_left(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_node = t._node_at_index(len(t) // 2)
        split_key = split_node.key
        left, right = t.split_at_node(split_node, keep_node_left=True)
        assert split_key in sorted_keys(left)
        assert split_key not in sorted_keys(right)
        assert len(left) + len(right) == len(keys)
        assert left.check_integrity(verbose=True)
        assert right.check_integrity(verbose=True)

    @pytest.mark.parametrize("seed", [0, 42, 999])
    def test_split_keep_node_right(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(200).tolist()
        t = build_tree(keys)
        split_node = t._node_at_index(len(t) // 2)
        split_key = split_node.key
        left, right = t.split_at_node(split_node, keep_node_right=True)
        assert split_key not in sorted_keys(left)
        assert split_key in sorted_keys(right)
        assert len(left) + len(right) == len(keys)
        assert left.check_integrity(verbose=True)
        assert right.check_integrity(verbose=True)

    def test_split_keep_both_raises_value_error(self):
        t = build_tree(range(10))
        node = t._node_at_index(5)
        with pytest.raises(ValueError):
            t.split_at_node(node, keep_node_left=True, keep_node_right=True)

    @pytest.mark.parametrize("seed", [0, 42])
    def test_split_and_rejoin(self, seed):
        """Splitting then joining should recover all keys."""
        rng = np.random.default_rng(seed)
        keys = rng.permutation(100).tolist()
        t = build_tree(keys)
        split_idx = len(t) // 2
        split_node = t._node_at_index(split_idx)
        split_key = split_node.key
        left, right = t.split_at_node(split_node)
        joined = TreeSet.join(left, right, key=split_key)
        assert sorted_keys(joined) == sorted(keys)
        assert joined.check_integrity(verbose=True)

    @pytest.mark.parametrize("split_idx", [0, 1, -1, -2])
    def test_split_at_boundary(self, split_idx):
        """Split at the first or last element."""
        keys = list(range(20))
        t = build_tree(keys)
        split_node = t._node_at_index(split_idx)
        left, right = t.split_at_node(split_node)
        assert left.check_integrity(verbose=True)
        assert right.check_integrity(verbose=True)


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

    @pytest.mark.parametrize("seed", [0, 42])
    def test_many_pop_at_index(self, seed):
        rng = np.random.default_rng(seed)
        keys = rng.permutation(2000).tolist()
        t = build_tree(keys)
        for _ in range(500):
            idx = rng.integers(0, len(t))
            t.pop_at_index(int(idx))
        assert t.check_integrity(verbose=True)
