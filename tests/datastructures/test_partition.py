"""Tests for tralda.datastructures.partition.Partition."""

from __future__ import annotations

import random

import pytest

from tralda.datastructures.partition import Partition, PartitionIterator


# ===========================================================================
# Construction
# ===========================================================================


class TestConstruction:
    def test_empty_partition_has_length_zero(self):
        p = Partition([])
        assert len(p) == 0

    @pytest.mark.parametrize(
        "sets, expected_len",
        [
            ([[1]], 1),
            ([[1, 2, 3]], 1),
            ([[1, 2], [3, 4]], 2),
            ([[1], [2], [3]], 3),
            ([[1, 2, 3], [4, 5], [6]], 3),
        ],
    )
    def test_len_equals_number_of_sets(self, sets, expected_len):
        p = Partition(sets)
        assert len(p) == expected_len

    @pytest.mark.parametrize(
        "sets",
        [
            [[1, 2], [3]],
            [["a", "b"], ["c"]],
            [[(0, 1)], [(2, 3)]],
        ],
    )
    def test_accepts_various_hashable_element_types(self, sets):
        p = Partition(sets)
        assert len(p) == len(sets)


# ===========================================================================
# in_same_set
# ===========================================================================


class TestInSameSet:
    @pytest.mark.parametrize(
        "sets, x, y, expected",
        [
            ([[1, 2, 3], [4, 5], [6]], 1, 2, True),
            ([[1, 2, 3], [4, 5], [6]], 1, 3, True),
            ([[1, 2, 3], [4, 5], [6]], 2, 3, True),
            ([[1, 2, 3], [4, 5], [6]], 4, 5, True),
            ([[1, 2, 3], [4, 5], [6]], 1, 4, False),
            ([[1, 2, 3], [4, 5], [6]], 1, 6, False),
            ([[1, 2, 3], [4, 5], [6]], 4, 6, False),
        ],
    )
    def test_basic_membership(self, sets, x, y, expected):
        p = Partition(sets)
        assert p.in_same_set(x, y) == expected

    def test_reflexive_same_set(self):
        p = Partition([[1, 2], [3]])
        assert p.in_same_set(1, 1) is True

    def test_symmetric(self):
        p = Partition([[1, 2], [3]])
        assert p.in_same_set(1, 2) == p.in_same_set(2, 1)

    def test_string_elements(self):
        p = Partition([["a", "b"], ["c"]])
        assert p.in_same_set("a", "b") is True
        assert p.in_same_set("a", "c") is False

    def test_missing_first_element_raises_key_error(self):
        p = Partition([[1, 2]])
        with pytest.raises(KeyError):
            p.in_same_set(99, 1)

    def test_missing_second_element_raises_key_error(self):
        p = Partition([[1, 2]])
        with pytest.raises(KeyError):
            p.in_same_set(1, 99)

    def test_both_missing_raises_key_error(self):
        p = Partition([[1, 2]])
        with pytest.raises(KeyError):
            p.in_same_set(98, 99)


# ===========================================================================
# separated_xy_z
# ===========================================================================


class TestSeparatedXYZ:
    @pytest.mark.parametrize(
        "sets, x, y, z, expected",
        [
            # x and y together, z separate
            ([[1, 2, 3], [4, 5], [6]], 1, 2, 4, True),
            ([[1, 2, 3], [4, 5], [6]], 1, 3, 6, True),
            ([[1, 2, 3], [4, 5], [6]], 4, 5, 1, True),
            # x and y NOT together
            ([[1, 2, 3], [4, 5], [6]], 1, 4, 6, False),
            # x and y together BUT z is also in the same set
            ([[1, 2, 3], [4, 5], [6]], 1, 2, 3, False),
        ],
    )
    def test_basic_cases(self, sets, x, y, z, expected):
        p = Partition(sets)
        assert p.separated_xy_z(x, y, z) == expected

    def test_reflexive_xy_different_z(self):
        p = Partition([[1, 2], [3]])
        assert p.separated_xy_z(1, 1, 3) is True

    def test_missing_x_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.separated_xy_z(99, 1, 3)

    def test_missing_y_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.separated_xy_z(1, 99, 3)

    def test_missing_z_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.separated_xy_z(1, 2, 99)


# ===========================================================================
# merge
# ===========================================================================


class TestMerge:
    # ── Return value ────────────────────────────────────────────────────────

    def test_merge_returns_smaller_set(self):
        p = Partition([[1, 2], [3, 4, 5]])
        smaller = p.merge(1, 3)
        assert smaller == {1, 2}

    def test_merge_returns_smaller_set_regardless_of_arg_order(self):
        p = Partition([[1, 2], [3, 4, 5]])
        smaller = p.merge(3, 1)  # larger repr first
        assert smaller == {1, 2}

    def test_merge_same_set_returns_empty_set(self):
        p = Partition([[1, 2], [3]])
        result = p.merge(1, 2)
        assert result == set()

    def test_merge_singleton_with_itself_returns_empty_set(self):
        p = Partition([[42]])
        result = p.merge(42, 42)
        assert result == set()

    # ── Structural effect ──────────────────────────────────────────────────

    @pytest.mark.parametrize(
        "sets, repr1, repr2, joined_elements",
        [
            ([[1, 2], [3, 4]], 1, 3, {1, 2, 3, 4}),
            ([[1], [2], [3]], 1, 2, {1, 2}),
            ([[1, 2, 3], [4, 5], [6]], 1, 4, {1, 2, 3, 4, 5}),
        ],
    )
    def test_merged_elements_are_in_same_set(self, sets, repr1, repr2, joined_elements):
        p = Partition(sets)
        p.merge(repr1, repr2)
        elements = list(joined_elements)
        for i in range(len(elements)):
            for j in range(i + 1, len(elements)):
                assert p.in_same_set(elements[i], elements[j])

    @pytest.mark.parametrize(
        "sets, repr1, repr2, before_count, after_count",
        [
            ([[1, 2], [3, 4]], 1, 3, 2, 1),
            ([[1], [2], [3]], 1, 2, 3, 2),
            ([[1, 2, 3], [4, 5], [6]], 1, 4, 3, 2),
        ],
    )
    def test_merge_decrements_set_count(self, sets, repr1, repr2, before_count, after_count):
        p = Partition(sets)
        assert len(p) == before_count
        p.merge(repr1, repr2)
        assert len(p) == after_count

    def test_merge_same_set_does_not_decrement_count(self):
        p = Partition([[1, 2], [3]])
        p.merge(1, 2)
        assert len(p) == 2

    def test_elements_outside_merged_sets_unaffected(self):
        p = Partition([[1, 2], [3, 4], [5]])
        p.merge(1, 3)
        # 5 should still be in its own separate set
        assert not p.in_same_set(1, 5)
        assert not p.in_same_set(3, 5)

    def test_merge_all_sets_into_one(self):
        p = Partition([[1], [2], [3], [4]])
        p.merge(1, 2)
        p.merge(1, 3)
        p.merge(1, 4)
        assert len(p) == 1
        for x in range(1, 5):
            for y in range(1, 5):
                assert p.in_same_set(x, y)
        assert len(p) == 1

    # ── Small-into-large correctness ───────────────────────────────────────

    @pytest.mark.parametrize(
        "large, small",
        [
            ([1, 2, 3, 4, 5], [6]),
            ([1, 2, 3], [4, 5]),
        ],
    )
    def test_small_into_large_direction(self, large, small):
        """The returned set must always be the smaller original set."""
        p = Partition([large, small])
        smaller = p.merge(large[0], small[0])
        assert smaller == set(small)

    def test_equal_size_merge_returns_repr1_set(self):
        """When both sets have the same size, repr1's set is returned (documented tie-break)."""
        p = Partition([[1], [2]])
        smaller = p.merge(1, 2)
        assert smaller == {1}

    # ── Error handling ─────────────────────────────────────────────────────

    def test_missing_repr1_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.merge(99, 1)

    def test_missing_repr2_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.merge(1, 99)

    # ── Return value is a snapshot, not a live view ────────────────────────

    def test_returned_set_is_snapshot_of_smaller_original(self):
        """Test that the set returned by ``merge`` is a snapshot of the smaller original set.

        ``merge`` returns the original smaller set object, which is disconnected from the partition
        after the merge. It is a snapshot of the pre-merge state, not a live view of any active set.
        """
        p = Partition([[1, 2], [3, 4, 5]])
        smaller = p.merge(1, 3)
        assert smaller == {1, 2}
        # The returned set is no longer reachable through the partition
        active_sets = list(p)
        assert not any(s is smaller for s in active_sets)


# ===========================================================================
# Iteration
# ===========================================================================


class TestIteration:
    def test_empty_partition_iterates_zero_sets(self):
        p = Partition([])
        assert list(p) == []

    @pytest.mark.parametrize(
        "sets",
        [
            [[1]],
            [[1, 2], [3]],
            [[1, 2, 3], [4, 5], [6]],
        ],
    )
    def test_iteration_yields_all_sets(self, sets):
        p = Partition(sets)
        result = list(p)
        assert len(result) == len(sets)
        expected = [set(s) for s in sets]
        assert sorted(str(s) for s in result) == sorted(str(s) for s in expected)

    def test_iteration_covers_all_elements(self):
        p = Partition([[1, 2, 3], [4, 5], [6]])
        all_elements = {elem for s in p for elem in s}
        assert all_elements == {1, 2, 3, 4, 5, 6}

    def test_iterated_sets_are_disjoint(self):
        p = Partition([[1, 2], [3, 4], [5]])
        sets = list(p)
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert sets[i].isdisjoint(sets[j])

    def test_iter_returns_partition_iterator(self):
        p = Partition([[1, 2]])
        assert isinstance(iter(p), PartitionIterator)

    def test_multiple_independent_iterators(self):
        p = Partition([[1, 2], [3]])
        it1 = iter(p)
        it2 = iter(p)
        # advance one iterator without affecting the other
        next(it1)
        s2 = next(it2)
        assert isinstance(s2, set)


# ===========================================================================
# Partition.__next__ stub (Bug 1)
# ===========================================================================


class TestPartitionIsNotItsOwnIterator:
    """``Partition`` is an iterable but not an iterator."""

    def test_iter_does_not_return_self(self):
        """``iter(partition)`` must return a PartitionIterator, not the partition."""
        p = Partition([[1, 2]])
        assert iter(p) is not p

    def test_calling_next_on_partition_raises_type_error(self):
        """``next(partition)`` raises ``TypeError`` pointing to ``PartitionIterator``."""
        p = Partition([[1, 2]])
        with pytest.raises(TypeError, match="PartitionIterator"):
            next(p)


# ===========================================================================
# Combined workflow
# ===========================================================================


class TestWorkflow:
    def test_full_merge_workflow(self):
        p = Partition([[1, 2, 3], [4, 5], [6]])
        assert len(p) == 3

        p.merge(1, 4)
        assert len(p) == 2
        assert p.in_same_set(2, 5)
        assert not p.in_same_set(2, 6)

        p.merge(2, 6)
        assert len(p) == 1
        for x in [1, 2, 3, 4, 5, 6]:
            for y in [1, 2, 3, 4, 5, 6]:
                assert p.in_same_set(x, y)

    def test_separated_xy_z_updates_after_merge(self):
        p = Partition([[1, 2], [3], [4]])
        assert p.separated_xy_z(1, 2, 3) is True
        p.merge(2, 3)
        # now 1, 2, 3 are all in the same set
        assert p.separated_xy_z(1, 2, 3) is False

    def test_large_partition_many_merges(self):
        n = 50
        p = Partition([[i] for i in range(n)])
        assert len(p) == n

        # merge pairs: 0-1, 2-3, ..., 48-49
        for i in range(0, n, 2):
            p.merge(i, i + 1)
        assert len(p) == n // 2

        # every even-odd pair must be in the same set
        for i in range(0, n, 2):
            assert p.in_same_set(i, i + 1)
        # elements from different pairs must be in different sets
        assert not p.in_same_set(0, 2)

    @pytest.mark.parametrize(
        "elements", [list(range(10)), list("abcdefghij"), list(range(0, 100, 10))]
    )
    def test_chain_merge_produces_single_set(self, elements):
        """Merging elements one by one from left to right must collapse to one set."""
        p = Partition([[e] for e in elements])
        for i in range(1, len(elements)):
            p.merge(elements[0], elements[i])
        assert len(p) == 1
        for x in elements:
            for y in elements:
                assert p.in_same_set(x, y)


# ===========================================================================
# separated_xy_z — previously uncovered bug path
# ===========================================================================


class TestSeparatedXYZMissingZWhenXYDiffer:
    """Regression tests for the short-circuit bug fixed in separated_xy_z.

    When x and y are in *different* sets the ``and`` short-circuits, so z was
    never looked up.  A missing z must always raise KeyError regardless of
    whether x and y are together or apart.
    """

    def test_missing_z_raises_when_xy_in_different_sets(self):
        p = Partition([[1, 2], [3]])
        # x=1 and y=3 are in DIFFERENT sets — this is the previously buggy path
        with pytest.raises(KeyError):
            p.separated_xy_z(1, 3, 99)

    @pytest.mark.parametrize(
        "sets, x, y, z",
        [
            # x and y together
            ([[1, 2], [3]], 1, 2, 99),
            # x and y apart — the previously uncovered path
            ([[1, 2], [3]], 1, 3, 99),
            ([[1], [2], [3]], 2, 3, 99),
        ],
    )
    def test_missing_z_always_raises_key_error(self, sets, x, y, z):
        p = Partition(sets)
        with pytest.raises(KeyError):
            p.separated_xy_z(x, y, z)

    @pytest.mark.parametrize(
        "sets, x, y, z",
        [
            # x missing, y and z in different sets
            ([[1, 2], [3]], 99, 1, 3),
            # y missing, x and z in different sets
            ([[1, 2], [3]], 1, 99, 3),
        ],
    )
    def test_missing_xy_always_raises_key_error(self, sets, x, y, z):
        p = Partition(sets)
        with pytest.raises(KeyError):
            p.separated_xy_z(x, y, z)

    def test_all_three_missing_raises_key_error(self):
        p = Partition([[1, 2], [3]])
        with pytest.raises(KeyError):
            p.separated_xy_z(97, 98, 99)


# ===========================================================================
# separated_xy_z — all three in the same set
# ===========================================================================


class TestSeparatedXYZAllSameSet:
    """When x, y, and z are all in the same set the result must be False."""

    @pytest.mark.parametrize(
        "sets, x, y, z",
        [
            ([[1, 2, 3]], 1, 2, 3),
            ([[1, 2, 3]], 1, 3, 2),
            ([[1, 2, 3, 4, 5]], 3, 4, 5),
        ],
    )
    def test_all_in_same_set_returns_false(self, sets, x, y, z):
        p = Partition(sets)
        assert p.separated_xy_z(x, y, z) is False

    def test_z_joins_xy_set_after_merge_returns_false(self):
        p = Partition([[1, 2], [3]])
        assert p.separated_xy_z(1, 2, 3) is True
        p.merge(1, 3)
        assert p.separated_xy_z(1, 2, 3) is False


# ===========================================================================
# PartitionIterator — exhaustion and StopIteration
# ===========================================================================


class TestPartitionIteratorExhaustion:
    def test_iterator_raises_stop_iteration_when_exhausted(self):
        p = Partition([[1], [2]])
        it = iter(p)
        next(it)
        next(it)
        with pytest.raises(StopIteration):
            next(it)

    def test_empty_partition_iterator_raises_immediately(self):
        p = Partition([])
        it = iter(p)
        with pytest.raises(StopIteration):
            next(it)

    def test_iterator_iter_returns_self(self):
        p = Partition([[1, 2]])
        it = iter(p)
        assert iter(it) is it

    def test_iterator_yields_correct_number_of_sets(self):
        sets = [[1, 2], [3, 4], [5]]
        p = Partition(sets)
        assert sum(1 for _ in iter(p)) == len(sets)

    def test_iterator_still_valid_after_full_traversal(self):
        """A fresh iterator obtained after a full traversal works correctly."""
        p = Partition([[1], [2], [3]])
        _ = list(p)  # exhaust one implicit iterator
        assert list(p) == list(p)  # two fresh iterators give the same result


# ===========================================================================
# merge — return-value invariants
# ===========================================================================


class TestMergeReturnValueInvariants:
    @pytest.mark.parametrize(
        "sets, repr1, repr2, expected_smaller",
        [
            # clearly smaller set is the first arg's set
            ([[1], [2, 3, 4]], 1, 2, {1}),
            # clearly smaller set is the second arg's set
            ([[1, 2, 3], [4]], 1, 4, {4}),
            # equal size — repr1's set is returned
            ([[1, 2], [3, 4]], 1, 3, {1, 2}),
            ([[1, 2], [3, 4]], 3, 1, {3, 4}),
        ],
    )
    def test_return_value_is_smaller_set(self, sets, repr1, repr2, expected_smaller):
        p = Partition(sets)
        result = p.merge(repr1, repr2)
        assert result == expected_smaller

    def test_returned_set_union_equals_merged_set(self):
        """The merged set must contain all elements; the returned set must be a subset of it."""
        p = Partition([[1, 2], [3, 4, 5]])
        smaller = p.merge(1, 3)
        # After merge there is exactly one set in the partition
        (merged,) = list(p)
        assert merged == {1, 2, 3, 4, 5}
        assert smaller.issubset(merged)

    def test_merge_same_set_returns_empty_set_not_none(self):
        """Idempotent merge must return an empty set, not None or some falsy value."""
        p = Partition([[1, 2, 3]])
        result = p.merge(1, 3)
        assert result == set()
        assert isinstance(result, set)


# ===========================================================================
# Randomised tests
# ===========================================================================


class TestRandomised:
    """Property-style tests driven by random seeds to increase structural coverage."""

    @pytest.mark.parametrize("seed", [0, 42, 123, 999, 7])
    def test_union_find_correctness(self, seed: int) -> None:
        """A reference union-find (dict-based) must agree with Partition on all queries."""
        rng = random.Random(seed)
        n = 40
        elements = list(range(n))

        # Reference: component id per element (path-compressed manually)
        comp: dict[int, int] = {e: e for e in elements}

        def find(x: int) -> int:
            while comp[x] != x:
                comp[x] = comp[comp[x]]
                x = comp[x]
            return x

        p = Partition([[e] for e in elements])

        for _ in range(30):
            a, b = rng.sample(elements, 2)
            p.merge(a, b)
            # Mirror in reference
            ra, rb = find(a), find(b)
            if ra != rb:
                comp[ra] = rb

        # Verify every pair
        for i in range(n):
            for j in range(i + 1, n):
                assert p.in_same_set(i, j) == (find(i) == find(j)), (
                    f"seed={seed}: in_same_set({i},{j}) disagrees with reference"
                )

    @pytest.mark.parametrize("seed", [0, 17, 256])
    def test_len_decrements_exactly_once_per_cross_set_merge(self, seed: int) -> None:
        """Each merge of two *distinct* sets must decrement len by exactly 1."""
        rng = random.Random(seed)
        n = 20
        p = Partition([[i] for i in range(n)])

        for _ in range(25):
            a, b = rng.sample(range(n), 2)
            before = len(p)
            same = p.in_same_set(a, b)
            p.merge(a, b)
            after = len(p)
            if same:
                assert after == before
            else:
                assert after == before - 1

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_iteration_always_covers_all_elements(self, seed: int) -> None:
        """After any sequence of merges, iterating the partition yields every element."""
        rng = random.Random(seed)
        n = 30
        elements = set(range(n))
        p = Partition([[e] for e in elements])

        for _ in range(15):
            a, b = rng.sample(list(elements), 2)
            p.merge(a, b)

        iterated = {elem for s in p for elem in s}
        assert iterated == elements

    @pytest.mark.parametrize("seed", [10, 20, 30])
    def test_iterated_sets_partition_elements(self, seed: int) -> None:
        """Sets yielded by the iterator must be pairwise disjoint and cover all elements."""
        rng = random.Random(seed)
        n = 25
        elements = set(range(n))
        p = Partition([[e] for e in elements])

        for _ in range(10):
            a, b = rng.sample(list(elements), 2)
            p.merge(a, b)

        sets = list(p)
        # pairwise disjoint
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                assert sets[i].isdisjoint(sets[j])
        # union equals original elements
        assert set().union(*sets) == elements

    @pytest.mark.parametrize("seed", [5, 50, 500])
    def test_separated_xy_z_consistent_with_in_same_set(self, seed: int) -> None:
        """separated_xy_z(x,y,z) must equal in_same_set(x,y) and not in_same_set(x,z)."""
        rng = random.Random(seed)
        n = 20
        elements = list(range(n))
        p = Partition([[e] for e in elements])

        for _ in range(12):
            a, b = rng.sample(elements, 2)
            p.merge(a, b)

        for _ in range(50):
            x, y, z = rng.sample(elements, 3)
            expected = p.in_same_set(x, y) and not p.in_same_set(x, z)
            assert p.separated_xy_z(x, y, z) == expected

    @pytest.mark.parametrize("seed", [11, 22, 33])
    def test_merge_smaller_set_returned_content(self, seed: int) -> None:
        """The set returned by merge must equal the smaller of the two pre-merge sets."""
        rng = random.Random(seed)
        n = 20
        elements = list(range(n))
        p = Partition([[e] for e in elements])

        # Record which elements are in each component using the reference structure
        comp: dict[int, set[int]] = {e: {e} for e in elements}

        def find_root(x: int) -> int:
            for root, members in comp.items():
                if x in members:
                    return root
            raise KeyError(x)

        for _ in range(15):
            a, b = rng.sample(elements, 2)
            root_a = find_root(a)
            root_b = find_root(b)

            if root_a == root_b:
                result = p.merge(a, b)
                assert result == set()
                continue

            set_a = comp.pop(root_a)
            set_b = comp.pop(root_b)
            expected_smaller = set_a if len(set_a) <= len(set_b) else set_b
            merged = set_a | set_b
            new_root = next(iter(merged))
            comp[new_root] = merged

            result = p.merge(a, b)
            assert result == expected_smaller
