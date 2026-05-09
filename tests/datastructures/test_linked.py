"""Tests for tralda.datastructures.linked (LinkedList and LinkedListNode)."""

from __future__ import annotations

import pytest

from tralda.datastructures.linked import LinkedList, LinkedListNode


# ===========================================================================
# LinkedListNode
# ===========================================================================


class TestLinkedListNode:
    """Unit tests for LinkedListNode."""

    @pytest.mark.parametrize("value", [0, 42, -1, "hello", 3.14, None, [], {"a": 1}])
    def test_get_returns_stored_value(self, value):
        node = LinkedListNode(value)
        assert node.get() == value

    def test_next_node_link(self):
        second = LinkedListNode("b")
        first = LinkedListNode("a", next_node=second)
        assert first._next is second

    def test_default_next_is_none(self):
        assert LinkedListNode("x")._next is None


# ===========================================================================
# LinkedList – construction
# ===========================================================================


class TestLinkedListConstruction:
    """Tests for __init__, __len__, and bool conversion."""

    def test_default_empty(self):
        ll = LinkedList()
        assert len(ll) == 0
        assert not bool(ll)

    @pytest.mark.parametrize(
        "args, expected",
        [
            ((1,), [1]),
            (("abc",), ["a", "b", "c"]),  # strings are Iterable → iterated char-by-char
            (([1, 2, 3],), [1, 2, 3]),
            (([10, 20], [30, 40]), [10, 20, 30, 40]),
            ((0, [1, 2]), [0, 1, 2]),
            ((range(5),), [0, 1, 2, 3, 4]),
        ],
    )
    def test_construction_from_args(self, args, expected):
        ll = LinkedList(*args)
        assert list(ll) == expected
        assert len(ll) == len(expected)

    @pytest.mark.parametrize("values", [[1], [1, 2], [1, 2, 3, 4, 5]])
    def test_bool_nonempty(self, values):
        assert bool(LinkedList(values))

    def test_bool_empty_false(self):
        assert not bool(LinkedList())


# ===========================================================================
# LinkedList – element access
# ===========================================================================


class TestLinkedListAccess:
    """Tests for first(), last(), first_node(), node_at(), and __getitem__."""

    @pytest.fixture
    def ll5(self) -> LinkedList:
        """A five-element list: [10, 20, 30, 40, 50]."""
        return LinkedList([10, 20, 30, 40, 50])

    @pytest.mark.parametrize(
        "index, expected",
        [(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)],
    )
    def test_getitem_positive(self, ll5, index, expected):
        assert ll5[index] == expected

    @pytest.mark.parametrize(
        "index, expected",
        [(-1, 50), (-2, 40), (-3, 30), (-4, 20), (-5, 10)],
    )
    def test_getitem_negative(self, ll5, index, expected):
        assert ll5[index] == expected

    @pytest.mark.parametrize("index", [5, 6, 100, -6, -7])
    def test_getitem_out_of_bounds_raises(self, ll5, index):
        with pytest.raises(IndexError):
            _ = ll5[index]

    @pytest.mark.parametrize("bad_index", ["0", 1.0, None])
    def test_node_at_wrong_type_raises(self, ll5, bad_index):
        with pytest.raises(TypeError):
            ll5.node_at(bad_index)

    def test_first_and_last(self, ll5):
        assert ll5.first() == 10
        assert ll5.last() == 50

    def test_first_node_is_linked_list_node(self, ll5):
        node = ll5.first_node()
        assert isinstance(node, LinkedListNode)
        assert node._value == 10

    @pytest.mark.parametrize(
        "index, expected",
        [(0, 10), (2, 30), (4, 50)],
    )
    def test_node_at_value(self, ll5, index, expected):
        assert ll5.node_at(index)._value == expected


# ===========================================================================
# LinkedList – iteration
# ===========================================================================


class TestLinkedListIteration:
    """Tests for __iter__ and LinkedListIterator."""

    @pytest.mark.parametrize(
        "values",
        [[], [1], [1, 2], [1, 2, 3, 4, 5], ["a", "b", "c"]],
    )
    def test_iterate_roundtrip(self, values):
        assert list(LinkedList(values)) == values

    def test_multiple_independent_iterators(self):
        ll = LinkedList([1, 2, 3])
        it1, it2 = iter(ll), iter(ll)
        assert next(it1) == 1
        assert next(it2) == 1
        assert next(it1) == 2

    def test_stop_iteration(self):
        it = iter(LinkedList([1]))
        next(it)
        with pytest.raises(StopIteration):
            next(it)


# ===========================================================================
# LinkedList – append / extend / append_left
# ===========================================================================


class TestLinkedListMutation:
    """Tests for append(), extend(), and append_left()."""

    @pytest.mark.parametrize("value", [0, "x", None, 3.14])
    def test_append_returns_node_with_correct_value(self, value):
        ll = LinkedList()
        node = ll.append(value)
        assert isinstance(node, LinkedListNode)
        assert node._value == value
        assert ll.last() == value

    @pytest.mark.parametrize(
        "initial, appended, expected",
        [
            ([], [1, 2, 3], [1, 2, 3]),
            ([1], [2, 3], [1, 2, 3]),
            ([1, 2], [3], [1, 2, 3]),
        ],
    )
    def test_extend_builds_correct_list(self, initial, appended, expected):
        ll = LinkedList(initial)
        ll.extend(appended)
        assert list(ll) == expected

    @pytest.mark.parametrize(
        "initial, prepended, expected",
        [
            ([2, 3], 1, [1, 2, 3]),
            ([1], 0, [0, 1]),
            ([], 5, [5]),
        ],
    )
    def test_append_left(self, initial, prepended, expected):
        ll = LinkedList(initial)
        node = ll.append_left(prepended)
        assert isinstance(node, LinkedListNode)
        assert ll.first() == prepended
        assert list(ll) == expected


# ===========================================================================
# LinkedList – concatenate
# ===========================================================================


class TestLinkedListConcatenate:
    """Tests for concatenate()."""

    @pytest.mark.parametrize(
        "left, right, expected",
        [
            ([1, 2], [3, 4], [1, 2, 3, 4]),
            ([], [1, 2], [1, 2]),
            ([1, 2], [], [1, 2]),
            ([], [], []),
            ([1], [2], [1, 2]),
            (list(range(5)), list(range(5, 10)), list(range(10))),
        ],
    )
    def test_concatenate(self, left, right, expected):
        a, b = LinkedList(left), LinkedList(right)
        result = a.concatenate(b)
        assert result is a
        assert list(a) == expected
        assert len(a) == len(expected)

    def test_concatenate_wrong_type_raises(self):
        with pytest.raises(TypeError):
            LinkedList([1]).concatenate([2, 3])

    def test_concatenate_last_pointer_updated(self):
        a = LinkedList([1, 2])
        b = LinkedList([3, 4])
        a.concatenate(b)
        assert a.last() == 4


# ===========================================================================
# LinkedList – insert_right_of
# ===========================================================================


class TestLinkedListInsertRightOf:
    """Tests for insert_right_of()."""

    @pytest.mark.parametrize(
        "initial, insert_at_index, value, expected",
        [
            ([1, 3], 0, 2, [1, 2, 3]),
            ([1, 2, 4], 1, 3, [1, 2, 3, 4]),
            ([1], 0, 2, [1, 2]),  # insert after last → goes via append path
            ([1, 2, 3], 2, 4, [1, 2, 3, 4]),  # insert after current last
        ],
    )
    def test_insert_right_of(self, initial, insert_at_index, value, expected):
        ll = LinkedList(initial)
        node = ll.node_at(insert_at_index)
        new_node = ll.insert_right_of(node, value)
        assert isinstance(new_node, LinkedListNode)
        assert new_node._value == value
        assert list(ll) == expected
        assert len(ll) == len(expected)


# ===========================================================================
# LinkedList – remove
# ===========================================================================


class TestLinkedListRemove:
    """Tests for remove()."""

    @pytest.mark.parametrize(
        "values, remove_value, expected",
        [
            ([1, 2, 3], 1, [2, 3]),  # remove first
            ([1, 2, 3], 2, [1, 3]),  # remove middle
            ([1, 2, 3], 3, [1, 2]),  # remove last
            ([42], 42, []),  # remove only element
            ([1, 2, 1], 1, [2, 1]),  # removes first occurrence only
        ],
    )
    def test_remove(self, values, remove_value, expected):
        ll = LinkedList(values)
        ll.remove(remove_value)
        assert list(ll) == expected
        assert len(ll) == len(expected)

    def test_remove_last_updates_last_pointer(self):
        ll = LinkedList([1, 2, 3])
        ll.remove(3)
        assert ll.last() == 2

    def test_remove_first_updates_first_pointer(self):
        ll = LinkedList([1, 2, 3])
        ll.remove(1)
        assert ll.first() == 2

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError):
            LinkedList([1, 2]).remove(99)

    def test_remove_from_empty_raises(self):
        with pytest.raises(KeyError):
            LinkedList().remove(1)


# ===========================================================================
# LinkedList – truncate / truncate_left
# ===========================================================================


class TestLinkedListTruncate:
    """Tests for truncate() and truncate_left()."""

    @pytest.mark.parametrize(
        "values, index, expected",
        [
            ([1, 2, 3, 4], 2, [1, 2]),
            ([1, 2, 3, 4], 1, [1]),
            ([1, 2, 3, 4], 3, [1, 2, 3]),
            ([1, 2], 0, []),
            ([1, 2], -1, []),
        ],
    )
    def test_truncate(self, values, index, expected):
        ll = LinkedList(values)
        ll.truncate(index)
        assert list(ll) == expected
        assert len(ll) == len(expected)

    @pytest.mark.parametrize(
        "values, n, expected",
        [
            ([1, 2, 3, 4], 1, [2, 3, 4]),
            ([1, 2, 3, 4], 2, [3, 4]),
            ([1, 2, 3, 4], 3, [4]),
            ([1, 2], 2, []),
            ([1, 2], 5, []),
        ],
    )
    def test_truncate_left(self, values, n, expected):
        ll = LinkedList(values)
        ll.truncate_left(n)
        assert list(ll) == expected
        assert len(ll) == len(expected)

    def test_truncate_updates_last_pointer(self):
        ll = LinkedList([1, 2, 3, 4])
        ll.truncate(2)
        assert ll.last() == 2

    def test_truncate_left_updates_first_pointer(self):
        ll = LinkedList([1, 2, 3, 4])
        ll.truncate_left(2)
        assert ll.first() == 3


# ===========================================================================
# LinkedList – popleft
# ===========================================================================


class TestLinkedListPopleft:
    """Tests for popleft()."""

    @pytest.mark.parametrize(
        "values, pop_count, expected_popped, expected_remaining",
        [
            ([1, 2, 3], 1, [1], [2, 3]),
            ([1, 2, 3], 2, [1, 2], [3]),
            ([1, 2, 3], 3, [1, 2, 3], []),
            ([99], 1, [99], []),
            ([1, 2], 1, [1], [2]),
        ],
    )
    def test_popleft_sequence(self, values, pop_count, expected_popped, expected_remaining):
        ll = LinkedList(values)
        popped = [ll.popleft() for _ in range(pop_count)]
        assert popped == expected_popped
        assert list(ll) == expected_remaining
        assert len(ll) == len(expected_remaining)

    def test_popleft_single_element_clears_both_pointers(self):
        ll = LinkedList([99])
        ll.popleft()
        assert ll._first is None
        assert ll._last is None

    def test_popleft_two_elements_does_not_corrupt_chain(self):
        ll = LinkedList([1, 2])
        ll.popleft()
        assert list(ll) == [2]
        assert ll._last._value == 2

    def test_popleft_empty_raises(self):
        with pytest.raises(IndexError):
            LinkedList().popleft()


# ===========================================================================
# LinkedList – clear and _count_actual
# ===========================================================================


class TestLinkedListClear:
    """Tests for clear() and the _count_actual debugging helper."""

    def test_clear_resets_all_state(self):
        ll = LinkedList([1, 2, 3])
        ll.clear()
        assert len(ll) == 0
        assert ll._first is None
        assert ll._last is None

    @pytest.mark.parametrize(
        "values",
        [[], [1], [1, 2], list(range(20))],
    )
    def test_count_actual_matches_len(self, values):
        ll = LinkedList(values)
        assert ll._count_actual() == len(ll)
