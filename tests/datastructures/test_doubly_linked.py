"""Tests for tralda.datastructures.doubly_linked (DLList and DLListNode)."""

from __future__ import annotations

import pytest

from tralda.datastructures.doubly_linked import DLList, DLListNode


# ===========================================================================
# DLListNode
# ===========================================================================


class TestDLListNode:
    """Unit tests for DLListNode."""

    @pytest.mark.parametrize("value", [0, 42, -1, "hello", 3.14, None, [], {"a": 1}])
    def test_get_returns_stored_value(self, value):
        node = DLListNode(value)
        assert node.get() == value

    def test_prev_and_next_links(self):
        a = DLListNode("a")
        b = DLListNode("b", prev_node=a)
        a._next = b
        assert b._prev is a
        assert a._next is b

    def test_default_links_are_none(self):
        node = DLListNode("x")
        assert node._prev is None
        assert node._next is None


# ===========================================================================
# DLList – construction
# ===========================================================================


class TestDLListConstruction:
    """Tests for __init__, __len__, and bool conversion."""

    def test_default_empty(self):
        dl = DLList()
        assert len(dl) == 0
        assert not bool(dl)

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
        dl = DLList(*args)
        assert list(dl) == expected
        assert len(dl) == len(expected)

    @pytest.mark.parametrize("values", [[1], [1, 2], [1, 2, 3, 4, 5]])
    def test_bool_nonempty(self, values):
        assert bool(DLList(values))

    def test_bool_empty_false(self):
        assert not bool(DLList())


# ===========================================================================
# DLList – element access
# ===========================================================================


class TestDLListAccess:
    """Tests for first(), last(), first_node(), node_at(), and __getitem__."""

    @pytest.fixture
    def dl5(self) -> DLList:
        """A five-element list: [10, 20, 30, 40, 50]."""
        return DLList([10, 20, 30, 40, 50])

    @pytest.mark.parametrize(
        "index, expected",
        [(0, 10), (1, 20), (2, 30), (3, 40), (4, 50)],
    )
    def test_getitem_positive(self, dl5, index, expected):
        assert dl5[index] == expected

    @pytest.mark.parametrize(
        "index, expected",
        [(-1, 50), (-2, 40), (-3, 30), (-4, 20), (-5, 10)],
    )
    def test_getitem_negative(self, dl5, index, expected):
        assert dl5[index] == expected

    @pytest.mark.parametrize("index", [5, 6, 100, -6, -7])
    def test_getitem_out_of_bounds_raises(self, dl5, index):
        with pytest.raises(IndexError):
            _ = dl5[index]

    @pytest.mark.parametrize("bad_index", ["0", 1.0, None])
    def test_node_at_wrong_type_raises(self, dl5, bad_index):
        with pytest.raises(TypeError):
            dl5.node_at(bad_index)

    def test_first_node_empty_raises(self):
        with pytest.raises(IndexError):
            DLList().first_node()

    def test_first_and_last(self, dl5):
        assert dl5.first() == 10
        assert dl5.last() == 50

    def test_first_node_is_dllistnode(self, dl5):
        node = dl5.first_node()
        assert isinstance(node, DLListNode)
        assert node._value == 10

    @pytest.mark.parametrize(
        "index, expected",
        [
            (0, 10),  # first (forward path)
            (1, 20),  # forward path
            (2, 30),  # midpoint
            (3, 40),  # backward path
            (4, 50),  # last (backward path)
        ],
    )
    def test_node_at_traversal_paths(self, dl5, index, expected):
        """node_at traverses from front or back depending on index position."""
        assert dl5.node_at(index)._value == expected

    def test_node_at_backward_links_intact(self, dl5):
        """Verify the backward (prev) chain is correctly set up."""
        for i in range(1, 5):
            node = dl5.node_at(i)
            assert node._prev is dl5.node_at(i - 1)


# ===========================================================================
# DLList – iteration
# ===========================================================================


class TestDLListIteration:
    """Tests for __iter__ and DLListIterator."""

    @pytest.mark.parametrize(
        "values",
        [[], [1], [1, 2], [1, 2, 3, 4, 5], ["a", "b", "c"]],
    )
    def test_iterate_roundtrip(self, values):
        assert list(DLList(values)) == values

    def test_multiple_independent_iterators(self):
        dl = DLList([1, 2, 3])
        it1, it2 = iter(dl), iter(dl)
        assert next(it1) == 1
        assert next(it2) == 1
        assert next(it1) == 2

    def test_stop_iteration(self):
        it = iter(DLList([1]))
        next(it)
        with pytest.raises(StopIteration):
            next(it)


# ===========================================================================
# DLList – append / extend / append_left
# ===========================================================================


class TestDLListMutation:
    """Tests for append(), extend(), and append_left()."""

    @pytest.mark.parametrize("value", [0, "x", None, 3.14])
    def test_append_returns_dllistnode_with_correct_value(self, value):
        dl = DLList()
        node = dl.append(value)
        assert isinstance(node, DLListNode)
        assert node._value == value
        assert dl.last() == value

    @pytest.mark.parametrize(
        "initial, appended, expected",
        [
            ([], [1, 2, 3], [1, 2, 3]),
            ([1], [2, 3], [1, 2, 3]),
            ([1, 2], [3, 4, 5], [1, 2, 3, 4, 5]),
        ],
    )
    def test_extend_builds_correct_list(self, initial, appended, expected):
        dl = DLList(initial)
        dl.extend(appended)
        assert list(dl) == expected

    @pytest.mark.parametrize(
        "initial, prepended, expected",
        [
            ([2, 3], 1, [1, 2, 3]),
            ([1], 0, [0, 1]),
            ([], 5, [5]),
        ],
    )
    def test_append_left(self, initial, prepended, expected):
        dl = DLList(initial)
        node = dl.append_left(prepended)
        assert isinstance(node, DLListNode)
        assert dl.first() == prepended
        assert list(dl) == expected

    def test_append_backward_link(self):
        dl = DLList([1])
        node = dl.append(2)
        assert node._prev._value == 1

    def test_append_left_forward_link(self):
        dl = DLList([2])
        dl.append_left(1)
        head = dl.first_node()
        assert head._next._value == 2
        assert head._next._prev is head


# ===========================================================================
# DLList – remove_node and remove
# ===========================================================================


class TestDLListRemove:
    """Tests for remove_node() and remove()."""

    @pytest.mark.parametrize(
        "values, remove_index, expected",
        [
            ([1, 2, 3], 0, [2, 3]),  # remove first
            ([1, 2, 3], 1, [1, 3]),  # remove middle
            ([1, 2, 3], 2, [1, 2]),  # remove last
            ([42], 0, []),  # remove only element
            ([1, 2, 3, 4, 5], 2, [1, 2, 4, 5]),  # remove middle of longer list
        ],
    )
    def test_remove_node_by_index(self, values, remove_index, expected):
        dl = DLList(values)
        node = dl.node_at(remove_index)
        dl.remove_node(node)
        assert list(dl) == expected
        assert len(dl) == len(expected)
        # Removed node's pointers should be cleared
        assert node._prev is None
        assert node._next is None

    @pytest.mark.parametrize(
        "values, remove_value, expected",
        [
            ([1, 2, 3], 1, [2, 3]),
            ([1, 2, 3], 2, [1, 3]),
            ([1, 2, 3], 3, [1, 2]),
            ([42], 42, []),
            ([1, 2, 1, 3], 1, [2, 1, 3]),  # removes first occurrence only
        ],
    )
    def test_remove_by_value(self, values, remove_value, expected):
        dl = DLList(values)
        dl.remove(remove_value)
        assert list(dl) == expected

    def test_remove_updates_first_pointer(self):
        dl = DLList([1, 2, 3])
        dl.remove(1)
        assert dl.first() == 2

    def test_remove_updates_last_pointer(self):
        dl = DLList([1, 2, 3])
        dl.remove(3)
        assert dl.last() == 2

    def test_remove_only_element_clears_pointers(self):
        dl = DLList([42])
        dl.remove_node(dl.first_node())
        assert dl._first is None
        assert dl._last is None

    def test_remove_missing_raises(self):
        with pytest.raises(KeyError):
            DLList([1, 2]).remove(99)

    def test_remove_from_empty_raises(self):
        with pytest.raises(KeyError):
            DLList().remove(1)


# ===========================================================================
# DLList – remove_range
# ===========================================================================


class TestDLListRemoveRange:
    """Tests for remove_range()."""

    @pytest.mark.parametrize(
        "values, index, length, expected",
        [
            ([1, 2, 3, 4, 5], 1, 2, [1, 4, 5]),  # remove [2,3] from middle
            ([1, 2, 3, 4, 5], 0, 2, [3, 4, 5]),  # remove from start (delegate to truncate_left)
            ([1, 2, 3, 4, 5], 2, 2, [1, 2, 5]),  # remove two in middle
            ([1, 2, 3, 4, 5], 3, 1, [1, 2, 3, 5]),  # remove single element from middle
            ([1, 2, 3], 1, 10, [1]),  # length exceeds bounds → truncate
            ([1, 2, 3], 0, 3, []),  # remove all → clears
        ],
    )
    def test_remove_range(self, values, index, length, expected):
        dl = DLList(values)
        dl.remove_range(index, length)
        assert list(dl) == expected
        assert len(dl) == len(expected)

    @pytest.mark.parametrize(
        "values, index, expected",
        [
            ([1, 2, 3, 4], 2, [1, 2]),  # no length → truncate from index
            ([1, 2, 3], 0, []),  # no length, from start → clears
        ],
    )
    def test_remove_range_no_length(self, values, index, expected):
        dl = DLList(values)
        dl.remove_range(index)
        assert list(dl) == expected

    @pytest.mark.parametrize("bad_index", ["1", 1.5, None])
    def test_remove_range_invalid_index_type_raises(self, bad_index):
        with pytest.raises(TypeError):
            DLList([1, 2, 3]).remove_range(bad_index)

    @pytest.mark.parametrize("bad_length", [0, -1, "2", 1.0])
    def test_remove_range_invalid_length_raises(self, bad_length):
        with pytest.raises(TypeError):
            DLList([1, 2, 3]).remove_range(1, bad_length)

    def test_remove_range_preserves_bidirectional_links(self):
        dl = DLList([1, 2, 3, 4, 5])
        dl.remove_range(1, 2)  # remove [2, 3] → [1, 4, 5]
        node1 = dl.node_at(0)
        node4 = dl.node_at(1)
        node5 = dl.node_at(2)
        assert node1._next is node4
        assert node4._prev is node1
        assert node4._next is node5
        assert node5._prev is node4


# ===========================================================================
# DLList – insert_right_of
# ===========================================================================


class TestDLListInsertRightOf:
    """Tests for insert_right_of()."""

    @pytest.mark.parametrize(
        "initial, insert_at_index, value, expected",
        [
            ([1, 3], 0, 2, [1, 2, 3]),
            ([1, 2, 4], 1, 3, [1, 2, 3, 4]),
            ([1], 0, 2, [1, 2]),  # insert after last → append path
            ([1, 2, 3], 2, 4, [1, 2, 3, 4]),  # insert after current last
            ([1, 2, 3, 5], 2, 4, [1, 2, 3, 4, 5]),
        ],
    )
    def test_insert_right_of(self, initial, insert_at_index, value, expected):
        dl = DLList(initial)
        node = dl.node_at(insert_at_index)
        new_node = dl.insert_right_of(node, value)
        assert isinstance(new_node, DLListNode)
        assert new_node._value == value
        assert list(dl) == expected
        assert len(dl) == len(expected)

    def test_insert_right_of_updates_backward_link(self):
        dl = DLList([1, 3])
        first = dl.first_node()
        new_node = dl.insert_right_of(first, 2)
        assert new_node._prev is first
        assert new_node._next._value == 3
        assert new_node._next._prev is new_node

    def test_insert_right_of_updates_last(self):
        dl = DLList([1, 2])
        dl.insert_right_of(dl.node_at(-1), 3)
        assert dl.last() == 3


# ===========================================================================
# DLList – truncate / truncate_left
# ===========================================================================


class TestDLListTruncate:
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
        dl = DLList(values)
        dl.truncate(index)
        assert list(dl) == expected
        assert len(dl) == len(expected)

    def test_truncate_severs_forward_and_backward_links(self):
        dl = DLList([1, 2, 3])
        dl.truncate(2)
        assert dl._last._next is None

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
        dl = DLList(values)
        dl.truncate_left(n)
        assert list(dl) == expected
        assert len(dl) == len(expected)

    def test_truncate_left_severs_backward_link(self):
        dl = DLList([1, 2, 3])
        dl.truncate_left(1)
        assert dl._first._prev is None

    def test_truncate_updates_last_pointer(self):
        dl = DLList([1, 2, 3, 4])
        dl.truncate(2)
        assert dl.last() == 2

    def test_truncate_left_updates_first_pointer(self):
        dl = DLList([1, 2, 3, 4])
        dl.truncate_left(2)
        assert dl.first() == 3


# ===========================================================================
# DLList – popright and popleft
# ===========================================================================


class TestDLListPop:
    """Tests for popright() and popleft()."""

    @pytest.mark.parametrize(
        "values, pop_count, expected_popped, expected_remaining",
        [
            ([1, 2, 3], 1, [1], [2, 3]),
            ([1, 2, 3], 2, [1, 2], [3]),
            ([1, 2, 3], 3, [1, 2, 3], []),
            ([99], 1, [99], []),
        ],
    )
    def test_popleft_sequence(self, values, pop_count, expected_popped, expected_remaining):
        dl = DLList(values)
        popped = [dl.popleft() for _ in range(pop_count)]
        assert popped == expected_popped
        assert list(dl) == expected_remaining

    @pytest.mark.parametrize(
        "values, pop_count, expected_popped, expected_remaining",
        [
            ([1, 2, 3], 1, [3], [1, 2]),
            ([1, 2, 3], 2, [3, 2], [1]),
            ([1, 2, 3], 3, [3, 2, 1], []),
            ([99], 1, [99], []),
        ],
    )
    def test_popright_sequence(self, values, pop_count, expected_popped, expected_remaining):
        dl = DLList(values)
        popped = [dl.popright() for _ in range(pop_count)]
        assert popped == expected_popped
        assert list(dl) == expected_remaining

    def test_popleft_single_element_clears_both_pointers(self):
        dl = DLList([42])
        dl.popleft()
        assert dl._first is None
        assert dl._last is None

    def test_popright_single_element_clears_both_pointers(self):
        dl = DLList([42])
        dl.popright()
        assert dl._first is None
        assert dl._last is None

    def test_popleft_empty_raises(self):
        with pytest.raises(IndexError):
            DLList().popleft()

    def test_popright_empty_raises(self):
        with pytest.raises(IndexError):
            DLList().popright()

    def test_interleaved_popleft_popright(self):
        dl = DLList([1, 2, 3, 4, 5])
        assert dl.popleft() == 1
        assert dl.popright() == 5
        assert dl.popleft() == 2
        assert dl.popright() == 4
        assert list(dl) == [3]


# ===========================================================================
# DLList – sublist
# ===========================================================================


class TestDLListSublist:
    """Tests for sublist()."""

    @pytest.mark.parametrize(
        "values, left_idx, right_idx, expected",
        [
            ([1, 2, 3], 0, 2, [1, 2, 3]),  # full range
            ([1, 2, 3, 4, 5], 1, 3, [2, 3, 4]),  # middle slice
            ([1, 2, 3], 1, 1, [2]),  # single node
            ([1, 2, 3], 0, 0, [1]),  # first node only
            ([1, 2, 3], 2, 2, [3]),  # last node only
        ],
    )
    def test_sublist(self, values, left_idx, right_idx, expected):
        dl = DLList(values)
        result = dl.sublist(dl.node_at(left_idx), dl.node_at(right_idx))
        assert result == expected

    def test_sublist_invalid_range_raises(self):
        dl = DLList([1, 2, 3])
        with pytest.raises(KeyError):
            dl.sublist(dl.node_at(2), dl.node_at(0))


# ===========================================================================
# DLList – clear and _count_actual
# ===========================================================================


class TestDLListClear:
    """Tests for clear() and the _count_actual debugging helper."""

    def test_clear_resets_all_state(self):
        dl = DLList([1, 2, 3])
        dl.clear()
        assert len(dl) == 0
        assert dl._first is None
        assert dl._last is None

    @pytest.mark.parametrize(
        "values",
        [[], [1], [1, 2], list(range(20))],
    )
    def test_count_actual_matches_len(self, values):
        dl = DLList(values)
        assert dl._count_actual() == len(dl)

    @pytest.mark.parametrize(
        "values",
        [[], [1], [1, 2], list(range(20))],
    )
    def test_count_actual_matches_len_after_mutations(self, values):
        """_count_actual must stay in sync after a sequence of operations."""
        dl = DLList(values)
        if len(dl) >= 2:
            dl.append(99)
            dl.popleft()
            dl.append_left(0)
        assert dl._count_actual() == len(dl)
