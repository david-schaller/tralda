"""Doubly-linked list.

The list enables access to single list elements in order to modify/delete values in constant time.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class DLListNode:
    """Doubly-linked list node."""

    __slots__ = ("_value", "_prev", "_next")

    def __init__(
        self, value: Any, prev_node: DLListNode | None = None, next_node: DLListNode | None = None
    ) -> None:
        """Constructor of the doubly-linked list node.

        Args:
            value: The value that the list node shall hold.
            prev_node: The predecessor node in the doubly-linked list.
            next_node: The successor node in the doubly-linked list.
        """
        self._value = value
        self._prev = prev_node
        self._next = next_node

    def get(self) -> Any:
        """Getter for the value.

        Returns:
            The value held by the list node.
        """
        return self._value


class DLList:
    """Doubly-linked list."""

    __slots__ = ("_first", "_last", "_count")

    def __init__(self, *args):
        """Constructor of the doubly-linked list.

        Args:
            args: A single value or an iterable of values to be added to the list.
        """
        self._first = None
        self._last = None
        self._count = 0
        for arg in args:
            if isinstance(arg, Iterable):
                for item in arg:
                    self.append(item)
            else:
                self.append(item)

    def __len__(self) -> int:
        """Number of items in the list.

        Returns:
            The number of items in the list.
        """
        return self._count

    def __nonzero__(self):
        """Returns whether the list is non-empty.

        Returns:
            Return True if the list is non-empty.
        """
        return self._count > 0

    def __iter__(self) -> DLListIterator:
        """Returns an iterator object for the list.

        Returns:
            An iterator for this list.
        """
        return DLListIterator(self)

    def __next__(self):
        pass

    def __getitem__(self, index: int) -> Any:
        """Access value at a specified index.

        Returns:
            The value at the specified index.
        """
        return self.node_at(index)._value

    def first(self) -> Any:
        """Return the first item in the list.

        Returns:
            The first item in the list.
        """
        return self._first._value

    def last(self) -> Any:
        """Return the last item in the list.

        Returns:
            The last item in the list.
        """
        return self._last._value

    def first_node(self) -> DLListNode:
        """Return the first list node.

        Returns:
            The first node of the list.
        """
        if self._first is None:
            raise IndexError("the list is empty")

        return self._first

    def node_at(self, index: int) -> DLListNode:
        """Return the list node at the specified index (in O(n) time).

        The index can also be negative in which case it is counted from the end (as for built-in
        Python lists).

        Returns:
            The list node at the specified index.

        Raises:
            TypeError: If 'index' is not an integer.
            IndexError: If the index is out of bounds.
        """
        if not isinstance(index, int):
            raise TypeError("index must be of type int")

        if index < (-self._count) or index >= self._count:
            raise IndexError(f"index {index} is out of bounds")

        if index < 0:
            index += self._count

        if index <= self._count // 2:
            # start from beginning
            node = self._first
            for _ in range(index):
                node = node._next

        else:
            # start from end
            node = self._last
            for _ in range(self._count - index - 1):
                node = node._prev

        return node

    def append(self, value: Any) -> DLList:
        """Append an item to the list.

        Args:
            value: The value to be inserted at the end of the list.

        Returns:
            The new end node of the list.
        """
        new_end = DLListNode(value, prev_node=self._last)
        if self._last:
            self._last._next = new_end

        self._last = new_end
        if not self._first:
            self._first = new_end

        self._count += 1

        return new_end

    def extend(self, iterable: Iterable) -> None:
        """Append multiple items to the list.

        Args:
            An iterable of values to be inserted at the end of the list.
        """
        for value in iterable:
            self.append(value)

    def append_left(self, value: Any) -> DLListNode:
        """Append an item to left side of the list.

        Args:
            value: The value to be inserted at the beginning of the list.

        Returns:
            The new start node of the list.
        """
        new_start = DLListNode(value, next_node=self._first)
        if self._first:
            self._first._prev = new_start

        self._first = new_start
        if not self._last:
            self._last = new_start

        self._count += 1

        return new_start

    def remove_node(self, node: DLListNode) -> None:
        """Remove an item by reference to the 'DLListNode' instance in O(1) time.

        Args:
            node: The node to be removed.
        """
        if node._prev:
            node._prev._next = node._next
        if node._next:
            node._next._prev = node._prev
        if self._first is node:
            self._first = node._next
        if self._last is node:
            self._last = node._prev

        node._prev, node._next = None, None
        self._count -= 1

    def remove(self, value: Any) -> None:
        """Remove an item by value (in O(n) time).

        Args:
            value: The value to be removed.

        Raises:
            KeyError: If the value was not found in the list.
        """
        node = self._first
        while node:
            if node._value == value:
                self.remove_node(node)
                return
            node = node._next

        raise KeyError("value {} is not in the doubly-linked list".format(value))

    def remove_range(self, index: int, length: int = None) -> None:
        """Removes a range from the index of the specified length.

        Removes the range [index, index+length) from the sequence. If no length is specified or
        index + length is out of bounds, the list gets truncated.

        Args:
            index: The start of the range to be removed.
            length: The length of the range to be removed.

        Raises:
            TypeError: If index is not an integer.
            TypeError: If length is not an integer or below 1.
        """
        if not isinstance(index, int):
            raise TypeError("index must be of type 'int'")
        elif index < 0:
            index = self._count + index

        if length is not None and (not isinstance(length, int) or length < 1):
            raise TypeError("length must be of type 'int' and >0")

        if length is None or index + length >= self._count:
            self.truncate(index)

        elif index == 0:
            self.truncate_left(length)

        else:
            cut_start = self.node_at(index)
            cut_end = self.node_at(index + length - 1)

            cut_start._prev._next = cut_end._next
            cut_end._next._prev = cut_start._prev
            cut_start._prev, cut_end._next = None, None

            self._count -= length

    def insert_right_of(self, node: DLListNode, value: Any) -> DLListNode:
        """Insert a new item right of a node of the list in O(1).

        Args:
            node: The list node next to which the item shall be inserted.
            value: The value to be inserted.

        Returns:
            The list node for the inserted value.
        """
        if node is self._last:
            new_node = self.append(value)
        else:
            new_node = DLListNode(value, prev_node=node, next_node=node._next)
            new_node._next._prev = new_node
            node._next = new_node
            self._count += 1

        return new_node

    def truncate(self, index: int) -> None:
        """Truncate all nodes starting from the specified index in O(n) time.

        Args:
            index: The index from which on the list is truncated.
        """
        if index <= 0:
            self.clear()
        else:
            new_end = self.node_at(index - 1)
            self._last = new_end
            if new_end._next:
                new_end._next._prev = None
                new_end._next = None
            self._count = index

    def truncate_left(self, n: int) -> None:
        """Truncate n nodes on the left side.

        Args:
            n: The number of items to be truncated.
        """
        if n >= self._count:
            self.clear()
        else:
            new_start = self.node_at(n)
            self._first = new_start
            if new_start._prev:
                new_start._prev._next = None
                new_start._prev = None
            self._count -= n

    def popright(self) -> Any:
        """Removes the last item of the list and returns its value.

        Returns:
            The former last item of the list which has been removed.

        Raises:
            IndexError: If the list is empty.
        """
        if self._last:
            value = self._last._value
            self.remove_node(self._last)
            return value

        raise IndexError("cannot pop from empty list")

    def popleft(self) -> Any:
        """Removes the first item of the list and returns its value.

        Returns:
            The former first item of the list which has been removed.

        Raises:
            IndexError: If the list is empty.
        """
        if self._first:
            value = self._first._value
            self.remove_node(self._first)
            return value

        raise IndexError("cannot pop from empty list")

    def clear(self) -> None:
        """Clear the list."""
        self._first = None
        self._last = None
        self._count = 0

    def sublist(self, left_node: DLListNode, right_node: DLListNode) -> list:
        """Sublist of items in the subsequence between two nodes of the list.

        Args:
            left_node: The left bound of the subsequence.
            right_node: The right bound of the subsequence.

        Returns:
            The values in the subsequence in the standard Python list.

        Raises:
            KeyError: If 'right_node' is not a successor of 'left_node' in the list.
        """
        result = []
        current = left_node

        while current:
            result.append(current._value)
            if current is right_node:
                return result

            current = current._next

        raise KeyError(
            f"{right_node} is not contained in the subsequence starting with {left_node}"
        )

    def _count_actual(self) -> int:
        """Counts the actual number of elements (for debugging purposes).

        Runs in O(n) time.

        Returns:
            The counted number of items in the list.
        """
        current = self._first
        counter = 0

        while current:
            counter += 1
            current = current._next

        return counter


class DLListIterator:
    """Iterator class for doubly-linked list."""

    def __init__(self, dllist: DLList) -> None:
        """Constructor of DLListIterator.

        Args:
            dllist: The doubly-linked list.
        """
        self.dllist = dllist
        self._current = dllist._first

    def __iter__(self) -> DLListIterator:
        """An iterator for the doubly-linked list."""
        return self

    def __next__(self) -> Any:
        """Return the next item in the list.

        Raises:
            StopIteration: If there are no more items.

        Returns:
            The next item in the iteration.
        """
        if self._current:
            x = self._current
            self._current = self._current._next
            return x._value
        else:
            raise StopIteration
