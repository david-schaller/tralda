"""Linked list.

The list enables access to single list elements in order to modify/delete values in constant time.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


class LinkedListNode:
    """Linked list node."""

    __slots__ = ("_value", "_next")

    def __init__(self, value: Any, next_node: LinkedListNode | None = None) -> None:
        """Constructor of the inked list node.

        Args:
            value: The value that the list node shall hold.
            next_node: The successor node in the doubly-linked list.
        """
        self._value = value
        self._next = next_node

    def get(self) -> Any:
        """Getter for the value.

        Returns:
            The value held by the list node.
        """
        return self._value


class LinkedList:
    """Linked list."""

    __slots__ = ("_first", "_last", "_count")

    def __init__(self, *args) -> None:
        """Constructor of the inked list.

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

    def __iter__(self) -> LinkedListIterator:
        """Returns an iterator object for the list.

        Returns:
            An iterator for this list.
        """
        return LinkedListIterator(self)

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

    def first_node(self) -> LinkedListNode:
        """Return the first list node.

        Returns:
            The first node of the list.
        """
        return self._first

    def node_at(self, index: int) -> LinkedListNode:
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
            raise TypeError("index must be of type 'int'")

        if index < (-self._count) or index >= self._count:
            raise IndexError(f"index {index} is out of bounds")

        if index < 0:
            index += self._count

        # search from beginning
        node = self._first
        for _ in range(index):
            node = node._next

        return node

    def append(self, value: Any) -> LinkedListNode:
        """Append an item to the list.

        Args:
            value: The value to be inserted at the end of the list.

        Returns:
            The new end node of the list.
        """
        new_end = LinkedListNode(value)
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

    def concatenate(self, other: LinkedList) -> LinkedList:
        """Merges two linked lists.

        The instance for which this function is called is extended, the other list should no longer
        be used.

        Args:
            other: The liked list to concatenate with this list.

        Returns:
            This LikedList instance.

        Raises:
            TypeError: If 'other' is not a LinkedList instance.
        """
        if not isinstance(other, LinkedList):
            raise TypeError("must be of type 'LinkedList'")

        if self._last:
            self._last._next = other._first
        else:
            self._first = other._first

        if other._last:
            self._last = other._last

        self._count += other._count

        return self

    def append_left(self, value: Any) -> LinkedListNode:
        """Append an item to left side of the list.

        Args:
            value: The value to be inserted at the beginning of the list.

        Returns:
            The new start node of the list.
        """
        new_start = LinkedListNode(value, next_node=self._first)
        self._first = new_start
        if not self._last:
            self._last = new_start

        self._count += 1

        return new_start

    def remove(self, value: Any) -> Any:
        """Remove an item by value (in O(n) time).

        Args:
            value: The value to be removed.

        Raises:
            KeyError: If the value was not found in the list.
        """
        node = self._first
        prev_node = None

        while node:
            if node._value == value:
                if prev_node:
                    prev_node._next = node._next
                else:
                    self._first = node._next

                if not node._next:
                    self._last = prev_node

                self._count -= 1
                return

            prev_node = node
            node = node._next

        raise KeyError(f"value {value} is not in the linked list")

    def insert_right_of(self, node: LinkedListNode, value: Any) -> LinkedListNode:
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
            new_node = LinkedListNode(value, next_node=node._next)
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
            self._count -= n

    def popleft(self) -> Any:
        """Removes the first item of the list and returns its value.

        Returns:
            The former first item of the list which has been removed.

        Raises:
            IndexError: If the list is empty.
        """
        if self._first:
            value = self._first._value
            self._first = self._first._next
            self._first._next = None
            self._count -= 1
            return value

        raise IndexError("cannot pop from empty linked list")

    def clear(self) -> None:
        """Clear the list."""
        self._first = None
        self._last = None
        self._count = 0

    def _count_actual(self):
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


class LinkedListIterator:
    """Iterator class for linked list."""

    def __init__(self, llist: LinkedList) -> None:
        """Constructor of LinkedListIterator.

        Args:
            llist: The linked list.
        """
        self.llist = llist
        self._current = llist._first

    def __iter__(self) -> LinkedListIterator:
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
