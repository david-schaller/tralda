"""Module for dynamic partitions

The implementation allows, after initialization, constant lookup whether two items are in the same
set of the partition and efficient merging of two sets (based on arbitrary representatives).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tralda.datastructures.doubly_linked import DLList


class Partition:
    """Dynamic partition implementation."""

    def __init__(self, iterable: Iterable) -> None:
        """Constructor of the partition.

        Args:
            iterable: An iterable of iterables defining the initial partition.
        """
        self._partition = DLList()
        self._item2set: dict[Any, DLList] = {}

        for set_ in iterable:
            dll_node = self._partition.append(set(set_))
            for x in set_:
                self._item2set[x] = dll_node

    def __len__(self) -> int:
        """Current number of sets in the partition.

        Returns:
            The current number of sets in the partition.
        """
        return len(self._partition)

    def __iter__(self) -> PartitionIterator:
        """An iterator for the dynamic partition.

        Returns:
            An iterator for this partition.
        """
        return PartitionIterator(self)

    def __next__(self):
        pass

    def in_same_set(self, x: Any, y: Any) -> bool:
        """Check whether two items are in the same set of the partition.

        Args:
            x: Item 1.
            y: Item 2.

        Returns:
            True if x and y are in the same set of the partition, False otherwise.

        Raises:
            KeyError: If x or y is not an item in the partition.
        """
        try:
            return self._item2set[x] is self._item2set[y]
        except KeyError:
            if x not in self._item2set:
                raise KeyError(f"{x} is not an item in the partition")
            else:
                raise KeyError(f"{y} is not an item in the partition")

    def separated_xy_z(self, x: Any, y: Any, z: Any) -> bool:
        """Check whether two items are in the same set but in a different set than the third.

        Args:
            x: Item 1.
            y: Item 2.
            z: Item 3.

        Returns:
            True if x and y are in the same set of the partition and z is in a different one, False
            otherwise.

        Raises:
            KeyError: If x, y, or z is not an item in the partition.
        """
        try:
            return (
                self._item2set[x] is self._item2set[y]
                and self._item2set[x] is not self._item2set[z]
            )
        except KeyError:
            if x not in self._item2set:
                raise KeyError(f"{x} is not an item in the partition")
            elif y not in self._item2set:
                raise KeyError(f"{y} is not an item in the partition")
            else:
                raise KeyError(f"{z} is not an item in the partition")

    def merge(self, repr1: Any, repr2: Any) -> set[Any]:
        """Merge two sets of the partition based on arbitrary representatives.

        The merging is done efficiently by adding the elements of the smaller set to the bigger set.
        That way, each element is moved at most log(n) times into a new set.

        Args:
            repr1: Arbitrary representative of the first set.
            repr2: Arbitrary representative of the second set.

        Raises:
            KeyError: If 'repr1' or 'repr2' is not an item in the partition.

        Returns:
            The smaller of the original two sets that now have been merged.
        """
        try:
            set1 = self._item2set[repr1]
        except KeyError:
            raise KeyError(f"{repr1} is not an item in the partition")

        try:
            set2 = self._item2set[repr2]
        except KeyError:
            raise KeyError(f"{repr2} is not an item in the partition")

        if set1 is set2:
            return set()

        # ensure that set1 is smaller
        if len(set2._value) < len(set1._value):
            set1, set2 = set2, set1

        for x in set1._value:
            self._item2set[x] = set2

        # the bigger set gets extended
        set2._value |= set1._value

        self._partition.remove_node(set1)

        return set1._value


class PartitionIterator:
    """Iterator class for Partition class."""

    def __init__(self, dynamic_partition: Partition) -> None:
        """Constructor of the PartitionIterator.

        Args:
            dynamic_partition: A Partition instance.
        """
        self._dynamic_partition = dynamic_partition
        self._current = self._dynamic_partition._partition._first

    def __iter__(self) -> PartitionIterator:
        """Iterator function.

        Returns:
            This PartitionIterator.
        """
        return self

    def __next__(self) -> set[Any]:
        """Return the next set in the partition.

        Raises:
            StopIteration: If there are no more sets in the partition.

        Returns:
            The next set in the partition.
        """
        if self._current:
            x = self._current
            self._current = self._current._next
            return x._value
        else:
            raise StopIteration
