"""Base classes for binary search trees."""

from __future__ import annotations

from typing import Any
from typing import Iterator
from typing import Iterable


class BinaryNode:
    __slots__ = (
        "key",
        "left",
        "right",
        "size",
        "height",
    )
    _attributes = ("key",)

    def __init__(self, key: Any) -> None:
        """Initialize the binary search tree node.

        Args:
            key: The key/label of the node. Keys must be unique within a binary search tree.
        """
        self.key: Any = key

        self.left: BinaryNode | None = None
        self.right: BinaryNode | None = None

        # stores number of elements in its subtree
        self.size: int = 1

        # height of the subtree
        self.height: int = 1

    def __str__(self) -> str:
        """String representation of the node.

        Returns:
            String representation.
        """
        return f"<node: {self.key}>"

    def get_attributes(self) -> tuple[Any]:
        """Attributes of this node.

        Returns:
            Attributes of this node.
        """
        return tuple(getattr(self, a) for a in self._attributes)

    def set_attributes(self, attributes: tuple[Any]) -> None:
        """Set the attributes of this node.

        Args:
            attributes: The attributes in the order as defined for this BinaryNode (sub)class.
        """
        for val, a in zip(attributes, self._attributes):
            setattr(self, a, val)

    def copy_attributes_to_node(self, other: BinaryNode) -> None:
        """Copy attributes of this node to another node.

        Args:
            other: The node into which to copy the attributes.

        Raises:
            TypeError: If this instance and other do not have exactly the same type.
        """
        # test both ways as subclasses are not allowed here
        if not (isinstance(other, type(self)) and isinstance(self, type(other))):
            TypeError(
                f"nodes must have the same type (self if {type(self)}, other is {type(other)})"
            )

        other.set_attributes(self.get_attributes())

    def update(self) -> None:
        """Update height and size of (the subtree under) the node."""
        if self.left and self.right:
            self.height = 1 + max(self.left.height, self.right.height)
            self.size = 1 + self.left.size + self.right.size
        elif self.left:
            self.height = 1 + self.left.height
            self.size = 1 + self.left.size
        elif self.right:
            self.height = 1 + self.right.height
            self.size = 1 + self.right.size
        else:
            self.height = 1
            self.size = 1

    def left_size(self) -> int:
        """Size of the left subtree.

        Returns:
            Size of the left subtree.
        """
        return self.left.size if self.left else 0

    def right_size(self) -> int:
        """Size of the right subtree.

        Returns:
            Size of the right subtree.
        """
        return self.right.size if self.right else 0

    def balance(self) -> int:
        """Balance factor of the node.

        Returns:
            Balance factor.
        """
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0

        return left_height - right_height

    def copy(self) -> BinaryNode:
        """A copy of this node.

        Returns:
            A copy of this node.
        """
        copy = self.__class__(*self.get_attributes())
        copy.height = self.height
        copy.size = self.size

        return copy


class BinaryTreeIterator:
    """Iterator for binary search trees."""

    __slots__ = ("tree", "_inorder_generator")

    def __init__(self, tree: BaseBinarySearchTree):
        """Initilize the tree iterator.

        Args:
            tree: The binary search tree.
        """
        self.tree = tree
        self._inorder_generator = self.tree._inorder_traversal()

    def __iter__(self) -> BaseBinarySearchTree:
        """Iterator function.

        Returns:
            An iterator for a binary search tree.
        """
        return self

    def __next__(self) -> Any:
        """The next item in the binary search tree.

        Returns:
            The next item.

        Raises:
            StopIteration: When no items are left.
        """
        try:
            node = next(self._inorder_generator)
            return node.key
        except StopIteration:
            raise StopIteration


class BaseBinarySearchTree:
    """Base class for binary search trees."""

    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator

    def __init__(self) -> None:
        """Initialize the balanced binary search tree."""
        self.root: BinaryNode | None = None

        # for temporarily storing the node attributes for insertion/popping in recursive functions
        self._temp_attributes: Any | None = None

    def __iter__(self) -> Iterator[Any]:
        """An iterator for the binary search tree.

        Returns:
            An iterator for the binary search tree.
        """
        return self.iterator_class(self)

    def __next__(self) -> None:
        pass

    def __nonzero__(self) -> bool:
        """Return whether the tree is non-empty.

        Returns:
            Whether the tree contains elements or not.
        """
        return bool(self.root)

    def __len__(self) -> int:
        """Number of elements in the tree.

        Returns:
            Number of elements.
        """
        return self.root.size if self.root else 0

    def __contains__(self, item: Any) -> bool:
        """Return whether the tree contains the item.

        Args:
            item: The item for which to check whether it is contained in the tree.

        Returns:
            Whether the tree contains the item.
        """
        return self._find(item) is not None

    def __getitem__(self, idx: int) -> Any:
        """Return the element at the index.

        Same as 'key_at_index(index)'.

        Args:
            idx: The index of the key to get.

        Returns:
            The key of the node at the index.
        """
        return self._node_at_index(idx).key

    def key_at_index(self, idx: int) -> Any:
        """Return the key at the index.

        Args:
            idx: The index of the key to get.

        Returns:
            The key of the node at the index.
        """
        return self._node_at_index(idx).key

    def _validate_index(self, idx: int) -> int:
        """Return the node at the index.

        Args:
            idx: The index to be validated.

        Returns:
            The input index or the corresponding positive index if the input was negative.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if idx < 0:
            if idx < -self.root.size:
                raise IndexError(f"index {idx} is out of range")
            else:
                idx += self.root.size

        if idx >= self.root.size:
            raise IndexError(f"index {idx} is out of range")

        return idx

    def _node_at_index(self, idx: int) -> BinaryNode:
        """Return the node at the index.

        Args:
            idx: The index.

        Returns:
            The node instance at the index.

        Raises:
            IndexError: If the index is out of bounds.
            RuntimeError: If the index seems valid but the node could not be found. A corrupted
                integrity of the tree datastructure could be the reason.
        """
        idx = self._validate_index(idx)
        current = self.root
        current_sum = 0

        while current:
            current_idx = current_sum + current.left_size()
            if idx == current_idx:
                return current
            elif idx < current_idx:
                current = current.left
            else:
                current = current.right
                current_sum = current_idx + 1

        raise RuntimeError(f"could not find node with index {idx}")

    def add(self, item: Any) -> None:
        """Add an item if not yet present.

        Args:
            item: The new item to be inserted.
        """
        try:
            self.insert(item)
        except (KeyError, ValueError):
            pass

    def insert(self, key: Any) -> None:
        """Insert an item.

        This function will throw a ValueError if the key is already present. If you do not want
        this behavior, use the function add() instead.

        Args:
            key: The new item to be inserted.

        Raises:
            KeyError: If the key already exists.
        """
        self._temp_attributes = (key,)
        self._insert_key(key)
        self._temp_attributes = None

    def remove(self, key: Any) -> None:
        """Remove a key from the tree.

        Args:
            key: The new item to be removed.

        Raises:
            KeyError: If the key is not in the tree.
        """
        self._delete_key(key)

    def discard(self, key: Any) -> None:
        """Remove a key from the tree if present.

        Args:
            key: The new item to be removed.
        """
        try:
            self._delete_key(key)
        except KeyError:
            pass

    def pop(self) -> Any:
        """Remove and return the greatest item.

        Returns:
            The greatest item in the tree.

        Raises:
            IndexError: If the tree is empty.
        """
        if not self.root:
            raise IndexError("pop from empty tree")

        return self.pop_at_index(len(self) - 1)

    def clear(self) -> None:
        """Removes all items from the tree."""
        self.root = None

    def difference_update(self, items: Iterable[Any]) -> None:
        """Discard all elements in the collection.

        Args:
            items: The items to be discarded.
        """
        for item in items:
            self.discard(item)

    def remove_at_index(self, idx: int) -> None:
        """Remove node at the index.

        Args:
            idx: The index of the element to be removed.

        Raises:
            IndexError: If the index is out of bounds.
        """
        self.pop_at_index(idx)

    def pop_at_index(self, idx: int) -> Any:
        """Remove item at the index.

        Args:
            idx: The index of the element to be removed and returned.

        Returns:
            The item at the index.

        Raises:
            IndexError: If the index is out of bounds.
        """
        idx = self._validate_index(idx)
        self._pop_at_index(idx)

        to_pop = tuple(self._temp_attributes)
        self._temp_attributes = None

        if len(to_pop) == 1:
            to_pop = to_pop[0]

        return to_pop

    def _find(self, key: Any) -> BinaryNode | None:
        """Find the node for the specified key.

        Args:
            key: The key to be searched.

        Returns:
            The corresponding tree node or None if the key was not found.
        """
        if not self.root:
            return None

        current = self.root
        while current:
            if key == current.key:
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right

    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.

        Args:
            key: The key to be inserted.

        Raises:
            KeyError: If the key already exists.
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("not implemented for base class")

    def _delete_key(self, key: Any) -> None:
        """Delete a key.

        Args:
            key: The key to be deleted.

        Raises:
            KeyError: If the key does not exist.
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("not implemented for base class")

    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index and return it.

        Args:
            idx: The index of the element to be removed.

        Raises:
            IndexError: If the index is out of bounds.
            NotImplementedError: If the child class does not implement this method.
        """
        raise NotImplementedError("not implemented for base class")

    def _smallest_in_subtree(self, node: BinaryNode) -> BinaryNode:
        """Return the left-most (smallest element) node in the subtree.

        Args:
            node: The node whose subtree is to be considered.

        Returns:
            The node with the smallest key in the subtree.
        """
        current = node
        while current.left:
            current = current.left

        return current

    def _largest_in_subtree(self, node: BinaryNode) -> BinaryNode:
        """Return the right-most (largest element) node in the subtree.

        Args:
            node: The node whose subtree is to be considered.

        Returns:
            The node with the largest key in the subtree.
        """
        current = node
        while current.right:
            current = current.right

        return current

    def _inorder_traversal(self) -> Iterator[BinaryNode]:
        """Generator for the nodes in a in-order traversal of the tree.

        Yields:
            All nodes of the tree in in-order.
        """

        def _inorder(node: BinaryNode) -> Iterator[BinaryNode]:
            if node.left:
                yield from _inorder(node.left)
            yield node
            if node.right:
                yield from _inorder(node.right)

        if self.root:
            yield from _inorder(self.root)
        else:
            yield from []

    def _postorder_traversal(self) -> Iterator[BinaryNode]:
        """Generator for the nodes in a post-order traversal of the tree.

        Yields:
            All nodes of the tree in post-order.
        """

        def _postorder(node: BinaryNode) -> Iterator[BinaryNode]:
            if node.left:
                yield from _postorder(node.left)
            if node.right:
                yield from _postorder(node.right)
            yield node

        if self.root:
            yield from _postorder(self.root)
        else:
            yield from []

    def copy(self) -> BaseBinarySearchTree:
        """Copy the tree.

        Returns:
            A copy of the tree.
        """
        tree_copy = self.__class__()
        if self.root:
            tree_copy.root = self._copy_subtree(self.root)

        return tree_copy

    def _copy_subtree(self, node: BinaryNode) -> BinaryNode:
        """Copy a subtree.

        Args:
            node: The node whose subtree to copy.

        Returns:
            The copied node (= the root of the copied subtree).
        """
        node_copy = node.copy()
        if node.left:
            node_copy.left = self._copy_subtree(node.left)
        if node.right:
            node_copy.right = self._copy_subtree(node.right)

        return node_copy

    def to_newick(self) -> str:
        """Newick representation of the tree.

        Intended for testing purpose.

        Returns:
            A Newick representation of the tree.
        """

        def _newick(node):
            if not (node.left or node.right):
                return str(node.key)
            else:
                if node.left and node.right:
                    s = f"({_newick(node.left)},{_newick(node.right)})"
                elif node.left:
                    s = f"({_newick(node.left)},-)"
                elif node.right:
                    s = f"(-,{_newick(node.right)})"
                else:
                    s = ""
                return s + str(node.key)

        return _newick(self.root) if self.root else ""

    def check_integrity(self, verbose: bool = False) -> bool:
        """Integrity check of the tree.

        Checks whether the size and heigth is correct in all subtrees. Intended for debugging and
        testing purpose.

        Args:
            verbose: If True print where the integrity check has failed.

        Returns:
            Whether all integrity checks have been passed.
        """
        for node in self._postorder_traversal():
            height_left, height_right, size = 0, 0, 1

            if node.left:
                height_left = node.left.height
                size += node.left.size

            if node.right:
                height_right = node.right.height
                size += node.right.size

            if node.height != 1 + max(height_left, height_right):
                if verbose:
                    print(f"height of node {node} is incorrect")
                return False

            if node.size != size:
                if verbose:
                    print(f"size of node {node} is incorrect")
                return False

        return True
