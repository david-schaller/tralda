"""AVL-tree implementation.

Balanced binary search tree implementation of a set (TreeSet) and dictionary (TreeDict).
"""

from __future__ import annotations

from enum import Enum
from typing import Any
from typing import Iterator

from tralda.datastructures.bst.base import BinaryNode
from tralda.datastructures.bst.base import BaseBinarySearchTree
from tralda.datastructures.bst.base import BinaryTreeIterator


class TreeSet(BaseBinarySearchTree):
    """AVL tree implementation of a sorted set."""

    __slots__ = ()

    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator

    def check_integrity(self, verbose: bool = False) -> bool:
        """Integrity check of the tree.

        Checks whether all children have a correct parent reference and the size and heigth is
        correct in all subtrees. Additionally, the AVL property is checked. Intended for debugging
        and testing purpose.

        Args:
            verbose: If True print where the integrity check has failed.

        Returns:
            Whether all integrity checks have been passed.
        """
        super().check_integrity(verbose=verbose)

        for node in self._inorder_traversal():
            if abs(node.balance()) > 1:
                if verbose:
                    print(f"node {node} is unbalanced")
                return False

        return True

    def _rotate_right(self, node: BinaryNode) -> BinaryNode:
        """Perform a right rotation on the node.

        Args:
            node: The node on which to perform a right rotation.

        Returns:
            The former left child of the node which is its parent after the rotation.
        """
        left_child = node.left
        node.left = left_child.right
        left_child.right = node

        node.update()
        left_child.update()

        return left_child

    def _rotate_left(self, node: BinaryNode) -> BinaryNode:
        """Perform a left rotation on the node.

        Args:
            node: The node on which to perform a left rotation.

        Returns:
            The former right child of the node which is its parent after the rotation.
        """
        right_child = node.right
        node.right = right_child.left
        right_child.left = node

        node.update()
        right_child.update()

        return right_child

    def _rebalance(self, node: BinaryNode) -> BinaryNode:
        """Rebalance the node.

        Args:
            node: The node to be rebalanced.

        Returns:
            The root of the tree after rebalancing.
        """
        node.update()
        balance = node.balance()

        if balance > 1:
            if node.left.balance() >= 0:
                # single rotation needed
                node = self._rotate_right(node)
            else:
                # double rotation needed
                node.left = self._rotate_left(node.left)
                node = self._rotate_right(node)
        elif balance < -1:
            if node.right.balance() <= 0:
                # single rotation needed
                node = self._rotate_left(node)
            else:
                # double rotation needed
                node.right = self._rotate_right(node.right)
                node = self._rotate_left(node)

        return node

    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.

        Args:
            key: The key to be inserted.
        """
        self.root = self._insert_and_rebalance(key, self.root)

    def _insert_and_rebalance(self, key: Any, node: BinaryNode | None) -> BinaryNode:
        """Recursive insertion and rebalancing.

        Args:
            key: The key to be inserted.
            node: The node under which the key must be inserted.

        Returns:
            The root of the subtree after inserting the key and rebalancing.

        Raises:
            KeyError: If the key already exists.
        """
        if node is None:
            return self.node_class(*self._temp_attributes)

        if key < node.key:
            node.left = self._insert_and_rebalance(key, node.left)
        elif key > node.key:
            node.right = self._insert_and_rebalance(key, node.right)
        else:
            raise KeyError(f"key {key} already exists")

        return self._rebalance(node)

    def _delete_key(self, key: Any) -> None:
        """Delete a key if present.

        Args:
            key: The key to be deleted.
        """
        self.root = self._delete_and_rebalance(key, self.root)

    def _delete_and_rebalance(self, key: Any, node: BinaryNode | None) -> BinaryNode:
        """Recursive deletion and rebalancing.

        Args:
            key: The key to be deleted.
            node: The node below which the key to be deleted is located.

        Returns:
            The root of the subtree after deletion and rebalancing.

        Raises:
            KeyError: If the key is not in the tree.
        """
        if node is None:
            raise KeyError(f"key {key} not found")

        if key < node.key:
            node.left = self._delete_and_rebalance(key, node.left)
        elif key > node.key:
            node.right = self._delete_and_rebalance(key, node.right)
        else:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                subst_node = node.right.smallest_in_subtree()
                subst_node.copy_attributes_to_node(node)
                # now find and delete subst_node
                node.right = self._delete_and_rebalance(node.key, node.right)

        if node is not None:
            node = self._rebalance(node)

        return node

    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index.

        Args:
            idx: The index of the element to be removed.
        """
        self.root = self._pop_at_index_and_rebalance(idx, self.root)

    def _pop_at_index_and_rebalance(self, idx: int, node: BinaryNode) -> BinaryNode:
        """Recursive deletion of a node at the index."""
        if node is None:
            raise ValueError(f"could not find node with index {idx}")

        node_idx = node.left_size()

        if idx == node_idx:
            self._temp_attributes = node.get_attributes()
            return self._delete_and_rebalance(node.key, node)
        elif idx < node_idx:
            node.left = self._pop_at_index_and_rebalance(idx, node.left)
        else:
            node.right = self._pop_at_index_and_rebalance(idx - node_idx - 1, node.right)

        if node is not None:
            node = self._rebalance(node)

        return node


class TreeDictNode(BinaryNode):
    __slots__ = ("value",)
    _attributes = (
        "key",
        "value",
    )

    def __init__(self, key: Any, value: Any) -> None:
        """Initialize the TreeDict node.

        Args:
            key: The key of the node. Keys must be unique within a binary search tree.
            value: The value associated with the key.
        """
        super().__init__(key)
        self.value = value


class TreeDictIteratorMode(Enum):
    KEY = 1
    VALUE = 2
    KEY_AND_VALUE = 3


class TreeDictIterator(BinaryTreeIterator):
    """Iterator for AVL-tree-based dictionary."""

    __slots__ = ("_mode",)

    def __init__(
        self,
        tree: TreeDict,
        mode: TreeDictIteratorMode = TreeDictIteratorMode.KEY,
    ) -> None:
        """Initilize the TreeDict iterator.

        Args:
            tree: The TreeDict instance.
            mode: What shall be iterated through.
        """
        super().__init__(tree)

        self._mode = mode

    def __iter__(self) -> TreeDictIterator:
        return self

    def __next__(self) -> Any | tuple[Any, Any]:
        """The next key, value or key-value pair in the binary search tree.

        Returns:
            The next key, value, or key-value pair.

        Raises:
            StopIteration: When no items are left.
        """
        try:
            node = next(self._inorder_generator)
            if self._mode == TreeDictIteratorMode.KEY:
                return node.key
            elif self._mode == TreeDictIteratorMode.VALUE:
                return node.value
            else:
                return (node.key, node.value)
        except StopIteration:
            raise StopIteration


class TreeDict(TreeSet):
    __slots__ = ()

    node_class = TreeDictNode
    iterator_class: Iterator[Any] = TreeDictIterator

    def __getitem__(self, key: Any) -> Any:
        """Return the value associated with the key.

        Overrides the method of the base class where the key at a specified index is returned.

        Args:
            key: The key.

        Returns:
            The value associated with the key.

        Raises:
            KeyError: If the key does not exist.
        """
        node = self._find(key)

        if not node:
            raise KeyError(f"key {key} is not in the TreeDict")

        return node.value

    def get(self, item: Any, default: Any = None) -> Any:
        """Return the value associated with the key or a default value if the key does not exist.

        Args:
            key: The key for which to get the asociated value
            default: The default to be returned if the key does not exist. Defaults to None.

        Returns:
            The value associated with the key or the default value.
        """
        node = self._find(item)

        if node:
            return node.value
        else:
            return default

    def keys(self) -> Iterator[Any]:
        """Iterator for the keys.

        Returns:
            An iterator for the keys.
        """
        return self.iterator_class(self, mode=TreeDictIteratorMode.KEY)

    def values(self) -> Iterator[Any]:
        """Iterator for the values.

        Returns:
            An iterator for the values.
        """
        return self.iterator_class(self, mode=TreeDictIteratorMode.VALUE)

    def items(self) -> Iterator[tuple[Any, Any]]:
        """Iterator for the key-value pairs.

        Returns:
            An iterator for the key-value pairs.
        """
        return self.iterator_class(self, mode=TreeDictIteratorMode.KEY_AND_VALUE)

    def value_at_index(self, idx: int) -> Any:
        """Return the value at the index.

        Args:
            idx: The index.

        Returns:
            The value of the node at the index.
        """
        return self._node_at_index(idx).value

    def key_and_value_at_index(self, idx: int) -> tuple[Any, Any]:
        """Return the key-value pair at the index.

        Args:
            idx: The index.

        Returns:
            The key-value pair of the node at the index.
        """
        node = self._node_at_index(idx)

        return (node.key, node.value)

    def add(self, key: Any, value: Any) -> None:
        """Insert a key and value.

        Args:
            key: The key to be added.
            value: The associated value.
        """
        self.insert(key, value)

    def insert(self, key: Any, value: Any) -> None:
        """Insert a key and value.

        Args:
            key: The key to be added.
            value: The associated value.

        Raises:
            If the key already exists.
        """
        self._temp_attributes = (key, value)
        self._insert_key(key)
        self._temp_attributes = None
