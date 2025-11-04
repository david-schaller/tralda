"""Simple binary search tree."""

from __future__ import annotations

from typing import Any, Iterator

from tralda.datastructures.bst.base import BinaryNode
from tralda.datastructures.bst.base import BaseBinarySearchTree
from tralda.datastructures.bst.base import BinaryTreeIterator


class BinarySearchTree(BaseBinarySearchTree):
    """Simple binary search tree."""

    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator

    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.

        Parameters
        ----------
        key : Any
            The key to be inserted.

        Raises
        ------
        ValueError
            If the key already exists.
        """
        self.root = self._recursive_insert(key, self.root)

    def _recursive_insert(self, key: Any, node: BinaryNode) -> BinaryNode:
        """Recursive insertion and rebalancing."""
        if node is None:
            return self.node_class(key)
        elif key < node.key:
            node.left = self._recursive_insert(key, node.left)
        elif key > node.key:
            node.right = self._recursive_insert(key, node.right)
        else:
            raise ValueError(f"key {key} already exists")

        node.update()

        return node

    def _delete_key(self, key: Any) -> None:
        """Delete a key if present.

        Parameters
        ----------
        key : Any
            The key to be deleted.
        """
        self.root = self._recursive_delete(key, self.root)

    def _recursive_delete(self, key: Any, node: BinaryNode) -> BinaryNode:
        """Recursive deletion."""
        if node is None:
            raise ValueError(f"key {key} not found")
        elif key < node.key:
            node.left = self._recursive_delete(key, node.left)
        elif key > node.key:
            node.right = self._recursive_delete(key, node.right)
        else:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                subst_node = self._smallest_in_subtree(node.right)
                node.set_attributes(subst_node.get_attributes())
                # now find and delete subst_node
                node.right = self._recursive_delete(node.key, node.right)

        if node is not None:
            node.update()

        return node

    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index and return it.

        Parameters
        ----------
        idx : int
            The index of the element to be removed.
        """
        self.root = self._recursive_pop_at_index(idx, self.root)

    def _recusive_pop_at_index(self, idx: int, node: BinaryNode) -> BinaryNode:
        """Recursive deletion of a node at the index."""
        if node is None:
            raise ValueError(f"could not find node with index {idx}")

        node_idx = node.left_size()

        if idx == node_idx:
            self._temp_attributes = node.get_attributes()
            return self._recursive_delete(node.key, node)
        elif idx < node_idx:
            node.left = self._recusive_pop_at_index(idx, node.left)
        else:
            node.right = self._recusive_pop_at_index(idx - node_idx - 1, node.right)

        if node is not None:
            node.update()

        return node
