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

        Args:
            key: The key to be inserted.
        """
        self.root = self._recursive_insert(key, self.root)

    def _recursive_insert(self, key: Any, node: BinaryNode | None) -> BinaryNode:
        """Recursive insertion and rebalancing.

        Args:
            key: The key to be inserted.
            node: The node below which to insert the new key or None if the node should be inserted
                at this position.

        Returns:
            The provided node or the newly created node if None was provided.

        Raises:
            KeyError: If the key already exists.
        """
        if node is None:
            return self.node_class(key)
        elif key < node.key:
            node.left = self._recursive_insert(key, node.left)
        elif key > node.key:
            node.right = self._recursive_insert(key, node.right)
        else:
            raise KeyError(f"key {key} already exists")

        node.update()

        return node

    def _delete_key(self, key: Any) -> None:
        """Delete a key if present.

        Args:
            key: The key to be deleted.
        """
        self.root = self._recursive_delete(key, self.root)

    def _recursive_delete(self, key: Any, node: BinaryNode | None) -> BinaryNode | None:
        """Recursive deletion.

        Args:
            key: The key to be deleted.
            node: The node below which the key to be deleted is located.

        Returns:
            The provided node, a node that is moved up, or None if a leaf node was deleted.

        Raises:
            KeyError: If the key is not in the tree.
        """
        if node is None:
            raise KeyError(f"key {key} not found")

        if key < node.key:
            node.left = self._recursive_delete(key, node.left)
        elif key > node.key:
            node.right = self._recursive_delete(key, node.right)
        else:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                subst_node = node.right.smallest_in_subtree()
                subst_node.copy_attributes_to_node(node)

                # now find and delete subst_node
                node.right = self._recursive_delete(node.key, node.right)

        if node is not None:
            node.update()

        return node

    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index and return it.

        Args:
            idx: The index of the element to be removed.
        """
        self.root = self._recursive_pop_at_index(idx, self.root)

    def _recusive_pop_at_index(self, idx: int, node: BinaryNode) -> BinaryNode | None:
        """Recursive deletion of a node at the index.

        Args:
            idx: The index of the element to be popped.
            node: The node below which the node with the specified index is located (or equal to
                this node).

        Returns:
            The provided node, a node that is moved up, or None if a leaf node was deleted.

        Raises:
            IndexError: If the key is not in the tree.
        """
        if node is None:
            raise IndexError(f"could not find node with index {idx}")

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
