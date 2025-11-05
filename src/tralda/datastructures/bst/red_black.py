"""Red-black tree implementation.

Balanced binary search tree implementation of a set (TreeSet) and dictionary (TreeDict).

References:
    [1] https://www.happycoders.eu/de/algorithmen/rot-schwarz-baum-java/
"""

from __future__ import annotations

from typing import Any
from typing import Iterator

from tralda.datastructures.bst.base import BinaryNode
from tralda.datastructures.bst.base import BaseBinarySearchTree
from tralda.datastructures.bst.base import BinaryTreeIterator


class RedBlackTreeNode(BinaryNode):
    __slots__ = (
        "parent",
        "is_red",
    )

    def __init__(self, key: Any) -> None:
        """Initialize the red-black tree node.

        Args:
            key: The key/label of the node. Keys must be unique within a binary search tree.
        """
        super().__init__(key)

        self.parent: RedBlackTreeNode | None = None
        self.is_red: bool = False

    def copy(self) -> RedBlackTreeNode:
        """A copy of this node.

        Returns:
            BinaryNode: A copy of this node.
        """
        copy = super().copy()
        copy.is_red = self.is_red

        return copy


class TemporyNilNode(RedBlackTreeNode):
    def __init__(self):
        """Initialize the temporary NIL node for red-black trees."""
        super().__init__(0)  # dummy key

        # as the tempory NIL nodes will be deleting after the reparing function is called, it
        # must not contribute to the height and size
        self.size: int = 0
        self.height: int = 0

    def update(self) -> None:
        """Update height and size of (the subtree under) the node."""
        # does nothing for tempory NIL nodes
        return


class TreeSet(BaseBinarySearchTree):
    """Red-black tree."""

    node_class = RedBlackTreeNode
    iterator_class: Iterator[Any] = BinaryTreeIterator

    def __init__(self) -> None:
        super().__init__()

    def _copy_subtree(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
        """Recursive auxiliary function to copy a subtree under the provided node.

        Args:
            node: Node whose subtree to copy.

        Returns:
            The root node of the copied subtree.
        """
        node_copy = node.copy()
        if node.left:
            node_copy.left = self._copy_subtree(node.left)
            node_copy.left.parent = node_copy
        if node.right:
            node_copy.right = self._copy_subtree(node.right)
            node_copy.right.parent = node_copy

        return node_copy

    def _traverse_to_root_and_update(self, node: RedBlackTreeNode) -> None:
        """Update all nodes on the path to the root.

        Args:
            node: Node for starting the traversal to the root.
        """
        node.update()

        if node.parent:
            self._traverse_to_root_and_update(node.parent)

    def _replace_child(
        self,
        parent: RedBlackTreeNode | None,
        old_child: RedBlackTreeNode,
        new_child: RedBlackTreeNode | None,
    ) -> None:
        """Replace the child of a parent node.

        Args:
            parent: The parent node.
            old_child: The current child node to be replaced.
            new_child: The replacement node.
        """
        if not parent:
            self.root = new_child
        elif parent.right is old_child:
            parent.right = new_child
        elif parent.left is old_child:
            parent.left = new_child
        else:
            raise RuntimeError("node is not a child of the provided parent")

        if new_child:
            new_child.parent = parent

    def _is_black(self, node: RedBlackTreeNode | None) -> bool:
        """Check if the node is black or None (= a black NIL node).

        Args:
            node: Node to check.

        Returns:
            Whether the node is black.
        """
        return node is None or not node.is_red

    def _rotate_right(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
        """Perform a right rotation on the node.

        Args:
            node: The node on which to perform a right rotation.

        Returns:
            The former left child of the node which is its parent after the rotation.
        """
        parent = node.parent
        left_child = node.left

        node.left = left_child.right
        if node.left:
            node.left.parent = node

        left_child.right = node
        node.parent = left_child
        self._replace_child(parent, node, left_child)

        # node may no longer be on the path from the inserted node to the root and, therefore, it
        # must be updated here already
        node.update()
        # left_child.update()

        return left_child

    def _rotate_left(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
        """Perform a left rotation on the node.

        Args:
            node: The node on which to perform a left rotation.

        Returns:
            The former right child of the node which is its parent after the rotation.
        """
        parent = node.parent
        right_child = node.right

        node.right = right_child.left
        if node.right:
            node.right.parent = node

        right_child.left = node
        node.parent = right_child
        self._replace_child(parent, node, right_child)

        # node may no longer be on the path from the inserted node to the root and, therefore, it
        # must be updated here already
        node.update()
        # right_child.update()

        return right_child

    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.

        Args:
            key: The key to be inserted.
        """
        node = self.root
        parent = None

        # search for the place to insert the new key
        while node:
            parent = node
            if key < node.key:
                node = node.left
            elif key > node.key:
                node = node.right
            else:
                raise ValueError(f"key {key} already exists")

        # insert the new node
        new_node = RedBlackTreeNode(key)
        new_node.is_red = True
        if not parent:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.parent = parent

        self._fix_after_insert(new_node)
        self._traverse_to_root_and_update(new_node)

    def _get_sibling(self, node: RedBlackTreeNode) -> RedBlackTreeNode | None:
        """Find the sibling of a node.

        Args:
            node: The node for which to find the sibling.

        Returns:
            The sibling node.
        """
        parent = node.parent

        if node is parent.left:
            return parent.right
        elif node is parent.right:
            return parent.left
        else:
            raise RuntimeError("node is not a child of its parent")

    def _get_uncle(self, parent: RedBlackTreeNode) -> RedBlackTreeNode | None:
        """Find the uncle starting from the parent node.

        Args:
            parent: The parent of the node for which to

        Returns:
            The uncle node.
        """
        grandparent = parent.parent

        if parent is grandparent.left:
            return grandparent.right
        elif parent is grandparent.right:
            return grandparent.left
        else:
            raise RuntimeError("node is not a child of its parent")

    def _fix_after_insert(self, node: RedBlackTreeNode) -> None:
        """Fix the red-black properties after insertion.

        Args:
            node: The node at which to start fixing the properties.
        """
        parent = node.parent

        # case 1: we have reached the root, which must be black
        if not parent:
            node.is_red = False
            return

        grandparent = parent.parent

        # case 2: the parent is black, nothing to do
        if not parent.is_red:
            return

        # now parent is red, therefore not the root and grandparent not None

        # get the uncle (which may be None)
        uncle = self._get_uncle(parent)

        # case 3: uncle is red -> recolor parent, grandparent and uncle
        if uncle and uncle.is_red:
            parent.is_red = False
            grandparent.is_red = True
            uncle.is_red = False

            # continue recursive at grandparent which is now red
            self._fix_after_insert(grandparent)

        # parent is left child of grandparent
        elif parent is grandparent.left:
            # case 4a: uncle is black and node is an "inner child"
            if node is parent.right:
                self._rotate_left(parent)
                # recoloring will be done in the following part
                parent = node

            # case 5a: uncle is black and node is an "outer child"
            self._rotate_right(grandparent)
            parent.is_red = False
            grandparent.is_red = True

        # parent is right child of grandparent
        else:
            # case 4b: uncle is black and node is an "inner child"
            if node is parent.left:
                self._rotate_right(parent)
                # recoloring will be done in the following part
                parent = node

            # case 5b: uncle is black and node is an "outer child"
            self._rotate_left(grandparent)
            parent.is_red = False
            grandparent.is_red = True

    def _delete_key(self, key: Any) -> None:
        """Delete a key.

        Args:
            key: The key to be deleted.
        """
        node = self._find(key)

        if not node:
            raise ValueError(f"key {key} not found")

        self._delete_node(node)

    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index and return it.

        Args:
            idx: The index of the element to be removed.
        """
        node = self._node_at_index(idx)
        self._temp_attributes = node.get_attributes()
        self._delete_node(node)

    def _delete_node(self, node: RedBlackTreeNode) -> None:
        """Delete a node from the tree.

        Args:
            node: The node to delete from the tree.
        """
        # node has zero or one child
        if not node.left or not node.right:
            # store the node at which to start to fix the red-black properties after deleting a node
            deleted_node_parent = node.parent
            moved_up_node = self._delete_node_with_zero_or_one_child(node)
            deleted_node_is_red = node.is_red

        # node has two children
        else:
            # find smallest node of right subtree ("inorder successor" of current node)
            inorder_successor = self._smallest_in_subtree(node.right)

            # copy inorder successor's data to current node (keep its color)
            inorder_successor.copy_attributes_to_node(node)

            # delete inorder successor instead (has at most one child)
            deleted_node_parent = inorder_successor.parent
            moved_up_node = self._delete_node_with_zero_or_one_child(inorder_successor)
            deleted_node_is_red = inorder_successor.is_red

        # we have to repair the red-black properties if the deleted node is red
        if not deleted_node_is_red:
            self._fix_after_delete(moved_up_node)

        if moved_up_node:
            self._traverse_to_root_and_update(moved_up_node)
        else:
            self._traverse_to_root_and_update(deleted_node_parent)

        # remove the temporary NIL node
        if isinstance(moved_up_node, TemporyNilNode):
            self._replace_child(moved_up_node.parent, moved_up_node, None)

    def _delete_node_with_zero_or_one_child(
        self,
        node: RedBlackTreeNode,
    ) -> RedBlackTreeNode | None:
        """Delete with zero or one child from the tree.

        Args:
            node: The node to delete from the tree.

        Returns:
            The node that was moved up or None if no node was moved up.
        """
        if node.left:
            self._replace_child(node.parent, node, node.left)
            return node.left

        if node.right:
            self._replace_child(node.parent, node, node.right)
            return node.right

        # node has no children
        new_child = None if node.is_red else TemporyNilNode()
        self._replace_child(node.parent, node, new_child)

        return new_child

    def _fix_after_delete(self, node: RedBlackTreeNode) -> None:
        """Fix the red-black properties after deletion of a node.

        Args:
            node: The node at which to fix the red-black properties.
        """
        # case 1: node is the root, end of recursion
        if node is self.root:
            return

        sibling = self._get_sibling(node)

        # case 2: red sibling
        if sibling.is_red:
            self._handle_red_sibling(node, sibling)
            # get new sibling for fall-through to cases 3-6
            sibling = self._get_sibling(node)

        # cases 3 and 4: black sibling with two black children
        if self._is_black(sibling.left) and self._is_black(sibling.right):
            sibling.is_red = True

            # case 3: black sibling with two black children and red parent
            if node.parent.is_red:
                node.parent.is_red = False
            # case 4: black sibling with two black children and black parent
            self._fix_after_delete(node.parent)

        # cases 5 and 6: black sibling with at least one red child
        else:
            self._handle_black_sibling_with_red_child(node, sibling)

    def _handle_red_sibling(self, node: RedBlackTreeNode, sibling: RedBlackTreeNode) -> None:
        """Fix the red-black properties after deletion of a node with a red sibling.

        Args:
            node: The node at which to fix the red-black properties.
            sibling: Its red sibling.
        """
        # recolor
        sibling.is_red = False
        node.parent.is_red = True

        # rotate
        if node is node.parent.left:
            self._rotate_left(node.parent)
        else:
            self._rotate_right(node.parent)

    def _handle_black_sibling_with_red_child(
        self,
        node: RedBlackTreeNode,
        sibling: RedBlackTreeNode,
    ) -> None:
        """Fix the red-black properties after deletion of a node with black sibling and red nephew.

        Args:
            node: The node at which to fix the red-black properties.
            sibling: Its black sibling with at least one red child.
        """
        node_is_left_child = node is node.parent.left

        # case 5: black sibling with at least one red child and outer nephew is black
        # --> recolor sibling and its child, and rotate around sibling
        if node_is_left_child and self._is_black(sibling.right):
            sibling.left.is_red = False
            sibling.is_red = True
            self._rotate_right(sibling)
            sibling = node.parent.right
        elif not node_is_left_child and self._is_black(sibling.left):
            sibling.right.is_red = False
            sibling.is_red = True
            self._rotate_left(sibling)
            sibling = node.parent.left

        # fall-through to case 6

        # case 6: black sibling with at least one red child and outer nephew is red
        # --> recolor sibling + parent + sibling's child, and rotate around parent
        sibling.is_red = node.parent.is_red
        node.parent.is_red = False
        if node_is_left_child:
            sibling.right.is_red = False
            self._rotate_left(node.parent)
        else:
            sibling.left.is_red = False
            self._rotate_right(node.parent)
