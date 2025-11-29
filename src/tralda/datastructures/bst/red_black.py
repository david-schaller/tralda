"""Red-black tree implementation.

Balanced binary search tree implementation of a set (TreeSet) and dictionary (TreeDict).

References:
    [1] https://en.wikipedia.org/wiki/Red-black_tree
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
        "_is_red",
        "black_height",
    )

    def __init__(self, key: Any) -> None:
        """Initialize the red-black tree node.

        Args:
            key: The key/label of the node. Keys must be unique within a binary search tree.
        """
        super().__init__(key)

        self.parent: RedBlackTreeNode | None = None
        self._is_red: bool = False

        # the black height is the number of black vertices on the path from the node to any of the
        # leaves below it where NIL node (= None) count as black nodes (hence, we intialize to 2)
        self.black_height: int = 2

    @property
    def direction(self) -> int:
        """Return whether the node is its parent's left or right child.

        Returns:
            The direction w.r.t. the parent (0 if it is the left child, 1 otherwise).

        Raises:
            RuntimeError: If the node has no parent.
        """
        try:
            return 0 if self is self.parent.left else 1
        except AttributeError as e:
            raise RuntimeError(f"node {self} has no parent") from e

    @property
    def is_red(self) -> bool:
        """Return whether the node is red."""
        return self._is_red

    @property
    def is_black(self) -> bool:
        """Return whether the node is black."""
        return not self._is_red

    def child(self, direction: int) -> RedBlackTreeNode | None:
        """Return the left (direction = 0) or right (direction = 1) child.

        Args:
            direction: The direction.

        Returns:
            The child at to the provided direction.
        """
        if direction == 0:
            return self.left
        else:
            return self.right

    def set_child(self, direction: int, node: RedBlackTreeNode | None) -> None:
        """Set the left (direction = 0) or right (direction = 1) child.

        Args:
            direction: The direction.
            node: The new child to be set.
        """
        if direction == 0:
            self.left = node
        else:
            self.right = node

    def turn_red(self) -> None:
        """Change the color to red (if necessary).

        If the node was black before, then the black height is reduced by 1.
        """
        if not self._is_red:
            self.black_height -= 1

        self._is_red = True

    def turn_black(self) -> None:
        """Change the color to black (if necessary).

        If the node was red before, then the black height is increased by 1.
        """
        if self._is_red:
            self.black_height += 1

        self._is_red = False

    def update(self) -> None:
        """Update height, size, and black height of (the subtree under) the node."""
        super().update()

        # update the black height (checking one child is enough)
        black_height_children = self.left.black_height if self.left else 1
        self.black_height = black_height_children + int(self.is_black)

    def copy(self) -> RedBlackTreeNode:
        """A copy of this node.

        Returns:
            A copy of this node.
        """
        copy = super().copy()
        copy._is_red = self._is_red
        copy.black_height = self.black_height

        return copy


class TreeSet(BaseBinarySearchTree):
    """Red-black tree."""

    node_class = RedBlackTreeNode
    iterator_class: Iterator[Any] = BinaryTreeIterator

    def __init__(self) -> None:
        """Constructor of the TreeSet class (red-black tree implementation)."""
        super().__init__()

    def check_integrity(self, verbose: bool = False) -> bool:
        """Integrity check of the tree.

        Checks whether the size and heigth is correct in all subtrees. Moreover, the red-black
        properties are checked. Intended for debugging and testing purpose.

        Args:
            verbose: If True print where the integrity check has failed.

        Returns:
            Whether all integrity checks have been passed.
        """
        if not super().check_integrity(verbose=verbose):
            return False

        # check the red-black properties
        for i, node in enumerate(self._postorder_traversal()):
            # (1) all patch from the root to a leaf (counting NIL nodes as black leaves) have the
            # same number of black nodes
            black_height_left = node.left.black_height if node.left else 1
            black_height_right = node.right.black_height if node.right else 1

            if black_height_left != black_height_right:
                if verbose:
                    print(
                        f"children of node {node} have different black height "
                        f"({black_height_left} vs. {black_height_right}, traversal step: {i})"
                    )
                return False

            expected_black_height = black_height_left + int(node.is_black)

            if expected_black_height != node.black_height:
                if verbose:
                    print(
                        f"node {node} should black height {expected_black_height} but has "
                        f"{node.black_height} (traversal step: {i})"
                    )
                return False

            # (2) a red node does not have a red child
            if node.is_red and node.parent and node.parent.is_red:
                if verbose:
                    print(f"node {node.parent} is red and has a red child {node}")
                return False

        return True

    @classmethod
    def join(cls, tree_left: TreeSet, tree_right: TreeSet, key: Any | None = None) -> TreeSet:
        """Join function for two red-black trees.

        Join two red-black trees T_L and T_R with x < key < y for all x in T_L and y in T_R. The
        resulting tree will contain all keys in T_L and T_R and additionally the new provided key.
        If no key is provided, a dummy key will be added during the join operation, which will be
        removed in the end, keeping the total O(log(n)) time complexity.

        NOTE: The input red-black tree instances should no longer be used afterwards.

        Args:
            tree_left: The tree T_L with the smaller elements.
            tree_right: The tree T_R with the larger elements.
            key: The new key to be added. If None, a dummy node will be temporarily inserted and
                removed in the end.

        Raises:
            TypeError: If one of the two trees is not an instance of the correct red-black tree
                type.

        Returns:
            The joined tree.
        """
        if not isinstance(tree_left, cls):
            raise TypeError(f"tree_left must be of type {cls}, but is {type(tree_left)}")
        if not isinstance(tree_right, cls):
            raise TypeError(f"tree_right must be of type {cls}, but is {type(tree_right)}")

        new_node = cls.node_class(0) if key is None else cls.node_class(key)

        # both trees are empty --> re-use instance tree_left
        if not tree_left and not tree_right:
            tree_left.root = new_node
            return tree_left
        # only tree_right is non-empty --> insert new_node as new smallest element
        elif not tree_left:
            parent = tree_right._smallest_in_subtree(tree_right.root)
            tree_right._insert_node(new_node, parent, 0)
            return tree_right
        # only tree_left is non-empty --> insert new_node as new largest element
        elif not tree_right:
            parent = tree_left._largest_in_subtree(tree_left.root)
            tree_left._insert_node(new_node, parent, 1)
            return tree_left

        # create a new tree if both trees are non-empty
        tree = cls()
        tree.root = tree._join(tree_left.root, tree_right.root, new_node)

        # update all nodes on the path from the new node to the root
        tree._traverse_to_root_and_update(new_node)

        # if no key was provided, remove the dummy node
        if key is None:
            tree._delete_node(new_node)

        return tree

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
        while node:
            node.update()
            node = node.parent

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

    def _rotate_subtree(self, node: RedBlackTreeNode, direction: int) -> RedBlackTreeNode:
        """Rotate the node and and one of its children.

        Args:
            node: The node that will become a child of one of its children.
            direction: The direction of the rotation (0 = left rotation, 1 = right rotation).

        Returns:
            The child of node that is now its parent, i.e., the new root of the subtree.
        """
        parent = node.parent
        new_root = node.child(1 - direction)
        new_child = new_root.child(direction)

        node.set_child(1 - direction, new_child)

        if new_child:
            new_child.parent = node

        new_root.set_child(direction, node)
        node.parent = new_root
        self._replace_child(parent, node, new_root)

        # node may no longer be on the path from the inserted / moved up node to the root and,
        # therefore, it must be updated here already
        node.update()

        return new_root

    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.

        Args:
            key: The key to be inserted.

        Raises:
            KeyError: If the key already exists.
        """
        node = self.root
        parent = None
        direction = 0

        # search for the place to insert the new key
        while node:
            parent = node
            if key < node.key:
                node = node.left
                direction = 0
            elif key > node.key:
                node = node.right
                direction = 1
            else:
                raise KeyError(f"key {key} already exists")

        # create and insert the new node, rebalance, update nodes on the way to the root
        new_node = self.node_class(key)
        self._insert_node(new_node, parent, direction)
        self._traverse_to_root_and_update(new_node)

    def _insert_node(
        self,
        node: RedBlackTreeNode,
        parent: RedBlackTreeNode | None,
        direction: int,
    ) -> None:
        """Insert a node at the specified location.

        Args:
            node: The new node to insert.
            parent: The parent below which to insert the new node.
            direction: The direction where to insert the new node below parent (0 = left, 1 =
                right).

        References:
            .. [1] https://en.wikipedia.org/wiki/Red-black_tree#Insertion
        """
        # cases follow the implementation in https://en.wikipedia.org/wiki/Red-black_tree#Insertion
        # (accessed Nov 29, 2025)

        node.turn_red()
        node.parent = parent

        # tree was empty before --> new node becoms the root
        if not parent:
            self.root = node
            return

        parent.set_child(direction, node)

        # rebalance the tree
        while parent:
            # case 1
            if parent.is_black:
                return

            grandparent = parent.parent

            if not grandparent:
                # case 4
                parent.turn_black()
                return

            direction = parent.direction
            uncle = grandparent.child(1 - direction)

            if not uncle or uncle.is_black:
                if node is parent.child(1 - direction):
                    # case 5
                    self._rotate_subtree(parent, direction)
                    node = parent
                    parent = grandparent.child(direction)

                # case 6
                self._rotate_subtree(grandparent, 1 - direction)
                parent.turn_black()
                grandparent.turn_red()
                return

            # case 2
            parent.turn_black()
            uncle.turn_black()
            grandparent.turn_red()
            node = grandparent

            parent = node.parent

    def _delete_key(self, key: Any) -> None:
        """Delete a key.

        Args:
            key: The key to be deleted.
        """
        node = self._find(key)

        if not node:
            raise KeyError(f"key {key} not found")

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

        References:
            .. [1] https://en.wikipedia.org/wiki/Red-black_tree#Removal
        """
        # node to delete has 2 children
        if node.left and node.right:
            # find smallest node of right subtree ("inorder successor" of current node)
            inorder_successor = self._smallest_in_subtree(node.right)

            # copy inorder successor's data to current node (keep its color)
            inorder_successor.copy_attributes_to_node(node)

            # continue with deleting the inorder sucessor instead, it can only have a right child
            # or no child at all
            node = inorder_successor

        parent = node.parent

        # node to delete has 1 child (single child must be red, deleted node must be black)
        if node.left or node.right:
            self._delete_node_with_one_child(node)
        # node to delete has no children and is the root
        elif not parent:
            self.root = None  # the tree is now empty
            return
        # node to delete has no children and is red --> simply delete the leaf
        elif node.is_red:
            self._replace_child(parent, node, None)
        # node to delete has no children and black --> deleting creates imbalance and requires
        # rebalancing
        else:
            self._delete_black_leaf(node)

        self._traverse_to_root_and_update(parent)

    def _delete_node_with_one_child(self, node: RedBlackTreeNode) -> None:
        """Delete with exactly one child from the tree.

        Args:
            node: The node to delete from the tree.
        """
        if node.left:
            self._replace_child(node.parent, node, node.left)
            node.left.turn_black()
        else:
            self._replace_child(node.parent, node, node.right)
            node.right.turn_black()

    def _delete_black_leaf(self, node: RedBlackTreeNode) -> None:
        """Remove a black leaf from the tree.

        Args:
            node: The black leaf node to be removed.
        """
        # cases follow the implementation in https://en.wikipedia.org/wiki/Red–black_tree#Removal
        # (accessed Nov 29, 2025)

        parent = node.parent
        sibling = None
        close_nephew = None
        distant_nephew = None
        direction = node.direction
        skip_case_5 = False

        parent.set_child(direction, None)

        while True:
            sibling = parent.child(1 - direction)
            distant_nephew = sibling.child(1 - direction)
            close_nephew = sibling.child(direction)

            if sibling.is_red:
                # case 3
                self._rotate_subtree(parent, direction)
                parent.turn_red()
                sibling.turn_black()
                sibling = close_nephew

                distant_nephew = sibling.child(1 - direction)
                if distant_nephew and distant_nephew.is_red:
                    skip_case_5 = True
                    break  # go to case 6
                close_nephew = sibling.child(direction)

                if close_nephew and close_nephew.is_red:
                    break  # go to case 5

                # case 4
                sibling.turn_red()
                parent.turn_black()
                return

            if distant_nephew and distant_nephew.is_red:
                skip_case_5 = True
                break  # go to case 6

            if close_nephew and close_nephew.is_red:
                break  # go to case 5

            if not parent:
                # case 1
                return

            if parent.is_red:
                # case 4
                sibling.turn_red()
                parent.turn_black()
                return

            # case 2
            sibling.turn_red()
            node = parent

            parent = node.parent
            if not parent:
                break
            direction = node.direction

        if not skip_case_5:
            # case 5
            self._rotate_subtree(sibling, 1 - direction)
            sibling.turn_red()
            close_nephew.turn_black()
            distant_nephew = sibling
            sibling = close_nephew

        # case 6
        self._rotate_subtree(parent, direction)
        if parent.is_red:
            sibling.turn_red()
        else:
            sibling.turn_black()
        parent.turn_black()
        distant_nephew.turn_black()

    def _join_subtrees(
        self,
        parent: RedBlackTreeNode,
        left_child: RedBlackTreeNode,
        right_child: RedBlackTreeNode,
    ) -> None:
        """Attach two nodes to a common parent.

        Args:
            parent: The parent node.
            left_child: The new left child.
            right_child: The new right child.
        """
        parent.left = left_child
        left_child.parent = parent
        parent.right = right_child
        right_child = parent

    def _join(
        self,
        root_left: RedBlackTreeNode,
        root_right: RedBlackTreeNode,
        new_node: RedBlackTreeNode,
    ) -> RedBlackTreeNode:
        """Join two non-empty red-black trees.

        Args:
            root_left: The root of the left tree (containing the smaller elements).
            root_right: The root of the right tree (containing the larger elements).
            new_node: The new node used to join the trees.

        Returns:
            The root of the joined tree.
        """
        if root_left.black_height > root_right.black_height:
            new_root = self._join_right(root_left, root_right, new_node)
            new_root.turn_black()
        elif root_left.black_height < root_right.black_height:
            new_root = self._join_left(root_left, root_right, new_node)
            new_root.turn_black()
        else:
            new_root = new_node
            self._join_subtrees(new_root, root_left, root_right)

        return new_root

    def _join_right(
        self,
        root_left: RedBlackTreeNode,
        root_right: RedBlackTreeNode,
        new_node: RedBlackTreeNode,
    ) -> RedBlackTreeNode:
        """Recursively move done the right spine of the left tree to join the trees.

        Args:
            root_left: The root of the left tree (containing the smaller elements).
            root_right: The root of the right tree (containing the larger elements).
            new_node: The new node used to join the trees.

        Returns:
            The root of the joined tree.
        """
        if root_left.is_black and root_left.black_height == root_right.black_height:
            new_node.turn_red()
            self._join_subtrees(new_node, root_left, root_right)
            return new_node

        root_left.right = self._join_right(root_left.right, root_right, new_node)
        root_left.right.parent = root_left

        if root_left.is_black and root_left.right.is_red and root_left.right.right.is_red:
            root_left.right.right.turn_black()
            return self._rotate_subtree(root_left, 0)  # left rotation

        return root_left

    def _join_left(
        self,
        root_left: RedBlackTreeNode,
        root_right: RedBlackTreeNode,
        new_node: RedBlackTreeNode,
    ) -> RedBlackTreeNode:
        """Recursively move done the left spine of the right tree to join the trees.

        Args:
            root_left: The root of the left tree (containing the smaller elements).
            root_right: The root of the right tree (containing the larger elements).
            new_node: The new node used to join the trees.

        Returns:
            The root of the joined tree.
        """
        if root_right.is_black and root_left.black_height == root_right.black_height:
            new_node.turn_red()
            self._join_subtrees(new_node, root_left, root_right)
            return new_node

        root_right.left = self._join_left(root_left, root_right.left, new_node)
        root_right.left.parent = root_right

        if root_right.is_black and root_right.left.is_red and root_right.left.left.is_red:
            root_right.left.left.turn_black()
            return self._rotate_subtree(root_right, 1)  # right rotation

        return root_right
