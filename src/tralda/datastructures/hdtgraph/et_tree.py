"""Euler tour tree as AVL-tree.

Implementation of an Euler tour tree for dynamic graph algorithm.

References:
    .. [1] Monika Rauch Henzinger, Valerie King. Randomized fully dynamic graph algorithms with
           polylogarithmic time per operation. J. ACM 46(4). July 1999. 502-536.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any
from typing import Iterator

from tralda.datastructures.bst.red_black import RedBlackTreeNode
from tralda.datastructures.bst.red_black import TreeSet
from tralda.datastructures.bst.base import BinaryTreeNodeIterator
from tralda.utils.graph_tools import sort_edge


class ETTreeNode(RedBlackTreeNode):
    __slots__ = (
        "active",
        "_num_active_nodes",
        "ett",
    )

    def __init__(self, key: Any, active: bool = False):
        """Constructor of ETTreeNode.

        Args:
            key: The key to be stored.
            active: Whether the node corresponds to the active occurrence of the key in the tree.
        """
        super().__init__(key)

        self.active = active
        self._num_active_nodes = int(active)

        # root nodes must be able to store a reference to the ETTree instance
        self.ett = None

    def get_root(self) -> ETTreeNode:
        """Walk to the root and return it.

        Returns:
            The root node under which this node is located.
        """
        current = self
        while current.parent:
            current = current.parent

        return current

    def update(self) -> None:
        """Update height, size, black height, and number of active occurrences in subtree."""
        super().update()

        self._num_active_nodes = int(self.active)

        if self.left:
            self._num_active_nodes += self.left._num_active_nodes
        if self.right:
            self._num_active_nodes += self.right._num_active_nodes

    def find_inorder_predecessor(self) -> ETTreeNode | None:
        """Find the inorder predecessor of the node (if existent).

        Returns:
            The inorder predecessor or None if the node is the smallest node in the tree.
        """
        if self.left:
            return self.left.largest_in_subtree()

        node = self
        while node.parent:
            if node is node.parent.right:
                return node.parent

            node = node.parent

        return None

    def find_inorder_successor(self) -> ETTreeNode | None:
        """Find the inorder successor of the node (if existent).

        Returns:
            The inorder successor or None if the node is the largest node in the tree.
        """
        if self.right:
            return self.right.smallest_in_subtree()

        node = self
        while node.parent:
            if node is node.parent.left:
                return node.parent

            node = node.parent

        return None

    def is_smaller(self, other: ETTreeNode) -> bool:
        """Determine whether the node appears before the other node in the Euler tour.

        Args:
            other: Another node in the Euler tour.

        Returns:
            True the node appears before the other node in the Euler tour, False otherwise.

        Raises:
            ValueError: If the nodes are not in the same tree (i.e., they are not connected to the
            same root node).
        """
        if self is other:
            return False

        path1 = self._path_to_root()
        path2 = other._path_to_root()

        if path1[-1] is not path2[-1]:
            raise ValueError(f"nodes {self} and {other} are not in the same Euler Tour tree")

        for i in range(-2, -min(len(path1), len(path2)) - 1, -1):
            if path1[i] is not path2[i]:
                return path1[i] is path1[i].parent.left

        if len(path1) < len(path2):
            return path2[-len(path1) - 1] is not path1[0].left
        else:
            return path1[-len(path2) - 1] is path2[0].left

    def _path_to_root(self) -> list[ETTreeNode]:
        """Get the path from this node to the root.

        Returns:
            A list containing all node instances in a traversal from this node to the root.
        """
        path = [self]
        current = self.parent

        while current:
            path.append(current)
            current = current.parent

        return path


class EdgeOccurrences:
    """Class for storing and maintaining the occurrences a1, b1, b2, a2 that represent an edge ab.

    Every edge is traversed two times in an Euler tour. Hence, an edge is associated with four (or
    three, if an endpoint is a leaf) occurrences of its endpoints. This class is used to store and
    maintain these occurrence for a given edge.
    For the stored occurrences a1, b1, b2, and a2, one of the following orders should hold in the
    Euler tour:
    - a1 < b1 <= b2 < a2
    - b2 < a2 <= a1 < b1
    """

    __slots__ = ["_a", "_b", "_a1", "_a2", "_b1", "_b2"]

    def __init__(self, a: Any, b: Any) -> None:
        """Constructor of the EdgeOccurrences class.

        Args:
            a: The first endpoint of the edge.
            b: The second endpoint of the edge.
        """
        # the keys (endpoints of the edge ab)
        self._a = a
        self._b = b

        # occurrence of the edge where a appears before b
        self._a1: ETTreeNode | None = None
        self._b1: ETTreeNode | None = None

        # occurrence of the edge where b appears before a
        self._b2: ETTreeNode | None = None
        self._a2: ETTreeNode | None = None

    def update(self, occurrence1: ETTreeNode, occurrence2: ETTreeNode) -> None:
        """Update one pair of consecutive occurrence.

        Args:
            occurrence1: The occurrence that comes first in the Euler Tour.
            occurrence2: The successor of occurrence1 in the Euler Tour.
        """
        if occurrence1.key == self._a:
            self._a1 = occurrence1
            self._b1 = occurrence2
        else:
            self._b2 = occurrence1
            self._a2 = occurrence2

    def get_sorted_occurrences(self) -> tuple[ETTreeNode, ETTreeNode, ETTreeNode, ETTreeNode]:
        """Return the occurrences such that a1 < b1 <= b2 < a2.

        Returns:
            The sorted occurrences.
        """
        if self._b1 is self._b2:
            return self._a1, self._b1, self._b2, self._a2
        elif self._a2 is self._a1:
            return self._b2, self._a2, self._a1, self._b1
        elif self._a1.is_smaller(self._a2):
            return self._a1, self._b1, self._b2, self._a2
        else:
            return self._b2, self._a2, self._a1, self._b1


class ETTree(TreeSet):
    """Euler Tour (ET) tree datastructure."""

    __slots__ = ()

    node_class = ETTreeNode
    iterator_class: Iterator[Any] = BinaryTreeNodeIterator

    def __init__(self) -> None:
        """Constructor of the ET tree datastructure."""
        super().__init__()

    @property
    def num_active_occurrences(self):
        """The number of active occurrences.

        Returns:
            The number of different items (= active occurrences) in the ET tree.
        """
        if self.root:
            return self.root._num_active_nodes
        else:
            return 0

    @classmethod
    def join(
        cls,
        tree_left: ETTree,
        tree_right: ETTree,
        node2active_occurrence: dict[Any, ETTreeNode],
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> TreeSet:
        """Join function for two Euler Tree trees.

        NOTE: The input tree instances should no longer be used afterwards.

        Args:
            tree_left: The tree representing the first part of the sequence.
            tree_right: The tree representing the second part of the sequence.
            node2active_occurrence: Dictionary mapping keys to active occurrences in this and
                possible other Euler Tour trees.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            The tree representing the joined parts of the sequence.
        """
        dummy_node = cls.node_class(0)

        tree = cls._join(tree_left, tree_right, dummy_node)

        node_to_remove = dummy_node
        if node_to_remove.left and node_to_remove.right:
            node_to_remove = tree._replace_node_to_remove(
                node_to_remove, node2active_occurrence, edge2occurrences
            )

        tree._delete_node(node_to_remove)

        return tree

    def delete_edge(
        self,
        edge: EdgeOccurrences,
        node2active_occurrence: dict[Any, ETTreeNode],
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> tuple[ETTree, ETTree]:
        """Delete an edge in the tree that this ET tree represents.

        Args:
            edge: The occurrences that represent the edge to delete.
            node2active_occurrence: Dictionary mapping keys to active occurrences in this and
                possible other Euler Tour trees.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            The two ET trees after the edge is deleted.
        """
        # occurrences sorted such that a1 < b1 <= b2 < a2
        a1, _, _, a2 = edge.get_sorted_occurrences()

        # since a2 is discarded, we may have to make a1 active
        if a2.active:
            a1.active = True
            node2active_occurrence[a1.key] = a1
            self._traverse_to_root_and_update(a1)

        # cut between a1 and b1 (removing a1 for now)
        tree_a_left, tree_b = self.split_at_node(a1)

        # cut between b2 and a2 (the node a2 is discarded)
        tree_b, tree_a_right = self.split_at_node(a2)

        # join the two parts corresponding to the component that contains a
        tree_a = ETTree._join(tree_a_left, tree_a_right, a1)
        a1_successor = a1.find_inorder_successor()
        if a1_successor:
            self._update_edge_occurrences(edge2occurrences, a1, a1_successor)

        # the roots store references to the ETTree instances
        tree_a.root.ett = tree_a
        tree_b.root.ett = tree_b

        return tree_a, tree_b

    def add_right_child_and_rebalance(
        self,
        parent: ETTreeNode | None,
        child: ETTreeNode,
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> None:
        """Add a right child to a node and rebalance the tree.

        If the provided parent node is None, the provided child node becomes the root of the tree.

        Args:
            parent: The parent node which must not have a right child. It can be None, in which
                case, the provided child node becomes the root.
            child: The node to attach as a child.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.
        """
        self._insert_node(child, parent, 1)

        if parent is not None:
            self._update_edge_occurrences(edge2occurrences, parent, child)

    def reroot(
        self,
        node: ETTreeNode,
        node2active_occurrence: dict[Any, ETTreeNode],
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> ETTree:
        """Change the root of the tree that this ET tree represents.

        Args:
            node: An occurrence of the new root.
            node2active_occurrence: Dictionary mapping keys to active occurrences in this and
                possible other Euler Tour trees.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            The re-rooted tree.
        """
        # do nothing if the tree is empty or only consists of one item
        if len(self) <= 1:
            return self

        tree_1, tree_2 = self.split_at_node(node, keep_node_right=True)

        if not tree_1:
            return tree_2

        node_to_remove = tree_1.root.smallest_in_subtree()
        last_in_tree2 = tree_2.root.largest_in_subtree()

        # since the node gets removed, we may need to set a new active node; the last occurrence
        # in tree2 has the same key
        if node_to_remove.active:
            last_in_tree2.active = True
            node2active_occurrence[last_in_tree2.key] = last_in_tree2
            tree_2._traverse_to_root_and_update(last_in_tree2)

        # remove the first occurrence of the old root (or the replacement node); as it is the
        # smallest node in tree_1, it is guaranteed to have at most one child, so this ETTreeNode
        # instance will be removed and we do not need to find a replacement node
        tree_1._delete_node(node_to_remove)

        # tack the first part to the end of the second part
        if tree_1:
            self._update_edge_occurrences(
                edge2occurrences, last_in_tree2, tree_1.root.smallest_in_subtree()
            )
        tree = ETTree.join(tree_2, tree_1, node2active_occurrence, edge2occurrences)

        # finally add a new occurrence of the new root at the end
        new_node = ETTreeNode(node.key, active=False)
        parent = tree.root.largest_in_subtree()
        tree.add_right_child_and_rebalance(parent, new_node, edge2occurrences)

        # the root stores a reference to the ETTree instance
        tree.root.ett = tree

        return tree

    def join_by_edge(
        self,
        node_a: ETTreeNode,
        node_b: ETTreeNode,
        tree_b: ETTree,
        node2active_occurrence: dict[Any, ETTreeNode],
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> ETTree:
        """Join two tree represented as Euler Tour trees by an edge ab.

        Args:
            node_a: An occurrence of the node a in the edge ab. Must be located in the Euler Tour
                tree.
            node_b: An occurrence of the node b in the edge ab.
            tree_b: The ETTree instance in which b is located.
            node2active_occurrence: Dictionary mapping keys to active occurrences in this Euler Tour
                tree, tree_b, and possible other Euler Tour trees.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            A new ETTree instance representing the new tree that is given by joining the two trees
            by the edge ab.
        """
        # reroot T' at b
        if node_b is not tree_b.root.smallest_in_subtree():
            tree_b = tree_b.reroot(node_b, node2active_occurrence, edge2occurrences)

        # get the last occurrence of b, which is needed for the edge representation
        node_b2 = tree_b.root.largest_in_subtree()

        # create a new occurrence of a
        node_a2 = ETTreeNode(node_a.key, active=False)

        # splice the sequence ET(T') and the new occurrence of a immediately after node_a

        # step 1: split the first sequence
        tree_left, tree_right = self.split_at_node(node_a)

        # step 2: update edge occurrences: new node_a2 to its successor
        if tree_right:
            self._update_edge_occurrences(
                edge2occurrences, node_a2, tree_right.root.smallest_in_subtree()
            )

        # step 3: join the sequence parts
        tree_left = ETTree._join(tree_left, tree_b, node_a)
        tree = ETTree._join(tree_left, tree_right, node_a2)

        # the root stores a reference to the ETTree instance
        tree.root.ett = tree

        self._update_edge_occurrences(edge2occurrences, node_a, node_b)
        self._update_edge_occurrences(edge2occurrences, node_b2, node_a2)

        return tree

    def check_integrity(
        self,
        verbose: bool = False,
        check_ett_properties: bool = True,
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences] | None = None,
    ) -> bool:
        """Integrity check of the tree.

        Checks whether the size and height is correct in all subtrees. Moreover, the red-black
        properties are checked. Intended for debugging and testing purpose.

        Args:
            verbose: If True print where the integrity check has failed.
            check_ett_properties: Whether to check certain ETT-specific properties. If False, only
                the red-black tree properties are checked.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            Whether all integrity checks have been passed.
        """
        if not super().check_integrity(verbose=verbose):
            return False

        # stop here if only red-black tree properties shall be checked
        if not check_ett_properties:
            return True

        # check that the number of active occurrences is correct in each subtree
        node2num_active = defaultdict(int)
        for i, node in enumerate(self._postorder_traversal()):
            node2num_active[node.key] += int(node.active)

            num_active_nodes = int(node.active)

            if node.left:
                num_active_nodes += node.left._num_active_nodes
            if node.right:
                num_active_nodes += node.right._num_active_nodes

            if num_active_nodes != node._num_active_nodes:
                if verbose:
                    print(
                        f"node {node} does not have the expected number of active occurrences "
                        f"({num_active_nodes} vs. {node._num_active_nodes}, traversal step: {i})"
                    )
                return False

        # check that all items have exactly one active occurrence
        if not all(count == 1 for count in node2num_active.values()):
            print(self.to_newick())
            for key, count in node2num_active.items():
                if verbose and count != 1:
                    print(f"key {key} has {count} active occurrences")
            return False

        if edge2occurrences is None:
            return True

        # check if all edges in the Euler Tour are correctly contained in edge2occurrences
        predecessor = None
        for node in self:
            if not predecessor:
                predecessor = node
                continue

            edge_occurrences = edge2occurrences.get(
                (predecessor.key, node.key), edge2occurrences.get((node.key, predecessor.key))
            )
            occurrence_list = (
                edge_occurrences._a1,
                edge_occurrences._a2,
                edge_occurrences._b1,
                edge_occurrences._b2,
            )

            for x in (predecessor, node):
                if x in occurrence_list:
                    continue
                if verbose:
                    print(f"{x} ({id(x)}) is not among the edge occurrences:")
                    for y in occurrence_list:
                        print(f"  - {y} ({id(y)})")
                return False

            predecessor = node

        return True

    def to_newick(self) -> str:
        """Return a Newick representation of the tree.

        Intended for testing purpose.

        Returns:
            A Newick representation of the tree.
        """

        def construct_newick(node):
            if not (node.left or node.right):
                return str(node.key)

            if node.left and node.right:
                s = "(" + construct_newick(node.left) + "," + construct_newick(node.right) + ")"
            elif node.left:
                s = "(" + construct_newick(node.left) + ",-)"
            elif node.right:
                s = "(-," + construct_newick(node.right) + ")"
            else:
                s = ""
            return s + str(node.key)

        if not self.root:
            return "<empty tree>"

        return construct_newick(self.root)

    def _replace_node_to_remove(
        self,
        node: ETTreeNode,
        node2active_occurrence: dict[Any, ETTreeNode],
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
    ) -> ETTreeNode:
        """Find the inorder successor of a node to be removed instead.

        The inorder successor is guaranteed to have at most one child (the right child). The
        attributes of this node will be copied into the original node to remove.

        Args:
            node: The node whose content shall be removed.
            node2active_occurrence: Dictionary mapping keys to active occurrences in this and
                possible other Euler Tour trees.
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.

        Returns:
            The instance of the inorder successor node which can now be removed instead.
        """
        # find smallest node of right subtree (inorder successor of current node)
        successor = node.right.smallest_in_subtree()

        # copy inorder successor's data to current node (but keep its color)
        successor.copy_attributes_to_node(node)
        node.active = successor.active
        self._traverse_to_root_and_update(node)  # TODO: needed?

        # update the dictionary if needed
        if successor.active:
            node2active_occurrence[successor.key] = node

        # update occurrences in edge dictionary
        predecessor = node.find_inorder_predecessor()
        if predecessor is not None:
            self._update_edge_occurrences(edge2occurrences, predecessor, node)
        successor2 = successor.find_inorder_successor()
        if successor2 is not None:
            self._update_edge_occurrences(edge2occurrences, node, successor2)

        return successor

    def _update_edge_occurrences(
        self,
        edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences],
        node1: ETTreeNode,
        node2: ETTreeNode,
    ) -> None:
        """Given an Euler Tour tree node and its new successor, update the edge occurrences.

        Args:
            edge2occurrences: Dictionary mapping edges to corresponding EdgeOccurrences instances.
            node1: An Euler Tour tree node.
            node2: The new successor of node1.
        """
        edge = sort_edge(node1.key, node2.key)

        edge_occurrences = edge2occurrences.get(edge)

        if not edge_occurrences:
            edge_occurrences = EdgeOccurrences(*edge)
            edge2occurrences[edge] = edge_occurrences

        edge_occurrences.update(node1, node2)
