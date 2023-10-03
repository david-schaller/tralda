# -*- coding: utf-8 -*-

"""
Red-black tree implementation.
    
Balanced binary search tree implementation of a set (TreeSet) and
dictionary (TreeDict).
"""

from __future__ import annotations

from typing import Any, Iterator

from tralda.datastructures.binary_tree_base import BinaryNode
from tralda.datastructures.binary_tree_base import BaseBinarySearchTree
from tralda.datastructures.binary_tree_base import BinaryTreeIterator


__author__ = 'David Schaller'


class RedBlackTreeNode(BinaryNode):
    
    __slots__ = ('parent', 'is_red',)
    
    def __init__(self, key: Any) -> None:
        super().__init__(key)
        self.parent = None
        self.is_red = False
        raise NotImplementedError('red-black trees are not yet implemented')
 

class TreeSet(BaseBinarySearchTree):
    """Red-black tree."""
    
    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator
    
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError('red-black trees are not yet implemented')
    
    
    def rotate_right(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
        """Perform a right rotation on the node.
        
        Parameters
        ----------
        node: RedBlackTreeNode
            The node on which to perform a right rotation.
        
        Returns
        -------
        RedBlackTreeNode
            The former left child of the node which is its parent after the
            rotation.
        """
        # the left child will become the new parent of the node
        left_child = node.left
        subtree_to_move = left_child.right
        
        if node.parent and (node is node.parent.right):
            node.parent.right = left_child
        elif node.parent and (node is node.parent.left):
            node.parent.left = left_child
            
        left_child.parent = node.parent
        node.parent = left_child
        left_child.right = node
        node.left = subtree_to_move
        
        if subtree_to_move:
            subtree_to_move.parent = node
        
        node.update()
        left_child.update()
        
        return left_child
        
        
    def rotate_left(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
        """Perform a left rotation on the node.
        
        Parameters
        ----------
        node: RedBlackTreeNode
            The node on which to perform a left rotation.
        
        Returns
        -------
        RedBlackTreeNode
            The former right child of the node which is its parent after the
            rotation.
        """
        # the right child will become the new parent of the node
        right_child = node.right
        subtree_to_move = right_child.left
        
        if node.parent and (node is node.parent.right):
            node.parent.right = right_child
        elif node.parent and (node is node.parent.left):
            node.parent.left = right_child
            
        right_child.parent = node.parent
        node.parent = right_child
        right_child.left = node
        node.right = subtree_to_move
        
        if subtree_to_move:
            subtree_to_move.parent = node
        
        node.update()
        right_child.update()
        
        return right_child
    
    
    def _rebalance(self, node: BinaryNode) -> BinaryNode:
        """Rebalance the tree starting from the specified node.

        Parameters
        ----------
        node : BinaryNode
            The node at which to start rebalancing.

        Returns
        -------
        BinaryNode
            The root of the tree after rebalancing.
        """        
        raise NotImplementedError
    
    
    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.
        
        Parameters
        ----------
        key : Any
            The key to be inserted.
        """
        raise NotImplementedError
    
    
    def _delete_key(self, key: Any) -> None:
        """Delete a key.
        
        Parameters
        ----------
        key : Any
            The key to be deleted.
        """
        raise NotImplementedError
    
    
    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index and return it.
        
        Parameters
        ----------
        idx : int
            The index of the element to be removed.
        """
        raise NotImplementedError('not implemented for base class')
    
    
    def check_integrity(self) -> bool:
        """Integrity check of the tree.
            
        Checks whether all children have a correct parent reference and the size
        and heigth is correct in all subtrees. Intended for testing purpose.
        
        Returns
        -------
        bool
            Whether all integrity checks have been passed.
        """
        raise NotImplementedError
