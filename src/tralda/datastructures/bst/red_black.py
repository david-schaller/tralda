# -*- coding: utf-8 -*-

"""
Red-black tree implementation.
    
Balanced binary search tree implementation of a set (TreeSet) and
dictionary (TreeDict).
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from tralda.datastructures.bst.base import BinaryNode
from tralda.datastructures.bst.base import BaseBinarySearchTree
from tralda.datastructures.bst.base import BinaryTreeIterator


__author__ = 'David Schaller'


class RedBlackTreeNode(BinaryNode):
    
    __slots__ = ('parent', 'is_red',)
    
    def __init__(self, key: Any) -> None:
        super().__init__(key)
        self.parent: Optional[RedBlackTreeNode] = None
        self.is_red: bool = False
        raise NotImplementedError('red-black trees are not yet implemented')
 

class TreeSet(BaseBinarySearchTree):
    """Red-black tree."""
    
    node_class = RedBlackTreeNode
    iterator_class: Iterator[Any] = BinaryTreeIterator
    
    def __init__(self) -> None:
        super().__init__()
        raise NotImplementedError('red-black trees are not yet implemented')
    
    
    def _replace_child(
        self, 
        parent: Optional[RedBlackTreeNode], 
        old_child: RedBlackTreeNode, 
        new_child: RedBlackTreeNode
    ) -> None:
        """Replace the child of a parent node.

        Parameters
        ----------
        parent: RedBlackTreeNode or None
            The parent node.
        old_child: RedBlackTreeNode
            The current child node to be replaced.
        new_child: RedBlackTreeNode
            The replacement node.
        """        
        if not parent:
            self.root = new_child
        elif parent.right is old_child:
            parent.right = new_child
        elif parent.left is old_child:
            parent.left = new_child
        else:
            raise RuntimeError('node is not a child of the provided parent')
        
        if new_child:
            new_child.parent = parent
    
    
    def _rotate_right(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
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
        parent = node.parent
        left_child = node.left
        
        node.left = left_child.right
        if node.left:
            node.left.parent = node.left
        
        left_child.right = node
        node.parent = left_child
        self._replace_child(parent, node, left_child)
        
        node.update()
        left_child.update()
        
        return left_child
        
        
    def _rotate_left(self, node: RedBlackTreeNode) -> RedBlackTreeNode:
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
        parent = node.parent
        right_child = node.right
        
        node.right = right_child.left
        if node.right:
            node.right.parent = node.right
        
        node.parent = right_child
        right_child.left = node
        self._replace_child(parent, node, right_child)
        
        node.update()
        right_child.update()
        
        return right_child
    
    
    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.
        
        Parameters
        ----------
        key : Any
            The key to be inserted.
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
                raise ValueError(f'key {key} already exists')
        
        # insert the new node
        new_node = RedBlackTreeNode(key)
        new_node.is_red = True
        if not parent:
            self.root = new_node
        elif key < parent.key:
            parent.left = new_node
        else:
            parent.right = new_node
        new_node.parent
        
        self._fix_after_insert(new_node)
        
    
    def _get_uncle(
        self, 
        parent: RedBlackTreeNode
    ) -> Optional[RedBlackTreeNode]:
        """Find the uncle starting from the parent node.

        Parameters
        ----------
        parent : RedBlackTreeNode
            The parent of the node for which to

        Returns
        -------
        RedBlackTreeNode or None
            The uncle node.
        """
        grandparent = parent.parent
        
        if parent is grandparent.left:
            return grandparent.right
        elif parent is grandparent.right:
            return grandparent.left
        else:
            raise RuntimeError(f'node is not a child of its parent')
    
    
    def _fix_after_insert(self, node: RedBlackTreeNode) -> None:
        """Fix the red-black properties after insertion.

        Parameters
        ----------
        node : RedBlackTreeNode
            The node at which to start fixing the properties.
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
