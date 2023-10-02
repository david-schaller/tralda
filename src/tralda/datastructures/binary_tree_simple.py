# -*- coding: utf-8 -*-

"""
Simple binary search tree.
"""

from __future__ import annotations

from typing import Any, Iterator

from tralda.datastructures.binary_tree_base import BinaryNode
from tralda.datastructures.binary_tree_base import BaseBinarySearchTree
from tralda.datastructures.binary_tree_base import BinaryTreeIterator


__author__ = 'David Schaller'
 

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
        """
        if not self.root:
            self.root = self.node_class(key)
        else:
            node = self._find_insert(key)
            
            if key < node.key:
                node.left = self.node_class(key)
                node.left.parent = node
            elif key > node.key:
                node.right = self.node_class(key)
                node.right.parent = node
    
    
    def _delete_node(self, node: BinaryNode) -> None:
        """Delete a node.
        
        Parameters
        ----------
        node : BinaryNode
            The node to be deleted.
        """
        if node.left and node.right:
            # replace by smallest in right subtree
            subst = self._smallest_in_subtree(node.right)
            subst_par = subst.parent
            
            if subst_par is not node:
                subst_par.left = subst.right
                if subst.right:
                    subst.right.parent = subst_par
                subst.right = node.right
                node.right.parent = subst
                
            subst.left = node.left
            node.left.parent = subst
            par = node.parent
            subst.parent = par
            
            if par:
                if node is par.left:
                    par.left = subst
                elif node is par.right:
                    par.right = subst
            else:
                self.root = subst
            
        elif node.left:
            par = node.parent
            node.left.parent = par
            
            if par:
                if node is par.left:
                    par.left = node.left
                elif node is par.right:
                    par.right = node.left
            else:
                self.root = node.left
                
        elif node.right:
            par = node.parent
            node.right.parent = par
            
            if par:
                if node is par.left:
                    par.left = node.right
                elif node is par.right:
                    par.right = node.right
            else:
                self.root = node.right
                
        else:
            par = node.parent
            if par:
                if node is par.left:
                    par.left = None
                elif node is par.right:
                    par.right = None
            else:
                self.root = None
        
        node.parent, node.left, node.right = None, None, None
