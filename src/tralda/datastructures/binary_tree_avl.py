# -*- coding: utf-8 -*-

"""
AVL-tree implementation.
    
Balanced binary search tree implementation of a set (TreeSet) and
dictionary (TreeDict).
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

from tralda.datastructures.binary_tree_base import BinaryNode
from tralda.datastructures.binary_tree_base import BaseBinarySearchTree
from tralda.datastructures.binary_tree_base import BinaryTreeIterator


__author__ = 'David Schaller'
 

class TreeSet(BaseBinarySearchTree):
    """AVL tree."""
    
    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator
    
    
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
        while node:
            node.update()
            while abs(node.balance()) > 1:
                if node.balance() > 1:
                    if node.left.balance() >= 0:
                        # single rotation needed
                        node.rightrotate()
                    else:      
                        # double rotation needed
                        node.left.leftrotate()
                        node.rightrotate()
                else:
                    if node.right.balance() <= 0:
                        # single rotation needed
                        node.leftrotate()
                    else:
                        # double rotation needed
                        node.right.rightrotate()
                        node.leftrotate()
                        
            if not node.parent:
                return node
            node = node.parent
    
    
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
                self.root = self._rebalance(node)
            elif key > node.key:
                node.right = self.node_class(key)
                node.right.parent = node
                self.root = self._rebalance(node)
    
    
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
            to_rebalance = subst.parent
            
            if to_rebalance is node:
                to_rebalance = subst
            else:
                to_rebalance.left = subst.right
                if subst.right:
                    subst.right.parent = to_rebalance
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
                
            self.root = self._rebalance(to_rebalance)
            
        elif node.left:
            par = node.parent
            node.left.parent = par
            
            if par:
                if node is par.left:
                    par.left = node.left
                elif node is par.right:
                    par.right = node.left
                self.root = self._rebalance(par)
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
                self.root = self._rebalance(par)
            else:
                self.root = node.right
                
        else:
            par = node.parent
            if par:
                if node is par.left:
                    par.left = None
                elif node is par.right:
                    par.right = None
                self.root = self._rebalance(par)
            else:
                self.root = None
        
        node.parent, node.left, node.right = None, None, None
    
    
    def check_integrity(self) -> bool:
        """Integrity check of the tree.
            
        Checks whether all children have a correct parent reference and the size
        and heigth is correct in all subtrees. Additionally, the AVL property is 
        checked. Intended for testing purpose.
        
        Returns
        -------
        bool
            Whether all integrity checks have been passed.
        """
        super().check_integrity()
        
        for node in self._inorder_traversal():
            if abs(node.balance()) > 1:
                print(f'node {node} is unbalanced')
                return False

        

class TreeDictNode(BinaryNode):
    
    __slots__ = ('value',)
    _attributes = ('key', 'value',)
    
    def __init__(self, key: Any, value: Any) -> None:
        """Initialize the TreeDict node.

        Parameters
        ----------
        key : Any
            The key of the node. Keys must be unique within a binary search 
            tree.
        value : Any
            The value associated with the key.
        """
        
        super().__init__(key)
        self.value = value
        
    
    def copy(self) -> 'TreeDictNode':
        """A copy of this node.

        Returns:
            TreeDictNode: A copy of this node.
        """ 
        copy= super().copy()
        copy.value = self.value
        
        return copy


class TreeDictIterator(BinaryTreeIterator):
    """Iterator for AVL-tree-based dictionary."""
    
    __slots__ = ('_mode',)
    
    def __init__(self, tree: 'TreeDict', mode: int = 1):
        """Initilize the TreeDict iterator.

        Parameters
        ----------
        tree : TreeDict
            The TreeDict instance.
        mode : int
            What shall be iterated through (1 = keys, 2 = values, 3 = key-value
            pairs).
        """
        super().__init__(tree)
        
        # What is returned?
        # 1 -- key
        # 2 -- value
        # 3 -- (key, value)
        self._mode = mode

    def __iter__(self):

        return self
    
    def __next__(self) -> Any | tuple[Any, Any]:
        """The next key, value or key-value pair in the binary search tree.
        
        Returns
        -------
        Any or tuple[Any, Any]
            The next key, value, or key-value pair.

        Raises
        ------
        StopIteration
            When no items are left.
        """
        node = self._find_next()
        if node:
            if self._mode == 1:
                return node.key
            elif self._mode == 2:
                return node.value
            else:
                return (node.key, node.value)
        else:
            raise StopIteration


class TreeDict(TreeSet):
    
    node_class = TreeDictNode
    iterator_class: Iterator[Any] = TreeDictIterator
        
    
    def __getitem__(self, key: Any) -> Any:
        """Return the value associated with the key.
        
        Overrides the method of the base class where the key at a specified
        index is returned.
        
        Parameters
        ----------
        key : Any
            The key.
            
        Returns
        -------
        Any
            The value associated with the key.
        
        Raises
        ------
        KeyError
            If the key does not exist.
        """
        node = self._find(key)
        
        if not node:
            raise KeyError(str(key))
            
        return node.value
    
    
    def get(self, item: Any, default: Any = None) -> Any:
        """Return the value associated with the key or a default value if the
        key does not exist.
        
        Parameters
        ----------
        key : Any
            The key.
        default : Any, optional
            The default to be returned if the key does not exist. Defaults to
            None.
            
        Returns
        -------
        Any
            The value associated with the key or the default value.
        """
        node = self._find(item)
        
        if node:
            return node.value
        else:
            return default
        
    
    def keys(self) -> Iterator[Any]:
        """Iterator for the keys.
        
        Returns
        -------
        Iterator[Any]
            An iterator for the keys.
        """
        return self.iterator_class(self, mode=1)
        
    
    def values(self) -> Iterator[Any]:
        """Iterator for the values.
        
        Returns
        -------
        Iterator[Any]
            An iterator for the values.
        """
        return self.iterator_class(self, mode=2)
    
    
    def items(self) -> Iterator[tuple[Any, Any]]:
        """Iterator for the key-value pairs.
        
        Returns
        -------
        Iterator[tuple[Any, Any]]
            An iterator for the key-value pairs.
        """
        return self.iterator_class(self, mode=3)
    
    
    def pop(self, key: Any) -> Any:
        """Remove a key from the tree and return its value.
        
        Parameters
        ----------
        key: Any
            The key to be removed.
        
        Returns
        -------
        Any
            The value associated with the key.
        
        Raises
        ------
        KeyError
            If key is not in the tree.
        """
        node = self._find(key)
        
        if not node:
            raise KeyError(str(key))
            
        self._delete_node(node)
        
        return node.value
    
    
    def value_at(self, idx) -> Any:
        """Return the value at the index.
        
        Parameters
        ----------
        idx : int
            The index.
            
        Returns
        -------
        Any
            The value of the node at the index.
        """
        return self._node_at(idx).value
    
    
    def key_and_value_at(self, index) -> tuple[Any, Any]:
        """Return the key-value pair at the index.
        
        Parameters
        ----------
        idx : int
            The index.
            
        Returns
        -------
        tuple[Any, Any]
            The key-value pair of the node at the index.
        """
        node = self._node_at(index)
        
        return (node.key, node.value)
    
    
    def pop_at(self, idx: int) -> tuple[Any, Any]:
        """Remove node at the index and return its key value pair.
        
        Parameters
        ----------
        idx : int
            The index.
        
        Returns
        -------
        tuple[Any, Any]
            The key-value pair of the node at the index.
        """
        node = self._node_at(idx)
        self._delete_node(node)
        
        return (node.key, node.value)
    
    
    def add(self, key: Any, value: Any) -> None:
        """Insert a key and value.
        
        Parameters
        ----------
        key: Any
            The key to be added.
        value: Any
            The associated value.
        """
        self.insert(key, value)
    
    
    def insert(self, key: Any, value: Any) -> None:
        """Insert a key and value.
        
        Parameters
        ----------
        key: Any
            The key to be added.
        value: Any
            The associated value.
        """
        if not self.root:
            self.root = self.node_class(key, value)
        else:
            node = self._find_insert(key)
            
            if key < node.key:
                node.left = self.node_class(key, value)
                node.left.parent = node
                self.root = self._rebalance(node)
            elif key > node.key:
                node.right = self.node_class(key, value)
                node.right.parent = node
                self.root = self._rebalance(node)
    
    
    def _insert_key(self, key: Any) -> None:
        raise NotImplementedError('cannot insert key without value')
        

if __name__ == '__main__':
    
    import random, time
    
    keys = [i for i in range(100000)]
    random.shuffle(keys)
    
    t = TreeDict()
    for key in keys:
        t.insert(key, None)
    # print(t.to_newick())
    
    t = t.copy()
    t.check_integrity()
    print(len(t))
    print(t.pop_at(-10000))
    print(len(t))
    print(t.key_and_value_at(980))
    
    l = [key for key in t.items()]
    print(l[-5:])
     
    s = set()
    for key in keys:
        s.add(key)
    
    # choosing a random element from a TreeSet/TreeDict can be done very fast
    # using a random integer and the pop_at() function
    start_time1 = time.time()
    for i in range(500):
        x = random.randint(0, len(t)-1)
        t.pop_at(x)
    end_time1 = time.time()
    print(len(t))
    
    start_time2 = time.time()
    for i in range(500):
        x = random.choice(tuple(s))
        s.remove(x)
    end_time2 = time.time()
    
    print(end_time1 - start_time1, end_time2 - start_time2)
    
    t.check_integrity()