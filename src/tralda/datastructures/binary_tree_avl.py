# -*- coding: utf-8 -*-

"""
AVL-tree implementation.
    
Balanced binary search tree implementation of a set (TreeSet) and
dictionary (TreeDict).
"""

from __future__ import annotations

from typing import Any, Iterator

from tralda.datastructures.binary_tree_base import BinaryNode
from tralda.datastructures.binary_tree_base import BaseBinarySearchTree
from tralda.datastructures.binary_tree_base import BinaryTreeIterator


__author__ = 'David Schaller'
 

class TreeSet(BaseBinarySearchTree):
    """AVL tree."""
    
    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator
    
    
    def rotate_right(self, node: BinaryNode) -> BinaryNode:
        """Perform a right rotation on the node.
        
        Parameters
        ----------
        node: BinaryNode
            The node on which to perform a right rotation.
        
        Returns
        -------
        BinaryNode
            The former left child of the node which is its parent after the
            rotation.
        """
        left_child = node.left
        node.left = left_child.right
        left_child.right = node
        
        node.update()
        left_child.update()
        
        return left_child
        
        
    def rotate_left(self, node: BinaryNode) -> BinaryNode:
        """Perform a left rotation on the node.
        
        Parameters
        ----------
        node: BinaryNode
            The node on which to perform a left rotation.
        
        Returns
        -------
        BinaryNode
            The former right child of the node which is its parent after the
            rotation.
        """
        right_child = node.right
        node.right = right_child.left
        right_child.left = node
        
        node.update()
        right_child.update()
        
        return right_child
    
    
    def _rebalance(self, node: BinaryNode) -> BinaryNode:
        """Rebalance the node.

        Parameters
        ----------
        node : BinaryNode
            The node to be rebalanced.

        Returns
        -------
        BinaryNode
            The root of the tree after rebalancing.
        """
        node.update()
        balance = node.balance()
        
        if balance > 1:
            if node.left.balance() >= 0:
                # single rotation needed
                node = self.rotate_right(node)
            else:      
                # double rotation needed
                node.left = self.rotate_left(node.left)
                node = self.rotate_right(node)
        elif balance < -1:
            if node.right.balance() <= 0:
                # single rotation needed
                node = self.rotate_left(node)
            else:
                # double rotation needed
                node.right = self.rotate_right(node.right)
                node = self.rotate_left(node)
        
        return node
    
    
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
        self.root = self._insert_and_rebalance(key, self.root)
        
    
    def _insert_and_rebalance(self, key: Any, node: BinaryNode) -> BinaryNode:
        """Recursive insertion and rebalancing.
        """
        if node is None:
            return self.node_class(*self._temp_attributes)
        elif key < node.key:
            node.left = self._insert_and_rebalance(key, node.left)
        elif key > node.key:
            node.right = self._insert_and_rebalance(key, node.right)
        else:
            raise ValueError(f'key {key} already exists')
        
        return self._rebalance(node)
    
    
    def _delete_key(self, key: Any) -> None:
        """Delete a key if present.
        
        Parameters
        ----------
        key : Any
            The key to be deleted.
        """
        self.root = self._delete_and_rebalance(key, self.root)
        
    
    def _delete_and_rebalance(self, key: Any, node: BinaryNode) -> BinaryNode:
        """Recursive deletion and rebalancing.
        """
        if node is None:
            raise ValueError(f'key {key} not found')
        elif key < node.key:
            node.left = self._delete_and_rebalance(key, node.left)
        elif key > node.key:
            node.right = self._delete_and_rebalance(key, node.right)
        else:
            if node.left is None:
                node = node.right
            elif node.right is None:
                node = node.left
            else:
                subst_node = self._smallest_in_subtree(node.right)
                node.set_attributes(subst_node.get_attributes())
                # now find and delete subst_node
                node.right = self._delete_and_rebalance(node.key, node.right)
        
        if node is not None:
            node = self._rebalance(node)
        
        return node
    
    
    def _pop_at_index(self, idx: int) -> None:
        """Remove item at the index.
        
        Parameters
        ----------
        idx : int
            The index of the element to be removed.
        """
        self.root = self._pop_at_index_and_rebalance(idx, self.root)
    
    
    def _pop_at_index_and_rebalance(
        self,
        idx: int,
        node: BinaryNode
    ) -> BinaryNode:
        """Recursive deletion of a node at the index.
        """
        if node is None:
            raise ValueError(f'could not find node with index {idx}')
        
        node_idx = node.left_size()
        
        if idx == node_idx:
            self._temp_attributes = node.get_attributes()
            return self._delete_and_rebalance(node.key, node)
        elif idx < node_idx:
            node.left = self._pop_at_index_and_rebalance(idx, node.left)
        else:
            node.right = self._pop_at_index_and_rebalance(idx - node_idx - 1, 
                                                          node.right)
        
        if node is not None:
            node = self._rebalance(node)
        
        return node
    
    
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
        try:
            node = next(self._inorder_generator)
            if self._mode == 1:
                return node.key
            elif self._mode == 2:
                return node.value
            else:
                return (node.key, node.value)
        except StopIteration:
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
    
    
    def value_at_index(self, idx) -> Any:
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
        return self._node_at_index(idx).value
    
    
    def key_and_value_at_index(self, idx) -> tuple[Any, Any]:
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
        node = self._node_at_index(idx)
        
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
        
        Raises
        ------
        ValueError
            If the key already exists.
        """
        self._temp_attributes = (key, value)
        self._insert_key(key)
        self._temp_attributes = None
        

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
    print(t.pop_at_index(-10000))
    print(len(t))
    print(t.key_and_value_at_index(980))
    
    l = [key for key in t.items()]
    print(l[-5:])
     
    s = set()
    for key in keys:
        s.add(key)
    
    # choosing a random element from a TreeSet/TreeDict can be done very fast
    # using a random integer and the pop_at_index() function
    start_time1 = time.time()
    for i in range(500):
        x = random.randint(0, len(t)-1)
        t.pop_at_index(x)
    end_time1 = time.time()
    print(len(t))
    
    start_time2 = time.time()
    for i in range(500):
        x = random.choice(tuple(s))
        s.remove(x)
    end_time2 = time.time()
    
    print(end_time1 - start_time1, end_time2 - start_time2)
    
    t.check_integrity()