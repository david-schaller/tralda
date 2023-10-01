# -*- coding: utf-8 -*-

"""Base classes for binary search trees.
"""

from typing import Optional, Any, Iterator, Iterable

__author__ = 'David Schaller'


class BinaryNode:
    
    __slots__ = ('key', 'parent', 'left', 'right', 'size', 'height',)
    _attributes = ('key',)
    
    def __init__(self, key: Any) -> None:
        """Initialize the binary search tree node.

        Parameters
        ----------
        key : Any
            The key/label of the node. Keys must be unique within a BST.
        """
        self.key: Any = key
        
        self.parent: Optional[BinaryNode] = None
        self.left: Optional[BinaryNode] = None
        self.right: Optional[BinaryNode] = None
        
        # stores number of elements in its subtree
        self.size: int = 1
        
        # height of the subtree
        self.height: int = 1
    
    
    def __str__(self) -> str:
        """String representation of the node.

        Returns
        -------
        str
            String representation.
        """
        return '<BST node: {}>'.format(self.key)
    
    
    def update(self) -> None:
        """Update height and size of (the subtree under) the node.
        """
        if self.left and self.right:
            self.height = 1 + max(self.left.height, self.right.height)
            self.size = 1 + self.left.size + self.right.size
        elif self.left:
            self.height = 1 + self.left.height
            self.size = 1 + self.left.size
        elif self.right:
            self.height = 1 + self.right.height
            self.size = 1 + self.right.size
        else:
            self.height = 1
            self.size = 1
    
    
    def left_size(self) -> int:
        """Size of the left subtree.

        Returns
        -------
        int
            Size of the left subtree.
        """        
        return self.left.size if self.left else 0
    
    
    def right_size(self) -> int:
        """Size of the right subtree.

        Returns
        -------
        int
            Size of the right subtree.
        """        
        return self.right.size if self.right else 0
    
    
    def balance(self) -> int:
        """Balance factor of the node.

        Returns
        -------
        int
            Balance factor.
        """        
        left_height = self.left.height if self.left else 0
        right_height = self.right.height if self.right else 0
        
        return left_height - right_height
    
    
    def copy(self) -> 'BinaryNode':
        """A copy of this node.

        Returns:
            BinaryNode: A copy of this node.
        """        
        copy = self.__class__(
            *(getattr(self, a) for a in self._attributes)
        )
        copy.height = self.height
        copy.size = self.size
        
        return copy
    
    
    def rightrotate(self) -> None:
        """Perform a right rotation on this node.
        """
        # the left child will become the new parent of this node
        left_child = self.left
        subtree_to_move = left_child.right
        
        if self.parent and (self is self.parent.right):
            self.parent.right = left_child
        elif self.parent and (self is self.parent.left):
            self.parent.left = left_child
            
        left_child.parent = self.parent
        self.parent = left_child
        left_child.right = self
        self.left = subtree_to_move
        
        if subtree_to_move:
            subtree_to_move.parent = self
        
        self.update()
        left_child.update()
        
        
    def leftrotate(self):
        """Perform a left rotation on this node.
        """
        # the right child will become the new parent of this node
        right_child = self.right
        subtree_to_move = right_child.left
        
        if self.parent and (self is self.parent.right):
            self.parent.right = right_child
        elif self.parent and (self is self.parent.left):
            self.parent.left = right_child
            
        right_child.parent = self.parent
        self.parent = right_child
        right_child.left = self
        self.right = subtree_to_move
        
        if subtree_to_move:
            subtree_to_move.parent = self
        
        self.update()
        right_child.update()


class BinaryTreeIterator:
    """Iterator for binary search trees."""
    
    __slots__ = ('tree', '_current', '_from')
    
    def __init__(self, tree: 'BaseBinarySearchTree'):
        """Initilize the tree iterator.

        Parameters
        ----------
        tree : BaseBinarySearchTree
            The binary search tree.
        """
        self.tree = tree
        self._current = self.tree.root
        
        # Where do I come from?
        # 1 -- up
        # 2 -- left
        # 3 -- right
        self._from = 1
        
        
    def __iter__(self) -> 'BaseBinarySearchTree':
        return self
        
    
    def __next__(self) -> Any:
        """The next item in the binary search tree.
        
        Returns
        -------
            The next item.

        Raises
        ------
        StopIteration
            When no items are left.
        """
        node = self._find_next()
        if node:
            return node.key
        else:
            raise StopIteration
    
    
    def _find_next(self) -> BinaryNode:
        """Finds the next node.
        
        Returns
        -------
        BinaryNode
            The next node.
        """
        while self._current:
            # coming from above
            if self._from == 1:
                if self._current.left:
                    self._current = self._current.left
                else:
                    self._from = 2
            # coming from left child --> return this node
            elif self._from == 2:
                x = self._current
                if self._current.right:
                    self._current = self._current.right
                    self._from = 1
                else:
                    self._current = self._current.parent
                    if self._current and self._current.left is x:
                        self._from = 2
                    elif self._current:
                        self._from = 3
                return x
            # coming from right child
            else:
                x = self._current
                self._current = self._current.parent
                if self._current and self._current.left is x:
                    self._from = 2
                elif self._current:
                    self._from = 3
 

class BaseBinarySearchTree:
    """Base class for binary search trees."""
    
    node_class = BinaryNode
    iterator_class: Iterator[Any] = BinaryTreeIterator
    
    def __init__(self) -> None:
        """Initialize the balanced binary search tree.
        """        
        self.root: Optional[BinaryNode] = None
    
    
    def __iter__(self) -> Iterator[Any]:
        """An iterator for the BST.

        Returns
        -------
        Iterator[Any]
            An iterator for the tree.
        """        
        return self.iterator_class(self)


    def __next__(self) -> None:
        pass
            
            
    def __nonzero__(self) -> bool:
        """Return whether the tree is non-empty.

        Returns
        -------
        bool
            Whether the tree contains elements or not.
        """        
        return bool(self.root)
      
    
    def __len__(self) -> int:
        """Number of elements in the tree.

        Returns
        -------
        int
            Number of elements.
        """        
        return self.root.size if self.root else 0
    
    
    def __contains__(self, item: Any) -> bool:
        """Return whether the tree contains the item.

        Parameters
        ----------
        item : Any
            The item.

        Returns
        -------
        bool
            Whether the tree contains the item.
        """        
        return self._find(item) is not None
    
    
    def __getitem__(self, idx: int) -> Any:
        """Return the element at the index.
        
        Same as 'key_at(index)'.
        
        Parameters
        ----------
        idx : int
            The index.
            
        Returns
        -------
        Any
            The key of the node at the index.
        """
        return self._node_at(idx).key
    
    
    def key_at(self, idx: int) -> Any:
        """Return the key at the index.
        
        Parameters
        ----------
        idx : int
            The index.
            
        Returns
        -------
        Any
            The key of the node at the index.
        """
        return self._node_at(idx).key
    
    
    def _node_at(self, idx: int) -> BinaryNode:
        """Return the node at the index.
        
        Parameters
        ----------
        idx : int
            The index.
            
        Returns
        -------
        Any
            The node instance at the index.
        
        Raises
        ------
        IndexError
            If the index is out of bounds.
        RuntimeError
            If the index seems valid but the node could not be found. A 
            corrupted integrity of the tree datastructure could be the reason.
        """
        
        if idx < 0:
            if idx < -self.root.size:
                raise IndexError(f'index {idx} is out of range')
            else:
                idx += self.root.size
        
        if idx >= self.root.size:
            raise IndexError(f'index {idx} is out of range')
        
        current = self.root
        current_sum = 0
        
        while current:
            current_idx = current_sum + current.left_size()
            if idx == current_idx:
                return current
            elif idx < current_idx:
                current = current.left
            else:
                current = current.right
                current_sum = current_idx + 1
                
        raise RuntimeError(f'could not find node with index {idx}')
    
    
    def add(self, item: Any) -> None:
        """Insert an item.
        
        Parameters
        ----------
        item : Any
            The new item to be inserted.
        """
        self._insert_key(item)
    
    
    def insert(self, key: Any) -> None:
        """Insert an item.
        
        The function 'add(item)' should be used instead for sets.
        
        Parameters
        ----------
        key : Any
            The new item to be inserted.
        """
        self.add(key)
                
                
    def remove(self, key: Any) -> None:
        """Remove a key from the tree.
        
        Parameters
        ----------
        key : Any
            The new item to be removed.
        
        Raises
        ------
        ValueError
            If the key is not in the tree.
        """
        node = self._find(key)
        
        if not node:
            raise KeyError(str(key))
            
        self._delete_node(node)
    
    
    def discard(self, key: Any) -> None:
        """Remove a key from the tree if present.
        
        Parameters
        ----------
        key : Any
            The new item to be removed.
        """
        node = self._find(key)
        if node:
            self._delete_node(node)
        
    
    def pop(self) -> Any:
        """Remove and return the greatest item.
        
        Returns
        -------
        Any
            The greatest item in the tree.
        
        Raises
        ------
        IndexError
            If the tree is empty.
        """
        if not self.root:
            raise IndexError('pop from empty tree')
        
        node = self._biggest_in_subtree(self.root)
        self._delete_node(node)
        
        return node.key
        
    
    def clear(self) -> None:
        """Removes all items from the tree.
        """
        self.root = None
        
    
    def difference_update(self, items: Iterable[Any]) -> None:
        """Discard all elements in the collection.
        
        Parameters
        ----------
        items : Iterable[Any]
            The items to be discarded.
        """
        for item in items:
            self.discard(item)
    
    
    def remove_at(self, idx: int) -> None:
        """Remove node at the index.
        
        Parameters
        ----------
        idx : int
            The index of the element to be removed.
        
        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        self._delete_node(self._node_at(idx))
    
    
    def pop_at(self, idx: int) -> Any:
        """Remove item at the index and return it.
        
        Parameters
        ----------
        idx : int
            The index of the element to be removed and returned.
        
        Returns
        -------
        Any
            The item at the index.
        
        Raises
        ------
        IndexError
            If the index is out of bounds.
        """
        node = self._node_at(idx)
        self._delete_node(node)
        
        return node.key
    
    
    def _find(self, key: Any) -> Optional[BinaryNode]:
        """Find the node for the specified key.
        
        Parameters
        ----------
        key : Any
            The key to be searched.
        
        Returns
        -------
        BinaryNode or None
            The corresponding tree node or None if the key was not found.
        """
        if not self.root:
            return None
        
        current = self.root
        while current:
            if key == current.key:
                return current
            elif key < current.key:
                current = current.left
            else:
                current = current.right
                
                
    def _find_insert(self, key):
        """Find the position where the new key must be inserted.
        
        Parameters
        ----------
        key : Any
            The key to be inserted.
        
        Returns
        -------
        BinaryNode or None
            The tree node as a child of which the tree node will be inserted.
        """
        current = self.root
        while True:
            if key < current.key and current.left:
                current = current.left
            elif key > current.key and current.right:
                current = current.right
            else:
                return current
    
    
    def _insert_key(self, key: Any) -> None:
        """Insert a key into the tree if not already present.
        
        Parameters
        ----------
        key : Any
            The key to be inserted.
        
        Raises
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        raise NotImplementedError('not implemented for BST base class')
    
    
    def _delete_node(self, node: BinaryNode) -> None:
        """Delete a node.
        
        Parameters
        ----------
        node : BinaryNode
            The node to be deleted.
        
        Raises
        -------
        NotImplementedError
            If the child class does not implement this method.
        """
        raise NotImplementedError('not implemented for BST base class')
    
    
    def _smallest_in_subtree(self, node: BinaryNode) -> BinaryNode:
        """Return the left-most (smallest element) node in the subtree.
        
        Parameters
        ----------
        node : BinaryNode
            The node whose subtree is to be considered.
        """
        current = node
        while current.left:
            current = current.left
        
        return current
    
    
    def _biggest_in_subtree(self, node):
        """Return the right-most (largest element) node in the subtree.
        
        Parameters
        ----------
        node : BinaryNode
            The node whose subtree is to be considered.
        """
        current = node
        while current.right:
            current = current.right

        return current
    
    
    def copy(self) -> 'BaseBinarySearchTree':
        """Copy the tree.
        
        Returns
        -------
        BaseBinarySearchTree
            A copy of the tree.
        """
        
        def _copy_subtree(node: BinaryNode, parent: Optional[BinaryNode] = None):
            node_copy = node.copy()
            node_copy.parent = parent
            if node.left:
                node_copy.left = _copy_subtree(node.left, parent=node_copy)
            if node.right:
                node_copy.right = _copy_subtree(node.right, parent=node_copy)
            return node_copy
        
        tree_copy = self.__class__()
        if self.root:
            tree_copy.root = _copy_subtree(self.root)
        return tree_copy
                
    
    def to_newick(self) -> str:
        """Newick representation of the tree.
        
        Intended for testing purpose.
        
        Returns
        -------
        str
            A Newick representation of the tree.
        """
        def _newick(node):
            if not (node.left or node.right):
                return str(node.key)
            else:
                if node.left and node.right:
                    s = f'({_newick(node.left)},{_newick(node.right)})'
                elif node.left:
                    s = f'({_newick(node.left)},-)'
                elif node.right:
                    s = f'(-,{_newick(node.right)})'
                else:
                    s = ''
                return s + str(node.key)
            
        return _newick(self.root) if self.root else ''
    
    
    def _inorder_traversal(self):
        """Generator for the nodes in a pre-order traversal of the tree.
        
        Yields
        ------
        TreeNode
            All nodes of the tree in pre-order.
        """
        
        def _inorder(node):
            if node.left: yield from _inorder(node.left)
            yield node
            if node.right: yield from _inorder(node.right)
        
        if self.root:
            yield from _inorder(self.root)
        else:
            yield from []
    
    
    def check_integrity(self, node: Optional[BinaryNode] = None) -> bool:
        """Recursive integrity check of the tree.
            
        Checks whether all children have a correct parent reference and the size
        and heigth is correct in all subtrees. Intended for testing purpose.
        
        Parameters
        ----------
        node : BinaryNode, optional
            The node at which to start the recursive integrity check.
        
        Returns
        -------
        bool
            Whether all integrity checks have been passed.
        """
        if not node:
            if not self.root:
                print('tree has no root')
                return False
            else:
                return self.check_integrity(self.root)
        else:
            height_left, height_right, size = 0, 0, 1
            
            if node.left:
                if (
                    node is not node.left.parent or 
                    not self.check_integrity(node.left)
                ):
                    print(f'check node (left): {node}')
                    return False
                height_left = node.left.height
                size += node.left.size
                
            if node.right:
                if (
                    node is not node.right.parent or
                    not self.check_integrity(node.right)
                ):
                    print(f'check node (right):{node}')
                    return False
                height_right = node.right.height
                size += node.right.size
            
            if node.height != 1 + max(height_left, height_right):
                print(f'height of node {node} is incorrect')
                return False
            
            if node.size != size:
                print(f'size of node {node} is incorrect')
                return False
            
            return True
