# -*- coding: utf-8 -*-

"""
Doubly-linked list.

Implementation of a doubly-linked list in Python 3. The list enables access
to single list elements in order to modify/delete values in constant time.
"""

import collections


__author__ = 'David Schaller'


class DLListNode:
    """Doubly-linked list node."""
    
    __slots__ = ('_value', '_prev', '_next')
    
    def __init__(self, value, prev_node=None, next_node=None):
        
        self._value = value
        self._prev = prev_node
        self._next = next_node
        
    
    def get(self):
        
        return self._value


class DLList:
    """Doubly-linked list."""
    
    __slots__ = ('_first', '_last', '_count')
    
    def __init__(self, *args):
        
        self._first = None
        self._last = None
        self._count = 0
        for arg in args:
            if isinstance(arg, collections.abc.Iterable):
                for item in arg:
                    self.append(item)
            else:
                self.append(item)
        
    
    def __len__(self):
        
        return self._count
    
    
    def __nonzero__(self):
        
        return True if self._count > 0 else False
    
    
    def __iter__(self):
        
        return DLListIterator(self)


    def __next__(self):
        
        pass
    
    
    def __getitem__(self, index):
        
        return self.node_at(index)._value
    
    
    def first(self):
        
        return self._first._value
    
    
    def last(self):
        
        return self._last._value
    
    
    def first_node(self):
        
        return self._first
    
    
    def node_at(self, index):
        
        if not isinstance(index, int):
            raise TypeError("index must be of type 'int'")
            
        if index < (-self._count) or index >= self._count:
            raise IndexError('index {} is out of bounds'.format(index))
            
        if index < 0:
            index += self._count
        
        if index <= self._count // 2:
            # start from beginning
            node = self._first
            for _ in range(index):
                node = node._next
        
        else:
            # start from end
            node = self._last
            for _ in range(self._count - index - 1):
                node = node._prev
            
        return node
    
    
    def append(self, value):
        
        new_end = DLListNode(value, prev_node=self._last)
        if self._last:
            self._last._next = new_end
        self._last = new_end
        if not self._first:
            self._first = new_end
        self._count += 1
        return new_end
    
    
    def extend(self, iterable):
        
        for value in iterable:
            self.append(value)
    
    
    def append_left(self, value):
        
        new_start = DLListNode(value, next_node=self._first)
        if self._first:
            self._first._prev = new_start
        self._first = new_start
        if not self._last:
            self._last = new_start
        self._count += 1
        return new_start
    
    
    def remove_node(self, node):
        """Remove an item by reference to the 'DLListNode' instance in O(1)."""
        
        if node._prev:
            node._prev._next = node._next
        if node._next:
            node._next._prev = node._prev
        if self._first is node:
            self._first = node._next
        if self._last is node:
            self._last = node._prev
        node._prev, node._next = None, None
        self._count -= 1
    
    
    def remove(self, value):
        """Remove an item by value in O(n)."""
        
        node = self._first
        while node:
            if node._value == value:
                self.remove_node(node)
                return
            node = node._next
            
        raise KeyError('value {} is not in the doubly-linked list'.format(value))
        
        
    def remove_range(self, index, length=None):
        """Removes a range from the index of the specified length.
        
        Removes the range [index, index+length) from the sequence. If no length
        is specified or index+length is out of bounds, the list gets truncated.
        """
        
        if not isinstance(index, int):
            raise TypeError("index must be of type 'int'")
        elif index < 0:
            index = self._count + index
            
        if length is not None and (not isinstance(length, int) or length < 1):
            raise TypeError("length must be of type 'int' and >0")
        
        if length is None or index + length >= self._count:
            self.truncate(index)
            
        elif index == 0:
            self.truncate_left(length)
            
        else:
            cut_start = self.node_at(index)
            cut_end = self.node_at(index + length - 1)
            
            cut_start._prev._next = cut_end._next
            cut_end._next._prev = cut_start._prev
            cut_start._prev, cut_end._next = None, None
            
            self._count -= length
    
    
    def insert_right_of(self, node, value):
        """Insert a new item right of a node of the list in O(1)."""
        
        if node is self._last:
            new_node = self.append(value)
            
        else:
            new_node = DLListNode(value,
                                  prev_node=node,
                                  next_node=node._next)
            new_node._next._prev = new_node
            node._next = new_node
            self._count += 1
        
        return new_node
            
    
    def truncate(self, index):
        """Truncate all nodes starting from the specified index."""
        
        if index <= 0:
            self.clear()
        else:
            new_end = self.node_at(index-1)
            self._last = new_end
            if new_end._next:
                new_end._next._prev = None
                new_end._next = None
            self._count = index
            
    
    def truncate_left(self, n):
        """Truncate n nodes on the left side."""
        
        if n >= self._count:
            self.clear()
        else:
            new_start = self.node_at(n)
            self._first = new_start
            if new_start._prev:
                new_start._prev._next = None
                new_start._prev = None
            self._count -= n
        
    
    def popright(self):
        """Removes the last element of the list and returns its value."""
        
        if self._last:
            value = self._last._value
            self.remove_node(self._last)
            return value
        else:
            return None
    
    
    def popleft(self):
        """Removes the first element of the list and returns its value."""
        
        if self._first:
            value = self._first._value
            self.remove_node(self._first)
            return value
        else:
            return None
    
    
    def clear(self):
        
        self._first = None
        self._last = None
        self._count = 0
        
    
    def sublist(self, left_node, right_node):
        """Sublist of items in the subsequence between two nodes of the list.
        
        Parameters
        ----------
        left_node : DLListNode
            The left bound of the subsequence.
        right_node : DLListNode
            The right bound of the subsequence.
        
        Returns
        -------
        list
            The values in the subsequence.
        
        Raises
        ------
        KeyError
            If 'right_node' is not a successor of 'left_node' in the list.
        """
        
        l = []
        current = left_node
        
        while current:
            l.append(current._value)
            if current is right_node:
                return l
            current = current._next
            
        raise KeyError('{} is not contained in the subsequence'\
                       'starting with {}'.format(right_node, left_node))
        
    
    def _count_actual(self):
        """Counts the actual number of elements."""
        
        current = self._first
        counter = 0
        
        while current:
            counter += 1
            current = current._next
            
        return counter
        

class DLListIterator:
    """Iterator class for doubly-linked list."""
    
    def __init__(self, dllist):
        
        self.dllist = dllist
        self._current = dllist._first
        
    
    def __next__(self):
        
        if self._current:
            x = self._current
            self._current = self._current._next
            return x._value
        else:
            raise StopIteration