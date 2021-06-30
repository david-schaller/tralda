# -*- coding: utf-8 -*-

"""
Linked list.

Implementation of a linked list in Python 3. The list enables access
to single list elements in order to modify/delete values in constant time.
"""

import collections


__author__ = 'David Schaller'


class LinkedListNode:
    """Linked list node."""
    
    __slots__ = ('_value', '_next')
    
    def __init__(self, value, next_node=None):
        
        self._value = value
        self._next = next_node
    
    
    def get(self):
        
        return self._value


class LinkedList:
    """Linked list."""
    
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
        
        return LinkedListIterator(self)


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
        
        # search from beginning
        node = self._first
        for _ in range(index):
            node = node._next
            
        return node
    
    
    def append(self, value):
        
        new_end = LinkedListNode(value)
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
            
    
    def concatenate(self, other):
        """Merges two linked lists.
        
        The instance for which this function is called is extended, the other
        list should no longer be used."""
        
        if not isinstance(other, LinkedList):
            raise TypeError("must be of type 'LinkedList'")
            
        if self._last:
            self._last._next = other._first
        else:
            self._first = other._first
        
        if other._last:
            self._last = other._last
            
        self._count += other._count
        
        return self
    
    
    def append_left(self, value):
        
        new_start = LinkedListNode(value, next_node=self._first)
        self._first = new_start
        if not self._last:
            self._last = new_start
        self._count += 1
        return new_start
    
    
    def remove(self, value):
        """Remove an item by value in O(n)."""
        
        node = self._first
        prev_node = None
        
        while node:
            if node._value == value:
                
                if prev_node:
                    prev_node._next = node._next
                else:
                    self._first = node._next
                    
                if not node._next:
                    self._last = prev_node
                
                self._count -= 1
                return
            
            prev_node = node
            node = node._next
            
        raise KeyError('value {} is not in the linked list'.format(value))
    
    
    def insert_right_of(self, node, value):
        """Insert a new item right of a node of the list in O(1)."""
        
        if node is self._last:
            new_node = self.append(value)
            
        else:
            new_node = LinkedListNode(value, next_node=node._next)
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
                new_end._next = None
            self._count = index
            
    
    def truncate_left(self, n):
        """Truncate n nodes on the left side."""
        
        if n >= self._count:
            self.clear()
        else:
            new_start = self.node_at(n)
            self._first = new_start
            self._count -= n
        
    
    # def popright(self):
    #     """Removes the last node of the list and returns its value."""
        
    #     pass
    
    
    def popleft(self):
        """Removes the first node of the list and returns its value."""
        
        if self._first:
            value = self._first._value
            self._first = self._first._next
            self._first._next = None
            self._count -= 1
            return value
        else:
            raise IndexError('pop from empty linked list')
    
    
    def clear(self):
        
        self._first = None
        self._last = None
        self._count = 0
        
    
    def _count_actual(self):
        """Counts the actual number of nodes."""
        
        current = self._first
        counter = 0
        
        while current:
            counter += 1
            current = current._next
            
        return counter
        

class LinkedListIterator:
    """Iterator class for linked list."""
    
    def __init__(self, llist):
        
        self.llist = llist
        self._current = llist._first
        
    
    def __next__(self):
        
        if self._current:
            x = self._current
            self._current = self._current._next
            return x._value
        else:
            raise StopIteration