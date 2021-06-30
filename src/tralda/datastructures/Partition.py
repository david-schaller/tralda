# -*- coding: utf-8 -*-

from tralda.datastructures.DoublyLinkedList import DLList


__author__ = 'David Schaller'


class Partition:
    """Dynamic partition implementation."""
    
    def __init__(self, iterable):
        
        self.P = DLList()       # partition
        self.lookup = {}        # item --> set in partition
        
        for s in iterable:
            dll_node = self.P.append(set(s))
            for x in s:
                self.lookup[x] = dll_node
                
    
    def __len__(self):
        
        return len(self.P)
    
    
    def __iter__(self):
        
        return PartitionIterator(self)


    def __next__(self):
        
        pass
              
                
    def in_same_set(self, x, y):
        
        return self.lookup[x] is self.lookup[y]
    
    
    def separated_xy_z(self, x, y, z):
        
        return (self.lookup[x] is     self.lookup[y] and
                self.lookup[x] is not self.lookup[z])
    
    
    def merge(self, repr1, repr2):
        
        set1 = self.lookup[repr1]
        set2 = self.lookup[repr2]
        
        if set1 is set2:
            return
        
        # ensure that set1 is smaller
        if len(set2._value) < len(set1._value):
            set1, set2 = set2, set1
        
        for x in set1._value:
            self.lookup[x] = set2
        set2._value |= set1._value
        
        self.P.remove_node(set1)
        
        return set1._value
        

class PartitionIterator:
    """Iterator class for Partition class."""
    
    def __init__(self, partition):
        
        self.partition = partition
        self._current = self.partition.P._first
        
    
    def __next__(self):
        
        if self._current:
            x = self._current
            self._current = self._current._next
            return x._value
        else:
            raise StopIteration
        
        
    
    