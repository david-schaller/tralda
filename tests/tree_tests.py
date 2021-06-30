# -*- coding: utf-8 -*-

import unittest

from tralda.datastructures import Tree


__author__ = 'David Schaller'


class TestTrees(unittest.TestCase):
    
    
    def test_integrity(self):
        
        tree = Tree.random_tree(20)
        
        self.assertTrue( tree._assert_integrity() )

    
    def test_traversal(self):
        
        tree = Tree.random_tree(20)
        pre = {v for v in tree.preorder()}
        post = {v for v in tree.postorder()}
        
        self.assertTrue( pre == post )
            

if __name__ == '__main__':
    
    unittest.main()