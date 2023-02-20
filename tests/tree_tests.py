# -*- coding: utf-8 -*-

import unittest, os, random

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
    
    
    def test_newick(self):
        
        tree = Tree.random_tree(20)
        
        for v in tree.preorder():
            v.dist = random.random()
        
        newick = tree.to_newick()
        
        tree2 = Tree.parse_newick(newick)
        
        self.assertTrue(tree.equal_topology(tree2))
        
        for v, v2 in zip(tree.preorder(), tree2.preorder()):
            self.assertEqual(v.label, v2.label)
            self.assertTrue(v.dist - v2.dist <= 1e-6)
    
    
    def test_serialization(self):
        
        tree = Tree.random_tree(50)
        
        tree.serialize('temp_serialized.pickle')
        tree.serialize('temp_serialized.json')
        
        tree_p = tree.load('temp_serialized.pickle')
        tree_j = tree.load('temp_serialized.json')
        
        os.remove('temp_serialized.pickle')
        os.remove('temp_serialized.json')
        
        # print('tree', tree.to_newick())
        # print('tree_p', tree_p.to_newick())
        # print('tree_j', tree_j.to_newick())
        
        self.assertTrue( tree.equal_topology(tree_p) and
                         tree.equal_topology(tree_j) )
            

if __name__ == '__main__':
    
    unittest.main()
