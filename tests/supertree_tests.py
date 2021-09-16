# -*- coding: utf-8 -*-

import unittest, random

from tralda.datastructures import Tree
from tralda.supertree import linear_common_refinement, BUILD_supertree, build_st


__author__ = 'David Schaller'


class TestSupertrees(unittest.TestCase):
    
    
    def get_partial_trees(self, tree, contraction_prob=0.9):
        
        partial_trees = []
        for i in range(10):
            T_i = tree.copy()
            edges = []
            for u, v in T_i.inner_edges():
                if random.random() < contraction_prob:
                    edges.append((u,v))
            T_i.contract(edges)
            partial_trees.append(T_i)
        return partial_trees
    
    
    def test_LinCR(self):
    
        tree = Tree.random_tree(50, binary=True)
        partial_trees = self.get_partial_trees(tree, contraction_prob=0.9)
        
        cr_tree = linear_common_refinement(partial_trees)
        
        all_true = True if cr_tree else False
        if cr_tree:
            for T_i in partial_trees:
                if not cr_tree.is_refinement(T_i):
                    all_true = False
                    
        self.assertTrue(all_true)
    
    
    def test_supertree_equal(self):
    
        tree = Tree.random_tree(50, binary=True)
        partial_trees = self.get_partial_trees(tree, contraction_prob=0.9)
        
        cr_tree = linear_common_refinement(partial_trees)
        b_tree = BUILD_supertree(partial_trees)
        bst_tree = build_st(partial_trees)
        
        all_true = True if cr_tree and b_tree and bst_tree else False
        
        if all_true:
            if not b_tree.equal_topology(bst_tree):
                all_true = False
            if not b_tree.equal_topology(cr_tree):
                all_true = False
                    
        self.assertTrue(all_true)
            

if __name__ == '__main__':
    
    unittest.main()