# -*- coding: utf-8 -*-

import unittest
import networkx as nx

from tralda.cograph import (random_cotree,
                            to_cograph,
                            to_cotree,
                            CographEditor,
                            complete_multipartite_completion,
                            )
import tralda.tools.GraphTools as gt


__author__ = 'David Schaller'


class TestCographPackage(unittest.TestCase):

    
    def test_is_cograph(self):
        
        cotree = random_cotree(100)
        cograph = to_cograph(cotree)
        
        self.assertTrue( to_cotree(cograph) )
        
        ce = CographEditor(cograph)
        ce.cograph_edit(run_number=10)
        self.assertEqual(ce.best_cost, 0)
        
        
    def test_is_no_cograph(self):
        
        graph = nx.Graph()
        graph.add_edge('a', 'b')
        graph.add_edge('b', 'c')
        graph.add_edge('c', 'd')
        
        self.assertFalse(to_cotree(graph))
        
        ce = CographEditor(graph)
        ce.cograph_edit(run_number=10)
        self.assertGreater(ce.best_cost, 0)
        
    
    def test_editing(self):
        
        graph = gt.random_graph(100, p=0.3)
        ce = CographEditor(graph)
        ce.cograph_edit(run_number=10)
        cograph = to_cograph(ce.best_T)
        
        self.assertTrue( to_cotree(cograph) )
    
    
    def test_compl_multipart_completion(self):
        
        cotree = random_cotree(20)
        sets, cmg = complete_multipartite_completion(cotree, supply_graph=True)
        orig_cograph = to_cograph(cotree)
        
        self.assertTrue(gt.is_subgraph(orig_cograph, cmg))
        

if __name__ == '__main__':
    
    unittest.main()