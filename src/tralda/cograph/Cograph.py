# -*- coding: utf-8 -*-

"""
Cographs and cotrees.

The linear cograph detection algorithm is an implementation of:
    D. G. Corneil, Y. Perl, and L. K. Stewart.
    A Linear Recognition Algorithm for Cographs.
    SIAM J. Comput., 14(4), 926–934 (1985).
    DOI: 10.1137/0214065
"""


import itertools, random
from collections import deque

import networkx as nx

from tralda.datastructures.Tree import Tree, TreeNode, LCA


__author__ = 'David Schaller'
    

class CotreeNode(TreeNode):
    
    __slots__ = ('aux_counter',)
    
    
    def __init__(self, ID, label=None, #parent=None,
                 aux_counter=0):
        
        super().__init__(ID, label=label)
        
        self.aux_counter = aux_counter      # auxiliary counting variable
        
    
    def __str__(self):
        
        if not self.children:
            return str(self.ID)
        
        elif self.label == 'series':
            return '<1>'
        
        elif self.label == 'parallel':
            return '<0>'
        
        else:
            return '<>'
        

class Cotree(Tree):
    
    # corresponding node type
    node_type = CotreeNode
    
    
    def __init__(self, root):
        super().__init__(root)
        
    
    def to_cograph(self):
        
        self.supply_leaves()
        G = nx.Graph()
        
        for v in self.root.leaves:
            G.add_node(v.ID)
        
        for u in self.preorder():
            if u.label == 'series':
                for v1, v2 in itertools.combinations(u.children, 2):
                    for l1, l2 in itertools.product(v1.leaves, v2.leaves):
                        G.add_edge(l1.ID, l2.ID)
        
        return G
    
    
    @staticmethod
    def cotree(G):
        """Checks if a graph is a cograph and returns its cotree.
        
        Linear O(|V| + |E|) implementation.
        """
        
        lcd = LinearCographDetector(G)
        return lcd.recognition()
        
        
    @staticmethod
    def cotree_naive(G):
        """Checks if a graph is a cograph and returns its cotree.
        
        Simple O(n^3) implementation.
        """
        
        def build_cotree(G, label=None):
            
            v = CotreeNode(None)
            v.label = label
            child_nodes = []
            
            if G.order() == 1:
                v.label = 'leaf'
                for ID in G.nodes():
                    v.ID = ID
                    
                return v
            
            child_nodes = []
            
            if v.label is None:
                ccs = [cc for cc in nx.connected_components(G)]
                
                if len(ccs) > 1:
                    v.label = 'parallel'
                else:
                    G = nx.complement(G)
                    ccs = [cc for cc in nx.connected_components(G)]
                    if len(ccs) > 1:
                        v.label = 'series'
                    else:
                        return False
            
            else:
                G = nx.complement(G)
                ccs = [cc for cc in nx.connected_components(G)]
                
                if len(ccs) == 1:
                    return False
            
            child_label = 'series' if v.label == 'parallel' else 'parallel'
            for cc in ccs:
                G_i = G.subgraph(cc).copy()
                v_i = build_cotree(G_i, label=child_label)
                if v_i:
                    child_nodes.append(v_i)
                else:
                    return False
            
            for child in child_nodes:
                v.children.append(child)
                child.parent = v
            
            return v
        
        root = build_cotree(G)
        
        if root:
            return Cotree(root)
        else:
            return False
        
    
    def complement(self, inplace=False):
        """Returns the cotree of the complement cograph."""
        
        tree = self if inplace else self.copy()
        
        for v in tree.inner_nodes():
            v.label = 'series' if v.label == 'parallel' else 'parallel'
        
        return tree
    
    
    def paths_of_length_2(self):
        """Generator for all paths of length 2 (edges) in the cograph."""
        
        self.supply_leaves()
        lca = LCA(self)
        
        for u in self.inner_nodes():
            
            if u.label == 'parallel':
                continue
            
            for v1, v2 in itertools.permutations(u.children, 2):
                for t1, t2 in itertools.combinations(v1.leaves, 2):
                    if lca(t1, t2).label == 'parallel':
                        for t3 in v2.leaves:
                            yield t1, t3, t2
    

    def copy(self, mapping=False):
        """Return a copy of the cotree.
        
        Constructs a deep copy of the tree, i.e. to the level of nodes.
        By default, the node attributes are all immutable data types.
        Hence, the original tree is not affected by operations on the copy.
        
        Parameters
        ----------
        mapping : bool
            If True, additionally return the mapping from original to copied
            nodes as dictionary.
        
        Returns
        -------
        Cotree or tuple of Cotree and dict
            A copy of the tree and optionally the mapping from original to 
            copied nodes.
        """
        
        if not self.root:
            return Cotree(None)
        
        orig_to_new = {}
        
        for orig in self.preorder():
            new = CotreeNode(orig.ID, label=orig.label, 
                             aux_counter=orig.aux_counter)
            orig_to_new[orig] = new
            if orig.parent:
                orig_to_new[orig.parent].add_child(new)
        
        if mapping:
            return Cotree(orig_to_new[self.root]), orig_to_new
        else:
            return Cotree(orig_to_new[self.root])
       
    
    @staticmethod
    def random_cotree(N, force_series_root=False):
        """Creates a random cotree."""
        
        root = CotreeNode(ID=0)
        cotree = Cotree(root)
        node_list = [root]
        nr, leaf_count = 1, 1
        
        while leaf_count < N:
            node = random.choice(node_list)
            
            # avoid nodes with outdegree 1
            if not node.children:
                new_child1 = CotreeNode(nr)
                new_child2 = CotreeNode(nr+1)
                node.add_child(new_child1)
                node.add_child(new_child2)
                node_list.extend(node.children)
                nr += 2
                leaf_count += 1
                
            # add only one child if there are already children
            elif node.children:
                new_child = CotreeNode(nr)
                node.add_child(new_child)
                node_list.append(new_child)
                nr += 1
                leaf_count += 1
        
        # assign labels ('series', 'parallel', 'leaf')
        for v in cotree.preorder():
            if not v.children:
                v.label == 'leaf'
            elif v.parent is None:
                if force_series_root:
                    v.label = 'series'
                else:
                    v.label = 'series' if random.random() < 0.5 else 'parallel'
            else:
                v.label = 'series' if v.parent.label == 'parallel' else 'parallel'
                
        return cotree
    

def linear_cograph_detection(G, return_cotree=True):
    
    lcd = LinearCographDetector(G)
    cotree = lcd.recognition()
    
    if not cotree:
        return False
    else:
        return cotree if return_cotree else True


class LinearCographDetector:
    
    def __init__(self, G):
        
        if not isinstance(G, nx.Graph):
            raise TypeError('not a NetworkX Graph')
        
        self.G = G
        self.V = [v for v in G.nodes()]
        
        self.T = Cotree(None)
        self.already_in_T = set()
        self.leaf_map = {}
        self.node_counter = 0
        
        self.marked = set()
        self.m_u_children = {}              # lists of marked and unmarked children
        self.mark_counter = 0
        self.unmark_counter = 0
        
        self.error_message = ''
    
    
    def recognition(self):
        
        if len(self.V) == 0:
            raise RuntimeError('empty graph in cograph recognition')
            return self.T
        
        elif len(self.V) == 1:
            self.T.root = CotreeNode(self.V[0], label='leaf')
            return self.T
        
        v1, v2 = self.V[0], self.V[1]
        self.already_in_T.update([v1, v2])
        
        R = CotreeNode(None, label='series')
        self.T.root = R
        
        if self.G.has_edge(v1, v2):
            v1_node = CotreeNode(v1, label='leaf')
            v2_node = CotreeNode(v2, label='leaf')
            R.add_child(v1_node)
            R.add_child(v2_node)
            self.node_counter = 3
        else:
            N = CotreeNode(None, label='parallel')
            R.add_child(N)
            v1_node = CotreeNode(v1, label='leaf')
            v2_node = CotreeNode(v2, label='leaf')
            N.add_child(v1_node)
            N.add_child(v2_node)
            self.node_counter = 4
            
        self.leaf_map[v1] = v1_node
        self.leaf_map[v2] = v2_node
        
        if len(self.V) == 2:
            self._remove_single_child_root()
            return self.T
        
        for x in self.V[2:]:
            
            # initialization (necessary?)
            self.marked.clear()
            self.m_u_children.clear()
            self.mark_counter = 0
            self.unmark_counter = 0
            self.already_in_T.add(x)        # add x for subsequent iterations
            
            # call procedure _MARK(x)
            self._MARK(x)
            
            # all nodes in T were marked and unmarked
            if self.node_counter == self.unmark_counter:
                R = self.T.root
                x_node = CotreeNode(x, label='leaf')
                R.add_child(x_node)
                self.node_counter += 1
                self.leaf_map[x] = x_node
                continue
            # no nodes in T were marked and unmarked
            elif self.mark_counter == 0:
                # d(R)=1
                if len(self.T.root.children) == 1:
                    N = self.T.root.children[0]
                    x_node = CotreeNode(x, label='leaf')
                    N.add_child(x_node)
                    self.node_counter += 1
                else:
                    R_old = self.T.root
                    R_new = CotreeNode(None, label='series')
                    N = CotreeNode(None, label='parallel')
                    R_new.add_child(N)
                    N.add_child(R_old)
                    self.T.root = R_new
                    
                    x_node = CotreeNode(x, label='leaf')
                    N.add_child(x_node)
                    self.node_counter += 3
                self.leaf_map[x] = x_node
                continue
            
            u = self._find_lowest()
            if not u:
                return False
            
            # label(u)=0 and |A|=1
            if u.label == 'parallel' and len(self.m_u_children[u]) == 1:
                w = self.m_u_children[u][0]
                if w.label == 'leaf':
                    new_node = CotreeNode(None, label='series')
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)
                    
                    x_node = CotreeNode(x, label='leaf')
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = CotreeNode(x, label='leaf')
                    w.add_child(x_node)
                    self.node_counter += 1 
            
            # label(u)=1 and |B|=1
            elif (u.label == 'series' and 
                  len(u.children) - len(self.m_u_children[u]) == 1):
                set_A = set(self.m_u_children[u])       # auxiliary set bounded by O(deg(x))
                w = None
                for child in u.children:
                    if child not in set_A:
                        w = child
                        break
                if w.label == 'leaf':
                    new_node = CotreeNode(None, label='parallel')
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)
                    
                    x_node = CotreeNode(x, label='leaf')
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = CotreeNode(x, label='leaf')
                    w.add_child(x_node)
                    self.node_counter += 1
            
            else:
                y = CotreeNode(None, label=u.label)
                for a in self.m_u_children[u]:
                    u.remove_child(a)
                    y.add_child(a)
                    
                if u.label == 'parallel':
                    new_node = CotreeNode(None, label='series')
                    u.add_child(new_node)
                    
                    new_node.add_child(y)
                    x_node = CotreeNode(x, label='leaf')
                    new_node.add_child(x_node)
                else:
                    par = u.parent
                    if par is not None:             # u was the root of T
                        par.remove_child(u)
                        par.add_child(y)
                    else:
                        self.T.root = y             # y becomes the new root
                    
                    new_node = CotreeNode(None, label='parallel')
                    y.add_child(new_node)
                    new_node.add_child(u)
                    x_node = CotreeNode(x, label='leaf')
                    new_node.add_child(x_node)
                self.node_counter += 3
                
            self.leaf_map[x] = x_node
        
        self._remove_single_child_root()
        return self.T
    
    
    def _MARK(self, x):
        
        for v in self.G.neighbors(x):
            if v in self.already_in_T:
                self.marked.add(self.leaf_map[v])
                self.mark_counter += 1
                
        queue = deque(self.marked)
        
        while queue:                        # contains only d(u)=md(u) nodes
            u = queue.popleft()
            self.marked.remove(u)           # unmark u
            self.unmark_counter += 1
            u.aux_counter = 0               # md(u) <- 0
            if u is not self.T.root:
                w = u.parent                # w <- parent(u)
                if w not in self.marked:
                    self.marked.add(w)      # mark w
                    self.mark_counter += 1
                w.aux_counter += 1
                if w.aux_counter == len(w.children):
                    queue.append(w)
                    
                if w in self.m_u_children:              # append u to list of
                    self.m_u_children[w].appendleft(u)  # marked and unmarked
                else:                                   # children of w
                    self.m_u_children[w] = deque([u])
                    
        if (self.marked and                             # any vertex remained marked
            len(self.T.root.children) == 1 and 
            self.T.root not in self.marked):
            
            self.marked.add(self.T.root)
            self.mark_counter += 1
    
    
    def _find_lowest(self):
        
        R = self.T.root
        y = 'Lambda'
        
        if R not in self.marked:        # R is not marked
            self.error_message = '(iii): R={}'.format(R)
            return False                # G+x is not a cograph (iii)
        else:
            if R.aux_counter != len(R.children) - 1:
                y = R
            self.marked.remove(R)
            R.aux_counter = 0
            u = w = R
        
        while self.marked:              # while there are mark vertices
            u = self.marked.pop()       # choose a arbitrary marked vertex u
            
            if y != 'Lambda':
                self.error_message = '(i) or (ii): y={}'.format(y)
                return False            # G+x is not a cograph (i) or (ii)
            
            if u.label == 'series':
                if u.aux_counter != len(u.children) - 1:
                    y = u
                if u.parent in self.marked:
                    self.error_message = '(i) and (vi): u={}'.format(u)
                    return False        # G+x is not a cograph (i) and (vi)
                else:
                    t = u.parent.parent
            else:
                y = u
                t = u.parent
            u.aux_counter = 0           # u was already unmarked above
            
            # check if the u-w path is part of the legitimate alternating path
            while t is not w:
                if t is R:
                    self.error_message = '(iv): t={}'.format(t)
                    return False        # G+x is not a cograph (iv)
                
                if t not in self.marked:
                    self.error_message = '(iii), (v) or (vi): t={}'.format(t)
                    return False        # G+x is not a cograph (iii), (v) or (vi)
                
                if t.aux_counter != len(t.children) - 1:
                    self.error_message = '(ii): t={}'.format(t)
                    return False        # G+x is not a cograph (ii)
                
                if t.parent in self.marked:
                    self.error_message = '(i): t={}'.format(t)
                    return False        # G+x is not a cograph (i)
                
                self.marked.remove(t)   # unmark t
                t.aux_counter = 0       # reset md(t)
                t = t.parent.parent
                
            w = u                       # rest w for next choice of marked vertex
        
        return u
    
    
    def _remove_single_child_root(self):
        
        if len(self.T.root.children) == 1:
        
            new_root = self.T.root.children[0]
            new_root.detach()
            self.T.root = new_root