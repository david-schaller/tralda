# -*- coding: utf-8 -*-

"""
Linear-time cograph detection.  
"""


from collections import deque

import networkx as nx

from tralda.datastructures.tree import Tree, TreeNode, LCA


__author__ = 'David Schaller'


class LinearCographDetector:
    """Linear cograph detection and cotree costruction.
    
    References
    ----------
    .. [1] D. G. Corneil, Y. Perl, and L. K. Stewart.
       A Linear Recognition Algorithm for Cographs.
       In: SIAM J. Comput., 14(4), 926–934 (1985).
       doi: 10.1137/0214065
    """
    
    def __init__(self, G):
        """Constructor of LinearCographDetector class.
        
        Parameters
        ----------
        G : networkx.Graph
            A graph.
        """
        
        if not isinstance(G, nx.Graph):
            raise TypeError('not a NetworkX Graph')
        
        self.G = G
        self.V = [v for v in G.nodes()]
        
        self.T = Tree(None)
        self.already_in_T = set()
        self.leaf_map = {}
        self.node_counter = 0
        
        self.marked = set()
        self.m_u_children = {}              # lists of marked and unmarked children
        self.mark_counter = 0
        self.unmark_counter = 0
        self.md = {}
        
        self.error_message = ''
    
    
    def recognition(self):
        """Run cotree construction.
        
        Returns
        -------
        Tree or bool
            The cotree representation of the graph, or False if it is not a
            cograph.
        """
        
        if len(self.V) == 0:
            raise RuntimeError('empty graph in cograph recognition')
            return self.T
        
        elif len(self.V) == 1:
            self.T.root = TreeNode(label=self.V[0])
            self.md[self.T.root] = 0
            return self.T
        
        v1, v2 = self.V[0], self.V[1]
        self.already_in_T.update([v1, v2])
        
        R = TreeNode(label='series')
        self.md[R] = 0
        self.T.root = R
        
        if self.G.has_edge(v1, v2):
            v1_node = TreeNode(label=v1)
            v2_node = TreeNode(label=v2)
            self.md[v1_node] = 0
            self.md[v2_node] = 0
            R.add_child(v1_node)
            R.add_child(v2_node)
            self.node_counter = 3
        else:
            N = TreeNode(label='parallel')
            self.md[N] = 0
            R.add_child(N)
            v1_node = TreeNode(label=v1)
            v2_node = TreeNode(label=v2)
            self.md[v1_node] = 0
            self.md[v2_node] = 0
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
                x_node = TreeNode(label=x)
                self.md[x_node] = 0
                R.add_child(x_node)
                self.node_counter += 1
                self.leaf_map[x] = x_node
                continue
            # no nodes in T were marked and unmarked
            elif self.mark_counter == 0:
                # d(R)=1
                if len(self.T.root.children) == 1:
                    N = self.T.root.children[0]
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
                    N.add_child(x_node)
                    self.node_counter += 1
                else:
                    R_old = self.T.root
                    R_new = TreeNode(label='series')
                    self.md[R_new] = 0
                    N = TreeNode(label='parallel')
                    self.md[N] = 0
                    R_new.add_child(N)
                    N.add_child(R_old)
                    self.T.root = R_new
                    
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
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
                if w.is_leaf():
                    new_node = TreeNode(label='series')
                    self.md[new_node] = 0
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)
                    
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
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
                if w.is_leaf():
                    new_node = TreeNode(label='parallel')
                    self.md[new_node] = 0
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)
                    
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
                    w.add_child(x_node)
                    self.node_counter += 1
            
            else:
                y = TreeNode(label=u.label)
                self.md[y] = 0
                for a in self.m_u_children[u]:
                    u.remove_child(a)
                    y.add_child(a)
                    
                if u.label == 'parallel':
                    new_node = TreeNode(label='series')
                    self.md[new_node] = 0
                    u.add_child(new_node)
                    
                    new_node.add_child(y)
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
                    new_node.add_child(x_node)
                else:
                    par = u.parent
                    if par is not None:             # u was the root of T
                        par.remove_child(u)
                        par.add_child(y)
                    else:
                        self.T.root = y             # y becomes the new root
                    
                    new_node = TreeNode(label='parallel')
                    self.md[new_node] = 0
                    y.add_child(new_node)
                    new_node.add_child(u)
                    x_node = TreeNode(label=x)
                    self.md[x_node] = 0
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
            self.md[u] = 0                  # md(u) <- 0
            if u is not self.T.root:
                w = u.parent                # w <- parent(u)
                if w not in self.marked:
                    self.marked.add(w)      # mark w
                    self.mark_counter += 1
                self.md[w] += 1
                if self.md[w] == len(w.children):
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
            if self.md[R] != len(R.children) - 1:
                y = R
            self.marked.remove(R)
            self.md[R] = 0
            u = w = R
        
        while self.marked:              # while there are mark vertices
            u = self.marked.pop()       # choose a arbitrary marked vertex u
            
            if y != 'Lambda':
                self.error_message = '(i) or (ii): y={}'.format(y)
                return False            # G+x is not a cograph (i) or (ii)
            
            if u.label == 'series':
                if self.md[u] != len(u.children) - 1:
                    y = u
                if u.parent in self.marked:
                    self.error_message = '(i) and (vi): u={}'.format(u)
                    return False        # G+x is not a cograph (i) and (vi)
                else:
                    t = u.parent.parent
            else:
                y = u
                t = u.parent
            self.md[u] = 0           # u was already unmarked above
            
            # check if the u-w path is part of the legitimate alternating path
            while t is not w:
                if t is R:
                    self.error_message = '(iv): t={}'.format(t)
                    return False        # G+x is not a cograph (iv)
                
                if t not in self.marked:
                    self.error_message = '(iii), (v) or (vi): t={}'.format(t)
                    return False        # G+x is not a cograph (iii), (v) or (vi)
                
                if self.md[t] != len(t.children) - 1:
                    self.error_message = '(ii): t={}'.format(t)
                    return False        # G+x is not a cograph (ii)
                
                if t.parent in self.marked:
                    self.error_message = '(i): t={}'.format(t)
                    return False        # G+x is not a cograph (i)
                
                self.marked.remove(t)   # unmark t
                self.md[t] = 0          # reset md(t)
                t = t.parent.parent
                
            w = u                       # rest w for next choice of marked vertex
        
        return u
    
    
    def _remove_single_child_root(self):
        
        if len(self.T.root.children) == 1:
        
            new_root = self.T.root.children[0]
            new_root.detach()
            self.T.root = new_root