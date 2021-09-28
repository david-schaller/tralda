# -*- coding: utf-8 -*-

"""Fast computation of a common refinement of trees on the same leaf set."""

from collections import deque

from tralda.datastructures.Tree import Tree, TreeNode
from tralda.tools.TreeTools import assert_leaf_sets_equal


__author__ = 'David Schaller'


def linear_common_refinement(trees):
    """Minimal common refinement for a set of trees with the same leaf set.
    
    All input trees must be phylogenetic and have the same set of leaf labels.
    In each tree, the leaves' label attributes must be set and unique.
    
    Parameters
    ----------
    trees : sequence of Tree instances
    
    Returns
    -------
    Tree or bool
        A common refinement of the input trees if existent, False otherwise.
        
    Raises
    ------
    TypeError
        In case of input instances that are not of type Tree.
    RuntimeError
        If the sequence contains empty trees or the tree do not share the same
        set of leaves.
        
    References
    ----------
    .. [1] D. Schaller, M. Hellmuth, P.F. Stadler (2021)
    A Simple Linear-Time Algorithm for the Common Refinement of Rooted
    Phylogenetic Trees on a Common Leaf Set.
    arXiv:2107.00072 [cs.DS]
    """
    
    cr = LinCR(trees)
    return cr.run()


class LinCR:
    """Minimal common refinement for a set of trees with the same leaf set.
    
    References
    ----------
    .. [1] D. Schaller, M. Hellmuth, P.F. Stadler (2021)
    A Simple Linear-Time Algorithm for the Common Refinement of Rooted
    Phylogenetic Trees on a Common Leaf Set.
    arXiv:2107.00072 [cs.DS]
    """
    
    def __init__(self, trees):
        
        self.L = assert_leaf_sets_equal(trees)
        
        if not self.L:
            raise RuntimeError('trees must have the same leaf labels')
        
        self.trees = trees
        
        
    def run(self):
        
        self._initialize()
        self._build_tree()
        
        if self.T:
            if not self.T.is_phylogenetic():
                return False
            if not self._check_displayed():
                return False
            return self.T
        else:
            return False
    
    
    def _initialize(self):
        """Initialize the lookup tables."""
    
        self.T = None           # resulting tree (candidate)
        
        self._leaf_set_cardinalities()
                                
        self.p = [{} for _ in range(len(self.trees))]
                                # v --> lca_Ti (L(Ti(v)))
        
        self.J = {}             # vertex in resulting tree --> indices of
                                # the trees with corresponding vertex
        self.vi_to_v = {}       # inner vertex in input tree --> 
                                # corresponding vertex in resulting tree
        
        self.Q = deque()        # queue
        self.V = set()          # vertices in resulting tree
        
        self.label_dict = {}
        
        for label in self.L:
            v = TreeNode(label=label)
            self.Q.append(v)
            self.V.add(v)
            self.label_dict[label] = v
            self.J[v] = {i: None for i in range(len(self.trees))}
            self.l[v] = 1 
        
        for i, T_i in enumerate(self.trees):
            for v_i in T_i.leaves():
                v = self.label_dict[v_i.label]
                self.p[i][v] = v_i
                self.J[v][i] = v_i
                self.vi_to_v[v_i] = v
        
    
    def _leaf_set_cardinalities(self):
        """Compute the number of leaves below each vertex."""
        
        self.l = {}
        
        for T_i in self.trees:
            for v in T_i.postorder():
                if v.is_leaf():
                    self.l[v] = 1
                else:
                    self.l[v] = sum(self.l[w] for w in v.children)
    
    
    def _build_tree(self):
        """Constructs the candidate for the common refinement tree."""
        
        self.root = None
        
        while self.Q:
            
            v = self.Q.popleft()
            u = None
            l_min = len(self.L)
            J_u = {}
            
            for i in range(len(self.trees)):
                
                if i in self.J[v]:
                    u2 = self.J[v][i].parent
                else:
                    u2 = self.p[i][v]
                
                if u is None or self.l[u2] < l_min:
                    u = u2
                    l_min = self.l[u2]
                    J_u.clear()
                    J_u[i] = u2
                elif self.l[u2] == l_min:
                    J_u[i] = u2
            
            if self.l[v] < l_min:
                if u in self.vi_to_v:
                    u = self.vi_to_v[u]
                else:
                    u = TreeNode()
                    self.J[u] = J_u
                    for i, u_i in J_u.items():
                        self.vi_to_v[u_i] = u
                        self.J[u][i] = u_i
                    self.l[u] = l_min
                u.add_child(v)
            else:
                return False
            
            if u not in self.V and l_min < len(self.L):
                
                self.Q.append(u)
                self.V.add(u)
                if len(self.V) > 2 * len(self.L) - 2:
                    return False
                
                for i in range(len(self.trees)):
                    if i in self.J[u]:
                        self.p[i][u] = self.J[u][i]
                    elif i in self.J[v]:
                        self.p[i][u] = self.J[v][i].parent
                    else:
                        self.p[i][u] = self.p[i][v]
                        
            elif l_min == len(self.L):
                self.root = u
        
        if not self.root:
            raise RuntimeError('could not determine root')
        
        self.T = Tree(self.root)
    
    
    def _check_displayed(self):
        """Checks whether all input trees are displayed by the constructed
        tree."""
        
        for i, T_i in enumerate(self.trees):
            
            T_copy, v_copy = self.T.copy(mapping=True)
            
            to_contract = [(v_copy[v].parent, v_copy[v])
                           for v in self.T.inner_nodes()
                           if i not in self.J[v] and v.parent]
            
            T_copy.contract(to_contract)
            
            for v_i in T_i.preorder():
                if v_i not in self.vi_to_v:
                    return False
                elif v_i.parent is None:
                    if v_copy[self.vi_to_v[v_i]].parent is not None:
                        return False
                elif v_i.parent not in self.vi_to_v:
                    return False
                elif v_copy[self.vi_to_v[v_i.parent]] is not \
                     v_copy[self.vi_to_v[v_i]].parent:
                    return False
        
        return True
