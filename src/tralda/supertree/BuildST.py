# -*- coding: utf-8 -*-

""" 
Implementation of the BuildST algorithm described by Deng & Fernández-Baca.

Builds a supertree from a profile of phylogenetic trees. Uses the algorithm
described by Deng and Fernández-Baca in 2016. The class 'BuildST' accepts a 
list of trees and computes the supertree if it exists.


References:
    - Yun Deng and David Fernández-Baca. Fast Compatibility Testing for
      Rooted Phylogenetic Trees. 27th Annual Symposium on Combinatorial
      Pattern Matching (CPM 2016). DOI: 10.4230/LIPIcs.CPM.2016.12
"""


from tralda.datastructures.Tree import Tree, TreeNode
from tralda.datastructures.DoublyLinkedList import DLList
from tralda.datastructures.hdtgraph.DynamicGraph import HDTGraph


__author__ = 'David Schaller'


def build_st(trees):
    """Supertree construction based on BuildST algorithm.
    
    Note that the graph connecting to trees in the input set whenever they
    share at least one leaf label must be connected. Otherwise, the
    implementation returns False, even though a supertree may exist.
    """
    
    st_builder = BuildST(trees)
    return st_builder.run()


class BuildST:
    """BuildST algorithm.
    
    Note that the graph connecting to trees in the input set whenever they
    share at least one leaf label must be connected. Otherwise, the
    implementation returns False, even though a supertree may exist.
    
    Parameters
    ----------
    trees : sequence of Tree instances
    
    References
    ----------
    - Yun Deng and David Fernández-Baca. Fast Compatibility Testing for
      Rooted Phylogenetic Trees. 27th Annual Symposium on Combinatorial
      Pattern Matching (CPM 2016). DOI: 10.4230/LIPIcs.CPM.2016.12
    """
    
    def __init__(self, trees):
        """Constructor for BuildST algorithm."""
        
        self.trees = trees
        self.Xp = {}                                    # leaf label --> XpNode
        self.marked = set()                             # set of marked nodes
        self.node_to_tree_index = {}
        self.list_pointer = {}                          # node --> DLL element
        self.singleton_pointer = {}                     # node --> DLL element
        self.hdt_graph = HDTGraph()
        
        self.fail_message = ''
        
    
    def run(self):
        """Build the supertree from the given tree list if existent."""
        
        self._prepare_trees()        # tree indices, Xp nodes

        U_init = self._initialize()
        if not U_init:
            return False
        root = self._buildst(U_init)
        if not root:
            return False
        else:
            return Tree(root)
    
    
    def _prepare_trees(self):
        
        for i in range(len(self.trees)):
            tree = self.trees[i]
            for node in tree.preorder():
                # access to tree index i in [k]
                self.node_to_tree_index[node] = i
                if not node.children:
                    if node.label not in self.Xp:
                        self.Xp[node.label] = XpNode(node.label)
                    self.Xp[node.label].leafnodes.append(node)
    
    
    def _initialize(self):
        
        Y = _ConnectedComp(len(self.trees), self, count=len(self.Xp))
        
        for i, tree in enumerate(self.trees):
            
            # Y_init.singleton is the set [k]
            self.singleton_pointer[tree.root] = Y.singleton.append(i)
            
            # append the root to DLList at Y.List[i]
            self.list_pointer[tree.root] = Y.List[i].append(tree.root)
            
            Y.initialize_tree_edges(tree)
            
            # mark root as part of set U_init
            self.marked.add(tree.root)
                          
        # glue together the leafnodes and the labelnodes in Xp
        for labelnode in self.Xp.values():
            for leafnode in labelnode.leafnodes:
                Y.add_edge(labelnode, leafnode)
                
        if self.hdt_graph:
            ett = self.hdt_graph.is_connected()
            if ett:
                Y.representative = self.trees[0].root
            else:
                self.failmessage = 'initialization failed because graph is '\
                                   'not fully connected'
                return False
            
        return Y
    
    
    def _buildst(self, U):
        
        # --------------------------------------------------
        # if |L(U)| = 1 then
        #    return the tree consisting of node r_U,
        #    labeled by the single species in L(U)
        # --------------------------------------------------
        if U.count == 1:
            labelnode = U.get_labelnodes()[0]
            return TreeNode(label=labelnode.label)
        
        r_U = TreeNode()      # create a node r_U
        
        # --------------------------------------------------
        # if |L(U)| = 2 then
        #    return the tree consisting of node r_U and two children
        #    each labeled by a different species in L(U)
        # --------------------------------------------------
        if U.count == 2:
            labels = U.get_labelnodes()
            if len(labels) != 2:
                self.failmessage = 'could not find 2 labeled nodes'
                return
            node1 = TreeNode(label=labels[0].label)
            node2 = TreeNode(label=labels[1].label)
            r_U.add_child(node1)
            r_U.add_child(node2)
            return r_U
        
        # --------------------------------------------------
        # foreach i in [k] such that |U(i)| = 1 do
        #    Let v be the single element in U(i)
        #    U = (U \ {v}) union Ch(v)
        # --------------------------------------------------
        W = [U]                             # list of conn. components
        J = [j for j in U.singleton]        # elements in singleton
        V = [U.List[i][0] for i in J]       # singleton nodes to be deleted
        
        for i in J:
            v_i = U.List[i][0]
            U.List[i].remove_node(self.list_pointer[v_i])
            U.singleton.remove_node(self.singleton_pointer[v_i])
            
            self.marked.discard(v_i)        # unmark v_i and
            for child in v_i.children:      # mark its children
                self.marked.add(child)
                self.list_pointer[child] = U.List[i].append(child)
        
        for v_i in V:
            Y_current = None
            
            # which component in W contains v_i?
            for Y in W:
                if Y.contains_key(v_i):
                    Y_current = Y
                    break
                
            if not Y_current:
                # print("Could not find connected component for v_i", v_i)
                return
            
            for u in v_i.children:
                conn_comp = Y_current.delete_edge(v_i, u)
                if len(conn_comp) == 2:
                    if v_i in conn_comp[0]:
                        
                        # (case 1.1) u is the last child of v_i
                        #            then conn_comp[0] is actually empty
                        if len(conn_comp[0]) == 1:
                            Y_current.set_component(component=conn_comp[1],
                                                    representative=u)
                            break
                        
                        # (case 1.2) two actual components emerge
                        else:
                            Y_current.set_component(component=conn_comp[0],
                                                    representative=v_i)
                            Y_new = _ConnectedComp(Y_current.k, self)
                            Y_new.set_component(component=conn_comp[1], 
                                                representative=u)
                            current_smaller = True \
                                if len(conn_comp[0]) < len(conn_comp[1]) \
                                else False
                    
                    elif v_i in conn_comp[1]:
                        
                        # (case 2.1) u is the last child of v_i
                        #            then conn_comp[1] is actually empty
                        if len(conn_comp[1]) == 1:
                            Y_current.set_component(component=conn_comp[0],
                                                    representative=u)
                            break
                        
                        # (case 2.2) two actual components emerge
                        else:
                            Y_current.set_component(component=conn_comp[1],
                                                    representative=v_i)
                            Y_new = _ConnectedComp(Y_current.k, self)
                            Y_new.set_component(component=conn_comp[0],
                                                representative=u)
                            current_smaller = True \
                                if len(conn_comp[1]) < len(conn_comp[0]) \
                                else False
                    
                    if current_smaller:
                        self._update_conn_comps(Y_current, Y_new,
                                                first_swap=True)
                    else:
                        self._update_conn_comps(Y_new, Y_current)
                    
                    W.append(Y_new)
                    
        # --------------------------------------------------
        # Let W_1, W_2, ..., W_p be the connected components
        # if p = 1 then
        #    return incompatible
        # --------------------------------------------------
        if len(W) == 1:
            self.failmessage = 'the trees are incompatible'
            return False
        
        # --------------------------------------------------
        # foreach j in [p] do
        #    Let t_j = BuildST(W_j)
        #    if t j is a tree then
        #       Add t_j to the set of subtrees of r_U
        #    else
        #       return incompatible
        # --------------------------------------------------
        for W_j in W:
            t_j = self._buildst(W_j)
            if t_j:
                r_U.add_child(t_j)
            else:
                return False
        
        # return the tree with root r_U
        return r_U
    
    
    def _update_conn_comps(self, Y1, Y2, first_swap=False):
        """Updates Y.count, Y.singleton and Y.List of two newly split components.
        
        Component Y1 is supposed to be the smaller one and is scanned for label
        nodes (Xp) and for marked nodes.
        Keyword argument:
            first_swap - if Y1 (smaller component) contains the information
                         about count, List and singleton instead of Y2
                         these attributes are swapped
        """
        
        if first_swap:
            Y1.count, Y2.count = Y2.count, Y1.count
            Y1.List, Y2.List = Y2.List, Y1.List
            Y1.singleton, Y2.singleton = Y2.singleton, Y1.singleton
            
        for v in Y1.keys():
            if isinstance(v, XpNode):
                Y1.count += 1
                Y2.count -= 1
            elif v in self.marked:
                i = self.node_to_tree_index[v]
                Y2.List[i].remove_node(self.list_pointer[v])
                self.list_pointer[v] = Y1.List[i].append(v)
                
                # v becomes singleton
                if len(Y2.List[i]) == 1:
                    singleton_node = Y2.List[i][0]
                    self.singleton_pointer[singleton_node] = Y2.singleton.append(i)
                # index i is no longer singleton, i.e. it must have been a singleton
                elif len(Y2.List[i]) == 0:
                    Y2.singleton.remove_node(self.singleton_pointer[v])
                
                # v becomes singleton
                if len(Y1.List[i]) == 1:                                        
                    self.singleton_pointer[v] = Y1.singleton.append(i)
                # index i is no longer singleton
                elif len(Y1.List[i]) == 2:
                    Y1.singleton.remove_node(self.singleton_pointer[Y1.List[i][0]])


class XpNode:
    """Special type of node for the set Xp (one for each leaf label).""" 
    
    def __init__(self, label):
        
        self.label = label
        self.leafnodes = []         # corresponding treenodes of the profile
    
    def __repr__(self):
        
        return '<XpNodeID:{}, {}>'.format(id(self), self.label)
    
    def __str__(self):
        
        return str(self.label)
        
    
class _ConnectedComp:
    """Connected component for D. & F.-B. algorithm (HDT data structure)."""
    
    def __init__(self, k, buildst, count=0, singleton=None, List=None):
        
        self.k = k
        self.buildst = buildst
        self.hdt_graph = buildst.hdt_graph
        self.count = count
        self.singleton = singleton if singleton else DLList()
        self.List = List if List else [DLList() for i in range(k)]
        
        # to find the component in the HDT datastructure
        self.representative = None
    
    
    def keys(self):
        """Generator for the elements in the component.
        
        Caution: If the HDT datastructure is used this will only work
        if a representative of the component is set correctly.
        """
        
        yield from self.hdt_graph.component_iterator(self.representative)
    
    
    def contains_key(self, key):
        """Check if a given key is element of the component."""
        
        if not self.representative: 
            raise RuntimeError('missing representative for the component (2)')
            
        return self.hdt_graph.connected(self.representative, key)
    
    
    def initialize_tree_edges(self, tree):
        """Add all edges of a tree to the graph datastructure."""
        
        self.hdt_graph.add_loose_tree(tree)
    
    
    def add_edge(self, u, v):
        """Add an edge to the graph datastructure."""
        
        self.hdt_graph.insert_edge(u, v)
    
    
    def delete_edge(self, u, v):
        """Delete an edge and return the 1 or 2 conn. components of u and v."""
        
        self.hdt_graph.delete_edge(u, v)
        ett1 = self.hdt_graph.get_component(u)
        ett2 = self.hdt_graph.get_component(v)
        if ett1 is ett2:
            return [ett1]
        else:
            return [ett1, ett2]
    
    
    def set_component(self, component=None, representative=None):
        """Set the component (self.representative)."""
        
        self.representative = representative
    
    
    def get_labelnodes(self):
        """Return the labelnodes in the connected component."""
        
        result = []
        if not self.representative:
            raise RuntimeError('missing representative for the component (1)')
            
        for node in self.hdt_graph.component_iterator(self.representative):
            if isinstance(node, XpNode):
                result.append(node)
                
        return result
