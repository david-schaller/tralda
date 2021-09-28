# -*- coding: utf-8 -*-

"""Algorithms for building trees from rooted triple sets."""


import itertools
import networkx as nx

from tralda.datastructures.Tree import Tree, TreeNode
from tralda.datastructures.Partition import Partition


__author__ = 'David Schaller'


def aho_graph(R, L, weighted=False, triple_weights=None):
    """Construct the auxiliary graph (Aho graph) for BUILD.
        
    Edges {a,b} are optionally weighted by the number of occurrences, resp.,
    sum of weights of triples of the form ab|x or ba|x.
    
    Parameters
    ----------
    R : collection of tuples
        A collection of triples.
    L : collection
        A collection of leaf labels.
    weighted : bool, optional
        If True, weight the edges in the resulting graph accordingto the
        number of corresponding triples (and their weights).
        The default is False.
    triple_weights : dict with tuples as keys and float values, optional
        The weights of the triples. The default is None, in which case every
        triple has weight one.
    
    Returns
    -------
    networkx.Graph
        The undirected Aho graph.
    """
    
    G = nx.Graph()
    G.add_nodes_from(L)

    for a, b, c in R:
        if not G.has_edge(a, b):
            G.add_edge(a, b)
        
        if weighted:
            if triple_weights:
                G[a][b]['weight'] = G[a][b].get('weight', 0.0) + \
                                    triple_weights[a, b, c]
            else:
                G[a][b]['weight'] = G[a][b].get('weight', 0.0) + 1.0
    
    return G


def _triple_connect(G, t):
    
    G.add_edge(t[0], t[1])
    G.add_edge(t[0], t[2])
    

def mtt_partition(L, R, F):
    """Construct the auxiliary partition for the MTT algorithm.
    
    Parameters
    ----------
    L : collection
        A collection of leaf labels.
    R : collection of tuples
        A collection of required triples.
    F : collection of tuples
        A collection of forbidden triples.
        
    Returns
    -------
    Partition
        The auxiliary partition for the MTT algorithm.
    networkx.Graph
        A graph representation of the auxiliary partition.
    """
    
    # auxiliary graph initialized as Aho graph
    G = aho_graph(R, L, weighted=False)
    
    # auxiliary partition
    P = Partition(nx.connected_components(G))
    
    if len(P) == 1:
        return P, G
    
    # aux. set of forbidden triples
    S = {t for t in F if P.separated_xy_z(*t)}
    
    # lookup of forb. triples to which u belongs
    L = {u: [] for u in L}
    for t in F:
        for u in t:
            L[u].append(t)
    
    while S:
        t = S.pop()
        _triple_connect(G, t)
        
        # merge returns the smaller of the two merged sets
        smaller_set = P.merge(t[0], t[2])
        
        # update S by traversing the L(u)
        for u in smaller_set:
            for t in L[u]:
                if t in S and not P.separated_xy_z(*t):
                    S.remove(t)
                    _triple_connect(G, t)
                elif t not in S and P.separated_xy_z(*t):
                    S.add(t)
    
    return P, G


class Build:
    """BUILD algorithm.
    
    References
    ----------
    .. [1] A. V. Aho, Y. Sagiv, T. G. Szymanski, and J. D. Ullman.
       Inferring a tree from lowest common ancestors with an application to
       the optimization of relational expressions.
       SIAM Journal on Computing, 10:405–421, 1981.
       doi: 10.1137/0210030.
    """
    
    def __init__(self, R, L, mincut=False, 
                 weighted_mincut=False, triple_weights=None):
        
        self.R = R
        self.L = L
        self.mincut = mincut
        self.weighted_mincut = weighted_mincut
        self.triple_weights = triple_weights
    
    
    def build_tree(self, return_root=False, print_info=False):
        """Build a tree displaying all triples in R if possible.
        
        Parameters
        ----------
        return_root : bool
            If True, return 'TreeNode' instead of 'Tree' instance.
        print_info : bool
            Print information about inconsistencies.
        
        Returns
        -------
        Tree or None
            The BUILD tree on leaf set L displaying all triples in R if the
            triple set R is consistent.
        """
        
        self.cut_value = 0
        self.cut_list = []
        self.print_info = print_info
        
        root = self._aho(self.L, self.R)
        if not root:
            if self.print_info: print('no such tree exists')
            return None
        else:
            return root if return_root else Tree(root)
    
    
    def _aho(self, L, R):
        """Recursive Aho-algorithm."""
        
        # trivial cases: only one or two leaves left in L
        if len(L) == 1:
            leaf = L.pop()
            return TreeNode(label=leaf)
        elif len(L) == 2:
            node = TreeNode()
            for _ in range(2):
                leaf = L.pop()
                node.add_child(TreeNode(label=leaf))
            return node
            
        help_graph = aho_graph(R, L, weighted=self.weighted_mincut,
                                     triple_weights=self.triple_weights)
        conn_comps = self._connected_components(help_graph)
        
        # return False if less than 2 connected components
        if len(conn_comps) <= 1:
            if self.print_info: print('Connected component:\n', conn_comps)
            return False
        
        # otherwise proceed recursively
        node = TreeNode()                   # place new inner node
        for cc in conn_comps:
            Li = set(cc)                    # list/dictionary --> set
            Ri = []
            for t in R:                     # construct triple subset
                if Li.issuperset(t):
                    Ri.append(t)
            Ti = self._aho(Li, Ri)          # recursive call
            if not Ti:
                return False                # raise False to previous call
            else:
                node.add_child(Ti)          # add root of the subtree
   
        return node
    
    
    def _connected_components(self, aho_graph):
        """Determines the connected components of the graph.
        
        And optionally executes a min cut if there is only one component."""
        
        conn_comps = list(nx.connected_components(aho_graph))
        if (not self.mincut) or len(conn_comps) > 1:
            return conn_comps
        else:
            # Stoer–Wagner algorithm
            cut_value, partition = nx.stoer_wagner(aho_graph)
            self.cut_value += cut_value
            if len(partition[0]) < len(partition[1]):
                smaller_comp = partition[0]
            else:
                smaller_comp = partition[1]
            for edge in aho_graph.edges():
                if ((edge[0] in smaller_comp and edge[1] not in smaller_comp)
                    or
                    (edge[1] in smaller_comp and edge[0] not in smaller_comp)):
                    self.cut_list.append(edge)
            return partition
        

class MTT:
    """MTT algorithm.
    
    References
    ----------
    .. [1] Y.-J. He, T. N. D. Huynh, J. Jansson, andW.-K. Sung.
       Inferring phylogenetic relationships avoiding forbidden rooted triplets.
       Journal of Bioinformatics and Computational Biology, 4: 59–74, 2006.
       doi: 10.1142/S0219720006001709.
    """
    
    def __init__(self, R, L, F=None):
        """Constructor for class MTT.
        
        Parameters
        ----------
        L : collection
            A collection of leaf labels.
        R : collection of tuples
            A collection of required triples.
        F : collection of tuples, optional
            A collection of forbidden triples.
        """
        
        self.R = R
        self.L = L
        
        # forbidden triples --> activates MTT if non-empty
        self.F = F
    
    
    def build_tree(self, return_root=False):
        """Build a tree displaying all triples in R if possible.
        
        Parameters
        ----------
        return_root : bool
            If True, return 'TreeNode' instead of 'Tree' instance.
        
        Returns
        -------
        Tree or bool
            A tree on leaf set L displaying all triples in R and none in F,
            if such a tree exists; False otherwise.
        """
        
        self.total_cost = 0
        
        if self.F:
            root = self._mtt(self.L, self.R, self.F)
        else:
            root = self._aho(self.L, self.R)
            
        return root if return_root else Tree(root)
    
    
    def _trivial_case(self, L):
        
        if len(L) == 1:
            leaf = L.pop()
            return TreeNode(label=leaf)
        
        elif len(L) == 2:
            node = TreeNode()
            for _ in range(2):
                leaf = L.pop()
                node.add_child(TreeNode(label=leaf))
            return node
    
    
    def _aho(self, L, R):
        """Recursive Aho algorithm."""
        
        # trivial case: one or two leaves left in L
        if len(L) <= 2:
            return self._trivial_case(L)
            
        aux_graph = aho_graph(R, L)
        partition = list(nx.connected_components(aux_graph))
        
        if len(partition) < 2:
            return False
        
        node = TreeNode()                   # place new inner node
        for s in partition:
            Li, Ri = set(s), []
            for t in R:                     # construct triple subset
                if Li.issuperset(t):
                    Ri.append(t)
            Ti = self._aho(Li, Ri)          # recursive call
            if not Ti:
                return False                # raise False to previous call
            else:
                node.add_child(Ti)          # add roots of the subtrees
   
        return node
    
    
    def _mtt(self, L, R, F):
        """Recursive MTT algorithm."""
        
        # trivial case: one or two leaves left in L
        if len(L) <= 2:
            return self._trivial_case(L)
        
        partition, aux_graph = mtt_partition(L, R, F)
        
        if len(partition) < 2:
            return False
        
        node = TreeNode()                   # place new inner node
        for s in partition:
            Li, Ri, Fi = set(s), [], []
            for Xi, X in ((Ri, R), (Fi, F)):
                for t in X:
                    if Li.issuperset(t):
                        Xi.append(t)
            Ti = self._mtt(Li, Ri, Fi)      # recursive call
            if not Ti:
                return False                # raise False to previous call
            else:
                node.add_child(Ti)          # add roots of the subtrees
   
        return node


def greedy_BUILD(R, L, triple_weights=None, return_root=False):
    """Greedy heuristic for triple consistency.
    
    Add triples one by one and checks consistency via BUILD.
    
    Parameters
    ----------
    R : collection of tuples
        A collection of triples.
    L : collection
        A collection of leaf labels.
    triple_weights : dict, optional
        Weights for the triples; default is None in which case all triples are
        uniformly weighted.
    return_root : bool, optional
        If True, return 'TreeNode' instead of 'Tree' instance.
        The default is False.
    
    Returns
    -------
    Tree or TreeNode
    """
        
    if triple_weights:
        triples = sorted(R,
                         key=lambda triple: triple_weights[triple],
                         reverse=True)
    else:
        triples = R
            
    consistent_triples = []
    root = None
    
    for t in triples:
        consistent_triples.append(t)
        build = Build(consistent_triples, L, mincut=False)
        new_root = build.build_tree(return_root=True)
        if new_root:
            root = new_root
        else:
            consistent_triples.pop()
    
    return root if return_root else Tree(root)
    

def best_pair_merge_first(R, L, triple_weights=None, return_root=False):
    """Wu’s (2004) Best-Pair-Merge-First (BPMF) heuristic.
    
    Modified version by Byrka et al. (2010) and added weights.
    
    Parameters
    ----------
    R : collection of tuples
        A collection of triples.
    L : collection
        A collection of leaf labels.
    triple_weights : dict, optional
        Weights for the triples; default is None in which case all triples are
        uniformly weighted.
    return_root : bool, optional
        If True, return 'TreeNode' instead of 'Tree' instance.
        The default is False.
    
    Returns
    -------
    Tree or TreeNode
    """
    
    # initialization
    nodes = {TreeNode(label=leaf): {leaf} for leaf in L}
    leaf_to_node = {}
    
    for node in nodes:
        leaf_to_node[node.label] = node
    
    # merging
    for i in range(len(L)-1):
        
        score = {(S_i, S_j): 0
                 for S_i, S_j in itertools.combinations(nodes.keys(), 2)}
        
        for x, y, z in R:
            
            w = triple_weights[(x,y,z)] if triple_weights else 1
            
            S_i, S_j, S_k = (leaf_to_node[x],
                             leaf_to_node[y],
                             leaf_to_node[z])
            
            if (S_i is not S_j) and (S_i is not S_k) and (S_j is not S_k):
                
                if (S_i, S_j) in score:
                    score[(S_i, S_j)] += 2 * w
                else:
                    score[(S_j, S_i)] += 2 * w
                    
                if (S_i, S_k) in score:
                    score[(S_i, S_k)] -= w
                else:
                    score[(S_k, S_i)] -= w
                    
                if (S_j, S_k) in score:
                    score[(S_j, S_k)] -= w
                else:
                    score[(S_k, S_j)] -= w
        
        current_max = float('-inf')
        S_i, S_j = None, None
        
        for pair, pair_score in score.items():
            
            if pair_score > current_max:
                current_max = pair_score
                S_i, S_j = pair
        
        # create new node S_k connecting S_i and S_j
        S_k = TreeNode()
        S_k.add_child(S_i)
        S_k.add_child(S_j)
        
        nodes[S_k] = nodes[S_i] | nodes[S_j]    # set union
        for leaf in nodes[S_k]:
            leaf_to_node[leaf] = S_k
        
        del nodes[S_i]
        del nodes[S_j]
    
    if len(nodes) != 1:
        raise RuntimeError('more than 1 node left')
    
    root = next(iter(nodes))
    return root if return_root else Tree(root)


def minimal_identifying_triple_set(tree):
    """Construct a minimal set of triples that identifies the tree.
    
    Parameters
    ----------
    tree : Tree
    
    Yields
    -------
    tuple of three ThreeNode instances
    
    References
    ----------
    - Stefan Grünewald, Mike Steel and M. Shel Swenson. Closure operations in
      phylogenetics. Mathematical Biosciences 208 (2007) 521–537.
      DOI: 10.1016/j.mbs.2006.11.005
    """
    
    # representative leaf for each vertex
    repres = {}
    
    for v in tree.postorder():
        repres[v] = repres[v.children[0]] if v.children else v.label
    
    for u, v in tree.inner_edges():
        W = [c for c in v.children]
        for v2 in u.children:
            if v is not v2:
                for i in range(len(W)-1):
                    yield (repres[W[i]], repres[W[i+1]],
                           repres[v2])
                    

def tree_profile_to_triples(trees):
    """Construct leaf set and reprentative triples from a profile of trees.
    
    Parameters
    ----------
    trees : sequence of Tree instances
    
    Returns
    -------
    tuple of two sets
        The first set contains all leaf labels that appear in the tree profile.
        The second set contains a representive set of triples.
    """
    
    L = set()
    R = set()
        
    for tree in trees:
        
        L.update(l.label for l in tree.leaves())
        R.update((*sorted(t[:2]), t[2])
                 for t in minimal_identifying_triple_set(tree))
    
    return L, R


def BUILD_supertree(trees):
    """Supertree construction based on the BUILD algorithm.
    
    Parameters
    ----------
    trees : sequence of Tree instances
    
    Returns
    -------
    Tree or bool
        A supertree for the input trees if existent, False otherwise.
    """
    
    L, R = tree_profile_to_triples(trees)
    
    build = Build(R, L, mincut=False)
    tree = build.build_tree()
    
    return tree if tree else False    
    