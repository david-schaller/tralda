# -*- coding: utf-8 -*-

"""
Implementation of a greedy solution for the cluster deletion problem for
cographs, and the complete multipartite graph completion problem.
"""

import itertools
import random

import networkx as nx

from tralda.datastructures.tree import Tree, LCA
from tralda.utils.graph_tools import complete_multipartite_graph_from_sets
from tralda.cograph.detection import LinearCographDetector


__author__ = 'David Schaller'


def to_cograph(cotree):
    """Returns the cograph corresponding to the cotree.
    
    Parameters
    ----------
    cotree : Tree
        A cotree, i.e., a Tree instance with inner vertex labels 'series' and
        'parallel'.
    
    Returns
    -------
    nexworkx.Graph
        The corresponding cograph with the leaf labels as vertices.
    """
    
    leaves = cotree.leaf_dict()
    G = nx.Graph()
    
    for v in leaves[cotree.root]:
        G.add_node(v.label)
    
    for u in cotree.preorder():
        if u.label == 'series':
            for v1, v2 in itertools.combinations(u.children, 2):
                for l1, l2 in itertools.product(leaves[v1], leaves[v2]):
                    G.add_edge(l1.label, l2.label)
    
    return G


def to_cotree(G):
    """Checks if a graph is a cograph and returns its cotree.
    
    Linear O(|V| + |E|) implementation.
    
    Parameters
    ----------
    G : nexworkx.Graph
        A graph.
    
    Returns
    -------
    Tree or bool
        The cotree representation of the graph, or False if it is not a cograph.
    
    References
    ----------
    .. [1] D. G. Corneil, Y. Perl, and L. K. Stewart.
       A Linear Recognition Algorithm for Cographs.
       In: SIAM J. Comput., 14(4), 926–934 (1985).
       doi: 10.1137/0214065
    """
    
    lcd = LinearCographDetector(G)
    return lcd.recognition()


def complement_cograph(cotree, inplace=False):
    """Returns the cotree of the complement cograph.
    
    Parameters
    ----------
    cotree : Tree
        A cotree, i.e., a Tree instance with inner vertex labels 'series' and
        'parallel'.
    
    Returns
    -------
    Tree
        The cotree of the complement cograph.
    """
    
    tree = cotree if inplace else cotree.copy()
    
    for v in tree.inner_nodes():
        v.label = 'series' if v.label == 'parallel' else 'parallel'
    
    return tree
    
    
def paths_of_length_2(cotree):
    """Generator for all paths of length 2 (edges) in the cograph.
    
    Parameters
    ----------
    cotree : Tree
        A cotree, i.e., a Tree instance with inner vertex labels 'series' and
        'parallel'.
    
    Yields
    ------
    tuple of three TreeNode instance
        All paths of length 2 in the corresponding cograph.
    """
    
    leaves = cotree.leaf_dict()
    lca = LCA(cotree)
    
    for u in cotree.inner_nodes():
        
        if u.label == 'parallel':
            continue
        
        for v1, v2 in itertools.permutations(u.children, 2):
            for t1, t2 in itertools.combinations(leaves[v1], 2):
                if lca(t1, t2).label == 'parallel':
                    for t3 in leaves[v2]:
                        yield t1, t3, t2
    

def random_cotree(N, force_series_root=False):
    """Creates a random cotree.
    
    Parameters
    ----------
    N : int
        The number of leaves in the resulting tree.
    force_series_root : bool
        If True, the cograph of the resulting cotree will be connected,
        otherwise it may be disconnected; the default is False.
    
    Returns
    -------
    Tree
        A cotree, i.e., a Tree instance with inner vertex labels 'series' and
        'parallel'.
    """
    
    cotree = Tree.random_tree(N)
    
    # assign labels ('series', 'parallel')
    for v in cotree.preorder():
        if v.is_leaf():
            continue
        elif v.parent is None:
            if force_series_root:
                v.label = 'series'
            else:
                v.label = 'series' if random.random() < 0.5 else 'parallel'
        else:
            v.label = 'series' if v.parent.label == 'parallel' else 'parallel'
            
    return cotree


def cluster_deletion(cograph):
    """Cluster deletion for cographs.
    
    Returns a partition of a cograph into disjoint cliques with a minimal
    number of edges between the cliques.
    
    Parameters
    ----------
    cograph : networkx.Graph or Tree
        The cograph for which cluster deletion shall be performed.
    
    Returns
    -------
    list of lists
        A partition where each sublist corresponds to a clique in a solution
        of the cluster deletion problem.
    
    Raises
    ------
    RuntimeError
        If the input is not a valid cograph or cotree.
    
    References
    ----------
    .. [1] Gao Y, Hare DR, Nastos J (2013) The cluster deletion problem for
    cographs. Discrete Math 313(23):2763–2771, DOI 10.1016/j.disc.2013.08.017.
    .. [2] Schaller D, Lafond M, Stadler PF, Wieseke N, Hellmuth M (2020)
    Indirect Identification of Horizontal Gene Transfer. (preprint)
    arXiv:2012.08897
    """
    
    cotree = cograph if isinstance(cograph, Tree) else to_cotree(cograph)
    
    if not cotree:
        raise RuntimeError('not a valid cograph/cotree')
    
    P = {}
    
    for u in cotree.postorder():
        
        P[u] = []
        if not u.children:
            P[u].append([u.label])
        
        elif u.label == 'parallel':
            for v in u.children:
                P[u].extend(P[v])
                
            # naive sorting can be replaced by k-way merge-sort
            P[u].sort(key=len, reverse=True)
        
        elif u.label == 'series':
            for v in u.children:
                for i, Q_i in enumerate(P[v]):
                    if i >= len(P[u]):
                        P[u].append([])
                    P[u][i].extend(Q_i)
        
        else:
            raise RuntimeError('invalid cotree')
    
    return P[cotree.root]


def complete_multipartite_completion(cograph, supply_graph=False):
    """Complete multipartite graph completion for cographs.
    
    Returns a partition of the vertex set corresponding to the (maximal)
    independent sets in an optimal edge completion of the cograph to a
    complete multipartite graph.
    
    Parameters
    ----------
    cograph : networkx.Graph or Tree
        The cograph for which complete multipartite graph completion shall be
        performed.
    supply_graph : bool, optional
        If True, the solution is additionally returned as a NetworkX Graph.
    
    Returns
    -------
    list of lists
        A partition where each sublist corresponds to a (maximal) independent
        set in a solution of the complete multipartite graph completion problem.
    networkx.Graph, optional
        The solution as a graph.
    
    Raises
    ------
    RuntimeError
        If the input is not a valid cograph or cotree.
    """
    
    cotree = cograph if isinstance(cograph, Tree) else to_cotree(cograph)
    
    if not cotree:
        raise RuntimeError('not a valid cograph/cotree')
        
    # complete multipartite graph completion is equivalent to 
    # cluster deletion in the complement cograph
    compl_cotree = complement_cograph(cotree, inplace=False)
    
    # clusters are then equivalent to the maximal independent sets
    independent_sets = cluster_deletion(compl_cotree)
    
    if not supply_graph:
        return independent_sets
    else:
        return (independent_sets,
                complete_multipartite_graph_from_sets(independent_sets))
