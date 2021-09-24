# -*- coding: utf-8 -*-

"""
Implementation of a greedy solution for the cluster deletion problem for
cographs, and the complete multipartite graph completion problem.
"""


__author__ = 'David Schaller'


from tralda.datastructures.Tree import Tree
from tralda.cograph.Cograph import to_cotree, complement_cograph
from tralda.tools.GraphTools import complete_multipartite_graph_from_sets


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
