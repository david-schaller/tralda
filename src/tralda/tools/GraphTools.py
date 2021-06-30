# -*- coding: utf-8 -*-

"""
Graph Tools.
"""

import itertools, random
import numpy as np
import networkx as nx


__author__ = 'David Schaller'


# ----------------------------------------------------------------------------
#                             Adjacency matrix
# ----------------------------------------------------------------------------

def build_adjacency_matrix(G):
    """Return an adjacency matrix."""
    
    # maps node --> row/column index
    index = {node: i for i, node in enumerate(G.nodes())}
    M = np.zeros( (len(index), len(index)), dtype=np.int8)
    
    for x, neighbors in G.adjacency():
        for y in neighbors:
            M[index[x], index[y]] = 1
            
    return M, index


# ----------------------------------------------------------------------------
#                             Graph comparison
# ----------------------------------------------------------------------------

def graphs_equal(G1, G2):
    """Returns whether two NetworkX graphs (directed or undirected) are equal."""
    
    if (not G1.order() == G2.order()) or (not G1.size() == G2.size()):
        return False
    
    if set(G1.nodes()) != set(G2.nodes()):
        return False
    
    for x, y in G1.edges():
        if not G2.has_edge(x, y):
            return False
    
    return True


def is_subgraph(G1, G2):
    """Returns whether G1 is a subgraph of G2."""
    
    if ((isinstance(G1, nx.Graph) and isinstance(G2, nx.DiGraph)) or
        (isinstance(G1, nx.DiGraph) and isinstance(G2, nx.Graph))):
        return False
    
    if G1.order() > G2.order() or G1.size() > G2.size():
        return False
    
    # vertex set is not a subset
    if not set(G1.nodes()) <= set(G2.nodes()):
        return False
    
    for x, y in G1.edges():
        if not G2.has_edge(x, y):
            return False
    
    return True


def symmetric_diff(G1, G2):
    """Returns the number of edges in the symmetric difference."""
    
    set1 = {x for x in G1.nodes()}
    set2 = {x for x in G2.nodes()}
    
    if set1 != set2:
        raise RuntimeError('graphs do not have the same vertex set')
        return
    
    sym_diff_number = 0
    
    if isinstance(G1, nx.DiGraph):
        generator = itertools.permutations(set1, 2)
    else:
        generator = itertools.combinations(set1, 2)
    
    for x, y in generator:
        if G1.has_edge(x, y) and not G2.has_edge(x, y):
            sym_diff_number += 1
        elif not G1.has_edge(x, y) and G2.has_edge(x, y):
            sym_diff_number += 1
    
    return sym_diff_number


def contingency_table(true_graph, graph, as_dict=True):
    """Contingency table for the edge sets of two graphs.
    
    The two graphs must have the same vertex set.
    """
    
    if (true_graph.order() != graph.order() or
        set(true_graph.nodes()) != set(graph.nodes())):
        raise ValueError('graphs must have the same vertex sets')
            
    
    tp, fp, fn = 0, 0, 0
    
    for u, v in graph.edges():
        if true_graph.has_edge(u,v):
            tp += 1
        else:
            fp += 1
    
    for u, v in true_graph.edges():
        if not graph.has_edge(u, v):
            fn += 1
    
    if isinstance(graph, nx.DiGraph):
        tn = (graph.order() * (graph.order()-1)) - (tp + fp + fn)
    else:
        tn = (graph.order() * (graph.order()-1) // 2) - (tp + fp + fn)
    
    if as_dict:
        return {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}
    else:
        return tp, tn, fp, fn


def performance(true_graph, graph):
    """Returns order, size, tp, tn, fp, fn, accuracy, precision and recall
    of a (directed or undirected) graph w.r.t. true graph."""
    
    tp, tn, fp, fn = contingency_table(true_graph, graph, as_dict=False)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else float('nan')
    precision = tp / (tp + fp) if tp + fp > 0 else float('nan')
    recall = tp / (tp + fn) if tp + fn > 0 else float('nan')
    
    return (graph.order(), graph.size(),
            tp, tn, fp, fn,
            accuracy, precision, recall)


def false_edges(true_graph, graph):
    """Returns a graph containing fn and a graph containg fp edges."""
    
    if isinstance(true_graph, nx.DiGraph):
        fn_graph = nx.DiGraph()
        fp_graph = nx.DiGraph()
    else:
        fn_graph = nx.Graph()
        fp_graph = nx.Graph()
        
    fn_graph.add_nodes_from(true_graph.nodes(data=True))
    fp_graph.add_nodes_from(true_graph.nodes(data=True))
        
    for u, v in graph.edges():
        if not true_graph.has_edge(u,v):
            fp_graph.add_edge(u, v)
    
    for u, v in true_graph.edges():
        if not graph.has_edge(u, v):
            fn_graph.add_edge(u, v)
        
    return fn_graph, fp_graph


# ----------------------------------------------------------------------------
#                             Graph coloring
# ----------------------------------------------------------------------------

def is_properly_colored(G):
    """Returns whether a (di)graph is properly colored.
    
    Raises a KeyError if, in any edge, a vertex has no 'color' attribute."""
    
    for u, v in G.edges():
        if G.nodes[u]['color'] == G.nodes[v]['color']:
            return False
    
    return True


def sort_by_colors(graph):
    
    color_dict = {}
    
    for v in graph.nodes():
        color = graph.nodes[v]['color']
        
        if color not in color_dict:
            color_dict[color] = [v]
        else:
            color_dict[color].append(v)
    
    return color_dict


def copy_node_attributes(from_graph, to_graph):
    """Copy node label and color from one graph to another."""
    
    for x in from_graph.nodes():
        if to_graph.has_node(x):
            to_graph.nodes[x]['label'] = from_graph.nodes[x]['label']
            to_graph.nodes[x]['color'] = from_graph.nodes[x]['color']


# ----------------------------------------------------------------------------
#                      Graph generation/manipulation
# ----------------------------------------------------------------------------

def random_graph(N, p=0.5):
    """Construct a random graph on N nodes.
    
    Keyword arguments:
        p - probability that an edge xy is inserted
    """
        
    G = nx.Graph()
    G.add_nodes_from(range(1, N+1))
    
    for x, y in itertools.combinations(range(1, N+1), 2):
        if random.random() < p:
            G.add_edge(x, y)
    
    return G


def disturb_graph(graph, insertion_prob, deletion_prob, inplace=False,
                  preserve_properly_colored=True):
    """Randomly insert/delete edges in a graph."""
    
    if not inplace:
        graph = graph.copy()
    
    for x, y in itertools.combinations(graph.nodes, 2):
        
        if (preserve_properly_colored and
            'color' in graph.nodes[x] and 'color' in graph.nodes[y] and
            graph.nodes[x]['color'] == graph.nodes[y]['color']):
            continue
        
        if not graph.has_edge(x, y) and random.random() <= insertion_prob:
            graph.add_edge(x, y)
        elif graph.has_edge(x, y) and random.random() <= deletion_prob:
            graph.remove_edge(x, y)
        
        # done for undirected graphs
        if not isinstance(graph, nx.DiGraph):
            continue
        
        # other direction for digraphs
        if not graph.has_edge(y, x) and random.random() <= insertion_prob:
            graph.add_edge(y, x)
        elif graph.has_edge(y, x) and random.random() <= deletion_prob:
            graph.remove_edge(y, x)
    
    return graph


# ----------------------------------------------------------------------------
#                      Complete multipartite graphs
# ----------------------------------------------------------------------------

def independent_sets(G):
    """Independent sets of a complete multipartite (i.e. a Fitch graph).
    
    Returns a partition of the graph's vertex set that corresponds to its
    set of independent set if the graph is complete multipartite, and
    False otherwise.
    """
    
    # independent sets are equivalent to cliques in the complement graph
    G = nx.complement(G)
    
    ccs = [list(cc) for cc in nx.connected_components(G)]
    
    for cc in ccs:
        for x, y in itertools.combinations(cc, 2):
            if not G.has_edge(x, y):                # not a clique
                return False
    
    return ccs


def complete_multipartite_graph_from_sets(partition):
    """Construct the complete multipartite graphs from the independent sets."""
    
    G = nx.Graph()
    
    for i in range(len(partition)):
        G.add_nodes_from(partition[i])
        
        for j in range(i+1, len(partition)):
            for x, y in itertools.product(partition[i], partition[j]):
                G.add_edge(x, y)
    
    return G