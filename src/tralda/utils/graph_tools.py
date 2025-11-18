"""Collection of functions for graph analysis, comparison, or manipulation."""

from __future__ import annotations

import itertools
import random
from collections.abc import Collection
from collections.abc import Sequence
from typing import Any

import numpy as np
import networkx as nx


# --------------------------------------------------------------------------------------------------
#                                      Adjacency matrix
# --------------------------------------------------------------------------------------------------


def build_adjacency_matrix(graph: nx.Graph | nx.DiGraph) -> np.ndarray:
    """Return an adjacency matrix.

    Args:
        graph: A graph.

    Returns:
        An adjacency matrix representing the provided graph.
    """
    # maps node --> row/column index
    index = {node: i for i, node in enumerate(graph.nodes())}
    matrix = np.zeros((len(index), len(index)), dtype=np.int8)

    for x, neighbors in graph.adjacency():
        for y in neighbors:
            matrix[index[x], index[y]] = 1

    return matrix, index


# --------------------------------------------------------------------------------------------------
#                                      Graph comparison
# --------------------------------------------------------------------------------------------------


def graphs_equal(graph1: nx.Graph | nx.DiGraph, graph2: nx.Graph | nx.DiGraph) -> bool:
    """Returns whether two NetworkX graphs (directed or undirected) are equal.

    Args:
        graph1: The first graph.
        graph2: The second graph.

    Returns:
        True if the two graphs have the same nodes and edges, False otherwise.
    """
    if (not graph1.order() == graph2.order()) or (not graph1.size() == graph2.size()):
        return False

    if set(graph1.nodes()) != set(graph2.nodes()):
        return False

    for x, y in graph1.edges():
        if not graph2.has_edge(x, y):
            return False

    return True


def is_subgraph(graph1: nx.Graph | nx.DiGraph, graph2: nx.Graph | nx.DiGraph) -> bool:
    """Returns whether graph1 is a subgraph of graph2.

    Args:
        graph1: The first graph.
        graph2: The second graph.

    Returns:
        True if graph1 is a subgraph of graph2, False otherwise.
    """
    if (isinstance(graph1, nx.Graph) and isinstance(graph2, nx.DiGraph)) or (
        isinstance(graph1, nx.DiGraph) and isinstance(graph2, nx.Graph)
    ):
        return False

    if graph1.order() > graph2.order() or graph1.size() > graph2.size():
        return False

    # vertex set is not a subset
    if not set(graph1.nodes()) <= set(graph2.nodes()):
        return False

    for x, y in graph1.edges():
        if not graph2.has_edge(x, y):
            return False

    return True


def symmetric_diff(graph1: nx.Graph | nx.DiGraph, graph2: nx.Graph | nx.DiGraph) -> int:
    """Returns the number of edges in the symmetric difference.

    Args:
        graph1: The first graph.
        graph2: The second graph.

    Returns:
        The number of edges in the symmetric difference.
    """
    set1 = {x for x in graph1.nodes()}
    set2 = {x for x in graph2.nodes()}

    if set1 != set2:
        raise RuntimeError("graphs do not have the same vertex set")
        return

    sym_diff_number = 0

    if isinstance(graph1, nx.DiGraph):
        generator = itertools.permutations(set1, 2)
    else:
        generator = itertools.combinations(set1, 2)

    for x, y in generator:
        if graph1.has_edge(x, y) and not graph2.has_edge(x, y):
            sym_diff_number += 1
        elif not graph1.has_edge(x, y) and graph2.has_edge(x, y):
            sym_diff_number += 1

    return sym_diff_number


def contingency_table(
    true_graph: nx.Graph | nx.DiGraph,
    graph: nx.Graph | nx.DiGraph,
    as_dict: bool = True,
) -> tuple[int, int, int, int] | dict[str, int]:
    """Contingency table for the edge sets of two graphs.

    The two graphs must have the same vertex set.

    Args:
        true_graph: The 'true' graph.
        graph: The graph whose edges are compared against the 'true' graph.
        as_dict: If True, the numbers of true positives, true negatives, false positives, and false
            negatives are returned as a dictionary.

    Returns:
        The true positives, true negatives, false positives, and false negatives. Optionally, as a
        dictionary with keys 'tp', 'tn', 'fp', and 'fn'.

    Raises:
        ValueError: If the graph do not have the same set of vertices.
    """
    if true_graph.order() != graph.order() or set(true_graph.nodes()) != set(graph.nodes()):
        raise ValueError("graphs must have the same vertex sets")

    tp, fp, fn = 0, 0, 0

    for u, v in graph.edges():
        if true_graph.has_edge(u, v):
            tp += 1
        else:
            fp += 1

    for u, v in true_graph.edges():
        if not graph.has_edge(u, v):
            fn += 1

    if isinstance(graph, nx.DiGraph):
        tn = (graph.order() * (graph.order() - 1)) - (tp + fp + fn)
    else:
        tn = (graph.order() * (graph.order() - 1) // 2) - (tp + fp + fn)

    if as_dict:
        return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}
    else:
        return tp, tn, fp, fn


def performance(
    true_graph: nx.Graph | nx.DiGraph,
    graph: nx.Graph | nx.DiGraph,
) -> tuple[int, int, int, int, int, int, float, float, float]:
    """Returns various metrics for the comparison of a graph against a reference graph.

    Args:
        true_graph: The 'true' graph.
        graph: The graph whose edges are compared against the 'true' graph.

    Returns:
        Order, size, tp, tn, fp, fn, accuracy, precision and recall of a (directed or undirected)
        graph w.r.t. 'true' graph.
    """
    tp, tn, fp, fn = contingency_table(true_graph, graph, as_dict=False)

    accuracy = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0 else float("nan")
    precision = tp / (tp + fp) if tp + fp > 0 else float("nan")
    recall = tp / (tp + fn) if tp + fn > 0 else float("nan")

    return (graph.order(), graph.size(), tp, tn, fp, fn, accuracy, precision, recall)


def false_edges(
    true_graph: nx.Graph | nx.DiGraph,
    graph: nx.Graph | nx.DiGraph,
) -> tuple[nx.Graph, nx.Graph] | tuple[nx.DiGraph, nx.DiGraph]:
    """Returns a graph containing false-negative and a graph containg false-positive edges.

    Args:
        true_graph: The 'true' graph.
        graph: The graph whose edges are compared against the 'true' graph.

    Returns:
        A graph containing false-negative and a graph containg false-positive edges.
    """
    if isinstance(true_graph, nx.DiGraph):
        fn_graph = nx.DiGraph()
        fp_graph = nx.DiGraph()
    else:
        fn_graph = nx.Graph()
        fp_graph = nx.Graph()

    fn_graph.add_nodes_from(true_graph.nodes(data=True))
    fp_graph.add_nodes_from(true_graph.nodes(data=True))

    for u, v in graph.edges():
        if not true_graph.has_edge(u, v):
            fp_graph.add_edge(u, v)

    for u, v in true_graph.edges():
        if not graph.has_edge(u, v):
            fn_graph.add_edge(u, v)

    return fn_graph, fp_graph


# --------------------------------------------------------------------------------------------------
#                                        Graph coloring
# --------------------------------------------------------------------------------------------------


def is_properly_colored(graph: nx.Graph | nx.DiGraph, color_attribute: str = "color") -> bool:
    """Returns whether a (di)graph is properly colored.

    A graph is properly color if, for any edge uv, the vertices u and v have different colors.

    Args:
        graph: The input graph whose vertices should have some color attribute.
        color_attribute: The vertex attribute that shall be used as color.

    Returns:
        Whether or not the graph is properly colored.

    Raises:
        KeyError: If, in any edge, a vertex does not have the color attribute.
    """
    for u, v in graph.edges():
        if graph.nodes[u][color_attribute] == graph.nodes[v][color_attribute]:
            return False

    return True


def sort_by_colors(
    graph: nx.Graph | nx.DiGraph,
    color_attribute: str = "color",
) -> dict[Any, list[Any]]:
    """Sort the vertices of a graph by color.

    Args:
        graph: The input graph whose vertices should have some color attribute.
        color_attribute: The vertex attribute that shall be used as color.

    Returns:
        A dictionary with the colors as keys and lists of corresponding vertices as values.

    Raises:
        KeyError: If any vertex does not have the color attribute.
    """
    color_dict = {}

    for v in graph.nodes():
        color = graph.nodes[v][color_attribute]

        if color not in color_dict:
            color_dict[color] = [v]
        else:
            color_dict[color].append(v)

    return color_dict


def copy_node_attributes(
    from_graph: nx.Graph | nx.DiGraph,
    to_graph: nx.Graph | nx.DiGraph,
    attributes: str | tuple[str, ...] | list[str] = ("label", "color"),
) -> None:
    """Copy node attributes from one graph to another.

    By default, the 'label' and 'color' attributes are copied.

    Args:
        from_graph: The source graph from which to copy node attributes.
        to_graph: The target graph to which to copy node attributes.
        attributes: The attributes to copy.
    """
    if isinstance(attributes, str):
        attributes = [attributes]

    for x in from_graph.nodes():
        if not to_graph.has_node(x):
            continue

        for attribute in attributes:
            to_graph.nodes[x][attribute] = from_graph.nodes[x][attribute]


# --------------------------------------------------------------------------------------------------
#                                Graph generation/manipulation
# --------------------------------------------------------------------------------------------------


def random_graph(number_of_nodes, p: float = 0.5) -> nx.Graph:
    """Construct a random graph with the specified number of nodes.

    The nodes are numbered continuously from 1 to number_of_nodes.

    Args:
        number_of_nodes: The number of nodes that the graph shall have.
        p: Probability that an edge xy is inserted.

    Returns:
        The generated random graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(1, number_of_nodes + 1))

    for x, y in itertools.combinations(range(1, number_of_nodes + 1), 2):
        if random.random() < p:
            graph.add_edge(x, y)

    return graph


def disturb_graph(
    graph: nx.Graph | nx.DiGraph,
    insertion_prob: float,
    deletion_prob: float,
    inplace: bool = False,
    preserve_properly_colored: bool = True,
    color_attribute: str = "color",
) -> nx.Graph | nx.DiGraph:
    """Randomly insert and/or delete edges in a graph.

    Args:
        graph: The input graph.
        insertion_prob: The probability with which a missing edge gets inserted.
        deletion_prob: The probability with which a present edge gets deleted.
        inplace: If True, manipulate the graph inplace. Otherwise, a copy of the graph gets
            manipulated and returned.
        preserve_properly_colored: Whether to preserve the property that the graph is properly
            colored.
        color_attribute: The vertex attribute that shall be used as color.

    Returns:
        The disturbed graph.
    """
    if not inplace:
        graph = graph.copy()

    for x, y in itertools.combinations(graph.nodes, 2):
        if (
            preserve_properly_colored
            and color_attribute in graph.nodes[x]
            and color_attribute in graph.nodes[y]
            and graph.nodes[x][color_attribute] == graph.nodes[y][color_attribute]
        ):
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


# --------------------------------------------------------------------------------------------------
#                                  Complete multipartite graphs
# --------------------------------------------------------------------------------------------------


def independent_sets(graph: nx.graph) -> list[list[Any]] | None:
    """Independent sets of a complete multipartite (i.e., a Fitch graph).

    Returns a partition of the graph's vertex set that corresponds to its set of independent set if
    the graph is complete multipartite, and None otherwise.

    Args:
        graph: The input graph.

    Returns:
        A list of lists that represent the independent set.
    """
    # independent sets are equivalent to cliques in the complement graph
    graph = nx.complement(graph)

    ccs = [list(cc) for cc in nx.connected_components(graph)]

    for cc in ccs:
        for x, y in itertools.combinations(cc, 2):
            if not graph.has_edge(x, y):  # not a clique
                return

    return ccs


def complete_multipartite_graph_from_sets(partition: Sequence[Collection[Any]]) -> nx.Graph:
    """Construct the complete multipartite graphs from the independent sets.

    Args:
        partition: A partition of the vertices for the graph to construct. The partition represents
            the independent sets of the graph.

    Returns:
        A complete multipartite graph whose independent sets are the sets in the input partition.
    """
    graph = nx.Graph()

    for i in range(len(partition)):
        graph.add_nodes_from(partition[i])

        for j in range(i + 1, len(partition)):
            for x, y in itertools.product(partition[i], partition[j]):
                graph.add_edge(x, y)

    return graph
