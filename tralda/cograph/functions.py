"""Module for cograph-related functions.

Implementation of a greedy solution for the cluster deletion problem for cographs, and the complete
multipartite graph completion problem.
"""

import itertools
import random
from collections.abc import Iterator
from typing import Any

import networkx as nx

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode
from tralda.datastructures.last_common_ancestor import LCA
from tralda.utils.graph_tools import complete_multipartite_graph_from_sets
from tralda.cograph.detection import LinearCographDetector


def to_cograph(cotree: Tree) -> nx.Graph:
    """Returns the cograph corresponding to the cotree.

    Args:
        cotree: A cotree, i.e., a Tree instance with inner vertex labels 'series' and 'parallel'.

    Returns:
        The corresponding cograph with the leaf labels as vertices.
    """
    leaves = cotree.leaf_dict()
    graph = nx.Graph()

    for v in leaves[cotree.root]:
        graph.add_node(v.label)

    for u in cotree.preorder():
        if u.label != "series":
            continue

        for v1, v2 in itertools.combinations(u.children, 2):
            for l1, l2 in itertools.product(leaves[v1], leaves[v2]):
                graph.add_edge(l1.label, l2.label)

    return graph


def to_cotree(graph: nx.Graph) -> Tree | None:
    """Checks if a graph is a cograph and returns its cotree.

    Linear O(|V| + |E|) implementation.

    Args:
        graph: A graph.

    Returns:
            The cotree representation of the graph, or None if it is not a cograph.

    References:
        .. [1] D. G. Corneil, Y. Perl, and L. K. Stewart. A Linear Recognition Algorithm for
               Cographs. In: SIAM J. Comput., 14(4), 926-934 (1985). DOI: 10.1137/0214065
    """
    return LinearCographDetector(graph).recognition()


def complement_cograph(cotree: Tree, inplace: bool = False) -> Tree:
    """Returns the cotree of the complement cograph.

    Args:
        cotree: A cotree, i.e., a Tree instance with inner vertex labels 'series' and 'parallel'.

    Returns:
        The cotree of the complement cograph.

    Raises:
        ValueError: If the label is not 'series' and 'parallel' for any inner node.
    """
    tree = cotree if inplace else cotree.copy()

    for v in tree.inner_nodes():
        if v.label == "series":
            v.label = "parallel"
        elif v.label == "parallel":
            v.label = "series"
        else:
            raise ValueError(f"inner node has label '{v.label}', must be 'series' or 'parallel'")

    return tree


def paths_of_length_2(cotree: Tree) -> Iterator[tuple[TreeNode, TreeNode, TreeNode]]:
    """Generator for all paths of length 2 (edges) in the cograph.

    Args:
        cotree: A cotree, i.e., a Tree instance with inner vertex labels 'series' and 'parallel'.

    Yields:
        All paths of length 2 in the corresponding cograph.

    Raises:
        ValueError: If the label is not 'series' and 'parallel' for any inner node.
    """
    leaves = cotree.leaf_dict()
    lca = LCA(cotree)

    for u in cotree.inner_nodes():
        if u.label == "parallel":
            continue
        elif u.label != "series":
            raise ValueError(f"inner node has label '{u.label}', must be 'series' or 'parallel'")

        for v1, v2 in itertools.permutations(u.children, 2):
            for t1, t2 in itertools.combinations(leaves[v1], 2):
                if lca(t1, t2).label != "parallel":
                    continue

                for t3 in leaves[v2]:
                    yield t1, t3, t2


def random_cotree(number_of_leaves: int, force_series_root: bool = False) -> Tree:
    """Creates a random cotree.

    Args:
        number_of_leaves: The number of leaves in the resulting tree.
        force_series_root: If True, the cograph of the resulting cotree will be connected, otherwise
            it may be disconnected.

    Returns:
        A cotree, i.e., a Tree instance with inner vertex labels 'series' and 'parallel'.
    """
    cotree = Tree.random_tree(number_of_leaves)

    # assign labels ('series', 'parallel')
    for v in cotree.preorder():
        if v.is_leaf():
            continue
        elif v.parent is None:
            if force_series_root:
                v.label = "series"
            else:
                v.label = "series" if random.random() < 0.5 else "parallel"
        else:
            v.label = "series" if v.parent.label == "parallel" else "parallel"

    return cotree


def cluster_deletion(cograph: nx.Graph | Tree) -> list[list[Any]]:
    """Cluster deletion for cographs.

    Returns a partition of a cograph into disjoint cliques with a minimal number of edges between
    the cliques.

    Args:
        cograph: The cograph (as a graph or cotree) for which cluster deletion shall be performed.

    Returns:
        A partition where each sublist corresponds to a clique in a solution of the cluster deletion
        problem.

    Raises:
        ValueError: If the input is not a valid cograph or cotree.

    References:
        .. [1] Gao Y, Hare DR, Nastos J (2013) The cluster deletion problem for cographs. Discrete
               Math 313(23):2763-2771, DOI: 10.1016/j.disc.2013.08.017.
        .. [2] Schaller D, Lafond M, Stadler PF, Wieseke N, Hellmuth M (2020) Indirect
               Identification of Horizontal Gene Transfer. Journal of Mathematical Biology 83(10),
               DOI: 10.1007/s00285-021-01631-0
    """
    cotree = cograph if isinstance(cograph, Tree) else to_cotree(cograph)

    if not cotree:
        raise ValueError("not a valid cograph/cotree")

    partition = {}

    for u in cotree.postorder():
        partition[u] = []
        if not u.children:
            partition[u].append([u.label])

        elif u.label == "parallel":
            for v in u.children:
                partition[u].extend(partition[v])

            # naive sorting can be replaced by k-way merge-sort
            partition[u].sort(key=len, reverse=True)

        elif u.label == "series":
            for v in u.children:
                for i, Q_i in enumerate(partition[v]):
                    if i >= len(partition[u]):
                        partition[u].append([])
                    partition[u][i].extend(Q_i)

        else:
            raise ValueError("invalid cotree")

    return partition[cotree.root]


def complete_multipartite_completion(
    cograph: nx.Graph | Tree,
    supply_graph: bool = False,
) -> list[list[Any]] | tuple[list[list[Any]], nx.Graph]:
    """Complete multipartite graph completion for cographs.

    Returns a partition of the vertex set corresponding to the (maximal) independent sets in an
    optimal edge completion of the cograph to a complete multipartite graph.

    Args:
        cograph: The cograph (as a graph or cotree) for which complete multipartite graph completion
            shall be performed.
        supply_graph: If True, the solution is additionally returned as a NetworkX Graph.

    Returns:
        A partition where each sublist corresponds to a (maximal) independent set in a solution of
        the complete multipartite graph completion problem. And, optionally, the solution as a
        graph.

    Raises:
        ValueError: If the input is not a valid cograph or cotree.
    """
    cotree = cograph if isinstance(cograph, Tree) else to_cotree(cograph)

    if not cotree:
        raise ValueError("not a valid cograph/cotree")

    # complete multipartite graph completion is equivalent to cluster deletion in the complement
    # cograph
    compl_cotree = complement_cograph(cotree, inplace=False)

    # clusters are then equivalent to the maximal independent sets
    independent_sets = cluster_deletion(compl_cotree)

    if not supply_graph:
        return independent_sets
    else:
        return (independent_sets, complete_multipartite_graph_from_sets(independent_sets))
