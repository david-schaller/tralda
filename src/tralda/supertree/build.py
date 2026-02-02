"""Algorithms for building trees from rooted triple sets."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any
from typing import Collection
import itertools
import networkx as nx

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode
from tralda.datastructures.partition import Partition


Triple = tuple[Any, Any, Any]


def aho_graph(
    R: Iterable[Triple],
    L: Collection[Any],
    weighted: bool = False,
    triple_weights: dict[Triple, int | float] | None = None,
) -> nx.Graph:
    """Construct the auxiliary graph (Aho graph) for the BUILD algorithm.

    Edges {a,b} are optionally weighted by the number of occurrences, resp., sum of weights of
    triples of the form ab|x or ba|x.

    Args:
        R: A collection of triples.
        L: A collection of leaf labels.
        weighted: If True, weight the edges in the resulting graph accordingto the number of
            corresponding triples (and their weights).
        triple_weights: Dictionary with tuples as keys and float values. The weights of the triples.
            The default is None, in which case every triple has weight one.

    Returns:
        The undirected Aho graph.
    """
    graph = nx.Graph()
    graph.add_nodes_from(L)

    for a, b, c in R:
        if not graph.has_edge(a, b):
            graph.add_edge(a, b)

        if weighted:
            if triple_weights:
                graph[a][b]["weight"] = graph[a][b].get("weight", 0.0) + triple_weights[a, b, c]
            else:
                graph[a][b]["weight"] = graph[a][b].get("weight", 0.0) + 1.0

    return graph


def _forbidden_triple_connect(graph: nx.Graph, triple: Triple) -> None:
    """Connect items according to a forbidden triple in the MTT algorithm.

    Args:
        graph: The graph in which to set edges.
        triple: The triple.
    """
    graph.add_edge(triple[0], triple[1])
    graph.add_edge(triple[0], triple[2])


def mtt_partition(
    L: Collection[Any], R: Iterable[Triple], F: Iterable[Triple]
) -> tuple[Partition, nx.Graph]:
    """Construct the auxiliary partition for the MTT algorithm.

    Args:
        L: A collection of leaf labels.
        R: An iterable of required triples.
        F: An iterable of forbidden triples.

    Returns:
        The auxiliary partition for the MTT algorithm and a graph representation of this auxiliary
        partition.
    """
    # auxiliary graph initialized as Aho graph
    graph = aho_graph(R, L, weighted=False)

    # auxiliary partition
    partition = Partition(nx.connected_components(graph))

    if len(partition) == 1:
        return partition, graph

    # set of remaining forbidden triples that need to be considered
    remaining_forbidden_triples = {t for t in F if partition.separated_xy_z(*t)}

    # lookup of forbidden triples to which u belongs
    leaf2forb_triples = {u: [] for u in L}
    for t in F:
        for u in t:
            leaf2forb_triples[u].append(t)

    while remaining_forbidden_triples:
        t = remaining_forbidden_triples.pop()
        _forbidden_triple_connect(graph, t)

        # merge returns the smaller of the two merged sets
        smaller_set = partition.merge(t[0], t[2])

        # update remaining forbidden triples by traversing the L(u)
        for u in smaller_set:
            for t in leaf2forb_triples[u]:
                if t in remaining_forbidden_triples and not partition.separated_xy_z(*t):
                    remaining_forbidden_triples.remove(t)
                    _forbidden_triple_connect(graph, t)
                elif t not in remaining_forbidden_triples and partition.separated_xy_z(*t):
                    remaining_forbidden_triples.add(t)

    return partition, graph


class Build:
    """BUILD algorithm.

    References:
        .. [1] A. V. Aho, Y. Sagiv, T. G. Szymanski, and J. D. Ullman. Inferring a tree from lowest
           common ancestors with an application to the optimization of relational expressions.
           SIAM Journal on Computing, 10:405-421, 1981. DOI: 10.1137/0210030.
    """

    def __init__(
        self,
        R: Iterable[Triple],
        L: Collection[Any],
        mincut: bool = False,
        weighted_mincut: bool = False,
        triple_weights: dict[Triple, int | float] | None = None,
    ):
        """Constructor for class implementing the BUILD algorithm.

        Args:
            R: An iterable of triples.
            L: A collection of leaf labels.
            mincut: If True, use MinCut to resolve triple inconsistencies. Otherwise, no tree is
                returned in case of such inconsistencies.
            weighted_mincut: If True, weight the edges in the resulting graph accordingto the number
                of corresponding triples (and their weights).
            triple_weights: Dictionary with tuples as keys and float values. The weights of the
                triples. The default is None, in which case every triple has weight one.
        """
        self.R = R
        self.L = L
        self.mincut = mincut
        self.weighted_mincut = weighted_mincut
        self.triple_weights = triple_weights

    def build_tree(
        self,
        return_root: bool = False,
        print_info: bool = False,
    ) -> Tree | TreeNode | None:
        """Build a tree displaying all triples in R if possible.

        Args:
            return_root: If True, return 'TreeNode' instead of 'Tree' instance.
            print_info: If True, print information about inconsistencies.

        Returns
            The BUILD tree on leaf set L displaying all triples in R if the triple set R is
            consistent.
        """
        self.cut_value = 0
        self.cut_list = []
        self.print_info = print_info

        root = self._aho(self.L, self.R)

        # case 1: building a tree was successful
        if root:
            return root if return_root else Tree(root)

        # case 2: inconsistent triple set --> return None
        if self.print_info:
            print("no such tree exists")

    def _aho(self, L: Collection[Any], R: Iterable[Triple]) -> TreeNode | None:
        """Recursive Aho-algorithm.

        Args:
            L: A collection of leaf labels.
            R: An iterable of triples.

        Returns:
            The root of the recursively built tree (or None if no tree exists).
        """
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

        help_graph = aho_graph(
            R, L, weighted=self.weighted_mincut, triple_weights=self.triple_weights
        )
        conn_comps = self._connected_components(help_graph)

        # return if less than 2 connected components
        if len(conn_comps) <= 1:
            if self.print_info:
                print("Connected component:\n", conn_comps)
            return

        # otherwise proceed recursively
        node = TreeNode()  # place new inner node
        for cc in conn_comps:
            Li = set(cc)  # list/dictionary --> set
            Ri = []
            for t in R:  # construct triple subset
                if Li.issuperset(t):
                    Ri.append(t)
            Ti = self._aho(Li, Ri)  # recursive call
            if not Ti:
                return  # return None to previous call
            else:
                node.add_child(Ti)  # add root of the subtree

        return node

    def _connected_components(self, aho_graph: nx.Graph) -> list[set[Any]] | Partition:
        """Determines the connected components of the graph.

        And optionally executes a min cut if there is only one component.

        Args:
            aho_graph: The Aho graph.

        Returns:
            The connected components of the Aho graph or the result of applying the MinCut algorithm
            for resolving triple inconsistencies.
        """
        conn_comps = list(nx.connected_components(aho_graph))
        if (not self.mincut) or len(conn_comps) > 1:
            return conn_comps

        # execute MinCut using the Stoer–Wagner algorithm
        cut_value, partition = nx.stoer_wagner(aho_graph)
        self.cut_value += cut_value

        if len(partition[0]) < len(partition[1]):
            smaller_comp = partition[0]
        else:
            smaller_comp = partition[1]

        for edge in aho_graph.edges():
            if (edge[0] in smaller_comp and edge[1] not in smaller_comp) or (
                edge[1] in smaller_comp and edge[0] not in smaller_comp
            ):
                self.cut_list.append(edge)

        return partition


class MTT:
    """MTT algorithm.

    References:
        .. [1] Y.-J. He, T. N. D. Huynh, J. Jansson, and W.-K. Sung.Inferring phylogenetic
           relationships avoiding forbidden rooted triplets. Journal of Bioinformatics and
           Computational Biology, 4: 59-74, 2006. DOI: 10.1142/S0219720006001709.
    """

    def __init__(
        self,
        R: Iterable[Triple],
        L: Collection[Any],
        F: Iterable[Triple] | None = None,
    ) -> None:
        """Constructor for class implementing the MTT algorithm.

        Args:
            L: A collection of leaf labels.
            R: An iterable of required triples.
            F: An iterable of forbidden triples.
        """
        self.R = R
        self.L = L

        # forbidden triples --> activates MTT if non-empty
        self.F = F

    def build_tree(self, return_root: bool = False) -> Tree | TreeNode | None:
        """Build a tree displaying all triples in R if possible.

        Args:
            return_root: If True, return 'TreeNode' instead of 'Tree' instance.

        Returns:
            A tree on leaf set L displaying all triples in R and none in F, if such a tree exists,
            and None otherwise.
        """
        self.total_cost = 0

        if self.F:
            root = self._mtt(self.L, self.R, self.F)
        else:
            root = self._aho(self.L, self.R)

        return root if return_root else Tree(root)

    def _trivial_case(self, L: Collection[Any]) -> TreeNode:
        """Base case of the recursion (1 or 2 leaves).

        Args:
            L: Collection of 1 or 2 leaves.

        Raises:
            RuntimeError: If L does not have 1 or 2 leaves.

        Returns:
            The created leaf node or inner node with two leaves.
        """
        if len(L) == 1:
            leaf = L.pop()
            return TreeNode(label=leaf)
        elif len(L) == 2:
            node = TreeNode()
            for _ in range(2):
                leaf = L.pop()
                node.add_child(TreeNode(label=leaf))
            return node
        else:
            raise RuntimeError(f"provided collection of leaves has size {len(L)}")

    def _aho(self, L: Collection[Any], R: Iterable[Triple]) -> TreeNode | None:
        """Recursive Aho algorithm.

        Args:
            L: A collection of leaf labels.
            R: An iterable of triples.

        Returns:
            The root of the recursively built tree (or None if no tree exists).
        """
        # trivial case: one or two leaves left in L
        if len(L) <= 2:
            return self._trivial_case(L)

        aux_graph = aho_graph(R, L)
        partition = list(nx.connected_components(aux_graph))

        if len(partition) < 2:
            return

        node = TreeNode()  # place new inner node
        for s in partition:
            Li, Ri = set(s), []
            for t in R:  # construct triple subset
                if Li.issuperset(t):
                    Ri.append(t)
            Ti = self._aho(Li, Ri)  # recursive call
            if not Ti:
                return  # return None to previous call
            else:
                node.add_child(Ti)  # add roots of the subtrees

        return node

    def _mtt(self, L: Collection[Any], R: Iterable[Triple], F: Iterable[Triple]) -> TreeNode | None:
        """Recursive MTT algorithm.

        Args:
            L: A collection of leaf labels.
            R: An iterable of required triples.
            F: An iterable of forbidden triples.

        Returns:
            The root of the recursively built tree (or None if no tree exists).
        """
        # trivial case: one or two leaves left in L
        if len(L) <= 2:
            return self._trivial_case(L)

        partition, _ = mtt_partition(L, R, F)

        if len(partition) < 2:
            return

        node = TreeNode()  # place new inner node
        for s in partition:
            Li, Ri, Fi = set(s), [], []
            for Xi, X in ((Ri, R), (Fi, F)):
                for t in X:
                    if Li.issuperset(t):
                        Xi.append(t)
            Ti = self._mtt(Li, Ri, Fi)  # recursive call
            if not Ti:
                return  # return None to previous call
            else:
                node.add_child(Ti)  # add roots of the subtrees

        return node


def greedy_build(
    R: Iterable[Triple],
    L: Collection[Any],
    triple_weights: dict[Triple, int | float] | None = None,
    return_root: bool = False,
) -> Tree | TreeNode:
    """Greedy heuristic for triple consistency.

    Add triples one by one and checks consistency via BUILD.

    Args:
        R: An iterable of triples.
        L: A collection of leaf labels.
        triple_weights: Weights for the triples; default is None in which case all triples are
            uniformly weighted.
        return_root: If True, return 'TreeNode' instead of 'Tree' instance.

    Returns:
        The constructed tree.
    """
    if triple_weights:
        triples = sorted(R, key=lambda triple: triple_weights[triple], reverse=True)
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


def best_pair_merge_first(
    R: Iterable[Triple],
    L: Collection[Any],
    triple_weights: dict[Triple, int | float] | None = None,
    return_root: bool = False,
) -> Tree | TreeNode:
    """Wu's (2004) Best-Pair-Merge-First (BPMF) heuristic.

    Modified version by Byrka et al. (2010) and added weights.

    Args:
        R: An iterable of triples.
        L: A collection of leaf labels.
        triple_weights: Weights for the triples; default is None in which case all triples are
            uniformly weighted.
        return_root: If True, return 'TreeNode' instead of 'Tree' instance.

    Returns:
        The constructed tree.
    """
    # initialization
    nodes = {TreeNode(label=leaf): {leaf} for leaf in L}
    leaf_to_node = {}

    for node in nodes:
        leaf_to_node[node.label] = node

    # merging
    for i in range(len(L) - 1):
        score = {(S_i, S_j): 0 for S_i, S_j in itertools.combinations(nodes.keys(), 2)}

        for x, y, z in R:
            w = triple_weights[(x, y, z)] if triple_weights else 1

            S_i, S_j, S_k = (leaf_to_node[x], leaf_to_node[y], leaf_to_node[z])

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

        current_max = float("-inf")
        S_i, S_j = None, None

        for pair, pair_score in score.items():
            if pair_score > current_max:
                current_max = pair_score
                S_i, S_j = pair

        # create new node S_k connecting S_i and S_j
        S_k = TreeNode()
        S_k.add_child(S_i)
        S_k.add_child(S_j)

        nodes[S_k] = nodes[S_i] | nodes[S_j]  # set union
        for leaf in nodes[S_k]:
            leaf_to_node[leaf] = S_k

        del nodes[S_i]
        del nodes[S_j]

    if len(nodes) != 1:
        raise RuntimeError("more than 1 node left")

    root = next(iter(nodes))

    return root if return_root else Tree(root)


def minimal_identifying_triple_set(tree: Tree) -> Iterator[Triple]:
    """Construct a minimal set of triples that identifies the tree.

    Args:
        tree: A Tree instance.

    Yields:
        Triples in minimal set of triples that identifies the tree.

    References:
        .. [1] Stefan Grünewald, Mike Steel and M. Shel Swenson. Closure operations in
               phylogenetics. Mathematical Biosciences 208 (2007) 521-537.
               DOI: 10.1016/j.mbs.2006.11.005
    """
    # representative leaf for each vertex
    repres = {}

    for v in tree.postorder():
        repres[v] = repres[v.children[0]] if v.children else v.label

    for u, v in tree.inner_edges():
        # convert to list for faster access
        v_children = [c for c in v.children]

        for v2 in u.children:
            if v is v2:
                continue

            for i in range(len(v_children) - 1):
                yield (repres[v_children[i]], repres[v_children[i + 1]], repres[v2])


def tree_profile_to_triples(trees: Iterable[Tree]) -> tuple[set[Any], set[Triple]]:
    """Construct leaf set and representative triples from a profile of trees.

    Args:
        trees: An iterable of Tree instances.

    Returns
        The first set contains all leaf labels that appear in the tree profilea, and the second set
        contains a representative set of triples.
    """
    leaves = set()
    triples = set()

    for tree in trees:
        leaves.update(leaf.label for leaf in tree.leaves())
        triples.update((*sorted(t[:2]), t[2]) for t in minimal_identifying_triple_set(tree))

    return leaves, triples


def build_supertree(trees: Iterable[Tree]) -> Tree | None:
    """Supertree construction based on the BUILD algorithm.

    Args:
        trees: An iterable of Tree instances.

    Returns:
        A supertree for the input trees if existent, None otherwise.
    """
    leaves, triples = tree_profile_to_triples(trees)

    build = Build(triples, leaves, mincut=False)

    return build.build_tree()
