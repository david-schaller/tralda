"""Implementation of the BuildST algorithm described by Deng & Fernández-Baca.

Builds a supertree from a profile of phylogenetic trees. Uses the algorithm described by Deng and
Fernández-Baca in 2016. The class 'BuildST' accepts a list of trees and computes the supertree if
it exists.

References:
    .. [1] Yun Deng and David Fernández-Baca. Fast Compatibility Testing for Rooted Phylogenetic
           Trees. 27th Annual Symposium on Combinatorial Pattern Matching (CPM 2016).
           DOI: 10.4230/LIPIcs.CPM.2016.12
"""

from __future__ import annotations

from collections.abc import Collection
from collections.abc import Iterator
from typing import Any
from typing import Union

from tralda.datastructures.tree import Tree, TreeNode
from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.doubly_linked import DLListNode
from tralda.datastructures.hdtgraph.dynamic_graph import HDTGraph


class _XpNode:
    """Special type of node for the set Xp (one for each leaf label)."""

    def __init__(self, label: Any) -> None:
        """Constructor of the _XpNode class

        Args:
            label: The label that the node shall hold.
        """
        self.label = label
        self.leafnodes = []  # corresponding treenodes of the profile

    def __repr__(self) -> str:
        """Return a string representation of the _XpNode instance.

        Returns:
            A string representation of the _XpNode instance.
        """
        return f"<_XpNodeID:{id(self)}, {self.label}>"

    def __str__(self) -> str:
        """Return a string representation of the label.

        Returns:
            A string representation of the label.
        """
        return str(self.label)


NodeType = Union[TreeNode, _XpNode]


def build_st(trees: Collection[Tree]) -> Tree | None:
    """Supertree construction based on BuildST algorithm.

    NOTE: The graph connecting to trees in the input set whenever they share at least one leaf
    label must be connected. Otherwise, the implementation returns None, even though a supertree may
    exist.

    Args:
        trees: A collection of Tree instances.

    Return:
        The constructed supertree if it exists; None otherwise.

    References:
    .. [1] Yun Deng and David Fernández-Baca. Fast Compatibility Testing for Rooted Phylogenetic
           Trees. 27th Annual Symposium on Combinatorial Pattern Matching (CPM 2016).
           DOI: 10.4230/LIPIcs.CPM.2016.12
    """
    st_builder = BuildST(trees)

    return st_builder.run()


class BuildST:
    """BuildST algorithm.

    NOTE: The graph connecting to trees in the input set whenever they share at least one leaf
    label must be connected. Otherwise, the implementation returns None, even though a supertree may
    exist.

    References:
    .. [1] Yun Deng and David Fernández-Baca. Fast Compatibility Testing for Rooted Phylogenetic
           Trees. 27th Annual Symposium on Combinatorial Pattern Matching (CPM 2016).
           DOI: 10.4230/LIPIcs.CPM.2016.12
    """

    def __init__(self, trees: Collection[Tree]) -> None:
        """Constructor for BuildST algorithm.

        Args:
            trees: A collection of Tree instances.
        """
        # collection of k trees
        self.trees = trees

        # leaf label --> _XpNode
        self.label2xp_node: dict[Any, _XpNode] = {}

        # set U (marked nodes)
        self.marked: set[TreeNode] = set()

        # node --> index of the tree
        self.node2tree_index: dict[TreeNode, int] = {}

        # node --> doubly-linked list element
        self.list_pointer: dict[TreeNode, DLListNode] = {}

        # node --> doubly-linked list element
        self.singleton_pointer: dict[TreeNode, DLListNode] = {}

        # the graph H_p using HDT datastructure for dynamic graph connectivity
        self.hdt_graph = HDTGraph()

        # save why tree construction failed
        self.fail_message = ""

    def run(self) -> Tree | None:
        """Build the supertree from the given tree list if existent.

        Returns:
            The constructed supertree if it exits; None otherwise.
        """
        self._prepare_trees()  # tree indices, Xp nodes

        U_init = self._initialize()
        if not U_init:
            return

        root = self._buildst(U_init)

        if not root:
            return
        else:
            return Tree(root)

    def _prepare_trees(self) -> None:
        """Initialize lookups node2tree_index and label2xp_node."""
        for i in range(len(self.trees)):
            tree = self.trees[i]
            for node in tree.preorder():
                self.node2tree_index[node] = i  # enable access to tree index i in [k]
                if node.children:
                    continue

                if node.label not in self.label2xp_node:
                    self.label2xp_node[node.label] = _XpNode(node.label)
                self.label2xp_node[node.label].leafnodes.append(node)

    def _initialize(self) -> _ConnectedComp:
        """Construct the initial auxiliary graph H_p(U_init) as described in the paper.

        Returns:
            The initial connected component of the auxiliary graph H_p.
        """
        conn_comp_Y = _ConnectedComp(
            len(self.trees),
            self.hdt_graph,
            self.trees[0].root,
            count=len(self.label2xp_node),
        )

        for i, tree in enumerate(self.trees):
            # Y_init.singleton is the set i in [k] such that |U(i)|= 1
            self.singleton_pointer[tree.root] = conn_comp_Y.singleton.append(i)

            # append the root to DLList at Y.List[i]
            self.list_pointer[tree.root] = conn_comp_Y.List[i].append(tree.root)

            # add all the edges in the trees
            conn_comp_Y.initialize_tree_edges(tree)

            # mark root as part of set U_init
            self.marked.add(tree.root)

        # glue together the leafnodes and the label nodes in Xp
        for label_node in self.label2xp_node.values():
            for leafnode in label_node.leafnodes:
                conn_comp_Y.add_edge(label_node, leafnode)

        if self.hdt_graph and not self.hdt_graph.is_connected():
            self.failmessage = "initialization failed because graph is not fully connected"
            return

        return conn_comp_Y

    def _buildst(self, U: _ConnectedComp) -> TreeNode | None:
        """Recursive function for constructing the supertree.

        Args:
            U: A connected component of the auxiliary graph H_p.

        Returns:
            The root of the constructed supertree for the input trees if existent, None otherwise.
        """
        # --------------------------------------------------
        # if |L(U)| = 1 then
        #    return the tree consisting of node r_U, labeled by the single species in L(U)
        # --------------------------------------------------
        if U.count == 1:
            label_node = U.get_label_nodes()[0]
            return TreeNode(label=label_node.label)

        r_U = TreeNode()  # create a node r_U

        # --------------------------------------------------
        # if |L(U)| = 2 then
        #    return the tree consisting of node r_U and two children each labeled by a different
        #    species in L(U)
        # --------------------------------------------------
        if U.count == 2:
            labels = U.get_label_nodes()
            if len(labels) != 2:
                self.failmessage = "could not find 2 labeled nodes"
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
        W = [U]  # list of connected components
        J = [j for j in U.singleton]  # elements in singleton
        V = [U.List[i][0] for i in J]  # singleton nodes to be deleted

        for i in J:
            v_i = U.List[i][0]
            U.List[i].remove_node(self.list_pointer[v_i])
            U.singleton.remove_node(self.singleton_pointer[v_i])

            # unmark v_i and mark its children
            self.marked.discard(v_i)
            for child in v_i.children:
                self.marked.add(child)
                self.list_pointer[child] = U.List[i].append(child)

        for v_i in V:
            Y_current = None

            # find the component in W that contains v_i
            for Y in W:
                if v_i in Y:
                    Y_current = Y
                    break

            if not Y_current:
                return

            for u in v_i.children:
                # the edge deletion method guarantees that v_i is in conn_comp[0]
                conn_comp = Y_current.delete_edge(v_i, u)

                if len(conn_comp) == 2:
                    # case 1: u is the last child of v_i, then conn_comp[0] is actually empty
                    if len(conn_comp[0]) == 1:
                        Y_current.representative = u
                        break

                    # case 2: two actual components emerge
                    Y_new = conn_comp[1]

                    if len(Y_current) < len(Y_new):
                        self._update_conn_comps(Y_current, Y_new, first_swap=True)
                    else:
                        self._update_conn_comps(Y_new, Y_current)

                    W.append(Y_new)

        # --------------------------------------------------
        # Let W_1, W_2, ..., W_p be the connected components
        # if p = 1 then
        #    return incompatible
        # --------------------------------------------------
        if len(W) == 1:
            self.failmessage = "the trees are incompatible"
            return

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
                return

        # return the tree with root r_U
        return r_U

    def _update_conn_comps(
        self,
        Y1: _ConnectedComp,
        Y2: _ConnectedComp,
        first_swap: bool = False,
    ) -> None:
        """Updates Y.count, Y.singleton and Y.List of two newly split components.

        Component Y1 is supposed to be the smaller one and is scanned for label nodes (Xp) and for
        marked nodes.

        Args:
            Y1: The first connected component (should be the smaller one).
            Y2: The second connected compnent.
            first_swap: Should be set to True if Y1 (i.e., the smaller component) contains the
                information about count, List and singleton instead of Y2. In this case, these
                attributes need to be swapped first.
        """
        if first_swap:
            Y1.count, Y2.count = Y2.count, Y1.count
            Y1.List, Y2.List = Y2.List, Y1.List
            Y1.singleton, Y2.singleton = Y2.singleton, Y1.singleton

        for v in Y1:
            if isinstance(v, _XpNode):
                Y1.count += 1
                Y2.count -= 1
                continue

            if v not in self.marked:
                continue

            i = self.node2tree_index[v]
            Y2.List[i].remove_node(self.list_pointer[v])
            self.list_pointer[v] = Y1.List[i].append(v)

            # v becomes singleton
            if len(Y2.List[i]) == 1:
                singleton_node = Y2.List[i][0]
                self.singleton_pointer[singleton_node] = Y2.singleton.append(i)
            # index i is no longer singleton, i.e., it must have been a singleton
            elif len(Y2.List[i]) == 0:
                Y2.singleton.remove_node(self.singleton_pointer[v])

            # v becomes singleton
            if len(Y1.List[i]) == 1:
                self.singleton_pointer[v] = Y1.singleton.append(i)
            # index i is no longer singleton
            elif len(Y1.List[i]) == 2:
                Y1.singleton.remove_node(self.singleton_pointer[Y1.List[i][0]])


class _ConnectedComp:
    """Connected component for D. & F.-B. algorithm (HDT data structure)."""

    def __init__(
        self,
        k: int,
        hdt_graph: HDTGraph,
        representative: NodeType,
        count: int = 0,
    ) -> None:
        """Constructor for connected component Y.

        Args:
            k: The number of tree for which to build a supertree.
            count: The cardinality of the overlap of Y and X_p (i.e., the number of unique leaf
                labels).
        """
        self.hdt_graph = hdt_graph

        # to find the component in the HDT datastructure
        self.representative = representative

        # the cardinality of the overlap of Y and X_p
        self.count = count

        # doubly-linked list that contains all indices i in [k] such that |U(i)| = 1
        self.singleton = DLList()

        # a list where, for each i in [k], Y.List[i] is a doubly-linked list consisting of the
        # elements of the intersection of Y and U(i)
        self.List: list[DLList] = [DLList() for _ in range(k)]

    def __len__(self) -> int:
        """Return the number of elements in this connected component.

        Returns:
            The number of elements in this connected component.
        """
        return self.hdt_graph.get_component(self.representative).num_active_occurrences

    def __iter__(self) -> Iterator[TreeNode | _XpNode]:
        """Iterator for the elements in the component.

        Caution: If the HDT datastructure is used this will only work if a representative of the
        component is set correctly.

        Yields:
            The elements in this component.
        """
        yield from self.hdt_graph.component_iterator(self.representative)

    def __next__(self):
        pass

    def __contains__(self, key: NodeType) -> bool:
        """Check if a given key is element of the component.

        Args:
            key: The item for which to check whether it is contained in this component.

        Returns:
            True if the item is contained.

        Raises:
            RuntimeError: If the representative has not been set.
        """
        if not self.representative:
            raise RuntimeError("missing representative for the component (2)")

        return self.hdt_graph.connected(self.representative, key)

    def initialize_tree_edges(self, tree: Tree) -> None:
        """Add all edges of a tree to the graph datastructure.

        Args:
            tree: A tree.
        """
        self.hdt_graph.add_loose_tree(tree)

    def add_edge(self, u: NodeType, v: NodeType) -> None:
        """Add an edge between two nodes to the graph datastructure.

        Args:
            u: The first node.
            v: The second node.
        """
        self.hdt_graph.insert_edge(u, v)

    def delete_edge(self, u: NodeType, v: NodeType) -> list[_ConnectedComp]:
        """Delete an edge and return the 1 or 2 connected components of u and v.

        The method returns the list of connected components into which this connected component
        breaks apart. It may contain only this connected component if the components do not change.
        The first return component is always this _ConnectedComp and contains u. If a second
        connected component is returned, that one is a new instance and contains v.

        Args:
            u: The first node.
            v: The second node.

        Returns:
            The list of connected components into which this connected component breaks apart (may
            contain only this connected component if the components do not change).
        """
        self.hdt_graph.delete_edge(u, v)

        # the first returned component will be this instance and contain u
        self.representative = u

        if self.hdt_graph.connected(u, v):
            return [self]
        else:
            new_component = _ConnectedComp(len(self.List), self.hdt_graph, v)
            return [self, new_component]

    def get_label_nodes(self) -> list[_XpNode]:
        """Return the label nodes in the connected component.

        Returns:
            A list of label nodes (_XpNode instances) in this components
        """
        result = []
        if not self.representative:
            raise RuntimeError("missing representative for the component (1)")

        for node in self.hdt_graph.component_iterator(self.representative):
            if isinstance(node, _XpNode):
                result.append(node)

        return result
