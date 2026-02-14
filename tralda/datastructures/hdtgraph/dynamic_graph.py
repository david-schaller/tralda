"""Poly-logarithmic dynamic graph algorithm.

Implementation of the poly-logarithmic dynamic graph structure described by Holm et al. 2001.
The datastructure uses Euler tour trees (Henzinger and King 1999) to determine in O(log n) time
whether two given nodes are connected.

References:
    .. [1] Jacob Holm, Kristian de Lichtenberg, and Mikkel Thorup. Poly-logarithmic deterministic
           fully-dynamic algorithms for connectivity, minimum spanning tree, 2-edge, and
           biconnectivity. J. ACM, 48(4):723-760, July 2001.
    .. [2] Monika Rauch Henzinger, Valerie King. Randomized fully dynamic graph algorithms with
           polylogarithmic time per operation. J. ACM 46(4). July 1999. 502-536.
"""

from __future__ import annotations

from typing import Any
from typing import Iterator

from tralda.datastructures.doubly_linked import DLList
from tralda.datastructures.hdtgraph.et_tree import ETTree
from tralda.datastructures.hdtgraph.et_tree import ETTreeNode
from tralda.datastructures.hdtgraph.et_tree import EdgeOccurrences
from tralda.datastructures.tree import Tree
from tralda.utils.graph_tools import sort_edge


class _Edge:
    """Class for storing edge attributes and associated references."""

    __slots__ = ["e", "level", "is_tree_edge", "dllist_entries"]

    def __init__(
        self,
        e: tuple[Any, Any],
        level: int = 0,
        is_tree_edge: bool = False,
    ) -> None:
        """Constructor od the _Edge class.

        Args:
            e: A tuple containing the two endpoints of the edge.
            level: The level of the edge.
            is_tree_edge: Whether the edge is a tree edge.
        """
        self.e = e  # edge as tuple (u, v)
        self.level = level
        self.is_tree_edge = is_tree_edge

        # reference to doubly-linked-list entries (both ends)
        self.dllist_entries = None


class _Level:
    """Class for maintaining the Euler Tour tree forest of a level in the HDT datastructure."""

    def __init__(self, index: int) -> None:
        """Constructor of the _Level class.

        Args:
            index: The index of the level in the HDT datastructure.
        """
        self.index = index
        self.forest: set[ETTree] = set()

        self.node2active_occurrence: dict[Any, ETTreeNode] = {}
        self.node2tree_edges: dict[Any, DLList] = {}
        self.node2nontree_edges: dict[Any, DLList] = {}

        # references to the occurrences of the nodes in the Euler Tour tree
        self.edge2occurrences: dict[tuple[Any, Any], EdgeOccurrences] = {}

    def connected(self, u: Any, v: Any) -> bool:
        """Determine whether u and v are connected on this level.

        Args:
            u: The first element.
            v: The second element.

        Returns:
            Whether u and v are connected on this level.
        """
        ettree_node1 = self.node2active_occurrence.get(u)
        ettree_node2 = self.node2active_occurrence.get(v)

        if not (ettree_node1 and ettree_node2):
            return False

        return ettree_node1.get_root() is ettree_node2.get_root()

    def add_node(self, v: Any) -> None:
        """Add a node and add it as a single-node Euler Tour tree to the forest on this level.

        Args:
            u: The element to add as a node on this level.
        """
        ett_node = self.add_loose_node(v)

        # Euler Tour tree that will contain only the new node
        ett = ETTree()
        ett.root = ett_node

        # the root stores a reference to the ETTree instance
        ett_node.ett = ett

        self.forest.add(ett)

    def add_loose_node(self, v: Any) -> ETTree:
        """Add a loose node on this level.

        This function does not create a Euler Tour tree that contains the new node.

        Args:
            u: The element to add as a node on this level.
        """
        ett_node = ETTreeNode(v, active=True)

        self.node2active_occurrence[v] = ett_node
        self.node2tree_edges[v] = DLList()
        self.node2nontree_edges[v] = DLList()

        return ett_node

    def connect(self, u: Any, v: Any, edge: _Edge | None = None) -> None:
        """Connect two Euler Tour trees by the edge (u, v) on this level.

        Args:
            u: The first element.
            v: The second element.
            edge: Reference to the edge (u, v). Use this if the edge is a tree edge on this level.
        """
        if u not in self.node2active_occurrence:
            self.add_node(u)

        if v not in self.node2active_occurrence:
            self.add_node(v)

        ett1 = self.node2active_occurrence[u].get_root().ett
        ett2 = self.node2active_occurrence[v].get_root().ett

        if ett1 is ett2:
            raise RuntimeError(f"nodes {u} and {v} on level {self.index} are already connected")

        new_ett = ett1.join_by_edge(
            self.node2active_occurrence[u],
            self.node2active_occurrence[v],
            ett2,
            self.node2active_occurrence,
            self.edge2occurrences,
        )

        if edge:
            edge.dllist_entries = (
                self.node2tree_edges[u].append(edge),
                self.node2tree_edges[v].append(edge),
            )

        self.forest.remove(ett1)
        self.forest.remove(ett2)
        self.forest.add(new_ett)

    def add_nontree_edge(self, edge: _Edge) -> None:
        """Add a non-tree edge to its end points on this level.

        Args:
            edge: The edge to add as a non-tree edge on this level.
        """
        u, v = edge.e
        if u not in self.node2active_occurrence:  # Can this case ever happen?
            self.add_node(u)
        if v not in self.node2active_occurrence:  # Can this case ever happen?
            self.add_node(v)

        edge.dllist_entries = (
            self.node2nontree_edges[u].append(edge),
            self.node2nontree_edges[v].append(edge),
        )

    def cut(self, u: Any, v: Any) -> None:
        """Cut the tree edge (u, v) on this level.

        Args:
            u: The first endpoint of the edge.
            v: The second endpoint of the edge.
        """
        if u not in self.node2active_occurrence:
            raise KeyError(f"could not find node {u} on level {self.index}")

        if v not in self.node2active_occurrence:
            raise KeyError(f"could not find node {v} on level {self.index}")

        ett = self.node2active_occurrence[u].get_root().ett

        if ett is not self.node2active_occurrence[v].get_root().ett:
            raise RuntimeError(f"nodes {u} and {v} are not connected on level {self.index}")

        ett1, ett2 = ett.delete_edge(
            self.edge2occurrences[u, v],
            self.node2active_occurrence,
            self.edge2occurrences,
        )

        self.forest.remove(ett)
        self.forest.add(ett1)
        self.forest.add(ett2)


class HDTGraph:
    """Undirected graph datastructure for poly-logarithmic connectivity queries.

    Implementation of the poly-logarithmic dynamic graph structure (HDT datastructure).

    References:
        .. [1] Jacob Holm, Kristian de Lichtenberg, and Mikkel Thorup. Poly-logarithmic
            deterministic fully-dynamic algorithms for connectivity, minimum spanning tree, 2-edge,
            and biconnectivity. J. ACM, 48(4):723-760, July 2001.
    """

    def __init__(self) -> None:
        """Constructor of the undirected graph for poly-logarithmic connectivity queries."""
        self._levels: list[_Level] = [_Level(0)]
        self._edges: dict[tuple[Any, Any], _Edge] = dict()

    def get_nodes(self) -> Iterator[Any]:
        """Generator for the nodes in the graph.

        Yields:
            The nodes in the graph.
        """
        yield from self._levels[0].node2active_occurrence.keys()

    def get_edges(self) -> Iterator[tuple[Any, Any]]:
        """Generator for the edges in the graph.

        Yields:
            The edges in the graph.
        """
        yield from self._edges.keys()

    def has_node(self, node: Any) -> bool:
        """Determine whether a node is in the graph.

        Args:
            node: The node for which to check if it is in the graph.

        Return:
            Whether the node is contained in the graph.
        """
        return node in self._levels[0].node2active_occurrence

    def has_edge(self, u: Any, v: Any) -> bool:
        """Determine whether an edge is in the graph.

        Args:
            u: The first node.
            v: The second node.

        Returns:
            Whether the graph has the undirected edge (u, v).
        """
        return sort_edge(u, v) in self._edges

    def connected(self, u: Any, v: Any, level: int = 0) -> bool:
        """Determine in O(log n) whether u and v are connected in the graph.

        Args:
            u: The first node.
            v: The second node.
            level: If provided, check for connectivity on that level. The default is 0 which checks
                for connectivity in the full current graph.

        Returns:
            Whether u and v are connected by a path in the graph.
        """
        return self._levels[level].connected(u, v)

    def get_component(self, node: Any) -> ETTree:
        """Return the connected component (ETT) of a given value.

        Args:
            node: A node in the graph.

        Returns:
            The connected component that contains the node.

        Raises:
            KeyError: If the node is not in the graph.
        """
        active_occurrence = self._levels[0].node2active_occurrence.get(node)

        if active_occurrence is None:
            raise KeyError(f"{node} is not a node in the graph")

        return active_occurrence.get_root().ett if active_occurrence is not None else None

    def insert_node(self, node: Any) -> None:
        """Insert a loose node into the graph.

        Args:
            node: The node to add to the graph.

        Raises:
            KeyError: If the node is already in the graph.
        """
        if node in self._levels[0].node2active_occurrence:
            raise KeyError(f"node {node} is already in the graph")

        self._levels[0].add_node(node)

    def insert_edge(self, u: Any, v: Any) -> None:
        """Insert an edge (u, v) into the graph.

        Args:
            u: The first node.
            v: The second node.
        """
        u, v = sort_edge(u, v)
        e = (u, v)

        if e in self._edges:
            return

        if self.connected(u, v):
            new_edge = _Edge(e, is_tree_edge=False)  # new edge on level 0
            self._levels[0].add_nontree_edge(new_edge)  # append nontree edge to u and v
        else:
            new_edge = _Edge(e, is_tree_edge=True)  # new edge on level 0
            self._levels[0].connect(u, v, edge=new_edge)  # connect + append tree edge to u and v

        self._edges[e] = new_edge

    def delete_edge(self, u: Any, v: Any) -> None:
        """Delete an edge from the graph.

        Args:
            u: The first endpoint of the edge.
            v: The second endpoint of the edge.
        """
        u, v = sort_edge(u, v)
        edge = self._edges.get((u, v))

        if edge is None:  # edge does not exist
            return

        self._edges.pop(edge.e)
        level = self._levels[edge.level]

        # e is not a tree edge --> simply delete e
        if not edge.is_tree_edge:
            if u in level.node2active_occurrence:
                level.node2nontree_edges[u].remove_node(edge.dllist_entries[0])
            if v in level.node2active_occurrence:
                level.node2nontree_edges[v].remove_node(edge.dllist_entries[1])

        # e is a tree edge --> search for replacement
        else:
            level.node2tree_edges[u].remove_node(edge.dllist_entries[0])
            level.node2tree_edges[v].remove_node(edge.dllist_entries[1])

            # remove the edge on all levels <= l(e)
            for i in range(level.index, -1, -1):
                self._levels[i].cut(u, v)

            self._replace(u, v, level)

    def add_loose_tree(self, tree: Tree) -> None:
        """Add the edges of a (rooted) tree as undirected edges.

        The elements of the 'Tree' instance must not yet be nodes in the graph.

        Args:
            tree: The tree whose nodes and edges shall be added to the graph.

        Raises:
            TypeError: If tree is not an instance of type Tree.
        """
        if not isinstance(tree, Tree):
            raise TypeError("instance of type 'Tree' required")

        # empty tree --> nothing to add
        if not tree.root:
            return

        node2active_occurrence = self._levels[0].node2active_occurrence
        node2tree_edges = self._levels[0].node2tree_edges
        edge2occurrences = self._levels[0].edge2occurrences

        ett = ETTree()

        # construct the Euler Tour tree
        previous_ett_node = None
        for node in tree.euler_generator():
            if node not in node2active_occurrence:
                ett_node = self._levels[0].add_loose_node(node)
            else:
                ett_node = ETTreeNode(node, active=False)

            ett.add_right_child_and_rebalance(previous_ett_node, ett_node, edge2occurrences)
            previous_ett_node = ett_node

        # create and add the _Edge instances (which are already represented in the ET tree)
        for node in tree.preorder():
            if not node.parent:
                continue

            e = sort_edge(node, node.parent)
            edge = _Edge(e, level=0, is_tree_edge=True)
            edge.dllist_entries = (
                node2tree_edges[e[0]].append(edge),
                node2tree_edges[e[1]].append(edge),
            )
            self._edges[e] = edge

        # the root stores a reference to the ETTree instance
        ett.root.ett = ett

        # finally, add the new component to the forest
        self._levels[0].forest.add(ett)

    def is_connected(self) -> bool:
        """Determine if the graph is connected (on level 0).

        Returns:
            Whether the graph is connected.
        """
        return len(self._levels[0].forest) == 1

    def component_iterator(self, representative: Any) -> Iterator[Any]:
        """Iterator over the connected component of a given value.

        Args:
            representative: A representative node of the component.

        Yields:
            The nodes in the connected component.
        """
        ett = self.get_component(representative)

        if ett is None:
            raise KeyError(f"representative node {representative} is not contained in the graph")

        for occ in ett:
            if occ.active:
                yield occ.key

    def print_ett_forest(self, level: str | int = "all") -> None:
        """Print Euler tour of the spanning forest.

        Intended for testing purpose.

        Args:
            level: Level of the ETT spanning forest to be printed. The default is "all" in which
                case all levels are printed.
        """
        if level == "all":
            for level_i in self._levels:
                print(f"----- Level {level_i.index} -----")
                for ett in level_i.forest:
                    print([occ.key for occ in ett], ett.num_active_occurrences)
        elif isinstance(level, int):
            for ett in self._levels[level].forest:
                print([occ.key for occ in ett], ett.num_active_occurrences)
        else:
            ValueError(f"invalid level: {level} of type {type(level)}")

    def _replace(self, u: Any, v: Any, level: _Level) -> None:
        """Search for an replacement edge to reconnect u and v on the given level.

        Args:
            u: The first node.
            v: The second node.
            level: The level on which to search for a replacement edge of (u, v).
        """
        ett1 = level.node2active_occurrence[u].get_root().ett
        ett2 = level.node2active_occurrence[v].get_root().ett

        if ett1.num_active_occurrences <= ett2.num_active_occurrences:
            smaller_ett = ett1
            root2 = ett2.root
        else:
            smaller_ett = ett2
            root2 = ett1.root

        self._raise_tree_edges(smaller_ett, level)
        found = None

        for occ in smaller_ett:
            if not occ.active:
                continue

            nontree_edges = level.node2nontree_edges[occ.key]

            while nontree_edges:
                edge = nontree_edges.popleft()

                # also remove the non-tree edge at the other node
                i = 1 - int(edge.e[0] != occ.key)
                other_end = edge.e[i]
                level.node2nontree_edges[other_end].remove_node(edge.dllist_entries[i])

                # edge is a replacement edge
                if (other_end in level.node2active_occurrence) and (
                    level.node2active_occurrence[other_end].get_root() is root2
                ):
                    found = edge
                    break

                # edge is not a replacement edge --> raise to next level
                edge.level += 1
                self._levels[level.index + 1].add_nontree_edge(edge)

            if found:
                break

        if found:
            found.is_tree_edge = True
            found.dllist_entries = None
            u, v = found.e[0], found.e[1]
            level.connect(u, v, found)

            # reconnect on all lower levels
            for i in range(level.index - 1, -1, -1):
                self._levels[i].connect(u, v)

        # no replament edge found --> try on lower levels if available
        elif level.index > 0:
            self._replace(u, v, self._levels[level.index - 1])

    def _raise_tree_edges(self, ett: ETTree, level: _Level) -> None:
        """Raise all tree edges of an Euler Tour tree (ETT) to the next level.

        Args:
            ett: The ETT whose edges shall be raised up a level.
            level: The current level of the edges represented by the ETT.
        """
        # create new level if necessary
        if level.index >= len(self._levels) - 1:
            self._levels.append(_Level(len(self._levels)))

        for occ in ett:
            if not occ.active:
                continue

            tree_edges = level.node2tree_edges[occ.key]

            for edge in tree_edges:
                if edge.level == level.index:
                    edge.level += 1
                    u, v = edge.e[0], edge.e[1]
                    self._levels[level.index + 1].connect(u, v, edge=edge)

            # remove all tree edges (of this ET tree) on this level
            tree_edges.clear()
