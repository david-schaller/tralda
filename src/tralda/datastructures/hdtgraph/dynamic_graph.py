"""Poly-logarithmic dynamic graph algorithm.

Implementation of the poly-logarithmic dynamic graph structure described by Holm et al. 2001.
The datastructure uses Euler tour trees (Henzinger and King 1999) to determine in O(log n) time
whether to given nodes are connected.

References:
    - Jacob Holm, Kristian de Lichtenberg, and Mikkel Thorup.
      Poly-logarithmic deterministic fully-dynamic algorithms for
      connectivity, minimum spanning tree, 2-edge, and biconnectivity.
      J. ACM, 48(4):723–760, July 2001.
    - Monika Rauch Henzinger, Valerie King. Randomized fully dynamic graph
      algorithms with polylogarithmic time per operation. J. ACM 46(4)
      July 1999. 502–536.
"""

from __future__ import annotations

from tralda.datastructures.hdtgraph.et_tree import ETTree
from tralda.datastructures.hdtgraph.et_tree import ETTreeNode
from tralda.datastructures.hdtgraph.et_tree import DGNode
from tralda.datastructures.tree import Tree


class Edge:
    __slots__ = ["e", "level", "tree_edge", "dllist_entries"]

    def __init__(self, e, level=0, tree_edge=False):
        self.e = e  # edge as tuple (u, v)
        self.level = level  # current level of this edge
        self.tree_edge = tree_edge  # is it a tree edge

        # reference to doubly-linked-list entries (both ends)
        self.dllist_entries = None

    def __eq__(self, other):
        return self.e == other.e

    def __hash__(self):
        return hash(self.e)

    def __repr__(self):
        tree = " tree" if self.tree_edge else ""
        return "{},{}{}".format(self.e, self.level, tree)


class Level:
    def __init__(self, index):
        self.index = index
        self.forest = set()  # spanning forest (ET trees)
        self.nodedict = dict()  # maps value --> DGNode (of this level)

    def connected(self, u, v):
        """Determine whether u and v are connected on this level."""
        node1, node2 = None, None
        if u in self.nodedict:
            node1 = self.nodedict[u]

        if v in self.nodedict:
            node2 = self.nodedict[v]

        if not (node1 and node2):
            return False

        return node1.active_occ.get_root() is node2.active_occ.get_root()

    def add_node(self, v):
        """Add a loose node on this level."""
        new_etnode = ETTreeNode(v, active=True)
        self.nodedict[v] = DGNode(v, active_occ=new_etnode)
        self.forest.add(
            ETTree(root=new_etnode, nodedict=self.nodedict, start=new_etnode, end=new_etnode)
        )

    def connect(self, u, v, edge=None):
        """Connect to ETTs by the edge (u,v) on this level.

        Keyword argument:
            edge - reference to the edge (u,v), use this if the edge is
                   a tree edge on this level, default=None
        """
        if u not in self.nodedict:
            self.add_node(u)

        if v not in self.nodedict:
            self.add_node(v)

        ett1 = self.nodedict[u].active_occ.get_root().ett
        ett2 = self.nodedict[v].active_occ.get_root().ett

        if ett1 is ett2:
            print("Nodes", u, "and", v, "on level", self.index, "are already connected!")
            return

        new_ett = ETTree.link(ett1, ett2, u, v)
        if not new_ett:
            print("Link operation was not successful!")
            return

        if edge:
            edge.dllist_entries = (
                self.nodedict[u].tree_edges.append(edge),
                self.nodedict[v].tree_edges.append(edge),
            )

        self.forest.remove(ett1)
        self.forest.remove(ett2)
        self.forest.add(new_ett)

        return True

    def add_nontree_edge(self, e):
        """Add a non-tree edge to its end points on this level."""

        u, v = e.e[0], e.e[1]
        if u not in self.nodedict:  # Can this case ever happen?
            self.add_node(u)
        if v not in self.nodedict:  # Can this case ever happen?
            self.add_node(v)
        e.dllist_entries = (
            self.nodedict[u].nontree_edges.append(e),
            self.nodedict[v].nontree_edges.append(e),
        )

    def cut(self, u, v):
        """Cut the tree edge (u,v) on this level."""
        if not (u in self.nodedict and v in self.nodedict):
            print("Could not find nodes", u, "and", v, "on level", self.index, "!")
            return

        ett1 = self.nodedict[u].active_occ.get_root().ett
        ett2 = self.nodedict[v].active_occ.get_root().ett

        if ett1 is not ett2:
            print("Nodes", u, "and", v, "are not connected on level", self.index, "!")

        result = ett1.cut(u, v)

        if not result:
            print("Nodes", u, "and", v, "on level", self.index, "are not connected!")
            return

        # result[0] is the original instance and therefore already in the forest
        self.forest.add(result[1])

        return True


class HDTGraph:
    def __init__(self):
        self.levels = [Level(0)]
        self.edges = dict()

    def get_nodes(self):
        """Generator for the nodes in the graph."""
        yield from self.levels[0].nodedict.keys()

    def get_edges(self):
        """Generator for the edges in the graph."""
        yield from self.edges.keys()

    def has_node(self, v):
        """Determine whether a node is in the graph."""
        return v in self.levels[0].nodehash

    def has_edge(self, u, v):
        """Determine whether an edge is in the graph."""
        try:
            if u > v:
                u, v = v, u
        except TypeError:
            if id(u) > id(v):
                u, v = v, u

        return (u, v) in self.edges

    def connected(self, u, v, level=0):
        """Determine in O(log n) whether u and v are connected in the graph."""
        return self.levels[level].connected(u, v)

    def get_component(self, u):
        """Return the connected component (ETT) of a given value."""
        if u in self.levels[0].nodedict:
            return self.levels[0].nodedict[u].active_occ.get_root().ett
        else:
            return None

    def insert_node(self, v):
        """Insert a loose node into the graph."""
        if v in self.levels[0].nodedict:
            print("Node", v, "is already in the graph.")
        else:
            self.levels[0].add_node(v)

    def insert_edge(self, u, v):
        """Insert an edge into the graph."""
        try:
            if u > v:
                u, v = v, u
        except TypeError:
            if id(u) > id(v):
                u, v = v, u
        e = (u, v)
        if e in self.edges:  # edge already exists
            return

        if self.connected(u, v):
            new_edge = Edge(e, tree_edge=False)  # new edge on level 0
            self.levels[0].add_nontree_edge(new_edge)  # append nontree edge to u and v
        else:
            new_edge = Edge(e, tree_edge=True)  # new edge on level 0
            self.levels[0].connect(u, v, edge=new_edge)  # connect + append tree edge to u and v
        self.edges[e] = new_edge

    def delete_edge(self, u, v):
        """Delete an edge from the graph."""
        try:
            if u > v:
                u, v = v, u
        except TypeError:
            if id(u) > id(v):
                u, v = v, u
        e = (u, v)
        if e not in self.edges:  # edge does not exist
            return
        e = self.edges[e]
        self.edges.pop(e.e)
        level = self.levels[e.level]

        # e is not a tree edge --> simply delete e
        if not e.tree_edge:
            if u in level.nodedict:
                level.nodedict[u].nontree_edges.remove_node(e.dllist_entries[0])
            if v in level.nodedict:
                level.nodedict[v].nontree_edges.remove_node(e.dllist_entries[1])

        # e is a tree edge --> search for replacement
        else:
            level.nodedict[u].tree_edges.remove_node(e.dllist_entries[0])
            level.nodedict[v].tree_edges.remove_node(e.dllist_entries[1])
            # remove e on all levels <= l(e)
            for i in range(level.index, -1, -1):
                if not self.levels[i].cut(u, v):
                    print("Something went wrong on level", i)
                    return

            self._replace(u, v, level)

    def _replace(self, u, v, level):
        """Search for an replacement edge to reconnect u and v on the given level."""
        ett1 = level.nodedict[u].active_occ.get_root().ett
        ett2 = level.nodedict[v].active_occ.get_root().ett

        if ett1.get_size() <= ett2.get_size():
            smaller_ett = ett1
            root2 = ett2.root
        else:
            smaller_ett = ett2
            root2 = ett1.root

        self._raise_tree_edges(smaller_ett, level)
        found = None

        for occ in smaller_ett:
            if occ.active:
                dgnode = level.nodedict[occ.value]
                while dgnode.nontree_edges:
                    f = dgnode.nontree_edges.popleft()
                    if f.e[0] == dgnode.value:
                        other_end = f.e[1]
                        level.nodedict[other_end].nontree_edges.remove_node(f.dllist_entries[1])
                    else:
                        other_end = f.e[0]
                        level.nodedict[other_end].nontree_edges.remove_node(f.dllist_entries[0])

                    # f is a replacement edge
                    if (other_end in level.nodedict) and (
                        level.nodedict[other_end].active_occ.get_root() is root2
                    ):
                        found = f
                        break
                    # f is not a replacement edge --> raise to next level
                    else:
                        f.level += 1
                        self.levels[level.index + 1].add_nontree_edge(f)
            if found:
                break

        if found:
            found.tree_edge = True
            found.dllist_entries = None
            u, v = found.e[0], found.e[1]
            level.connect(u, v, found)
            for i in range(level.index - 1, -1, -1):  # reconnect on all lower levels
                self.levels[i].connect(u, v)
        elif level.index > 0:
            self._replace(u, v, self.levels[level.index - 1])

    def _raise_tree_edges(self, ett, level):
        """Raise all tree edges of an ETT to the next level."""
        if level.index >= len(self.levels) - 1:
            self.levels.append(Level(len(self.levels)))  # create new level

        for occ in ett:
            if occ.active:
                dgnode = level.nodedict[occ.value]
                for tree_edge in dgnode.tree_edges:
                    if tree_edge.level == level.index:
                        tree_edge.level += 1
                        u, v = tree_edge.e[0], tree_edge.e[1]
                        self.levels[level.index + 1].connect(u, v, edge=tree_edge)

                # remove all tree edges (of this ET tree) on this level
                dgnode.tree_edges.clear()

    def _add_tree_edges(self, node):
        nodedict = self.levels[0].nodedict
        for child in node.children:
            nodedict[child] = DGNode(child)
            if id(node) < id(child):
                e = (node, child)
            else:
                e = (child, node)

            new_edge = Edge(e, level=0, tree_edge=True)
            new_edge.dllist_entries = (
                nodedict[e[0]].tree_edges.append(new_edge),
                nodedict[e[1]].tree_edges.append(new_edge),
            )
            self.edges[e] = new_edge
            self._add_tree_edges(child)

    def add_loose_tree(self, tree):
        """Add the edges of a tree as undirected edges.

        The elements of the 'Tree' instance must not yet be nodes in the graph.
        """

        if not isinstance(tree, Tree):
            raise TypeError("instance of type 'Tree' required")

        nodedict = self.levels[0].nodedict
        nodedict[tree.root] = DGNode(tree.root)
        self._add_tree_edges(tree.root)
        ett = ETTree.initialize_from_tree(tree, nodedict=nodedict)
        if not ett:
            return
        self.levels[0].forest.add(ett)

    def is_connected(self):
        """Determine if the graph is connected (on level 0)."""

        if len(self.levels[0].forest) == 1:
            for ett in self.levels[0].forest:
                return ett
        else:
            return False

    def component_iterator(self, representative):
        """Iterator over the connected component of a given value."""

        for occ in self.get_component(representative):
            if occ.active:
                yield occ.value

    def print_ett_forest(self, level=0):
        """Print Euler tour of the spanning forest.

        Intended for testing purpose.
        Keyword argument:
            level - level of the ETT spanning forest to be printed,
                    choose "all" for printing all levels, default=0
        """

        if level == "all":
            for level_i in self.levels:
                print("----- Level {} -----".format(level_i.index))
                for ett in level_i.forest:
                    print(ett.ET_to_list(), ett.get_size())
        else:
            for ett in self.levels[level].forest:
                print(ett.ET_to_list(), ett.get_size())
