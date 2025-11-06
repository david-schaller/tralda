"""Module for the main tree datastructure in tralda."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Iterator
from typing import Any
import itertools
import json
import os
import pickle
import random

import networkx as nx

import tralda.datastructures.doubly_linked as dll


class TreeNode:
    """Tree nodes for class Tree.

    Attributes:
        parent (TreeNode): Parent node of this node.
        children (dll.DLList): Child nodes of this node in a doubly-linked list.
    """

    def __init__(self, **kwargs) -> None:
        """Constructor for TreeNode class.

        Args:
            **kwargs: Set arbitrary node attributes using key=value.
        """
        self.parent = None
        # reference to doubly-linked list element in the parents' children
        self._par_dll_node = None

        self.children = dll.DLList()

        self.__dict__.update(kwargs)

    def __str__(self) -> str:
        """Return the label of the tree node if available.

        Returns:
            The label of the tree node if available or an empty string.
        """
        return str(self.label) if hasattr(self, "label") else ""

    def __repr__(self) -> str:
        """Return a string representation of the tree node.

        If available, the `label` attribute will be contained, otherwise the object ID of the
        TreeNode instance.

        Returns:
            A string representation of the tree node.
        """
        return f"<TN: {self.label if hasattr(self, 'label') else id(self)}>"

    def attributes(self) -> Iterator[tuple[str, Any]]:
        """A generator for the node attributes.

        Yields:
            Pairs of str and the type of the corresponding value.
        """
        for key, value in self.__dict__.items():
            if key not in ("parent", "children", "_par_dll_node"):
                yield key, value

    def add_child(self, child_node: TreeNode) -> None:
        """Add a node as a child of this node.

        Does nothing if the node is already a child node of this node.

        Args:
            child_node: The node to add as a new child to this node.
        """
        if child_node.parent is None:
            child_node.parent = self
            child_node._par_dll_node = self.children.append(child_node)
        elif child_node.parent is not self:
            child_node.parent.remove_child(child_node)
            child_node.parent = self
            child_node._par_dll_node = self.children.append(child_node)

    def add_child_right_of(self, child_node: TreeNode, right_of: TreeNode) -> None:
        """Add a node as a child of this node as a right sibling of one of its children.

        Can also be used to change the position of a child node, i.e., to detach it and re-insert
        it to the right of the specified node.

        Args:
            child_node: The node to add as a new child to this node.
            right_of: The child of this node right of which 'child_node' gets inserted.

        Raises:
            KeyError: If 'right_of' is not a child of this node.
        """
        if right_of.parent is not self:
            raise KeyError(f"{right_of} is not a child of node {self}")

        if child_node.parent is not None:
            child_node.parent.remove_child(child_node)

        child_node.parent = self
        child_node._par_dll_node = self.children.insert_right_of(right_of._par_dll_node, child_node)

    def remove_child(self, child_node: TreeNode) -> None:
        """Remove a child node of this node.

        Args:
            child_node: The node to be removed from the list of children.

        Raises:
            KeyError: If the supplied node is not a child of this node.
        """
        if child_node.parent is self:
            self.children.remove_node(child_node._par_dll_node)
            child_node.parent = None
            child_node._par_dll_node = None
        else:
            raise KeyError(f"{child_node} is not a child of node {self}")

    def detach(self) -> None:
        """Detach this node from its parent.

        The node has no parent afterwards.
        """
        if self.parent is not None:
            self.parent.remove_child(self)
        else:
            self.parent = None
            self._par_dll_node = None

    def is_leaf(self) -> bool:
        """Return True if the node is a leaf, False otherwise.

        Returns:
            True if the node is a leaf, i.e., it has no children, else False.
        """
        return not self.children

    def child_subsequence(self, left_node: TreeNode, right_node: TreeNode) -> list[TreeNode]:
        """Consecutive subsequence of children within a left and right bound.

        Args:
            left_node: The left bound of the subsequence.
            right_node: The right bound of the subsequence.

        Returns:
            The children in the subsequence.

        Raises:
            KeyError: If 'right_node' or 'left_node' is not a child of this node.
        """
        if left_node.parent is not self:
            raise KeyError(f"{left_node} is not a child of node {self}")
        if right_node.parent is not self:
            raise KeyError(f"{right_node} is not a child of node {self}")

        return self.children.sublist(left_node._par_dll_node, right_node._par_dll_node)


class Tree:
    """Rooted tree whose nodes may have an arbitrary number of children.

    Attributes:
        root (TreeNode): The root node of the tree.
    """

    def __init__(self, arg: TreeNode | str) -> None:
        """Constructor for the class Tree.

        Args:
            arg: The root node for the newly created tree or a Newick representation of a tree.

        Raises:
            TypeError: If 'arg' is neither a TreeNode instance nor a string.
        """
        if isinstance(arg, TreeNode) or arg is None:
            self.root = arg
        elif isinstance(arg, str):
            self.root = Tree._parse_newick_and_return_root(arg)
        else:
            raise TypeError(f"Tree cannot be initialized with argument of type {type(arg)}")

    def __len__(self) -> int:
        """Size of the tree (number of nodes).

        Runs in O(n) where n is the size of the tree.

        Returns:
            The number of nodes in the tree.
        """
        return sum(1 for _ in self.preorder())

    def height(self) -> int:
        """Height of the tree (number of edges in the longest path from the root to a leaf).

        Runs in O(n) where n is the size of the tree. Return -1 if the tree is empty.

        Returns:
            The height of the tree.
        """
        if not self.root:
            return -1

        return max(level for _, level in self.preorder_and_level())

    def leaves(self) -> Iterator[TreeNode]:
        """Generator for leaves of the tree.

        Yields:
            The leaf nodes of the tree.
        """

        def _leaves(node: TreeNode) -> Iterator[TreeNode]:
            if not node.children:
                yield node
            else:
                for child in node.children:
                    yield from _leaves(child)

        if self.root:
            yield from _leaves(self.root)
        else:
            yield from []

    def preorder(self) -> Iterator[TreeNode]:
        """Generator for a pre-order traversal of the tree.

        Yields:
            All nodes of the tree in pre-order.
        """

        def _preorder(node: TreeNode) -> Iterator[TreeNode]:
            yield node
            for child in node.children:
                yield from _preorder(child)

        if self.root:
            yield from _preorder(self.root)
        else:
            yield from []

    def preorder_and_level(self) -> Iterator[tuple[TreeNode, int]]:
        """Generator for a pre-order traversal with node levels.

        Yields:
            Nodes and their level (distance from the root) in pre-order.
        """

        def _preorder_level(node: TreeNode, level: int) -> Iterator[tuple[TreeNode, int]]:
            yield (node, level)
            for child in node.children:
                yield from _preorder_level(child, level + 1)

        if self.root:
            yield from _preorder_level(self.root, 0)
        else:
            yield from []

    def traverse_subtree(self, node: TreeNode) -> Iterator[TreeNode]:
        """Generator for a pre-order traversal of the subtree rooted at the given node.

        Args:
            node: The node whose subtree to traverse.

        Yields:
            All nodes in the subtree rooted at the given node in pre-order.
        """
        yield node

        for child in node.children:
            yield from self.traverse_subtree(child)

    def postorder(self) -> Iterator[TreeNode]:
        """Generator for post-order traversal of the tree.

        Yields:
            All nodes of the tree in post-order.
        """

        def _postorder(node: TreeNode) -> Iterator[TreeNode]:
            for child in node.children:
                yield from _postorder(child)
            yield node

        if self.root:
            yield from _postorder(self.root)
        else:
            yield from []

    def inner_nodes(self) -> Iterator[TreeNode]:
        """Generator for inner nodes in pre-order.

        Inner nodes are those nodes that do not have any children.

        Yields:
            All inner nodes of the tree in pre-order.
        """

        def _inner_nodes(node: TreeNode) -> Iterator[TreeNode]:
            if node.children:
                yield node
                for child in node.children:
                    yield from _inner_nodes(child)

        if self.root:
            yield from _inner_nodes(self.root)
        else:
            yield from []

    def edges(self) -> Iterator[tuple[TreeNode, TreeNode]]:
        """Generator for all edges of the tree.

        Yields:
            All edges of the tree as tuples of TreeNode instances.
        """

        def _edges(node: TreeNode) -> Iterator[tuple[TreeNode, TreeNode]]:
            for child in node.children:
                yield (node, child)
                yield from _edges(child)

        if self.root:
            yield from _edges(self.root)
        else:
            yield from []

    def edges_sibling_order(self) -> Iterator[tuple[TreeNode, TreeNode, int]]:
        """Generator for all edges of the tree with sibling order.

        Yields:
            Edges uv as tuples (u, v, nr) where nr is the index of v in the list of children of
            node u.
        """

        def _edges_sibling_order(node: TreeNode) -> Iterator[tuple[TreeNode, TreeNode, int]]:
            i = 0
            for child in node.children:
                yield (node, child, i)
                yield from _edges_sibling_order(child)
                i += 1

        if self.root:
            yield from _edges_sibling_order(self.root)
        else:
            yield from []

    def inner_edges(self) -> Iterator[tuple[TreeNode, TreeNode]]:
        """Generator for all inner edges of the tree.

        Yields:
            All inner edges uv of the tree, i.e., edges for which the child v of u is not a leaf.
        """

        def _inner_edges(node: TreeNode) -> Iterator[tuple[TreeNode, TreeNode]]:
            for child in node.children:
                if child.children:
                    yield (node, child)
                    yield from _inner_edges(child)

        if self.root:
            yield from _inner_edges(self.root)
        else:
            yield from []

    def euler_generator(self) -> Iterator[TreeNode]:
        """Generator for an Euler tour of the tree.

        Yields:
            Nodes in an Euler tour of the tree.
        """

        def _euler_generator(node: TreeNode) -> Iterator[TreeNode]:
            yield node
            for child in node.children:
                yield from _euler_generator(child)
                yield node

        if self.root:
            yield from _euler_generator(self.root)
        else:
            yield from []

    def euler_and_level(self) -> Iterator[tuple[TreeNode, int]]:
        """Generator for an Euler tour with node levels.

        Yields:
            Nodes and their level (distance from the root) in an Euler tour of the tree.
        """

        def _euler_level(node: TreeNode, level: int) -> Iterator[tuple[TreeNode, int]]:
            yield (node, level)
            for child in node.children:
                yield from _euler_level(child, level + 1)
                yield (node, level)

        if self.root:
            yield from _euler_level(self.root, 0)
        else:
            yield from []

    def leaf_dict(self) -> dict[TreeNode, list[TreeNode]]:
        """Leaves in the subtree rooted at each node.

        Computes the list of leaves for every node in the tree containing the leaf nodes in the
        subtree rooted at the node.

        Returns:
            The leaves under each vertex (as a dictionary with TreeNode keys and lists of TreeNode
            objects as values).
        """
        leaves = {}

        for v in self.postorder():
            if not v.children:
                leaves[v] = [v]
            else:
                leaves[v] = []
                for child in v.children:
                    leaves[v].extend(leaves[child])

        return leaves

    def contract(self, edges: Iterable[tuple[TreeNode]], inplace: bool = True) -> Tree:
        """Contract edges in the tree.

        Args:
            edges: The edges to be contracted in the tree.
            inplace: If True, the edges are contracted in this tree instance, otherwise a copy is
                returned and the original tree is not affected.

        Returns:
            The original tree with contracted edges or a copy in which the contractions are
            performed.
        """
        contracted = set()

        if not inplace:
            T_copy, mapping = self.copy(mapping=True)

        for _, v in edges:
            # avoid trying to contract the same edge multiple times
            if v in contracted:
                continue

            if inplace:
                self.delete_and_reconnect(v)
            else:
                T_copy.delete_and_reconnect(mapping[v])

            contracted.add(v)

        return self if inplace else T_copy

    def get_triples(self, label_only: bool = False) -> list[tuple[Any, Any, Any]]:
        """Retrieve a list of all (rooted) triples of the tree.

        A tree displays a triple ab|c on the leaf nodes a, b and c if the last common ancestor of a
        and b is a (proper) descendant of the last common ancestor of a and c (b and c).

        Args:
            label_only: If True, the triples are represented by the label attribute of the nodes.

        Returns:
            List of tuples where each tuple (a, b, c) represents the triple ab|c (=ba|c), i.e., the
            first two items are closer related in the tree.
        """
        if label_only:
            return [(a.label, b.label, c.label) for a, b, c in self._triple_generator()]
        else:
            return [t for t in self._triple_generator()]

    def _triple_generator(self) -> Iterator[tuple[TreeNode, TreeNode, TreeNode]]:
        """Iterator for all (rooted) triples displayed by the tree.

        Yields:
            The triples displayed by the tree.
        """
        leaves = self.leaf_dict()

        for u in self.preorder():
            for v1, v2 in itertools.permutations(u.children, 2):
                if len(leaves[v2]) <= 1:
                    continue

                for c in leaves[v1]:
                    for a, b in itertools.combinations(leaves[v2], 2):
                        yield a, b, c

    def delete_and_reconnect(self, node: TreeNode) -> TreeNode | None:
        """Delete a node from the tree and reconnect its parent and children.

        The function preserves the 'sibling order' of the remaining nodes of the tree.

        Args:
            node: The node to be deleted.

        Returns:
            The parent of the node, if it could be deleted, or None, if the node could not be
            deleted, i.e., it has no parent.
        """
        parent = node.parent

        if not parent:
            return None
        else:
            # copy list of children to edit edges
            children = [child for child in node.children]

            for child in children:
                parent.add_child_right_of(child, node)

            parent.remove_child(node)
            node.children.clear()

        return parent

    def random_leaves(self, proportion: int | float) -> list[TreeNode]:
        """A random sample of the leaves.

        Args:
            proportion: The proportion of the sample w.r.t. the full set of leaves.

        Returns:
            A randomly samples list of the leaves of the tree.

        Raises:
            If 'proportion' is not a number between 0 and 1.
        """
        if not isinstance(proportion, (float, int)) or proportion < 0 or proportion > 1:
            raise ValueError("proportion must be a number 0 <= p <= 1")

        leaves = [v for v in self.leaves()]
        k = round(proportion * len(leaves))

        return random.sample(leaves, k)

    def is_binary(self) -> bool:
        """Check whether the tree is a binary tree.

        Nodes in (rooted) binary trees are either leaves or have exactly two children.

        Returns:
            True if the tree is binary, else False.
        """
        for v in self.preorder():
            if len(v.children) == 1 or len(v.children) > 2:
                return False

        return True

    def is_phylogenetic(self) -> bool:
        """Check whether the tree is a phylogetic tree.

        Nodes in (rooted) phylogentic trees are either leaves or have at least two children.

        Returns:
            True if the tree is phylogenetic, else False.
        """
        for v in self.preorder():
            if len(v.children) == 1:
                return False

        return True

    def get_hierarchy(self) -> set[tuple[Any]]:
        """Hierarchy set on the leaf labels defined by the tree.

        Every (phylogenetic) tree can be represented by a hierarchy on the set of its leaves. The
        label attributes of the leaf nodes must be set and unique for each leaf.

        Returns:
            A set of tuples representing the hierarchy where the leaves are represented by their
            labels.
        """
        leaves = self.leaf_dict()
        hierarchy = set()

        for v in self.preorder():
            hierarchy.add(tuple(sorted(leaf.label for leaf in leaves[v])))

        return hierarchy

    def equal_topology(self, other: Tree, verbose: bool = False) -> bool:
        """Compare the tree topology based on the leaf labels.

        Only works for phylogenetic trees with unique leaf labels.

        Args:
            other: Another tree which this tree shall be compared to.
            verbose: If True, print where the equality check failed (the first time).

        Returns:
            True if the topologies are equal, else False.
        """
        hierarchy1 = sorted(self.get_hierarchy())
        hierarchy2 = sorted(other.get_hierarchy())

        if len(hierarchy1) != len(hierarchy2):
            if verbose:
                print(
                    f"Unequal sizes of the hierarchy sets: {len(hierarchy1)} and {len(hierarchy2)}"
                )
            return False

        for i in range(len(hierarchy1)):
            if hierarchy1[i] != hierarchy2[i]:
                if verbose:
                    print(f"Hierarchies not equal:\n{hierarchy1[i]}\n{hierarchy2[i]}")
                return False

        return True

    def is_refinement(self, other: Tree) -> bool:
        """Checks whether the tree is a refinement of 'other' based on the leaf labels.

        Only works for phylogenetic trees with unique leaf labels.

        Args:
            other: Another tree which this tree shall be compared to.

        Returns:
            True if the tree is refinement of 'other', else False.
        """
        hierarchy1 = sorted(self.get_hierarchy())
        hierarchy2 = sorted(other.get_hierarchy())

        if len(hierarchy1) < len(hierarchy2):
            return False

        i1, i2 = 0, 0
        while i2 < len(hierarchy2):
            if i1 >= len(hierarchy1):
                return False

            if hierarchy1[i1] == hierarchy2[i2]:
                i1 += 1
                i2 += 1
            else:
                i1 += 1

        return True

    def _assert_integrity(self) -> bool:
        """Check whether all nodes in the tree are connected properly to their parents.

        Raises:
            RuntimeError: If the is a loop or a child of a node is not connected to that node via
                the 'parent' attribute.

        Returns:
            True if the integrity check has been passed successfully.
        """
        for v in self.preorder():
            for child in v.children:
                if child is v:
                    raise RuntimeError(f"loop at {v}")
                if child.parent is not v:
                    raise RuntimeError(f"Tree invalid for {v} and {child}")

        return True

    def copy(self, mapping: bool = False) -> Tree | tuple[Tree, dict[TreeNode, TreeNode]]:
        """Return a copy of the tree.

        Constructs a copy of the tree to the level of nodes, i.e., the attributes are only copied as
        references. If the node attributes are all immutable data types, the original tree is not
        affected by operations on the copy.

        Args:
            mapping: If True, additionally return the mapping from original to copied nodes as
                dictionary.

        Returns:
            A copy of the tree and optionally the mapping from original to copied nodes.
        """
        if not self.root:
            return Tree(None)

        orig_to_new = {}

        for orig in self.preorder():
            new = TreeNode()
            orig_to_new[orig] = new
            if orig.parent:
                orig_to_new[orig.parent].add_child(new)

            # shallow copy of the node attributes
            for key, value in orig.attributes():
                setattr(new, key, value)

        if mapping:
            return Tree(orig_to_new[self.root]), orig_to_new
        else:
            return Tree(orig_to_new[self.root])

    # --------------------------------------------------------------------------
    #                         TREE  <--->  NEWICK
    # --------------------------------------------------------------------------

    def to_newick(self, node: TreeNode | None = None) -> str:
        """Newick representation of the tree.

        Args:
            node: The node whose subtree shall be returned as a Newick string, the default is None,
                in which case the whole tree is returned in Newick format.

        Returns:
            A newick representation of the (sub)tree.
        """

        def _to_newick(node: TreeNode) -> str:
            node_str = str(node)

            # add colon and distance if available
            if hasattr(node, "dist"):
                node_str += f":{node.dist}"

            if not node.children:
                return node_str
            else:
                s = ""
                for child in node.children:
                    s += _to_newick(child) + ","
                return f"({s[:-1]}){node_str}"

        if node:
            return _to_newick(node) + ";"
        elif self.root:
            return _to_newick(self.root) + ";"
        else:
            return ";"

    @staticmethod
    def _parse_newick_and_return_root(newick: str) -> TreeNode:
        """Parses trees in Newick format and returns the root.

        If available (after colons in the Newick strings), the distance is stored in the 'dist'
        attribute of the nodes. Moreover, labels are converted to integers if possible.

        Args:
            newick: A tree in Newick format.

        Returns:
            The root of the parsed tree.

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the input is not a valid Newick string.
        """

        def _parse_subtree(subroot: TreeNode, subtree_string: str) -> None:
            """Recursive function to parse the subtrees.

            Args:
                subroot: The tree node to which parsed children are attached.
                subtree_string: The string containing the children to parse.

            Raises:
                ValueError: If the input is not a valid Newick string.
            """
            children = _split_children(subtree_string)

            for child in children:
                node = TreeNode()
                subroot.add_child(node)
                end = -1

                # the child has subtrees
                if child[0] == "(":
                    end = child.rfind(")")
                    if end == -1:
                        raise ValueError("invalid Newick string")
                    # recursive call
                    _parse_subtree(node, child[1:end])

                child = child[end + 1 :].strip()

                label = child

                if child.find(":") != -1:
                    label, dist = child.rsplit(":", 1)

                    try:
                        node.dist = float(dist)
                    except ValueError:
                        raise ValueError(f"invalid distance in Newick string: {dist}")

                # convert label to integer if possible
                node.label = int(label) if label.isdigit() else label

        def _split_children(child_string: str) -> list[str]:
            """Splits a given string by all ',' that are not enclosed by parentheses.

            Args:
                child_string: The string to split into the direct children.

            Returns:
                The splitted string.

            Raises:
                ValueError: If the input is not a valid Newick string.
            """
            stack = 0
            children = []
            current = ""

            for c in child_string:
                if (stack == 0) and c == ",":
                    children.append(current)
                    current = ""
                elif c == "(":
                    stack += 1
                    current += c
                elif c == ")":
                    if stack <= 0:
                        raise ValueError("invalid Newick string")
                    stack -= 1
                    current += c
                else:
                    current += c

            children.append(current.strip())
            return children

        if not isinstance(newick, str):
            raise TypeError("Newick parser needs a 'str' as input")

        end = newick.find(";")
        if end != -1:
            newick = newick[:end]

        temp_root = TreeNode()
        _parse_subtree(temp_root, newick)

        if temp_root.children:
            root = temp_root.children[0]
            # remove the parent temp_root
            root.detach()
            return root
        else:
            raise ValueError("invalid Newick string")

    @classmethod
    def parse_newick(cls, newick: str) -> Tree:
        """Parses trees in Newick format into Tree object.

        If available (after colons in the Newick strings), the distance is stored in the 'dist'
        attribute of the nodes. Moreover, labels are converted to integers if possible.

        NOTE: Do not use this function for serialization and reloading Tree objects. Use the
        'serialize()' function instead.

        Args:
            newick: A tree in Newick format.

        Returns:
            The parsed tree.

        Raises:
            TypeError: If the input is not a string.
            ValueError: If the input is not a valid Newick string.
        """
        return cls(cls._parse_newick_and_return_root(newick))

    # --------------------------------------------------------------------------
    #                         TREE  <--->  NETWORKX
    # --------------------------------------------------------------------------

    def to_nx(self) -> tuple[nx.DiGraph, int]:
        """Convert a Tree into a NetworkX DiGraph.

        The attributes correspond to the node attributes in the resulting graph. The nodes of the
        resulting graph correspond to the object ids of the TreeNode instances belonging to the
        tree.

        Returns:
            A graph representation of the tree and the object id of the root (and thus the
            corresponding node in the graph) in order to be able to completely reconstruct the tree.
        """
        self._assert_integrity()
        graph = nx.DiGraph()

        if not self.root:
            return graph, None

        for v in self.preorder():
            graph.add_node(id(v))
            for key, value in v.attributes():
                graph.nodes[id(v)][key] = value

        for u, v, sibling_nr in self.edges_sibling_order():
            if u is v:
                raise RuntimeError(f"loop at {u} and {v}")
            graph.add_edge(id(u), id(v))
            graph.nodes[id(v)]["sibling_nr"] = sibling_nr

        return graph, id(self.root)

    @classmethod
    def parse_nx(cls, graph: nx.DiGraph, root: int):
        """Convert a NetworkX DiGraph version back into a Tree.

        Args:
            graph: A tree represented as a Networkx DiGraph.
            root: The node in the graph corresponding to the root.

        Returns:
            The reconstructed tree.
        """
        number_of_leaves = 0

        if root is None:
            return cls(None)

        def _build_tree(graphnode, parent=None):
            nonlocal number_of_leaves

            treenode = TreeNode()

            if parent:
                parent.add_child(treenode)

            for key, value in graph.nodes[graphnode].items():
                setattr(treenode, key, value)

            children = sorted(
                graph.neighbors(graphnode), key=lambda item: graph.nodes[item]["sibling_nr"]
            )

            for c in children:
                _build_tree(c, parent=treenode)
            if graph.out_degree(graphnode) == 0:
                number_of_leaves += 1

            return treenode

        tree = cls(_build_tree(root))
        tree.number_of_species = number_of_leaves

        return tree

    # --------------------------------------------------------------------------
    #                           SERIALIZATION
    # --------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Convert the tree into a nested dictionary.

        Returns:
            A dict representation of the tree,

        Raises:
            RuntimeError: If the tree is empty.
        """

        def _to_dict(node: TreeNode) -> dict[str, Any]:
            node_dict = {k: v for k, v in node.attributes()}

            for i, child in enumerate(node.children):
                node_dict[f"_child{i}"] = _to_dict(child)

            return node_dict

        if self.root:
            return _to_dict(self.root)
        else:
            raise RuntimeError("cannot convert empty tree to dict")

    @classmethod
    def parse_dict(cls, tree_dict: dict[str, Any]) -> Tree:
        """Convert a dictionary representation back into a Tree.

        Args:
            tree_dict: A dictionary representation of the tree.

        Returns:
            The reconstructed tree.
        """

        def _parse_dict(node_dict: dict[str, Any]) -> TreeNode:
            """Recursive function for reconstructing the subtree represented by the dict.

            Args:
                node_dict: Dict representing a (sub)tree.

            Returns:
                The root node of reconstructed (sub)tree.
            """
            node = TreeNode()
            children = {}

            for k, v in node_dict.items():
                if k.startswith("_child"):
                    children[int(k[6:])] = _parse_dict(v)
                else:
                    setattr(node, k, v)

            for i in sorted(children):
                node.add_child(children[i])

            return node

        return cls(_parse_dict(tree_dict))

    @staticmethod
    def _infer_serialization_mode(filename: str) -> str:
        """Infer the serialization mode based on the file ending (.json or .pickle).

        Args:
            filename: The file name.

        Raises:
            ValueError: If the file extension is not supported.

        Returns:
            The inferred serialization mode.
        """
        _, file_ext = os.path.splitext(filename)

        if file_ext.lower() == ".json":
            return "json"
        elif file_ext.lower() == ".pickle":
            return "pickle"
        else:
            raise ValueError(
                "serialization format is not supplied and could not be inferred from file extension"
            )

    def serialize(self, filename: str, mode: str = None) -> None:
        """Serialize the tree using pickle or json.

        Args:
            filename: The filename (including the path) of the file to be created.
            mode: The serialization mode. Supported are pickle and json. The default is None in
                which case the mode is inferred from the file extension.

        Raises:
            If the serialization mode is unknown or could not be inferred.
        """
        if not mode:
            mode = Tree._infer_serialization_mode(filename)

        if mode == "json":
            with open(filename, "w") as f:
                json.dump(self.to_dict(), f)
        elif mode == "pickle":
            tree_nx, root_id = self.to_nx()
            with open(filename, "wb") as f:
                pickle.dump((tree_nx, root_id), f)
        else:
            raise ValueError(f"serialization mode '{mode}' not supported")

    @classmethod
    def load(cls, filename: str, mode: bool = None) -> Tree:
        """Reload a Tree from a file (pickle or json).

        Args:
            filename: The filename (including the path) of the file to be loaded.
            mode: The serialization mode. Supported are pickle and json. The default is None in
                which case the mode is inferred from the file extension.

        Returns:
            The tree reloaded from file.

        Raises:
            ValueError: If the serialization mode is unknown or could not be inferred.
        """
        if not mode:
            mode = Tree._infer_serialization_mode(filename)

        if mode == "json":
            with open(filename, "r") as f:
                tree_dict = json.load(f)
            return Tree.parse_dict(tree_dict)
        elif mode == "pickle":
            with open(filename, "rb") as f:
                return Tree.parse_nx(*pickle.load(f))
        else:
            raise ValueError(f"serialization mode '{mode}' not supported")

    # --------------------------------------------------------------------------
    #                     Print the tree to the console
    # --------------------------------------------------------------------------

    def _lines_for_print_tree(self, child_indentation: int) -> list[str]:
        """Auxiliary function for generating the lines to print the tree.

        Args:
            child_indentation: The additional indentation to be added for each level.

        Returns:
            The lines to be printed.
        """
        symbols = ("\u2500", "\u2502", "\u2514", "\u251c")

        nodes = [node for node in self.preorder()]
        node_index = {node: i for i, node in enumerate(nodes)}
        n = len(node_index)

        lines = [[] for _ in range(n)]

        for node in self.preorder():
            lines[node_index[node]].append(str(node))

            if not node.children:
                continue

            last_child = node.children.last()

            for i in range(node_index[node] + 1, node_index[last_child] + 1):
                descendant = nodes[i]
                if descendant is last_child:
                    lines[node_index[descendant]].append(
                        symbols[2] + (child_indentation - 1) * symbols[0]
                    )
                elif node is descendant.parent:
                    lines[node_index[descendant]].append(
                        symbols[3] + (child_indentation - 1) * symbols[0]
                    )
                else:
                    lines[node_index[descendant]].append(symbols[1] + (child_indentation - 1) * " ")

            for descendant in self.traverse_subtree(last_child):
                if descendant is not last_child:
                    lines[node_index[descendant]].append(child_indentation * " ")

        lines = ["".join(line) for line in lines]

        return lines

    def print_tree(self, child_indentation: int = 3) -> None:
        """Print a representation of the tree to the console.

        Args:
            child_indentation: The indentation added per level of the nodes. Must be >= 1.
        """
        if not isinstance(child_indentation, int) or child_indentation < 1:
            raise ValueError("child indentation must be an integer >= 1")

        for line in self._lines_for_print_tree(child_indentation):
            print(line)

    # --------------------------------------------------------------------------
    #                             RANDOM TREE
    # --------------------------------------------------------------------------

    @classmethod
    def random_tree(cls, number_of_leaves, binary: bool = False) -> Tree:
        """A simple function to generate a random tree.

        The tree is generated by iteratively sampling a random node to which a new leaf is attached
        (or two leaves if the sampled node was a leaf). Each step increases the number of leaves by
        one. The procedure ends as soon as the desired number of leaves is reached.
        The resulting tree is always phylogenetic, i.e., each inner node has at least two children.

        Args:
            number_of_leaves: The desired number of leaves.
            binary: If True, the resulting tree is binary, otherwise it may contain multifurcations.

        Returns:
            A randomly generated tree with the specified number of leaves.

        Raises:
            TypeError: If 'number_of_leaves' is not an integer >= 1.
        """
        if not (isinstance(number_of_leaves, int)) or number_of_leaves < 1:
            raise TypeError("N must be an 'int' > 0")

        root = TreeNode(label=0)
        tree = cls(root)
        node_list = [root]
        nr, leaf_count = 1, 1

        while leaf_count < number_of_leaves:
            node = random.choice(node_list)

            if not node.children:
                # to be phylogenetic at least two children must be added
                new_child1 = TreeNode(label=nr)
                new_child2 = TreeNode(label=nr + 1)
                node.add_child(new_child1)
                node.add_child(new_child2)
                node_list.extend(node.children)
                nr += 2
                leaf_count += 1
            elif node.children and not binary:
                # add only one child if there are already children
                new_child = TreeNode(label=nr)
                node.add_child(new_child)
                node_list.append(new_child)
                nr += 1
                leaf_count += 1

        return tree
