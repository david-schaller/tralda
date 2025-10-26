# tralda

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi version](https://img.shields.io/badge/pypi-v1.1.1-blue.svg)](https://pypi.org/project/tralda/)

A Python library for **tr**ee **al**gorithms and **da**ta structures.

## Installation

The package requires Python 3.7 or higher.

#### Easy installation with pip

The `tralda` package is available on [PyPI](https://pypi.org/project/tralda/):

    pip install tralda

For details about how to install Python packages see [here](https://packaging.python.org/tutorials/installing-packages/).

#### Installation with the setup file

Alternatively, you can download or clone the repo, go to the root folder of package and install it using the command:

    python setup.py install

#### Dependencies

The package has several dependencies (which are installed automatically when using `pip`):
* [NetworkX](https://networkx.github.io/)
* [Numpy](https://numpy.org)

## Usage and description

### Tree data structure

The class `Tree` implements the tree data structure which is essential for most of the modules in the package and can be imported from the subpackage `tralda.datastructures`.
It provides methods for tree traversals and manipulation, output in Newick format, as well as the linear-time computation of last common ancestors by Bender & Farach-Colton (class `LCA` which is initialized with an instance of type `Tree`).
`Tree` instances can be serialized in pickle or json format.

<details>
<summary>Overview of the functions of the class TreeNode: (Click to expand)</summary>

| Function | Description |
| --- | --- |
| `attributes()` | generator for the node attributes |
| `add_child(child_node)` | add `child_node` as a child |
| `add_child_right_of(child_node, right_of)` | add `child_node` as a child to the right of `right_of` |
| `remove_child(child_node)` | remove the child `child_node` |
| `detach()` | remove the node from its parent's children |
| `is_leaf()` | check if the node is a leaf |
| `child_subsequence(left_node, right_node)` | list of children between `left_node` and `right_node`

</details>

<details>

<summary>Overview of the functions of the class Tree: (Click to expand)</summary>

| Function | Description |
| --- | --- |
| `leaves()` | generator for the leaf nodes |
| `preorder()` | generator for preorder (=top-down) traversal |
| `postorder()` | generator for postorder (=bottom-up) traversal |
| `inner_vertices()` | generator for inner nodes |
| `edges()` | generator for the edges of the tree |
| `euler_generator()` | generator for an Euler tour |
| `leaf_dict()` | compute the `list` of leaf nodes in the subtree of each node, and return them as a `dict` |
| `contract(edges)` | contract all edges in the collection `edges` |
| `get_triples()` | return a list of all triples that are displayed by the tree |
| `delete_and_reconnect(node)` | delete `node` and reconnect its children to its parent |
| `copy()` | construct a copy of the tree (node attributes are only copied as references, but mutable data types should be avoided as node attribute values) |
| `to_newick()` | return a `str` representation of the tree in Newick format |
| `random_tree(N, binary=False)` | return a random tree with `N` leaves that is optionally forced to be binary; new children are stepwise attached to randomly selected nodes until `N` are reached |
| `to_nx()` | return a NetworkX `DiGraph` version of the tree (with the ids of the `TreeNode` instances as nodes) and its `root` (also represented by the id) |
| `parse_nx(G, root)` | convert a tree encoded as a NetworkX `DiGraph` (together with the `root`) back into a `Tree` |
| `serialize(filename, mode=None)` | serialize a tree in JSON or pickle format specified by `mode`; default is `None`, in which case the mode is inferred from the filename ending |
| `load(filename, mode=None)` | load a tree from file in JSON or pickle format specified by `mode`; default is `None`, in which case the mode is inferred from the filename ending |
| `is_binary()` | check if the tree is binary |
| `is_phylogenetic()` | check if the tree is phylogenetic (all inner nodes have at least one child) |
| `equal_topology(other)` | check whether this tree and `other` have the same topology based on the leaves' `label` attributes |
| `is_refinement` | check whether this tree refines `other` based on the leaves' `label` attributes |

</details>

<details>

<summary>Overview of the functions of the class LCA: (Click to expand)</summary>

| Function | Description |
| --- | --- |
| `get(a, b)` | get the last common ancestor of nodes a and b |
| `displays_triple(a, b, c)` | check whether the triple ab|c is displayed |
| `are_comparable(u, v)` | check whether `u` and `v` are comparable in terms of the ancestor relation |
| `ancestor_or_equal(u, v)` | check whether `u` is equal to or an ancestor of `v` |
| `ancestor_not_equal(u, v)` | check whether `u` is a strict ancestor of `v` |
| `descendant_or_equal(u, v)` | check whether `u` is equal to or a descendant of `v` |
| `descendant_not_equal(u, v)` | check whether `u` is a strict descendant of `v` |
| `consistent_triples(triples)` | `list` with the subset of `triples` that are displayed by the tree |
| `consistent_triple_generator` | generator for the items in `triples` that are displayed |

</details>

<details>
<summary>Example usage: (Click to expand)</summary>

    from tralda.datastructures import Tree, LCA

    # construct a random tree with 20 leaves
    tree = Tree.random_tree(20)

    # serialization, reload via Tree.load('path/to/file.json')
    tree.serialize('path/to/file.json')

    # linear-time processing of the tree for constant-time
    # last common ancestor queries
    lca_T = LCA(tree)

    # l.c.a. queries via 'TreeNode' instances or labels (if the nodes
    # in the tree have the label attribute set)
    print( lca_T(4, 7) )

    # triple queries (e.g. is 3 5 | 2 displayed?)
    print( lca_T.displays_triple(3, 5, 2) )

</details>

### Supertree computation

The subpackage `tralda.supertree` implements a number of algorithms for the computation of supertrees:
* BUILD (Aho et al. 1981), class `Build` or function `BUILD_supertree`
* BuildST (Deng & Fernández-Baca 2016), class `BuildST` or function `build_st`
* Loose_Cons_Tree (Jansson et al. 2016), class `LooseConsensusTree` or function `loose_consensus_tree`
* LinCR (Schaller et al. 2021), class `LinCR` or function `linear_common_refinement`

The latter two algorithms compute the loose consensus tree and the common refinement, resp., for a sequence of trees on the same set of leaves in linear time.

### Cographs and cotrees

The subpackage `tralda.cograph` contains an efficient algorithm for cograph recognition and heuristics for cograph editing:
* function `to_cotree` recognizes cographs and returns a `Tree` representation in the positive case (Corneil et al. 1985)
* function `edit_to_cograph` edits an arbitrary graph to a cograph (algorithm from Crespelle 2019) and returns a `Tree` representation

### Other data structures

The following auxiliary data structures can be imported from the subpackage `tralda.datastructures`:
* linked list: class `LinkedList`
* doubly-linked list: class `DoublyLinkedList`
* HDT dynamic graph data structure (Holm, de Lichtenberg & Thorup in 2001): class `HDTGraph`
* AVL trees: classes `TreeSet` and `TreeDict` implement data structures for sorted sets and dictionaries, respectively

## Citation and references

If you use `tralda` in your project or code from it, please consider citing:

* **Schaller, D., Hellmuth, M., Stadler, P.F. (2021) A Simple Linear-Time Algorithm for the Common Refinement of Rooted Phylogenetic Trees on a Common Leaf Set.**

Additional references to algorithms that were implemented are given in the source code.

Please report any bugs and questions in the [Issues](https://github.com/david-schaller/tralda/issues) section.
Also, feel free to make suggestions for improvement and/or new functionalities.
