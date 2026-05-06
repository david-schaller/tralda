# User Guide

This guide walks through the main features of tralda step by step.

## Overview

`tralda` provides a collection of efficient algorithms and data structures centered around rooted
trees, with a focus on phylogenetics and combinatorial algorithms. It is designed to serve as a
building block for research software in computational biology and related fields.

**[Trees](trees.md)** — The `Tree` and `TreeNode` classes in `tralda.datastructures` offer a
flexible rooted-tree representation with support for traversals (preorder, postorder, level-order),
subtree operations, tree generation, and utilities such as topology comparison and hierarchy
extraction.

**[Last Common Ancestors](lca.md)** — An $O(n)$-time/space preprocessing structure for
$O(1)$ LCA queries, based on the algorithm by Bender et al. (2005).
Available as `LCA` in `tralda.datastructures`.

**[Supertrees and Common Refinements](supertrees.md)** — The `tralda.supertree` subpackage
implements several algorithms for constructing supertrees and consensus trees from a set of
(partial) input trees:

- **BUILD** (Aho et al. 1981) — classic triple-based supertree construction.
- **BuildST** (Deng & Fernández-Baca 2016) — fast compatibility testing and supertree construction
  for rooted phylogenetic trees.
- **LinCR** (Schaller et al. 2021) — linear-time algorithm for the minimal common
  refinement of rooted phylogenetic trees on a common leaf set.
- **Loose consensus tree** (Jansson et al. 2016) — linear-time construction of the loose
  consensus tree for trees with the same leaf set.

**[Cographs](cograph.md)** — `tralda.cograph` offers linear-time cograph recognition
(Corneil et al. 1985) with cotree construction, as well as a heuristic for cograph editing
(Crespelle 2021) that modifies a graph with a near-minimum number of edge insertions/deletions to
make it a cograph.

**[Dynamic Graph Connectivity](dynamic_connectivity.md)** — The `HDTGraph` class in
`tralda.datastructures.hdtgraph` implements the poly-logarithmic dynamic graph structure described
by Holm et al. (2001), supporting edge insertions and deletions with $O(\log^2 n)$ amortized cost
while answering connectivity queries in $O(\log n)$.

**[Balanced Binary Search Trees](bst.md)** — `tralda.datastructures.bst` provides AVL trees and
red-black trees (ordered sets and dictionaries) with $O(\log n)$ insertion, deletion, and lookup,
as well as efficient split and join operations.

**[Linked Lists](linked_lists.md)** — Singly and doubly linked list implementations, with the
doubly linked variant (`DLList`) supporting $O(1)$ node removal given a direct node reference.

**[Dynamic Partition](dynamic_partition.md)** — A dynamic partition data structure that
supports efficient set merges using a small-into-large (weighted-union) strategy, giving
$O(n \log n)$ total cost over all merge operations.

**[Utils](utils.md)** — Helper functions for working with trees and graphs, collected in
`tralda.utils`.
