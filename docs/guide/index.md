# User Guide

This guide walks through the main features of tralda step by step.

## Overview

`tralda` provides a collection of efficient algorithms and data structures centered around rooted
trees, with a focus on phylogenetics and combinatorial algorithms. It is designed to serve as a
building block for research software in computational biology and related fields.

**Tree data structure** — The `Tree` and `TreeNode` classes in `tralda.datastructures` offer a
flexible rooted-tree representation with support for traversals (preorder, postorder, level-order),
subtree operations, tree generation, and utilities such as topology comparison and hierarchy
extraction.

**Lowest common ancestor (LCA)** — An $O(n)$-time/space preprocessing structure for
$O(1)$ LCA queries, based on the algorithm by Bender et al. (2005).
Available as `LCA` in `tralda.datastructures`.

**Supertree computation** — The `tralda.supertree` subpackage implements several algorithms for
constructing supertrees and consensus trees from a set of (partial) input trees:
- **BUILD** (Aho et al. 1981) — classic triple-based supertree construction.
- **BuildST** (Deng & Fernández-Baca 2016) — fast compatibility testing and supertree construction
  for rooted phylogenetic trees.
- **LinCR** (Schaller et al. 2021) — linear-time algorithm for the minimal common
  refinement of rooted phylogenetic trees on a common leaf set.
- **Loose consensus tree** (Jansson et al. 2016) — linear-time construction of the loose
  consensus tree for trees with the same leaf set.

**Balanced binary search trees** — `tralda.datastructures.bst` provides AVL trees and red-black
trees (ordered sets and dictionaries) with $O(\log n)$ insertion, deletion, and lookup, as well as
efficient split and join operations.

**Dynamic graph connectivity** — The `HDTGraph` class in `tralda.datastructures.hdtgraph`
implements the poly-logarithmic dynamic graph structure described by Holm et al. (2001), supporting
edge insertions and deletions with $O(\log^2 n)$ amortized cost while answering connectivity
queries in $O(\log n / \log \log n)$.

**Cograph detection and editing** — `tralda.cograph` offers linear-time cograph recognition
(Corneil et al. 1985) with cotree construction, as well as a heuristic for cograph editing
(Crespelle 2021) that modifies a graph with a near-minimum number of edge insertions/deletions to
make it a cograph.
