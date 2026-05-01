# tralda

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![pypi version](https://img.shields.io/pypi/v/tralda)](https://pypi.org/project/tralda/)
[![Python](https://img.shields.io/pypi/pyversions/tralda)](https://pypi.org/project/tralda/)
[![PyPI downloads](https://img.shields.io/pypi/dm/tralda)](https://pypi.org/project/tralda/)
[![DOI](https://img.shields.io/badge/DOI-10.1186/s13015--021--00202--8-blue)](https://doi.org/10.1186/s13015-021-00202-8)

A Python library for **tr**ee **al**gorithms and **da**ta structures.

## Overview

`tralda` provides a collection of efficient algorithms and data structures centered around rooted
trees, with a focus on phylogenetics and combinatorial algorithms. It is designed to serve as a
building block for research software in computational biology and related fields.

**Tree data structure** — The `Tree` and `TreeNode` classes in `tralda.datastructures` offer a
flexible rooted-tree representation with support for traversals (preorder, postorder, level-order),
subtree operations, tree generation, and utilities such as topology comparison and hierarchy
extraction.

**Lowest common ancestor (LCA)** — An $\mathcal{O}(n)$-time/space preprocessing structure for
$\mathcal{O}(1)$ LCA queries, based on the algorithm by Bender et al. (2005).
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
trees (ordered sets and dictionaries) with $\mathcal{O}(\log n)$ insertion, deletion, and lookup,
as well as efficient split and join operations.

**Dynamic graph connectivity** — The `HDTGraph` class in `tralda.datastructures.hdtgraph`
implements the poly-logarithmic dynamic graph structure described by Holm et al. (2001), supporting
edge insertions and deletions with $\mathcal{O}(\log^2 n)$ amortized cost while answering
connectivity queries in $\mathcal{O}(\log n / \log \log n)$.

**Cograph detection and editing** — `tralda.cograph` offers linear-time cograph recognition
(Corneil et al. 1985) with cotree construction, as well as a heuristic for cograph editing
(Crespelle 2021) that modifies a graph with a near-minimum number of edge insertions/deletions to
make it a cograph.

## Installation and usage

The package requires Python 3.10 or higher.
The `tralda` package is available on [PyPI](https://pypi.org/project/tralda/):

```bash
pip install tralda
```

See also the page [Installation](https://david-schaller.github.io/tralda/installation/) in the
documentation.

The [tralda documentation](https://david-schaller.github.io/tralda/) contains a user manual
with example code as well as the
[API reference](https://david-schaller.github.io/tralda/api/trees/).

If you want to contribute to the project, please see the
[contributing guidelines](https://david-schaller.github.io/tralda/contributing/).
Please report any bugs and questions in the
[Issues](https://github.com/david-schaller/tralda/issues) section.

## Citation and references

If you use `tralda` in your project or code from it, please consider citing:

> Schaller, D., Hellmuth, M., Stadler, P.F. (2021)
> A simpler linear-time algorithm for the common refinement of rooted phylogenetic trees on a common
> leaf set. *Algorithms for Molecular Biology* 16, 23 (2021).
> https://doi.org/10.1186/s13015-021-00202-8

Additional references to algorithms that were implemented are given in the source code.
