# tralda

[![license: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

A Python library for **tr**ee **al**gorithms and **da**ta structures.

## Installation

The package requires Python 3.5 or higher.

#### Easy installation with pip

The `tralda` package is available on PyPI:

    pip install tralda

For details about how to install Python packages see [here](https://packaging.python.org/tutorials/installing-packages/).
    
#### Dependencies

The package has several dependencies (which are installed automatically when using `pip`):
* [NetworkX](https://networkx.github.io/)
* [Numpy](https://numpy.org)

## Usage and description

### Tree data structures

The following classes can be imported from the subpackage `tralda.datastructures`.
The class `Tree` implements the basic tree data structure which are essential for most of the modules in the package.
It provides methods for tree traversals and manipulation, output in Newick format, as well as the efficient computation of last common ancestors (class `LCA` which is initialized with an instance of type `Tree`).

The classes `TreeSet` and `TreeDict` implement data structures (AVL trees) for sorted sets and dictionaries, respectively.

### Supertree computation

The subpackage `tralda.supertree` implements a number of algorithms for the computation of supertrees:
* BUILD (Aho et al. 1981), class `Build` or function `BUILD_supertree`
* BuildST (Deng & Fernández-Baca 2016), class `BuildST` or function `build_st`
* LinCR (Schaller et al. 2021), class `LinCR` or function `linear_common_refinement`

The LinCR algorithm computes a supertree for a sequence of trees on the same set of leaves, i.e., a common refinement.

### Cograph editing

The subpackage `tralda.cograph` contains an efficient algorithm for cograph recognition and heuristics for cograph editing:
* class `Cotree` inherits from `Tree`
* function `linear_cograph_detection` recognizes cographs and returns a `Cotree` representation in the positive case (Corneil et al. 1985)
* function `edit_to_cograph` edits an arbitrary graph to a cograph (algorithm from Crespelle 2019) and returns a `Cotree` representation

### Other data structures

The following auxiliary data structures can be imported from the subpackage `tralda.datastructures`:
* linked list: class `LinkedList`
* doubly-linked list: class `DoublyLinkedList`
* HDT dynamic graph data structure (Holm, de Lichtenberg & Thorup in 2001): class `HDTGraph`

## Citation and references

If you use `tralda` in your project or code from it, please consider citing:

* **Schaller, D., Hellmuth, M., Stadler, P.F. (2021) A Linear-Time Algorithm for the Common Refinement of Rooted Phylogenetic Trees on a Common Leaf Set.**

Additional references to algorithms that were implemented are given in the source code.

Please report any bugs and questions in the [Issues](https://github.com/david-schaller/tralda/issues) section.
Also, feel free to make suggestions for improvement and/or new functionalities.