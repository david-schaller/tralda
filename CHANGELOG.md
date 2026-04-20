# Change Log

## 2.0.1

* Cograph detection and editing functions now accept any graph object that implements the methods 
  `nodes()`, `has_edge()`, and `neighbors()`, as in `networkx.Graph`. This allows for more flexible
  usage of these functions without requiring a specific graph library.

## 2.0.0

### Added

* functions for basic tree properties: size and height
* Newick parser in main `Tree` class
* `Tree` can be initialized with a Newick string (in addition to `TreeNode`)
* Fix that sibling order is preserved in edge contraction
* function `print_tree()` of `Tree` prints a representation of the tree to the console

### Changed

* JSON serialization without NetworkX
* Change from GNU GENERAL PUBLIC LICENSE to MIT LICENSE

### Removed

## 1.0.1

Released on May 4, 2022.
