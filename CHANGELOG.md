# Changelog

## [2.0.2] - 2026-05-06

### 🐛 Bug fixes

- Bug in initialization of (double-)linked lists from multiple non-iterable arguments fixed:
  `LinkedList(1, 2, 3)` now correctly creates a list with three elements.
- Some type hints and docstrings added or corrected.

### 📚 Documentation

- Set up documentation using MkDocs, including:
  - A user guide with examples for all main features.
  - API reference generated from docstrings.
  - Pages with instructions for installation, contribution, and citing the project.
- Removed large parts of the previous documentation in `README.md`.

### 🎨 Style

- Changes to docstrings necessary for correct rendering in the new documentation format.

### 🔖 Release

- Added a workflow using GitHub Actions that automatically builds and publishes the documentation
  upon pushes to the main branch (and develop branch in subfolder `dev/`).

## [2.0.1] - 2026-04-20

### ♻️ Refactorings

- Cograph detection and editing functions now accept any graph object that implements the methods 
  `nodes()`, `has_edge()`, and `neighbors()`, as in `networkx.Graph`. This allows for more flexible
  usage of these functions without requiring a specific graph library.

### 🔖 Release

- Added a release workflow using GitHub Actions that automatically builds and publishes the
  package to PyPI on new releases.

## [2.0.0] - 2026-02-15

### 🚨 Breaking changes

- Renamed all modules using snake_case instead of camelCase.
- Updated package structure (and imports accordingly).

### 🌟 Features

- Balanced binary search trees (AVL and red-black trees) added to `tralda.datastructures.bst`
  implementing ordered sets (`TreeSet`) and dictionaries (`TreeDict`) with $O(\log n)$ insertion,
  deletion, and lookup. The `TreeSet` based on red-black trees also supports efficient split and
  join operations.
- Functions for basic tree properties: size and height.
- Function `print_tree()` of `Tree` prints a representation of the tree to the console.

### ♻️ Refactorings

- `HDTGraph` now uses `tralda.datastructures.bst` for its internal balanced binary search trees,
  replacing the previous custom implementation. This change improves code maintainability.
- Move `LCA` into a separate module `lca` in `tralda.datastructures` to better organize the
  codebase.

### 🐛 Bug fixes

- In function `edit_to_cograph`, use function instead of non-existing method call.
- Fix unclosed file handle in tree serialization.

### 🧪 Tests

- Updated benchmarking code for comparing supertree algorithms.

### 🎨 Style

- Changed to Google style for docstrings.
- Introduced typing hints for all functions and methods.
- Maximal line length is now 100 characters.

### 📦 Build

- Introduced [uv](https://docs.astral.sh/uv/) as package and project manager.
- Set up [pre-commit](https://pre-commit.com/) hooks for code formatting and linting using `ruff` and `codespell`.

## [1.1.1] - 2025-10-26

### 🐛 Bug fixes

- Added missing `__iter__` methods in iterator classes to fix compatibility with Python 3.13.

## [1.1.0] - 2023-04-27

### 🌟 Features

- Newick parser in main `Tree` class.
- `Tree` can be initialized with a Newick string (in addition to `TreeNode`).

### ♻️ Refactorings

- JSON serialization without NetworkX.

### 🔖 Release

- Changed license from GNU GENERAL PUBLIC LICENSE to MIT LICENSE.

## [1.0.1] - 2022-05-04

### 🐛 Bug fixes

- Fix that sibling order is preserved in edge contraction.
- Add missing `src/tralda/tools/__init__.py`.

## [1.0.0] - 2021-09-28

### 🌟 Features

- Linear LCA preprocessing structure added, based on the algorithm by Bender et al. (2005).
- Loose consensus tree construction added, based on the algorithm by Jansson et al. (2016).

## [0.0.2] - 2021-07-01

Initial public release of `tralda`.
