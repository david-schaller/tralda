# Utils

`tralda.utils` provides two modules of helper functions that are used internally and are also
available for general use.  Full signatures and docstrings are in the
[API reference](../api/utils.md).

## Graph tools — `tralda.utils.graph_tools`

```python
from tralda.utils import graph_tools
```

| Function | Purpose |
|---|---|
| `sort_edge(u, v)` | Return the edge $uv$ as a tuple with the smaller endpoint first |
| `build_adjacency_matrix(graph)` | Build a NumPy adjacency matrix and a node-to-index mapping |
| `graphs_equal(g1, g2)` | Check whether two NetworkX graphs have the same vertices and edges |
| `is_subgraph(g1, g2)` | Check whether `g1` is a subgraph of `g2` |
| `symmetric_diff(g1, g2)` | Count edges in exactly one of the two graphs |
| `contingency_table(true, pred)` | Compute TP / TN / FP / FN for edge-set comparison |
| `performance(true, pred)` | Compute accuracy, precision, and recall for edge-set comparison |
| `false_edges(true, pred)` | Return separate graphs of false-negative and false-positive edges |
| `is_properly_colored(graph)` | Check that no edge connects two vertices of the same color |
| `sort_by_colors(graph)` | Group vertices by their color attribute |
| `copy_node_attributes(src, dst)` | Copy node attributes from one graph to another |
| `random_graph(n, p)` | Generate a random Erdős–Rényi graph |
| `disturb_graph(graph, ...)` | Randomly add or remove edges from a graph |
| `independent_sets(graph)` | Return all independent sets of an undirected graph |
| `complete_multipartite_graph_from_sets(partition)` | Build a complete multipartite graph from a partition of vertices |

## Tree tools — `tralda.utils.tree_tools`

```python
from tralda.utils import tree_tools
```

| Function | Purpose |
|---|---|
| `assert_leaf_sets_equal(trees)` | Verify that all trees share the same set of leaf labels; returns that set or `None` |
