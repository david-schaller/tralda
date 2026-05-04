# Dynamic Graph Connectivity

## Background

### The fully dynamic connectivity problem

A **fully dynamic graph** supports the interleaving of three operations on an undirected
graph $G = (V, E)$:

- **insert** an edge,
- **delete** an edge, and
- **query** whether two vertices are connected.

The challenge lies in keeping connectivity information up to date efficiently as edges are
inserted and deleted.  The naive approach — recomputing a spanning forest from scratch after
every update — costs $O(m)$ per operation.  The goal is a *poly-logarithmic* bound.

### The HDT algorithm

Holm, de Lichtenberg, and Thorup (2001) gave the first *deterministic* algorithm achieving
this goal.  Their structure (referred to here as the **HDT datastructure**) maintains a
spanning forest $F$ of $G$ and assigns every edge a *level* $\ell(e) \in \{0, \dots,
\lfloor\log_2 n\rfloor\}$.  Two invariants are preserved at all times:

1. $F$ is a maximum spanning forest of $G$ with respect to edge levels.
2. Every tree in the spanning forest restricted to edges of level $\geq i$ contains at most
   $\lfloor n / 2^i \rfloor$ vertices.

When a tree edge $(u, v)$ is deleted, the algorithm searches for a *replacement edge* by
scanning non-tree edges incident to the smaller of the two resulting components, promoting
any non-replacement edge to the next level.  Because levels can only increase (and are
bounded by $\lfloor \log_2 n \rfloor$), the total work can be charged via an amortisation
argument.

The resulting complexity is:

| Operation | Amortised cost |
|---|---|
| `insert_edge` | $O(\log^2 n)$ |
| `delete_edge` | $O(\log^2 n)$ |
| `connected` | $O(\log n)$ |

where $n = |V|$.  Unlike earlier algorithms, this bound is *independent* of the degrees of
the vertices.

!!! note "References"
    Jacob Holm, Kristian de Lichtenberg, and Mikkel Thorup. Poly-logarithmic deterministic
    fully-dynamic algorithms for connectivity, minimum spanning tree, 2-edge, and biconnectivity.
    *J. ACM*, 48(4):723–760, July 2001.
    [DOI: 10.1145/502090.502095](https://doi.org/10.1145/502090.502095)

### Euler Tour (ET) trees

The spanning forest $F$ is maintained as a collection of **Euler Tour trees**.  An Euler
Tour tree represents a rooted tree $T$ by the sequence of nodes visited in a DFS Euler
tour of $T$ (each node appears once per incident edge, plus once as the starting point).
This sequence is stored as the in-order traversal of a **balanced BST**, so that the tour
can be split and joined in $O(\log n)$ time.  These two operations correspond to cutting
and linking edges of $T$.

In tralda, the BST underlying each ET tree is the
[red-black tree](bst.md) implementation provided by
`tralda.datastructures.bst.red_black`.  Each BST node additionally stores the count of
*active occurrences* (= unique vertices) in its subtree, enabling $O(\log n)$ size queries
for any component.

!!! note "References"
    Monika R. Henzinger and Valerie King. Randomized fully dynamic graph algorithms with
    polylogarithmic time per operation. *J. ACM*, 46(4):502–536, July 1999.
    [DOI: 10.1145/320211.320215](https://doi.org/10.1145/320211.320215)


### Usage in BuildST

The `BuildST` / `build_st` algorithm (Deng & Fernández-Baca 2016) for fast supertree
compatibility testing maintains the **display graph** of the input profile as an
`HDTGraph`.  During the recursive decomposition of the algorithm, internal nodes of the
input trees are successively replaced by their children (edge deletions in the display
graph), and the connected components of the resulting graph determine how the supertree
is assembled.  The $O(M_\mathcal{P} \log^2 M_\mathcal{P})$ overall running time of
BuildST follows directly from the HDT bounds applied to a graph of size
$O(M_\mathcal{P})$.

See the [Supertrees](supertrees.md) guide for full usage details of `build_st`.

!!! note "References"
    Yun Deng and David Fernández-Baca. Fast Compatibility Testing for Rooted Phylogenetic Trees.
    *27th Annual Symposium on Combinatorial Pattern Matching (CPM 2016)*.
    [DOI: 10.4230/LIPIcs.CPM.2016.12](https://doi.org/10.4230/LIPIcs.CPM.2016.12)


## Module overview

The `HDTGraph` class is exported from `tralda.datastructures`:

```python
from tralda.datastructures import HDTGraph
```

The underlying ET tree and node classes can be imported directly if needed:

```python
from tralda.datastructures.hdtgraph.et_tree import ETTree, ETTreeNode
```


## Creating a graph and inserting edges

`HDTGraph` is initialised as an empty graph.  Nodes are created implicitly when an edge is
inserted; they can also be added without any edges via `insert_node`.

```python
from tralda.datastructures import HDTGraph

G = HDTGraph()

# Insert edges — nodes are created automatically
G.insert_edge(1, 2)
G.insert_edge(2, 3)
G.insert_edge(3, 4)
G.insert_edge(1, 4)  # creates a cycle

# Insert a node without any edges
G.insert_node(5)

print(G.has_node(3))          # True
print(G.has_edge(2, 3))       # True
print(G.has_edge(1, 3))       # False
```

!!! note
    Parallel edges are not supported.  Inserting an edge that already exists is a no-op.


## Connectivity queries

The central operation is `connected(u, v)`, which runs in $O(\log n)$ time:

```python
print(G.connected(1, 4))   # True  — connected via 1-2-3-4 or directly 1-4
print(G.connected(1, 5))   # False — node 5 is isolated
```

`is_connected()` checks whether the *entire* graph is connected (a single component):

```python
print(G.is_connected())    # False — node 5 is isolated
G.insert_edge(4, 5)
print(G.is_connected())    # True
```


## Deleting edges

Edge deletion is handled automatically.  If the deleted edge is a *non-tree* edge, it is
simply removed.  If it is a *tree* edge, the algorithm searches for a replacement edge to
reconnect the spanning forest (if one exists) and promotes any non-replacement edges encountered
along the way.

```python
G = HDTGraph()
for u, v in [(1, 2), (2, 3), (3, 4), (4, 5)]:
    G.insert_edge(u, v)

print(G.connected(1, 5))   # True

G.delete_edge(2, 3)

print(G.connected(1, 5))   # False — graph is now split
print(G.connected(1, 2))   # True
print(G.connected(3, 5))   # True
```

!!! note
    `delete_edge` is silent if the edge does not exist.


## Iterating over nodes, edges, and components

```python
G = HDTGraph()
for u, v in [(1, 2), (2, 3), (5, 6)]:
    G.insert_edge(u, v)

# Iterate over all nodes
print(list(G.get_nodes()))   # [1, 2, 3, 5, 6] (order not guaranteed)

# Iterate over all edges
print(list(G.get_edges()))

# Iterate over the nodes of the connected component containing node 1
print(list(G.component_iterator(1)))   # [1, 2, 3]
print(list(G.component_iterator(5)))   # [5, 6]
```


## Adding a tree efficiently

When a tree (a `tralda.datastructures.Tree` instance) is available and its nodes are not
yet part of the graph, `add_loose_tree` constructs the corresponding ET tree directly from
the Euler tour in $O(n \log n)$ time rather than inserting edges one by one:

```python
from tralda.datastructures import HDTGraph, Tree

T = Tree.parse_newick("((a,b),(c,d));")

G = HDTGraph()
G.add_loose_tree(T)   # adds all nodes and edges of T at once

print(G.connected(T.root, T.root.children[0]))   # True
```

`add_loose_tree` requires that none of the tree's nodes are already present in the graph.
