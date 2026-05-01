# Last Common Ancestors

## Background

Given a rooted tree $T$ with root $\rho$, the **last common ancestor** (LCA) of two nodes
$u$ and $v$ — also called their *lowest* common ancestor — is the unique node
$\text{lca}(u, v)$ that is an ancestor of both $u$ and $v$ and lies as deep as possible in
the tree (i.e. as far from the root as possible).

Equivalently, $\text{lca}(u, v)$ is the node at which the paths from $\rho$ to $u$ and from
$\rho$ to $v$ diverge.  In particular, $\text{lca}(u, u) = u$ and, if $u$ is an ancestor of
$v$, then $\text{lca}(u, v) = u$.

LCA queries arise naturally in many tree algorithms: checking whether a rooted triple $ab|c$
is displayed by a tree, comparing the depths of nodes, or determining whether one node is an
ancestor of another all reduce to LCA queries.


## Algorithm and complexity

The `LCA` class pre-processes the tree once and then answers individual queries in $O(1)$ time.

The algorithm works via a reduction to the **±1 Range Minimum Query (RMQ)** problem:

1. Perform an **Euler tour** of $T$ (each node is visited once per incident edge) and record the
   depth of every visit in an array $L$.
2. For each node $v$, record the index of its *first* occurrence in the Euler tour.
3. The LCA of $u$ and $v$ corresponds to the node with minimum depth in the sub-array of $L$
   between the first occurrences of $u$ and $v$.
4. The ±1 property of consecutive depths in an Euler tour allows a **sparse table** to be built
   in $O(n)$ time that answers range-minimum queries in $O(1)$.

Overall:

| Phase | Complexity |
|---|---|
| Preprocessing | $O(n)$ |
| Single LCA query | $O(1)$ |

where $n$ is the number of nodes in the tree.

!!! note "References"
    The implementation follows the algorithm described in:

    M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena, P. Sumazin.
    *Lowest common ancestors in trees and directed acyclic graphs.*
    Journal of Algorithms, 57(2):75–94, 2005.
    [DOI: 10.1016/j.jalgor.2005.08.001](https://doi.org/10.1016/j.jalgor.2005.08.001)


## The `LCA` class

`LCA` is provided by `tralda.datastructures`:

```python
from tralda.datastructures import LCA
```

Construct an instance by passing a `Tree` object.  The preprocessing runs immediately during
construction:

```python
from tralda.datastructures import Tree, LCA

T = Tree.parse_newick("((a,b),(c,(d,e)));")
lca = LCA(T)
```


## Querying the LCA

The instance is callable, and also exposes a `get()` method — both are equivalent:

```python
u = lca("a", "b")   # lca of leaves "a" and "b" → inner node above them
v = lca.get("c", "e")

# nodes can be passed by label (str/int) or as TreeNode objects directly
root = T.root
w = lca(root, "d")  # lca of the root with any other node is always the root
```

Nodes can be identified either by their `label` attribute or by passing the `TreeNode` object
directly.  When using labels, every node that should be queryable must have its `label` attribute
set, and all labels in the tree must be unique — duplicate labels will cause undefined behavior.


## Ancestor and descendant tests

Because $u$ is an ancestor of $v$ if and only if $\text{lca}(u, v) = u$, the `LCA` instance
exposes several convenience predicates:

```python
lca.ancestor_or_equal(u, v)      # True if u is an ancestor of v (or u == v)
lca.ancestor_not_equal(u, v)     # True if u is a strict ancestor of v
lca.descendant_or_equal(u, v)    # True if u is a descendant of v (or u == v)
lca.descendant_not_equal(u, v)   # True if u is a strict descendant of v
lca.are_comparable(u, v)         # True if u is an ancestor or descendant of v
```


## Rooted triple queries

The LCA structure makes triple queries efficient.  The tree displays the rooted triple
$ab|c$ if and only if $\text{lca}(a, b)$ is a proper descendant of $\text{lca}(a, c)$
(equivalently of $\text{lca}(b, c)$):

```python
lca.displays_triple("a", "b", "c")   # True if the tree displays ab|c
```

To filter a collection of triples to those displayed by the tree:

```python
triples = [("a", "b", "c"), ("a", "c", "d"), ("d", "e", "a")]

consistent = lca.consistent_triples(triples)           # list
# or lazily:
for t in lca.consistent_triple_generator(triples):
    print(t)
```


## Edges and nodes as arguments

Methods that accept node arguments also accept **edges** (as a two-element tuple or list
`[parent, child]`), treating the edge as equivalent to its child node:

```python
edge = ("c", "d")   # labels, or pass actual TreeNode objects
lca.are_comparable(edge, "a")
```


## Important: validity after tree modification

!!! warning "The `LCA` structure is not updated automatically"
    The internal data structures of an `LCA` instance are built once from the tree at
    construction time.  **If the tree is modified afterwards — nodes added or removed,
    edges rewired, labels changed — the `LCA` instance is no longer guaranteed to return
    correct results.**

    Always construct a fresh `LCA` instance after modifying the tree:

    ```python
    T = Tree.parse_newick("((a,b),(c,d));")
    lca = LCA(T)

    # ... modify T, e.g. by adding a child ...
    new_leaf = TreeNode(label="e")
    T.root.children.first.data.add_child(new_leaf)

    # lca is now stale — rebuild it:
    lca = LCA(T)
    ```
