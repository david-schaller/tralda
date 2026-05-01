# Trees

## Background

A **rooted tree** $T$ consists of a set of nodes $V$ with a distinguished root $\rho \in V$, where
every non-root node $v$ has a unique parent $\text{par}(v)$. Nodes without children are called
**leaves**; all other nodes are **inner nodes**. The **subtree** rooted at $v$, written $T(v)$,
is the tree induced by $v$ and all its descendants.

A rooted tree is called **phylogenetic** if every inner node has at least two children.  It is
**binary** if every inner node has *exactly* two children.  In phylogenetics, the leaves carry
species labels, and the tree represents the evolutionary relationships among those species.

A key concept is the **cluster** (or **clade**) of a node $v$: the set of labels of all leaves in
$T(v)$.  The family of all clusters forms a **hierarchy** — a laminar family on the leaf label set.
Two trees are **topologically equal** if and only if their hierarchies are identical; $T$ is a
**refinement** of $T'$ if every cluster of $T'$ also appears in $T$.

A **rooted triple** $ab|c$ is displayed by $T$ when the last common ancestor of leaves $a$ and $b$
is a proper descendant of the last common ancestor of $a$ (or $b$) and $c$.


## The `Tree` and `TreeNode` classes

The main tree data structure is provided by `tralda.datastructures`:

```python
from tralda.datastructures import Tree, TreeNode
```

A `TreeNode` can hold arbitrary attributes set as keyword arguments or via direct attribute
assignment.  The special attribute `label` is used by many tree operations that need to identify
leaves.

```python
# Build a small tree manually
root = TreeNode(label="root")
a    = TreeNode(label="a")
b    = TreeNode(label="b")
c    = TreeNode(label="c")

root.add_child(a)
root.add_child(b)
b.add_child(c)

T = Tree(root)
```

A `Tree` is constructed from its root node.  An empty tree is created with `Tree(None)`.


## Traversals

The standard traversals are available as generators, so they work efficiently even for very large
trees:

```python
for v in T.preorder():    # root before children
    print(v)

for v in T.postorder():   # children before parent
    print(v)

for v in T.leaves():
    print(v.label)

for v in T.inner_nodes():
    print(v)
```

The method `preorder_and_level()` additionally yields the depth of each node (distance from the
root):

```python
for v, depth in T.preorder_and_level():
    print(f"{'  ' * depth}{v}")
```

The method `euler_generator()` produces an Euler tour — each node appears once per incident
edge — which is the basis for the efficient LCA structure described in the [LCA guide](lca.md).


## Newick format

Trees can be parsed from and serialized to the standard
[Newick format](https://en.wikipedia.org/wiki/Newick_format):

```python
T = Tree.parse_newick("((a,b),(c,(d,e)));")
print(T.to_newick())  # ((a,b),(c,(d,e)));
```

Branch lengths after colons are stored in the `dist` attribute of each node:

```python
T = Tree.parse_newick("((a:0.1,b:0.2):0.5,c:0.3);")
for v, _ in T.preorder_and_level():
    if hasattr(v, "dist"):
        print(v.label, v.dist)
```


## Modifying a tree

Child nodes are managed through `add_child()` and `remove_child()` on `TreeNode`.
The `detach()` method disconnects a node from its parent without removing it from memory.

To *suppress* a node (remove it and reconnect its children to its parent, preserving sibling
order), use `Tree.delete_and_reconnect()`:

```python
# contract a list of edges (suppress the child node of each edge)
T.contract([(parent, child) for parent, child in T.inner_edges()])
```

The method `contract()` accepts a list of `(parent, child)` edges and processes them in one pass.
Pass `inplace=False` to leave the original tree untouched and receive a modified copy instead.


## Hierarchies, topology comparison, and triples

```python
# The set of clusters as frozensets of leaf labels
hierarchy = T.get_hierarchy()

# Check topological equality (based on leaf labels)
T1.equal_topology(T2)          # True / False

# Check whether T1 is a refinement of T2
T1.is_refinement(T2)           # True / False

# All rooted triples displayed by the tree
triples = T.get_triples(label_only=True)  # list of (a, b, c) meaning ab|c
```


## Random trees

`Tree.random_tree()` generates a random phylogenetic tree with a given number of leaves.  Set
`binary=True` to restrict every inner node to exactly two children:

```python
T = Tree.random_tree(20)           # random phylogenetic tree, 20 leaves
T_bin = Tree.random_tree(20, binary=True)  # random binary tree, 20 leaves
```

This is useful for quickly testing algorithms or benchmarking.


## Serialization

Trees can be saved and reloaded using JSON or pickle.  The recommended approach is `serialize()`
and `load()`, which handle the format automatically based on the file extension:

```python
T.serialize("my_tree.json")
T2 = Tree.load("my_tree.json")

T.serialize("my_tree.pickle")
T3 = Tree.load("my_tree.pickle")
```

For interoperability with NetworkX, `to_nx()` converts a tree to a `DiGraph` and `parse_nx()`
reconstructs it:

```python
import networkx as nx

graph, root_id = T.to_nx()
T_reconstructed = Tree.parse_nx(graph, root_id)
```


## Printing

`print_tree()` renders a compact ASCII representation of the tree to the console, which is handy
for quick inspection:

```python
T = Tree.parse_newick("((a,b),(c,(d,e)));")
T.print_tree()
```

```
├──
│  ├──a
│  └──b
└──
   ├──c
   └──
      ├──d
      └──e
```
