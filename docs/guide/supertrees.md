# Supertrees and Common Refinements

## Background

### Rooted triples and consistency

A **rooted triple** $ab|c$ is a binary phylogenetic tree on three leaves $a$, $b$, $c$ in which
$a$ and $b$ are more closely related to each other than either is to $c$.  Formally, a tree $T$
**displays** $ab|c$ if

$$
\text{lca}_T(a,b) \prec_T \text{lca}_T(a,c) = \text{lca}_T(b,c).
$$

A set of triples $\mathcal{R}$ is **consistent** if there exists a tree that displays every triple
in $\mathcal{R}$.  Given $\mathcal{R}$ and a leaf set $L$, the **BUILD algorithm** (Aho et al.
1981) decides consistency and, if consistent, constructs the unique **least-resolved** tree
$\texttt{BUILD}(\mathcal{R}, L)$ displaying $\mathcal{R}$.


### Clusters, hierarchies, and supertrees

The **cluster** of a node $v$ in a rooted tree $T$ is the set $\mathcal{L}(T(v))$ of leaf labels
in the subtree rooted at $v$.  The family $\mathcal{H}(T)$ of all clusters forms a **hierarchy**
— a laminar family — on the leaf set $\mathcal{L}(T)$.

Given a collection of rooted phylogenetic trees $T_1, \dots, T_k$ with (potentially different)
leaf sets, a **supertree** is a tree $T$ with leaf set
$L = \bigcup_{i=1}^k \mathcal{L}(T_i)$ that **displays** every input tree $T_i$, meaning
$\mathcal{H}(T_i) \subseteq \{\, C \cap \mathcal{L}(T_i) \mid C \in \mathcal{H}(T),\,
C \cap \mathcal{L}(T_i) \ne \emptyset \,\}$.


### Common refinements

When all input trees share the same leaf set $L$, a tree $T^*$ is a **refinement** of $T$ if
$\mathcal{H}(T) \subseteq \mathcal{H}(T^*)$, i.e., $T$ can be obtained from $T^*$ by contracting
inner edges.  The **minimal common refinement** of $T_1, \dots, T_k$ is the unique tree $T$
satisfying

$$
\mathcal{H}(T) = \bigcup_{i=1}^k \mathcal{H}(T_i),
$$

provided such a tree exists.  A common refinement exists if and only if all clusters from all
input trees are pairwise compatible (no two clusters partially overlap).


## Module overview

All supertree algorithms are provided by `tralda.supertree`:

```python
from tralda.supertree import (
    # BUILD-based and related heuristics
    Build, build_supertree, tree_profile_to_triples,
    greedy_build, best_pair_merge_first,
    # BuildST
    BuildST, build_st,
    # Common refinement
    LinCR, linear_common_refinement,
    # Consensus
    LooseConsensusTree, loose_consensus_tree, merge_all,
)
```

The module implements four main algorithms:

| Algorithm | Input | Complexity | Notes |
|---|---|---|---|
| **BUILD** | triple set $\mathcal{R}$, leaf set $L$ | $O(\lvert\mathcal{R}\rvert \cdot \lvert L\rvert)$ | exact; returns `None` if inconsistent |
| **BuildST** | trees $T_1,\dots,T_k$ | $O(N \log^2 N)$, $N = \sum \lvert\mathcal{L}(T_i)\rvert$ | exact; trees may have different leaf sets |
| **LinCR** | trees $T_1,\dots,T_k$ with a **common** leaf set | $O(k \lvert L \rvert)$ | exact; returns `None` if incompatible |
| **Loose consensus** | trees $T_1,\dots,T_k$ with a **common** leaf set | $O(k \lvert L \rvert)$ | always returns a tree |


## BUILD — triple-based supertree construction

BUILD (Aho et al. 1981) is the classical algorithm for supertree construction.  It operates on a
set of rooted triples rather than on the trees directly.

### Convenience function

The simplest entry point is `build_supertree()`, which extracts representative triples from a
tree profile and runs BUILD in one step:

```python
from tralda.datastructures import Tree
from tralda.supertree import build_supertree, tree_profile_to_triples

T1 = Tree.parse_newick("((a,b),c);")
T2 = Tree.parse_newick("((b,c),d);")

T = build_supertree([T1, T2])
if T is not None:
    T.print_tree()
else:
    print("no supertree exists")
```

`tree_profile_to_triples()` can be used to inspect the extracted triple set:

```python
leaves, triples = tree_profile_to_triples([T1, T2])
print(leaves)   # set of all leaf labels
print(triples)  # set of representative triples as (a, b, c) tuples meaning ab|c
```

### The `Build` class

For more control — including MinCut-based heuristics for inconsistent triple sets — use the
`Build` class directly:

```python
from tralda.supertree import Build

leaves = {"a", "b", "c", "d"}
triples = [("a", "b", "c"), ("b", "c", "d"), ("a", "d", "c")]  # inconsistent

builder = Build(triples, leaves)
T = builder.build_tree()   # returns None if inconsistent

# Use MinCut to resolve inconsistencies (heuristic)
builder_mc = Build(triples, leaves, mincut=True)
T_approx = builder_mc.build_tree()
T_approx.print_tree()
```

With `mincut=True`, when a triple set is inconsistent the algorithm applies the Stoer–Wagner
minimum cut to split connected components and produce a tree anyway.  The resulting tree does
*not* necessarily display all input triples.


### Heuristics for inconsistent triple sets

Two heuristics are available when the triple set is inconsistent and an approximate solution is
acceptable:

**Greedy BUILD** adds triples one by one in decreasing weight order, skipping any triple that
would create an inconsistency:

```python
from tralda.supertree import greedy_build

T = greedy_build(triples, leaves)
```

**Best-Pair-Merge-First (BPMF)** (Wu 2004, modified by Byrka et al. 2010) scores all pairs of
subtrees by how strongly the weighted triple set supports merging them, and greedily merges the
best-scoring pair at each step:

```python
from tralda.supertree import best_pair_merge_first

T = best_pair_merge_first(triples, leaves)
```

Both functions accept an optional `triple_weights` dictionary mapping each triple to a numeric
weight.


## BuildST — fast compatibility testing

The function `build_st()` implements the algorithm of Deng & Fernández-Baca (2016), which tests
compatibility and constructs a supertree directly from the input trees without first converting
them to a triple set.  Unlike BUILD, the input trees may have **different** leaf sets.

```python
from tralda.supertree import build_st

T1 = Tree.parse_newick("((a,b),c);")
T2 = Tree.parse_newick("((b,c),d);")
T3 = Tree.parse_newick("(d,(a,c));")

T = build_st([T1, T2, T3])
if T is not None:
    print(T.to_newick())
```

!!! note
    The graph that connects two input trees whenever they share at least one leaf label must be
    connected.  If this condition is not met, `build_st()` returns `None` even though a supertree
    might exist.  It is planned to relax this requirement in a future release, allowing the
    algorithm to return a supertree by operating on each connected component of the input profile
    separately and merging the results at the end.

The underlying `BuildST` class exposes a `run()` method for the same functionality:

```python
from tralda.supertree import BuildST

builder = BuildST([T1, T2, T3])
T = builder.run()
```


## LinCR — linear-time common refinement

When all input trees share the **same leaf set**, the minimal common refinement can be computed
in $O(k \lvert L \rvert)$ time using the `LinCR` algorithm (Schaller, Hellmuth & Stadler 2021).
This is asymptotically faster than both BUILD and BuildST for this special case.

```python
from tralda.supertree import linear_common_refinement

T1 = Tree.parse_newick("((a,b),c,d);")
T2 = Tree.parse_newick("((a,b,c),d);")

T = linear_common_refinement([T1, T2])
if T is not None:
    print(T.to_newick())   # (((a,b),c),d);  — the minimal common refinement
else:
    print("trees are incompatible — no common refinement exists")
```

The `LinCR` class can be used directly and provides access to intermediate results:

```python
from tralda.supertree import LinCR

cr = LinCR([T1, T2])
T = cr.run()
```

!!! note
    `run()` may only be called once per `LinCR` instance.  Construct a new instance to repeat the
    computation.

The algorithm works bottom-up: starting from the leaves, it builds the parent function of the
candidate refinement tree by computing, for each node $v$, the quantity

$$
p_i(v) := \text{lca}_{T_i}(\mathcal{L}(T(v)))
$$

for every input tree $T_i$.  The parent of $v$ in the common refinement is then the
$\preceq_T$-minimal element among all such $p_i(v)$ values (and the parents of $v$ in any tree
that contains $v$ as an inner node).  Correctness of the resulting tree is verified at the end by
checking that it displays all input trees.

### Comparison with other algorithms

For trees on a common leaf set:

- **LinCR** is $O(k|L|)$ in time and space — optimal for this setting.
- **BuildST** is $O(N \log^2 N)$ with $N = k|L|$ — nearly linear with a poly-logarithmic factor.
- **BUILD** requires first extracting a triple set of size $O(k|L|^2)$ for poorly resolved trees,
  leading to $O(k|L|^3)$ in the worst case.

In practice, LinCR is significantly faster than BuildST and BUILD for large $k$ or $|L|$.


## Loose consensus tree

The **loose consensus tree** of $T_1, \dots, T_k$ (all on the same leaf set $L$) contains
exactly the clusters that appear in at least one $T_i$ *and* are compatible with all clusters of
all other input trees.  When the input trees are compatible, the loose consensus tree coincides
with the minimal common refinement.

```python
from tralda.supertree import loose_consensus_tree

T1 = Tree.parse_newick("((a,b),(c,d));")
T2 = Tree.parse_newick("(((a,b),c),d);")  # incompatible with T1

T = loose_consensus_tree([T1, T2])
T.print_tree()   # only clusters compatible with both trees are retained
```

To merge trees that are **known to be compatible** without the overhead of compatibility checking,
use `merge_all()`:

```python
from tralda.supertree import merge_all

T = merge_all([T1, T2, T3])   # undefined behavior if trees are incompatible
```

!!! warning
    `merge_all()` assumes that all input trees are pairwise compatible.  If they are not, the
    result is undefined.  Use `linear_common_refinement()` or `loose_consensus_tree()` when
    compatibility is not guaranteed.
