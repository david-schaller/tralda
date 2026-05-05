# Dynamic Partition

## Background

A **partition** of a set $U$ is a collection of pairwise-disjoint, non-empty subsets whose
union equals $U$.  A *dynamic* partition supports efficient merging of two of its sets.

`Partition` uses a **small-into-large** (weighted-union) strategy: when merging two sets, the
elements of the smaller set are moved into the larger set and their lookup pointers are updated.
Because an element can only be moved into a strictly larger set, each element is re-assigned at
most $O(\log n)$ times in total, giving the following amortised guarantees:

| Operation | Time |
|---|---|
| `in_same_set` | $O(1)$ |
| `separated_xy_z` | $O(1)$ |
| `merge` | $O(\lvert S_{\text{small}} \rvert)$ per call; $O(n \log n)$ over all merges |

## Dynamic partition in `tralda`

```python
from tralda.datastructures.partition import Partition
```

### Construction

`Partition` is initialised from an iterable of iterables, where each inner iterable defines one
set of the partition:

```python
p = Partition([[1, 2, 3], [4, 5], [6]])

print(len(p))  # 3 — number of sets
```

### Membership queries

```python
print(p.in_same_set(1, 2))   # True
print(p.in_same_set(1, 4))   # False
```

`separated_xy_z` checks the common pattern "*x and y are together, but z is not*":

```python
print(p.separated_xy_z(1, 2, 4))   # True  — {1,2,3} vs {4,5}
print(p.separated_xy_z(1, 4, 6))   # False — 1 and 4 are not in the same set
```

Both queries run in $O(1)$ via a dictionary lookup.

### Merging sets

`merge` takes one representative element from each of the two sets to merge.  Any element of
the set can serve as its representative:

```python
p.merge(1, 4)               # merge the set containing 1 with the set containing 4

print(len(p))               # 2 — now two sets remain
print(p.in_same_set(2, 5))  # True — previously separate sets now joined
```

`merge` returns the *smaller* of the two original sets (the one whose elements were moved), or
an empty set if both representatives already belong to the same set.

### Iterating over sets

```python
for subset in p:
    print(subset)   # each subset is a plain Python set
```