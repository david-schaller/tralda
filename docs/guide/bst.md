# Balanced Binary Search Trees

## Background

A **binary search tree (BST)** is a rooted binary tree in which every node holds a **key**, and
for each node $v$ the following invariant holds:

- all keys in the *left* subtree of $v$ are strictly smaller than $v$'s key, and
- all keys in the *right* subtree of $v$ are strictly greater than $v$'s key.

This invariant makes search, insertion, and deletion all run in $O(h)$ time, where $h$ is the
height of the tree.  For a random tree $h = O(\log n)$, but in the worst case a plain BST
degrades to $O(n)$.  **Self-balancing** BSTs keep $h = O(\log n)$ at all times by performing
local restructuring operations (rotations) after each insertion or deletion.

`tralda.datastructures.bst` provides two such implementations:

| Class | Balancing strategy | `join` / `split` |
|---|---|---|
| `TreeSet` / `TreeDict` (AVL) | AVL — balance factor kept in $[-1, 1]$ | no |
| `TreeSet` (red-black) | Red-black properties | yes |

Both implementations maintain the subtree **size** and **height** in every node, enabling
$O(\log n)$ order-statistic queries (rank, access by index).


## AVL tree — `TreeSet` and `TreeDict`

The AVL-tree implementations are the main public interface and are re-exported directly from
`tralda.datastructures`:

```python
from tralda.datastructures import TreeSet, TreeDict
```

### `TreeSet` — sorted set

`TreeSet` is a sorted set backed by an AVL tree.  All standard set-like operations run in
$O(\log n)$ time.

```python
from tralda.datastructures import TreeSet

s = TreeSet()

# insertion
s.insert(5)     # raises KeyError if key already present
s.add(3)        # silently ignores duplicates
s.add(7)
s.add(3)        # no-op

# membership and size
print(5 in s)   # True
print(len(s))   # 3

# removal
s.remove(3)          # raises KeyError if absent
s.discard(99)        # silent no-op if absent
last = s.pop()       # remove and return the largest element

# iteration in sorted order
for key in s:
    print(key)
```

#### Access by index

Because each node stores its subtree size, elements can be looked up by their **rank** (0-based
index in sorted order) in $O(\log n)$ time:

```python
s = TreeSet()
for x in [10, 20, 30, 40, 50]:
    s.add(x)

print(s[0])                  # 10 — smallest
print(s[-1])                 # 50 — largest
print(s.key_at_index(2))     # 30

val = s.pop_at_index(1)      # remove and return element at index 1 (= 20)
```

Negative indices are supported and follow the same convention as Python lists.

#### Bulk removal

```python
s.difference_update([10, 30])   # discard all elements in the iterable
s.clear()                       # remove everything
```

### `TreeDict` — sorted dictionary

`TreeDict` extends `TreeSet` so that each key carries an associated **value**, analogous to a
`dict` but kept in sorted key order.

```python
from tralda.datastructures import TreeDict

d = TreeDict()

d.insert("b", 2)
d.add("a", 1)     # add() is an alias for insert() on TreeDict
d.add("c", 3)

# key-based access
print(d["b"])              # 2
print(d.get("z", 0))       # 0 — default for missing keys

# index-based access
print(d.key_at_index(0))             # "a"
print(d.value_at_index(0))           # 1
print(d.key_and_value_at_index(1))   # ("b", 2)

# iteration
for k in d.keys():
    print(k)

for v in d.values():
    print(v)

for k, v in d.items():
    print(k, v)
```

Removing entries works the same way as for `TreeSet`:

```python
d.remove("b")
d.discard("z")         # silent no-op
d.pop_at_index(0)      # removes the entry with the smallest key
```


## Red-black tree

The red-black tree in `tralda.datastructures.bst.red_black` provides a `TreeSet` with two
additional bulk operations — **join** and **split** — which are not available on the AVL tree.

```python
from tralda.datastructures.bst.red_black import TreeSet as RBTreeSet
```

!!! note
    The red-black `TreeSet` supports the same basic interface as the AVL `TreeSet` (insertion,
    deletion, membership, index access, iteration), but is not re-exported from
    `tralda.datastructures` by default.  Import it directly from
    `tralda.datastructures.bst.red_black`.

### `join`

`join` merges two disjoint trees $T_L$ and $T_R$ — where every key in $T_L$ is strictly less
than every key in $T_R$ — into a single balanced tree in $O(\log n)$ time.  An optional
separator key between the two trees may be provided:

```python
left = RBTreeSet()
for x in [1, 2, 3]:
    left.add(x)

right = RBTreeSet()
for x in [7, 8, 9]:
    right.add(x)

# IMPORTANT: run only one of the following two options, not both!

# without separator key — a dummy node is used internally and then removed
merged = RBTreeSet.join(left, right)

# with an explicit separator key that lies between left and right
merged = RBTreeSet.join(left, right, key=5)
```

!!! warning
    After a `join`, the original `left` and `right` instances must not be used.  Their internal
    state is consumed by the operation.

### `split`

`split_at_node` splits a tree at a given node into two trees $T_L$ (containing keys $\le$ the
split key) and $T_R$ (containing keys $\ge$ the split key) in $O(\log n)$ time.  The split
node itself can optionally be retained in either the left or the right tree:

```python
t = RBTreeSet()
for x in [1, 3, 5, 7, 9]:
    t.add(x)

# find the internal node for key 5
node = t._find(5)   # returns the RedBlackTreeNode with key 5

# IMPORTANT: run only one of the following two options, not both!

left, right = t.split_at_node(node)
# left  contains {1, 3}  (keys strictly less than 5)
# right contains {7, 9}  (keys strictly greater than 5)

# keep the split key in the left tree
left, right = t.split_at_node(node, keep_node_left=True)
# left  contains {1, 3, 5}
# right contains {7, 9}
```

!!! warning
    After a `split_at_node`, the original tree instance must not be used.


## The simple BST

`tralda.datastructures.bst.simple` contains an unbalanced `BinarySearchTree` that is **not
self-balancing** and intended for internal use and testing only.  It has the same interface as
`TreeSet` but degrades to $O(n)$ in the worst case.  Prefer `TreeSet` for any production use.


## Choosing between AVL and red-black

- Use the **AVL `TreeSet` / `TreeDict`** (from `tralda.datastructures`) for the general case.
  AVL trees maintain stricter balance than red-black trees and therefore tend to be faster for
  lookup-heavy workloads.
- Use the **red-black `TreeSet`** when you need `join` or `split` — these bulk operations are
  not available on the AVL tree.
