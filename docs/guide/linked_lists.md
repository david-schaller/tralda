# Linked Lists

## Background

A **singly-linked list** stores each element in a node that holds a value and a pointer to the
next node.  Appending to either end runs in $O(1)$; locating a node by index requires an $O(n)$
traversal from one end.  Because there is no backward pointer, removing a node requires scanning
from the front to find its predecessor first.

A **doubly-linked list** augments each node with an additional pointer to its predecessor.  The
key benefit is $O(1)$ node removal given a direct node reference — no traversal needed.
`DLList` also optimises index lookup by starting from whichever end is closer, halving the
average traversal length.

| Class | Module | Backward traversal | $O(1)$ remove by node |
|---|---|---|---|
| `LinkedList` | `tralda.datastructures.linked` | no | no |
| `DLList` | `tralda.datastructures.doubly_linked` | yes | yes |

Both lists expose their internal node objects (`LinkedListNode` / `DLListNode`).  Callers can
retain a reference to a node and use it later for $O(1)$ insertion next to it or, for `DLList`,
$O(1)$ removal.

## Singly-linked list — `LinkedList`

```python
from tralda.datastructures.linked import LinkedList
```

### Construction and basic access

```python
lst = LinkedList([10, 20, 30])

print(len(lst))     # 3
print(lst.first())  # 10
print(lst.last())   # 30
print(lst[1])       # 20  (O(n) traversal)
```

The constructor accepts an iterable; individual values can also be passed as positional
arguments.

### Appending and prepending

```python
lst.append(40)       # O(1) — add to the right
lst.append_left(5)   # O(1) — add to the left
lst.extend([50, 60]) # O(k) — append k items
```

### Inserting next to a node

Because `append` and `append_left` return the new node, you can keep a reference and later
insert an element immediately to its right in $O(1)$:

```python
node = lst.append(100)
lst.insert_right_of(node, 101)  # O(1)
```

### Removal

```python
lst.remove(20)    # O(n) — scan for value, raises KeyError if absent
lst.popleft()     # O(1) — remove and return the first element
```

Truncation removes a suffix or prefix in $O(n)$:

```python
lst.truncate(3)       # keep only the first 3 elements
lst.truncate_left(2)  # discard the first 2 elements
```

### Concatenation

`concatenate` splices another list onto the right end of this list in $O(1)$.  The other list
must not be used afterwards:

```python
a = LinkedList([1, 2, 3])
b = LinkedList([4, 5, 6])

a.concatenate(b)  # a now contains [1, 2, 3, 4, 5, 6]
```

### Iteration

```python
for value in lst:
    print(value)
```

---

## Doubly-linked list — `DLList`

```python
from tralda.datastructures.doubly_linked import DLList
```

`DLList` provides the same interface as `LinkedList` and extends it with backward-pointer
operations.

### Construction and basic access

```python
lst = DLList([10, 20, 30])

print(len(lst))     # 3
print(lst.first())  # 10
print(lst.last())   # 30
print(lst[1])       # 20  (starts from the closer end)
```

### Appending and prepending

```python
lst.append(40)       # O(1) — add to the right
lst.append_left(5)   # O(1) — add to the left
lst.extend([50, 60]) # O(k)
```

### O(1) removal by node reference

The main advantage of `DLList` over `LinkedList` is that a node can be removed directly in
$O(1)$ without any list traversal:

```python
node = lst.append(99)
lst.remove_node(node)  # O(1)
```

`remove` by value is also available but requires $O(n)$ traversal:

```python
lst.remove(20)  # O(n), raises KeyError if absent
```

### Inserting next to a node

```python
node = lst.first_node()
lst.insert_right_of(node, 15)  # O(1) — inserts 15 right after the first node
```

### Popping from either end

```python
lst.popleft()   # O(1)
lst.popright()  # O(1)
```

### Range removal

`remove_range` removes a contiguous slice of the list.  The start and end indices are resolved in
$O(n)$; the removal itself is $O(1)$ for internal ranges:

```python
lst = DLList([0, 1, 2, 3, 4, 5])
lst.remove_range(2, 3)   # remove 3 elements starting at index 2 → [0, 1, 5]
```

If `length` is omitted (or would reach past the end), the list is truncated from `index`
onwards:

```python
lst.remove_range(2)   # equivalent to lst.truncate(2)
```

### Sublist

`sublist` returns the values between two node references as a plain Python list:

```python
lst = DLList([10, 20, 30, 40, 50])
left  = lst.first_node()          # node for 10
right = lst.node_at(2)            # node for 30

print(lst.sublist(left, right))   # [10, 20, 30]
```

### Iteration

```python
for value in lst:
    print(value)
```
