# tralda

tralda is an open-source Python library for for **tr**ee **al**gorithms and **da**ta structures.
It provides efficient implementations of algorithms for tree manipulation and analysis, as well as
a tree data structure with various methods for tree traversal and manipulation.

## Quick start

```bash
pip install tralda
```

```python
from tralda.datastructures import Tree

tree = Tree.random_tree(10)
tree.print_tree()
```

Output:

```
0
├──1
└──2
   ├──3
   │  ├──5
   │  ├──6
   │  │  ├──12
   │  │  └──13
   │  └──10
   ├──4
   ├──7
   │  ├──8
   │  ├──9
   │  └──14
   └──11
```

See the [User Guide](guide/index.md) for a full walkthrough, or jump straight to the
[API Reference](api/trees.md).
