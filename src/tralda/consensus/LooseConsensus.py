#!/usr/bin/env python3

"""
Construction of a loose consensus tree.
"""


__author__ = 'David Schaller'


from tralda.datastructures.Tree import Tree, TreeNode, LCA


def merge_trees(T1, T2):
    
    # list of leaves ordered as in T1
    L = [l for l in T1.leaves()]
    n = len(L)
    
    # bijection such that clusters in T1 form sequences os consecutive integers
    f = {l: i for i, l in enumerate(L)}
    
    # dict storing the minimal f(x) for each vertex in T2
    m = {}
    for v in T2.postorder():
        m[v] = min(f[c] for c in v.children) if v.children else f[v]
    
    # dict storing for each vertex the number edges from the root
    depth = {}
    for v in T2.preorder():
        depth[v] = depth[v.parent] + 1 if v.parent else 0
    
    counting_sort = [[] for i in range(n)]
    for v in T2.preorder():
        counting_sort[m(v)].append(v)
    sorted_vertices = [v for sublist in counting_sort for v in sublist]
    
    # copy T2 while ordering all children according to counting_sort
    orig_to_new = {}
    for orig in sorted_vertices:
        new = TreeNode(orig.ID, label=orig.label)
        orig_to_new[orig] = new
        if orig.parent:
            orig_to_new[orig.parent].add_child(new)
    T2 = Tree(orig_to_new[T2.root])
    
    leaf_rank = {l.ID: i for i, l in enumerate(T2.leaves())}
    
    start, stop = {}, {}
    for u in T1.postorder():
        if not u.children:
            start[u] = leaf_rank[u.ID]
            stop[u] = leaf_rank[u.ID]
        else:
            start[u] = min(start[c] for c in u.children)
            stop[u] = max(stop[c] for c in u.children)
    
    # vertex in T2 with smallest depth whose leftmost leaf is x
    x_left, leftmost = {}, {}
    # vertex in T2 with smallest depth whose rightmost leaf is x
    x_right, rightmost = {}, {}
    
    for v in T2.postorder():
        if not v.children:
            x_left[v] = v
            leftmost[v] = v
            x_right[v] = v
            rightmost[v] = v
        else:
            cur_leftmost = leftmost[v.children[0]]
            leftmost[v] = cur_leftmost
            x_left[cur_leftmost] = v
            
            cur_rightmost = rightmost[v.children[-1]]
            rightmost[v] = cur_rightmost
            x_right[cur_rightmost] = v
    
    lca = LCA(T2)