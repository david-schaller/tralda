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
    f = {l.ID: i for i, l in enumerate(L)}
    
    # dict storing the minimal f(x) for each vertex in T2
    m = {}
    for v in T2.postorder():
        m[v] = min(m[c] for c in v.children) if v.children else f[v.ID]
    
    counting_sort = [[] for i in range(n)]
    for v in T2.preorder():
        counting_sort[m[v]].append(v)
    sorted_vertices = [v for sublist in counting_sort for v in sublist]
    
    # copy T2 while ordering all children according to counting_sort
    orig_to_new = {}
    for orig in sorted_vertices:
        new = TreeNode(orig.ID, label=orig.label)
        orig_to_new[orig] = new
        if orig.parent:
            orig_to_new[orig.parent].add_child(new)
    T2 = Tree(orig_to_new[T2.root])
    
    # dict storing for each vertex the number edges from the root
    depth = {}
    for v in T2.preorder():
        depth[v] = depth[v.parent] + 1 if v.parent else 0
    
    L_T2 = list(T2.leaves())
    leaf_rank = {l.ID: i for i, l in enumerate(L_T2)}
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
    
    for u in T1.postorder():
        
        if not u.children:
            continue
        
        a = L_T2[start[u]]
        b = L_T2[stop[u]]
        r_u = lca(a, b)
        
        leftmost_child = r_u.children[0]
        rightmost_child = r_u.children[-1]
        
        d_u = x_left[a] if depth[x_left[a]] > depth[r_u] else leftmost_child
        e_u = x_right[b] if depth[x_right[b]] > depth[r_u] else rightmost_child
        
        if d_u is leftmost_child and e_u is rightmost_child:
            continue
        
        c = TreeNode(-1)
        depth[c] = depth[r_u] + 1
        if d_u is not leftmost_child:
            x_left[a] = c
        if e_u is not rightmost_child:
            x_right[b] = c
        r_u.add_child_right_of(c, e_u)
        
        c_children = r_u.child_subsequence(d_u, e_u)
        for child in c_children:
            c.add_child(child)
    
    return T2

            
if __name__ == '__main__':
    
    import random
    
    tree = Tree.random_tree(10, binary=True)
    partial_trees = []
    for i in range(10):
        T_i = tree.copy()
        edges = []
        for u, v in T_i.inner_edges():
            if random.random() < 0.8:
                edges.append((u,v))
        T_i.contract(edges)
        partial_trees.append(T_i)
    
    T1 = partial_trees[0]
    T2 = partial_trees[1]
    T = merge_trees(T1, T2)
    print(T1.to_newick())
    print(T2.to_newick())
    print(T.to_newick())
    print(T.is_refinement(T1))
    print(T.is_refinement(T2))