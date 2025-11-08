"""Module for the construction of loose consensus trees.

References:
    .. [1] J. Jansson, C. Shen, W.-K. Sung. Improved Algorithms for Constructing Consensus Trees.
           J. ACM 63, 3, Article 28 (June 2016), 24 pages. DOI: 10.1145/2925985.
"""

from __future__ import annotations

from collections.abc import Collection

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode
from tralda.datastructures.last_common_ancestor import LCA
from tralda.utils.tree_tools import assert_leaf_sets_equal


def loose_consensus_tree(trees: Collection[Tree]) -> Tree | None:
    """Compute the loose consensus tree for trees with the same leaf set.

    All input trees must be phylogenetic and have the same set of leaf labels. In each tree, the
    leaves' label attributes must be set and unique.

    Parameters:
        trees: A collection of of Tree instances.

    Returns:
        The loose consensus tree; None if 'trees' is empty.

    References:
        .. [1] J. Jansson, C. Shen, W.-K. Sung. Improved Algorithms for Constructing Consensus
               Trees. J. ACM 63, 3, Article 28 (June 2016), 24 pages. DOI: 10.1145/2925985.
    """
    loose_consensus = LooseConsensusTree(trees)

    return loose_consensus.run()


class LooseConsensusTree:
    """Construction of the loose consensus tree.

    References:
        .. [1] J. Jansson, C. Shen, W.-K. Sung. Improved Algorithms for Constructing Consensus
               Trees. J. ACM 63, 3, Article 28 (June 2016), 24 pages. DOI: 10.1145/2925985.
    """

    def __init__(self, trees: Collection[Tree]) -> None:
        """_summary_

        Args:
            trees: A collection of of Tree instances.

        Raises:
            ValueError: If the trees do not have the same leaf labels.
        """
        if not assert_leaf_sets_equal(trees):
            raise ValueError("trees must have the same leaf labels")

        self.trees = trees

        # the loose consensus tree to be contracted
        self.T: Tree | None = None

        # the total number of edges that were contracted
        self.total_contracted = 0

    def run(self) -> Tree | None:
        """Run the construction of the loose consensus tree.

        Returns:
            The loose consensus tree; None if the collection of trees is empty.
        """
        self.total_contracted = 0

        k = len(self.trees)
        if k == 0:
            return
        elif k == 1:
            return self.trees[0].copy()

        tree = self.trees[0]
        for j in range(1, k):
            compatible_tree, x = one_way_compatible(
                tree,
                self.trees[j],
                return_no_of_contractions=True,
            )
            self.total_contracted += x
            tree = merge_trees(compatible_tree, self.trees[j])

        for j in range(k):
            tree, x = one_way_compatible(tree, self.trees[j], return_no_of_contractions=True)
            self.total_contracted += x

        self.T = tree

        return self.T


def merge_all(trees: Collection[Tree]):
    """Common refinement for compatible trees with the same leaf set.

    All input trees must be phylogenetic and have the same set of leaf labels. In each tree, the
    leaves' label attributes must be set and unique.

    WARNING: The behavior of the function is undefined if the trees are not compatible, i.e., if no
    common refinement exists.

    Args:
        trees: A collection of of Tree instances.

    Returns:
        A common refinement of the input trees; None if 'trees' is empty.
    """
    if not assert_leaf_sets_equal(trees):
        raise RuntimeError("trees must have the same leaf labels")

    k = len(trees)
    if k == 0:
        return None
    elif k == 1:
        return trees[0].copy()

    merged_tree = trees[0]
    for j in range(1, k):
        merged_tree = merge_trees(merged_tree, trees[j])

    return merged_tree


def _preprocess(
    tree1: Tree,
    tree2: Tree,
) -> tuple[
    Tree,
    list[TreeNode],
    LCA,
    dict[TreeNode, int],
    dict[TreeNode, int],
    dict[TreeNode, int],
    dict[TreeNode, TreeNode],
    dict[TreeNode, TreeNode],
]:
    """Auxiliary preprocessing function for merge_trees() and one_way_compatible().

    The returned tuples contains:
    - A copied version of T2 with re-ordered children.
    - A list of leaves in the copied T2.
    - An LCA instance for the copied T2.
    - A dictionary with the depth of each vertex in the copied T2.
    - The 'start' dictionary.
    - The 'stop' dictionary.
    - Dict mapping a vertex x to the vertex in T2 with smallest depth whose leftmost leaf is x.
    - Dict mapping a vertex x to the vertex in T2 with smallest depth whose rightmost leaf is x.

    Args:
        tree1: The tree T1.
        tree2: The tree T2.

    Returns:
        A tuple containing the copied tree and preprocessed lookups etc. (see above).
    """
    # bijection such that clusters in T1 form sequences of consecutive integers
    f = {leaf.label: i for i, leaf in enumerate(tree1.leaves())}

    # dict storing the minimal f(x) for each vertex in T2
    m = {}
    for v in tree2.postorder():
        m[v] = min(m[c] for c in v.children) if v.children else f[v.label]

    counting_sort = [[] for _ in range(len(f))]
    for v in tree2.preorder():
        counting_sort[m[v]].append(v)
    sorted_vertices = [v for sublist in counting_sort for v in sublist]

    # copy T2 while ordering all children according to counting_sort
    orig_to_new = {}
    for orig in sorted_vertices:
        new = TreeNode()
        orig_to_new[orig] = new
        if orig.parent:
            orig_to_new[orig.parent].add_child(new)
        for key, value in orig.attributes():
            setattr(new, key, value)
    tree2 = Tree(orig_to_new[tree2.root])

    # dict storing for each vertex the number edges from the root
    depth = {}
    for v in tree2.preorder():
        depth[v] = depth[v.parent] + 1 if v.parent else 0

    L_tree2 = list(tree2.leaves())
    leaf_rank = {leaf.label: i for i, leaf in enumerate(L_tree2)}
    start = {}
    stop = {}

    for u in tree1.postorder():
        if not u.children:
            start[u] = leaf_rank[u.label]
            stop[u] = leaf_rank[u.label]
        else:
            start[u] = min(start[c] for c in u.children)
            stop[u] = max(stop[c] for c in u.children)

    # vertex in T2 with smallest depth whose leftmost leaf is x
    x_left, leftmost = {}, {}

    # vertex in T2 with smallest depth whose rightmost leaf is x
    x_right, rightmost = {}, {}

    for v in tree2.postorder():
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

    lca = LCA(tree2)

    return tree2, L_tree2, lca, depth, start, stop, x_left, x_right


def merge_trees(tree1, tree2):
    """Common refinement of two compatible trees T1 and T2 with the same leaf set.

    Both input trees must be phylogenetic and have the same set of leaf labels. In each tree, the
    leaves' label attributess must be set and unique.

    Args:
        tree1: The tree T1.
        tree2: The tree T2.

    Returns:
        A tree whose clusters are the union of the cluster of T1 and T2.

    References:
        .. [1] J. Jansson, C. Shen, W.-K. Sung. Improved Algorithms for Constructing Consensus
               Trees. J. ACM 63, 3, Article 28 (June 2016), 24 pages. DOI: 10.1145/2925985.
    """
    # T2 gets copied in _preprocess, so input trees remain unchanged
    tree2, L_tree2, lca, depth, start, stop, x_left, x_right = _preprocess(tree1, tree2)

    for u in tree1.postorder():
        if not u.children:
            continue

        a = L_tree2[start[u]]
        b = L_tree2[stop[u]]
        r_u = lca(a, b)

        leftmost_child = r_u.children[0]
        rightmost_child = r_u.children[-1]

        d_u = x_left[a] if depth[x_left[a]] > depth[r_u] else leftmost_child
        e_u = x_right[b] if depth[x_right[b]] > depth[r_u] else rightmost_child

        if d_u is leftmost_child and e_u is rightmost_child:
            continue

        c = TreeNode()
        depth[c] = depth[r_u] + 1
        if d_u is not leftmost_child:
            x_left[a] = c
        if e_u is not rightmost_child:
            x_right[b] = c
        r_u.add_child_right_of(c, e_u)

        c_children = r_u.child_subsequence(d_u, e_u)
        for child in c_children:
            c.add_child(child)

    return tree2


def one_way_compatible(
    tree1: Tree,
    tree2: Tree,
    return_no_of_contractions: bool = False,
) -> Tree | tuple[Tree, int]:
    """Remove all clusters in a tree T1 that are incompatible with tree T2.

    Both input trees must be phylogenetic and have the same set of leaf labels. In each tree, the
    leaves' label attributes must be set and unique.

    Args:
        tree1: The tree T1.
        tree2: The tree T2.

    Returns:
        A copy of T1 with all clusters removed (i.e. the corresponding edges are contracted) that
        are incompatible with T2.

    Raises:
        RuntimeError: If a vertex unexpectedly does not have a parent.

    References:
        .. [1] J. Jansson, C. Shen, W.-K. Sung. Improved Algorithms for Constructing Consensus
               Trees. J. ACM 63, 3, Article 28 (June 2016), 24 pages. DOI: 10.1145/2925985.
    """
    # T2 gets copied in _preprocess, so input trees remain unchanged
    tree2, L_tree2, lca, depth, start, stop, x_left, x_right = _preprocess(tree1, tree2)

    # size of all cluster in T1
    size = {}

    # edges uv such that L(T1(v)) is a cluster that is not compatible with T2
    bad_edges = []

    for u in tree1.postorder():
        size[u] = sum(size[c] for c in u.children) if u.children else 1

        if not u.children:
            continue

        a = L_tree2[start[u]]
        b = L_tree2[stop[u]]
        r_u = lca(a, b)

        leftmost_child = r_u.children[0]
        rightmost_child = r_u.children[-1]

        d_u = x_left[a] if depth[x_left[a]] > depth[r_u] else leftmost_child
        e_u = x_right[b] if depth[x_right[b]] > depth[r_u] else rightmost_child

        if (
            (d_u.parent is not r_u)
            or (e_u.parent is not r_u)
            or (size[u] != stop[u] - start[u] + 1)
        ):
            # this should never appear
            if not u.parent:
                raise RuntimeError(f"vertex {u.parent} has no parent")

            bad_edges.append((u.parent, u))

    tree1 = tree1.contract(bad_edges, inplace=False)

    if return_no_of_contractions:
        return tree1, len(bad_edges)
    else:
        return tree1
