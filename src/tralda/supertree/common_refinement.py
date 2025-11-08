"""Fast computation of a common refinement of trees on the same leaf set.

References:
    .. [1] D. Schaller, M. Hellmuth, P.F. Stadler (2021). A simpler linear-time algorithm for the
           common refinement of rooted phylogenetic trees on a common leaf set.
           Algorithms Mol Biol. 16(1):23. DOI: 10.1186/s13015-021-00202-8.
"""

from __future__ import annotations

from collections.abc import Collection
from collections import deque

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode
from tralda.utils.tree_tools import assert_leaf_sets_equal


def linear_common_refinement(trees: Collection[Tree]) -> Tree | None:
    """Minimal common refinement for a set of trees with the same leaf set.

    All input trees must be phylogenetic and have the same set of leaf labels. In each tree, the
    leaves' label attributes must be set and unique.

    Args:
        trees: A collection of of Tree instances.

    Returns:
        A common refinement of the input trees if existent, None otherwise.

    References:
    .. [1] D. Schaller, M. Hellmuth, P.F. Stadler (2021). A simpler linear-time algorithm for the
           common refinement of rooted phylogenetic trees on a common leaf set.
           Algorithms Mol Biol. 16(1):23. DOI: 10.1186/s13015-021-00202-8.
    """
    cr = LinCR(trees)

    return cr.run()


class LinCR:
    """Minimal common refinement for a set of trees with the same leaf set in linear time.

    References:
    .. [1] D. Schaller, M. Hellmuth, P.F. Stadler (2021). A simpler linear-time algorithm for the
           common refinement of rooted phylogenetic trees on a common leaf set.
           Algorithms Mol Biol. 16(1):23. DOI: 10.1186/s13015-021-00202-8.
    """

    def __init__(self, trees: Collection[Tree]) -> None:
        """Constructor of the LinCR class.

        Args:
            trees: A collection of of Tree instances.

        Raises:
            ValueError: If the trees do not have the same leaf labels.
        """
        self.trees = trees
        self.L = assert_leaf_sets_equal(trees)

        if not self.L:
            raise ValueError("trees must have the same leaf labels")

        # ensure that the run() method is only called once
        self.already_run = False

        self.T: Tree | None = None  # the common refinement (CR) candidate
        self.queue = deque[TreeNode]()

        # vertices in the CR
        self.vertices_in_cr: set[TreeNode] = set()

        # vertex v in an input trees --> number of leaves below v
        self.vertex2num_leaves: dict[TreeNode, int] = {}

        # list of dict per tree Ti mapping the nodes v in the constructed tree --> lca_Ti (L(Ti(v)))
        self.p: list[dict[TreeNode, TreeNode]] = [{} for _ in range(len(self.trees))]

        # vertex in the common refinement --> indices of the trees with corresponding vertex
        self.J: dict[TreeNode, dict[int, TreeNode | None]] = {}

        # inner vertex in input tree --> corresponding vertex in the common refinement
        self.vi_to_v: dict[TreeNode, TreeNode] = {}

    def run(self) -> Tree | None:
        """Run the the algorithm for constructing the minimal common refinement.

        Returns:
            The minimal common refinement if it exits; None otherwise.

        Raises:
            ValueError: If the method was already called for this instance.
        """
        if self.already_run:
            raise ValueError("the 'run' method can only be run once")

        self.already_run = True
        self._initialize()
        self._build_tree()

        if self.T and self.T.is_phylogenetic() and self._check_displayed():
            return self.T

        # otherwise return None

    def _initialize(self) -> None:
        """Initialize the lookups."""
        # compute the number of leaves below each vertex
        for T_i in self.trees:
            for v in T_i.postorder():
                if v.is_leaf():
                    self.vertex2num_leaves[v] = 1
                else:
                    self.vertex2num_leaves[v] = sum(self.vertex2num_leaves[w] for w in v.children)

        # label --> leaf vertex in the common refinement
        label2vertex_in_cr = {}

        # initialize the other mappings
        for label in self.L:
            v = TreeNode(label=label)
            self.queue.append(v)
            self.vertices_in_cr.add(v)
            label2vertex_in_cr[label] = v
            self.J[v] = {i: None for i in range(len(self.trees))}
            self.vertex2num_leaves[v] = 1

        for i, T_i in enumerate(self.trees):
            for v_i in T_i.leaves():
                v = label2vertex_in_cr[v_i.label]
                self.p[i][v] = v_i
                self.J[v][i] = v_i
                self.vi_to_v[v_i] = v

    def _build_tree(self) -> None:
        """Constructs the candidate for the common refinement tree.

        Raises:
            RuntimeError: If the root could not be determined.
        """
        self.root = None

        while self.queue:
            v = self.queue.popleft()
            u = None
            l_min = len(self.L)
            J_u = {}

            for i in range(len(self.trees)):
                if i in self.J[v]:
                    u2 = self.J[v][i].parent
                else:
                    u2 = self.p[i][v]

                if u is None or self.vertex2num_leaves[u2] < l_min:
                    u = u2
                    l_min = self.vertex2num_leaves[u2]
                    J_u.clear()
                    J_u[i] = u2
                elif self.vertex2num_leaves[u2] == l_min:
                    J_u[i] = u2

            if self.vertex2num_leaves[v] < l_min:
                if u in self.vi_to_v:
                    u = self.vi_to_v[u]
                else:
                    u = TreeNode()
                    self.J[u] = J_u
                    for i, u_i in J_u.items():
                        self.vi_to_v[u_i] = u
                        self.J[u][i] = u_i
                    self.vertex2num_leaves[u] = l_min
                u.add_child(v)
            else:
                return

            if u not in self.vertices_in_cr and l_min < len(self.L):
                self.queue.append(u)
                self.vertices_in_cr.add(u)
                if len(self.vertices_in_cr) > 2 * len(self.L) - 2:
                    return

                for i in range(len(self.trees)):
                    if i in self.J[u]:
                        self.p[i][u] = self.J[u][i]
                    elif i in self.J[v]:
                        self.p[i][u] = self.J[v][i].parent
                    else:
                        self.p[i][u] = self.p[i][v]

            elif l_min == len(self.L):
                self.root = u

        if not self.root:
            raise RuntimeError("could not determine root")

        self.T = Tree(self.root)

    def _check_displayed(self) -> bool:
        """Checks whether all input trees are displayed by the constructed tree.

        Returns:
            True if all trees are displayed by the constructed tree candidate, False otherwise.
        """
        for i, T_i in enumerate(self.trees):
            T_copy, v_copy = self.T.copy(mapping=True)

            to_contract = [
                (v_copy[v].parent, v_copy[v])
                for v in self.T.inner_nodes()
                if i not in self.J[v] and v.parent
            ]

            T_copy.contract(to_contract)

            for v_i in T_i.preorder():
                if v_i not in self.vi_to_v:
                    return False
                elif v_i.parent is None:
                    if v_copy[self.vi_to_v[v_i]].parent is not None:
                        return False
                elif v_i.parent not in self.vi_to_v:
                    return False
                elif v_copy[self.vi_to_v[v_i.parent]] is not v_copy[self.vi_to_v[v_i]].parent:
                    return False

        return True
