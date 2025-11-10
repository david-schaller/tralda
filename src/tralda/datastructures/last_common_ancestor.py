"""Efficient computation of last/least common ancestors in trees.

References:
    .. [1] M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena, P. Sumazin. Lowest common
           ancestors in trees and directed acyclic graphs. In: Journal of Algorithms. 57, Nr. 2,
           November 2005, S. 75-94. DOI: 10.1016/j.jalgor.2005.08.001
    .. [2] https://cp-algorithms.com/data_structures/sparse-table.html
    .. [3] https://cp-algorithms.com/graph/lca_farachcoltonbender.html
"""

from __future__ import annotations

from typing import Iterable
from typing import Iterator
from typing import Union

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode


NodeType = Union[TreeNode, int]
NodeOrEdge = Union[NodeType, tuple[NodeType, NodeType], list[NodeType]]
Triple = tuple[NodeType, NodeType, NodeType]


class LCA:
    """Compute last common ancestors in a tree efficiently.

    Uses a reduction to a +/-1 Range minimum query (RMQ) problem and a sparse table implementation.
    - Preprocessing complexity: O(n) where n is the number of nodes in the tree
    - Query complexity: O(1)

    References:
        .. [1] M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena, P. Sumazin. Lowest common
               ancestors in trees and directed acyclic graphs. In: Journal of Algorithms. 57, Nr. 2,
               November 2005, S. 75-94. DOI: 10.1016/j.jalgor.2005.08.001
        .. [2] https://cp-algorithms.com/data_structures/sparse-table.html
        .. [3] https://cp-algorithms.com/graph/lca_farachcoltonbender.html
    """

    def __init__(self, tree: Tree) -> None:
        """Constructor for class LCA.

        Args:
            tree: The Tree instance for which this instance will allow efficient last common
                ancestor queries.

        Raises:
            If 'tree' is not a Tree instance.
        """
        if not isinstance(tree, Tree):
            raise TypeError("tree must be of type 'Tree'")

        self._tree = tree

        self._V = [v for v in self._tree.preorder()]
        self._index = {v: i for i, v in enumerate(self._V)}

        # store labels for queries via label
        self._label_dict = {v.label: v for v in self._V if hasattr(v, "label")}

        self._euler_tour = []
        # levels of the vertices in the Euler tour
        self._L = []
        # repres. of vertices in the Euler tour (index of first occurence)
        self._R = [None for _ in range(len(self._V))]

        for j, (v, level) in enumerate(self._tree.euler_and_level()):
            i = self._index[v]
            self._euler_tour.append(i)
            self._L.append(level)
            if self._R[i] is None:
                self._R[i] = j

        # build sparse table for range minimum query (RMQ)

        # initialize array and lookup dictionaries
        n = len(self._L)
        self._log_2 = self._precompute_logs(n)
        self._block_size = max(1, self._log_2[n] // 2)
        self._block_count = (n + self._block_size - 1) // self._block_size
        self._A = [0 for _ in range(self._block_count)]
        self._B = [0 for _ in range(self._block_count)]
        self._block_identifier = [0 for _ in range(self._block_count)]
        self._blocks: dict[int, list[int]] = {}
        self._block_st: dict[int, list[list[int]]] = {}

        # O(n) preprocessing
        self._A_st = self._linear_preprocessing()

    def __call__(self, a: NodeType, b: NodeType) -> TreeNode:
        """Last common ancestor of two nodes.

        Args:
            a: A node or its label in the tree corresponding to this LCA instance.
            b: A node or its label in the tree corresponding to this LCA instance.

        Returns:
            The last common ancestor of a and b.
        """
        return self._get_lca(self._label_to_treenode(a), self._label_to_treenode(b))

    def get(self, a: NodeType, b: NodeType) -> TreeNode:
        """Last common ancestor of two nodes.

        Args:
            a: A node or its label in the tree corresponding to this LCA instance.
            b: A node or its label in the tree corresponding to this LCA instance.

        Returns:
            The last common ancestor of a and b.
        """
        return self._get_lca(self._label_to_treenode(a), self._label_to_treenode(b))

    def displays_triple(self, a: NodeType, b: NodeType, c: NodeType) -> bool:
        """Return whether the tree displays the rooted triple ab|c (= ba|c).

        Args:
            a: A node or its label in the tree corresponding to this LCA instance.
            b: A node or its label in the tree corresponding to this LCA instance.
            c: A node or its label in the tree corresponding to this LCA instance.

        Returns:
            True if the tree displays the triple ab|c (= ba|c); False otherwise.
        """
        try:
            return self._has_triple(
                self._label_to_treenode(a), self._label_to_treenode(b), self._label_to_treenode(c)
            )
        except KeyError:
            return False

    def are_comparable(self, u: NodeOrEdge, v: NodeOrEdge) -> bool:
        """Returns True if two nodes/edges are comparable in the tree.

        Two nodes/edges are comparable if one lies on the unique path from the other to the root of
        the tree.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u and v are comparable in the tree, else False.
        """
        return self._are_comparable(self._label_to_treenode(u), self._label_to_treenode(v))

    def ancestor_or_equal(self, u: NodeOrEdge, v: NodeOrEdge) -> bool:
        """Return True if u is equal to or an ancestor of v.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u is equal or an ancestor of v, else False.
        """
        return self._ancestor_or_equal(self._label_to_treenode(u), self._label_to_treenode(v))

    def ancestor_not_equal(self, u: NodeOrEdge, v: NodeOrEdge) -> bool:
        """Return True if u is a strict ancestor of v.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u is a strict ancestor of v, else False.
        """
        u = self._label_to_treenode(u)
        v = self._label_to_treenode(v)

        return u != v and self._ancestor_or_equal(u, v)

    def descendant_or_equal(self, u: NodeOrEdge, v: NodeOrEdge) -> bool:
        """Return True if u is equal to or a descendant of v.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u is equal or a descendant of v, else False.
        """
        return self.ancestor_or_equal(v, u)

    def descendant_not_equal(self, u: NodeOrEdge, v: NodeOrEdge) -> bool:
        """Return True if u is a strict descendant of v.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u is a strict descendant of v, else False.
        """
        return self.ancestor_not_equal(v, u)

    def consistent_triples(self, triples: Iterable[Triple]) -> list[Triple]:
        """List with the subset of triples that are displayed by the tree.

        Args:
            triples: An iterable of triples of which each may or may not be displayed by the tree.

        Returns:
            A list containing the subset of the input list that are displayed by the tree.
        """
        return [t for t in triples if self.displays_triple(*t)]

    def consistent_triple_generator(self, triples: Iterable[Triple]) -> Iterator[Triple]:
        """Generator for the items in 'triples' that are displayed.

        Args:
            triples: An iterable of triples of which each may or may not be displayed by the tree.

        Yields:
            The triples in the input list that is displayed by the tree.
        """
        for t in triples:
            if self.displays_triple(*t):
                yield t

    def _precompute_logs(self, x: int) -> list[int]:
        """Efficiently pre-compute the ceil(log2(x)) values.

        Args:
            x: The highest integer for which to pre-compute ceil(log2(x)).

        Returns:
            A list of computed ceil(log2(x)) values.
        """
        log_2 = [0 for _ in range(x + 1)]
        log_2[0] = -1
        for i in range(1, x + 1):
            log_2[i] = int(log_2[i // 2]) + 1

        return log_2

    def _linear_preprocessing(self) -> list[list[int]]:
        """Run the O(n) preprocessing.

        Returns:
            The range minimum query (RQM) sparse table.
        """
        n = len(self._L)

        for b in range(self._block_count):
            i = self._block_size * b
            j = 0
            block = [0 for _ in range(self._block_size)]

            current_min = self._L[i]
            self._A[b] = current_min
            self._B[b] = i

            for j in range(1, self._block_size):
                i += 1
                if i < n and self._L[i] < current_min:
                    current_min = self._L[i]
                    self._A[b] = current_min
                    self._B[b] = i
                if i >= n or self._L[i - 1] < self._L[i]:
                    block[j] = block[j - 1] + 1
                    self._block_identifier[b] += 1 << (j - 1)
                else:
                    block[j] = block[j - 1] - 1

            # precompute the corresponding RMQ sparse table if new block type is encountered
            b_id = self._block_identifier[b]
            if b_id not in self._blocks:
                self._blocks[b_id] = block
                self._block_st[b_id] = self._rmq_sparse_table(block)

        # precompute the RMQ sparse table for array A
        return self._rmq_sparse_table(self._A)

    def _rmq_sparse_table(self, A: list[int]) -> list[list[int]]:
        """Compute the RMQ sparse table for an array A.

        Args:
            A: The array for which to compute the RMQ sparse table.

        Returns:
            The RMQ sparse table for array A.
        """
        n = len(A)

        k = self._log_2[n]

        # sparse table for look up
        st = [[0 for _ in range(k + 1)] for _ in range(n)]

        # initialize the intervals with length 1
        for i in range(n):
            st[i][0] = i

        # dynamic programming: compute values from smaller to bigger intervals
        for j in range(1, k + 1):
            # compute minimum value for all intervals with size 2^j
            for i in range(n - (1 << j) + 1):
                if A[st[i][j - 1]] < A[st[i + (1 << (j - 1))][j - 1]]:
                    st[i][j] = st[i][j - 1]
                else:
                    st[i][j] = st[i + (1 << (j - 1))][j - 1]

        return st

    def _sparse_table_query(self, L: list[int], st: list[list[int]], i: int, j: int) -> int:
        """Sparse table query."""
        k = self._log_2[j - i + 1]
        if L[st[i][k]] < L[st[j - (1 << k) + 1][k]]:
            return st[i][k]
        else:
            return st[j - (1 << k) + 1][k]

    def _block_query(self, b: int, left: int, right: int) -> int:
        """Block query."""
        b_id = self._block_identifier[b]
        return (
            self._sparse_table_query(self._blocks[b_id], self._block_st[b_id], left, right)
            + b * self._block_size
        )

    def _rmq_query(self, i: int, j: int) -> int:
        """RQM query."""
        b_i = i // self._block_size
        b_j = j // self._block_size

        if b_i == b_j:
            return self._block_query(b_i, i % self._block_size, j % self._block_size)

        pos1 = self._block_query(b_i, i % self._block_size, self._block_size - 1)
        pos2 = self._block_query(b_j, 0, j % self._block_size)
        pos = pos1 if self._L[pos1] < self._L[pos2] else pos2

        if b_i + 1 < b_j:
            b = self._sparse_table_query(self._A, self._A_st, b_i + 1, b_j - 1)
            pos3 = self._B[b]
            pos = pos if self._L[pos] < self._L[pos3] else pos3

        return pos

    def _label_to_treenode(self, v: NodeOrEdge) -> TreeNode | tuple[TreeNode, TreeNode]:
        """Map labels of a node or edge to the corresponding TreeNode instance(s).

        Args:
            v: A node or edge represented by the TreeNode instances themselves or the node labels.

        Returns:
            The corresponding TreeNode instance or, in case of an edge, tuple of TreeNode instances.
        """
        if isinstance(v, TreeNode):
            return v
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            return (self._label_to_treenode(v[0]), self._label_to_treenode(v[1]))
        else:
            return self._label_dict[v]

    def _get_lca(self, v1: TreeNode, v2: TreeNode) -> TreeNode:
        """Get the last common ancestor of two tree nodes in the tree.

        Args:
            v1: A node in the tree corresponding to this LCA instance.
            v2: A node in the tree corresponding to this LCA instance.

        Returns:
            The last common ancestor of v1 and v2.
        """
        if v1 is v2:
            return v1

        r1 = self._R[self._index[v1]]
        r2 = self._R[self._index[v2]]
        if r1 > r2:
            r1, r2 = r2, r1

        return self._V[self._euler_tour[self._rmq_query(r1, r2)]]

    def _has_triple(self, a: TreeNode, b: TreeNode, c: TreeNode) -> bool:
        """Return whether the tree displays the rooted triple ab|c (= ba|c).

        Args:
            a: A node in the tree corresponding to this LCA instance.
            b: A node in the tree corresponding to this LCA instance.
            c: A node in the tree corresponding to this LCA instance.

        Returns:
            True if the tree displays the triple ab|c (= ba|c); False otherwise.
        """
        if a is b:
            return False
        lca_ab = self._get_lca(a, b)

        return lca_ab is not self._get_lca(lca_ab, c)

    def _are_comparable(
        self,
        u: TreeNode | tuple[TreeNode, TreeNode],
        v: TreeNode | tuple[TreeNode, TreeNode],
    ) -> bool:
        """Returns True if two nodes/edges are comparable in the tree.

        Two nodes/edges are comparable if one lies on the unique path from the other to the root of
        the tree.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u and v are comparable in the tree, else False.
        """
        return self._ancestor_or_equal(u, v) or self._ancestor_or_equal(v, u)

    def _ancestor_or_equal(
        self,
        u: TreeNode | tuple[TreeNode, TreeNode],
        v: TreeNode | tuple[TreeNode, TreeNode],
    ) -> bool:
        """Return True if u is equal to or an ancestor of v.

        Args:
            u: A node or edge in the tree corresponding to this LCA instance.
            v: A node or edge in the tree corresponding to this LCA instance.

        Returns:
            True if u is equal or an ancestor of v, else False.
        """
        # both are nodes
        if isinstance(u, TreeNode) and isinstance(v, TreeNode):
            return u is self._get_lca(u, v)

        # u node, v edge
        elif isinstance(u, TreeNode) and isinstance(v, tuple):
            return u is self._get_lca(u, v[0])

        # u edge, v node
        elif isinstance(u, tuple) and isinstance(v, TreeNode):
            return u[1] is self._get_lca(u[1], v)

        # both are edges
        elif isinstance(u, tuple) and isinstance(v, tuple):
            return u[1] is v[1] or u[1] is self._get_lca(u[1], v[0])
