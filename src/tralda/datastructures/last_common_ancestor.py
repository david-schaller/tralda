"""Efficient computation of last/least common ancestors in trees."""

from __future__ import annotations

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode


class LCA:
    """Compute last common ancestors in a tree efficiently.

    Uses a reduction to a +/-1 Range minimum query (RMQ) problem and a sparse
    table implementation.
    Preprocessing complexity: O(n)
    Query complexity: O(1)
    where n is the number of nodes in the tree.

    References
    ----------
    .. [1] M. A. Bender, M. Farach-Colton, G. Pemmasani, S. Skiena, P. Sumazin.
       Lowest common ancestors in trees and directed acyclic graphs.
       In: Journal of Algorithms. 57, Nr. 2, November 2005, S. 75–94.
       ISSN 0196-6774. doi:10.1016/j.jalgor.2005.08.001.
    .. [2] https://cp-algorithms.com/data_structures/sparse-table.html
    .. [3] https://cp-algorithms.com/graph/lca_farachcoltonbender.html
    """

    def __init__(self, tree):
        """Constructor for class LCA.

        Parameters
        ----------
        tree : Tree
            The Tree instance for which this instance will allow efficient
            last common ancestor queries.

        Raises
        ------
        TypeError
            If `tree` is not a Tree instance.
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

        j = 0
        for v, level in self._tree.euler_and_level():
            i = self._index[v]
            self._euler_tour.append(i)
            self._L.append(level)
            if self._R[i] is None:
                self._R[i] = j
            j += 1

        # build sparse table for range minimum query (RMQ)
        self._precompute_logs(len(self._L))

        # # O(n log n)-preprocessing version
        # self._st = self._RMQ_sparse_table(self._L)

        # O(n) preprocessing
        self._linear_preprocessing()

    def __call__(self, a, b):
        """Last common ancestor of two nodes.

        Parameters
        ----------
        a : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.
        b : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.

        Returns
        -------
        TreeNode
            The last common ancestor of `a` and `b`.
        """

        return self._get_lca(self._label_to_treenode(a), self._label_to_treenode(b))

    def get(self, a, b):
        """Last common ancestor of two nodes.

        Parameters
        ----------
        a : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.
        b : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.

        Returns
        -------
        TreeNode
            The last common ancestor of `a` and `b`.
        """

        return self._get_lca(self._label_to_treenode(a), self._label_to_treenode(b))

    def displays_triple(self, a, b, c):
        """Return whether the tree displays the rooted triple ab|c (= ba|c).

        Parameters
        ----------
        a : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.
        b : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.
        c : TreeNode or int
            A node or its label in the tree corresponding to this LCA instance.

        Returns
        -------
        bool
            True if the tree displays the triple ab|c (= ba|c).
        """

        try:
            return self._has_triple(
                self._label_to_treenode(a), self._label_to_treenode(b), self._label_to_treenode(c)
            )
        except KeyError:
            return False

    def are_comparable(self, u, v):
        """Returns True if two nodes/edges are comparable in the tree.

        Two nodes/edges are comparable if one lies on the unique path from the
        other to the root of the tree.

        Parameters
        ----------
        u : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.
        v : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.

        Return
        ------
        bool
            True if `u` and `v` are comparable in the tree, else False.
        """

        return self._are_comparable(self._label_to_treenode(u), self._label_to_treenode(v))

    def ancestor_or_equal(self, u, v):
        """Return True if u is equal to or an ancestor of v.

        Parameters
        ----------
        u : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.
        v : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.

        Return
        ------
        bool
            True if `u` is equal or an ancestor of `v`, else False.
        """

        return self._ancestor_or_equal(self._label_to_treenode(u), self._label_to_treenode(v))

    def ancestor_not_equal(self, u, v):
        """Return True if u is a strict ancestor of v.

        Parameters
        ----------
        u : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.
        v : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.

        Return
        ------
        bool
            True if `u` is a strict ancestor of `v`, else False.
        """

        u = self._label_to_treenode(u)
        v = self._label_to_treenode(v)

        return u != v and self._ancestor_or_equal(u, v)

    def descendant_or_equal(self, u, v):
        """Return True if u is equal to or a descendant of v.

        Parameters
        ----------
        u : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.
        v : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.

        Return
        ------
        bool
            True if `u` is equal or a descendant of `v`, else False.
        """

        return self.ancestor_or_equal(v, u)

    def descendant_not_equal(self, u, v):
        """Return True if u is a strict descendant of v.

        Parameters
        ----------
        u : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.
        v : TreeNode or int or tuple of two TreeNode or int objects
            An node or edge in the tree corresponding to this LCA instance.

        Return
        ------
        bool
            True if `u` is a strict descendant of `v`, else False.
        """

        return self.ancestor_not_equal(v, u)

    def consistent_triples(self, triples):
        """List with the subset of triples that are displayed by the tree.

        Parameters
        ----------
        triples : an iterable object of tuples of three TreeNode or int objects
            Input triples of which each may or may not be displayed by the tree.

        Returns
        -------
        list of tuples of three TreeNode of int objects
            Representing the subset of the input list that are displayed by the
            tree.
        """

        return [t for t in triples if self.displays_triple(*t)]

    def consistent_triple_generator(self, triples):
        """Generator for the items in 'triples' that are displayed.

        Parameters
        ----------
        triples : an iterable object of tuples of three TreeNode or int objects
            Input triples of which each may or may not be displayed by the tree.

        Yields
        -------
        tuple of three TreeNode of int objects
            For each triple in the input list that is displayed by the tree.
        """

        for t in triples:
            if self.displays_triple(*t):
                yield t

    def _precompute_logs(self, x):
        self.log_2 = [0 for _ in range(x + 1)]
        self.log_2[0] = -1
        for i in range(1, x + 1):
            self.log_2[i] = int(self.log_2[i // 2]) + 1

    def _linear_preprocessing(self):
        n = len(self._L)
        self.block_size = max(1, self.log_2[n] // 2)
        self.block_count = (n + self.block_size - 1) // self.block_size

        self.A = [None for _ in range(self.block_count)]
        self.B = [None for _ in range(self.block_count)]
        self.block_identifier = [0 for _ in range(self.block_count)]
        self.blocks = {}
        self.block_st = {}

        for b in range(self.block_count):
            i = self.block_size * b
            j = 0
            block = [0 for _ in range(self.block_size)]

            current_min = self._L[i]
            self.A[b] = current_min
            self.B[b] = i

            for j in range(1, self.block_size):
                i += 1
                if i < n and self._L[i] < current_min:
                    current_min = self._L[i]
                    self.A[b] = current_min
                    self.B[b] = i
                if i >= n or self._L[i - 1] < self._L[i]:
                    block[j] = block[j - 1] + 1
                    self.block_identifier[b] += 1 << (j - 1)
                else:
                    block[j] = block[j - 1] - 1

            # precompute the corresponding RMQ sparse table if new block type
            # is encountered
            b_id = self.block_identifier[b]
            if b_id not in self.blocks:
                self.blocks[b_id] = block
                self.block_st[b_id] = self._RMQ_sparse_table(block)

        # precompute the RMQ sparse table for array A
        self.A_st = self._RMQ_sparse_table(self.A)

    def _RMQ_sparse_table(self, A):
        n = len(A)
        # self._precompute_logs(n)

        K = self.log_2[n]

        # sparse table for look up
        st = [[0 for j in range(K + 1)] for i in range(n)]

        # initialize the intervals with length 1
        for i in range(n):
            st[i][0] = i

        # dynamic programming: compute values from smaller to bigger intervals
        for j in range(1, K + 1):
            # compute minimum value for all intervals with size 2^j
            for i in range(n - (1 << j) + 1):
                if A[st[i][j - 1]] < A[st[i + (1 << (j - 1))][j - 1]]:
                    st[i][j] = st[i][j - 1]
                else:
                    st[i][j] = st[i + (1 << (j - 1))][j - 1]

        return st

    def _sparse_table_query(self, L, st, i, j):
        k = self.log_2[j - i + 1]
        if L[st[i][k]] < L[st[j - (1 << k) + 1][k]]:
            return st[i][k]
        else:
            return st[j - (1 << k) + 1][k]

    def _block_query(self, b, left, right):
        b_id = self.block_identifier[b]
        return (
            self._sparse_table_query(self.blocks[b_id], self.block_st[b_id], left, right)
            + b * self.block_size
        )

    def _RMQ_query(self, i, j):
        b_i = i // self.block_size
        b_j = j // self.block_size

        if b_i == b_j:
            return self._block_query(b_i, i % self.block_size, j % self.block_size)

        pos1 = self._block_query(b_i, i % self.block_size, self.block_size - 1)
        pos2 = self._block_query(b_j, 0, j % self.block_size)
        pos = pos1 if self._L[pos1] < self._L[pos2] else pos2

        if b_i + 1 < b_j:
            b = self._sparse_table_query(self.A, self.A_st, b_i + 1, b_j - 1)
            pos3 = self.B[b]
            pos = pos if self._L[pos] < self._L[pos3] else pos3

        return pos

    def _RMQ_query_OLD(self, i, j):
        # query function if O(n log n)-preprocessing is used

        return self._sparse_table_query(self._L, self._st, i, j)

    def _label_to_treenode(self, v):
        if isinstance(v, TreeNode):
            return v
        elif isinstance(v, (tuple, list)) and len(v) == 2:
            return (self._label_to_treenode(v[0]), self._label_to_treenode(v[1]))
        else:
            return self._label_dict[v]

    def _get_lca(self, v1, v2):
        if v1 is v2:
            return v1

        r1 = self._R[self._index[v1]]
        r2 = self._R[self._index[v2]]
        if r1 > r2:
            r1, r2 = r2, r1

        return self._V[self._euler_tour[self._RMQ_query(r1, r2)]]

    def _has_triple(self, a, b, c):
        if a is b:
            return False
        lca_ab = self._get_lca(a, b)
        return lca_ab is not self._get_lca(lca_ab, c)

    def _are_comparable(self, u, v):
        return self._ancestor_or_equal(u, v) or self._ancestor_or_equal(v, u)

    def _ancestor_or_equal(self, u, v):
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
