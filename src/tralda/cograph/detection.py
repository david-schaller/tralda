"""Linear-time cograph detection.

References:
    .. [1] D. G. Corneil, Y. Perl, and L. K. Stewart. A Linear Recognition Algorithm for Cographs.
           In: SIAM J. Comput., 14(4), 926-934 (1985). DOI: 10.1137/0214065
"""

from collections import defaultdict
from collections import deque
from typing import Any

import networkx as nx

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode


class LinearCographDetector:
    """Linear cograph detection and cotree costruction.

    References:
        .. [1] D. G. Corneil, Y. Perl, and L. K. Stewart. A Linear Recognition Algorithm for
               Cographs. In: SIAM J. Comput., 14(4), 926-934 (1985). DOI: 10.1137/0214065
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Constructor of LinearCographDetector class.

        Args:
            graph: A graph.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("not a NetworkX Graph")

        self.graph = graph
        self.V: list[Any] = [v for v in graph.nodes()]

        self.tree = Tree(None)
        self.already_in_tree: set[Any] = set()
        self.leaf_map: dict[Any, TreeNode] = {}
        self.node_counter = 0

        self.marked: set[TreeNode] = set()
        self.m_u_children: dict[
            TreeNode, list[TreeNode]
        ] = {}  # lists of marked and unmarked children
        self.mark_counter = 0
        self.unmark_counter = 0

        # md is set to zero for all new nodes including leaves as they are added to the tree
        self.md: dict[TreeNode, int] = defaultdict(int)

        self.error_message = ""

    def recognition(self) -> Tree | None:
        """Run cotree construction.

        Function COGRAPH-RECOGNITION from Corneil et al. 1985.

        Returns:
            The cotree representation of the graph, or None if it is not a cograph.
        """
        if len(self.V) == 0:
            raise RuntimeError("empty graph in cograph recognition")
        elif len(self.V) == 1:
            self.tree.root = TreeNode(label=self.V[0])

            return self.tree

        v1, v2 = self.V[0], self.V[1]
        self.already_in_tree.update([v1, v2])

        R = TreeNode(label="series")
        self.tree.root = R

        if self.graph.has_edge(v1, v2):
            v1_node = TreeNode(label=v1)
            v2_node = TreeNode(label=v2)
            R.add_child(v1_node)
            R.add_child(v2_node)
            self.node_counter = 3
        else:
            N = TreeNode(label="parallel")
            R.add_child(N)
            v1_node = TreeNode(label=v1)
            v2_node = TreeNode(label=v2)
            N.add_child(v1_node)
            N.add_child(v2_node)
            self.node_counter = 4

        self.leaf_map[v1] = v1_node
        self.leaf_map[v2] = v2_node

        if len(self.V) == 2:
            self._remove_single_child_root()
            return self.tree

        for x in self.V[2:]:
            # initialization (necessary?)
            self.marked.clear()
            self.m_u_children.clear()
            self.mark_counter = 0
            self.unmark_counter = 0
            self.already_in_tree.add(x)  # add x for subsequent iterations

            self._mark(x)

            # all nodes in T were marked and unmarked
            if self.node_counter == self.unmark_counter:
                R = self.tree.root
                x_node = TreeNode(label=x)
                R.add_child(x_node)
                self.node_counter += 1
                self.leaf_map[x] = x_node
                continue
            # no nodes in T were marked and unmarked
            elif self.mark_counter == 0:
                # d(R)=1
                if len(self.tree.root.children) == 1:
                    N = self.tree.root.children[0]
                    x_node = TreeNode(label=x)
                    N.add_child(x_node)
                    self.node_counter += 1
                else:
                    R_old = self.tree.root
                    R_new = TreeNode(label="series")
                    N = TreeNode(label="parallel")
                    R_new.add_child(N)
                    N.add_child(R_old)
                    self.tree.root = R_new

                    x_node = TreeNode(label=x)
                    N.add_child(x_node)
                    self.node_counter += 3
                self.leaf_map[x] = x_node
                continue

            u = self._find_lowest()
            if not u:
                return

            # label(u)=0 and |A|=1
            if u.label == "parallel" and len(self.m_u_children[u]) == 1:
                w = self.m_u_children[u][0]
                if w.is_leaf():
                    new_node = TreeNode(label="series")
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)

                    x_node = TreeNode(label=x)
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = TreeNode(label=x)
                    w.add_child(x_node)
                    self.node_counter += 1

            # label(u)=1 and |B|=1
            elif u.label == "series" and len(u.children) - len(self.m_u_children[u]) == 1:
                set_A = set(self.m_u_children[u])  # auxiliary set bounded by O(deg(x))
                w = None
                for child in u.children:
                    if child not in set_A:
                        w = child
                        break
                if w.is_leaf():
                    new_node = TreeNode(label="parallel")
                    u.remove_child(w)
                    u.add_child(new_node)
                    new_node.add_child(w)

                    x_node = TreeNode(label=x)
                    new_node.add_child(x_node)
                    self.node_counter += 2
                else:
                    x_node = TreeNode(label=x)
                    w.add_child(x_node)
                    self.node_counter += 1

            else:
                y = TreeNode(label=u.label)
                for a in self.m_u_children[u]:
                    u.remove_child(a)
                    y.add_child(a)

                if u.label == "parallel":
                    new_node = TreeNode(label="series")
                    u.add_child(new_node)

                    new_node.add_child(y)
                    x_node = TreeNode(label=x)
                    new_node.add_child(x_node)
                else:
                    par = u.parent
                    if par is not None:  # u was the root of T
                        par.remove_child(u)
                        par.add_child(y)
                    else:
                        self.tree.root = y  # y becomes the new root

                    new_node = TreeNode(label="parallel")
                    y.add_child(new_node)
                    new_node.add_child(u)
                    x_node = TreeNode(label=x)
                    new_node.add_child(x_node)
                self.node_counter += 3

            self.leaf_map[x] = x_node

        self._remove_single_child_root()

        return self.tree

    def _mark(self, x: Any) -> None:
        """Function MARK from Corneil et al. 1985.

        Args:
            x: A node in the input graph.
        """
        for v in self.graph.neighbors(x):
            if v in self.already_in_tree:
                self.marked.add(self.leaf_map[v])
                self.mark_counter += 1

        queue = deque(self.marked)

        while queue:  # contains only d(u)=md(u) nodes
            u = queue.popleft()
            self.marked.remove(u)  # unmark u
            self.unmark_counter += 1
            self.md[u] = 0  # md(u) <- 0
            if u is not self.tree.root:
                w = u.parent  # w <- parent(u)
                if w not in self.marked:
                    self.marked.add(w)  # mark w
                    self.mark_counter += 1
                self.md[w] += 1
                if self.md[w] == len(w.children):
                    queue.append(w)

                # append u to list of marked and unmarked children of w
                if w in self.m_u_children:
                    self.m_u_children[w].appendleft(u)
                else:
                    self.m_u_children[w] = deque([u])

        if (
            self.marked  # any vertex remained marked
            and len(self.tree.root.children) == 1
            and self.tree.root not in self.marked
        ):
            self.marked.add(self.tree.root)
            self.mark_counter += 1

    def _find_lowest(self):
        """Function FIND-LOWEST from Corneil et al. 1985.

        This function checks whether G+x is a cograph and, if so, returns u, the lowest marked
        vertex of T.

        Returns:
            The lowest marked vertex in the current tree.
        """
        R = self.tree.root
        y = "Lambda"

        if R not in self.marked:  # R is not marked
            self.error_message = "(iii): R={}".format(R)
            return  # G+x is not a cograph (iii)
        else:
            if self.md[R] != len(R.children) - 1:
                y = R
            self.marked.remove(R)
            self.md[R] = 0
            u = w = R

        # while there are marked vertices choose a arbitrary marked vertex u
        while self.marked:
            u = self.marked.pop()

            if y != "Lambda":
                self.error_message = "(i) or (ii): y={}".format(y)
                return  # G+x is not a cograph (i) or (ii)

            if u.label == "series":
                if self.md[u] != len(u.children) - 1:
                    y = u
                if u.parent in self.marked:
                    self.error_message = "(i) and (vi): u={}".format(u)
                    return  # G+x is not a cograph (i) and (vi)
                else:
                    t = u.parent.parent
            else:
                y = u
                t = u.parent
            self.md[u] = 0  # u was already unmarked above

            # check if the u-w path is part of the legitimate alternating path
            while t is not w:
                if t is R:
                    self.error_message = "(iv): t={}".format(t)
                    return  # G+x is not a cograph (iv)

                if t not in self.marked:
                    self.error_message = "(iii), (v) or (vi): t={}".format(t)
                    return  # G+x is not a cograph (iii), (v) or (vi)

                if self.md[t] != len(t.children) - 1:
                    self.error_message = "(ii): t={}".format(t)
                    return  # G+x is not a cograph (ii)

                if t.parent in self.marked:
                    self.error_message = "(i): t={}".format(t)
                    return  # G+x is not a cograph (i)

                self.marked.remove(t)  # unmark t
                self.md[t] = 0  # reset md(t)
                t = t.parent.parent

            w = u  # rest w for next choice of marked vertex

        return u

    def _remove_single_child_root(self):
        """If necessary, remove the root that has only one child v and make v the new root."""
        if len(self.tree.root.children) == 1:
            new_root = self.tree.root.children[0]
            new_root.detach()
            self.tree.root = new_root
