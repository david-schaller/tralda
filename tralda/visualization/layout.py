"""Tree layout computation for visualization.

This module separates layout (position assignment) from rendering. A :class:`TreeLayout` instance
computes all geometric information for a tree once and can then be consumed by any renderer
(matplotlib, plotly, etc.).

Internally, two canonical coordinates are assigned to every node:

* *depth* — distance from the root along the primary axis (the x-axis in horizontal mode).
* *leaf_rank* — fractional position along the perpendicular axis (the y-axis in horizontal mode).
  Leaves receive integer ranks ``0, 1, …, n_leaves - 1``; internal nodes receive a rank derived
  from their children according to the chosen :class:`NodeRankMode`.

After construction, the final screen-space ``(x, y)`` coordinates for the chosen :class:`LayoutMode`
are available in :attr:`TreeLayout.positions`.
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Literal

from tralda.datastructures.tree import Tree
from tralda.datastructures.tree import TreeNode


class EdgeLengthMode(Enum):
    """How edge lengths are determined along the depth axis.

    Attributes:
        ATTR: Use a node attribute for the length of the edge to its parent.
        UNIFORM: Every edge has length 1. Leaves may end at different depths.
        EVEN: Edges are scaled so that all leaves end at the same depth (cladogram style). Each
            internal node is placed by distributing the remaining depth evenly among the remaining
            edges on the deepest root-to-leaf path through that node.
        RANK: Internal nodes are placed at their topological depth (number of edges from the root).
            Leaves are extended to the maximum depth so they are all aligned.
    """

    ATTR = "attr"
    UNIFORM = "uniform"
    EVEN = "even"
    RANK = "rank"


class NodeRankMode(Enum):
    """How the perpendicular rank of an internal node is derived from its children's ranks.

    Attributes:
        MEAN: The node is placed at the mean rank of its first and last child (default).
        FIRST: The node is placed at the rank of its first (leftmost) child.
        LAST: The node is placed at the rank of its last (rightmost) child.
    """

    MEAN = "mean"
    FIRST = "first"
    LAST = "last"


class LayoutMode(Enum):
    """Orientation of the tree drawing.

    Attributes:
        HORIZONTAL: Root on the left, leaves on the right.
        VERTICAL: Root at the top, leaves at the bottom.
        CIRCULAR: Root at the centre, leaves on the outside.
    """

    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"
    CIRCULAR = "circular"


class TreeLayout:
    """Computes and stores the geometric layout of a tree for visualization.

    Depths and leaf ranks are computed once during construction. The final screen-space positions
    are derived from these canonical coordinates according to the chosen :class:`LayoutMode` and
    stored in :attr:`positions`. Calling :meth:`compute_positions` again with a different
    :class:`LayoutMode` re-derives the positions without repeating the depth computation.

    Attributes:
        tree (Tree): The tree being laid out.
        edge_length_mode (EdgeLengthMode): The edge-length mode used during construction.
        edge_length_attr (str): The node attribute read as edge length in ``ATTR`` mode.
        layout_mode (LayoutMode): The current layout orientation.
        node_rank_mode (NodeRankMode): How the rank of internal nodes is derived.
        depths (dict[TreeNode, float]): Canonical depth of every node from the root.
        leaf_ranks (dict[TreeNode, float]): Perpendicular rank of every node. Leaves receive
            integer ranks ``0, 1, …``; internal nodes receive a rank derived from their children
            according to :attr:`node_rank_mode`.
        max_depth (float): Maximum depth value (depth at which the outermost leaves sit).
        leaf_count (int): Total number of leaves.
        positions (dict[TreeNode, tuple[float, float]]): Final ``(x, y)`` screen-space coordinates.
        label_angle (dict[TreeNode, float]): Suggested text rotation in degrees for each leaf.
            0 = horizontal text; positive values rotate counter-clockwise.
        label_ha (dict[TreeNode, str]): Suggested ``horizontalalignment`` for each leaf label.
        label_va (dict[TreeNode, str]): Suggested ``verticalalignment`` for each leaf label.
    """

    def __init__(
        self,
        tree: Tree,
        edge_length_mode: EdgeLengthMode
        | Literal["attr", "uniform", "even", "rank"] = EdgeLengthMode.ATTR,
        edge_length_attr: str = "dist",
        layout_mode: LayoutMode
        | Literal["horizontal", "vertical", "circular"] = LayoutMode.HORIZONTAL,
        node_rank_mode: NodeRankMode | Literal["mean", "first", "last"] = NodeRankMode.MEAN,
    ) -> None:
        """Construct a layout for the given tree.

        Args:
            tree: The tree to lay out.
            edge_length_mode: How edge lengths along the depth axis are determined. Accepts an
                :class:`EdgeLengthMode` member or its string value (``"attr"``, ``"uniform"``,
                ``"even"``, ``"rank"``).
            edge_length_attr: Name of the :class:`~tralda.datastructures.TreeNode` attribute used
                as the edge length when *edge_length_mode* is ``ATTR``. Default is ``"dist"``.
            layout_mode: Orientation of the resulting drawing. Accepts a :class:`LayoutMode` member
                or its string value (``"horizontal"``, ``"vertical"``, ``"circular"``).
            node_rank_mode: How the perpendicular rank of internal nodes is derived from their
                children. Accepts a :class:`NodeRankMode` member or its string value
                (``"mean"``, ``"first"``, ``"last"``).
        """
        self.tree = tree
        self.edge_length_mode: EdgeLengthMode = (
            EdgeLengthMode(edge_length_mode)
            if not isinstance(edge_length_mode, EdgeLengthMode)
            else edge_length_mode
        )
        self.edge_length_attr = edge_length_attr
        self.layout_mode: LayoutMode = (
            LayoutMode(layout_mode) if not isinstance(layout_mode, LayoutMode) else layout_mode
        )
        self.node_rank_mode: NodeRankMode = (
            NodeRankMode(node_rank_mode)
            if not isinstance(node_rank_mode, NodeRankMode)
            else node_rank_mode
        )

        self.depths: dict[TreeNode, float] = {}
        self.leaf_ranks: dict[TreeNode, float] = {}
        self.max_depth: float = 0.0
        self.leaf_count: int = 0

        self.positions: dict[TreeNode, tuple[float, float]] = {}
        self.label_angle: dict[TreeNode, float] = {}
        self.label_ha: dict[TreeNode, str] = {}
        self.label_va: dict[TreeNode, str] = {}

        self._compute()

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def parent_edge(self, node: TreeNode) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return the screen-space endpoints of the edge from a node to its parent.

        For ``CIRCULAR`` mode the returned coordinates are straight-line Cartesian endpoints;
        renderers are responsible for drawing the elbow (radial segment + arc).

        Args:
            node: The node whose parent edge is requested.

        Returns:
            A tuple ``(parent_pos, node_pos)`` of ``(x, y)`` coordinate pairs, or ``None`` if
                *node* is the root.
        """
        if node.parent is None:
            return None

        return self.positions[node.parent], self.positions[node]

    def children_connector(
        self, node: TreeNode
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return the screen-space endpoints of the connector between a node's outermost children.

        In ``HORIZONTAL`` mode this is a vertical line segment; in ``VERTICAL`` mode a horizontal
        segment. In ``CIRCULAR`` mode the two endpoints on the arc at the parent radius are
        returned — renderers should draw a circular arc rather than a straight line.

        Args:
            node: The internal node whose child connector is requested.

        Returns:
            A tuple ``(start, end)`` of ``(x, y)`` coordinate pairs, or ``None`` if *node* is a
            leaf.
        """
        if not node.children:
            return None

        first = list(node.children)[0]
        last = list(node.children)[-1]

        if self.layout_mode is LayoutMode.HORIZONTAL:
            x = self.positions[node][0]
            return (x, self.positions[first][1]), (x, self.positions[last][1])
        elif self.layout_mode is LayoutMode.VERTICAL:
            y = self.positions[node][1]
            return (self.positions[first][0], y), (self.positions[last][0], y)
        else:  # CIRCULAR — arc endpoints at parent radius
            r = self.depths[node]
            theta_first = self._theta(self.leaf_ranks[first])
            theta_last = self._theta(self.leaf_ranks[last])
            return (
                (r * math.cos(theta_first), r * math.sin(theta_first)),
                (r * math.cos(theta_last), r * math.sin(theta_last)),
            )

    def ghost_segment(
        self, node: TreeNode
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Return the screen-space endpoints of the ghost (leaf-extension) segment for a node.

        A ghost segment is a dashed or otherwise styled line that extends a short-branched leaf
        to the maximum depth so that all leaves are visually aligned. It is only relevant for edge
        length modes where leaves can end at different depths (``ATTR`` or ``UNIFORM``). It is the
        responsibility of renderers to decide whether to draw a ghost segment.

        Args:
            node: The node for which the ghost segment is requested.

        Returns:
            A tuple ``(start, end)`` of ``(x, y)`` coordinate pairs representing the extension from
                the actual leaf position to the maximum depth, or ``None`` if *node* is not a leaf,
                or the leaf is already at :attr:`max_depth`.
        """
        if node.children:
            return None
        if self.max_depth - self.depths[node] < 1e-12:
            return None

        if self.layout_mode is LayoutMode.HORIZONTAL:
            x0, y = self.positions[node]
            return (x0, y), (self.max_depth, y)
        elif self.layout_mode is LayoutMode.VERTICAL:
            x, y0 = self.positions[node]
            return (x, y0), (x, self.max_depth)
        else:  # CIRCULAR
            theta = self._theta(self.leaf_ranks[node])
            r0 = self.depths[node]
            r1 = self.max_depth
            return (
                (r0 * math.cos(theta), r0 * math.sin(theta)),
                (r1 * math.cos(theta), r1 * math.sin(theta)),
            )

    def compute_positions(
        self,
        layout_mode: LayoutMode | Literal["horizontal", "vertical", "circular"] | None = None,
    ) -> None:
        """Transform canonical coordinates into screen-space positions.

        Called automatically during construction. Can be called again to switch layout modes
        without recomputing depths and leaf ranks.

        Args:
            layout_mode: Override the layout mode set during construction. Accepts a
                :class:`LayoutMode` member or its string value. If ``None``, the instance's
                current :attr:`layout_mode` is used.
        """
        if layout_mode is not None:
            self.layout_mode = (
                LayoutMode(layout_mode) if not isinstance(layout_mode, LayoutMode) else layout_mode
            )

        if self.layout_mode is LayoutMode.HORIZONTAL:
            self._positions_horizontal()
        elif self.layout_mode is LayoutMode.VERTICAL:
            self._positions_vertical()
        elif self.layout_mode is LayoutMode.CIRCULAR:
            self._positions_circular()

    # ------------------------------------------------------------------
    # Internal computation
    # ------------------------------------------------------------------

    def _compute(self) -> None:
        """Run the full layout pipeline."""
        if not self.tree.root:
            return

        self._compute_depths()
        self._compute_leaf_ranks()
        self.compute_positions()

    def _compute_depths(self) -> None:
        """Dispatch depth computation to the method matching :attr:`edge_length_mode`."""
        mode = self.edge_length_mode

        if mode is EdgeLengthMode.ATTR:
            self._depths_from_attr()
        elif mode is EdgeLengthMode.UNIFORM:
            self._depths_uniform()
        elif mode is EdgeLengthMode.EVEN:
            self._depths_even()
        elif mode is EdgeLengthMode.RANK:
            self._depths_rank()

    def _depths_from_attr(self) -> None:
        """Compute depths using a per-node attribute as the edge length."""
        attr = self.edge_length_attr
        for v in self.tree.preorder():
            if v.parent is None:
                self.depths[v] = 0.0
            else:
                self.depths[v] = self.depths[v.parent] + float(getattr(v, attr, 0.0))

        self.max_depth = max(self.depths.values()) if self.depths else 0.0

    def _depths_uniform(self) -> None:
        """Compute depths with every edge having length 1."""
        for v in self.tree.preorder():
            if v.parent is None:
                self.depths[v] = 0.0
            else:
                self.depths[v] = self.depths[v.parent] + 1.0

        self.max_depth = max(self.depths.values()) if self.depths else 0.0

    def _depths_even(self) -> None:
        """Compute depths so that all leaves end at depth 1 (cladogram style).

        For each internal node *v* with parent at depth *d*, the assigned depth is::

            depth[v] = d + (1 - d) / (subtree_height[v] + 1)

        where ``subtree_height[v]`` is the number of edges to the deepest leaf in *v*'s subtree.
        This distributes the remaining depth evenly among the remaining edges on the deepest
        root-to-leaf path through *v*, guaranteeing that every leaf reaches depth 1.
        """
        # Pass 1 (post-order): height of each subtree (edges to the deepest leaf).
        subtree_height: dict[TreeNode, int] = {}
        for v in self.tree.postorder():
            if not v.children:
                subtree_height[v] = 0
            else:
                subtree_height[v] = 1 + max(subtree_height[c] for c in v.children)

        # Pass 2 (pre-order): assign depths.
        for v in self.tree.preorder():
            if v.parent is None:
                self.depths[v] = 0.0
            elif subtree_height[v] == 0:
                self.depths[v] = 1.0
            else:
                self.depths[v] = self.depths[v.parent] + (1.0 - self.depths[v.parent]) / (
                    subtree_height[v] + 1
                )

        self.max_depth = max(self.depths.values()) if self.depths else 0.0

    def _depths_rank(self) -> None:
        """Compute depths using topological depth; extend leaves to the maximum depth."""
        topo_max = 0.0
        for v in self.tree.preorder():
            if v.parent is None:
                self.depths[v] = 0.0
            else:
                self.depths[v] = self.depths[v.parent] + 1.0
            if self.depths[v] > topo_max:
                topo_max = self.depths[v]

        for v in self.tree.leaves():
            self.depths[v] = topo_max

        self.max_depth = topo_max

    def _compute_leaf_ranks(self) -> None:
        """Assign perpendicular ranks to every node.

        Leaves receive consecutive integer ranks in left-to-right order. The rank of an internal
        node is determined by :attr:`node_rank_mode`.
        """
        leaf_index = 0
        for v in self.tree.postorder():
            if not v.children:
                self.leaf_ranks[v] = float(leaf_index)
                leaf_index += 1
            else:
                first = list(v.children)[0]
                last = list(v.children)[-1]
                if self.node_rank_mode is NodeRankMode.MEAN:
                    self.leaf_ranks[v] = (self.leaf_ranks[first] + self.leaf_ranks[last]) / 2.0
                elif self.node_rank_mode is NodeRankMode.FIRST:
                    self.leaf_ranks[v] = self.leaf_ranks[first]
                else:  # LAST
                    self.leaf_ranks[v] = self.leaf_ranks[last]

        self.leaf_count = leaf_index

    def _positions_horizontal(self) -> None:
        """Set positions for horizontal layout (root left, leaves right)."""
        for v in self.tree.preorder():
            self.positions[v] = (self.depths[v], self.leaf_ranks[v])

        for v in self.tree.leaves():
            self.label_angle[v] = 0.0
            self.label_ha[v] = "left"
            self.label_va[v] = "center"

    def _positions_vertical(self) -> None:
        """Set positions for vertical layout (root top, leaves bottom)."""
        for v in self.tree.preorder():
            self.positions[v] = (self.leaf_ranks[v], self.depths[v])

        for v in self.tree.leaves():
            self.label_angle[v] = -90.0
            self.label_ha[v] = "left"
            self.label_va[v] = "center"

    def _positions_circular(self) -> None:
        """Set positions for circular layout (root centre, leaves outside)."""
        for v in self.tree.preorder():
            r = self.depths[v]
            theta = self._theta(self.leaf_ranks[v])
            self.positions[v] = (r * math.cos(theta), r * math.sin(theta))

        for v in self.tree.leaves():
            theta = self._theta(self.leaf_ranks[v])
            angle_deg = math.degrees(theta)
            # Normalise to (-180, 180].
            angle_deg = (angle_deg + 180.0) % 360.0 - 180.0
            self.label_angle[v] = angle_deg
            # Labels on the right half point rightward; labels on the left half are flipped.
            if -90.0 <= angle_deg <= 90.0:
                self.label_ha[v] = "left"
            else:
                self.label_ha[v] = "right"
            self.label_va[v] = "center"

    def _theta(self, rank: float) -> float:
        """Convert a leaf rank to an angle in radians for circular layout.

        Angles run from 0 to just below 2π, distributed evenly across leaf ranks.

        Args:
            rank: The leaf rank to convert.

        Returns:
            The corresponding angle in radians.
        """
        if self.leaf_count <= 1:
            return 0.0

        return 2.0 * math.pi * rank / self.leaf_count
