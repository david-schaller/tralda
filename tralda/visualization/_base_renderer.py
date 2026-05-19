"""Shared base class and utilities for tree renderers.

This module contains logic that is common to all renderer backends:

* :class:`BaseRenderer` — shared constructor state and pure-Python geometry helpers
  (:meth:`~BaseRenderer._build_positions`, :meth:`~BaseRenderer._rescale_seg`,
  :meth:`~BaseRenderer._ghost_label_anchor`).
* :func:`arc_theta_range` — circular-arc parameter computation used by both the matplotlib and
  Plotly renderers.

Each backend subclasses :class:`BaseRenderer` and adds its own output-specific methods
(``render()``, drawing helpers, figure-sizing, etc.).
"""

from __future__ import annotations

import math

from tralda.datastructures.tree import TreeNode
from tralda.visualization.layout import LayoutMode
from tralda.visualization.layout import NodeRankMode
from tralda.visualization.layout import TreeLayout
from tralda.visualization.style import TreeStyle


class BaseRenderer:
    """Common base for tree renderers.

    Holds the constructor parameters that every renderer shares and provides pure-Python geometry
    helpers that are identical across backends.

    Attributes:
        layout (TreeLayout): The tree layout being rendered.
        tree_style (TreeStyle): Styling configuration.
        rescale_depth (bool): Whether to normalise the depth axis to ``[0, 1]``.
        show_labels (bool): Whether to draw leaf labels.
        show_internal_labels (bool): Whether to draw labels for internal nodes.
        show_ghost_segments (bool): Whether to extend short leaves to the maximum depth.
    """

    def __init__(
        self,
        layout: TreeLayout,
        *,
        tree_style: TreeStyle | None = None,
        rescale_depth: bool = True,
        show_labels: bool = True,
        show_internal_labels: bool = False,
        show_ghost_segments: bool = True,
    ) -> None:
        self.layout = layout
        self.tree_style: TreeStyle = tree_style if tree_style is not None else TreeStyle()
        self.rescale_depth = rescale_depth
        self.show_labels = show_labels
        self.show_internal_labels = show_internal_labels
        self.show_ghost_segments = show_ghost_segments

    # ----------------------------------------------------------------------------------------------
    # Shared geometry helpers
    # ----------------------------------------------------------------------------------------------

    def _build_positions(self) -> dict[TreeNode, tuple[float, float]]:
        """Return layout positions, optionally normalising the depth axis to ``[0, 1]``.

        Returns:
            Mapping from tree nodes to ``(x, y)`` positions in data coordinates.
        """
        positions = dict(self.layout.positions)

        if not self.rescale_depth or self.layout.max_depth == 0.0:
            return positions

        scale = 1.0 / self.layout.max_depth
        mode = self.layout.layout_mode

        if mode is LayoutMode.HORIZONTAL:
            return {v: (x * scale, y) for v, (x, y) in positions.items()}
        elif mode is LayoutMode.VERTICAL:
            return {v: (x, y * scale) for v, (x, y) in positions.items()}
        else:  # CIRCULAR: depth is encoded in the radius — scale uniformly.
            return {v: (x * scale, y * scale) for v, (x, y) in positions.items()}

    def _rescale_seg(
        self,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        mode: LayoutMode,
    ) -> tuple[float, float, float, float]:
        """Apply depth-axis rescaling to a line-segment endpoint pair.

        Used to rescale ghost-segment coordinates before rendering.

        Args:
            x0: Start x-coordinate.
            y0: Start y-coordinate.
            x1: End x-coordinate.
            y1: End y-coordinate.
            mode: Current layout mode.

        Returns:
            Rescaled ``(x0, y0, x1, y1)``.
        """
        if not self.rescale_depth or self.layout.max_depth == 0.0:
            return x0, y0, x1, y1
        scale = 1.0 / self.layout.max_depth
        if mode is LayoutMode.HORIZONTAL:
            return x0 * scale, y0, x1 * scale, y1
        elif mode is LayoutMode.VERTICAL:
            return x0, y0 * scale, x1, y1 * scale
        else:
            return x0 * scale, y0 * scale, x1 * scale, y1 * scale

    def _ghost_label_anchor(
        self,
        v: TreeNode,
        x: float,
        y: float,
        mode: LayoutMode,
    ) -> tuple[float, float]:
        """Return the label anchor position for a node, shifted to the ghost-segment tip if present.

        When a leaf has a ghost segment (because it is shorter than the maximum depth), its label
        should be anchored at the far end of the ghost segment so that all leaf labels are
        horizontally aligned regardless of actual branch length.

        For internal nodes and leaves without a ghost segment the original ``(x, y)`` is returned
        unchanged.

        Args:
            v: The node whose label anchor is requested.
            x: Current x position of the node.
            y: Current y position of the node.
            mode: Current layout mode.

        Returns:
            ``(x, y)`` of the label anchor point.
        """
        seg = self.layout.ghost_segment(v) if v.is_leaf() else None
        if seg is None:
            return x, y

        (_, _), (x1_raw, y1_raw) = seg
        scale = (
            (1.0 / self.layout.max_depth)
            if (self.rescale_depth and self.layout.max_depth > 0)
            else 1.0
        )

        if mode is LayoutMode.HORIZONTAL:
            return x1_raw * scale, y
        elif mode is LayoutMode.VERTICAL:
            return x, y1_raw * scale
        else:  # CIRCULAR
            return x1_raw * scale, y1_raw * scale

    def _parent_edge_segment(
        self,
        vx: float,
        vy: float,
        px: float,
        py: float,
        mode: LayoutMode,
    ) -> tuple[float, float, float, float]:
        """Return ``(x0, y0, x1, y1)`` for the parent-to-child edge segment.

        For HORIZONTAL and VERTICAL layouts this is a single axis-aligned segment.  For CIRCULAR
        layouts the segment starts at the point on the parent's radius that is collinear with the
        origin and the child, so that the connector arc and the radial segment meet exactly.

        Args:
            vx: Child x position.
            vy: Child y position.
            px: Parent x position.
            py: Parent y position.
            mode: Current layout mode.

        Returns:
            Segment endpoints ``(x0, y0, x1, y1)``.
        """
        if mode is LayoutMode.HORIZONTAL:
            return px, vy, vx, vy
        elif mode is LayoutMode.VERTICAL:
            return vx, py, vx, vy

        # CIRCULAR — radial segment at the child's angle
        r_parent = math.hypot(px, py)
        if r_parent > 1e-12:
            theta_v = math.atan2(vy, vx)
            start_x = r_parent * math.cos(theta_v)
            start_y = r_parent * math.sin(theta_v)
        else:
            start_x, start_y = px, py
        return start_x, start_y, vx, vy

    def _child_connector_range(
        self,
        v: TreeNode,
        positions: dict[TreeNode, tuple[float, float]],
    ) -> tuple[list[TreeNode], tuple[float, float], tuple[float, float]]:
        """Return ``(children, first_pos, last_pos)`` for the child connector of an internal node.

        ``first_pos`` and ``last_pos`` define the start and end of the connector bar (axis-aligned
        segment or arc).  When :attr:`~tralda.visualization.layout.NodeRankMode.NODE` rank mode is
        active the connector starts at the parent node's own position rather than the first
        child's position.

        Args:
            v: An internal node (must have at least one child).
            positions: Mapping from nodes to ``(x, y)`` positions.

        Returns:
            A 3-tuple ``(children, first_pos, last_pos)``.
        """
        children = list(v.children)
        if self.layout.node_rank_mode is NodeRankMode.NODE:
            first_pos = positions[v]
        else:
            first_pos = positions[children[0]]
        last_pos = positions[children[-1]]

        return children, first_pos, last_pos


# --------------------------------------------------------------------------------------------------
# Arc utility
# --------------------------------------------------------------------------------------------------


def arc_theta_range(
    px: float,
    py: float,
    first_pos: tuple[float, float],
    last_pos: tuple[float, float],
) -> tuple[float, float, float, int] | None:
    """Compute arc parameters for a circular-layout child connector.

    Both the matplotlib and Plotly renderers draw a circular arc at the parent's radius spanning
    from the angular projection of the first child to that of the last child.  This function
    computes the shared parameters; each renderer then samples the arc in its own way.

    Angles are normalised to ``[0, 2π)`` so that ``theta1 ≤ theta2`` and ``linspace`` / a simple
    loop sweeps the correct counterclockwise arc without additional heuristics.

    Args:
        px: X-coordinate of the parent node (defines the arc radius).
        py: Y-coordinate of the parent node.
        first_pos: ``(x, y)`` of the first (smallest-rank) child.
        last_pos: ``(x, y)`` of the last (largest-rank) child.

    Returns:
        ``(theta1, theta2, r, n_pts)`` where *theta1* and *theta2* are the start and end angles
            in radians, *r* is the arc radius, and *n_pts* is a suggested point count for smooth
            sampling.  Returns ``None`` if the radius is negligibly small (``< 1e-12``).
    """
    r = math.hypot(px, py)
    if r < 1e-12:
        return None

    theta1 = math.atan2(first_pos[1], first_pos[0])
    theta2 = math.atan2(last_pos[1], last_pos[0])
    if theta1 < 0.0:
        theta1 += 2.0 * math.pi
    if theta2 < 0.0:
        theta2 += 2.0 * math.pi
    if theta2 < theta1:
        theta2 += 2.0 * math.pi

    n_pts = max(3, int((theta2 - theta1) * 30) + 2)

    return theta1, theta2, r, n_pts
