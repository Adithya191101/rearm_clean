#!/usr/bin/env python3
"""Pick and place area bounds for the reBot B601-DM, carved from the MEASURED
reachable band.

WHY A SEPARATE MODULE
These constants are the single source of truth for where the can may be spawned
(PICK_AREA) and where it is dropped (PLACE_AREA). pick_scene.py imports and
re-exports the bounds, and the wall-safety tests check their corners. Keeping
them here -- pure Python, no Isaac import -- lets every consumer share ONE
definition.

THE REACHABLE BAND (measured on the GPU, horizontal side approach)
The arm can only SIDE-grasp: approach axis horizontal (world +X into the can),
never top-down. With that approach the reachable front workspace for the grasp TCP
was measured at x in [0.30, 0.45], z in [0.20, 0.30]. The bottom-up source pose
places the side-grasp TCP 0.09 m above the table, at world z = 0.24. The areas
below stay comfortably inside x in [0.30, 0.45] and are validated corner-by-corner
by the focused wall-safety tests before any BT run.

FRAME: base_link. Area z values are support-surface heights. The upside-down
can's semantic bottom-center is one can height above PICK_AREA z.
"""

from __future__ import annotations

# Table-top height (world/base_link z). The can rests here; matches
# pick_scene.CAN_BASE_Z and the worktop top.
TABLE_Z = 0.15

# Physical TCP height above the support surface in the bottom-up source pose.
SOURCE_GRASP_HEIGHT_ABOVE_TABLE_M = 0.09

# PICK AREA -- one end of the centered pick/wall/place row. The can remains
# inside the measured side-grasp band and clears the wall's two-inch envelope.
PICK_AREA = {
    "x": (0.34, 0.40),
    "y": (-0.12, 0.00),
    "z": TABLE_Z,
}

# PLACE AREA -- the opposite end of the centered row. Its center is symmetric
# with the pick center around the wall, leaving the full attached can and
# gripper outside the hard two-inch safety envelope.
PLACE_AREA = {
    "x": (0.36, 0.38),
    "y": (0.34, 0.36),
    "z": TABLE_Z,
}

def _centre(area) -> tuple:
    return (
        0.5 * (area["x"][0] + area["x"][1]),
        0.5 * (area["y"][0] + area["y"][1]),
        area["z"],
    )


def pick_centre() -> tuple:
    """Centre of PICK_AREA (can base-centre in base_link)."""
    return _centre(PICK_AREA)


def place_centre() -> tuple:
    """Centre of PLACE_AREA (can base-centre in base_link)."""
    return _centre(PLACE_AREA)


def pick_area_corners():
    """The four corners of PICK_AREA as (name, x, y, z) tuples, for plan-checking."""
    (x0, x1), (y0, y1), z = PICK_AREA["x"], PICK_AREA["y"], PICK_AREA["z"]
    return [
        ("pick_x0y0", x0, y0, z),
        ("pick_x0y1", x0, y1, z),
        ("pick_x1y0", x1, y0, z),
        ("pick_x1y1", x1, y1, z),
    ]


def place_area_corners():
    """The four corners of PLACE_AREA as (name, x, y, z) tuples."""
    (x0, x1), (y0, y1), z = PLACE_AREA["x"], PLACE_AREA["y"], PLACE_AREA["z"]
    return [
        ("place_x0y0", x0, y0, z),
        ("place_x0y1", x0, y1, z),
        ("place_x1y0", x1, y0, z),
        ("place_x1y1", x1, y1, z),
    ]
