"""Cylinder geometry: what gap a grasp actually needs.

WHY NOT JUST COMPARE AGAINST THE DIAMETER
"gap > 66 mm" is only the right test when the jaws close PERPENDICULAR to the can
axis. Close along the axis instead and the jaws have to clear the 101 mm height,
not the 66 mm diameter -- so a check hard-coded to the diameter would wave through
a grasp that is 35 mm too narrow. Since the grasp set here deliberately contains
several orientations, the required width is computed from each grasp's actual
closing direction rather than assumed.

THE SUPPORT WIDTH
For a finite cylinder of radius r and height h with unit axis ``a``, the width of
its projection onto a unit direction ``n`` is

    W(n) = 2 r sqrt(1 - (a.n)^2) + h |a.n|

  * n perpendicular to the axis  (a.n = 0)  ->  W = 2r  = 0.066 m, the diameter
  * n parallel to the axis       (|a.n| = 1) ->  W = h   = 0.101 m, the height

This is the width of the WHOLE cylinder, which for a tilted closing direction is
larger than what a short jaw pad near the can's mid-height would actually have to
clear. That over-estimate is deliberate: it errs towards demanding a wider gap,
and the failure this module exists to catch is a gap that is too NARROW.

WHICH AXIS IS THE CLOSING AXIS
The jaws travel along +/-Y of ``gripper_link``, and ``gripper_tcp`` is offset from
``gripper_link`` by a pure translation (rpy 0,0,0), so the closing direction is
the TCP's Y axis. The approach direction is the TCP's +X axis. Both facts are
read off rebot_b601dm_gripper.xacro; neither is a convention this package is free
to choose, and confusing the two produces a grasp that is 35 mm out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class CylinderSpec:
    """An upright finite cylinder, in its own object frame.

    ``origin_at_base`` records where the asset's origin sits, because it differs
    between conventions and getting it wrong is a half-height error. The NVIDIA
    stock soup can places its origin at the CENTRE OF THE BASE.
    """

    name: str
    diameter_m: float
    height_m: float
    origin_at_base: bool = True
    axis: tuple = (0.0, 0.0, 1.0)

    @property
    def radius_m(self) -> float:
        return 0.5 * self.diameter_m

    def axis_unit(self) -> np.ndarray:
        a = np.asarray(self.axis, dtype=float)
        n = float(np.linalg.norm(a))
        if n == 0.0:
            raise ValueError(f"{self.name}: cylinder axis must be non-zero")
        return a / n

    def centroid_offset(self) -> np.ndarray:
        """Object-frame position of the cylinder's centroid."""
        return self.axis_unit() * (0.5 * self.height_m if self.origin_at_base else 0.0)


# The one object this stage picks. The same dimensions are serialized into the
# committed grasp file, which the focused authoring test regenerates exactly.
SOUP_CAN = CylinderSpec(
    name="soup_can",
    diameter_m=0.066,
    height_m=0.101,
    origin_at_base=True,
    axis=(0.0, 0.0, 1.0),
)


def _unit(v: Sequence[float], what: str) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{what} must be a 3-vector, got shape {arr.shape}")
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise ValueError(f"{what} must be non-zero")
    return arr / n


def cylinder_support_width(cyl: CylinderSpec, closing_direction: Sequence[float]) -> float:
    """Width of ``cyl`` projected onto ``closing_direction``, in metres.

    This is the minimum jaw gap that clears the object along that direction. See
    the module docstring for the formula and for why it is an over-estimate when
    the direction is oblique to the axis.
    """
    n = _unit(closing_direction, "closing direction")
    a = cyl.axis_unit()
    dot = abs(float(np.dot(a, n)))
    # Guard the sqrt against dot drifting a hair past 1.0 through rounding.
    perp = math.sqrt(max(0.0, 1.0 - dot * dot))
    return cyl.diameter_m * perp + cyl.height_m * dot


def axial_extent(cyl: CylinderSpec) -> tuple:
    """(min, max) coordinate along the axis, in the object frame."""
    base = 0.0 if cyl.origin_at_base else -0.5 * cyl.height_m
    return base, base + cyl.height_m


def point_outside_cylinder(cyl: CylinderSpec, point: Sequence[float]) -> bool:
    """True if an object-frame point lies strictly outside the cylinder.

    Used to check that a PREGRASP pose has actually cleared the can, rather than
    sitting inside it -- which a pregrasp offset applied along the wrong sign of
    the approach axis would do, while still looking like a plausible number.
    """
    p = np.asarray(point, dtype=float)
    if p.shape != (3,):
        raise ValueError(f"point must be a 3-vector, got shape {p.shape}")
    a = cyl.axis_unit()
    along = float(np.dot(p, a))
    radial = float(np.linalg.norm(p - along * a))
    lo, hi = axial_extent(cyl)
    return radial > cyl.radius_m or along < lo or along > hi
