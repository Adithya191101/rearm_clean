"""Unit contracts for preserving FoundationPose top-versus-bottom detection."""

import math
from pathlib import Path
import sys

import numpy as np
import pytest


MOTION_BEHAVIORS = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "isaac_ros_manipulation"
    / "isaac_ros_manipulation_orchestration"
    / "isaac_ros_manipulation_orchestration"
    / "behaviors"
    / "motion_behaviors"
)
sys.path.insert(0, str(MOTION_BEHAVIORS))

from axial_object_pose import canonicalize_axial_pose


HALF_HEIGHT_M = 0.0505
MESH_TOP_AXIS = (0.0, 1.0, 0.0)


def _rotation_x(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def _mesh_pose(rotation_x_degrees: float) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = _rotation_x(rotation_x_degrees)
    pose[:3, 3] = (0.37, 0.01, 0.2005)
    return pose


def test_upright_mesh_axis_becomes_upright_base_center_pose():
    canonical, axis_z = canonicalize_axial_pose(
        _mesh_pose(90.0),
        half_height_m=HALF_HEIGHT_M,
        top_axis_in_mesh=MESH_TOP_AXIS,
    )

    assert axis_z == pytest.approx(1.0)
    assert canonical[:3, :3] == pytest.approx(np.eye(3))
    assert canonical[:3, 3] == pytest.approx((0.37, 0.01, 0.15))


def test_bottom_up_mesh_axis_preserves_downward_object_axis():
    canonical, axis_z = canonicalize_axial_pose(
        _mesh_pose(-90.0),
        half_height_m=HALF_HEIGHT_M,
        top_axis_in_mesh=MESH_TOP_AXIS,
    )

    assert axis_z == pytest.approx(-1.0)
    assert canonical[:3, :3] == pytest.approx(
        np.diag([1.0, -1.0, -1.0])
    )
    assert canonical[:3, 3] == pytest.approx((0.37, 0.01, 0.251))


def test_sideways_axis_is_rejected_instead_of_faking_a_pose():
    with pytest.raises(ValueError, match="not vertical enough"):
        canonicalize_axial_pose(
            _mesh_pose(0.0),
            half_height_m=HALF_HEIGHT_M,
            top_axis_in_mesh=MESH_TOP_AXIS,
        )
