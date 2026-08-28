#!/usr/bin/env python3
"""Geometry and USD authoring for the mapped pick-to-place obstacle."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WALL_SAFETY_SCHEMA_VERSION = 1
MINIMUM_WALL_CLEARANCE_M = 2.0 * 0.0254


@dataclass(frozen=True)
class WallSpec:
    """Axis-aligned static wall authored in the base/world frame."""

    prim_path: str
    center_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    color_rgb: tuple[float, float, float]

    def __post_init__(self) -> None:
        values = (*self.center_xyz_m, *self.size_xyz_m, *self.color_rgb)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("wall geometry and color must be finite")
        if any(size <= 0.0 for size in self.size_xyz_m):
            raise ValueError("wall dimensions must be positive")
        if any(not 0.0 <= channel <= 1.0 for channel in self.color_rgb):
            raise ValueError("wall color channels must be in [0, 1]")
        if not self.prim_path.startswith("/"):
            raise ValueError("wall prim path must be absolute")

    @property
    def min_xyz_m(self) -> tuple[float, float, float]:
        return tuple(
            center - 0.5 * size
            for center, size in zip(self.center_xyz_m, self.size_xyz_m)
        )

    @property
    def max_xyz_m(self) -> tuple[float, float, float]:
        return tuple(
            center + 0.5 * size
            for center, size in zip(self.center_xyz_m, self.size_xyz_m)
        )

    @property
    def top_z_m(self) -> float:
        return self.max_xyz_m[2]

    def blocks_transfer(
        self,
        start_xyz_m: tuple[float, float, float],
        end_xyz_m: tuple[float, float, float],
        *,
        object_radius_m: float,
        object_height_m: float,
    ) -> bool:
        """Return whether a straight, level carried object intersects the wall."""
        start = tuple(float(value) for value in start_xyz_m)
        end = tuple(float(value) for value in end_xyz_m)
        radius = float(object_radius_m)
        height = float(object_height_m)
        if (
            not all(math.isfinite(value) for value in (*start, *end, radius, height))
            or radius < 0.0
            or height <= 0.0
        ):
            raise ValueError("transfer geometry must be finite and positive")

        dy = end[1] - start[1]
        if math.isclose(dy, 0.0):
            return False
        t = (self.center_xyz_m[1] - start[1]) / dy
        if not 0.0 <= t <= 1.0:
            return False
        x_at_wall = start[0] + t * (end[0] - start[0])
        min_x, _, min_z = self.min_xyz_m
        max_x, _, max_z = self.max_xyz_m
        overlaps_x = min_x - radius <= x_at_wall <= max_x + radius
        object_base_z = start[2] + t * (end[2] - start[2])
        overlaps_z = (
            object_base_z < max_z
            and object_base_z + height > min_z
        )
        return overlaps_x and overlaps_z


# The wall sits entirely in the gap between PICK_AREA and PLACE_AREA.
# Its 310 mm top invalidates the prior approximately 289 mm carried-can base
# path. The 40 mm width blocks the direct center-to-center path while leaving
# enough of the arm's measured x workspace for the attached can to detour
# laterally around either edge.
TRANSFER_WALL = WallSpec(
    prim_path="/World/transfer_wall",
    center_xyz_m=(0.37, 0.16, 0.23),
    size_xyz_m=(0.04, 0.025, 0.16),
    color_rgb=(0.82, 0.20, 0.06),
)

# The camera-mapped ESDF remains the primary world representation, but a hard
# MoveIt object prevents a sparse or stale depth update from making the wall
# traversable. Inflate each face by the same clearance required by the runtime
# telemetry gate, so a collision-free static-scene plan also has a two-inch
# shell around the physical wall.
PLANNING_WALL = WallSpec(
    prim_path="/transfer_wall_safety_envelope",
    center_xyz_m=TRANSFER_WALL.center_xyz_m,
    size_xyz_m=tuple(
        size + 2.0 * MINIMUM_WALL_CLEARANCE_M
        for size in TRANSFER_WALL.size_xyz_m
    ),
    color_rgb=TRANSFER_WALL.color_rgb,
)


def moveit_scene_text(spec: WallSpec = PLANNING_WALL) -> str:
    """Serialize one box in MoveIt's plain-text ``.scene`` format."""
    object_name = spec.prim_path.rsplit("/", 1)[-1]
    if not object_name or any(character.isspace() for character in object_name):
        raise ValueError("MoveIt scene object name must be non-empty and whitespace-free")

    def row(values: Sequence[float]) -> str:
        return " ".join(f"{float(value):.9g}" for value in values)

    return "\n".join((
        "(noname)+",
        f"* {object_name}",
        row(spec.center_xyz_m),
        "0 0 0 1",
        "1",
        "box",
        row(spec.size_xyz_m),
        "0 0 0",
        "0 0 0 1",
        "0 0 0 0",
        "0",
        ".",
        "",
    ))


@dataclass(frozen=True)
class CollisionSphere:
    """One planner collision sphere expressed in its owning link frame."""

    link_name: str
    center_xyz_m: tuple[float, float, float]
    radius_m: float

    def __post_init__(self) -> None:
        values = (*self.center_xyz_m, self.radius_m)
        if (
            not self.link_name
            or not all(math.isfinite(value) for value in values)
            or self.radius_m <= 0.0
        ):
            raise ValueError("collision sphere must have a link and finite radius")


def load_planner_collision_spheres(path: str | Path) -> tuple[CollisionSphere, ...]:
    """Load XRDF collision spheres, including the planner's link buffers."""
    import yaml

    document = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        sphere_groups = document["geometry"]["collision_spheres"]["spheres"]
        buffers = document["collision"]["buffer_distance"]
    except (KeyError, TypeError) as exc:
        raise ValueError("XRDF is missing collision sphere geometry") from exc

    spheres = []
    for link_name, records in sphere_groups.items():
        buffer_m = float(buffers.get(link_name, 0.0))
        if not math.isfinite(buffer_m) or buffer_m < 0.0:
            raise ValueError(f"invalid collision buffer for {link_name}")
        for record in records:
            center = tuple(float(value) for value in record["center"])
            if len(center) != 3:
                raise ValueError(f"invalid sphere center for {link_name}")
            spheres.append(CollisionSphere(
                link_name=str(link_name),
                center_xyz_m=center,
                radius_m=float(record["radius"]) + buffer_m,
            ))
    if not spheres:
        raise ValueError("XRDF contains no collision spheres")
    return tuple(spheres)


def quaternion_xyzw_rotation_matrix(values: Sequence[float]) -> np.ndarray:
    """Return a rotation matrix for a finite, normalized XYZW quaternion."""
    quaternion = np.asarray(values, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm < 1.0e-12:
        raise ValueError("quaternion must be non-zero")
    x, y, z, w = quaternion / norm
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


def transform_collision_spheres(
    spheres: Sequence[CollisionSphere],
    frame_transforms_xyzw: dict[str, Sequence[float]],
) -> np.ndarray:
    """Transform link-local spheres into world ``[x, y, z, radius]`` rows."""
    result = np.empty((len(spheres), 4), dtype=float)
    frame_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, sphere in enumerate(spheres):
        if sphere.link_name not in frame_cache:
            try:
                transform = np.asarray(
                    frame_transforms_xyzw[sphere.link_name],
                    dtype=float,
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing live transform for {sphere.link_name}"
                ) from exc
            if transform.shape != (7,) or not np.all(np.isfinite(transform)):
                raise ValueError(
                    f"invalid live transform for {sphere.link_name}"
                )
            frame_cache[sphere.link_name] = (
                transform[:3],
                quaternion_xyzw_rotation_matrix(transform[3:]),
            )
        translation, rotation = frame_cache[sphere.link_name]
        result[index, :3] = (
            rotation @ np.asarray(sphere.center_xyz_m, dtype=float)
            + translation
        )
        result[index, 3] = sphere.radius_m
    return result


def cylinder_cover_spheres(
    center_xyz_m: Sequence[float],
    axis_xyz: Sequence[float],
    *,
    radius_m: float,
    height_m: float,
    count: int = 7,
) -> np.ndarray:
    """Conservatively cover an oriented cylinder with a short sphere chain."""
    center = np.asarray(center_xyz_m, dtype=float)
    axis = np.asarray(axis_xyz, dtype=float)
    if (
        center.shape != (3,)
        or axis.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(axis))
        or not math.isfinite(radius_m)
        or not math.isfinite(height_m)
        or radius_m <= 0.0
        or height_m <= 0.0
        or count < 1
    ):
        raise ValueError("cylinder geometry must be finite and positive")
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-12:
        raise ValueError("cylinder axis must be non-zero")
    axis /= axis_norm

    slab_height = height_m / count
    offsets = (
        -0.5 * height_m
        + (np.arange(count, dtype=float) + 0.5) * slab_height
    )
    cover_radius = math.hypot(radius_m, 0.5 * slab_height)
    rows = np.empty((count, 4), dtype=float)
    rows[:, :3] = center[None, :] + offsets[:, None] * axis[None, :]
    rows[:, 3] = cover_radius
    return rows


def project_point_to_finite_cylinder(
    point_xyz_m: Sequence[float],
    center_xyz_m: Sequence[float],
    axis_xyz: Sequence[float],
    *,
    radius_m: float,
    height_m: float,
) -> np.ndarray:
    """Return the closest point in a solid, oriented finite cylinder."""
    point = np.asarray(point_xyz_m, dtype=float)
    center = np.asarray(center_xyz_m, dtype=float)
    axis = np.asarray(axis_xyz, dtype=float)
    if (
        point.shape != (3,)
        or center.shape != (3,)
        or axis.shape != (3,)
        or not np.all(np.isfinite(point))
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(axis))
        or not math.isfinite(radius_m)
        or not math.isfinite(height_m)
        or radius_m <= 0.0
        or height_m <= 0.0
    ):
        raise ValueError("cylinder projection requires finite, positive geometry")

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-12:
        raise ValueError("cylinder axis must be non-zero")
    axis /= axis_norm

    offset = point - center
    axial_distance = float(np.dot(offset, axis))
    radial_offset = offset - axial_distance * axis
    radial_distance = float(np.linalg.norm(radial_offset))
    if radial_distance > radius_m:
        radial_offset *= radius_m / radial_distance
    axial_distance = float(np.clip(
        axial_distance,
        -0.5 * height_m,
        0.5 * height_m,
    ))
    return center + axial_distance * axis + radial_offset


@dataclass(frozen=True)
class CylinderAabbClearance:
    """Certified clearance bounds with a feasible closest-pair witness."""

    lower_bound_m: float
    upper_bound_m: float
    cylinder_witness_m: np.ndarray
    box_witness_m: np.ndarray
    iterations: int
    converged: bool

    @property
    def uncertainty_m(self) -> float:
        return self.upper_bound_m - self.lower_bound_m


def _finite_cylinder_support_point(
    center: np.ndarray,
    axis: np.ndarray,
    *,
    radius_m: float,
    half_height_m: float,
    direction: np.ndarray,
) -> np.ndarray:
    """Return the cylinder point furthest along ``direction``."""
    axial_component = float(np.dot(axis, direction))
    radial_direction = direction - axial_component * axis
    radial_norm = float(np.linalg.norm(radial_direction))
    axial_sign = 1.0 if axial_component >= 0.0 else -1.0
    point = center + axial_sign * half_height_m * axis
    if radial_norm > 1.0e-15:
        point = point + radius_m * radial_direction / radial_norm
    return point


def finite_cylinder_aabb_clearance_bounds(
    center_xyz_m: Sequence[float],
    axis_xyz: Sequence[float],
    *,
    radius_m: float,
    height_m: float,
    minimum_xyz_m: Sequence[float],
    maximum_xyz_m: Sequence[float],
    tolerance_m: float = 1.0e-9,
    max_iterations: int = 128,
) -> CylinderAabbClearance:
    """Return certified lower/upper bounds for cylinder-to-box clearance.

    Alternating projection supplies a feasible witness pair and therefore an
    upper bound. The separating support planes normal to that pair supply a
    lower bound. If nearly parallel features converge slowly, the lower bound
    remains conservative instead of failing or overstating safety.
    """
    center = np.asarray(center_xyz_m, dtype=float)
    axis = np.asarray(axis_xyz, dtype=float)
    minimum = np.asarray(minimum_xyz_m, dtype=float)
    maximum = np.asarray(maximum_xyz_m, dtype=float)
    if (
        center.shape != (3,)
        or axis.shape != (3,)
        or minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(center))
        or not np.all(np.isfinite(axis))
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(maximum <= minimum)
        or not math.isfinite(radius_m)
        or not math.isfinite(height_m)
        or radius_m <= 0.0
        or height_m <= 0.0
        or not math.isfinite(tolerance_m)
        or tolerance_m <= 0.0
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("cylinder and AABB must be finite, positive geometry")

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1.0e-12:
        raise ValueError("cylinder axis must be non-zero")
    axis /= axis_norm

    box_point = np.clip(center, minimum, maximum)
    best_lower_m = 0.0
    best_upper_m = float("inf")
    best_cylinder_point = center.copy()
    best_box_point = box_point.copy()
    converged = False

    for iteration in range(1, max_iterations + 1):
        cylinder_point = project_point_to_finite_cylinder(
            box_point,
            center,
            axis,
            radius_m=radius_m,
            height_m=height_m,
        )
        next_box_point = np.clip(cylinder_point, minimum, maximum)
        separation = next_box_point - cylinder_point
        upper_bound_m = float(np.linalg.norm(separation))
        if upper_bound_m < best_upper_m:
            best_upper_m = upper_bound_m
            best_cylinder_point = cylinder_point.copy()
            best_box_point = next_box_point.copy()

        if upper_bound_m <= tolerance_m:
            best_lower_m = 0.0
            best_upper_m = 0.0
            best_cylinder_point = cylinder_point.copy()
            best_box_point = next_box_point.copy()
            converged = True
            break

        direction = separation / upper_bound_m
        cylinder_support = _finite_cylinder_support_point(
            center,
            axis,
            radius_m=radius_m,
            half_height_m=0.5 * height_m,
            direction=direction,
        )
        box_support = np.where(-direction >= 0.0, maximum, minimum)
        lower_bound_m = max(
            0.0,
            float(np.dot(direction, box_support - cylinder_support)),
        )
        best_lower_m = max(
            best_lower_m,
            min(lower_bound_m, best_upper_m),
        )
        if best_upper_m - best_lower_m <= tolerance_m:
            converged = True
            break
        box_point = next_box_point

    best_lower_m = min(best_lower_m, best_upper_m)
    if best_lower_m <= tolerance_m:
        best_lower_m = 0.0
    return CylinderAabbClearance(
        lower_bound_m=best_lower_m,
        upper_bound_m=best_upper_m,
        cylinder_witness_m=best_cylinder_point,
        box_witness_m=best_box_point,
        iterations=iteration,
        converged=converged,
    )


def finite_cylinder_aabb_clearance(
    center_xyz_m: Sequence[float],
    axis_xyz: Sequence[float],
    *,
    radius_m: float,
    height_m: float,
    minimum_xyz_m: Sequence[float],
    maximum_xyz_m: Sequence[float],
    tolerance_m: float = 1.0e-9,
    max_iterations: int = 128,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return a conservative clearance and feasible cylinder/box witnesses."""
    result = finite_cylinder_aabb_clearance_bounds(
        center_xyz_m,
        axis_xyz,
        radius_m=radius_m,
        height_m=height_m,
        minimum_xyz_m=minimum_xyz_m,
        maximum_xyz_m=maximum_xyz_m,
        tolerance_m=tolerance_m,
        max_iterations=max_iterations,
    )
    return (
        result.lower_bound_m,
        result.cylinder_witness_m,
        result.box_witness_m,
    )


def sphere_aabb_clearances(
    world_spheres: np.ndarray,
    minimum_xyz_m: Sequence[float],
    maximum_xyz_m: Sequence[float],
) -> np.ndarray:
    """Return signed sphere-to-box distances; zero means geometric contact."""
    values = np.asarray(world_spheres, dtype=float)
    minimum = np.asarray(minimum_xyz_m, dtype=float)
    maximum = np.asarray(maximum_xyz_m, dtype=float)
    if (
        values.ndim != 2
        or values.shape[1] != 4
        or values.shape[0] < 1
        or minimum.shape != (3,)
        or maximum.shape != (3,)
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(minimum))
        or not np.all(np.isfinite(maximum))
        or np.any(values[:, 3] <= 0.0)
        or np.any(maximum <= minimum)
    ):
        raise ValueError("spheres and AABB must be finite, positive geometry")

    box_center = 0.5 * (minimum + maximum)
    box_half_extent = 0.5 * (maximum - minimum)
    offset = np.abs(values[:, :3] - box_center) - box_half_extent
    outside = np.linalg.norm(np.maximum(offset, 0.0), axis=1)
    inside = np.minimum(np.max(offset, axis=1), 0.0)
    return outside + inside - values[:, 3]


def minimum_sphere_aabb_clearance(
    world_spheres: np.ndarray,
    minimum_xyz_m: Sequence[float],
    maximum_xyz_m: Sequence[float],
) -> tuple[float, int]:
    """Return the minimum signed clearance and the responsible sphere index."""
    clearances = sphere_aabb_clearances(
        world_spheres,
        minimum_xyz_m,
        maximum_xyz_m,
    )
    index = int(np.argmin(clearances))
    return float(clearances[index]), index


def validate_wall_safety_state(payload: Any) -> dict:
    """Validate and return one simulation-to-recorder wall-safety record."""
    if not isinstance(payload, dict):
        raise ValueError("wall-safety state must be a JSON object")
    if payload.get("schema_version") != WALL_SAFETY_SCHEMA_VERSION:
        raise ValueError("unsupported wall-safety schema version")
    try:
        sample = payload["sample"]
        observed = payload["minimum_observed"]
        contact = payload["contact"]
        numeric = (
            payload["sim_time_s"],
            sample["robot_clearance_m"],
            sample["can_clearance_m"],
            sample["clearance_m"],
            observed["robot_clearance_m"],
            observed["can_clearance_m"],
            observed["clearance_m"],
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("wall-safety state is missing required fields") from exc
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("wall-safety clearances must be finite")
    if (
        not isinstance(contact.get("current"), bool)
        or not isinstance(contact.get("ever"), bool)
        or not isinstance(contact.get("events"), int)
        or contact["events"] < 0
    ):
        raise ValueError("wall-safety contact fields are invalid")
    if observed["clearance_m"] > sample["clearance_m"] + 1.0e-12:
        raise ValueError("observed clearance cannot exceed current clearance")
    return payload


def wall_safety_failure_reason(
    payload: Any,
    *,
    minimum_clearance_m: float = MINIMUM_WALL_CLEARANCE_M,
) -> str | None:
    """Return why a completed run fails the wall gate, or ``None``."""
    if not math.isfinite(minimum_clearance_m) or minimum_clearance_m < 0.0:
        raise ValueError("minimum clearance must be finite and non-negative")
    state = validate_wall_safety_state(payload)
    if state["contact"]["ever"]:
        return "PhysX reported robot/can contact with the transfer wall"
    observed = float(state["minimum_observed"]["clearance_m"])
    if observed < minimum_clearance_m:
        return (
            f"minimum wall clearance {observed:.4f} m is below "
            f"{minimum_clearance_m:.4f} m"
        )
    return None


def atomic_write_json(path: str | Path, payload: Any) -> None:
    """Replace a JSON state file atomically so readers never see partial data."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def create_transfer_wall(stage: Any, spec: WallSpec = TRANSFER_WALL) -> Any:
    """Create the visible static collision geometry that nvblox must map."""
    from pxr import Gf, UsdGeom, UsdPhysics

    wall = UsdGeom.Cube.Define(stage, spec.prim_path)
    wall.CreateSizeAttr(1.0)
    wall.CreateDisplayColorAttr([Gf.Vec3f(*spec.color_rgb)])
    xform = UsdGeom.Xformable(wall)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*spec.center_xyz_m))
    xform.AddScaleOp().Set(Gf.Vec3f(*spec.size_xyz_m))
    collision = UsdPhysics.CollisionAPI.Apply(wall.GetPrim())
    collision.CreateCollisionEnabledAttr().Set(True)
    return wall.GetPrim()


def trajectory_contrast_metrics(
    mapped_positions: Sequence[Sequence[float]],
    cleared_positions: Sequence[Sequence[float]],
    *,
    sample_count: int = 101,
) -> dict[str, float]:
    """Compare normalized joint paths returned for the same endpoints."""
    mapped = np.asarray(mapped_positions, dtype=float)
    cleared = np.asarray(cleared_positions, dtype=float)
    if (
        mapped.ndim != 2
        or cleared.ndim != 2
        or mapped.shape[0] < 2
        or cleared.shape[0] < 2
        or mapped.shape[1] != cleared.shape[1]
        or sample_count < 2
        or not np.isfinite(mapped).all()
        or not np.isfinite(cleared).all()
    ):
        raise ValueError("trajectories must be finite 2D paths with matching joints")

    samples = np.linspace(0.0, 1.0, sample_count)

    def resample(path: np.ndarray) -> np.ndarray:
        source = np.linspace(0.0, 1.0, path.shape[0])
        return np.column_stack(
            [np.interp(samples, source, path[:, joint]) for joint in range(path.shape[1])]
        )

    mapped_sampled = resample(mapped)
    cleared_sampled = resample(cleared)

    def chord_deviation(path: np.ndarray) -> float:
        chord = (
            path[0][None, :]
            + samples[:, None] * (path[-1] - path[0])[None, :]
        )
        return float(np.linalg.norm(path - chord, axis=1).max())

    mapped_deviation = chord_deviation(mapped_sampled)
    cleared_deviation = chord_deviation(cleared_sampled)
    ratio = (
        mapped_deviation / cleared_deviation
        if cleared_deviation > 1e-9
        else math.inf
    )
    return {
        "mapped_chord_deviation_rad": mapped_deviation,
        "cleared_chord_deviation_rad": cleared_deviation,
        "mapped_to_cleared_ratio": ratio,
        "max_path_separation_rad": float(
            np.linalg.norm(mapped_sampled - cleared_sampled, axis=1).max()
        ),
        "start_separation_rad": float(
            np.linalg.norm(mapped_sampled[0] - cleared_sampled[0])
        ),
        "goal_separation_rad": float(
            np.linalg.norm(mapped_sampled[-1] - cleared_sampled[-1])
        ),
    }
