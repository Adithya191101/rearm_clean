#!/usr/bin/env python3
"""Runtime adapter for Seeed's official reBot B601-DM Isaac Sim asset.

The vendor USD is the production visual and PhysX articulation.  It needs two
runtime-only adaptations:

* nine nested rigid bodies must receive independent, world-preserving transform
  stacks before PhysX initializes;
* those repair transforms must be refreshed from the PhysX tensor view while
  rendering because ``resetXformStack`` prevents normal PhysX-to-USD writeback.

The referenced asset is never modified.  All authored opinions live in the
anonymous stage/session layers used by the running scene.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


WS_DIR = Path(__file__).resolve().parent.parent
ASSET_PATH = (
    WS_DIR / "usd" / "vendor" / "reBot_B601_DM" / "reBot_B601_DM.usda"
)
EXTRINSICS_PATH = WS_DIR / "config" / "extrinsics_sim.yaml"

# Seeed-Projects/reBot-Isaacsim, usd/reBot_B601_DM/reBot_B601_DM.usda.
# The USD tree is unchanged between c3ee253 and the latest verified cb824be.
OFFICIAL_ROOT_LAYER_SHA256 = (
    "6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84"
)

ROBOT_PRIM_PATH = "/World/reBot_B601_DM"
ARTICULATION_ROOT_PATH = f"{ROBOT_PRIM_PATH}/Geometry/base_link"

EXPECTED_DOF_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "gripper_joint1",
    "gripper_joint2",
]
EXPECTED_LOWER = np.array(
    [-2.8, -3.14, -3.14, -1.87, -1.57, -3.14, 0.0, 0.0],
    dtype=np.float64,
)
EXPECTED_UPPER = np.array(
    [2.8, 0.0, 0.0, 1.57, 1.57, 3.14, 0.0715, 0.0715],
    dtype=np.float64,
)

RUNTIME_KP = np.array(
    [500.0, 1500.0, 1000.0, 150.0, 80.0, 50.0, 5000.0, 5000.0],
    dtype=np.float64,
)
RUNTIME_KD = np.array(
    [60.0, 96.0, 76.0, 18.0, 10.0, 7.0, 41.28, 41.28],
    dtype=np.float64,
)

PHYSICS_DT = 1.0 / 120.0
LIMIT_ATOL = 2.0e-4

LINK_NAMES = (
    "base_link",
    "link1",
    "link2",
    "link3",
    "link4",
    "link5",
    "link6",
    "gripper_link",
    "gripper_left",
    "gripper_right",
)
REPAIRED_LINK_NAMES = LINK_NAMES[1:]
EXPECTED_NESTED_RIGID_BODY_COUNT = len(REPAIRED_LINK_NAMES)

LINK6_PATH = (
    f"{ARTICULATION_ROOT_PATH}/link1/link2/link3/link4/link5/link6"
)
GRIPPER_LINK_PATH = f"{LINK6_PATH}/gripper_link"
FINGER_PATHS = (
    f"{GRIPPER_LINK_PATH}/gripper_left",
    f"{GRIPPER_LINK_PATH}/gripper_right",
)
GRIPPER_TCP_PATH = f"{GRIPPER_LINK_PATH}/gripper_tcp"
CAMERA_MOUNT_PATH = f"{LINK6_PATH}/camera_mount"
CAMERA_LINK_PATH = f"{CAMERA_MOUNT_PATH}/camera_link"
CAMERA_OPTICAL_FRAME_PATH = (
    f"{CAMERA_LINK_PATH}/camera_color_optical_frame"
)
CAMERA_DEPTH_FRAME_PATH = (
    f"{CAMERA_OPTICAL_FRAME_PATH}/camera_depth_optical_frame"
)
CAMERA_PRIM_PATH = f"{CAMERA_OPTICAL_FRAME_PATH}/rgbd_camera"

# Keep the ROS planning model's established TCP definition.  The vendor jaw
# midpoint was measured about 4 mm away from this frame; that physical tolerance
# remains a simulation concern and must not rewrite the URDF/XRDF contract.
GRIPPER_TCP_XYZ_M = (-0.0443, 0.0, 0.0)


def asset_sha256(path: Path = ASSET_PATH) -> str:
    """Return the root-layer checksum used to pin the official asset."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_unique_named_paths(
    paths: Iterable[str],
    expected_names: Sequence[str],
) -> dict[str, str]:
    """Select exactly one path for each expected leaf name."""
    by_name: dict[str, list[str]] = {name: [] for name in expected_names}
    for path_value in paths:
        path = str(path_value)
        name = path.rstrip("/").rsplit("/", 1)[-1]
        if name in by_name:
            by_name[name].append(path)

    invalid = {
        name: matches
        for name, matches in by_name.items()
        if len(matches) != 1
    }
    if invalid:
        detail = ", ".join(
            f"{name}={matches}" for name, matches in invalid.items()
        )
        raise RuntimeError(f"vendor prim discovery is not unique: {detail}")
    return {name: matches[0] for name, matches in by_name.items()}


def discover_named_prim_paths(
    stage: Any,
    root_path: str,
    expected_names: Sequence[str],
) -> dict[str, str]:
    """Discover named prims below a composed USD subtree."""
    from pxr import Usd

    root = stage.GetPrimAtPath(root_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"vendor prim root does not resolve: {root_path}")
    return select_unique_named_paths(
        (str(prim.GetPath()) for prim in Usd.PrimRange(root)),
        expected_names,
    )


def discover_link_paths(stage: Any) -> dict[str, str]:
    """Resolve the ten vendor rigid bodies, excluding same-named visuals."""
    from pxr import Usd, UsdPhysics

    root = stage.GetPrimAtPath(ARTICULATION_ROOT_PATH)
    if not root or not root.IsValid():
        raise RuntimeError(
            f"vendor articulation root does not resolve: "
            f"{ARTICULATION_ROOT_PATH}"
        )
    return select_unique_named_paths(
        (
            str(prim.GetPath())
            for prim in Usd.PrimRange(root)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ),
        LINK_NAMES,
    )


def _matrix_numpy(matrix: Any) -> np.ndarray:
    return np.array(
        [[matrix[i][j] for j in range(4)] for i in range(4)],
        dtype=np.float64,
    )


def nested_rigid_body_issues(
    stage: Any,
    robot_prim_path: str = ROBOT_PRIM_PATH,
) -> list[dict[str, str]]:
    """Return nested rigid bodies that PhysX will reject without a reset."""
    from pxr import UsdGeom, UsdPhysics

    prefix = robot_prim_path.rstrip("/") + "/"
    rigids = [
        prim
        for prim in stage.Traverse()
        if str(prim.GetPath()).startswith(prefix)
        and prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    rigid_paths = {str(prim.GetPath()) for prim in rigids}
    issues: list[dict[str, str]] = []
    for prim in rigids:
        ancestor = prim.GetParent()
        rigid_ancestor = None
        while ancestor and ancestor.IsValid():
            if str(ancestor.GetPath()) in rigid_paths:
                rigid_ancestor = str(ancestor.GetPath())
                break
            ancestor = ancestor.GetParent()
        if rigid_ancestor and not UsdGeom.Xformable(prim).GetResetXformStack():
            issues.append(
                {
                    "body_path": str(prim.GetPath()),
                    "rigid_ancestor_path": rigid_ancestor,
                    "problem": "missing resetXformStack",
                }
            )
    return issues


def repair_nested_rigid_body_xforms(
    stage: Any,
    issues: Sequence[dict[str, str]],
) -> dict[str, Any]:
    """Repair nested rigid bodies in the session layer without moving them."""
    from pxr import Usd, UsdGeom

    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    cached = {
        item["body_path"]: cache.GetLocalToWorldTransform(
            stage.GetPrimAtPath(item["body_path"])
        )
        for item in issues
    }
    before = {
        path: _matrix_numpy(matrix) for path, matrix in cached.items()
    }

    session_layer = stage.GetSessionLayer()
    with Usd.EditContext(stage, Usd.EditTarget(session_layer)):
        # AddXformOp must observe the attribute composed by the previous edit,
        # so these high-level USD calls intentionally do not use ChangeBlock.
        for path, world_matrix in cached.items():
            xform = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            repair_op = xform.AddTransformOp(
                precision=UsdGeom.XformOp.PrecisionDouble,
                opSuffix="b601PhysxRepair",
            )
            repair_op.Set(world_matrix)
            xform.SetXformOpOrder([repair_op], resetXformStack=True)

    verify = UsdGeom.XformCache(Usd.TimeCode.Default())
    after = {
        path: _matrix_numpy(
            verify.GetLocalToWorldTransform(stage.GetPrimAtPath(path))
        )
        for path in cached
    }
    max_error = max(
        (
            float(np.max(np.abs(after[path] - before[path])))
            for path in cached
        ),
        default=0.0,
    )
    remaining = nested_rigid_body_issues(stage)
    return {
        "applied": True,
        "persistent": False,
        "edit_layer": session_layer.identifier,
        "repaired_body_paths": list(cached),
        "repaired_count": len(cached),
        "remaining_issue_count": len(remaining),
        "remaining_issues": remaining,
        "max_initial_world_matrix_error": max_error,
    }


def repair_vendor_stage(stage: Any) -> dict[str, Any]:
    """Apply and validate the mandatory nine-body vendor session repair."""
    links = discover_link_paths(stage)
    issues = nested_rigid_body_issues(stage)
    issue_names = {
        item["body_path"].rsplit("/", 1)[-1] for item in issues
    }
    if (
        len(issues) != EXPECTED_NESTED_RIGID_BODY_COUNT
        or issue_names != set(REPAIRED_LINK_NAMES)
    ):
        raise RuntimeError(
            "expected the nine known vendor nested-body issues, found "
            f"{len(issues)} names={sorted(issue_names)}"
        )

    repair = repair_nested_rigid_body_xforms(stage, issues)
    if (
        repair["remaining_issue_count"] != 0
        or repair["max_initial_world_matrix_error"] > 1.0e-10
    ):
        raise RuntimeError(f"vendor nested-body repair failed: {repair}")
    repair["link_paths"] = links
    return repair


def _rpy_matrix(rpy: Sequence[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]],
        dtype=np.float64,
    )
    ry = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]],
        dtype=np.float64,
    )
    rz = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz @ ry @ rx


def _set_local_transform(
    prim: Any,
    xyz: Sequence[float],
    rpy: Sequence[float],
) -> None:
    from pxr import Gf, UsdGeom

    rotation = _rpy_matrix(rpy)
    matrix = Gf.Matrix4d(1.0)
    matrix.SetRotateOnly(
        Gf.Matrix3d(*[float(value) for value in rotation.T.flatten()])
    )
    matrix.SetTranslateOnly(
        Gf.Vec3d(*[float(value) for value in xyz])
    )
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTransformOp(
        precision=UsdGeom.XformOp.PrecisionDouble
    ).Set(matrix)


def load_sim_extrinsics(path: Path = EXTRINSICS_PATH) -> dict[str, Any]:
    """Load the same simulation camera contract used by URDF generation."""
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data.get("profile") != "sim":
        raise RuntimeError(
            f"{path} declares profile {data.get('profile')!r}, expected 'sim'"
        )
    if data["camera_mount"]["parent"] != "link6":
        raise RuntimeError("simulation camera mount must remain parented to link6")
    if data["intrinsics"]["optical_frame"] != "camera_color_optical_frame":
        raise RuntimeError("simulation optical frame contract changed")
    return data


def author_ros_frames_and_camera(
    stage: Any,
    extrinsics: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Add the existing ROS TCP/camera frames below the moving vendor links."""
    from pxr import Gf, Sdf, UsdGeom

    link_paths = discover_link_paths(stage)
    ext = load_sim_extrinsics() if extrinsics is None else extrinsics

    gripper_tcp_path = f"{link_paths['gripper_link']}/gripper_tcp"
    tcp = UsdGeom.Xform.Define(stage, gripper_tcp_path)
    _set_local_transform(tcp.GetPrim(), GRIPPER_TCP_XYZ_M, (0.0, 0.0, 0.0))

    mount_path = f"{link_paths['link6']}/{ext['camera_mount']['child']}"
    mount = UsdGeom.Xform.Define(stage, mount_path)
    _set_local_transform(
        mount.GetPrim(),
        ext["camera_mount"]["xyz"],
        ext["camera_mount"]["rpy"],
    )

    camera_link_path = f"{mount_path}/{ext['camera_link']['child']}"
    camera_link = UsdGeom.Xform.Define(stage, camera_link_path)
    _set_local_transform(
        camera_link.GetPrim(),
        ext["camera_link"]["xyz"],
        ext["camera_link"]["rpy"],
    )

    optical_name = ext["intrinsics"]["optical_frame"]
    optical_path = f"{camera_link_path}/{optical_name}"
    optical = UsdGeom.Xform.Define(stage, optical_path)
    _set_local_transform(
        optical.GetPrim(),
        (0.0, 0.0, 0.0),
        ext["optical_frame_rpy"],
    )

    depth_path = f"{optical_path}/camera_depth_optical_frame"
    depth = UsdGeom.Xform.Define(stage, depth_path)
    _set_local_transform(
        depth.GetPrim(), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    )

    camera_path = f"{optical_path}/rgbd_camera"
    camera = UsdGeom.Camera.Define(stage, camera_path)
    intrinsics = ext["intrinsics"]
    width = int(intrinsics["width"])
    height = int(intrinsics["height"])
    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    horizontal_aperture = 20.955
    focal_length = fx * horizontal_aperture / width
    vertical_aperture = (
        horizontal_aperture * (height / width) * (fx / fy)
    )
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(horizontal_aperture)
    camera.CreateVerticalApertureAttr(vertical_aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.02, 10.0))
    _set_local_transform(
        camera.GetPrim(), (0.0, 0.0, 0.0), (math.pi, 0.0, 0.0)
    )
    camera.GetPrim().CreateAttribute(
        "rebot:sourceExtrinsics", Sdf.ValueTypeNames.String, custom=True
    ).Set(str(EXTRINSICS_PATH.name))
    camera.GetPrim().CreateAttribute(
        "rebot:resolution", Sdf.ValueTypeNames.Int2, custom=True
    ).Set(Gf.Vec2i(width, height))
    camera.GetPrim().CreateAttribute(
        "rebot:fps", Sdf.ValueTypeNames.Int, custom=True
    ).Set(int(intrinsics["fps"]))

    expected = {
        "gripper_tcp": GRIPPER_TCP_PATH,
        "camera_mount": CAMERA_MOUNT_PATH,
        "camera_link": CAMERA_LINK_PATH,
        "camera_color_optical_frame": CAMERA_OPTICAL_FRAME_PATH,
        "camera_depth_optical_frame": CAMERA_DEPTH_FRAME_PATH,
        "rgbd_camera": CAMERA_PRIM_PATH,
    }
    actual = {
        "gripper_tcp": gripper_tcp_path,
        "camera_mount": mount_path,
        "camera_link": camera_link_path,
        "camera_color_optical_frame": optical_path,
        "camera_depth_optical_frame": depth_path,
        "rgbd_camera": camera_path,
    }
    if actual != expected:
        raise RuntimeError(
            f"vendor ROS frame hierarchy changed: actual={actual}"
        )
    return actual


def _rotation_from_xyzw(quaternion: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError(f"invalid PhysX link quaternion: {q}")
    x, y, z, w = q / norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


class LinkUsdSync:
    """Write live PhysX link transforms into the session repair operations."""

    def __init__(
        self,
        stage: Any,
        articulation: Any,
        repair_body_paths: Sequence[str],
    ) -> None:
        from pxr import UsdGeom

        self._stage = stage
        self._targets: list[tuple[int, Any, str]] = []
        self.rebind(articulation)
        for path in repair_body_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                continue
            repair_ops = [
                op
                for op in UsdGeom.Xformable(prim).GetOrderedXformOps()
                if op.GetOpName() == "xformOp:transform:b601PhysxRepair"
            ]
            name = path.rsplit("/", 1)[-1]
            if len(repair_ops) == 1 and name in self._body_names:
                self._targets.append(
                    (self._body_names.index(name), repair_ops[0], name)
                )

    @property
    def link_count(self) -> int:
        return len(self._targets)

    @property
    def link_names(self) -> tuple[str, ...]:
        return tuple(target[2] for target in self._targets)

    @property
    def body_names(self) -> tuple[str, ...]:
        """Return PhysX body names in the same order as ``link_transforms``."""
        return tuple(self._body_names)

    def rebind(self, articulation: Any) -> None:
        """Rebind after a reset rebuilds the articulation tensor view."""
        view = articulation._articulation_view
        self._physics_view = view._physics_view
        self._body_names = list(view.body_names)

    def link_transforms(self) -> np.ndarray:
        """Return current PhysX world poses as ``xyz + quaternion_xyzw`` rows."""
        transforms = self._physics_view.get_link_transforms()
        if hasattr(transforms, "numpy"):
            transforms = transforms.numpy()
        array = np.asarray(transforms, dtype=np.float64)
        return array.reshape(-1, len(self._body_names), 7)[0]

    def push(self) -> int:
        """Synchronize all repaired links to their current PhysX world poses."""
        from pxr import Gf, Sdf, Usd

        links = self.link_transforms()
        if not np.all(np.isfinite(links)):
            raise RuntimeError("non-finite PhysX link transform during USD sync")

        with Usd.EditContext(
            self._stage,
            Usd.EditTarget(self._stage.GetSessionLayer()),
        ), Sdf.ChangeBlock():
            for link_index, op, _ in self._targets:
                position = links[link_index, :3]
                rotation = _rotation_from_xyzw(links[link_index, 3:])
                matrix = Gf.Matrix4d(1.0)
                matrix.SetRotateOnly(
                    Gf.Matrix3d(
                        *[
                            float(value)
                            for value in rotation.T.flatten()
                        ]
                    )
                )
                matrix.SetTranslateOnly(
                    Gf.Vec3d(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    )
                )
                op.Set(matrix)
        return len(self._targets)
