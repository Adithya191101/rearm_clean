"""ROS-free command-line configuration for the pick-and-place goal client."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Sequence


@dataclass(frozen=True)
class DropTarget:
    """Drop pose expressed in the behavior tree's planning frame."""

    frame_id: str
    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float

    def validate(self) -> None:
        """Reject values that cannot form a valid ROS pose."""
        if not self.frame_id or any(char.isspace() for char in self.frame_id):
            raise ValueError("frame_id must be non-empty and contain no whitespace")

        values = (self.x, self.y, self.z, self.qx, self.qy, self.qz, self.qw)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("drop pose values must be finite")

        quaternion_norm = math.sqrt(
            self.qx ** 2 + self.qy ** 2 + self.qz ** 2 + self.qw ** 2
        )
        if not math.isclose(quaternion_norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
            raise ValueError(
                f"drop quaternion must be normalized; norm={quaternion_norm:.6f}"
            )


def compose_drop_target(
    *,
    frame_id: str,
    object_position: Sequence[float],
    object_quaternion_xyzw: Sequence[float],
    grasp_position: Sequence[float],
    grasp_quaternion_wxyz: Sequence[float],
) -> DropTarget:
    """Compose a desired object pose with an object-relative TCP grasp."""
    if len(object_position) != 3 or len(grasp_position) != 3:
        raise ValueError("object and grasp positions must each have 3 values")
    if len(object_quaternion_xyzw) != 4 or len(grasp_quaternion_wxyz) != 4:
        raise ValueError("object and grasp quaternions must each have 4 values")

    values = (
        *object_position,
        *object_quaternion_xyzw,
        *grasp_position,
        *grasp_quaternion_wxyz,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("pose composition values must be finite")

    ox, oy, oz, ow = (float(value) for value in object_quaternion_xyzw)
    gw, gx, gy, gz = (float(value) for value in grasp_quaternion_wxyz)
    object_norm = math.sqrt(ox * ox + oy * oy + oz * oz + ow * ow)
    grasp_norm = math.sqrt(gx * gx + gy * gy + gz * gz + gw * gw)
    if not math.isclose(object_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("object quaternion must be normalized")
    if not math.isclose(grasp_norm, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("grasp quaternion must be normalized")

    # Rotate the object-relative grasp translation into the target frame.
    vx, vy, vz = (float(value) for value in grasp_position)
    rotation = (
        (
            1.0 - 2.0 * (oy * oy + oz * oz),
            2.0 * (ox * oy - oz * ow),
            2.0 * (ox * oz + oy * ow),
        ),
        (
            2.0 * (ox * oy + oz * ow),
            1.0 - 2.0 * (ox * ox + oz * oz),
            2.0 * (oy * oz - ox * ow),
        ),
        (
            2.0 * (ox * oz - oy * ow),
            2.0 * (oy * oz + ox * ow),
            1.0 - 2.0 * (ox * ox + oy * oy),
        ),
    )
    rx = rotation[0][0] * vx + rotation[0][1] * vy + rotation[0][2] * vz
    ry = rotation[1][0] * vx + rotation[1][1] * vy + rotation[1][2] * vz
    rz = rotation[2][0] * vx + rotation[2][1] * vy + rotation[2][2] * vz

    # Hamilton product: target_object rotation * object-relative grasp rotation.
    qx = ow * gx + ox * gw + oy * gz - oz * gy
    qy = ow * gy - ox * gz + oy * gw + oz * gx
    qz = ow * gz + ox * gy - oy * gx + oz * gw
    qw = ow * gw - ox * gx - oy * gy - oz * gz
    target = DropTarget(
        frame_id=frame_id,
        x=float(object_position[0]) + rx,
        y=float(object_position[1]) + ry,
        z=float(object_position[2]) + rz,
        qx=qx,
        qy=qy,
        qz=qz,
        qw=qw,
    )
    target.validate()
    return target


@dataclass(frozen=True)
class GoalOptions:
    """Validated options consumed by the ROS action client."""

    target: DropTarget
    server_timeout_s: float
    send_timeout_s: float
    result_timeout_s: float
    require_complete: bool


def _finite_float(raw: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{raw!r} is not a number") from exc
    if not math.isfinite(value):
        raise argparse.ArgumentTypeError("value must be finite")
    return value


def _positive_float(raw: str) -> float:
    value = _finite_float(raw)
    if value <= 0.0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return value


def _frame_id(raw: str) -> str:
    value = raw.strip()
    if not value or any(char.isspace() for char in value):
        raise argparse.ArgumentTypeError(
            "frame id must be non-empty and contain no whitespace"
        )
    return value


def build_parser(default_target: DropTarget) -> argparse.ArgumentParser:
    """Build the public CLI while retaining the validated drop pose defaults."""
    parser = argparse.ArgumentParser(
        description=(
            "Send one perception-driven SINGLE_BIN pick-and-place goal. "
            "Object detection and grasp selection remain workflow-configured."
        )
    )
    parser.add_argument("--frame-id", type=_frame_id, default=default_target.frame_id)
    parser.add_argument("--drop-x", type=_finite_float, default=default_target.x)
    parser.add_argument("--drop-y", type=_finite_float, default=default_target.y)
    parser.add_argument("--drop-z", type=_finite_float, default=default_target.z)
    parser.add_argument("--qx", type=_finite_float, default=default_target.qx)
    parser.add_argument("--qy", type=_finite_float, default=default_target.qy)
    parser.add_argument("--qz", type=_finite_float, default=default_target.qz)
    parser.add_argument("--qw", type=_finite_float, default=default_target.qw)
    parser.add_argument("--server-timeout", type=_positive_float, default=120.0)
    parser.add_argument("--send-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--result-timeout", type=_positive_float, default=300.0)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Treat PARTIAL_SUCCESS as a failure; default preserves legacy behavior.",
    )
    return parser


def parse_goal_options(
    argv: Sequence[str] | None,
    *,
    default_target: DropTarget,
) -> GoalOptions:
    """Parse and validate goal options without importing ROS."""
    default_target.validate()
    args = build_parser(default_target).parse_args(argv)
    target = DropTarget(
        frame_id=args.frame_id,
        x=args.drop_x,
        y=args.drop_y,
        z=args.drop_z,
        qx=args.qx,
        qy=args.qy,
        qz=args.qz,
        qw=args.qw,
    )
    try:
        target.validate()
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    return GoalOptions(
        target=target,
        server_timeout_s=args.server_timeout,
        send_timeout_s=args.send_timeout,
        result_timeout_s=args.result_timeout,
        require_complete=args.require_complete,
    )
