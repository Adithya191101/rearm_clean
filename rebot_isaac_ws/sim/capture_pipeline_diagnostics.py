#!/usr/bin/env python3
"""Record a synchronized presentation view of the live manipulation pipeline.

This runs inside the Isaac ROS workflow container while ``pick_scene.py`` runs
on the host with ``--record``. The video combines:

* a large, center-cropped passive Isaac Sim observer frame;
* Grounding DINO and FoundationPose reprojected onto that observer;
* an independent high, far-right live view of the full room; and
* robot-masked nvblox input depth from both fixed scene cameras.

Raw depth and ESDF statistics remain in the machine-readable manifest without
competing for screen space. Frames and the manifest are written to one output
directory. Encode the frames with ffmpeg after capture so an interrupted run
still leaves inspectable evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import cv2
from geometry_msgs.msg import Point, Vector3
from nvblox_msgs.srv import EsdfAndGradients
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from vision_msgs.msg import Detection2DArray, Detection3DArray

from pipeline_diagnostics_render import (
    crop_to_fill,
    decode_color_image,
    decode_mask_image,
    draw_detections,
    draw_observer_perception_overlay,
    draw_pose_overlay,
    extract_detections,
    extract_poses,
    pose_object_world_cylinder_points,
    stamp_seconds,
)
from presentation_views import (
    CAMERA_HORIZONTAL_APERTURE_MM,
    MAIN_CAMERA_EYE,
    MAIN_CAMERA_FOCAL_LENGTH_MM,
    MAIN_CAMERA_TARGET,
    PERCEPTION_CAMERA_EYE,
    PERCEPTION_CAMERA_TARGET,
)
from transfer_obstacle import (
    MINIMUM_WALL_CLEARANCE_M,
    TRANSFER_WALL,
    validate_wall_safety_state,
    wall_safety_failure_reason,
)


DEPTH_TOPICS = {
    "scene_cam_0": "/scene_cam_0/depth/image_raw",
    "scene_cam_1": "/scene_cam_1/depth/image_raw",
    "wrist": "/front_stereo_camera/depth/ground_truth",
    "masked_cam_0": "/cumotion/camera_1/world_depth",
    "masked_cam_1": "/cumotion/camera_2/world_depth",
}
ESDF_SERVICE = "/nvblox_node/get_esdf_and_gradient"
GROUNDING_DINO_IMAGE_TOPIC = "/object_detection_server/image_rect"
# The action-server result topics are advertised but never published by the
# upstream server implementation. Subscribe to the backend outputs that the
# servers themselves consume and return in their action results.
GROUNDING_DINO_DETECTIONS_TOPIC = "/detections"
FOUNDATIONPOSE_IMAGE_TOPIC = "/foundation_pose_server/image"
FOUNDATIONPOSE_CAMERA_INFO_TOPIC = "/foundation_pose_server/camera_info"
FOUNDATIONPOSE_POSE_TOPIC = "/pose_estimation/output"
FOUNDATIONPOSE_MASK_TOPICS = (
    "/foundation_pose_server/segmented_mask",
    "/segmentation",
)
SCENE_CAMERA_INFO_TOPIC = "/scene_cam_0/camera_info"
UNKNOWN_ESDF_VALUE = -1000.0
WORKSPACE_MIN = (-0.10, -0.35, -0.05)
WORKSPACE_MAX = (0.65, 0.60, 0.65)
WALL_MIN = TRANSFER_WALL.min_xyz_m
WALL_MAX = TRANSFER_WALL.max_xyz_m
NVBLOX_DISPLAY_MIN_Z = 0.17
PANEL_WIDTH = 480
PANEL_HEIGHT = 540
IMAGE_HEIGHT = 452
SIM_PANEL_WIDTH = 1440
MOSAIC_SIZE = (SIM_PANEL_WIDTH + PANEL_WIDTH, PANEL_HEIGHT * 2)
MASKED_VIEW_HEIGHT = 231
PERCEPTION_OVERLAY_HOLD_S = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sim-frame-dir", type=Path, required=True)
    parser.add_argument("--wide-sim-frame-dir", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=180.0)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--esdf-period", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=88)
    parser.add_argument(
        "--sim-state-file",
        type=Path,
        help="atomic wall-clearance/contact state written by pick_scene.py",
    )
    parser.add_argument(
        "--minimum-wall-clearance-m",
        type=float,
        default=MINIMUM_WALL_CLEARANCE_M,
        help="fail capture validation below this measured clearance",
    )
    parser.add_argument(
        "--stop-file",
        type=Path,
        help="finish cleanly when this file appears",
    )
    return parser.parse_args()


def decode_depth(message: Image) -> np.ndarray:
    encoding = message.encoding.upper()
    if encoding == "32FC1":
        dtype = np.dtype(">f4" if message.is_bigendian else "<f4")
        scale = 1.0
    elif encoding in ("16UC1", "MONO16"):
        dtype = np.dtype(">u2" if message.is_bigendian else "<u2")
        scale = 0.001
    else:
        raise ValueError(f"unsupported depth encoding: {message.encoding}")

    row_values = message.step // dtype.itemsize
    values = np.frombuffer(message.data, dtype=dtype)
    depth = values.reshape(message.height, row_values)[:, :message.width]
    return depth.astype(np.float32, copy=True) * scale


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image is None or image.size == 0:
        return np.full((height, width, 3), 24, dtype=np.uint8)
    source_h, source_w = image.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized_w = max(1, int(round(source_w * scale)))
    resized_h = max(1, int(round(source_h * scale)))
    resized = cv2.resize(
        image,
        (resized_w, resized_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    canvas = np.full((height, width, 3), 18, dtype=np.uint8)
    x = (width - resized_w) // 2
    y = (height - resized_h) // 2
    canvas[y:y + resized_h, x:x + resized_w] = resized
    return canvas


def text(
    image: np.ndarray,
    value: str,
    xy: tuple[int, int],
    *,
    scale: float = 0.52,
    color: tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
) -> None:
    max_width = image.shape[1] - xy[0] - 8
    (rendered_width, _), _ = cv2.getTextSize(
        value,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        thickness,
    )
    if rendered_width > max_width > 0:
        scale = max(0.30, scale * max_width / rendered_width)
    cv2.putText(
        image,
        value,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_panel(
    image: np.ndarray | None,
    title: str,
    details: list[str],
    *,
    alert: bool = False,
    crop_image: bool = False,
) -> np.ndarray:
    panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 13, dtype=np.uint8)
    title_color = (80, 160, 255) if alert else (90, 220, 170)
    text(panel, title, (14, 27), scale=0.67, color=title_color, thickness=2)
    panel[38:38 + IMAGE_HEIGHT] = (
        crop_to_fill(image, PANEL_WIDTH, IMAGE_HEIGHT)
        if crop_image
        else fit_image(image, PANEL_WIDTH, IMAGE_HEIGHT)
    )
    y = 511
    for line in details[:2]:
        text(panel, line, (12, y), scale=0.47)
        y += 20
    return panel


def make_sim_panel(
    image: np.ndarray | None,
    *,
    elapsed_s: float,
    age_s: float | None,
    wall_safety: dict | None,
    minimum_wall_clearance_m: float,
) -> np.ndarray:
    """Render the manipulation as the dominant, uncluttered video surface."""
    panel = crop_to_fill(image, SIM_PANEL_WIDTH, MOSAIC_SIZE[1])
    overlay = panel[0:124, 0:930]
    dimmed = np.full(overlay.shape, 8, dtype=np.uint8)
    cv2.addWeighted(overlay, 0.35, dimmed, 0.65, 0.0, dst=overlay)
    alert = age_s is None or age_s > 2.0
    title_color = (80, 160, 255) if alert else (90, 220, 170)
    text(
        panel,
        "Isaac Sim - live pick and place",
        (18, 34),
        scale=0.78,
        color=title_color,
        thickness=2,
    )
    text(
        panel,
        f"elapsed={elapsed_s:.1f}s  {age_label(age_s)}",
        (18, 66),
        scale=0.55,
    )
    if wall_safety is None:
        safety_label = "WALL SAFETY - waiting for simulation telemetry"
        safety_color = (80, 160, 255)
    else:
        current_mm = 1000.0 * wall_safety["sample"]["clearance_m"]
        minimum_mm = (
            1000.0 * wall_safety["minimum_observed"]["clearance_m"]
        )
        required_mm = 1000.0 * minimum_wall_clearance_m
        contacted = wall_safety["contact"]["ever"]
        below_requirement = minimum_mm + 1.0e-9 < required_mm
        safety_label = (
            f"WALL CLEARANCE {current_mm:.0f} mm  "
            f"MIN {minimum_mm:.0f} mm  REQ {required_mm:.0f} mm  |  "
            f"{'CONTACT' if contacted else 'NO CONTACT'}"
        )
        safety_color = (
            (70, 70, 245)
            if contacted or below_requirement
            else (90, 220, 170)
        )
    text(
        panel,
        safety_label,
        (18, 101),
        scale=0.61,
        color=safety_color,
        thickness=2,
    )
    return panel


def make_wide_panel(
    image: np.ndarray | None,
    *,
    age_s: float | None,
) -> np.ndarray:
    """Render the independent high room observer in the top-right tile."""
    return make_panel(
        image,
        "High wide room view - live",
        [age_label(age_s), "far-right observer"],
        alert=age_s is None or age_s > 2.0,
        crop_image=True,
    )


def render_depth(depth: np.ndarray | None) -> tuple[np.ndarray, dict]:
    if depth is None:
        return np.full((480, 640, 3), 24, dtype=np.uint8), {
            "valid_fraction": 0.0,
            "min_m": None,
            "max_m": None,
        }
    valid = np.isfinite(depth) & (depth > 0.01) & (depth < 10.0)
    stats = {
        "valid_fraction": float(valid.mean()),
        "min_m": float(np.min(depth[valid])) if np.any(valid) else None,
        "max_m": float(np.max(depth[valid])) if np.any(valid) else None,
    }
    normalized = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        low, high = np.percentile(depth[valid], [2.0, 98.0])
        if high - low < 0.05:
            high = low + 0.05
        scaled = 1.0 - np.clip((depth - low) / (high - low), 0.0, 1.0)
        normalized[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
    colored = cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO)
    colored[~valid] = (10, 10, 10)
    return colored, stats


def image_age(now: float, arrival: float | None) -> float | None:
    return None if arrival is None else max(0.0, now - arrival)


def age_label(age: float | None) -> str:
    return "no messages" if age is None else f"age={age:.2f}s"


class PipelineCapture(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(
            "pipeline_diagnostics_capture",
            parameter_overrides=[Parameter("use_sim_time", value=True)],
        )
        self.args = args
        self.depth: dict[str, np.ndarray] = {}
        self.depth_stamp: dict[str, float] = {}
        self.depth_arrival: dict[str, float] = {}
        self.depth_count = {name: 0 for name in DEPTH_TOPICS}
        self.decode_errors: dict[str, str] = {}
        self._depth_subscriptions = []
        for name, topic_name in DEPTH_TOPICS.items():
            self._depth_subscriptions.append(
                self.create_subscription(
                    Image,
                    topic_name,
                    lambda message, stream=name: self.on_depth(stream, message),
                    qos_profile_sensor_data,
                )
            )

        self.color: dict[str, np.ndarray] = {}
        self.color_stamp: dict[str, float] = {}
        self.color_arrival: dict[str, float] = {}
        self.color_count = {"grounding_dino": 0, "foundation_pose": 0}
        self.perception_errors: dict[str, str] = {}
        self.grounding_dino_detections: list[dict] = []
        self.grounding_dino_stamp: float | None = None
        self.grounding_dino_arrival: float | None = None
        self.grounding_dino_count = 0
        self.foundation_object_world_points: np.ndarray | None = None
        self.scene_camera_info: CameraInfo | None = None
        self.scene_camera_info_arrival: float | None = None
        self.scene_camera_info_count = 0
        self.foundation_poses: list[dict] = []
        self.foundation_pose_frame = ""
        self.foundation_pose_stamp: float | None = None
        self.foundation_pose_arrival: float | None = None
        self.foundation_pose_count = 0
        self.foundation_camera_info: CameraInfo | None = None
        self.foundation_camera_info_stamp: float | None = None
        self.foundation_camera_info_arrival: float | None = None
        self.foundation_camera_info_count = 0
        self.foundation_mask: np.ndarray | None = None
        self.foundation_mask_topic: str | None = None
        self.foundation_mask_stamp: float | None = None
        self.foundation_mask_arrival: float | None = None
        self.foundation_mask_count = {
            topic_name: 0 for topic_name in FOUNDATIONPOSE_MASK_TOPICS
        }
        self._perception_subscriptions = [
            self.create_subscription(
                Image,
                GROUNDING_DINO_IMAGE_TOPIC,
                lambda message: self.on_color("grounding_dino", message),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Detection2DArray,
                GROUNDING_DINO_DETECTIONS_TOPIC,
                self.on_grounding_dino_detections,
                10,
            ),
            self.create_subscription(
                CameraInfo,
                SCENE_CAMERA_INFO_TOPIC,
                self.on_scene_camera_info,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Image,
                FOUNDATIONPOSE_IMAGE_TOPIC,
                lambda message: self.on_color("foundation_pose", message),
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                CameraInfo,
                FOUNDATIONPOSE_CAMERA_INFO_TOPIC,
                self.on_foundation_camera_info,
                qos_profile_sensor_data,
            ),
            self.create_subscription(
                Detection3DArray,
                FOUNDATIONPOSE_POSE_TOPIC,
                self.on_foundation_pose,
                10,
            ),
        ]
        for topic_name in FOUNDATIONPOSE_MASK_TOPICS:
            self._perception_subscriptions.append(
                self.create_subscription(
                    Image,
                    topic_name,
                    lambda message, topic=topic_name: self.on_foundation_mask(
                        topic, message
                    ),
                    qos_profile_sensor_data,
                )
            )

        self.esdf_client = self.create_client(EsdfAndGradients, ESDF_SERVICE)
        self.esdf_future = None
        self.esdf_grid: np.ndarray | None = None
        self.esdf_origin: np.ndarray | None = None
        self.esdf_voxel: float | None = None
        self.esdf_stamp: float | None = None
        self.esdf_arrival: float | None = None
        self.esdf_error: str | None = None
        self.esdf_count = 0
        self.last_esdf_request = 0.0
        self.latest_sim_path: Path | None = None
        self.latest_sim_image: np.ndarray | None = None
        self.latest_sim_mtime: float | None = None
        self.latest_wide_sim_path: Path | None = None
        self.latest_wide_sim_image: np.ndarray | None = None
        self.latest_wide_sim_mtime: float | None = None
        self.sim_state: dict | None = None
        self.sim_state_mtime_ns: int | None = None
        self.sim_state_arrival: float | None = None
        self.sim_state_count = 0
        self.sim_state_error: str | None = None

    def on_depth(self, name: str, message: Image) -> None:
        try:
            self.depth[name] = decode_depth(message)
            self.depth_stamp[name] = stamp_seconds(message)
            self.depth_arrival[name] = time.monotonic()
            self.depth_count[name] += 1
            self.decode_errors.pop(name, None)
        except Exception as exc:  # noqa: BLE001
            self.decode_errors[name] = str(exc)

    def on_color(self, name: str, message: Image) -> None:
        try:
            self.color[name] = decode_color_image(message)
            self.color_stamp[name] = stamp_seconds(message)
            self.color_arrival[name] = time.monotonic()
            self.color_count[name] += 1
            self.perception_errors.pop(f"{name}_image", None)
        except Exception as exc:  # noqa: BLE001
            self.perception_errors[f"{name}_image"] = str(exc)

    def on_grounding_dino_detections(
        self,
        message: Detection2DArray,
    ) -> None:
        self.grounding_dino_detections = extract_detections(message)
        self.grounding_dino_stamp = stamp_seconds(message)
        self.grounding_dino_arrival = time.monotonic()
        self.grounding_dino_count += 1

    def on_scene_camera_info(self, message: CameraInfo) -> None:
        self.scene_camera_info = message
        self.scene_camera_info_arrival = time.monotonic()
        self.scene_camera_info_count += 1

    def on_foundation_camera_info(self, message: CameraInfo) -> None:
        self.foundation_camera_info = message
        self.foundation_camera_info_stamp = stamp_seconds(message)
        self.foundation_camera_info_arrival = time.monotonic()
        self.foundation_camera_info_count += 1

    def on_foundation_pose(self, message: Detection3DArray) -> None:
        self.foundation_poses = extract_poses(message)
        self.foundation_pose_frame = message.header.frame_id
        self.foundation_pose_stamp = stamp_seconds(message)
        self.foundation_pose_arrival = time.monotonic()
        self.foundation_pose_count += 1
        self.foundation_object_world_points = None

    def on_foundation_mask(self, topic_name: str, message: Image) -> None:
        try:
            self.foundation_mask = decode_mask_image(message)
            self.foundation_mask_topic = topic_name
            self.foundation_mask_stamp = stamp_seconds(message)
            self.foundation_mask_arrival = time.monotonic()
            self.foundation_mask_count[topic_name] += 1
            self.perception_errors.pop(f"mask:{topic_name}", None)
        except Exception as exc:  # noqa: BLE001
            self.perception_errors[f"mask:{topic_name}"] = str(exc)

    def request_esdf(self, now: float) -> None:
        if self.esdf_future is not None or now - self.last_esdf_request < self.args.esdf_period:
            return
        if not self.esdf_client.service_is_ready():
            self.esdf_error = "service unavailable"
            return
        request = EsdfAndGradients.Request()
        request.update_esdf = True
        request.visualize_esdf = True
        request.use_aabb = True
        request.frame_id = "base_link"
        request.aabb_min_m = Point(
            x=WORKSPACE_MIN[0],
            y=WORKSPACE_MIN[1],
            z=WORKSPACE_MIN[2],
        )
        request.aabb_size_m = Vector3(
            x=WORKSPACE_MAX[0] - WORKSPACE_MIN[0],
            y=WORKSPACE_MAX[1] - WORKSPACE_MIN[1],
            z=WORKSPACE_MAX[2] - WORKSPACE_MIN[2],
        )
        self.esdf_future = self.esdf_client.call_async(request)
        self.last_esdf_request = now

    def collect_esdf(self) -> None:
        if self.esdf_future is None or not self.esdf_future.done():
            return
        future = self.esdf_future
        self.esdf_future = None
        try:
            response = future.result()
            if response is None or not response.success:
                raise RuntimeError("service returned success=false")
            dimensions = [
                int(dimension.size)
                for dimension in response.esdf_and_gradients.layout.dim
            ]
            grid = np.asarray(
                response.esdf_and_gradients.data,
                dtype=np.float32,
            ).reshape(dimensions)
            if grid.ndim != 3:
                raise RuntimeError(f"expected 3-D ESDF, got shape={grid.shape}")
            self.esdf_grid = grid
            self.esdf_origin = np.asarray(
                [
                    response.origin_m.x,
                    response.origin_m.y,
                    response.origin_m.z,
                ],
                dtype=float,
            )
            self.esdf_voxel = float(response.voxel_size_m)
            self.esdf_stamp = (
                float(response.header.stamp.sec)
                + float(response.header.stamp.nanosec) * 1.0e-9
            )
            self.esdf_arrival = time.monotonic()
            self.esdf_count += 1
            self.esdf_error = None
        except Exception as exc:  # noqa: BLE001
            self.esdf_error = str(exc)

    @staticmethod
    def _read_latest_frame(
        frame_dir: Path,
        current_path: Path | None,
    ) -> tuple[Path | None, np.ndarray | None, float | None]:
        try:
            latest = max(frame_dir.glob("frame_*.jpg"))
        except (ValueError, FileNotFoundError):
            return current_path, None, None
        if latest == current_path:
            return current_path, None, None
        image = cv2.imread(str(latest), cv2.IMREAD_COLOR)
        if image is None:
            return current_path, None, None
        return latest, image, latest.stat().st_mtime

    def update_sim_images(self) -> None:
        path, image, mtime = self._read_latest_frame(
            self.args.sim_frame_dir,
            self.latest_sim_path,
        )
        if image is not None:
            self.latest_sim_path = path
            self.latest_sim_image = image
            self.latest_sim_mtime = mtime

        wide_path, wide_image, wide_mtime = self._read_latest_frame(
            self.args.wide_sim_frame_dir,
            self.latest_wide_sim_path,
        )
        if wide_image is not None:
            self.latest_wide_sim_path = wide_path
            self.latest_wide_sim_image = wide_image
            self.latest_wide_sim_mtime = wide_mtime

    def update_sim_state(self) -> None:
        path = self.args.sim_state_file
        if path is None:
            return
        try:
            mtime_ns = path.stat().st_mtime_ns
            if mtime_ns == self.sim_state_mtime_ns:
                return
            state = validate_wall_safety_state(
                json.loads(path.read_text(encoding="utf-8"))
            )
            self.sim_state = state
            self.sim_state_mtime_ns = mtime_ns
            self.sim_state_arrival = time.monotonic()
            self.sim_state_count += 1
            self.sim_state_error = None
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self.sim_state_error = str(exc)

    def update_perception_world_geometry(self) -> None:
        if (
            self.foundation_object_world_points is not None
            or not self.foundation_poses
            or self.foundation_pose_frame != "scene_cam_0"
        ):
            return
        points = pose_object_world_cylinder_points(
            self.foundation_poses[0],
            eye=PERCEPTION_CAMERA_EYE,
            target=PERCEPTION_CAMERA_TARGET,
        )
        if points is not None:
            self.foundation_object_world_points = points

    def render_esdf(self) -> tuple[np.ndarray, dict]:
        canvas = np.full((480, 640, 3), 20, dtype=np.uint8)
        if self.esdf_grid is None:
            text(canvas, "Waiting for nvblox ESDF...", (145, 235), scale=0.7)
            return canvas, {
                "shape": None,
                "occupied": 0,
                "near_surface": 0,
                "unknown": 0,
                "minimum": None,
                "maximum": None,
                "wall_occupied": 0,
                "wall_near_surface": 0,
                "negative_zero": 0,
                "elevated_surface": 0,
            }

        grid = self.esdf_grid
        known = np.isfinite(grid) & (grid != UNKNOWN_ESDF_VALUE)
        negative_zero = known & (grid == 0.0) & np.signbit(grid)
        occupied = known & ((grid < 0.0) | negative_zero)
        near_surface = known & (np.abs(grid) <= 0.0075)
        projected = np.full(grid.shape[:2], np.nan, dtype=np.float32)
        for x_index in range(grid.shape[0]):
            column = np.where(known[x_index], grid[x_index], np.nan)
            with np.errstate(all="ignore"):
                projected[x_index] = np.nanmin(column, axis=1)

        map_image = np.full((*projected.shape, 3), (36, 36, 40), dtype=np.uint8)
        known_xy = np.isfinite(projected)
        positive = known_xy & (projected > 0.0075)
        if np.any(positive):
            free = np.clip(projected / 0.20, 0.0, 1.0)
            map_image[..., 0][positive] = np.asarray(
                160.0 - 100.0 * free[positive], dtype=np.uint8
            )
            map_image[..., 1][positive] = np.asarray(
                80.0 + 110.0 * free[positive], dtype=np.uint8
            )
            map_image[..., 2][positive] = 35
        if self.esdf_origin is not None and self.esdf_voxel:
            z_centers = (
                self.esdf_origin[2]
                + (np.arange(grid.shape[2], dtype=np.float32) + 0.5)
                * self.esdf_voxel
            )
        else:
            z_centers = np.arange(grid.shape[2], dtype=np.float32)
        elevated = occupied & (
            z_centers[None, None, :] >= NVBLOX_DISPLAY_MIN_Z
        )
        elevated_heights = np.where(
            elevated,
            z_centers[None, None, :],
            -np.inf,
        ).max(axis=2)
        elevated_xy = np.isfinite(elevated_heights)
        if np.any(elevated_xy):
            normalized_height = np.zeros(
                elevated_heights.shape, dtype=np.float32)
            normalized_height[elevated_xy] = np.clip(
                (
                    elevated_heights[elevated_xy] - NVBLOX_DISPLAY_MIN_Z
                ) / (WORKSPACE_MAX[2] - NVBLOX_DISPLAY_MIN_Z),
                0.0,
                1.0,
            )
            height_colors = cv2.applyColorMap(
                np.asarray(normalized_height * 255.0, dtype=np.uint8),
                cv2.COLORMAP_TURBO,
            )
            map_image[elevated_xy] = height_colors[elevated_xy]
        map_image = np.flipud(np.transpose(map_image, (1, 0, 2)))

        plot_x0, plot_y0 = 64, 30
        plot_w, plot_h = 520, 410
        canvas[plot_y0:plot_y0 + plot_h, plot_x0:plot_x0 + plot_w] = cv2.resize(
            map_image,
            (plot_w, plot_h),
            interpolation=cv2.INTER_NEAREST,
        )

        def world_to_pixel(x_value: float, y_value: float) -> tuple[int, int]:
            x_fraction = (
                (x_value - WORKSPACE_MIN[0])
                / (WORKSPACE_MAX[0] - WORKSPACE_MIN[0])
            )
            y_fraction = (
                (y_value - WORKSPACE_MIN[1])
                / (WORKSPACE_MAX[1] - WORKSPACE_MIN[1])
            )
            return (
                int(round(plot_x0 + x_fraction * plot_w)),
                int(round(plot_y0 + (1.0 - y_fraction) * plot_h)),
            )

        wall_a = world_to_pixel(WALL_MIN[0], WALL_MIN[1])
        wall_b = world_to_pixel(WALL_MAX[0], WALL_MAX[1])
        cv2.rectangle(canvas, wall_a, wall_b, (255, 255, 255), 2)
        text(canvas, "transfer wall", (wall_a[0] + 5, wall_a[1] - 6), scale=0.42)
        cv2.rectangle(
            canvas,
            (plot_x0, plot_y0),
            (plot_x0 + plot_w, plot_y0 + plot_h),
            (120, 120, 120),
            1,
        )
        text(canvas, "x", (plot_x0 + plot_w + 10, plot_y0 + plot_h), scale=0.5)
        text(canvas, "y", (plot_x0 - 18, plot_y0 + 16), scale=0.5)
        text(
            canvas,
            "color = highest mapped surface above table",
            (155, 466),
            scale=0.44,
        )

        wall_occupied = 0
        wall_near = 0
        if self.esdf_origin is not None and self.esdf_voxel:
            indices = np.indices(grid.shape, dtype=np.float32)
            centers = [
                self.esdf_origin[axis]
                + (indices[axis] + 0.5) * self.esdf_voxel
                for axis in range(3)
            ]
            in_wall = (
                (centers[0] >= WALL_MIN[0] - self.esdf_voxel)
                & (centers[0] <= WALL_MAX[0] + self.esdf_voxel)
                & (centers[1] >= WALL_MIN[1] - self.esdf_voxel)
                & (centers[1] <= WALL_MAX[1] + self.esdf_voxel)
                & (centers[2] >= WALL_MIN[2] - self.esdf_voxel)
                & (centers[2] <= WALL_MAX[2] + self.esdf_voxel)
            )
            wall_occupied = int(np.count_nonzero(occupied & in_wall))
            wall_near = int(np.count_nonzero(near_surface & in_wall))

        known_values = grid[known]
        return canvas, {
            "shape": list(grid.shape),
            "occupied": int(np.count_nonzero(occupied)),
            "near_surface": int(np.count_nonzero(near_surface)),
            "unknown": int(np.count_nonzero(~known)),
            "minimum": (
                float(np.min(known_values)) if known_values.size else None
            ),
            "maximum": (
                float(np.max(known_values)) if known_values.size else None
            ),
            "wall_occupied": wall_occupied,
            "wall_near_surface": wall_near,
            "negative_zero": int(np.count_nonzero(negative_zero)),
            "elevated_surface": int(np.count_nonzero(elevated)),
        }

    def depth_stats(self, name: str, now: float) -> dict:
        _, stats = render_depth(self.depth.get(name))
        age = image_age(now, self.depth_arrival.get(name))
        stats.update({
            "age_s": age,
            "stamp_s": self.depth_stamp.get(name),
            "messages": self.depth_count[name],
            "error": self.decode_errors.get(name),
        })
        return stats

    def masked_panel(self, now: float) -> tuple[np.ndarray, dict]:
        left, left_stats = render_depth(self.depth.get("masked_cam_0"))
        right, right_stats = render_depth(self.depth.get("masked_cam_1"))
        ages = [
            image_age(now, self.depth_arrival.get("masked_cam_0")),
            image_age(now, self.depth_arrival.get("masked_cam_1")),
        ]
        stats = {
            "cam_0": {
                **left_stats,
                "age_s": ages[0],
                "messages": self.depth_count["masked_cam_0"],
            },
            "cam_1": {
                **right_stats,
                "age_s": ages[1],
                "messages": self.depth_count["masked_cam_1"],
            },
        }
        alert = any(age is None or age > 2.0 for age in ages)

        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), 13, dtype=np.uint8)
        title_color = (80, 160, 255) if alert else (90, 220, 170)
        text(
            panel,
            "Robot-masked depth to nvblox",
            (14, 27),
            scale=0.67,
            color=title_color,
            thickness=2,
        )
        top_y = 38
        bottom_y = top_y + MASKED_VIEW_HEIGHT
        panel[top_y:bottom_y] = crop_to_fill(
            left, PANEL_WIDTH, MASKED_VIEW_HEIGHT)
        panel[bottom_y:bottom_y + MASKED_VIEW_HEIGHT] = crop_to_fill(
            right, PANEL_WIDTH, MASKED_VIEW_HEIGHT)
        cv2.line(
            panel,
            (0, bottom_y),
            (PANEL_WIDTH, bottom_y),
            (235, 235, 235),
            2,
        )
        for label, y in (("scene camera 0", top_y), ("scene camera 1", bottom_y)):
            overlay = panel[y:y + 32, 0:190]
            dimmed = np.full(overlay.shape, 8, dtype=np.uint8)
            cv2.addWeighted(overlay, 0.35, dimmed, 0.65, 0.0, dst=overlay)
            text(panel, label, (10, y + 23), scale=0.52, thickness=2)
        text(
            panel,
            (
                f"cam0 {100 * left_stats['valid_fraction']:.1f}%  "
                f"cam1 {100 * right_stats['valid_fraction']:.1f}% valid"
            ),
            (12, 523),
            scale=0.47,
        )
        return panel, stats

    def grounding_dino_panel(
        self,
        now: float,
    ) -> tuple[np.ndarray, dict]:
        image_arrival = self.color_arrival.get("grounding_dino")
        image_snapshot_age = image_age(now, image_arrival)
        detection_age = image_age(now, self.grounding_dino_arrival)
        rendered = draw_detections(
            self.color.get("grounding_dino"),
            self.grounding_dino_detections,
        )
        best = (
            max(
                self.grounding_dino_detections,
                key=lambda item: (
                    item["score"] if item["score"] is not None else -1.0
                ),
            )
            if self.grounding_dino_detections
            else None
        )
        best_label = "no boxes"
        if best is not None:
            score = (
                "n/a" if best["score"] is None else f"{best['score']:.2f}"
            )
            best_label = f"best={best['class_id']} score={score}"
        stats = {
            "image": {
                "topic": GROUNDING_DINO_IMAGE_TOPIC,
                "age_s": image_snapshot_age,
                "stamp_s": self.color_stamp.get("grounding_dino"),
                "messages": self.color_count["grounding_dino"],
                "error": self.perception_errors.get("grounding_dino_image"),
            },
            "detections": {
                "topic": GROUNDING_DINO_DETECTIONS_TOPIC,
                "age_s": detection_age,
                "stamp_s": self.grounding_dino_stamp,
                "messages": self.grounding_dino_count,
                "count": len(self.grounding_dino_detections),
                "items": self.grounding_dino_detections,
            },
        }
        details = [
            (
                f"image {age_label(image_snapshot_age)}  "
                f"detections {age_label(detection_age)}"
            ),
            (
                f"boxes={len(self.grounding_dino_detections)}  "
                f"snapshots={self.grounding_dino_count}  {best_label}"
            ),
        ]
        return make_panel(
            rendered,
            "Grounding DINO - 2-D detections",
            details,
            alert=image_snapshot_age is None or detection_age is None,
            crop_image=True,
        ), stats

    def foundation_pose_panel(
        self,
        now: float,
    ) -> tuple[np.ndarray, dict]:
        image_snapshot_age = image_age(
            now, self.color_arrival.get("foundation_pose"))
        pose_age = image_age(now, self.foundation_pose_arrival)
        camera_info_age = image_age(
            now, self.foundation_camera_info_arrival)
        mask_age = image_age(now, self.foundation_mask_arrival)
        rendered, axes_drawn = draw_pose_overlay(
            self.color.get("foundation_pose"),
            self.foundation_poses,
            self.foundation_camera_info,
            self.foundation_mask,
        )
        mask_fraction = (
            None
            if self.foundation_mask is None
            else float(np.count_nonzero(self.foundation_mask))
            / float(self.foundation_mask.size)
        )
        stats = {
            "image": {
                "topic": FOUNDATIONPOSE_IMAGE_TOPIC,
                "age_s": image_snapshot_age,
                "stamp_s": self.color_stamp.get("foundation_pose"),
                "messages": self.color_count["foundation_pose"],
                "error": self.perception_errors.get("foundation_pose_image"),
            },
            "pose": {
                "topic": FOUNDATIONPOSE_POSE_TOPIC,
                "frame_id": self.foundation_pose_frame,
                "age_s": pose_age,
                "stamp_s": self.foundation_pose_stamp,
                "messages": self.foundation_pose_count,
                "count": len(self.foundation_poses),
                "axes_drawn": axes_drawn,
                "items": self.foundation_poses,
            },
            "camera_info": {
                "topic": FOUNDATIONPOSE_CAMERA_INFO_TOPIC,
                "age_s": camera_info_age,
                "stamp_s": self.foundation_camera_info_stamp,
                "messages": self.foundation_camera_info_count,
            },
            "mask": {
                "topic": self.foundation_mask_topic,
                "age_s": mask_age,
                "stamp_s": self.foundation_mask_stamp,
                "messages": self.foundation_mask_count,
                "nonzero_fraction": mask_fraction,
            },
        }
        pose_label = (
            "waiting for 6-DoF pose"
            if not self.foundation_poses
            else (
                "xyz=(%.3f,%.3f,%.3f)m q=(%.2f,%.2f,%.2f,%.2f)"
                % (
                    *self.foundation_poses[0]["position_m"],
                    *self.foundation_poses[0]["orientation_xyzw"],
                )
            )
        )
        details = [
            (
                f"image {age_label(image_snapshot_age)}  "
                f"pose {age_label(pose_age)}  mask {age_label(mask_age)}"
            ),
            pose_label,
        ]
        return make_panel(
            rendered,
            "FoundationPose - mask + 6-DoF",
            details,
            alert=(
                image_snapshot_age is None
                or pose_age is None
                or camera_info_age is None
                or axes_drawn == 0
            ),
            crop_image=True,
        ), stats

    def compose(self, frame_number: int, started: float) -> tuple[np.ndarray, dict]:
        now = time.monotonic()
        self.update_sim_images()
        self.update_sim_state()
        self.update_perception_world_geometry()
        _, esdf_stats = self.render_esdf()

        sim_age = (
            None
            if self.latest_sim_mtime is None
            else max(0.0, time.time() - self.latest_sim_mtime)
        )
        sim_panel = make_sim_panel(
            self.latest_sim_image,
            elapsed_s=now - started,
            age_s=sim_age,
            wall_safety=self.sim_state,
            minimum_wall_clearance_m=self.args.minimum_wall_clearance_m,
        )
        _, grounding_dino_stats = (
            self.grounding_dino_panel(now)
        )
        _, foundation_pose_stats = (
            self.foundation_pose_panel(now)
        )
        wide_sim_age = (
            None
            if self.latest_wide_sim_mtime is None
            else max(0.0, time.time() - self.latest_wide_sim_mtime)
        )
        wide_sim_panel = make_wide_panel(
            self.latest_wide_sim_image,
            age_s=wide_sim_age,
        )
        wrist_stats = self.depth_stats("wrist", now)
        esdf_age = image_age(now, self.esdf_arrival)
        cam0_stats = self.depth_stats("scene_cam_0", now)
        cam1_stats = self.depth_stats("scene_cam_1", now)
        masked_panel, masked_stats = self.masked_panel(now)

        foundation_pose_ready = (
            bool(self.foundation_poses)
            and self.foundation_camera_info is not None
        )
        overlay_arrival = (
            self.foundation_pose_arrival
            if foundation_pose_ready
            else self.grounding_dino_arrival
        )
        overlay_age = image_age(now, overlay_arrival)
        overlay_active = (
            overlay_age is not None
            and overlay_age <= PERCEPTION_OVERLAY_HOLD_S
        )
        best_detection = (
            max(
                self.grounding_dino_detections,
                key=lambda item: (
                    item["score"] if item["score"] is not None else -1.0
                ),
            )
            if self.grounding_dino_detections
            else None
        )
        observer_overlay_stats = {
            "active": overlay_active,
            "age_s": overlay_age,
            "hold_s": PERCEPTION_OVERLAY_HOLD_S,
            "detection_box_drawn": False,
            "detection_box_source": "foundationpose_cylinder_outline",
            "pose_axes_drawn": 0,
        }
        if overlay_active and self.latest_sim_image is not None:
            sim_panel, drawn = draw_observer_perception_overlay(
                sim_panel,
                source_size=(
                    self.latest_sim_image.shape[1],
                    self.latest_sim_image.shape[0],
                ),
                detection=best_detection,
                object_world_points=self.foundation_object_world_points,
                pose=(
                    self.foundation_poses[0]
                    if foundation_pose_ready
                    and self.foundation_pose_frame == "scene_cam_0"
                    else None
                ),
                perception_eye=PERCEPTION_CAMERA_EYE,
                perception_target=PERCEPTION_CAMERA_TARGET,
                observer_eye=MAIN_CAMERA_EYE,
                observer_target=MAIN_CAMERA_TARGET,
                observer_focal_length_mm=MAIN_CAMERA_FOCAL_LENGTH_MM,
                observer_horizontal_aperture_mm=(
                    CAMERA_HORIZONTAL_APERTURE_MM
                ),
            )
            observer_overlay_stats.update(drawn)

        mosaic = np.hstack([
            sim_panel,
            np.vstack([
                wide_sim_panel,
                masked_panel,
            ]),
        ])
        assert mosaic.shape[1::-1] == MOSAIC_SIZE
        manifest = {
            "frame": frame_number,
            "elapsed_s": now - started,
            "sim": {
                "path": (
                    str(self.latest_sim_path)
                    if self.latest_sim_path is not None
                    else None
                ),
                "age_s": sim_age,
                "high_wide": {
                    "path": (
                        str(self.latest_wide_sim_path)
                        if self.latest_wide_sim_path is not None
                        else None
                    ),
                    "age_s": wide_sim_age,
                },
                "perception_overlay": observer_overlay_stats,
            },
            "wall_safety": {
                "path": (
                    str(self.args.sim_state_file)
                    if self.args.sim_state_file is not None
                    else None
                ),
                "age_s": image_age(now, self.sim_state_arrival),
                "updates": self.sim_state_count,
                "error": self.sim_state_error,
                "state": self.sim_state,
            },
            "perception": {
                "display": "main_observer_overlay",
                "grounding_dino": grounding_dino_stats,
                "foundation_pose": foundation_pose_stats,
                "scene_camera_info": {
                    "topic": SCENE_CAMERA_INFO_TOPIC,
                    "age_s": image_age(
                        now, self.scene_camera_info_arrival),
                    "messages": self.scene_camera_info_count,
                },
                "errors": self.perception_errors,
            },
            "depth": {
                "wrist": wrist_stats,
                "scene_cam_0": cam0_stats,
                "scene_cam_1": cam1_stats,
                "masked": masked_stats,
            },
            "esdf": {
                **esdf_stats,
                "age_s": esdf_age,
                "stamp_s": self.esdf_stamp,
                "responses": self.esdf_count,
                "error": self.esdf_error,
            },
        }
        return mosaic, manifest


def main() -> int:
    args = parse_args()
    if (
        args.duration <= 0.0
        or args.fps <= 0.0
        or args.esdf_period <= 0.0
        or args.minimum_wall_clearance_m < 0.0
    ):
        raise ValueError("duration, fps, and esdf-period must be positive")
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError("jpeg-quality must be in [1, 100]")

    output_dir = args.output_dir
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    args.sim_frame_dir.mkdir(parents=True, exist_ok=True)
    args.wide_sim_frame_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "capture_manifest.jsonl"
    summary_path = output_dir / "capture_summary.json"
    ready_path = output_dir / "capture_ready.json"

    rclpy.init()
    node = PipelineCapture(args)
    started = time.monotonic()
    next_frame = started
    frame_period = 1.0 / args.fps
    frame_number = 0
    final_manifest = None
    stopped_by_file = False

    if not node.esdf_client.wait_for_service(timeout_sec=30.0):
        node.get_logger().warning(f"{ESDF_SERVICE} not available at capture start")

    ready_path.write_text(
        json.dumps({"ready": True, "started_unix_s": time.time()}) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("w", encoding="utf-8") as manifest_file:
        while time.monotonic() - started < args.duration:
            if args.stop_file is not None and args.stop_file.exists():
                stopped_by_file = True
                break
            now = time.monotonic()
            node.request_esdf(now)
            rclpy.spin_once(node, timeout_sec=0.02)
            node.collect_esdf()
            now = time.monotonic()
            if now < next_frame:
                continue
            mosaic, frame_manifest = node.compose(frame_number, started)
            frame_path = frames_dir / f"frame_{frame_number:05d}.jpg"
            cv2.imwrite(
                str(frame_path),
                mosaic,
                [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality],
            )
            manifest_file.write(json.dumps(frame_manifest, sort_keys=True) + "\n")
            manifest_file.flush()
            final_manifest = frame_manifest
            frame_number += 1
            next_frame += frame_period
            if next_frame < now - frame_period:
                next_frame = now + frame_period

    wall_safety_error = None
    if args.sim_state_file is not None:
        if node.sim_state is None:
            wall_safety_error = (
                "no valid wall-safety telemetry was received from the simulator"
            )
        else:
            wall_safety_error = wall_safety_failure_reason(
                node.sim_state,
                minimum_clearance_m=args.minimum_wall_clearance_m,
            )
            state_age = image_age(time.monotonic(), node.sim_state_arrival)
            if state_age is None or state_age > 2.0:
                wall_safety_error = (
                    f"wall-safety telemetry is stale ({state_age!r} s)"
                )

    summary = {
        "frames": frame_number,
        "fps": args.fps,
        "duration_requested_s": args.duration,
        "duration_actual_s": time.monotonic() - started,
        "stopped_by_file": stopped_by_file,
        "mosaic_width": MOSAIC_SIZE[0],
        "mosaic_height": MOSAIC_SIZE[1],
        "depth_messages": node.depth_count,
        "perception_messages": {
            "grounding_dino_image": node.color_count["grounding_dino"],
            "grounding_dino_detections": node.grounding_dino_count,
            "foundation_pose_image": node.color_count["foundation_pose"],
            "foundation_pose_output": node.foundation_pose_count,
            "foundation_pose_camera_info": node.foundation_camera_info_count,
            "foundation_pose_masks": node.foundation_mask_count,
            "scene_camera_info": node.scene_camera_info_count,
        },
        "esdf_responses": node.esdf_count,
        "wall_safety": {
            "path": (
                str(args.sim_state_file)
                if args.sim_state_file is not None
                else None
            ),
            "updates": node.sim_state_count,
            "minimum_required_m": args.minimum_wall_clearance_m,
            "passed": wall_safety_error is None,
            "error": wall_safety_error,
            "final_state": node.sim_state,
        },
        "final": final_manifest,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    node.destroy_node()
    rclpy.shutdown()
    print(
        f"CAPTURE DONE: {frame_number} frames -> {frames_dir}",
        flush=True,
    )
    if wall_safety_error is not None:
        print(f"WALL SAFETY FAILED: {wall_safety_error}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
