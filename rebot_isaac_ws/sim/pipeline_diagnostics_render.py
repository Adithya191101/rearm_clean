"""ROS-message decoding and overlays for the pipeline diagnostics recorder."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


SOUP_CAN_MODEL_RADIUS_M = 0.034
SOUP_CAN_MODEL_HALF_HEIGHT_M = 0.051
SOUP_CAN_MODEL_HALF_EXTENT_M = (
    SOUP_CAN_MODEL_RADIUS_M,
    SOUP_CAN_MODEL_HALF_HEIGHT_M,
    SOUP_CAN_MODEL_RADIUS_M,
)


def crop_to_fill(
    image: np.ndarray | None,
    width: int,
    height: int,
    *,
    empty_value: int = 24,
) -> np.ndarray:
    """Resize and center-crop an image to fill an exact BGR frame."""
    if width <= 0 or height <= 0:
        raise ValueError("output dimensions must be positive")
    if image is None or image.size == 0:
        return np.full((height, width, 3), empty_value, dtype=np.uint8)

    source_height, source_width = image.shape[:2]
    scale = max(width / source_width, height / source_height)
    resized_width = max(width, int(round(source_width * scale)))
    resized_height = max(height, int(round(source_height * scale)))
    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )
    x = (resized_width - width) // 2
    y = (resized_height - height) // 2
    return np.ascontiguousarray(resized[y:y + height, x:x + width])


def stamp_seconds(message: Any) -> float:
    """Return a ROS message header stamp as fractional seconds."""
    stamp = message.header.stamp
    return float(stamp.sec) + float(stamp.nanosec) * 1.0e-9


def _byte_rows(message: Any, packed_row_bytes: int) -> np.ndarray:
    height = int(message.height)
    step = int(message.step)
    if height <= 0 or packed_row_bytes <= 0:
        raise ValueError("image dimensions must be positive")
    if step < packed_row_bytes:
        raise ValueError(
            f"image step {step} is smaller than packed row {packed_row_bytes}"
        )
    values = np.frombuffer(message.data, dtype=np.uint8)
    required = height * step
    if values.size < required:
        raise ValueError(
            f"image data has {values.size} bytes, expected at least {required}"
        )
    return values[:required].reshape(height, step)[:, :packed_row_bytes]


def decode_color_image(message: Any) -> np.ndarray:
    """Decode common sensor_msgs/Image color encodings into OpenCV BGR."""
    encoding = str(message.encoding).lower()
    channels_by_encoding = {
        "rgb8": 3,
        "bgr8": 3,
        "rgba8": 4,
        "bgra8": 4,
        "mono8": 1,
        "8uc1": 1,
        "8uc3": 3,
        "8uc4": 4,
    }
    channels = channels_by_encoding.get(encoding)
    if channels is None:
        raise ValueError(f"unsupported color encoding: {message.encoding}")

    width = int(message.width)
    rows = _byte_rows(message, width * channels)
    pixels = np.ascontiguousarray(
        rows.reshape(int(message.height), width, channels)
    )
    if encoding == "rgb8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    if encoding == "rgba8":
        return cv2.cvtColor(pixels, cv2.COLOR_RGBA2BGR)
    if encoding in ("bgra8", "8uc4"):
        return cv2.cvtColor(pixels, cv2.COLOR_BGRA2BGR)
    if channels == 1:
        return cv2.cvtColor(pixels[:, :, 0], cv2.COLOR_GRAY2BGR)
    return pixels.copy()


def decode_mask_image(message: Any) -> np.ndarray:
    """Decode a segmentation image into a binary uint8 mask."""
    encoding = str(message.encoding).lower()
    if encoding in ("rgb8", "bgr8", "rgba8", "bgra8", "8uc3", "8uc4"):
        color = decode_color_image(message)
        return np.asarray(np.any(color != 0, axis=2), dtype=np.uint8) * 255

    scalar_encodings = {
        "mono8": (np.dtype("u1"), 1),
        "8uc1": (np.dtype("u1"), 1),
        "mono16": (
            np.dtype(">u2" if message.is_bigendian else "<u2"),
            2,
        ),
        "16uc1": (
            np.dtype(">u2" if message.is_bigendian else "<u2"),
            2,
        ),
        "32fc1": (
            np.dtype(">f4" if message.is_bigendian else "<f4"),
            4,
        ),
    }
    spec = scalar_encodings.get(encoding)
    if spec is None:
        raise ValueError(f"unsupported mask encoding: {message.encoding}")

    dtype, item_size = spec
    width = int(message.width)
    byte_rows = _byte_rows(message, width * item_size)
    values = np.ascontiguousarray(byte_rows).view(dtype).reshape(
        int(message.height), width
    )
    valid = np.isfinite(values) & (values > 0)
    return np.asarray(valid, dtype=np.uint8) * 255


def extract_detections(message: Any) -> list[dict]:
    """Convert Detection2DArray entries into JSON-safe overlay records."""
    records = []
    for detection in message.detections:
        center = detection.bbox.center.position
        width = float(detection.bbox.size_x)
        height = float(detection.bbox.size_y)
        result = (
            max(
                detection.results,
                key=lambda item: float(item.hypothesis.score),
            )
            if detection.results
            else None
        )
        class_id = (
            str(result.hypothesis.class_id) if result is not None else "unknown"
        )
        score = (
            float(result.hypothesis.score) if result is not None else None
        )
        records.append({
            "id": str(getattr(detection, "id", "")),
            "class_id": class_id,
            "score": score,
            "center_x": float(center.x),
            "center_y": float(center.y),
            "width": width,
            "height": height,
            "x_min": float(center.x) - 0.5 * width,
            "y_min": float(center.y) - 0.5 * height,
            "x_max": float(center.x) + 0.5 * width,
            "y_max": float(center.y) + 0.5 * height,
        })
    return records


def draw_detections(
    image: np.ndarray | None,
    detections: list[dict],
) -> np.ndarray | None:
    """Draw Grounding DINO boxes and scores over the source image."""
    if image is None:
        return None
    rendered = image.copy()
    height, width = rendered.shape[:2]
    for index, detection in enumerate(detections):
        x0 = int(round(np.clip(detection["x_min"], 0, width - 1)))
        y0 = int(round(np.clip(detection["y_min"], 0, height - 1)))
        x1 = int(round(np.clip(detection["x_max"], 0, width - 1)))
        y1 = int(round(np.clip(detection["y_max"], 0, height - 1)))
        color = ((40 + 70 * index) % 220, 215, 255)
        cv2.rectangle(rendered, (x0, y0), (x1, y1), color, 3)
        label = detection["class_id"]
        if detection["score"] is not None:
            label += f" {detection['score']:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            2,
        )
        label_y = max(label_height + baseline + 3, y0)
        cv2.rectangle(
            rendered,
            (x0, label_y - label_height - baseline - 4),
            (min(width - 1, x0 + label_width + 8), label_y + 2),
            color,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (x0 + 4, label_y - baseline - 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
    return rendered


def extract_poses(message: Any) -> list[dict]:
    """Convert Detection3DArray results into JSON-safe 6-DoF records."""
    records = []
    for detection in message.detections:
        if not detection.results:
            continue
        result = max(
            detection.results,
            key=lambda item: float(item.hypothesis.score),
        )
        pose = result.pose.pose
        records.append({
            "id": str(getattr(detection, "id", "")),
            "class_id": str(result.hypothesis.class_id),
            "score": float(result.hypothesis.score),
            "position_m": [
                float(pose.position.x),
                float(pose.position.y),
                float(pose.position.z),
            ],
            "orientation_xyzw": [
                float(pose.orientation.x),
                float(pose.orientation.y),
                float(pose.orientation.z),
                float(pose.orientation.w),
            ],
        })
    return records


def quaternion_rotation_matrix(orientation_xyzw: list[float]) -> np.ndarray:
    """Return the normalized quaternion's 3x3 rotation matrix."""
    x, y, z, w = np.asarray(orientation_xyzw, dtype=float)
    norm = float(np.linalg.norm([x, y, z, w]))
    if not np.isfinite(norm) or norm < 1.0e-12:
        raise ValueError("pose quaternion must be finite and non-zero")
    x, y, z, w = (np.asarray([x, y, z, w]) / norm).tolist()
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w),
         2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z),
         2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w),
         1.0 - 2.0 * (x * x + y * y)],
    ])


def look_at_camera_basis(
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return world-space right, up, and forward axes for a Z-up camera."""
    eye_array = np.asarray(eye, dtype=float)
    forward = np.asarray(target, dtype=float) - eye_array
    forward_norm = float(np.linalg.norm(forward))
    if not np.isfinite(forward_norm) or forward_norm < 1.0e-9:
        raise ValueError("camera eye and target must be distinct")
    forward /= forward_norm
    right = np.cross(forward, np.asarray([0.0, 0.0, 1.0]))
    right_norm = float(np.linalg.norm(right))
    if not np.isfinite(right_norm) or right_norm < 1.0e-9:
        raise ValueError("camera view cannot be parallel to world up")
    right /= right_norm
    up = np.cross(right, forward)
    return right, up, forward


def optical_points_to_world(
    points: np.ndarray,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> np.ndarray:
    """Transform ROS optical points (+X right, +Y down, +Z forward) to world."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("optical points must have shape (N, 3)")
    right, up, forward = look_at_camera_basis(eye, target)
    optical_to_world = np.column_stack((right, -up, forward))
    return values @ optical_to_world.T + np.asarray(eye, dtype=float)


def project_world_points(
    points: np.ndarray,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    focal_length_mm: float,
    horizontal_aperture_mm: float,
    image_size: tuple[int, int],
) -> np.ndarray:
    """Project world points into a square-pixel pinhole observer image."""
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError("world points must have shape (N, 3)")
    width, height = image_size
    if (
        width <= 0
        or height <= 0
        or focal_length_mm <= 0.0
        or horizontal_aperture_mm <= 0.0
    ):
        raise ValueError("camera dimensions and filmback must be positive")
    right, up, forward = look_at_camera_basis(eye, target)
    relative = values - np.asarray(eye, dtype=float)
    camera_x = relative @ right
    camera_y = relative @ up
    camera_z = relative @ forward
    focal_pixels = focal_length_mm * width / horizontal_aperture_mm
    projected = np.full((values.shape[0], 2), np.nan, dtype=float)
    visible = np.isfinite(camera_z) & (camera_z > 1.0e-6)
    projected[visible, 0] = (
        0.5 * width
        + focal_pixels * camera_x[visible] / camera_z[visible]
    )
    projected[visible, 1] = (
        0.5 * height
        - focal_pixels * camera_y[visible] / camera_z[visible]
    )
    return projected


def map_center_crop_points(
    points: np.ndarray,
    *,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Map source-image coordinates through ``crop_to_fill``."""
    source_width, source_height = source_size
    output_width, output_height = output_size
    if min(source_width, source_height, output_width, output_height) <= 0:
        raise ValueError("crop dimensions must be positive")
    scale = max(
        output_width / source_width,
        output_height / source_height,
    )
    offset_x = 0.5 * (source_width * scale - output_width)
    offset_y = 0.5 * (source_height * scale - output_height)
    mapped = np.asarray(points, dtype=float).copy()
    mapped[:, 0] = mapped[:, 0] * scale - offset_x
    mapped[:, 1] = mapped[:, 1] * scale - offset_y
    return mapped


def axis_aligned_projected_box(
    projected_corners: np.ndarray,
    projected_center: np.ndarray,
    *,
    image_size: tuple[int, int],
    minimum_extent_px: int = 4,
) -> tuple[int, int, int, int] | None:
    """Bound projected 3-D corners with a validated image-axis rectangle."""
    corners = np.asarray(projected_corners, dtype=float)
    center = np.asarray(projected_center, dtype=float)
    width, height = image_size
    if (
        corners.ndim != 2
        or corners.shape[1] != 2
        or corners.shape[0] < 2
        or center.shape not in ((2,), (1, 2))
        or width <= 0
        or height <= 0
        or minimum_extent_px < 1
        or not np.all(np.isfinite(corners))
        or not np.all(np.isfinite(center))
    ):
        return None
    center = center.reshape(2)
    minimum = np.min(corners, axis=0)
    maximum = np.max(corners, axis=0)
    if (
        not 0.0 <= center[0] < width
        or not 0.0 <= center[1] < height
        or np.any(center < minimum)
        or np.any(center > maximum)
    ):
        return None
    x0 = int(np.clip(round(minimum[0]), 0, width - 1))
    y0 = int(np.clip(round(minimum[1]), 0, height - 1))
    x1 = int(np.clip(round(maximum[0]), 0, width - 1))
    y1 = int(np.clip(round(maximum[1]), 0, height - 1))
    if x1 - x0 < minimum_extent_px or y1 - y0 < minimum_extent_px:
        return None
    return x0, y0, x1, y1


def pose_object_world_corners(
    pose: dict,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    half_extent_xyz_m: tuple[float, float, float] = (
        SOUP_CAN_MODEL_HALF_EXTENT_M
    ),
) -> np.ndarray | None:
    """Transform a FoundationPose object extent into world-space corners."""
    try:
        origin = np.asarray(pose["position_m"], dtype=float)
        rotation = quaternion_rotation_matrix(pose["orientation_xyzw"])
    except (KeyError, TypeError, ValueError):
        return None
    half_extent = np.asarray(half_extent_xyz_m, dtype=float)
    if (
        origin.shape != (3,)
        or half_extent.shape != (3,)
        or not np.all(np.isfinite(origin))
        or not np.all(np.isfinite(half_extent))
        or np.any(half_extent <= 0.0)
    ):
        return None
    signs = np.asarray([
        [-1.0, -1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, 1.0, 1.0],
        [1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
    ])
    optical = signs * half_extent
    optical = optical @ rotation.T + origin
    return optical_points_to_world(optical, eye=eye, target=target)


def pose_object_world_cylinder_points(
    pose: dict,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    radius_m: float = SOUP_CAN_MODEL_RADIUS_M,
    half_height_m: float = SOUP_CAN_MODEL_HALF_HEIGHT_M,
    circumference_samples: int = 72,
) -> np.ndarray | None:
    """Transform the physical can cylinder outline into world-space points."""
    try:
        origin = np.asarray(pose["position_m"], dtype=float)
        rotation = quaternion_rotation_matrix(pose["orientation_xyzw"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        origin.shape != (3,)
        or not np.all(np.isfinite(origin))
        or not np.isfinite(radius_m)
        or not np.isfinite(half_height_m)
        or radius_m <= 0.0
        or half_height_m <= 0.0
        or circumference_samples < 8
    ):
        return None

    # The FoundationPose soup-can model is axial along local Y. Sampling both
    # end rings gives the exact projected convex outline without the diagonal
    # overreach of a rectangular mesh extent.
    angles = np.linspace(
        0.0,
        2.0 * np.pi,
        num=circumference_samples,
        endpoint=False,
    )
    ring_x = radius_m * np.cos(angles)
    ring_z = radius_m * np.sin(angles)
    optical = np.vstack([
        np.column_stack((
            ring_x,
            np.full(circumference_samples, y),
            ring_z,
        ))
        for y in (-half_height_m, half_height_m)
    ])
    optical = optical @ rotation.T + origin
    return optical_points_to_world(optical, eye=eye, target=target)


def pose_axes_world_points(
    pose: dict,
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
    axis_length_m: float = 0.08,
) -> np.ndarray | None:
    """Transform a FoundationPose triad from its optical frame into world."""
    if axis_length_m <= 0.0:
        return None
    try:
        origin = np.asarray(pose["position_m"], dtype=float)
        rotation = quaternion_rotation_matrix(pose["orientation_xyzw"])
    except (KeyError, TypeError, ValueError):
        return None
    if origin.shape != (3,) or not np.all(np.isfinite(origin)):
        return None
    optical = np.vstack((
        origin,
        origin + rotation[:, 0] * axis_length_m,
        origin + rotation[:, 1] * axis_length_m,
        origin + rotation[:, 2] * axis_length_m,
    ))
    return optical_points_to_world(optical, eye=eye, target=target)


def draw_observer_perception_overlay(
    image: np.ndarray,
    *,
    source_size: tuple[int, int],
    detection: dict | None,
    object_world_points: np.ndarray | None,
    pose: dict | None,
    perception_eye: tuple[float, float, float],
    perception_target: tuple[float, float, float],
    observer_eye: tuple[float, float, float],
    observer_target: tuple[float, float, float],
    observer_focal_length_mm: float,
    observer_horizontal_aperture_mm: float,
) -> tuple[np.ndarray, dict]:
    """Draw cross-camera DINO and FoundationPose results on an observer view."""
    rendered = image.copy()
    output_size = (rendered.shape[1], rendered.shape[0])
    stats = {
        "detection_box_drawn": False,
        "detection_box_source": "foundationpose_cylinder_outline",
        "pose_axes_drawn": 0,
    }

    def project(points: np.ndarray) -> np.ndarray:
        projected = project_world_points(
            points,
            eye=observer_eye,
            target=observer_target,
            focal_length_mm=observer_focal_length_mm,
            horizontal_aperture_mm=observer_horizontal_aperture_mm,
            image_size=source_size,
        )
        return map_center_crop_points(
            projected,
            source_size=source_size,
            output_size=output_size,
        )

    if detection is not None and object_world_points is not None:
        box = project(object_world_points)
        center = project(np.mean(object_world_points, axis=0, keepdims=True))
        rectangle = axis_aligned_projected_box(
            box,
            center,
            image_size=output_size,
        )
        if rectangle is not None:
            x0, y0, x1, y1 = rectangle
            cv2.rectangle(
                rendered,
                (x0, y0),
                (x1, y1),
                (40, 215, 255),
                4,
                cv2.LINE_AA,
            )
            score = detection.get("score")
            label = f"DINO  {detection.get('class_id', 'object')}"
            if score is not None:
                label += f"  {score:.2f}"
            label_x = int(np.clip(x0, 4, output_size[0] - 8))
            label_y = int(np.clip(y0 - 9, 25, output_size[1] - 8))
            (label_width, label_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                2,
            )
            cv2.rectangle(
                rendered,
                (label_x, label_y - label_height - baseline - 5),
                (
                    min(output_size[0] - 1, label_x + label_width + 10),
                    label_y + 3,
                ),
                (40, 215, 255),
                -1,
            )
            cv2.putText(
                rendered,
                label,
                (label_x + 5, label_y - baseline),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (12, 12, 12),
                2,
                cv2.LINE_AA,
            )
            stats["detection_box_drawn"] = True

    if pose is not None:
        world_axes = pose_axes_world_points(
            pose,
            eye=perception_eye,
            target=perception_target,
        )
        if world_axes is not None:
            axes = project(world_axes)
            if np.all(np.isfinite(axes)):
                axes = np.round(axes).astype(int)
                origin = tuple(axes[0])
                colors = {
                    "X": (40, 40, 245),
                    "Y": (40, 220, 40),
                    "Z": (245, 90, 40),
                }
                for index, label in enumerate(("X", "Y", "Z"), start=1):
                    endpoint = tuple(axes[index])
                    cv2.arrowedLine(
                        rendered,
                        origin,
                        endpoint,
                        colors[label],
                        4,
                        cv2.LINE_AA,
                        tipLength=0.18,
                    )
                    cv2.putText(
                        rendered,
                        label,
                        endpoint,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        colors[label],
                        2,
                        cv2.LINE_AA,
                    )
                cv2.circle(
                    rendered,
                    origin,
                    5,
                    (245, 245, 245),
                    -1,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    rendered,
                    "FoundationPose  6-DoF",
                    (origin[0] + 12, origin[1] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (235, 235, 235),
                    2,
                    cv2.LINE_AA,
                )
                stats["pose_axes_drawn"] = 1
    return rendered, stats


def project_pose_axes(
    pose: dict,
    camera_info: Any,
    *,
    axis_length_m: float = 0.08,
) -> dict[str, tuple[int, int]] | None:
    """Project an object-frame XYZ triad into its source camera image."""
    if camera_info is None or axis_length_m <= 0.0:
        return None
    intrinsics = np.asarray(camera_info.k, dtype=float)
    if intrinsics.size != 9:
        return None
    fx, fy = intrinsics[0], intrinsics[4]
    cx, cy = intrinsics[2], intrinsics[5]
    if not np.all(np.isfinite([fx, fy, cx, cy])) or fx <= 0.0 or fy <= 0.0:
        return None

    origin = np.asarray(pose["position_m"], dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)) or origin[2] <= 1.0e-6:
        return None
    try:
        rotation = quaternion_rotation_matrix(pose["orientation_xyzw"])
    except ValueError:
        return None

    points = {
        "origin": origin,
        "x": origin + rotation[:, 0] * axis_length_m,
        "y": origin + rotation[:, 1] * axis_length_m,
        "z": origin + rotation[:, 2] * axis_length_m,
    }
    projected = {}
    for name, point in points.items():
        if not np.all(np.isfinite(point)) or point[2] <= 1.0e-6:
            continue
        projected[name] = (
            int(round(fx * point[0] / point[2] + cx)),
            int(round(fy * point[1] / point[2] + cy)),
        )
    return projected if "origin" in projected else None


def draw_pose_overlay(
    image: np.ndarray | None,
    poses: list[dict],
    camera_info: Any,
    mask: np.ndarray | None,
) -> tuple[np.ndarray | None, int]:
    """Overlay FoundationPose segmentation and projected object axes."""
    if image is None:
        return None, 0
    rendered = image.copy()
    if mask is not None:
        resized_mask = cv2.resize(
            mask,
            (rendered.shape[1], rendered.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        )
        selected = resized_mask > 0
        tint = np.full(rendered.shape, (210, 70, 190), dtype=np.uint8)
        blended = cv2.addWeighted(rendered, 0.62, tint, 0.38, 0.0)
        rendered[selected] = blended[selected]

    colors = {
        "x": (40, 40, 245),
        "y": (40, 220, 40),
        "z": (245, 90, 40),
    }
    axes_drawn = 0
    for pose in poses:
        projected = project_pose_axes(pose, camera_info)
        if projected is None:
            continue
        origin = projected["origin"]
        for axis in ("x", "y", "z"):
            endpoint = projected.get(axis)
            if endpoint is None:
                continue
            cv2.arrowedLine(
                rendered,
                origin,
                endpoint,
                colors[axis],
                4,
                cv2.LINE_AA,
                tipLength=0.18,
            )
            cv2.putText(
                rendered,
                axis.upper(),
                endpoint,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                colors[axis],
                2,
                cv2.LINE_AA,
            )
        cv2.circle(rendered, origin, 5, (245, 245, 245), -1, cv2.LINE_AA)
        axes_drawn += 1
    return rendered, axes_drawn
