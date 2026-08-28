"""Contracts and rendering tests for the synchronized pipeline capture."""

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest


WS_DIR = Path(__file__).resolve().parents[1]
SIM_DIR = WS_DIR / "sim"
sys.path.insert(0, str(SIM_DIR))

from pipeline_diagnostics_render import (
    axis_aligned_projected_box,
    crop_to_fill,
    decode_color_image,
    decode_mask_image,
    draw_detections,
    draw_observer_perception_overlay,
    draw_pose_overlay,
    extract_detections,
    extract_poses,
    map_center_crop_points,
    optical_points_to_world,
    pose_axes_world_points,
    pose_object_world_cylinder_points,
    pose_object_world_corners,
    project_world_points,
    project_pose_axes,
)
from presentation_views import (
    CAMERA_HORIZONTAL_APERTURE_MM,
    MAIN_CAMERA_EYE,
    MAIN_CAMERA_FOCAL_LENGTH_MM,
    MAIN_CAMERA_TARGET,
    PERCEPTION_CAMERA_EYE,
    PERCEPTION_CAMERA_TARGET,
)


CAPTURE_SCRIPT = SIM_DIR / "capture_pipeline_diagnostics.py"
PICK_SCENE = SIM_DIR / "pick_scene.py"


def _image(encoding, width, height, step, data, *, bigendian=False):
    return SimpleNamespace(
        encoding=encoding,
        width=width,
        height=height,
        step=step,
        data=data,
        is_bigendian=bigendian,
    )


def _hypothesis(class_id="soup_can", score=0.91):
    return SimpleNamespace(
        hypothesis=SimpleNamespace(class_id=class_id, score=score)
    )


def _pose_record(z=1.0):
    return {
        "id": "can",
        "class_id": "soup_can",
        "score": 0.97,
        "position_m": [0.0, 0.0, z],
        "orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
    }


def test_color_decoder_respects_ros_row_padding_and_rgb_order():
    message = _image(
        "rgb8",
        width=2,
        height=2,
        step=8,
        data=bytes([
            255, 0, 0, 0, 255, 0, 99, 99,
            0, 0, 255, 255, 255, 255, 88, 88,
        ]),
    )

    decoded = decode_color_image(message)

    assert decoded.tolist() == [
        [[0, 0, 255], [0, 255, 0]],
        [[255, 0, 0], [255, 255, 255]],
    ]


def test_crop_to_fill_uses_the_center_without_letterboxing():
    source = np.zeros((4, 8, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(8, dtype=np.uint8)

    rendered = crop_to_fill(source, 4, 4)

    assert rendered.shape == (4, 4, 3)
    assert rendered[0, :, 0].tolist() == [2, 3, 4, 5]


def test_crop_to_fill_rejects_invalid_output_dimensions():
    with pytest.raises(ValueError, match="dimensions must be positive"):
        crop_to_fill(np.zeros((2, 2, 3), dtype=np.uint8), 0, 4)


def test_mask_decoder_handles_padded_16bit_images():
    values = np.asarray([[0, 2, 99], [4, 0, 88]], dtype="<u2")
    message = _image(
        "16UC1",
        width=2,
        height=2,
        step=6,
        data=values.tobytes(),
    )

    decoded = decode_mask_image(message)

    assert decoded.tolist() == [[0, 255], [255, 0]]


def test_color_decoder_rejects_unsupported_encoding():
    message = _image("yuv422", 1, 1, 2, b"\x00\x00")

    with pytest.raises(ValueError, match="unsupported color encoding"):
        decode_color_image(message)


def test_detection_records_and_overlay_use_real_bbox_and_confidence():
    detection = SimpleNamespace(
        id="42",
        bbox=SimpleNamespace(
            center=SimpleNamespace(position=SimpleNamespace(x=50.0, y=40.0)),
            size_x=20.0,
            size_y=10.0,
        ),
        results=[_hypothesis()],
    )
    records = extract_detections(
        SimpleNamespace(detections=[detection])
    )
    source = np.zeros((100, 120, 3), dtype=np.uint8)

    rendered = draw_detections(source, records)

    assert records == [{
        "id": "42",
        "class_id": "soup_can",
        "score": pytest.approx(0.91),
        "center_x": 50.0,
        "center_y": 40.0,
        "width": 20.0,
        "height": 10.0,
        "x_min": 40.0,
        "y_min": 35.0,
        "x_max": 60.0,
        "y_max": 45.0,
    }]
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(source) == 0


def test_foundation_pose_record_and_axes_projection():
    pose = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0, z=1.0),
        orientation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
    )
    result = _hypothesis(score=0.97)
    result.pose = SimpleNamespace(pose=pose)
    message = SimpleNamespace(
        detections=[SimpleNamespace(id="can", results=[result])]
    )
    records = extract_poses(message)
    camera_info = SimpleNamespace(
        k=[100.0, 0.0, 320.0, 0.0, 100.0, 240.0, 0.0, 0.0, 1.0]
    )

    projected = project_pose_axes(
        records[0], camera_info, axis_length_m=0.1)

    assert records == [_pose_record()]
    assert projected == {
        "origin": (320, 240),
        "x": (330, 240),
        "y": (320, 250),
        "z": (320, 240),
    }


def test_pose_overlay_requires_pose_in_front_of_camera():
    camera_info = SimpleNamespace(
        k=[100.0, 0.0, 50.0, 0.0, 100.0, 50.0, 0.0, 0.0, 1.0]
    )
    source = np.zeros((100, 100, 3), dtype=np.uint8)
    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[20:30, 20:30] = 255

    rendered, axes_drawn = draw_pose_overlay(
        source, [_pose_record()], camera_info, mask)
    behind = project_pose_axes(_pose_record(z=-1.0), camera_info)

    assert axes_drawn == 1
    assert np.count_nonzero(rendered) > 0
    assert behind is None


def test_cross_camera_pose_reprojection_targets_the_observed_can():
    pose = {
        "position_m": [-0.035245, -0.006822, 0.769301],
        "orientation_xyzw": [
            0.092414,
            0.892424,
            0.335625,
            0.287046,
        ],
    }

    world_axes = pose_axes_world_points(
        pose,
        eye=PERCEPTION_CAMERA_EYE,
        target=PERCEPTION_CAMERA_TARGET,
    )
    projected = project_world_points(
        world_axes,
        eye=MAIN_CAMERA_EYE,
        target=MAIN_CAMERA_TARGET,
        focal_length_mm=MAIN_CAMERA_FOCAL_LENGTH_MM,
        horizontal_aperture_mm=CAMERA_HORIZONTAL_APERTURE_MM,
        image_size=(1920, 1080),
    )
    cropped = map_center_crop_points(
        projected,
        source_size=(1920, 1080),
        output_size=(1440, 1080),
    )

    assert world_axes[0] == pytest.approx(
        [0.371399, -0.000867, 0.201950],
        abs=1.0e-5,
    )
    assert cropped[0] == pytest.approx([732.10, 693.38], abs=0.1)


def test_pose_cylinder_outline_draws_tight_dino_box_on_observer():
    detection = {
        "class_id": "soup can",
        "score": 0.82,
    }
    pose = _pose_record()
    points = pose_object_world_cylinder_points(
        pose,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 1.0, 0.0),
    )
    image = np.zeros((100, 100, 3), dtype=np.uint8)

    rendered, stats = draw_observer_perception_overlay(
        image,
        source_size=(100, 100),
        detection=detection,
        object_world_points=points,
        pose=pose,
        perception_eye=(0.0, 0.0, 0.0),
        perception_target=(0.0, 1.0, 0.0),
        observer_eye=(0.0, 0.0, 0.0),
        observer_target=(0.0, 1.0, 0.0),
        observer_focal_length_mm=24.0,
        observer_horizontal_aperture_mm=24.0,
    )

    assert points.shape == (144, 3)
    assert stats == {
        "detection_box_drawn": True,
        "detection_box_source": "foundationpose_cylinder_outline",
        "pose_axes_drawn": 1,
    }
    assert np.count_nonzero(rendered) > 0


def test_recorded_pose_cylinder_outline_is_tighter_than_mesh_box():
    pose = {
        "position_m": [-0.035458289, -0.007049954, 0.769755006],
        "orientation_xyzw": [
            0.090937442,
            0.893769836,
            0.329117323,
            0.290839478,
        ],
    }
    world_points = pose_object_world_cylinder_points(
        pose,
        eye=PERCEPTION_CAMERA_EYE,
        target=PERCEPTION_CAMERA_TARGET,
    )
    world_corners = pose_object_world_corners(
        pose,
        eye=PERCEPTION_CAMERA_EYE,
        target=PERCEPTION_CAMERA_TARGET,
    )
    world_center = np.mean(world_points, axis=0, keepdims=True)
    projected_points = map_center_crop_points(
        project_world_points(
            world_points,
            eye=MAIN_CAMERA_EYE,
            target=MAIN_CAMERA_TARGET,
            focal_length_mm=MAIN_CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=CAMERA_HORIZONTAL_APERTURE_MM,
            image_size=(1920, 1080),
        ),
        source_size=(1920, 1080),
        output_size=(1440, 1080),
    )
    projected_corners = map_center_crop_points(
        project_world_points(
            world_corners,
            eye=MAIN_CAMERA_EYE,
            target=MAIN_CAMERA_TARGET,
            focal_length_mm=MAIN_CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=CAMERA_HORIZONTAL_APERTURE_MM,
            image_size=(1920, 1080),
        ),
        source_size=(1920, 1080),
        output_size=(1440, 1080),
    )
    projected_center = map_center_crop_points(
        project_world_points(
            world_center,
            eye=MAIN_CAMERA_EYE,
            target=MAIN_CAMERA_TARGET,
            focal_length_mm=MAIN_CAMERA_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=CAMERA_HORIZONTAL_APERTURE_MM,
            image_size=(1920, 1080),
        ),
        source_size=(1920, 1080),
        output_size=(1440, 1080),
    )

    rectangle = axis_aligned_projected_box(
        projected_points,
        projected_center,
        image_size=(1440, 1080),
    )
    cuboid_rectangle = axis_aligned_projected_box(
        projected_corners,
        projected_center,
        image_size=(1440, 1080),
    )

    assert rectangle is not None
    x0, y0, x1, y1 = rectangle
    assert x0 < projected_center[0, 0] < x1
    assert y0 < projected_center[0, 1] < y1
    assert (x0, y0, x1, y1) == pytest.approx(
        (663, 572, 805, 816),
        abs=2,
    )
    assert x1 - x0 < cuboid_rectangle[2] - cuboid_rectangle[0]
    assert y1 - y0 < cuboid_rectangle[3] - cuboid_rectangle[1]


@pytest.mark.parametrize(
    "pose",
    [
        {"position_m": [0.0, 0.0], "orientation_xyzw": [0, 0, 0, 1]},
        {"position_m": [0.0, 0.0, 1.0], "orientation_xyzw": [0, 0, 0, 0]},
        {
            "position_m": [0.0, float("nan"), 1.0],
            "orientation_xyzw": [0, 0, 0, 1],
        },
    ],
)
def test_pose_cylinder_outline_rejects_malformed_pose(pose):
    assert pose_object_world_cylinder_points(
        pose,
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 1.0, 0.0),
    ) is None


def test_pose_cylinder_outline_rejects_undersampled_circle():
    assert pose_object_world_cylinder_points(
        _pose_record(),
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 1.0, 0.0),
        circumference_samples=7,
    ) is None


def test_axis_aligned_box_rejects_invalid_center_or_extent():
    corners = np.asarray([[10.0, 10.0], [20.0, 20.0]])

    assert axis_aligned_projected_box(
        corners,
        np.asarray([float("nan"), 15.0]),
        image_size=(100, 100),
    ) is None
    assert axis_aligned_projected_box(
        corners,
        np.asarray([15.0, 15.0]),
        image_size=(100, 100),
        minimum_extent_px=20,
    ) is None


def test_optical_to_world_uses_ros_down_axis():
    transformed = optical_points_to_world(
        np.asarray([[1.0, 2.0, 3.0]]),
        eye=(0.0, 0.0, 0.0),
        target=(0.0, 1.0, 0.0),
    )

    assert transformed[0] == pytest.approx([1.0, 3.0, -2.0])


def test_capture_uses_main_overlay_high_wide_view_and_masked_depth():
    source = CAPTURE_SCRIPT.read_text()
    compose = source[source.index("    def compose("):source.index("\ndef main()")]

    assert '"/object_detection_server/image_rect"' in source
    assert 'GROUNDING_DINO_DETECTIONS_TOPIC = "/detections"' in source
    assert '"/foundation_pose_server/image"' in source
    assert '"/foundation_pose_server/camera_info"' in source
    assert 'FOUNDATIONPOSE_POSE_TOPIC = "/pose_estimation/output"' in source
    assert '"/foundation_pose_server/segmented_mask"' in source
    assert '"/segmentation"' in source
    assert "PANEL_WIDTH = 480" in source
    assert "SIM_PANEL_WIDTH = 1440" in source
    assert (
        "MOSAIC_SIZE = (SIM_PANEL_WIDTH + PANEL_WIDTH, PANEL_HEIGHT * 2)"
        in source
    )
    assert "make_sim_panel(" in compose
    assert "grounding_dino_panel" in compose
    assert "foundation_pose_panel" in compose
    assert "foundation_pose_ready" in compose
    assert "draw_observer_perception_overlay(" in compose
    assert "pose_object_world_cylinder_points(" in source
    assert '"detection_box_source": "foundationpose_cylinder_outline"' in compose
    assert "detection_box_world_corners" not in source
    assert "PERCEPTION_OVERLAY_HOLD_S" in compose
    assert "wide_sim_panel" in compose
    assert "make_wide_panel(" in compose
    assert '"--wide-sim-frame-dir"' in source
    assert 'SCENE_CAMERA_INFO_TOPIC = "/scene_cam_0/camera_info"' in source
    assert '"display": "main_observer_overlay"' in compose
    assert "masked_panel" in compose
    assert "np.vstack([" in compose
    assert "wide_sim_panel," in compose
    assert "masked_panel," in compose
    assert "perception_panel" not in compose
    assert "esdf_panel" not in compose
    assert "nvblox 3-D ESDF - top-down projection" not in compose
    assert '"Scene camera 0 - raw depth"' not in compose
    assert '"Scene camera 1 - raw depth"' not in compose
    assert '"Wrist depth - perception input"' not in compose
    assert '"Robot-masked depth to nvblox"' in source
    assert '"scene camera 0"' in source
    assert '"scene camera 1"' in source
    assert '"--stop-file"' in source
    assert '"--sim-state-file"' in source
    assert "WALL CLEARANCE" in source
    assert "REQ {required_mm:.0f} mm" in source
    assert "minimum_wall_clearance_m=self.args.minimum_wall_clearance_m" in compose
    assert "wall_safety_failure_reason(" in source
    assert "WALL_MIN = TRANSFER_WALL.min_xyz_m" in source
    assert "WALL_MAX = TRANSFER_WALL.max_xyz_m" in source
    assert "args.stop_file.exists()" in source
    assert '"perception": {' in source
    assert '"grounding_dino": grounding_dino_stats' in source
    assert '"foundation_pose": foundation_pose_stats' in source


def test_observer_capture_defaults_to_full_hd_high_quality_frames():
    source = PICK_SCENE.read_text()

    assert '"--record-width", type=int, default=MAIN_RECORD_SIZE[0]' in source
    assert '"--record-height", type=int, default=MAIN_RECORD_SIZE[1]' in source
    assert '"--record-wide-dir"' in source
    assert '"--record-wide-width", type=int, default=WIDE_RECORD_SIZE[0]' in source
    assert '"--record-wide-height", type=int, default=WIDE_RECORD_SIZE[1]' in source
    assert '"--record-jpeg-quality", type=int, default=92' in source
    assert 'camera_path="/World/record_camera_high_wide"' in source
    assert "WIDE_CAMERA_EYE" in source
    assert "quality=_args.record_jpeg_quality" in source


def test_main_observer_sits_below_and_inside_perception_camera_fixture():
    observer_eye = np.asarray(MAIN_CAMERA_EYE)
    perception_eye = np.asarray(PERCEPTION_CAMERA_EYE)

    assert observer_eye[2] < perception_eye[2]
    assert observer_eye[1] > perception_eye[1]
    assert np.linalg.norm(observer_eye[:2] - perception_eye[:2]) < 0.12
    assert MAIN_CAMERA_FOCAL_LENGTH_MM == 18.0
