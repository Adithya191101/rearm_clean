"""Contracts for the camera-mapped transfer wall and cuMotion ESDF path."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

WS_DIR = Path(__file__).resolve().parents[1]
SIM_DIR = WS_DIR / "sim"
sys.path.insert(0, str(SIM_DIR))

from physical_grasp import CAN_COLLIDER
from pick_area import (
    PICK_AREA,
    PLACE_AREA,
    TABLE_Z,
    pick_centre,
    place_centre,
)
from transfer_obstacle import (
    MINIMUM_WALL_CLEARANCE_M,
    PLANNING_WALL,
    TRANSFER_WALL,
    CollisionSphere,
    WallSpec,
    atomic_write_json,
    cylinder_cover_spheres,
    finite_cylinder_aabb_clearance,
    finite_cylinder_aabb_clearance_bounds,
    load_planner_collision_spheres,
    minimum_sphere_aabb_clearance,
    moveit_scene_text,
    project_point_to_finite_cylinder,
    sphere_aabb_clearances,
    transform_collision_spheres,
    validate_wall_safety_state,
    wall_safety_failure_reason,
)

WORKFLOW_CONFIG = (
    WS_DIR / "config" / "workflows" / "rebot_sim_launch_params.yaml"
)
CUMOTION_LAUNCH = (
    WS_DIR / "src" / "isaac_ros_manipulation"
    / "isaac_ros_manipulation_bringup" / "launch" / "include"
    / "cumotion.launch.py"
)
NVBLOX_LAUNCH = (
    WS_DIR / "src" / "isaac_ros_manipulation"
    / "isaac_ros_manipulation_bringup" / "launch" / "include"
    / "nvblox.launch.py"
)
MANIPULATION_CONFIG = (
    WS_DIR / "src" / "isaac_ros_manipulation"
    / "isaac_ros_manipulation_ros_python_utils"
    / "isaac_ros_manipulation_ros_python_utils" / "config.py"
)
WORKFLOW_RUNNER = WS_DIR / "sim" / "run" / "launch_workflow_live.sh"
PICK_SCENE = WS_DIR / "sim" / "pick_scene.py"
DRIVER_UTILS = (
    WS_DIR / "src" / "rebot_b601dm_isaac"
    / "isaac_ros_manipulation_rebot_driver_utils"
)
CAMERA_INFO_RELAY = (
    DRIVER_UTILS / "isaac_ros_manipulation_rebot_driver_utils"
    / "nvblox_camera_info_relay.py"
)
DRIVER_LAUNCH = DRIVER_UTILS / "launch" / "rebot_driver.launch.py"
REBOT_URDF = (
    WS_DIR / "src" / "rebot_b601dm_isaac"
    / "isaac_ros_manipulation_rebot_robot_description"
    / "urdf" / "rebot_sim.urdf.xacro"
)
REBOT_SRDF = (
    WS_DIR / "src" / "rebot_b601dm_isaac"
    / "isaac_ros_manipulation_rebot_robot_description"
    / "srdf" / "rebot.srdf.xacro"
)
PICK_GOAL = WS_DIR / "sim" / "send_pick_goal.py"
ATTACH_OBJECT = (
    WS_DIR / "src" / "isaac_ros_manipulation"
    / "isaac_ros_manipulation_orchestration"
    / "isaac_ros_manipulation_orchestration" / "behaviors"
    / "motion_behaviors" / "attach_object.py"
)
GRIPPER_XRDF = WS_DIR / "config" / "xrdf" / "rebot_b601dm_gripper.xrdf"
STATIC_WALL_SCENE = (
    WS_DIR / "config" / "scene_objects" / "rebot_transfer_wall.scene"
)
BT_CONFIG = (
    WS_DIR / "config" / "pick_and_place"
    / "multi_object_pick_and_place_behavior_tree_params.yaml"
)

# Fixed wall from the numerical cases that originally exposed conservative
# cylinder-clearance bugs. Keep these solver fixtures independent of scene
# layout changes.
CYLINDER_REGRESSION_WALL_MIN = (0.38, 0.0375, 0.15)
CYLINDER_REGRESSION_WALL_MAX = (0.42, 0.0625, 0.31)


def test_wall_separates_pick_from_place_and_leaves_a_place_corridor():
    min_x, min_y, min_z = TRANSFER_WALL.min_xyz_m
    max_x, max_y, _ = TRANSFER_WALL.max_xyz_m
    _, place_y, _ = place_centre()

    assert min_y > PICK_AREA["y"][1]
    assert max_y < PLACE_AREA["y"][0]
    assert min_z == pytest.approx(TABLE_Z)
    assert pick_centre() == pytest.approx((0.37, -0.06, TABLE_Z))
    assert TRANSFER_WALL.center_xyz_m == pytest.approx((0.37, 0.16, 0.23))
    assert place_centre() == pytest.approx((0.37, 0.35, TABLE_Z))
    assert (
        pick_centre()[0]
        == TRANSFER_WALL.center_xyz_m[0]
        == place_centre()[0]
        == pytest.approx(0.37)
    )
    assert min_x < pick_centre()[0] < max_x
    assert (
        place_y - CAN_COLLIDER.radius_m
        > PLANNING_WALL.max_xyz_m[1]
    )
    assert TRANSFER_WALL.size_xyz_m[0] == pytest.approx(0.04)


def test_static_planning_wall_is_a_hard_two_inch_envelope_of_physical_wall():
    physical_min = np.asarray(TRANSFER_WALL.min_xyz_m)
    physical_max = np.asarray(TRANSFER_WALL.max_xyz_m)
    planning_min = np.asarray(PLANNING_WALL.min_xyz_m)
    planning_max = np.asarray(PLANNING_WALL.max_xyz_m)

    assert MINIMUM_WALL_CLEARANCE_M == pytest.approx(2.0 * 0.0254)
    assert physical_min - planning_min == pytest.approx(
        [MINIMUM_WALL_CLEARANCE_M] * 3
    )
    assert planning_max - physical_max == pytest.approx(
        [MINIMUM_WALL_CLEARANCE_M] * 3
    )
    assert STATIC_WALL_SCENE.read_text() == moveit_scene_text()

    workflow = yaml.safe_load(WORKFLOW_CONFIG.read_text())
    assert workflow["moveit_collision_objects_scene_file"] == (
        "$ISAAC_ROS_WS/config/scene_objects/rebot_transfer_wall.scene"
    )


def test_moveit_model_owns_world_frame_before_static_collision_objects_arrive():
    driver_source = DRIVER_LAUNCH.read_text()
    urdf_source = REBOT_URDF.read_text()
    srdf_source = REBOT_SRDF.read_text()

    assert '<link name="world" />' in urdf_source
    assert 'name="$(arg prefix)world_to_base_link" type="fixed"' in urdf_source
    assert '<parent link="world" />' in urdf_source
    assert '<child link="$(arg prefix)base_link" />' in urdf_source
    assert "virtual_joint" not in srdf_source
    assert "world_to_base_link_publisher" not in driver_source


def test_direct_route_to_place_intersects_hard_wall_envelope():
    previous_carry_base_z = 0.289
    start = (*pick_centre()[:2], previous_carry_base_z)
    end = (*place_centre()[:2], previous_carry_base_z)

    assert PLANNING_WALL.blocks_transfer(
        start,
        end,
        object_radius_m=CAN_COLLIDER.radius_m,
        object_height_m=CAN_COLLIDER.height_m,
    )


def test_over_wall_detour_has_clearance():
    clear_base_z = TRANSFER_WALL.top_z_m + 0.02
    start = (*pick_centre()[:2], clear_base_z)
    end = (*place_centre()[:2], clear_base_z)

    assert not TRANSFER_WALL.blocks_transfer(
        start,
        end,
        object_radius_m=CAN_COLLIDER.radius_m,
        object_height_m=CAN_COLLIDER.height_m,
    )


def test_upright_place_target_stays_in_area_and_clears_the_wall():
    x, y, z = place_centre()

    assert PLACE_AREA["x"][0] < x < PLACE_AREA["x"][1]
    assert PLACE_AREA["y"][0] < y < PLACE_AREA["y"][1]
    assert (x, y) == pytest.approx((0.37, 0.35))
    assert z == pytest.approx(TABLE_Z)
    clearance, _, _ = finite_cylinder_aabb_clearance(
        (x, y, z + 0.5 * CAN_COLLIDER.height_m),
        (0.0, 0.0, 1.0),
        radius_m=CAN_COLLIDER.radius_m,
        height_m=CAN_COLLIDER.height_m,
        minimum_xyz_m=TRANSFER_WALL.min_xyz_m,
        maximum_xyz_m=TRANSFER_WALL.max_xyz_m,
    )
    assert clearance > MINIMUM_WALL_CLEARANCE_M


def test_trajectory_contrast_detects_mapped_detour():
    from transfer_obstacle import trajectory_contrast_metrics

    cleared = [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]
    mapped = [[0.0, 0.0], [0.5, 1.0], [1.0, 1.0]]

    metrics = trajectory_contrast_metrics(mapped, cleared)

    assert metrics["mapped_chord_deviation_rad"] > 0.3
    assert metrics["cleared_chord_deviation_rad"] == pytest.approx(0.0)
    assert metrics["mapped_to_cleared_ratio"] == float("inf")
    assert metrics["max_path_separation_rad"] > 0.4


def test_trajectory_contrast_rejects_invalid_paths():
    from transfer_obstacle import trajectory_contrast_metrics

    with pytest.raises(ValueError, match="finite 2D paths"):
        trajectory_contrast_metrics([[0.0]], [[0.0]])


def test_wall_spec_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="positive"):
        WallSpec("/World/bad", (0.0, 0.0, 0.0), (1.0, 0.0, 1.0), (1, 0, 0))


def test_signed_sphere_aabb_clearance_distinguishes_clear_touch_and_penetration():
    spheres = np.asarray([
        [2.0, 0.5, 0.5, 0.5],
        [2.0, 0.5, 0.5, 1.0],
        [2.0, 0.5, 0.5, 1.1],
    ])

    clearances = sphere_aabb_clearances(
        spheres,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )

    assert clearances == pytest.approx([0.5, 0.0, -0.1])
    minimum, index = minimum_sphere_aabb_clearance(
        spheres,
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 1.0),
    )
    assert minimum == pytest.approx(-0.1)
    assert index == 2


def test_planner_spheres_transform_from_live_physx_frames_with_buffers():
    loaded = load_planner_collision_spheres(GRIPPER_XRDF)
    sphere = CollisionSphere("link", (1.0, 0.0, 0.0), 0.2)
    half_turn_about_z_xyzw = (
        0.0,
        0.0,
        2 ** -0.5,
        2 ** -0.5,
    )

    world = transform_collision_spheres(
        [sphere],
        {"link": [1.0, 2.0, 3.0, *half_turn_about_z_xyzw]},
    )

    assert len(loaded) == 677
    assert loaded[0].radius_m == pytest.approx(0.0425)
    assert world[0] == pytest.approx([1.0, 3.0, 3.0, 0.2])


def test_can_sphere_chain_conservatively_covers_cylinder_end_corner():
    spheres = cylinder_cover_spheres(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        radius_m=0.033,
        height_m=0.101,
        count=7,
    )
    end_corner = np.asarray([0.033, 0.0, 0.0505])
    distances = np.linalg.norm(spheres[:, :3] - end_corner, axis=1)

    assert spheres.shape == (7, 4)
    assert np.min(distances - spheres[:, 3]) <= 1.0e-12


def test_finite_cylinder_projection_clamps_radius_and_both_end_caps():
    projected = project_point_to_finite_cylinder(
        (2.0, 0.0, 3.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        radius_m=1.0,
        height_m=2.0,
    )

    assert projected == pytest.approx([1.0, 0.0, 1.0])


@pytest.mark.parametrize(
    ("minimum", "maximum", "expected_clearance"),
    (
        ((2.0, -0.5, -0.5), (3.0, 0.5, 0.5), 1.0),
        ((-0.5, -0.5, 2.0), (0.5, 0.5, 3.0), 1.0),
        ((-0.5, -0.5, 0.5), (0.5, 0.5, 1.5), 0.0),
    ),
)
def test_finite_cylinder_clearance_handles_side_end_cap_and_intersection(
    minimum,
    maximum,
    expected_clearance,
):
    clearance, cylinder_witness, wall_witness = finite_cylinder_aabb_clearance(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        radius_m=1.0,
        height_m=2.0,
        minimum_xyz_m=minimum,
        maximum_xyz_m=maximum,
    )

    assert clearance == pytest.approx(expected_clearance)
    assert np.linalg.norm(cylinder_witness - wall_witness) == pytest.approx(
        expected_clearance
    )


def test_finite_cylinder_avoids_sphere_cover_false_contact_for_tilted_can():
    center = (0.39543897800522196, 0.04335232468566873, 0.3876307389945705)
    axis = (0.08025913995428624, -0.07215930051898442, -0.9941586924643415)
    radius_m = 0.033
    height_m = 0.101

    result = finite_cylinder_aabb_clearance_bounds(
        center,
        axis,
        radius_m=radius_m,
        height_m=height_m,
        minimum_xyz_m=CYLINDER_REGRESSION_WALL_MIN,
        maximum_xyz_m=CYLINDER_REGRESSION_WALL_MAX,
    )
    clearance = result.lower_bound_m
    can_witness = result.cylinder_witness_m
    wall_witness = result.box_witness_m
    old_spheres = cylinder_cover_spheres(
        center,
        axis,
        radius_m=radius_m,
        height_m=height_m,
    )
    old_clearance, _ = minimum_sphere_aabb_clearance(
        old_spheres,
        CYLINDER_REGRESSION_WALL_MIN,
        CYLINDER_REGRESSION_WALL_MAX,
    )

    assert clearance == pytest.approx(0.0240072463)
    assert not result.converged
    assert result.upper_bound_m == pytest.approx(0.0240564720)
    assert result.uncertainty_m < 0.00005
    assert old_clearance < 0.002
    assert np.linalg.norm(can_witness - wall_witness) == pytest.approx(
        result.upper_bound_m
    )
    assert wall_witness == pytest.approx([0.38, 0.0625, 0.31])
    assert project_point_to_finite_cylinder(
        wall_witness,
        center,
        axis,
        radius_m=radius_m,
        height_m=height_m,
    ) == pytest.approx(can_witness)


def test_finite_cylinder_clearance_converges_for_nearly_vertical_startup_pose():
    clearance, _, _ = finite_cylinder_aabb_clearance(
        (0.4100000500744335, -0.11000000709452035, 0.20050001692772118),
        (
            -2.831402547663013e-07,
            4.897081994244066e-08,
            -0.9999999999999587,
        ),
        radius_m=0.033,
        height_m=0.101,
        minimum_xyz_m=CYLINDER_REGRESSION_WALL_MIN,
        maximum_xyz_m=CYLINDER_REGRESSION_WALL_MAX,
    )

    assert clearance == pytest.approx(0.1145000071)


def test_finite_cylinder_clearance_fails_closed_for_slow_parallel_projection():
    result = finite_cylinder_aabb_clearance_bounds(
        (0.431, -0.106, 0.201),
        (-0.025, 0.0, -(1.0 - 0.025 ** 2) ** 0.5),
        radius_m=0.033,
        height_m=0.101,
        minimum_xyz_m=CYLINDER_REGRESSION_WALL_MIN,
        maximum_xyz_m=CYLINDER_REGRESSION_WALL_MAX,
        tolerance_m=1.0e-12,
        max_iterations=128,
    )

    # The old implementation raised after 128 iterations at this pose. The
    # replacement returns a certified lower bound and exposes the small gap to
    # its feasible upper witness instead of overstating safety.
    assert not result.converged
    assert result.lower_bound_m == pytest.approx(0.1108287951)
    assert result.upper_bound_m == pytest.approx(0.1108721466)
    assert 0.0 < result.uncertainty_m < 0.00005
    assert np.linalg.norm(
        result.cylinder_witness_m - result.box_witness_m
    ) == pytest.approx(result.upper_bound_m)


def test_finite_cylinder_clearance_rejects_degenerate_axis():
    with pytest.raises(ValueError, match="axis"):
        finite_cylinder_aabb_clearance(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            radius_m=1.0,
            height_m=2.0,
            minimum_xyz_m=(2.0, -0.5, -0.5),
            maximum_xyz_m=(3.0, 0.5, 0.5),
        )


def test_wall_safety_state_is_atomic_and_enforces_clearance_and_contact(tmp_path):
    state = {
        "schema_version": 1,
        "sim_time_s": 12.0,
        "sample": {
            "robot_clearance_m": 0.070,
            "can_clearance_m": 0.060,
            "clearance_m": 0.060,
        },
        "minimum_observed": {
            "robot_clearance_m": 0.065,
            "can_clearance_m": 0.055,
            "clearance_m": 0.055,
        },
        "contact": {"current": False, "ever": False, "events": 0},
    }
    path = tmp_path / "wall_state.json"

    atomic_write_json(path, state)
    loaded = validate_wall_safety_state(json.loads(path.read_text()))

    assert loaded == state
    assert wall_safety_failure_reason(loaded) is None
    assert "below" in wall_safety_failure_reason(
        loaded,
        minimum_clearance_m=0.060,
    )
    loaded["contact"]["ever"] = True
    assert "PhysX" in wall_safety_failure_reason(loaded)
    assert not (tmp_path / ".wall_state.json.tmp").exists()


def test_place_marker_is_a_visible_visual_only_drop_sheet():
    source = PICK_SCENE.read_text()
    marker_source = source[
        source.index("def add_area_markers("):
        source.index("\ndef _box(", source.index("def add_area_markers("))
    ]

    assert "place_sheet_size_m = (0.12, 0.12)" in marker_source
    assert '"place_area_marker"' in marker_source
    assert "sx = max(x1 - x0, minimum_size_m[0])" in marker_source
    assert "sy = max(y1 - y0, minimum_size_m[1])" in marker_source
    assert "UsdPhysics.CollisionAPI.Apply" not in marker_source
    assert "UsdPhysics.RigidBodyAPI.Apply" not in marker_source


def test_wall_is_visible_static_collision_geometry():
    obstacle_source = (SIM_DIR / "transfer_obstacle.py").read_text()
    scene_source = PICK_SCENE.read_text()

    assert "UsdPhysics.CollisionAPI.Apply" in obstacle_source
    assert "RigidBodyAPI" not in obstacle_source
    assert "create_transfer_wall(stage)" in scene_source
    assert '"--no-transfer-wall"' not in scene_source
    assert "bind_physics_material(wall, grip_material)" in scene_source
    assert '"--transfer-wall-command-file"' in scene_source
    assert 'command["wall_visible"]' in scene_source
    assert "imageable.MakeInvisible()" in scene_source
    assert "PhysX collider remains enabled" in scene_source
    assert '"--record-state-file"' in scene_source
    assert "load_planner_collision_spheres(" in scene_source
    assert "transform_collision_spheres(" in scene_source
    assert "finite_cylinder_aabb_clearance_bounds(" in scene_source
    assert '"can_witness_m"' in scene_source
    assert '"wall_witness_m"' in scene_source
    assert "PhysxContactReportAPI.Apply" in scene_source
    assert "subscribe_contact_report_events" in scene_source
    assert "contact = contact_data[index]" in scene_source
    assert "contact_data[start:stop]" not in scene_source
    assert "minimum_observed" in scene_source


def test_scene_cameras_are_synchronously_throttled_for_mapping():
    scene_source = PICK_SCENE.read_text()
    constants_source = (
        WS_DIR / "src" / "isaac_ros_manipulation"
        / "isaac_ros_manipulation_ros_python_utils"
        / "isaac_ros_manipulation_ros_python_utils" / "constants.py"
    ).read_text()

    assert "SCENE_CAM_W, SCENE_CAM_H = 640, 480" in scene_source
    assert "SCENE_CAM_FRAME_SKIP = 3" in scene_source
    for helper in ("Rgb", "Info", "Depth"):
        assert (
            f'("{helper}.inputs:frameSkipCount", SCENE_CAM_FRAME_SKIP)'
            in scene_source
        )
    assert "SCENE_CAM_IMAGE_WIDTH = 640" in constants_source
    assert "SCENE_CAM_IMAGE_HEIGHT = 480" in constants_source


def test_active_workflow_enables_two_camera_nvblox():
    config = yaml.safe_load(WORKFLOW_CONFIG.read_text())

    assert config["camera_type"] == "ISAAC_SIM"
    assert config["num_cameras"] == 2
    assert config["enable_nvblox"] == "true"
    assert config["nvblox_global_frame"] == "base_link"
    assert config["moveit_collision_objects_scene_file"] == (
        "$ISAAC_ROS_WS/config/scene_objects/rebot_transfer_wall.scene"
    )


def test_cumotion_segments_both_static_depth_streams():
    source = CUMOTION_LAUNCH.read_text()

    for camera in ("scene_cam_0", "scene_cam_1"):
        assert f"'/{camera}/depth/image_raw'" in source
        assert f"'/{camera}/camera_info'" in source
    assert "get_isaac_sim_depth_topics(num_cameras: int)" in source
    assert "json.dumps([depth for depth, _ in cameras])" in source


def test_nvblox_consumes_both_robot_masked_depth_streams():
    source = NVBLOX_LAUNCH.read_text()

    for camera in ("scene_cam_0", "scene_cam_1"):
        assert f"'/{camera}/rgb/image_raw'" in source
        assert f"'/{camera}/camera_info'" in source
    assert "f'/cumotion/camera_{index + 1}/world_depth'" in source
    assert "f'/nvblox/scene_cam_{index}/depth/camera_info'" in source
    assert "get_sim_remappings(num_cameras)" in source
    assert "1 <= num_cameras <= 2" in source
    assert "'use_color': False" in source
    assert "'use_depth': True" in source
    assert "'integrate_depth_rate_hz': 20.0" in source
    assert "'static_mapper.workspace_bounds_type': 'bounding_box'" in source
    assert "'static_mapper.workspace_bounds_min_corner_x_m': -0.10" in source
    assert "'static_mapper.workspace_bounds_max_corner_y_m': 0.60" in source


def test_nvblox_camera_info_relay_uses_masked_depth_timestamps():
    source = CAMERA_INFO_RELAY.read_text()

    for camera in ("scene_cam_0", "scene_cam_1"):
        assert f'"/{camera}/camera_info"' in source
        assert f'"/nvblox/{camera}/depth/camera_info"' in source
    for camera in (1, 2):
        assert f'"/cumotion/camera_{camera}/world_depth"' in source
    assert "aligned.header = copy.deepcopy(depth.header)" in source


def test_core_config_preserves_two_camera_isaac_sim_mapping():
    source = MANIPULATION_CONFIG.read_text()

    assert "1 <= self.num_cameras <= 2" in source
    assert "self.num_cameras = '1'" not in source


def test_docker_runtime_uses_complete_overlay_for_mapping_files():
    source = WORKFLOW_RUNNER.read_text()
    driver_source = DRIVER_LAUNCH.read_text()
    packages = (WS_DIR / "docker" / "overlay-packages.txt").read_text()

    assert ":/opt/ros/" not in source
    assert "docker/verify_overlay.sh" in source
    assert "isaac_ros_manipulation_bringup" in packages
    assert "isaac_ros_manipulation_ros_python_utils" in packages
    assert "sim/nvblox_camera_info_relay.py" not in source
    assert "executable='nvblox_camera_info_relay'" in driver_source


def test_attachment_uses_the_configured_grasp_frame_for_esdf_clearing():
    behavior_source = ATTACH_OBJECT.read_text()
    runner_source = WORKFLOW_RUNNER.read_text()
    grasp_config = yaml.safe_load(
        (WS_DIR / "config" / "rebot_grasps_soup_can.yaml").read_text()
    )
    attach_config = yaml.safe_load(BT_CONFIG.read_text())[
        "behavior_tree_params"
    ]["multi_object_pick_and_place"]["attach_object"]

    assert "frame_id=self.grasp_frame" in behavior_source
    assert "_attachment_mesh_pose(" in behavior_source
    assert (
        "grasp_pose_object_matrix @ object_pose_mesh_matrix"
        in behavior_source
    )
    assert attach_config["shape"] == "CUSTOM_MESH"
    assert attach_config["scale"] == [1.0, 1.0, 1.0]
    mesh_pose = grasp_config["object_geometry"]["mesh_pose_in_object"]
    assert mesh_pose["position"] == pytest.approx([0.0, 0.0, 0.0505])
    assert mesh_pose["orientation"]["w"] == pytest.approx(2 ** -0.5)
    assert mesh_pose["orientation"]["xyz"] == pytest.approx(
        [2 ** -0.5, 0.0, 0.0]
    )

    # The source is bottom-up, so object +Z points down while carried. Grasping
    # 11 mm from the semantic base leaves 90 mm of the 101 mm can hanging below
    # the TCP. The composed mesh center must preserve both ends, not approximate
    # the can as a point at the gripper.
    height_m = grasp_config["object_geometry"]["height_m"]
    grasp_z_m = grasp_config["grasps"]["grasp_0"]["position"][2]
    mesh_center_in_tcp_z_m = mesh_pose["position"][2] - grasp_z_m
    assert mesh_center_in_tcp_z_m == pytest.approx(0.0395)
    assert mesh_center_in_tcp_z_m - 0.5 * height_m == pytest.approx(-0.011)
    assert mesh_center_in_tcp_z_m + 0.5 * height_m == pytest.approx(0.090)

    assert ":/opt/ros/" not in runner_source
    packages = (WS_DIR / "docker" / "overlay-packages.txt").read_text()
    assert "isaac_ros_manipulation_orchestration" in packages


def test_release_pose_clears_mapped_tabletop_for_gravity_placement():
    source = PICK_GOAL.read_text()

    assert "RELEASE_TCP_HEIGHT_ABOVE_TABLE_M = 0.130" in source
    assert "UPRIGHT_OBJECT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)" in source
    assert "pa.place_centre()" in source
    assert "object_bottom_z" in source
    assert "float(grasp.position[2])" in source
    assert "compose_drop_target(" in source
