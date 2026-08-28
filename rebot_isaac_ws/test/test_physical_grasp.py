"""Contracts for the contact-driven can collider and pick gate."""

from pathlib import Path
import sys

import pytest

SIM_DIR = Path(__file__).resolve().parents[1] / 'sim'
sys.path.insert(0, str(SIM_DIR))

from physical_grasp import (
    CAN_COLLIDER,
    CONTACT_OFFSET_M,
    DYNAMIC_FRICTION,
    FINGER_LINK_NAMES,
    FOLLOWER_JAW_JOINT,
    GRIPPER_MAX_EFFORT_N,
    JAW_DRIVE_DAMPING_N_S_PER_M,
    JAW_DRIVE_STIFFNESS_N_PER_M,
    MIMIC_DAMPING_RATIO,
    MIMIC_NATURAL_FREQUENCY_HZ,
    mimic_api_instances,
    PHYSICS_HZ,
    REST_OFFSET_M,
    SOLVER_POSITION_ITERATIONS,
    SOLVER_VELOCITY_ITERATIONS,
    STATIC_FRICTION,
    CanColliderSpec,
)


WS_DIR = Path(__file__).resolve().parents[1]
PICK_SCENE = WS_DIR / 'sim' / 'pick_scene.py'
CLOSE_BEHAVIOR = (
    WS_DIR / 'src' / 'isaac_ros_manipulation'
    / 'isaac_ros_manipulation_orchestration'
    / 'isaac_ros_manipulation_orchestration'
    / 'behaviors' / 'motion_behaviors' / 'close_gripper.py'
)
SCENE_LAUNCH = WS_DIR / 'sim' / 'run' / 'launch_scene_live.sh'
WORKFLOW_LAUNCH = WS_DIR / 'sim' / 'run' / 'launch_workflow_live.sh'
DRIVER_UTILS = (
    WS_DIR / 'src' / 'rebot_b601dm_isaac'
    / 'isaac_ros_manipulation_rebot_driver_utils'
)
DRIVER_LAUNCH = DRIVER_UTILS / 'launch' / 'rebot_driver.launch.py'
GRIPPER_BRIDGE = (
    DRIVER_UTILS / 'isaac_ros_manipulation_rebot_driver_utils'
    / 'sim_gripper_bridge.py'
)
BT_CONFIGS = (
    WS_DIR / 'config' / 'pick_and_place'
    / 'multi_object_pick_and_place_behavior_tree_params.yaml',
)


def test_can_collider_matches_canonical_geometry():
    assert CAN_COLLIDER.radius_m == pytest.approx(0.033, abs=0.001)
    assert CAN_COLLIDER.height_m == pytest.approx(0.101, abs=0.001)
    assert CAN_COLLIDER.center_z_m == pytest.approx(0.0505, abs=0.0005)
    assert CAN_COLLIDER.mass_kg == pytest.approx(0.349)


@pytest.mark.parametrize(
    'radius,height,mass',
    [
        (0.0, 0.101, 0.349),
        (0.033, -0.101, 0.349),
        (0.033, 0.101, float('nan')),
    ],
)
def test_can_collider_rejects_invalid_properties(radius, height, mass):
    with pytest.raises(ValueError, match='positive'):
        CanColliderSpec(radius, height, mass)


def test_contact_parameters_match_validated_physics_probe():
    assert STATIC_FRICTION == pytest.approx(1.2)
    assert DYNAMIC_FRICTION == pytest.approx(1.1)
    assert CONTACT_OFFSET_M == pytest.approx(0.001)
    assert REST_OFFSET_M == pytest.approx(0.0)
    assert PHYSICS_HZ == pytest.approx(120.0)
    assert GRIPPER_MAX_EFFORT_N == pytest.approx(60.0)
    assert JAW_DRIVE_STIFFNESS_N_PER_M == pytest.approx(5000.0)
    assert JAW_DRIVE_DAMPING_N_S_PER_M == pytest.approx(41.28)
    assert MIMIC_NATURAL_FREQUENCY_HZ == pytest.approx(1000.0)
    assert MIMIC_DAMPING_RATIO == pytest.approx(1.0)
    assert SOLVER_POSITION_ITERATIONS == 32
    assert SOLVER_VELOCITY_ITERATIONS == 4
    assert FINGER_LINK_NAMES == ('gripper_left', 'gripper_right')
    assert FOLLOWER_JAW_JOINT == 'gripper_joint2'


def test_finger_refinement_replaces_solid_collision_hulls():
    source = (WS_DIR / 'sim' / 'physical_grasp.py').read_text()
    assert '"/collisions/" in str(prim.GetPath())' in source
    assert '"/visuals/" in str(prim.GetPath())' in source
    assert 'source_collision.CreateCollisionEnabledAttr().Set(False)' in source
    assert 'UsdPhysics.CollisionAPI.Apply(replacement)' in source
    assert '"convexDecomposition"' in source


def test_mimic_schema_instances_select_only_physx_mimic_apis():
    assert mimic_api_instances([
        'UsdPhysicsDriveAPI:linear',
        'PhysxMimicJointAPI:rotX',
        'PhysxMimicJointAPI:transX',
    ]) == ('rotX', 'transX')


def test_pick_scene_has_no_kinematic_attachment_or_pose_writer():
    source = PICK_SCENE.read_text()
    forbidden = (
        'place_upright_kinematically',
        'update_grasp',
        'can_pose_writer',
        'set_world_pose(',
        'CreateKinematicEnabledAttr().Set(True)',
    )
    for token in forbidden:
        assert token not in source
    assert 'disable_runtime_jaw_mimic' not in source
    assert 'configure_dynamic_can(root)' in source
    assert 'create_can_collider(' in source
    assert 'refine_finger_colliders(' in source
    assert 'configure_independent_jaw_drives(stage, ROBOT_PRIM)' in source
    assert 'configure_runtime_jaw_mimic(stage, ROBOT_PRIM)' not in source
    assert 'PublishCanTf' not in source


def test_pick_scene_starts_and_gates_the_can_bottom_up():
    source = PICK_SCENE.read_text()
    add_can = source[source.index('def add_soup_can('):source.index(
        '\ndef assert_physics(', source.index('def add_soup_can('))]

    assert 'CAN_INITIAL_ROOT_Z = CAN_BASE_Z + CAN_COLLIDER.height_m' in source
    assert 'CAN_INITIAL_ROTATION_DEG = (180.0, 0.0, 0.0)' in source
    assert 'Gf.Vec3d(x, y, CAN_INITIAL_ROOT_Z)' in add_can
    assert 'rxf.AddRotateXYZOp().Set(Gf.Vec3f(*CAN_INITIAL_ROTATION_DEG))' in add_can
    assert 'if settled_upright > -0.95:' in source
    assert '"can did not remain bottom-up: upright_z=%.3f"' in source


def test_pick_scene_has_one_required_end_to_end_configuration():
    source = PICK_SCENE.read_text()
    removed_bypasses = (
        '--no-camera',
        '--no-can',
        '--no-tf',
        '--hold-home',
        '--no-appearance',
        '--no-markers',
        '--no-transfer-wall',
        '--no-environment',
        '--no-scene-cams',
        '--plain-can',
    )
    for option in removed_bypasses:
        assert option not in source


def test_pick_scene_uses_paired_force_limited_jaw_drives():
    source = PICK_SCENE.read_text()
    assert 'jaw_drive_indices' in source
    assert 'for index in jaw_drive_indices:' in source
    assert 'for i in jaw_drive_indices:' in source
    assert (
        'per_jaw_effort = GRIPPER_MAX_EFFORT_N / len(jaw_drive_indices)'
        in source
    )


def test_pick_close_requires_contact_before_lift():
    source = CLOSE_BEHAVIOR.read_text()
    assert 'if reached_goal or stalled:' not in source
    assert 'if stalled:' in source
    assert 'if reached_goal:' in source
    assert 'refusing to lift' in source


def test_sim_gripper_close_contact_cannot_be_disabled():
    bridge = GRIPPER_BRIDGE.read_text()
    launch = DRIVER_LAUNCH.read_text()

    assert 'require_contact_on_close' not in bridge
    assert 'completed = stalled if closing else reached' in bridge
    assert 'reached close target %.4f without contact' in bridge
    assert "executable='sim_gripper_bridge'" in launch


def test_scene_runs_through_repository_isaac_launcher():
    source = SCENE_LAUNCH.read_text()
    assert 'launch_isaacsim.sh" --python' in source
    assert '"$WS/sim/pick_scene.py"' in source
    assert '--duration "$DURATION" "$@"' in source


def test_static_scene_cameras_use_visible_non_physical_realsense_assets():
    source = PICK_SCENE.read_text()
    fixture_start = source.index('def add_scene_camera_fixture(')
    fixture_end = source.index('\ndef create_scene_camera(', fixture_start)
    fixture = source[fixture_start:fixture_end]

    assert 'add_scene_camera_fixture(stage, spec, -fwd, transform)' in source
    assert source.count('"visible_fixture": True') == 2
    assert '"visible_fixture": False' not in source
    assert 'REALSENSE_D455_URL' in source
    assert 'REALSENSE_D455_MODEL_PRIM = "RSD455"' in source
    assert (
        'REALSENSE_D455_COLOR_CAMERA = '
        '"Camera_OmniVision_OV9782_Color"'
    ) in source
    assert 'asset_root.GetPrim().GetReferences().AddReference(' in fixture
    assert 'GetLocalTransformation()' in fixture
    assert 'color_camera_transform.GetInverse()' in fixture
    assert 'make_scene_camera_asset_visual_only(' in source
    assert 'prim.RemoveAPI(UsdPhysics.RigidBodyAPI)' in source
    assert 'prim.RemoveAPI(UsdPhysics.CollisionAPI)' in source
    assert 'embedded_camera.SetActive(False)' in fixture
    assert 'REALSENSE_D455_VIEWPORT_MESH' not in source
    assert 'AddRotateYOp()' not in fixture
    assert 'AddRotateZOp()' not in fixture
    assert 'COLOR_OPTICAL_Y_OFFSET' not in fixture
    assert '"%s/housing" % housing_frame_path' not in fixture
    assert '"%s/bench_clamp" % mount_path' in fixture
    assert '"%s/post" % mount_path' in fixture
    assert 'collision=True' not in fixture
    assert 'UsdPhysics.CollisionAPI' not in fixture


def test_docker_workflow_builds_contact_gated_close_behavior_in_overlay():
    source = WORKFLOW_LAUNCH.read_text()
    packages = (WS_DIR / 'docker' / 'overlay-packages.txt').read_text()

    assert ':/opt/ros/' not in source
    assert 'docker/verify_overlay.sh' in source
    assert 'isaac_ros_manipulation_orchestration' in packages
    assert 'sim/gripper_stub_server.py' not in source
    assert 'sim/nvblox_camera_info_relay.py' not in source


@pytest.mark.parametrize('config_path', BT_CONFIGS)
def test_close_effort_matches_simulated_gripper_limit(config_path):
    import yaml

    document = yaml.safe_load(config_path.read_text())
    close = document['behavior_tree_params'][
        'multi_object_pick_and_place']['close_gripper']
    assert close['close_position'] == pytest.approx(0.019)
    assert close['max_effort'] == pytest.approx(GRIPPER_MAX_EFFORT_N)
