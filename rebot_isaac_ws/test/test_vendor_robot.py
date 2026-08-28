"""Contracts for the official Seeed USD production runtime."""

import ast
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml


WS_DIR = Path(__file__).resolve().parents[1]
SIM_DIR = WS_DIR / "sim"
sys.path.insert(0, str(SIM_DIR))

import vendor_robot as vendor


PICK_SCENE = SIM_DIR / "pick_scene.py"
PHYSICAL_GRASP = SIM_DIR / "physical_grasp.py"
VENDOR_README = WS_DIR / "usd" / "vendor" / "README.md"
ISAAC_LAUNCHER = WS_DIR.parent / "launch_isaacsim.sh"
INITIAL_POSITIONS = (
    WS_DIR
    / "src"
    / "rebot_b601dm_isaac"
    / "isaac_ros_manipulation_rebot_robot_description"
    / "config"
    / "initial_positions.yaml"
)
XRDF_DIR = WS_DIR / "config" / "xrdf"
FULL_URDF = (
    WS_DIR
    / "src"
    / "rebot_b601dm_description"
    / "urdf"
    / "generated"
    / "rebot_b601dm_full.urdf"
)
KINEMATICS_TEST = (
    WS_DIR
    / "src"
    / "rebot_b601dm_description"
    / "test"
    / "test_kinematics.py"
)


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(
            isinstance(target, ast.Name) and target.id == name
            for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name} is not a module-level literal in {path}")


def _urdf_fk():
    spec = importlib.util.spec_from_file_location(
        "rebot_description_kinematics",
        KINEMATICS_TEST,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.UrdfFK(FULL_URDF.read_text())


def test_official_seeed_root_layer_is_pinned_byte_for_byte():
    assert vendor.ASSET_PATH.is_file()
    assert vendor.asset_sha256() == vendor.OFFICIAL_ROOT_LAYER_SHA256
    assert vendor.OFFICIAL_ROOT_LAYER_SHA256 in VENDOR_README.read_text()


def test_vendor_articulation_contract_has_exactly_eight_ordered_dofs():
    assert vendor.EXPECTED_DOF_NAMES == [
        "joint1",
        "joint2",
        "joint3",
        "joint4",
        "joint5",
        "joint6",
        "gripper_joint1",
        "gripper_joint2",
    ]
    assert len(vendor.RUNTIME_KP) == len(vendor.EXPECTED_DOF_NAMES) == 8
    assert len(vendor.RUNTIME_KD) == len(vendor.EXPECTED_DOF_NAMES) == 8
    assert np.all(vendor.RUNTIME_KP > 0.0)
    assert np.all(vendor.RUNTIME_KD > 0.0)
    assert "if dof_names != expected_dof_order:" in (
        PICK_SCENE.read_text()
    )


def test_neutral_start_pose_is_interior_and_synchronized_across_runtime_inputs():
    expected = {
        "joint1": -0.6,
        "joint2": -0.75,
        "joint3": -0.75,
        "joint4": 0.0,
        "joint5": 0.0,
        "joint6": 0.0,
    }
    arm_q = np.array(list(expected.values()), dtype=np.float64)

    assert np.all(arm_q > vendor.EXPECTED_LOWER[:6])
    assert np.all(arm_q < vendor.EXPECTED_UPPER[:6])
    assert _literal_assignment(PICK_SCENE, "START_Q") == expected
    assert yaml.safe_load(INITIAL_POSITIONS.read_text()) == expected

    tool_free_defaults = yaml.safe_load(
        (XRDF_DIR / "rebot_b601dm_tool_free.xrdf").read_text()
    )["default_joint_positions"]
    gripper_defaults = yaml.safe_load(
        (XRDF_DIR / "rebot_b601dm_gripper.xrdf").read_text()
    )["default_joint_positions"]
    assert tool_free_defaults == expected
    assert gripper_defaults == {**expected, "gripper_joint1": 0.0715}


def test_start_pose_places_horizontal_gripper_high_and_away_from_can():
    expected = _literal_assignment(PICK_SCENE, "START_Q")
    fk = _urdf_fk()
    tcp = fk.fk("gripper_tcp", expected)
    link5 = fk.fk("link5", expected)
    link6 = fk.fk("link6", expected)
    gripper = fk.fk("gripper_link", expected)
    terminal_direction = link6[:3, 3] - link5[:3, 3]
    terminal_direction /= np.linalg.norm(terminal_direction)
    gripper_rail_axis = gripper[:3, 1]
    rail_terminal_alignment = abs(float(np.dot(
        terminal_direction,
        gripper_rail_axis,
    )))
    can_xy = np.asarray([0.37, -0.06])

    assert tcp[:3, 3] == pytest.approx(
        [0.236728, -0.162011, 0.371653],
        abs=1.0e-5,
    )
    assert tcp[2, 3] - 0.31 > 0.05
    assert np.linalg.norm(tcp[:2, 3] - can_xy) > 0.15
    assert expected["joint4"] == expected["joint5"] == 0.0
    assert expected["joint6"] == 0.0
    assert rail_terminal_alignment < 0.01
    assert abs(float(gripper_rail_axis[2])) < 1.0e-4


def test_vendor_runtime_paths_match_the_composed_asset_hierarchy():
    assert vendor.ROBOT_PRIM_PATH == "/World/reBot_B601_DM"
    assert vendor.ARTICULATION_ROOT_PATH.endswith("/Geometry/base_link")
    assert vendor.LINK6_PATH.endswith("/link4/link5/link6")
    assert vendor.FINGER_PATHS == (
        f"{vendor.GRIPPER_LINK_PATH}/gripper_left",
        f"{vendor.GRIPPER_LINK_PATH}/gripper_right",
    )
    assert vendor.CAMERA_PRIM_PATH.endswith(
        "/camera_color_optical_frame/rgbd_camera"
    )
    assert vendor.EXPECTED_NESTED_RIGID_BODY_COUNT == 9
    assert "prim.HasAPI(UsdPhysics.RigidBodyAPI)" in (
        SIM_DIR / "vendor_robot.py"
    ).read_text()
    assert "repair = vendor.repair_vendor_stage(stage)" in (
        PICK_SCENE.read_text()
    )


def test_named_path_discovery_rejects_missing_or_duplicate_links():
    paths = [
        "/World/reBot/Geometry/base_link",
        "/World/reBot/Geometry/base_link/link1",
        "/World/reBot/Geometry/base_link/link1/link2",
    ]
    assert vendor.select_unique_named_paths(
        paths, ("base_link", "link1", "link2")
    ) == {
        "base_link": paths[0],
        "link1": paths[1],
        "link2": paths[2],
    }

    with pytest.raises(RuntimeError, match="link2=\\[\\]"):
        vendor.select_unique_named_paths(paths[:2], ("base_link", "link2"))
    with pytest.raises(RuntimeError, match="link1="):
        vendor.select_unique_named_paths(
            paths + ["/Other/link1"], ("base_link", "link1")
        )


def test_vendor_camera_frames_reuse_the_ros_extrinsics_contract():
    extrinsics = vendor.load_sim_extrinsics()
    assert extrinsics["camera_mount"]["parent"] == "link6"
    assert extrinsics["camera_mount"]["xyz"] == [0.05, 0.0, 0.06]
    assert extrinsics["intrinsics"]["optical_frame"] == (
        "camera_color_optical_frame"
    )
    assert extrinsics["intrinsics"]["width"] == 1280
    assert extrinsics["intrinsics"]["height"] == 720
    assert vendor.GRIPPER_TCP_XYZ_M == (-0.0443, 0.0, 0.0)


def test_pick_scene_uses_vendor_visuals_and_physics_by_default():
    source = PICK_SCENE.read_text()
    assert "VENDOR_ASSET_PATH = vendor.ASSET_PATH" in source
    assert "USD_PATH = _args.usd or str(VENDOR_ASSET_PATH)" in source
    assert "vendor.repair_vendor_stage(stage)" in source
    assert "vendor.author_ros_frames_and_camera(stage)" in source
    assert "SingleArticulation(prim_path=articulation_root" in source
    assert "vendor.RUNTIME_KP.copy()" in source
    assert "vendor.RUNTIME_KD.copy()" in source
    assert "apply_rebot_appearance" not in source
    assert 'sys.stderr.write("FATAL pick_scene:' in source


def test_pick_scene_wires_tensor_sync_into_every_rendered_step():
    source = PICK_SCENE.read_text()
    assert "vendor.LinkUsdSync(stage, arm, repaired_body_paths)" in source
    assert "link_sync.push()" in source
    assert source.count("sim.step(render=True)") == 1
    assert "def step_rendered() -> None:" in source

    adapter = (SIM_DIR / "vendor_robot.py").read_text()
    assert "get_link_transforms()" in adapter
    assert "def link_transforms(self)" in adapter
    assert "def body_names(self)" in adapter
    assert "xformOp:transform:b601PhysxRepair" in adapter
    assert "Sdf.ChangeBlock()" in adapter


def test_pick_scene_uses_reference_lighting_and_observer_framing():
    source = PICK_SCENE.read_text()
    assert '"inputs:intensity": 800.0' in source
    assert '"inputs:intensity": 2200.0' in source
    assert '"inputs:angle": 2.0' in source
    assert "RECORD_CAMERA_TARGET = MAIN_CAMERA_TARGET" in source
    assert "RECORD_CAMERA_EYE_OFFSET = MAIN_CAMERA_EYE_OFFSET" in source
    assert "RECORD_CAMERA_FOCAL_LENGTH_MM = MAIN_CAMERA_FOCAL_LENGTH_MM" in source


def test_ros_topic_and_frame_names_remain_unchanged():
    source = PICK_SCENE.read_text()
    expected = (
        'TOPIC_CLOCK = "/clock"',
        'TOPIC_JOINT_STATES = "/isaac_joint_states"',
        'TOPIC_JOINT_COMMANDS = "/isaac_joint_commands"',
        'TOPIC_RGB = "/front_stereo_camera/left/image_raw"',
        'TOPIC_CAMERA_INFO = "/front_stereo_camera/left/camera_info"',
        'TOPIC_DEPTH = "/front_stereo_camera/depth/ground_truth"',
        'CAMERA_FRAME_ID = "camera_color_optical_frame"',
    )
    for contract in expected:
        assert contract in source


def test_physical_grasp_supports_vendor_colliders_and_independent_jaws():
    source = PHYSICAL_GRASP.read_text()
    assert "authored vendor mesh collider" in source
    assert "prim.HasAPI(UsdPhysics.MeshCollisionAPI)" in source
    assert "prim.HasAPI(UsdPhysics.RigidBodyAPI)" in source
    assert "instances = mimic_api_instances" in source
    assert "for instance in instances:" in source
    assert "CreateEnabledSelfCollisionsAttr(False)" in source


def test_isaac_launcher_removes_inherited_host_ros_python_paths():
    source = ISAAC_LAUNCHER.read_text()
    assert 'if [[ ! -d "$runtime_dir" || ! -w "$runtime_dir" ]]' in source
    assert 'runtime_dir="/tmp"' in source
    assert 'export ROS_LOG_DIR="$ros_log_dir"' in source
    assert "unset PYTHONPATH AMENT_PREFIX_PATH" in source
    assert '[[ "$entry" == /opt/ros/* ]] && continue' in source
    assert "unset ROS_DISTRO" not in source
    assert "unset RMW_IMPLEMENTATION" not in source


def test_isaac_launcher_forces_a_writable_ros_log_directory(tmp_path):
    isaac_root = tmp_path / "isaacsim"
    isaac_root.mkdir()
    python_launcher = isaac_root / "python.sh"
    python_launcher.write_text(
        "#!/usr/bin/env bash\n"
        'printf "%s\\n" "${ROS_LOG_DIR:-missing}"\n'
        '[[ -d "$ROS_LOG_DIR" && -w "$ROS_LOG_DIR" ]]\n'
    )
    python_launcher.chmod(0o755)
    sim_launcher = isaac_root / "isaac-sim.sh"
    sim_launcher.write_text("#!/usr/bin/env bash\nexit 0\n")
    sim_launcher.chmod(0o755)

    env = os.environ.copy()
    env["ISAACSIM_ROOT"] = str(isaac_root)
    env["ROS_LOG_DIR"] = str(tmp_path / "inherited-read-only-log-dir")
    result = subprocess.run(
        [str(ISAAC_LAUNCHER), "--python", "unused.py"],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == f"/tmp/rebot-isaac-ros-{os.getuid()}"
