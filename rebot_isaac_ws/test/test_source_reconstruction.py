"""Contracts for the source-only repository reconstruction."""

from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]
WS = REPO_ROOT / "rebot_isaac_ws"
LOCK_FILE = REPO_ROOT / "dependencies.lock"
FETCH_SCRIPT = REPO_ROOT / "scripts" / "fetch_sources.sh"
PATCH_FILE = REPO_ROOT / "patches" / "isaac_ros_manipulation.patch"

EXPECTED_DEPENDENCIES = {
    "isaac_ros_manipulation": {
        "url": (
            "https://github.com/NVIDIA-ISAAC-ROS/"
            "isaac_ros_manipulation.git"
        ),
        "revision": "6ef8d72fee82f5aa0bae207962e9a17ff4306f90",
        "destination": "rebot_isaac_ws/src/isaac_ros_manipulation",
    },
    "topic_based_ros2_control": {
        "url": (
            "https://github.com/karanchahal-nv/"
            "topic_based_ros2_control.git"
        ),
        "revision": "7ee291ab13adba52ab5889deb9e520009fe2283d",
        "destination": "rebot_isaac_ws/src/topic_based_ros2_control",
    },
    "rebot_arm_controller": {
        "url": (
            "https://github.com/Seeed-Projects/"
            "reBotArmController_ROS2.git"
        ),
        "revision": "39fbea54c7235b1c38bd025fc2e7308e42bd2fbe",
        "destination": (
            "rebot_isaac_ws/.upstream/reBotArmController_ROS2"
        ),
    },
    "rebot_isaacsim": {
        "url": "https://github.com/Seeed-Projects/reBot-Isaacsim.git",
        "revision": "c3ee253ca113ea3514da442684ef5d4894219374",
        "destination": "rebot_isaac_ws/.upstream/reBot-Isaacsim",
    },
}

EXPECTED_OVERLAY_PACKAGES = {
    "isaac_ros_manipulation_bringup",
    "isaac_ros_manipulation_orchestration",
    "isaac_ros_manipulation_pick_and_place",
    "isaac_ros_manipulation_ros_python_utils",
    "isaac_ros_manipulation_test_utils",
    "isaac_ros_manipulation_rebot_driver_utils",
    "isaac_ros_manipulation_rebot_robot_description",
    "rebot_b601dm_description",
    "rebot_b601dm_perception",
    "topic_based_ros2_control",
}

VISUAL_FILES = {
    "base_link.STL",
    "gripper_left.STL",
    "gripper_link.STL",
    "gripper_right.STL",
    "link1.STL",
    "link2.STL",
    "link3.STL",
    "link4.STL",
    "link5.STL",
    "link6.STL",
}


def _dependencies() -> dict[str, dict[str, str]]:
    dependencies = {}
    for raw_line in LOCK_FILE.read_text().splitlines():
        if not raw_line or raw_line.startswith("#"):
            continue
        name, url, revision, destination, sparse_path = raw_line.split("|")
        dependencies[name] = {
            "url": url,
            "revision": revision,
            "destination": destination,
            "sparse_path": sparse_path,
        }
    return dependencies


def _git(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip()


def test_dependency_lock_uses_exact_reviewed_revisions():
    dependencies = _dependencies()
    assert set(dependencies) == set(EXPECTED_DEPENDENCIES)
    assert len({
        dependency["destination"]
        for dependency in dependencies.values()
    }) == len(dependencies)

    for name, expected in EXPECTED_DEPENDENCIES.items():
        actual = dependencies[name]
        assert actual["url"] == expected["url"]
        assert actual["revision"] == expected["revision"]
        assert len(actual["revision"]) == 40
        int(actual["revision"], 16)
        assert actual["destination"] == expected["destination"]
        assert not actual["destination"].startswith("/")
        assert ".." not in Path(actual["destination"]).parts


def test_generated_checkouts_have_locked_heads_and_remotes():
    for name, dependency in _dependencies().items():
        checkout = REPO_ROOT / dependency["destination"]
        assert (checkout / ".git").is_dir(), (
            f"{name} is missing; run ./scripts/fetch_sources.sh"
        )
        assert _git("rev-parse", "HEAD", cwd=checkout) == (
            dependency["revision"]
        )
        assert _git("remote", "get-url", "origin", cwd=checkout) == (
            dependency["url"]
        )


def test_isaac_ros_patch_is_complete_and_reversible():
    isaac_source = WS / "src" / "isaac_ros_manipulation"
    subprocess.run(
        [
            "git",
            "-C",
            str(isaac_source),
            "apply",
            "--reverse",
            "--check",
            str(PATCH_FILE),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    patch = PATCH_FILE.read_text()
    assert "axial_object_pose.py" in patch
    assert "wait_for_fresh_joint_state.py" in patch
    assert "test_orchestrator_scheduling.py" in patch
    assert "launch/workflows/core.launch.py" in patch
    assert "setup_perception_models.py" in patch
    assert "/tmp/rearm_clean" not in patch
    assert "/home/" not in patch


def test_foundationpose_setup_does_not_require_unused_synthetica_models():
    setup_script = (
        WS
        / "src"
        / "isaac_ros_manipulation"
        / "isaac_ros_manipulation_asset_bringup"
        / "scripts"
        / "setup_perception_models.py"
    ).read_text()
    assert "Synthetica detr models not found" not in setup_script


def test_upstream_core_launch_accepts_rebot_gripper():
    core_launch = (
        WS
        / "src"
        / "isaac_ros_manipulation"
        / "isaac_ros_manipulation_bringup"
        / "launch"
        / "workflows"
        / "core.launch.py"
    ).read_text()
    assert "'rebot_parallel'," in core_launch


def test_materialized_visual_meshes_are_byte_identical_to_seeed():
    source = (
        WS
        / ".upstream"
        / "reBotArmController_ROS2"
        / "src"
        / "rebotarm_bringup"
        / "description"
        / "meshes_b601_gripper"
    )
    generated = (
        WS / "src" / "rebot_b601dm_description" / "meshes" / "visual"
    )
    assert {path.name for path in generated.glob("*.STL")} == VISUAL_FILES
    for name in VISUAL_FILES:
        assert (source / name).read_bytes() == (generated / name).read_bytes()


def test_vendor_usd_is_a_verified_generated_link():
    vendor_link = WS / "usd" / "vendor" / "reBot_B601_DM"
    assert vendor_link.is_symlink()
    assert vendor_link.readlink() == Path(
        "../../.upstream/reBot-Isaacsim/usd/reBot_B601_DM"
    )
    root_layer = vendor_link / "reBot_B601_DM.usda"
    assert hashlib.sha256(root_layer.read_bytes()).hexdigest() == (
        "6b9d39de1200732c581c91e895bee412844e101006fb0c3df54259d81ee28e84"
    )


def test_overlay_build_contains_only_runtime_packages():
    package_file = WS / "docker" / "overlay-packages.txt"
    packages = {
        line
        for raw_line in package_file.read_text().splitlines()
        if (line := raw_line.strip()) and not line.startswith("#")
    }
    assert packages == EXPECTED_OVERLAY_PACKAGES

    excluded_project_packages = {
        "rebot_b601dm_adapters",
        "rebot_b601dm_calibration",
        "rebot_b601dm_orchestration",
        "rebot_b601dm_verification",
    }
    assert packages.isdisjoint(excluded_project_packages)
    for package in excluded_project_packages:
        assert not (WS / "src" / package).exists()


def test_clean_overlay_excludes_development_only_sim_tools():
    excluded = {
        "blooper_scenarios.py",
        "drop_plan_probe.py",
        "grasp_verify.py",
        "load_vendor_asset.py",
        "obstacle_map_probe.py",
        "physical_grasp_probe.py",
    }
    assert excluded.isdisjoint(path.name for path in (WS / "sim").glob("*.py"))


def test_bootstrap_prepares_sources_before_building():
    bootstrap = (REPO_ROOT / "scripts" / "bootstrap.sh").read_text()
    fetch_index = bootstrap.index('"$REPO_ROOT/scripts/fetch_sources.sh"')
    preflight_index = bootstrap.index('"$REPO_ROOT/scripts/preflight.sh"')
    docker_index = bootstrap.index("docker build")
    assert fetch_index < preflight_index < docker_index
    assert "BOOTSTRAP RESULT status=success" in bootstrap
    assert FETCH_SCRIPT.stat().st_mode & 0o111


def test_model_installer_sources_ros_without_nounset():
    installer = (REPO_ROOT / "scripts" / "install_models.sh").read_text()
    assert (
        "set +u\n"
        "    source /opt/ros/jazzy/setup.bash\n"
        "    set -u"
    ) in installer
    assert (
        "/opt/ros/jazzy/lib/isaac_ros_foundationpose_models_install/"
        "install_foundationpose_models.sh"
    ) in installer
    assert (
        "/opt/ros/jazzy/lib/isaac_ros_grounding_dino_models_install/"
        "install_grounding_dino_models.sh"
    ) in installer
    assert "CUDA_MPS_PIPE_DIRECTORY=/tmp/no-mps-model-install" in installer
    assert "--maxShapes=input1:42x160x160x6,input2:42x160x160x6" in (
        installer
    )
    assert "--maxShapes=input1:252x160x160x6,input2:252x160x160x6" in (
        installer
    )


def test_prepare_does_not_load_host_pytest_plugins():
    reproduce = (WS / "docker" / "reproduce.sh").read_text()
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1" in reproduce
    assert '"$PYTHON" -m pytest -q "$WS/test"' in reproduce


def test_overlay_manifest_uses_only_tracked_reconstruction_inputs():
    build_script = (WS / "docker" / "build_overlay.sh").read_text()
    container_runner = (WS / "docker" / "run.sh").read_text()

    assert "dependencies.lock" in build_script
    assert "dependencies_sha256=" in build_script
    assert "repos.yaml" not in build_script
    assert (
        '"${REPO_ROOT}/dependencies.lock:/workspaces/dependencies.lock:ro"'
        in container_runner
    )


def test_managed_demo_keeps_failure_capture_and_encoding_in_one_command():
    runner = (REPO_ROOT / "scripts" / "run_demo.sh").read_text()
    assert "--capture FILE" in runner
    assert "capture_pipeline_diagnostics.py" in runner
    assert "capture_ready.json" in runner
    assert "capture_summary.json" not in runner
    assert "SCENE_RENDER_FPS=60" in runner
    assert "scene_record_every=$((SCENE_RENDER_FPS / CAPTURE_FPS))" in runner
    assert '--record-every "$scene_record_every"' in runner
    assert "ffmpeg -y" in runner
    assert 'if [[ "$goal_status" -ne 0 ]]' in runner
    assert 'exit "$goal_status"' in runner
