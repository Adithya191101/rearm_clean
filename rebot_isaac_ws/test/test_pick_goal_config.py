"""Unit tests for the ROS-free pick-goal command-line contract."""

from pathlib import Path
import sys

import pytest


SIM_DIR = Path(__file__).resolve().parents[1] / "sim"
sys.path.insert(0, str(SIM_DIR))

from pick_goal_config import DropTarget, compose_drop_target, parse_goal_options


SEND_PICK_GOAL = SIM_DIR / "send_pick_goal.py"


DEFAULT_TARGET = DropTarget(
    frame_id="base_link",
    x=0.350,
    y=0.250,
    z=0.280,
    qx=0.0,
    qy=0.0,
    qz=0.0,
    qw=1.0,
)


def test_defaults_preserve_the_validated_target_and_timeouts():
    options = parse_goal_options([], default_target=DEFAULT_TARGET)

    assert options.target == DEFAULT_TARGET
    assert options.server_timeout_s == pytest.approx(120.0)
    assert options.send_timeout_s == pytest.approx(30.0)
    assert options.result_timeout_s == pytest.approx(300.0)
    assert options.require_complete is False


def test_all_public_goal_overrides_are_applied():
    options = parse_goal_options(
        [
            "--frame-id",
            "map",
            "--drop-x",
            "0.4",
            "--drop-y",
            "-0.2",
            "--drop-z",
            "0.3",
            "--qx",
            "0",
            "--qy",
            "0",
            "--qz",
            "1",
            "--qw",
            "0",
            "--server-timeout",
            "10",
            "--send-timeout",
            "5",
            "--result-timeout",
            "60",
            "--require-complete",
        ],
        default_target=DEFAULT_TARGET,
    )

    assert options.target == DropTarget(
        frame_id="map",
        x=0.4,
        y=-0.2,
        z=0.3,
        qx=0.0,
        qy=0.0,
        qz=1.0,
        qw=0.0,
    )
    assert options.server_timeout_s == pytest.approx(10.0)
    assert options.send_timeout_s == pytest.approx(5.0)
    assert options.result_timeout_s == pytest.approx(60.0)
    assert options.require_complete is True


@pytest.mark.parametrize("value", ("nan", "inf", "-inf"))
def test_non_finite_pose_values_are_rejected(value):
    with pytest.raises(SystemExit):
        parse_goal_options(
            ["--drop-x", value],
            default_target=DEFAULT_TARGET,
        )


def test_non_normalized_quaternion_is_rejected():
    with pytest.raises(SystemExit, match="quaternion must be normalized"):
        parse_goal_options(
            ["--qw", "2.0"],
            default_target=DEFAULT_TARGET,
        )


@pytest.mark.parametrize("value", ("0", "-1"))
def test_non_positive_timeouts_are_rejected(value):
    with pytest.raises(SystemExit):
        parse_goal_options(
            ["--result-timeout", value],
            default_target=DEFAULT_TARGET,
        )


def test_invalid_default_target_is_rejected_before_argument_parsing():
    with pytest.raises(ValueError, match="frame_id"):
        parse_goal_options(
            [],
            default_target=DropTarget(
                frame_id="",
                x=0.0,
                y=0.0,
                z=0.0,
                qx=0.0,
                qy=0.0,
                qz=0.0,
                qw=1.0,
            ),
        )


def test_upright_object_pose_composes_to_reachable_bottom_first_tcp_target():
    target = compose_drop_target(
        frame_id="base_link",
        object_position=(0.35, 0.25, 0.269),
        object_quaternion_xyzw=(0.0, 0.0, 0.0, 1.0),
        grasp_position=(0.0, 0.0, 0.011),
        grasp_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
    )

    assert target == DropTarget(
        frame_id="base_link",
        x=pytest.approx(0.35),
        y=pytest.approx(0.25),
        z=pytest.approx(0.28),
        qx=pytest.approx(0.0),
        qy=pytest.approx(0.0),
        qz=pytest.approx(0.0),
        qw=pytest.approx(1.0),
    )


def test_pose_composition_rejects_non_normalized_input_quaternion():
    with pytest.raises(ValueError, match="object quaternion must be normalized"):
        compose_drop_target(
            frame_id="base_link",
            object_position=(0.0, 0.0, 0.0),
            object_quaternion_xyzw=(2.0, 0.0, 0.0, 0.0),
            grasp_position=(0.0, 0.0, 0.0),
            grasp_quaternion_wxyz=(1.0, 0.0, 0.0, 0.0),
        )


def test_live_goal_requests_an_upright_object_release():
    source = SEND_PICK_GOAL.read_text()
    default_target = source[
        source.index("def _default_target()"):
        source.index("\n\ndef main(", source.index("def _default_target()"))
    ]

    assert "UPRIGHT_OBJECT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)" in source
    assert "RELEASE_TCP_HEIGHT_ABOVE_TABLE_M = 0.130" in source
    assert "pa.place_centre()" in default_target
    assert "object_bottom_z" in default_target
    assert "RELEASE_TCP_HEIGHT_ABOVE_TABLE_M" in default_target
    assert "float(grasp.position[2])" in default_target
    assert "object_quaternion_xyzw=UPRIGHT_OBJECT_QUAT_XYZW" in default_target
    assert "INVERTED_OBJECT_QUAT_XYZW" not in source
