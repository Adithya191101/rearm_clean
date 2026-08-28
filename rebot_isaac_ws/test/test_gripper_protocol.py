"""Unit tests for the ROS-free simulated gripper contract."""

import math
from pathlib import Path
import sys

import pytest

DRIVER_UTILS_MODULE_DIR = (
    Path(__file__).resolve().parents[1]
    / 'src' / 'rebot_b601dm_isaac'
    / 'isaac_ros_manipulation_rebot_driver_utils'
    / 'isaac_ros_manipulation_rebot_driver_utils'
)
sys.path.insert(0, str(DRIVER_UTILS_MODULE_DIR))

from gripper_protocol import (
    ClosingStallDetector,
    contact_stall_detected,
    estimated_drive_effort,
    effective_jaw_target,
    finite_effort_magnitude,
    JAW_CONTACT_M,
    JAW_CONTACT_WINDOW_M,
    JAW_CLOSE_SPEED_M_PER_S,
    JAW_COMMAND_JOINTS,
    JAW_JOINTS,
    JAW_MAX_OPEN_M,
    jaw_target_reached,
    paired_drive_effort,
    paired_jaw_target_reached,
    paired_jaw_position,
    ramped_jaw_target,
    symmetric_jaw_position,
)


@pytest.mark.parametrize('requested', [0.0, 0.030, JAW_CONTACT_M])
def test_force_close_commands_the_requested_target(requested):
    target, closing = effective_jaw_target(requested)
    assert closing is True
    assert target == pytest.approx(requested)


def test_open_command_preserves_requested_clearance():
    target, closing = effective_jaw_target(0.045)
    assert closing is False
    assert target == pytest.approx(0.045)


def test_open_command_is_clamped_to_the_joint_limit():
    target, closing = effective_jaw_target(1.0)
    assert closing is False
    assert target == pytest.approx(JAW_MAX_OPEN_M)


def test_close_feedback_uses_the_contact_side_of_the_target():
    assert jaw_target_reached(
        0.0304, 0.030, closing=True)
    assert not jaw_target_reached(
        JAW_CONTACT_M, 0.030, closing=True)


def test_open_feedback_requires_clearance():
    assert jaw_target_reached(0.0711, JAW_MAX_OPEN_M, closing=False)
    assert not jaw_target_reached(0.065, JAW_MAX_OPEN_M, closing=False)


def test_open_feedback_requires_both_jaws_to_clear_requested_target():
    assert paired_jaw_target_reached(
        (0.0518, JAW_MAX_OPEN_M), 0.045, closing=False)
    assert not paired_jaw_target_reached(
        (0.040, JAW_MAX_OPEN_M), 0.045, closing=False)


def test_close_feedback_requires_both_jaws_to_reach_target():
    assert paired_jaw_target_reached(
        (0.0192, 0.0194), 0.019, closing=True)
    assert not paired_jaw_target_reached(
        (0.0192, 0.024), 0.019, closing=True)


def test_close_stall_requires_travel_then_a_stable_stop():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not detector.observe(0.0715, now_sec=0.0)
    assert not detector.observe(0.0670, now_sec=0.1)
    assert not detector.observe(0.0651, now_sec=0.2)
    assert not detector.observe(0.0651, now_sec=0.6)
    assert detector.observe(0.0651, now_sec=0.71)


def test_stationary_open_jaw_is_not_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not detector.observe(0.0715, now_sec=0.0)
    assert not detector.observe(0.0715, now_sec=1.0)


def test_loaded_stall_short_of_target_is_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector, 0.0715, 0.030, 10.0, now_sec=0.0)
    assert not contact_stall_detected(
        detector, 0.0400, 0.030, 10.0, now_sec=0.1)
    assert not contact_stall_detected(
        detector, JAW_CONTACT_M, 0.030, 10.0, now_sec=0.2)
    assert contact_stall_detected(
        detector, JAW_CONTACT_M, 0.030, 10.0, now_sec=0.71)


def test_stall_tracks_travel_before_preload_can_report_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector,
        0.0715,
        0.019,
        40.0,
        estimated_effort_n=0.0,
        now_sec=0.0,
        preload_complete=False,
    )
    assert not contact_stall_detected(
        detector,
        0.0325,
        0.019,
        40.0,
        estimated_effort_n=10.0,
        now_sec=0.1,
        preload_complete=False,
    )
    assert contact_stall_detected(
        detector,
        0.0325,
        0.019,
        40.0,
        estimated_effort_n=6.5,
        now_sec=0.7,
        preload_complete=True,
    )


def test_wide_loaded_stall_is_not_can_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector, 0.0715, 0.030, 40.0, now_sec=0.0)
    assert not contact_stall_detected(
        detector, 0.0668, 0.030, 40.0, now_sec=0.1)
    assert not contact_stall_detected(
        detector, 0.0668, 0.030, 40.0, now_sec=0.7)


def test_contact_window_covers_nominal_can_width_only():
    assert JAW_CONTACT_M + JAW_CONTACT_WINDOW_M < 0.05


def test_physical_gripper_commands_and_checks_both_jaws():
    assert JAW_COMMAND_JOINTS == (
        'gripper_joint1', 'gripper_joint2')
    assert JAW_JOINTS == ('gripper_joint1', 'gripper_joint2')


def test_asymmetric_physical_jaws_report_mean_aperture_coordinate():
    assert paired_jaw_position(0.0447, 0.0203) == pytest.approx(0.0325)


@pytest.mark.parametrize(
    'first,second',
    [
        (None, 0.034),
        (0.034, None),
        (math.nan, 0.034),
    ],
)
def test_paired_jaw_feedback_requires_two_finite_positions(first, second):
    assert paired_jaw_position(first, second) is None


def test_symmetric_jaw_feedback_returns_paired_position():
    assert symmetric_jaw_position(0.0340, 0.0342) == pytest.approx(0.0341)


@pytest.mark.parametrize(
    'first,second',
    [
        (None, 0.034),
        (0.034, None),
        (0.034, 0.040),
        (math.nan, 0.034),
    ],
)
def test_missing_or_asymmetric_jaw_feedback_is_rejected(first, second):
    assert symmetric_jaw_position(first, second) is None


def test_reaching_close_target_in_empty_air_is_not_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector, 0.0715, 0.030, 10.0, now_sec=0.0)
    assert not contact_stall_detected(
        detector, 0.0400, 0.030, 10.0, now_sec=0.1)
    assert not contact_stall_detected(
        detector, 0.0300, 0.030, 10.0, now_sec=0.2)
    assert not contact_stall_detected(
        detector, 0.0300, 0.030, 10.0, now_sec=1.0)


def test_drive_effort_is_derived_from_blocked_tracking_error():
    assert estimated_drive_effort(
        JAW_CONTACT_M, 0.030, 10.0) == pytest.approx(10.0)
    assert estimated_drive_effort(
        0.030, 0.030, 10.0) == pytest.approx(0.0)


def test_bilateral_drive_effort_uses_less_loaded_jaw():
    assert paired_drive_effort(
        (0.0447, 0.0203), 0.019, 40.0) == pytest.approx(6.5)


def test_bilateral_drive_effort_rejects_an_unloaded_jaw():
    assert paired_drive_effort(
        (0.0447, 0.019), 0.019, 40.0) == pytest.approx(0.0)


def test_close_target_ramps_at_validated_speed():
    assert ramped_jaw_target(
        0.0715, 0.019, 1.0) == pytest.approx(
            0.0715 - JAW_CLOSE_SPEED_M_PER_S)
    assert ramped_jaw_target(
        0.0715, 0.019, 20.0) == pytest.approx(0.019)


def test_bilateral_effort_override_prevents_one_sided_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector,
        0.0715,
        0.019,
        40.0,
        estimated_effort_n=0.0,
        now_sec=0.0,
    )
    assert not contact_stall_detected(
        detector,
        0.033,
        0.019,
        40.0,
        estimated_effort_n=0.0,
        now_sec=0.1,
    )
    assert not contact_stall_detected(
        detector,
        0.033,
        0.019,
        40.0,
        estimated_effort_n=0.0,
        now_sec=0.7,
    )


def test_stall_without_enough_drive_load_is_not_contact():
    detector = ClosingStallDetector(timeout_sec=0.5)

    assert not contact_stall_detected(
        detector, 0.0715, 0.030, 0.5, now_sec=0.0)
    assert not contact_stall_detected(
        detector, 0.03075, 0.030, 0.5, now_sec=0.1)
    assert not contact_stall_detected(
        detector,
        0.03075,
        0.030,
        0.5,
        measured_effort_n=math.nan,
        now_sec=0.7,
    )


@pytest.mark.parametrize(
    'effort,expected',
    [
        (None, 0.0),
        (math.nan, 0.0),
        (math.inf, 0.0),
        (-2.5, 2.5),
    ],
)
def test_effort_feedback_is_finite_and_nonnegative(effort, expected):
    assert finite_effort_magnitude(effort) == pytest.approx(expected)


@pytest.mark.parametrize('measured', [None, math.nan, math.inf])
def test_missing_or_nonfinite_feedback_never_completes(measured):
    assert not jaw_target_reached(measured, JAW_CONTACT_M, closing=True)


def test_nonfinite_command_is_rejected():
    with pytest.raises(ValueError, match='finite'):
        effective_jaw_target(math.nan)


def test_negative_command_is_rejected():
    with pytest.raises(ValueError, match='negative'):
        effective_jaw_target(-0.001)
