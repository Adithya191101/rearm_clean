# Copyright (c) 2026 reBot Isaac integration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
"""ROS-free constants and predicates for the simulated parallel gripper."""

from __future__ import annotations

import math
import time


GRIPPER_ACTION_NAME = "/gripper_adapter/gripper_cmd"
GRIPPER_COMMAND_TOPIC = "/sim_gripper/joint_command"
ISAAC_JOINT_STATES_TOPIC = "/isaac_joint_states"

DRIVING_JAW_JOINT = "gripper_joint1"
FOLLOWING_JAW_JOINT = "gripper_joint2"
JAW_JOINTS = (DRIVING_JAW_JOINT, FOLLOWING_JAW_JOINT)
# The imported compliant mimic preserves unloaded symmetry but transmits almost
# no force to one finger under contact. Two equal, force-limited PhysX drives
# model the mechanically coupled jaws without attaching or moving the object.
JAW_COMMAND_JOINTS = JAW_JOINTS
JAW_CONTACT_M = 0.034
JAW_CONTACT_WINDOW_M = 0.006
JAW_MAX_OPEN_M = 0.0715
JAW_RELEASE_M = JAW_MAX_OPEN_M
JAW_POSITION_TOLERANCE_M = 0.0005
JAW_SYMMETRY_TOLERANCE_M = 0.002
JAW_CLOSE_SPEED_M_PER_S = 0.005
JAW_STALL_MIN_TRAVEL_M = 0.001
JAW_STALL_PROGRESS_M = 0.0001
JAW_STALL_TIMEOUT_SEC = 0.5
JAW_DRIVE_STIFFNESS_N_PER_M = 5000.0
JAW_CONTACT_EFFORT_N = 1.0


class ClosingStallDetector:
    """Detect contact after closing motion reaches a stable physical stop."""

    def __init__(
        self,
        *,
        min_travel_m: float = JAW_STALL_MIN_TRAVEL_M,
        progress_m: float = JAW_STALL_PROGRESS_M,
        timeout_sec: float = JAW_STALL_TIMEOUT_SEC,
    ) -> None:
        self._min_travel_m = float(min_travel_m)
        self._progress_m = float(progress_m)
        self._timeout_sec = float(timeout_sec)
        self._start_m = None
        self._best_m = None
        self._last_progress_sec = None

    def observe(self, measured_m: float | None, now_sec: float | None = None) -> bool:
        if measured_m is None:
            return False
        measured = float(measured_m)
        now = time.monotonic() if now_sec is None else float(now_sec)
        if not math.isfinite(measured) or not math.isfinite(now):
            return False

        if self._start_m is None:
            self._start_m = measured
            self._best_m = measured
            self._last_progress_sec = now
            return False

        if measured < self._best_m - self._progress_m:
            self._best_m = measured
            self._last_progress_sec = now

        travelled = self._start_m - self._best_m
        stable_for = now - self._last_progress_sec
        return (
            travelled >= self._min_travel_m
            and stable_for >= self._timeout_sec
        )


def effective_jaw_target(
    requested_m: float,
    contact_m: float = JAW_CONTACT_M,
    max_open_m: float = JAW_MAX_OPEN_M,
) -> tuple[float, bool]:
    """
    Return the physical drive target and whether this is a close command.

    Force-close goals are not clamped at the expected can width. The drive must
    target the requested narrower position so PhysX contact creates a sustained
    tracking error and normal force.
    """
    requested = float(requested_m)
    contact = float(contact_m)
    max_open = float(max_open_m)
    if not all(math.isfinite(value) for value in (requested, contact, max_open)):
        raise ValueError("jaw positions must be finite")
    if not 0.0 <= contact <= max_open:
        raise ValueError("jaw contact must be inside the simulated joint limits")
    if requested < 0.0:
        raise ValueError("jaw position must not be negative")

    closing = requested <= contact
    if closing:
        return requested, True
    # Open commands are clearance requests, not requests for maximum travel.
    # Requiring full travel made a successful release fail when one unloaded jaw
    # stopped beyond the requested clearance but short of its mechanical limit.
    return min(requested, max_open), False


def estimated_drive_effort(
    measured_m: float | None,
    target_m: float,
    max_effort_n: float,
    *,
    stiffness_n_per_m: float = JAW_DRIVE_STIFFNESS_N_PER_M,
) -> float:
    """
    Estimate closing load from the position-drive error.

    Isaac's ROS joint-state publisher does not consistently expose articulation
    force across CPU/GPU backends. The position drive still obeys ``F = k*x``,
    so its blocked tracking error is a deterministic load estimate.
    """
    if measured_m is None:
        return 0.0
    measured = float(measured_m)
    target = float(target_m)
    maximum = float(max_effort_n)
    stiffness = float(stiffness_n_per_m)
    if not all(math.isfinite(value) for value in (
            measured, target, maximum, stiffness)):
        return 0.0
    if maximum < 0.0 or stiffness <= 0.0:
        return 0.0
    load = max(0.0, measured - target) * stiffness
    return min(load, maximum) if maximum > 0.0 else load


def finite_effort_magnitude(measured_effort_n: float | None) -> float:
    """Return a finite, non-negative effort suitable for action feedback."""
    if measured_effort_n is None:
        return 0.0
    try:
        magnitude = abs(float(measured_effort_n))
    except (TypeError, ValueError):
        return 0.0
    return magnitude if math.isfinite(magnitude) else 0.0


def ramped_jaw_target(
    start_m: float,
    target_m: float,
    elapsed_sec: float,
    *,
    speed_m_per_s: float = JAW_CLOSE_SPEED_M_PER_S,
) -> float:
    """Ramp a close target slowly enough to avoid impact-driven ejection."""
    start = float(start_m)
    target = float(target_m)
    elapsed = float(elapsed_sec)
    speed = float(speed_m_per_s)
    if not all(math.isfinite(value) for value in (
            start, target, elapsed, speed)):
        raise ValueError("jaw ramp inputs must be finite")
    if speed <= 0.0:
        raise ValueError("jaw close speed must be positive")
    return max(target, start - speed * max(0.0, elapsed))


def paired_jaw_position(
    first_m: float | None,
    second_m: float | None,
) -> float | None:
    """Return the mean physical aperture coordinate for two finite jaws."""
    if first_m is None or second_m is None:
        return None
    try:
        first = float(first_m)
        second = float(second_m)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(first) or not math.isfinite(second):
        return None
    return 0.5 * (first + second)


def paired_jaw_target_reached(
    jaw_positions_m: tuple[float, float] | None,
    target_m: float,
    closing: bool,
    tolerance_m: float = JAW_POSITION_TOLERANCE_M,
) -> bool:
    """Require both physical jaws to reach the requested target."""
    if jaw_positions_m is None or len(jaw_positions_m) != len(JAW_JOINTS):
        return False
    try:
        positions = tuple(float(position) for position in jaw_positions_m)
        target = float(target_m)
        tolerance = float(tolerance_m)
    except (TypeError, ValueError):
        return False
    if (
        not all(math.isfinite(position) for position in positions)
        or not math.isfinite(target)
        or not math.isfinite(tolerance)
        or tolerance < 0.0
    ):
        return False
    if closing:
        return max(positions) <= target + tolerance
    return min(positions) >= target - tolerance


def paired_drive_effort(
    jaw_positions_m: tuple[float, float] | None,
    target_m: float,
    max_effort_n: float,
) -> float:
    """Estimate bilateral load using the less-loaded physical jaw drive."""
    if jaw_positions_m is None or len(jaw_positions_m) != len(JAW_JOINTS):
        return 0.0
    maximum = finite_effort_magnitude(max_effort_n)
    if maximum <= 0.0:
        return 0.0
    per_jaw_maximum = maximum / len(JAW_JOINTS)
    efforts = [
        estimated_drive_effort(position, target_m, per_jaw_maximum)
        for position in jaw_positions_m
    ]
    return min(efforts)


def symmetric_jaw_position(
    first_m: float | None,
    second_m: float | None,
    *,
    tolerance_m: float = JAW_SYMMETRY_TOLERANCE_M,
) -> float | None:
    """Return the paired-jaw position only when both jaws agree."""
    if first_m is None or second_m is None:
        return None
    try:
        first = float(first_m)
        second = float(second_m)
        tolerance = float(tolerance_m)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (first, second, tolerance)):
        return None
    if tolerance < 0.0 or abs(first - second) > tolerance:
        return None
    return 0.5 * (first + second)


def contact_stall_detected(
    detector: ClosingStallDetector,
    measured_m: float | None,
    target_m: float,
    max_effort_n: float,
    *,
    measured_effort_n: float | None = None,
    estimated_effort_n: float | None = None,
    now_sec: float | None = None,
    minimum_effort_n: float = JAW_CONTACT_EFFORT_N,
    contact_m: float = JAW_CONTACT_M,
    contact_window_m: float = JAW_CONTACT_WINDOW_M,
    preload_complete: bool = True,
) -> bool:
    """Track closing travel and report a loaded stall after full preload."""
    if measured_m is None:
        return False
    motion_stalled = detector.observe(measured_m, now_sec=now_sec)
    if not preload_complete:
        return False
    if jaw_target_reached(measured_m, target_m, closing=True):
        return False
    if not motion_stalled:
        return False
    try:
        measured = float(measured_m)
        window = float(contact_window_m)
        maximum_contact = float(contact_m) + window
    except (TypeError, ValueError):
        return False
    if (
        not math.isfinite(maximum_contact)
        or window < 0.0
        or measured > maximum_contact
    ):
        return False

    estimated = (
        estimated_drive_effort(measured_m, target_m, max_effort_n)
        if estimated_effort_n is None
        else finite_effort_magnitude(estimated_effort_n)
    )
    measured_effort = finite_effort_magnitude(measured_effort_n)
    return max(estimated, measured_effort) >= float(minimum_effort_n)


def jaw_target_reached(
    measured_m: float | None,
    target_m: float,
    closing: bool,
    tolerance_m: float = JAW_POSITION_TOLERANCE_M,
) -> bool:
    """Return whether measured jaw feedback satisfies a close/open target."""
    if measured_m is None:
        return False
    measured = float(measured_m)
    if not math.isfinite(measured):
        return False
    if closing:
        return measured <= float(target_m) + float(tolerance_m)
    return measured >= float(target_m) - float(tolerance_m)
