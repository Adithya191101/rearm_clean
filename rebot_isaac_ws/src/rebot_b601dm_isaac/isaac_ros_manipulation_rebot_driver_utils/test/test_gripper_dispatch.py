# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""
Prove a reBot gripper never receives Robotiq values.

The defect this guards: ``compute_gripper_action_name`` /
``compute_gripper_positions`` / ``compute_gripper_settle_time`` used to end with a
bare ``return <robotiq value>``, so ANY gripper without an explicit branch
inherited Robotiq's numbers. For the reBot parallel jaw that is wrong in three
independent ways at once:

* polarity is inverted (Robotiq open=0.0/close=0.65; reBot open=0.0715/close=0.0),
  so "open" would command fully closed;
* units differ (radians of knuckle rotation vs metres of prismatic travel);
* magnitude overtravels by ~9x (0.65 against a 0.0715 m jaw limit).


Nothing raises in that scenario, which is exactly why it needs a test.


Runs on a plain host: no rclpy, no launch context, no container.
"""

from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    _ROBOTIQ_GRIPPER_TYPES,
    compute_gripper_action_name,
    compute_gripper_positions,
    compute_gripper_settle_time,
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import GripperType

import pytest

REBOT = GripperType.REBOT_PARALLEL.value

# Values the Robotiq branches return, restated here as literals on purpose. If a
# future edit changes the Robotiq numbers, this test still asserts the reBot does
# not receive the OLD Robotiq numbers, and the dedicated Robotiq test below
# catches the change itself.
ROBOTIQ_POSITIONS = (0.0, 0.65)
ROBOTIQ_ACTION_NAME = '/robotiq_gripper_controller/gripper_cmd'

# From the URDF/SRDF: jaw travel is 0.0 (closed) to 0.0715 (open) METRES per jaw.
REBOT_OPEN = 0.0715
REBOT_CLOSED = 0.0


DISPATCHERS = (
    compute_gripper_action_name,
    compute_gripper_positions,
    compute_gripper_settle_time,
)


def test_rebot_positions_are_not_robotiq_positions():
    """The headline assertion: no Robotiq leakage into a reBot gripper."""
    assert compute_gripper_positions(REBOT) != ROBOTIQ_POSITIONS


def test_rebot_positions_are_the_srdf_values():
    open_position, close_position = compute_gripper_positions(REBOT)
    assert open_position == pytest.approx(REBOT_OPEN)
    assert close_position == pytest.approx(REBOT_CLOSED)


def test_rebot_open_is_greater_than_closed():
    """Polarity, stated independently of the literals above."""
    open_position, close_position = compute_gripper_positions(REBOT)
    assert open_position > close_position, (
        'reBot jaws are prismatic with 0.0 = closed. An open value below the '
        'closed value is the Robotiq polarity and would command a close on open.')


def test_rebot_open_is_within_the_jaw_limit():
    """Robotiq's 0.65 would overtravel a 0.0715 m jaw by ~9x."""
    open_position, close_position = compute_gripper_positions(REBOT)
    for value in (open_position, close_position):
        assert 0.0 <= value <= REBOT_OPEN


def test_rebot_action_name_is_not_the_robotiq_action():
    assert compute_gripper_action_name(REBOT) != ROBOTIQ_ACTION_NAME


def test_rebot_action_name_is_rebot_specific():
    assert 'rebot' in compute_gripper_action_name(REBOT)


def test_rebot_settle_time_is_a_float():
    assert isinstance(compute_gripper_settle_time(REBOT), float)


@pytest.mark.parametrize('dispatcher', DISPATCHERS)
def test_unknown_gripper_raises_instead_of_falling_through(dispatcher):
    """
    An unnamed future gripper must not inherit Robotiq values either.

    This is the belt-and-braces half of the fix: even without an explicit
    REBOT_PARALLEL branch, the tail of each dispatcher now refuses.
    """
    with pytest.raises(NotImplementedError) as excinfo:
        dispatcher('some_gripper_that_does_not_exist')
    assert 'Robotiq' in str(excinfo.value)


@pytest.mark.parametrize('dispatcher', DISPATCHERS)
def test_grav_still_dispatches(dispatcher):
    """Regression guard: the guard must not break the pre-existing Grav branch."""
    dispatcher(GripperType.GRAV.value)


@pytest.mark.parametrize('robotiq', _ROBOTIQ_GRIPPER_TYPES)
def test_robotiq_grippers_still_get_robotiq_values(robotiq):
    """The guard must not have broken the grippers that DO want these values."""
    assert compute_gripper_positions(robotiq) == ROBOTIQ_POSITIONS
    assert compute_gripper_action_name(robotiq) == ROBOTIQ_ACTION_NAME


def test_rebot_parallel_is_not_in_the_robotiq_allowlist():
    """The allowlist is what makes the fall-through safe; keep reBot out of it."""
    assert REBOT not in _ROBOTIQ_GRIPPER_TYPES


def test_gripper_type_lookup_resolves_rebot():
    assert GripperType.get_gripper_type(REBOT) is GripperType.REBOT_PARALLEL


def test_every_gripper_type_has_an_explicit_dispatch():
    """
    No enum member may rely on the fall-through.

    Catches the case where a gripper is added to the enum (so it validates as a
    legal ``gripper_type`` launch arg) but no dispatcher branch is written for it.
    """
    for gripper in GripperType:
        for dispatcher in DISPATCHERS:
            dispatcher(gripper.value)
