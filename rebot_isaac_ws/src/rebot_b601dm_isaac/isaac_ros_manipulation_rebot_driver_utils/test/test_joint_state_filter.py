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
Tests for the Isaac Sim -> MoveIt joint state filter.

Host-runnable: ``joint_state_filter`` imports neither ``rclpy`` nor
``sensor_msgs``, so the whole transform is exercised without a ROS install. The
matching node (``src/isaac_sim_joint_parser_node.py``) is an I/O shell whose only
untested part is publish/subscribe wiring, covered by the container test in
``test_rebot_startup.py``.


Two properties matter most and neither is enforced by the type system:

*  the OUTPUT ORDER is pinned to MoveIt's expectation (six arm joints followed
   by the driving jaw), not inherited from the incoming message.
*  an unexpected name set is REJECTED, not trimmed. A best-effort filter yields a
   structurally valid but short message and MoveIt then reports "no joint state
   received" from a node that is publishing happily.
"""

from isaac_ros_manipulation_rebot_driver_utils.joint_state_filter import (
    ARM_JOINT_NAMES,
    DRIVING_JAW_JOINT_NAME,
    filter_arm_joint_state,
    JointStateLengthError,
    MIMIC_JAW_JOINT_NAME,
    prefixed,
    UnexpectedJointNamesError,
)

import pytest

# Isaac Sim's articulation order. Deliberately NOT the MoveIt order: joint6 comes
# before joint5, and the jaws are interleaved rather than appended, so a filter
# that slices or preserves incoming order gets a different answer than one that
# gathers by name.
ISAAC_ORDER = (
    'joint1', 'joint2', 'gripper_joint1', 'joint3',
    'joint6', 'joint4', 'gripper_joint2', 'joint5',
)

ISAAC_POSITIONS = (0.11, 0.22, 0.03, 0.33, 0.66, 0.44, 0.03, 0.55)
ISAAC_VELOCITIES = (1.1, 1.2, 1.9, 1.3, 1.6, 1.4, 1.9, 1.5)
ISAAC_EFFORTS = (2.1, 2.2, 2.9, 2.3, 2.6, 2.4, 2.9, 2.5)


def _isaac_message(**overrides):
    kwargs = {
        'name': ISAAC_ORDER,
        'position': ISAAC_POSITIONS,
        'velocity': ISAAC_VELOCITIES,
        'effort': ISAAC_EFFORTS,
    }
    kwargs.update(overrides)
    return kwargs


def test_eight_isaac_names_yield_arm_joints_plus_the_driving_jaw():
    """The headline assertion from the brief."""
    result = filter_arm_joint_state(**_isaac_message())
    assert len(result.name) == 7
    assert result.name == ARM_JOINT_NAMES + (DRIVING_JAW_JOINT_NAME,)


def test_output_order_is_moveit_order_not_incoming_order():
    result = filter_arm_joint_state(**_isaac_message())
    assert result.name == (
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
        'gripper_joint1')
    # Prove the fixture would have caught order preservation: the arm joints as
    # they appear in the incoming message are NOT in MoveIt order.
    incoming_arm_order = tuple(
        n for n in ISAAC_ORDER if n in ARM_JOINT_NAMES)
    assert incoming_arm_order != result.name


def test_positions_follow_their_joint_not_their_index():
    result = filter_arm_joint_state(**_isaac_message())
    by_name = dict(zip(result.name, result.position))
    assert by_name == {
        'joint1': pytest.approx(0.11),
        'joint2': pytest.approx(0.22),
        'joint3': pytest.approx(0.33),
        'joint4': pytest.approx(0.44),
        'joint5': pytest.approx(0.55),
        'joint6': pytest.approx(0.66),
        'gripper_joint1': pytest.approx(0.03),
    }


def test_velocity_and_effort_are_gathered_by_index_not_sliced():
    """
    A leading slice would pair joint5's velocity with joint3.

    The fixture values encode their joint (velocity 1.<n> for joint<n>), so a
    slice-off-the-front implementation produces detectably wrong pairings.
    """
    result = filter_arm_joint_state(**_isaac_message())
    assert result.velocity == pytest.approx(
        (1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.9))
    assert result.effort == pytest.approx(
        (2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.9))
    # The naive first-six slice differs, so the assertion above is load-bearing.
    assert result.velocity != pytest.approx(ISAAC_VELOCITIES[:6])


def test_default_includes_driving_jaw_but_excludes_mimic():
    result = filter_arm_joint_state(**_isaac_message())
    assert DRIVING_JAW_JOINT_NAME in result.name
    assert MIMIC_JAW_JOINT_NAME not in result.name


def test_explicit_arm_only_mode_excludes_both_jaws():
    result = filter_arm_joint_state(
        **_isaac_message(), include_driving_jaw=False)
    assert result.name == ARM_JOINT_NAMES
    assert DRIVING_JAW_JOINT_NAME not in result.name
    assert MIMIC_JAW_JOINT_NAME not in result.name


def test_unexpected_name_is_rejected_not_silently_dropped():
    """
    The brief's second explicit requirement.

    A renamed articulation must fail loudly, not yield a short message.
    """
    names = ISAAC_ORDER[:-1] + ('shoulder_pan_joint',)
    with pytest.raises(UnexpectedJointNamesError) as excinfo:
        filter_arm_joint_state(name=names, position=ISAAC_POSITIONS)
    assert 'shoulder_pan_joint' in str(excinfo.value)
    # The SUBSTITUTION above also removes joint5, so the missing-joint check can
    # satisfy this test on its own; see the additive test below, which isolates
    # the unknown-name check. Keep both: this one is the realistic shape of a
    # renamed articulation, the other is the one that pins the check.
    assert 'Unrecognised' in str(excinfo.value)


def test_an_ADDED_unknown_name_is_rejected():
    """
    Negative control: isolates the unknown-name check from every other check.

    Appending a name (rather than substituting one) leaves all six arm joints
    present, introduces no duplicate and keeps the arrays index-aligned, so this
    message is rejectable ONLY by the unknown-name check. The substitution-shaped
    test above still passes when that check is deleted, because dropping joint5
    trips the missing-joint check instead -- a mutation run caught exactly that.
    """
    names = ISAAC_ORDER + ('shoulder_pan_joint',)
    positions = ISAAC_POSITIONS + (9.9,)
    with pytest.raises(UnexpectedJointNamesError) as excinfo:
        filter_arm_joint_state(name=names, position=positions)
    assert 'shoulder_pan_joint' in str(excinfo.value)
    assert 'Unrecognised' in str(excinfo.value)


def test_prefixed_articulation_is_rejected_when_no_prefix_configured():
    names = tuple(f'rebot_{n}' for n in ISAAC_ORDER)
    with pytest.raises(UnexpectedJointNamesError):
        filter_arm_joint_state(name=names, position=ISAAC_POSITIONS)


def test_prefixed_articulation_is_accepted_with_a_matching_prefix():
    names = tuple(f'rebot_{n}' for n in ISAAC_ORDER)
    result = filter_arm_joint_state(
        name=names, position=ISAAC_POSITIONS, prefix='rebot_')
    assert result.name == prefixed(
        ARM_JOINT_NAMES + (DRIVING_JAW_JOINT_NAME,), 'rebot_')


def test_missing_arm_joint_is_rejected():
    names = tuple(n for n in ISAAC_ORDER if n != 'joint4')
    positions = ISAAC_POSITIONS[:len(names)]
    with pytest.raises(UnexpectedJointNamesError) as excinfo:
        filter_arm_joint_state(name=names, position=positions)
    assert 'joint4' in str(excinfo.value)


def test_missing_driving_jaw_is_rejected_when_requested():
    names = tuple(n for n in ISAAC_ORDER if n != DRIVING_JAW_JOINT_NAME)
    positions = ISAAC_POSITIONS[:len(names)]
    with pytest.raises(UnexpectedJointNamesError):
        filter_arm_joint_state(
            name=names, position=positions, include_driving_jaw=True)


def test_duplicate_name_is_rejected():
    names = ISAAC_ORDER[:-1] + ('joint1',)
    with pytest.raises(UnexpectedJointNamesError) as excinfo:
        filter_arm_joint_state(name=names, position=ISAAC_POSITIONS)
    assert 'joint1' in str(excinfo.value)


def test_an_ADDED_duplicate_name_is_rejected():
    """
    Negative control: isolates the duplicate check from every other check.

    As with the unknown-name pair above, the substitution-shaped test also drops
    joint5 and so passes on the missing-joint check alone. Appending the duplicate
    keeps every arm joint present and every name known, leaving the duplicate
    check as the only thing that can reject the message.
    """
    names = ISAAC_ORDER + ('joint1',)
    positions = ISAAC_POSITIONS + (9.9,)
    with pytest.raises(UnexpectedJointNamesError) as excinfo:
        filter_arm_joint_state(name=names, position=positions)
    assert 'Duplicate' in str(excinfo.value)
    assert 'joint1' in str(excinfo.value)


def test_short_position_array_is_rejected():
    with pytest.raises(JointStateLengthError):
        filter_arm_joint_state(
            name=ISAAC_ORDER, position=ISAAC_POSITIONS[:-1])


def test_short_velocity_array_is_rejected():
    with pytest.raises(JointStateLengthError):
        filter_arm_joint_state(
            name=ISAAC_ORDER,
            position=ISAAC_POSITIONS,
            velocity=ISAAC_VELOCITIES[:-1])


def test_empty_velocity_and_effort_become_zeros():
    """Isaac Sim may publish an empty effort array; that is not an error."""
    result = filter_arm_joint_state(
        name=ISAAC_ORDER, position=ISAAC_POSITIONS)
    assert result.velocity == (0.0,) * 7
    assert result.effort == (0.0,) * 7


def test_arm_only_message_is_accepted_when_driving_jaw_is_disabled():
    result = filter_arm_joint_state(
        name=ARM_JOINT_NAMES,
        position=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
        include_driving_jaw=False)
    assert result.position == pytest.approx((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


def test_arm_only_message_is_rejected_by_default():
    with pytest.raises(UnexpectedJointNamesError, match=DRIVING_JAW_JOINT_NAME):
        filter_arm_joint_state(
            name=ARM_JOINT_NAMES,
            position=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0))


def test_arm_joint_names_match_the_shared_launch_util():
    """
    The parser and MoveIt must agree on the joint list.

    ``compute_joint_names`` feeds MoveIt/cuMotion; this filter feeds
    robot_state_publisher. A divergence means TF and planning disagree about
    which joints exist, and neither side errors.
    """
    from isaac_ros_manipulation_ros_python_utils.launch_utils import (
        _UR_ARM_JOINTS,
    )
    assert ARM_JOINT_NAMES == tuple(f'joint{i}' for i in range(1, 7))
    # Sanity: we are not accidentally comparing against the UR list.
    assert ARM_JOINT_NAMES != _UR_ARM_JOINTS
