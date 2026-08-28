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
Pure-Python joint-state filtering for the reBot B601-DM Isaac Sim bridge.

Deliberately free of ``rclpy`` / ``sensor_msgs`` imports so the transform can be
unit tested with plain ``python3 -m pytest`` on a host with no ROS install.
``src/isaac_sim_joint_parser_node.py`` is the (untestable-without-ROS) I/O shell
around :func:`filter_arm_joint_state`.

Why this exists
---------------
Isaac Sim publishes every articulated joint on ``/isaac_joint_states``: the six
arm joints plus BOTH jaw joints. ``gripper_joint2`` is a ``<mimic>`` of
``gripper_joint1`` in the URDF, so ``robot_state_publisher`` and MoveIt derive it
themselves and reject it as an input -- forwarding it makes the jaw state doubly
specified. This module reduces the incoming message to the arm joints in a FIXED
order and rejects anything it does not recognise.

Rejection, not best-effort
--------------------------
An unexpected name set raises instead of being silently trimmed or reordered. A
best-effort filter that drops what it does not know produces a message that is
structurally valid but semantically wrong (e.g. a renamed or re-prefixed
articulation yields an empty or short message, and MoveIt then reports "no
joint state received" from a node that is happily publishing). Raising surfaces
the mismatch at the boundary where the names actually disagree.

Complete MoveIt state
---------------------
``srdf/rebot.srdf.xacro`` declares ``gripper_joint1`` as an active joint, so
MoveIt's CurrentStateMonitor requires it even though ros2_control commands only
the six arm joints. The driving jaw is therefore included by default.
``gripper_joint2`` is never republished: it is the ``<mimic>`` follower and
consumers derive it from ``gripper_joint1``.
"""

from typing import NamedTuple, Sequence, Tuple

# Output order. This is the order MoveIt expects for the ``rebot_arm`` planning
# group and is identical to what
# ``isaac_ros_manipulation_ros_python_utils.launch_utils.compute_joint_names``
# returns for ``RobotType.REBOT``. Isaac Sim's own order comes from the USD
# articulation and is NOT guaranteed to match, which is why the output order is
# pinned here rather than inherited from the incoming message.
ARM_JOINT_NAMES: Tuple[str, ...] = (
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6',
)

# The jaw joints, present in the Isaac Sim articulation. gripper_joint1 is the
# driven joint; gripper_joint2 mimics it (multiplier +1.0) and is never
# republished, because a consumer that derives it from gripper_joint1 would then
# have to reconcile its own value against ours.
DRIVING_JAW_JOINT_NAME: str = 'gripper_joint1'
MIMIC_JAW_JOINT_NAME: str = 'gripper_joint2'
JAW_JOINT_NAMES: Tuple[str, ...] = (
    DRIVING_JAW_JOINT_NAME, MIMIC_JAW_JOINT_NAME)

# Every joint the reBot Isaac Sim articulation is allowed to contain. A name
# outside this set means the USD, the URDF or the prefix has changed and the
# filter's assumptions no longer hold.
KNOWN_JOINT_NAMES: Tuple[str, ...] = ARM_JOINT_NAMES + JAW_JOINT_NAMES


class JointStateFilterError(ValueError):
    """Base class for every rejection raised by :func:`filter_arm_joint_state`."""


class UnexpectedJointNamesError(JointStateFilterError):
    """Raised when the incoming name set is not the expected articulation."""


class JointStateLengthError(JointStateFilterError):
    """Raised when the position/velocity/effort arrays do not match ``name``."""


class FilteredJointState(NamedTuple):
    """MoveIt joint state, with arm joints first and optional driving jaw."""

    name: Tuple[str, ...]
    position: Tuple[float, ...]
    velocity: Tuple[float, ...]
    effort: Tuple[float, ...]


def prefixed(names: Sequence[str], prefix: str = '') -> Tuple[str, ...]:
    """
    Apply a TF/joint name prefix to ``names``.

    The reBot description macros take a ``prefix`` arg defaulting to ``''``, so
    the common case is a no-op. Kept explicit so the node can be brought up
    against a prefixed articulation without editing this module.
    """
    if not prefix:
        return tuple(names)
    return tuple(f'{prefix}{name}' for name in names)


def _check_array_length(label: str, values: Sequence[float], expected: int) -> None:
    """Reject a non-empty array whose length does not match ``name``."""
    if len(values) not in (0, expected):
        raise JointStateLengthError(
            f'JointState.{label} has {len(values)} entries but JointState.name '
            f'has {expected}. Per-index pairing is impossible, so the message '
            f'is rejected rather than paired against the wrong joints.')


def filter_arm_joint_state(
    name: Sequence[str],
    position: Sequence[float],
    velocity: Sequence[float] = (),
    effort: Sequence[float] = (),
    prefix: str = '',
    include_driving_jaw: bool = True,
) -> FilteredJointState:
    """
    Reduce an Isaac Sim joint state to the arm joints, in MoveIt's order.

    Args
    ----
        name: Joint names as published by Isaac Sim, in the USD articulation's
            order.
        position: Positions, index-aligned with ``name``. Must be the same
            length as ``name``.
        velocity: Velocities, index-aligned with ``name``. May be empty, in
            which case zeros are emitted.
        effort: Efforts, index-aligned with ``name``. May be empty, in which
            case zeros are emitted.
        prefix: Joint name prefix in use (``''`` for the default reBot
            description).
        include_driving_jaw: Append ``gripper_joint1`` to the output. On by
            default so MoveIt's active-joint state is complete. The mimic
            follower ``gripper_joint2`` is never emitted.

    Returns
    -------
        FilteredJointState: The six arm joints in :data:`ARM_JOINT_NAMES` order
        followed by the driving jaw when requested (prefixed if ``prefix`` is
        set), with velocity and effort gathered BY INDEX rather than sliced off
        the front of the incoming arrays.
        Isaac Sim's order is set by the USD articulation and need not match the
        output order, so a leading slice can pair a velocity with the wrong
        joint.

    Raises
    ------
        UnexpectedJointNamesError: If ``name`` contains a duplicate, contains a
            joint outside :data:`KNOWN_JOINT_NAMES`, is missing any arm joint,
            or is missing ``gripper_joint1`` while ``include_driving_jaw`` is
            set.
        JointStateLengthError: If ``position`` is not the same length as
            ``name``, or if a non-empty ``velocity``/``effort`` is not.

    """
    names = list(name)
    wanted = ARM_JOINT_NAMES
    if include_driving_jaw:
        wanted = wanted + (DRIVING_JAW_JOINT_NAME,)
    expected_arm = prefixed(wanted, prefix)
    expected_known = set(prefixed(KNOWN_JOINT_NAMES, prefix))

    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise UnexpectedJointNamesError(
            f'Duplicate joint name(s) {duplicates} in JointState.name '
            f'({names}). Which index owns the joint is ambiguous, so the '
            f'message is rejected.')

    unknown = [n for n in names if n not in expected_known]
    if unknown:
        raise UnexpectedJointNamesError(
            f'Unrecognised joint name(s) {unknown} in JointState.name '
            f'({names}). Expected a subset of {sorted(expected_known)}. The '
            f'articulation, the URDF or the joint prefix has changed; '
            f'rejecting rather than silently dropping the joint.')

    missing = [n for n in expected_arm if n not in names]
    if missing:
        raise UnexpectedJointNamesError(
            f'Missing required joint(s) {missing} from JointState.name '
            f'({names}). A partial state cannot be republished: MoveIt would '
            f'plan from a state it never received.')

    # position must be complete: every arm joint needs a value.
    if len(position) != len(names):
        raise JointStateLengthError(
            f'JointState.position has {len(position)} entries but '
            f'JointState.name has {len(names)}.')
    _check_array_length('velocity', velocity, len(names))
    _check_array_length('effort', effort, len(names))

    index_of = {n: i for i, n in enumerate(names)}

    def gather(values: Sequence[float]) -> Tuple[float, ...]:
        if not values:
            return tuple(0.0 for _ in expected_arm)
        return tuple(float(values[index_of[n]]) for n in expected_arm)

    return FilteredJointState(
        name=tuple(expected_arm),
        position=tuple(float(position[index_of[n]]) for n in expected_arm),
        velocity=gather(velocity),
        effort=gather(effort),
    )
