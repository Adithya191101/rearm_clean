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
Tests for the REBOT enum patch and ``ReBotDriverConfig``.

A real ``launch.LaunchContext`` is constructed here -- ``launch`` is importable on
this host and ``perform_substitution`` works without a running ROS graph -- so
these exercise the same code path bringup does, not a mock of it.


The frame overrides are the highest-value assertion in this file.
``DriverConfig.__init__`` derives ``gripper_frame`` as ``f'{prefix}gripper_frame'``
and ``grasp_frame`` as ``f'{prefix}grasp_frame'``. NEITHER FRAME EXISTS in the
reBot URDF, which publishes ``gripper_link`` and ``gripper_tcp``. Without the
overrides the failure is a TF lookup timeout at grasp time, far from its cause,
and it does not appear at launch.
"""

from isaac_ros_manipulation_rebot_driver_utils.config import (
    REBOT_GRASP_FRAME, REBOT_GRIPPER_FRAME, ReBotDriverConfig,
)
from isaac_ros_manipulation_ros_python_utils.config import (
    _GRIPPER_COLLISION_LINKS, DriverConfig,
)
from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    compute_frame_prefix, compute_joint_names, get_robot_type,
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    GripperType, RobotType,
)

from launch import LaunchContext

import pytest

# The names DriverConfig would derive. If these ever become real links in the
# reBot description, the override becomes unnecessary -- but until then, a config
# reporting these names is reporting frames that are not in TF.
BASE_DERIVED_GRIPPER_FRAME = 'gripper_frame'
BASE_DERIVED_GRASP_FRAME = 'grasp_frame'


REBOT_ARGS = {
    'robot_type': 'REBOT',
    'gripper_type': 'rebot_parallel',
    'use_sim_time': 'true',
    'headless': 'false',
    'log_level': 'error',
    'workflow_type': 'POSE_TO_POSE',
    'controller_spawner_timeout': '10',
    'urdf_path': '/tmp/rebot_sim.urdf.xacro',
    'srdf_path': '/tmp/rebot.srdf.xacro',
    'joint_limits_file_path': '/tmp/joint_limits.yaml',
    'kinematics_file_path': '/tmp/kinematics_sim.yaml',
    'moveit_controllers_file_path': '/tmp/moveit_sim_controllers.yaml',
    'ros2_controllers_file_path': '/tmp/ros2_control_controllers_sim.yaml',
}


def make_context(**overrides) -> LaunchContext:
    """Build a LaunchContext with the reBot launch args already resolved."""
    args = dict(REBOT_ARGS)
    args.update(overrides)
    context = LaunchContext()
    for key, value in args.items():
        context.launch_configurations[key] = value
    return context

# ---------------------------------------------------------------------------
# Enum patch
# ---------------------------------------------------------------------------


def test_robot_type_enum_has_rebot():
    assert 'REBOT' in RobotType.names()
    assert str(RobotType.REBOT) == 'REBOT'


def test_get_robot_type_resolves_rebot():
    assert get_robot_type(make_context()) is RobotType.REBOT


def test_gripper_type_enum_has_rebot_parallel():
    assert GripperType.REBOT_PARALLEL.value == 'rebot_parallel'


def test_gripper_collision_links_entry_exists():
    links = _GRIPPER_COLLISION_LINKS[GripperType.REBOT_PARALLEL]
    assert links == ['gripper_link', 'gripper_left', 'gripper_right']


def test_gripper_collision_links_exclude_the_tcp():
    """gripper_tcp is a geometry-free frame; it has no collisions to disable."""
    links = _GRIPPER_COLLISION_LINKS[GripperType.REBOT_PARALLEL]
    assert 'gripper_tcp' not in links


def test_gripper_collision_links_are_not_the_robotiq_links():
    rebot = _GRIPPER_COLLISION_LINKS[GripperType.REBOT_PARALLEL]
    for other in (GripperType.ROBOTIQ_2F_140, GripperType.ROBOTIQ_2F_85,
                  GripperType.GRAV):
        assert rebot != _GRIPPER_COLLISION_LINKS[other]

# ---------------------------------------------------------------------------
# Frame prefix and joint names
# ---------------------------------------------------------------------------


def test_frame_prefix_defaults_to_empty_when_prefix_undeclared():
    """The reBot xacro defaults prefix to ''; an undeclared arg must not raise."""
    assert compute_frame_prefix(make_context()) == ''


def test_frame_prefix_uses_the_prefix_arg_not_tf_prefix_or_robot_sn():
    context = make_context(prefix='left_', tf_prefix='WRONG_', robot_sn='SN123')
    assert compute_frame_prefix(context) == 'left_'


def test_frame_prefix_does_not_raise_on_empty_robot_sn():
    """
    Regression guard against routing reBot through the Flexiv helper.

    ``_flexiv_frame_prefix`` raises ValueError when robot_sn is empty. The reBot
    has no serial-numbered frames, so an empty robot_sn is normal.
    """
    assert compute_frame_prefix(make_context(robot_sn='')) == ''


def test_joint_names_are_the_six_arm_joints():
    assert compute_joint_names(make_context()) == [
        'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']


def test_joint_names_have_no_fabricated_joint7():
    """Flexiv uses range(1, 8); copying that invents a joint not in the URDF."""
    assert 'joint7' not in compute_joint_names(make_context())


def test_joint_names_exclude_the_jaw_joints():
    names = compute_joint_names(make_context())
    assert 'gripper_joint1' not in names
    assert 'gripper_joint2' not in names


def test_joint_names_are_prefixed():
    assert compute_joint_names(make_context(prefix='left_'))[0] == 'left_joint1'

# ---------------------------------------------------------------------------
# ReBotDriverConfig
# ---------------------------------------------------------------------------


def test_base_class_would_derive_nonexistent_frames():
    """
    Establishes that the override is load-bearing, not decorative.

    If this ever fails because the base class stopped deriving these names, the
    override tests below are testing nothing and should be revisited.
    """
    base = DriverConfig(make_context())
    assert base.gripper_frame == BASE_DERIVED_GRIPPER_FRAME
    assert base.grasp_frame == BASE_DERIVED_GRASP_FRAME


def test_gripper_frame_is_overridden_to_a_real_link():
    config = ReBotDriverConfig(make_context())
    assert config.gripper_frame == REBOT_GRIPPER_FRAME == 'gripper_link'
    assert config.gripper_frame != BASE_DERIVED_GRIPPER_FRAME


def test_grasp_frame_is_overridden_to_the_tcp():
    config = ReBotDriverConfig(make_context())
    assert config.grasp_frame == REBOT_GRASP_FRAME == 'gripper_tcp'
    assert config.grasp_frame != BASE_DERIVED_GRASP_FRAME


def test_frame_overrides_respect_the_prefix():
    config = ReBotDriverConfig(make_context(prefix='left_'))
    assert config.gripper_frame == 'left_gripper_link'
    assert config.grasp_frame == 'left_gripper_tcp'
    assert config.base_frame == 'left_base_link'


def test_config_carries_rebot_gripper_positions_not_robotiq():
    config = ReBotDriverConfig(make_context())
    assert config.gripper_open_position == pytest.approx(0.0715)
    assert config.gripper_close_position == pytest.approx(0.0)
    assert (config.gripper_open_position,
            config.gripper_close_position) != (0.0, 0.65)


def test_config_rejects_a_non_rebot_robot_type():
    """
    Identity guard, matching the UR and Flexiv configs.

    Without it, a mismatched robot_type silently yields a config whose joint
    names and frames belong to a different robot.
    """
    with pytest.raises(ValueError) as excinfo:
        ReBotDriverConfig(make_context(robot_type='FLEXIV', robot_sn='SN1'))
    assert 'REBOT' in str(excinfo.value)


def test_sim_remaps_joint_states_to_the_parsed_topic():
    config = ReBotDriverConfig(make_context())
    assert config.remapped_joint_states['/joint_states'] == (
        '/isaac_parsed_joint_states')
    assert config.remapped_joint_states[
        '/controller_manager/robot_description'] == '/robot_description'


def test_insertion_frame_is_left_at_the_base_derived_name():
    """
    Documented deliberate omission, asserted so it stays deliberate.

    The reBot has no insertion frame and no measured offset to author one from.
    Only GEAR_ASSEMBLY reads it, and that workflow is unsupported here.
    """
    config = ReBotDriverConfig(make_context())
    assert config.insertion_frame == 'insertion_frame'
