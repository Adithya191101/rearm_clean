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
Host-runnable structural checks on the sim ros2_control block and controllers.

``xacro`` is not installed outside the container, so the macro cannot be
expanded here. It does not need to be: every property asserted below (the
hardware plugin, the two Isaac Sim topics, the joint list, the interface names)
is a literal in the file, and ``xacro`` files are well-formed XML, so
``xml.etree`` reads them directly. Only the ``${...}`` initial_value expressions
are opaque, and none of these assertions depend on their values.

Why this file exists
--------------------
The topic names and the six joint names were previously asserted ONLY by
``test_rebot_startup.py``, which is container-only and has never run. A mutation
run confirmed that renaming ``/isaac_joint_commands`` in the xacro broke nothing
that executes on this host. These are cheap literals to check and the failure
mode of getting them wrong is a silent one -- ``TopicBasedSystem`` publishes to
whatever topic it is told and Isaac Sim listens on the one it was told, so a
mismatch produces an arm that simply never moves.
"""

import os
import xml.etree.ElementTree as ElementTree

from isaac_ros_manipulation_rebot_driver_utils.joint_state_filter import (
    ARM_JOINT_NAMES, DRIVING_JAW_JOINT_NAME, MIMIC_JAW_JOINT_NAME,
)

import pytest

import yaml

_XACRO_NS = {'xacro': 'http://www.ros.org/wiki/xacro'}

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_DESCRIPTION_PKG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(_TEST_DIR)),
    'isaac_ros_manipulation_rebot_robot_description')
_ROS2_CONTROL_XACRO = os.path.join(
    _DESCRIPTION_PKG_DIR, 'urdf', 'rebot_sim.ros2_control.xacro')
_CONTROLLERS_YAML = os.path.join(
    _DESCRIPTION_PKG_DIR, 'config', 'ros2_control_controllers_sim.yaml')
_MOVEIT_CONTROLLERS_YAML = os.path.join(
    _DESCRIPTION_PKG_DIR, 'config', 'moveit_sim_controllers.yaml')

# Isaac Sim's own topic names. TopicBasedSystem is a bridge, so both ends must
# agree with the Isaac Sim action graph; these match the UR sim reference block.
JOINT_COMMANDS_TOPIC = '/isaac_joint_commands'
JOINT_STATES_TOPIC = '/isaac_joint_states'

# Shared with rebot_driver_utils.ARM_CONTROLLER_NAME and hardcoded in the
# upstream shared test harness.
ARM_CONTROLLER_NAME = 'scaled_joint_trajectory_controller'


@pytest.fixture(scope='module')
def ros2_control_block():
    root = ElementTree.parse(_ROS2_CONTROL_XACRO).getroot()
    macro = root.find('xacro:macro', _XACRO_NS)
    assert macro is not None, 'No xacro:macro in the ros2_control xacro'
    block = macro.find('ros2_control')
    assert block is not None, 'No <ros2_control> inside the macro'
    return block


@pytest.fixture(scope='module')
def controllers():
    with open(_CONTROLLERS_YAML) as handle:
        return yaml.safe_load(handle)


@pytest.fixture(scope='module')
def moveit_controllers():
    with open(_MOVEIT_CONTROLLERS_YAML) as handle:
        return yaml.safe_load(handle)


# ---------------------------------------------------------------------------
# The TopicBasedSystem bridge
# ---------------------------------------------------------------------------

def test_hardware_plugin_is_topic_based(ros2_control_block):
    """Not a real hardware interface: sim is bridged over topics."""
    hardware = ros2_control_block.find('hardware')
    assert hardware.findtext('plugin') == (
        'topic_based_ros2_control/TopicBasedSystem')


def test_bridge_topics_are_the_isaac_sim_topics(ros2_control_block):
    """
    Both topic params must name the Isaac Sim topics, spelled exactly.

    Getting either wrong produces no error anywhere: ros2_control publishes into
    the void and the arm never moves. The Flexiv reference uses its own
    /rizon_arm_command because it interposes a gripper driver node that merges
    arm and jaw commands; there is no such node for the reBot, so commands go
    straight to Isaac Sim on its own topic, matching the UR sim block.
    """
    hardware = ros2_control_block.find('hardware')
    params = {
        param.get('name'): param.text
        for param in hardware.findall('param')
    }
    assert params['joint_commands_topic'] == JOINT_COMMANDS_TOPIC
    assert params['joint_states_topic'] == JOINT_STATES_TOPIC


def test_the_two_bridge_topics_are_distinct(ros2_control_block):
    """A copy-paste that points both at one topic makes the plugin echo itself."""
    hardware = ros2_control_block.find('hardware')
    params = {p.get('name'): p.text for p in hardware.findall('param')}
    assert params['joint_commands_topic'] != params['joint_states_topic']


# ---------------------------------------------------------------------------
# Joint list
# ---------------------------------------------------------------------------

def test_ros2_control_joints_are_the_six_arm_joints(ros2_control_block):
    names = tuple(j.get('name') for j in ros2_control_block.findall('joint'))
    assert names == ARM_JOINT_NAMES


def test_ros2_control_has_no_seventh_joint(ros2_control_block):
    """range(1, 8) is the 7-DoF Flexiv value; joint7 is not in the reBot URDF."""
    names = {j.get('name') for j in ros2_control_block.findall('joint')}
    assert 'joint7' not in names


def test_ros2_control_excludes_the_jaw_joints(ros2_control_block):
    """
    The jaws are not controller_manager joints.

    gripper_joint2 <mimic>s gripper_joint1 so it has no independent command
    interface, and the jaws are driven through a gripper action rather than the
    trajectory controller. Both reference vendors' blocks carry arm joints only.
    """
    names = {j.get('name') for j in ros2_control_block.findall('joint')}
    assert DRIVING_JAW_JOINT_NAME not in names
    assert MIMIC_JAW_JOINT_NAME not in names


# ---------------------------------------------------------------------------
# Interfaces
# ---------------------------------------------------------------------------

def test_every_joint_exposes_the_same_interfaces(ros2_control_block):
    for joint in ros2_control_block.findall('joint'):
        command = [c.get('name') for c in joint.findall('command_interface')]
        state = [s.get('name') for s in joint.findall('state_interface')]
        assert command == ['position', 'velocity'], joint.get('name')
        assert state == ['position', 'velocity', 'effort'], joint.get('name')


def test_position_state_interfaces_carry_an_initial_value(ros2_control_block):
    """
    Without it the arm starts at 0.0, which is inside the table for this model.

    initial_positions.yaml is loaded by the wrapping urdf.xacro and interpolated
    into these attributes.
    """
    for joint in ros2_control_block.findall('joint'):
        position_state = joint.find("state_interface[@name='position']")
        initial = position_state.find("param[@name='initial_value']")
        assert initial is not None, joint.get('name')
        assert initial.text.startswith('${initial_positions['), (
            joint.get('name'))


def test_command_interface_limits_are_present_and_ordered(ros2_control_block):
    """Require min < max on every command interface, with both present."""
    for joint in ros2_control_block.findall('joint'):
        for interface in joint.findall('command_interface'):
            minimum = interface.find("param[@name='min']")
            maximum = interface.find("param[@name='max']")
            assert minimum is not None and maximum is not None, (
                f"{joint.get('name')}/{interface.get('name')}")
            assert float(minimum.text) < float(maximum.text), (
                f"{joint.get('name')}/{interface.get('name')}")


# ---------------------------------------------------------------------------
# Controller configs must agree with the ros2_control block
# ---------------------------------------------------------------------------

def test_arm_controller_joints_match_the_ros2_control_block(
        ros2_control_block, controllers):
    """
    A controller claiming a joint the hardware does not expose fails to activate.

    That failure surfaces as a spawner timeout, which reads as a race rather than
    a config mismatch.
    """
    block_joints = [j.get('name') for j in ros2_control_block.findall('joint')]
    configured = controllers[ARM_CONTROLLER_NAME]['ros__parameters']['joints']
    assert configured == block_joints


def test_arm_controller_is_registered_with_the_controller_manager(controllers):
    types = controllers['controller_manager']['ros__parameters']
    assert types[ARM_CONTROLLER_NAME]['type'] == (
        'joint_trajectory_controller/JointTrajectoryController')
    assert types['joint_state_broadcaster']['type'] == (
        'joint_state_broadcaster/JointStateBroadcaster')


def test_arm_controller_command_interfaces_are_a_subset_of_the_hardware(
        ros2_control_block, controllers):
    """
    The controller may claim fewer interfaces than the hardware offers, never more.

    The hardware exposes position AND velocity commands; the controller claims
    position only, which is intentional -- TopicBasedSystem forwards a position
    command as the commanded joint state and Isaac Sim's articulation controller
    does the rest.
    """
    joint = ros2_control_block.findall('joint')[0]
    available_command = {
        c.get('name') for c in joint.findall('command_interface')}
    available_state = {
        s.get('name') for s in joint.findall('state_interface')}
    params = controllers[ARM_CONTROLLER_NAME]['ros__parameters']
    assert set(params['command_interfaces']) <= available_command
    assert set(params['state_interfaces']) <= available_state


def test_moveit_controller_name_and_joints_agree(moveit_controllers,
                                                 controllers):
    """
    Require the MoveIt entry to name the same controller and the same joints.

    A mismatch here means MoveIt plans a trajectory and then sends it to a
    FollowJointTrajectory action that does not exist, or sends joint names the
    controller rejects.
    """
    manager = moveit_controllers['moveit_simple_controller_manager']
    assert manager['controller_names'] == [ARM_CONTROLLER_NAME]
    assert manager[ARM_CONTROLLER_NAME]['joints'] == (
        controllers[ARM_CONTROLLER_NAME]['ros__parameters']['joints'])
    assert manager[ARM_CONTROLLER_NAME]['type'] == 'FollowJointTrajectory'
