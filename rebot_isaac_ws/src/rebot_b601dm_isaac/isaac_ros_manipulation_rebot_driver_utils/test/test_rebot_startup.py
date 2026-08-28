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
r"""
BYOR step-4 startup test for the reBot B601-DM sim driver. CONTAINER ONLY.

=============================================================================
THIS TEST HAS NEVER BEEN EXECUTED. It cannot run on a plain host: it needs the
controller_manager, the topic_based_ros2_control plugin, MoveIt, xacro and an
installed (colcon) overlay, none of which exist outside the Isaac ROS
manipulation container. Do not read a green plain-``pytest`` run as evidence
that it passed -- under plain ``pytest`` every assertion below is SKIPPED.
=============================================================================

Gated on ``ENABLE_MANIPULATOR_TESTING=on_robot``, matching
``isaac_ros_manipulation_flexiv_driver_utils/test/test_drivers_sim.py``. Without
that variable the launch description brings up only a ``static_transform_publisher``
and every assertion skips, so the file is collectible (and lintable) everywhere.

To run it, inside the container::

    colcon build --packages-select isaac_ros_manipulation_rebot_driver_utils
    source install/setup.bash
    ENABLE_MANIPULATOR_TESTING=on_robot launch_test \\
        src/rebot_b601dm_isaac/isaac_ros_manipulation_rebot_driver_utils/test/\\
test_rebot_startup.py

Isaac Sim does NOT need to be running. That is deliberate and is what makes this
a startup test rather than an integration test:

*  ``topic_based_ros2_control/TopicBasedSystem`` publishes its initial commanded
   position on ``/isaac_joint_commands`` as soon as the arm controller is
   activated, with no state feedback required. That proves the ros2_control graph
   came up and that the joint NAMES in ``rebot_sim.ros2_control.xacro`` are the
   six this robot has.
*  the joint parser is driven by a synthetic ``/isaac_joint_states`` message
   published by this test, standing in for Isaac Sim's articulation. That proves
   the node's subscribe/publish wiring and its parameter defaults, the only parts
   of the parser not already covered on the host by ``test_joint_state_filter.py``.

What this does NOT cover: whether Isaac Sim's actual USD articulation uses these
joint names, and whether the gripper action name in ``compute_gripper_action_name``
matches a server that exists. Both need a real scene.
"""

import os
import unittest

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node as RosNode
import launch_testing

import pytest
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

DRIVER_UTILS_PKG = 'isaac_ros_manipulation_rebot_driver_utils'
WORKFLOW_PARAMS_FILENAME = 'rebot_sim_launch_params.yaml'

# Topics from urdf/rebot_sim.ros2_control.xacro and
# src/isaac_sim_joint_parser_node.py. Restated as literals so a silent rename in
# either file is caught here rather than at bringup.
JOINT_COMMANDS_TOPIC = '/isaac_joint_commands'
JOINT_STATES_TOPIC = '/isaac_joint_states'
PARSED_JOINT_STATES_TOPIC = '/isaac_parsed_joint_states'

# The six arm joints, in MoveIt order. Must match ARM_JOINT_NAMES in
# joint_state_filter.py and compute_joint_names(RobotType.REBOT).
EXPECTED_ARM_JOINTS = (
    'joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6')

DRIVING_JAW = 'gripper_joint1'
MIMIC_JAW = 'gripper_joint2'
EXPECTED_MOVEIT_JOINTS = EXPECTED_ARM_JOINTS + (DRIVING_JAW,)

# Isaac Sim's articulation order: not the MoveIt order, jaws interleaved. Same
# fixture shape as test_joint_state_filter.py, so a parser that preserves the
# incoming order is detectable here too.
ISAAC_ORDER = (
    'joint1', 'joint2', DRIVING_JAW, 'joint3',
    'joint6', 'joint4', MIMIC_JAW, 'joint5',
)
ISAAC_POSITIONS = (0.11, 0.22, 0.03, 0.33, 0.66, 0.44, 0.03, 0.55)

RUN_TEST = (
    os.environ.get('ENABLE_MANIPULATOR_TESTING', '').lower() == 'on_robot'
)

SKIP_REASON = (
    'Container only. Set ENABLE_MANIPULATOR_TESTING=on_robot and run under '
    'launch_test inside the Isaac ROS manipulation container.'
)


@pytest.mark.rostest
def generate_test_description():
    """Launch the reBot sim driver, or a placeholder node when disabled."""
    test_actions = []
    # controller_manager + MoveIt + xacro expansion is slow; 1s is enough for the
    # placeholder node.
    ready_delay_sec = 1.0

    if RUN_TEST:
        ready_delay_sec = 15.0
        driver_launch_dir = os.path.join(
            get_package_share_directory(DRIVER_UTILS_PKG), 'launch')

        # The package's own params file, read from share/params rather than from
        # $ISAAC_ROS_MANIPULATION_WORKFLOW_CONFIG_DIR. Reading the installed copy
        # keeps the test independent of whether the workflow config dir has been
        # populated; resolution through that variable is covered on the host by
        # test_workflow_config.py.
        params = load_yaml_params(
            os.path.join(
                get_package_share_directory(DRIVER_UTILS_PKG),
                'params', WORKFLOW_PARAMS_FILENAME),
            package_name=DRIVER_UTILS_PKG)
        params.update({
            'headless': 'true',
            'use_sim_time': 'true',
            'enable_rviz_visualization': 'false',
            'enable_foxglove_visualization': 'false',
        })

        test_actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([
                    driver_launch_dir, '/rebot_driver.launch.py']),
                launch_arguments={
                    key: str(value) for key, value in params.items()
                }.items()))
    else:
        # Something must be launched or launch_testing has nothing to shut down.
        test_actions.append(
            RosNode(
                package='tf2_ros',
                executable='static_transform_publisher',
                name='static_transform_publisher',
                output='screen',
                arguments=['0', '0', '0', '0', '0', '0', 'world', 'base_link'],
            )
        )

    return LaunchDescription([
        TimerAction(period=0.0, actions=test_actions),
        TimerAction(
            period=ready_delay_sec,
            actions=[launch_testing.actions.ReadyToTest()]),
    ])


@pytest.mark.skipif(not RUN_TEST, reason=SKIP_REASON)
class TestReBotSimStartup(unittest.TestCase):
    """Assert the sim ros2_control graph and the joint parser came up."""

    @classmethod
    def setUpClass(cls):
        cls.received_commands = []
        cls.received_parsed = []
        if not RUN_TEST:
            # Deliberately no rclpy.init(): under a plain pytest run this class
            # is collected, and initialising a ROS context as a side effect of
            # collection would be a surprise.
            cls.node = None
            return
        rclpy.init()
        cls.node = Node('rebot_startup_test_node')
        cls.command_sub = cls.node.create_subscription(
            JointState,
            JOINT_COMMANDS_TOPIC,
            lambda msg: cls.received_commands.append(msg),
            10)
        cls.parsed_sub = cls.node.create_subscription(
            JointState,
            PARSED_JOINT_STATES_TOPIC,
            lambda msg: cls.received_parsed.append(msg),
            10)
        cls.joint_state_pub = cls.node.create_publisher(
            JointState, JOINT_STATES_TOPIC, 10)

    @classmethod
    def tearDownClass(cls):
        if cls.node is None:
            return
        cls.node.destroy_node()
        rclpy.shutdown()

    def _spin_until(self, predicate, timeout_sec):
        end_time = self.node.get_clock().now().nanoseconds + int(
            timeout_sec * 1e9)
        while (self.node.get_clock().now().nanoseconds < end_time
               and not predicate()):
            rclpy.spin_once(self.node, timeout_sec=0.25)
        return predicate()

    def test_ros2_control_publishes_the_six_arm_joints(self):
        """
        Assert TopicBasedSystem commands exactly joint1..joint6.

        Silence here means the ros2_control graph did not come up: either the
        controller spawner timed out (controller_spawner_timeout, 10s) or the
        arm controller name in config/ros2_control_controllers_sim.yaml does not
        match ARM_CONTROLLER_NAME in rebot_driver_utils.py.

        A message with the WRONG names means rebot_sim.ros2_control.xacro and the
        URDF disagree -- e.g. a fabricated joint7 copied from the 7-DoF Flexiv
        block, or a jaw joint added to a block that must carry arm joints only.
        """
        if not RUN_TEST:
            self.skipTest(SKIP_REASON)
        timeout_sec = 30.0
        self.assertTrue(
            self._spin_until(
                lambda: len(self.received_commands) > 0, timeout_sec),
            f'{JOINT_COMMANDS_TOPIC} received no messages within '
            f'{timeout_sec}s. topic_based_ros2_control/TopicBasedSystem should '
            f'publish initial joint positions as soon as '
            f'scaled_joint_trajectory_controller activates.')

        msg = self.received_commands[0]
        self.assertEqual(
            set(msg.name), set(EXPECTED_ARM_JOINTS),
            f'Commanded joint names mismatch: {list(msg.name)}')
        self.assertEqual(
            len(msg.name), len(EXPECTED_ARM_JOINTS),
            f'Expected {len(EXPECTED_ARM_JOINTS)} commanded joints, got '
            f'{len(msg.name)}: {list(msg.name)}')

    def test_ros2_control_does_not_command_the_jaw_joints(self):
        """
        The jaws are not ros2_control joints.

        gripper_joint2 <mimic>s gripper_joint1, so it has no independent command
        interface; the jaws are driven through a gripper action instead. Either
        name appearing here means the ros2_control block picked up gripper joints
        it cannot actuate.
        """
        if not RUN_TEST:
            self.skipTest(SKIP_REASON)
        self.assertTrue(
            self._spin_until(lambda: len(self.received_commands) > 0, 30.0),
            f'{JOINT_COMMANDS_TOPIC} received no messages.')
        names = set(self.received_commands[0].name)
        self.assertNotIn(DRIVING_JAW, names)
        self.assertNotIn(MIMIC_JAW, names)

    def test_ros2_control_does_not_command_a_seventh_joint(self):
        """The reBot arm is 6-DoF; range(1, 8) is the Flexiv value."""
        if not RUN_TEST:
            self.skipTest(SKIP_REASON)
        self.assertTrue(
            self._spin_until(lambda: len(self.received_commands) > 0, 30.0),
            f'{JOINT_COMMANDS_TOPIC} received no messages.')
        self.assertNotIn('joint7', set(self.received_commands[0].name))

    def test_joint_parser_filters_a_synthetic_isaac_message(self):
        """
        End-to-end wiring check for isaac_sim_joint_parser_node.py.

        The filtering logic itself is covered on the host by
        test_joint_state_filter.py. What is exercised here and nowhere else: the
        node is actually installed as an executable, its subscription and
        publication topics resolve to the names the rest of the graph uses, and
        its 'prefix'/'include_driving_jaw' parameter defaults ('' and True) hold
        when the node is launched with no parameter overrides.
        """
        if not RUN_TEST:
            self.skipTest(SKIP_REASON)
        message = JointState()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.name = list(ISAAC_ORDER)
        message.position = list(ISAAC_POSITIONS)

        timeout_sec = 15.0
        end_time = self.node.get_clock().now().nanoseconds + int(
            timeout_sec * 1e9)
        # Republished in a loop: the parser may not have finished matching its
        # subscription when the first message goes out, and JointState is
        # published VOLATILE so a late subscriber gets nothing retroactively.
        while (self.node.get_clock().now().nanoseconds < end_time
               and not self.received_parsed):
            self.joint_state_pub.publish(message)
            rclpy.spin_once(self.node, timeout_sec=0.25)

        self.assertTrue(
            self.received_parsed,
            f'{PARSED_JOINT_STATES_TOPIC} received nothing within '
            f'{timeout_sec}s of publishing on {JOINT_STATES_TOPIC}.')

        parsed = self.received_parsed[0]
        self.assertEqual(
            tuple(parsed.name), EXPECTED_MOVEIT_JOINTS,
            f'Parsed joint names are not the complete MoveIt state order: '
            f'{list(parsed.name)}')
        by_name = dict(zip(parsed.name, parsed.position))
        # joint6 arrives fifth and joint5 eighth in ISAAC_ORDER, so index-based
        # forwarding would swap these two.
        self.assertAlmostEqual(by_name['joint5'], 0.55, places=5)
        self.assertAlmostEqual(by_name['joint6'], 0.66, places=5)
