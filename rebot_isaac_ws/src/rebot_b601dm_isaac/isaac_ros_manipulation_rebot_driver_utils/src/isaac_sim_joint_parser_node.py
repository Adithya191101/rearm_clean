#!/usr/bin/env python3
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
Parse Isaac Sim joint states for the reBot B601-DM with parallel gripper.

I/O shell only. All filtering logic lives in
``isaac_ros_manipulation_rebot_driver_utils.joint_state_filter``, which imports
neither ``rclpy`` nor ``sensor_msgs`` so it can be unit tested without a ROS
install. See that module's docstring for why the arm joints are republished and
why an unexpected name set is rejected.


Topic chain
-----------
    /isaac_joint_states  (Isaac Sim, ALL joints incl. the mimic jaw)
        -> this node
        -> /isaac_parsed_joint_states
        -> (remapped to /joint_states) robot_state_publisher -> /tf


QoS is the plain depth-10 default on both endpoints, matching the UR and Flexiv
parsers. Isaac Sim publishes joint states with default (RELIABLE / VOLATILE)
QoS, so no override is needed; a SENSOR_DATA (BEST_EFFORT) profile here would
fail to match the publisher and the node would receive nothing.
"""
from isaac_ros_manipulation_rebot_driver_utils.joint_state_filter import (
    filter_arm_joint_state, JointStateFilterError,
)

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import JointState

# Seconds between repeats of the same rejection message. A rejection is a
# per-message condition on a stream that runs at simulation rate, so an
# unthrottled log would emit tens of thousands of identical lines and bury the
# first one.
REJECTION_LOG_THROTTLE_SEC = 5.0


class JointParser(Node):
    """Republish the reBot arm and driving jaw from Isaac Sim."""

    def __init__(self):
        super().__init__('joint_parser')
        # Declared so the prefix can be set without editing this file if the
        # description is ever brought up with a non-empty prefix. Defaults to
        # '' to match rebot_b601dm_description's xacro default.
        self.declare_parameter('prefix', '')
        self.declare_parameter('include_driving_jaw', True)
        self._prefix = self.get_parameter('prefix').value
        self._include_driving_jaw = self.get_parameter(
            'include_driving_jaw').value

        self.subscription = self.create_subscription(
            JointState,
            'isaac_joint_states',
            self.listener_callback,
            10
        )
        self.publisher = self.create_publisher(
            JointState, 'isaac_parsed_joint_states', 10)

    def listener_callback(self, msg):
        try:
            filtered = filter_arm_joint_state(
                name=msg.name,
                position=msg.position,
                velocity=msg.velocity,
                effort=msg.effort,
                prefix=self._prefix,
                include_driving_jaw=self._include_driving_jaw,
            )
        except JointStateFilterError as error:
            # Drop the message and say so. Publishing a partial or reordered
            # state would be worse than publishing nothing: MoveIt would plan
            # from joint values attributed to the wrong joints.
            self.get_logger().error(
                f'Rejected /isaac_joint_states message: {error}',
                throttle_duration_sec=REJECTION_LOG_THROTTLE_SEC)
            return

        new_msg = JointState()
        new_msg.header = msg.header
        new_msg.name = list(filtered.name)
        new_msg.position = list(filtered.position)
        new_msg.velocity = list(filtered.velocity)
        new_msg.effort = list(filtered.effort)
        self.publisher.publish(new_msg)


def main(args=None):
    rclpy.init(args=args)
    node = JointParser()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
