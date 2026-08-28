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
"""ROS node exposing the physical Isaac Sim gripper action."""

import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node

from .gripper_protocol import GRIPPER_ACTION_NAME, JAW_CONTACT_M
from .sim_gripper_bridge import SimGripperBridge


class SimGripperNode(Node):
    """Serve GripperCommand goals using measured PhysX jaw feedback."""

    def __init__(self):
        super().__init__('sim_gripper_bridge')
        self.declare_parameter('gripper_action_name', GRIPPER_ACTION_NAME)
        self.declare_parameter('jaw_contact_m', JAW_CONTACT_M)
        self.declare_parameter('feedback_timeout_sec', 20.0)
        name = str(self.get_parameter('gripper_action_name').value)
        contact = float(self.get_parameter('jaw_contact_m').value)
        timeout = float(self.get_parameter('feedback_timeout_sec').value)
        self._bridge = SimGripperBridge(
            self,
            action_name=name,
            contact_m=contact,
            feedback_timeout_sec=timeout,
            callback_group=ReentrantCallbackGroup(),
        )


def main(args=None):
    """Run the simulated gripper action server."""
    rclpy.init(args=args)
    node = SimGripperNode()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
