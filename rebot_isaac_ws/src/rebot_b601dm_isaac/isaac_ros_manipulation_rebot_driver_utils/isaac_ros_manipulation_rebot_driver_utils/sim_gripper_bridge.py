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
"""ROS action-to-joint-command bridge for the Isaac Sim gripper."""

from __future__ import annotations

import math
import threading
import time

from control_msgs.action import GripperCommand
from rclpy.action import ActionServer
from sensor_msgs.msg import JointState

from .gripper_protocol import (
    ClosingStallDetector,
    contact_stall_detected,
    effective_jaw_target,
    finite_effort_magnitude,
    GRIPPER_COMMAND_TOPIC,
    ISAAC_JOINT_STATES_TOPIC,
    JAW_COMMAND_JOINTS,
    JAW_JOINTS,
    JAW_MAX_OPEN_M,
    paired_drive_effort,
    paired_jaw_target_reached,
    paired_jaw_position,
    ramped_jaw_target,
)


class SimGripperBridge:
    """Translate ``GripperCommand`` goals into measured Isaac jaw motion."""

    def __init__(
        self,
        node,
        *,
        action_name: str,
        contact_m: float,
        feedback_timeout_sec: float,
        callback_group,
        command_topic: str = GRIPPER_COMMAND_TOPIC,
        joint_states_topic: str = ISAAC_JOINT_STATES_TOPIC,
    ) -> None:
        self._node = node
        self._contact_m = float(contact_m)
        self._feedback_timeout_sec = float(feedback_timeout_sec)
        self._condition = threading.Condition()
        self._jaw_position = None
        self._jaw_effort = None
        self._jaw_positions = None

        self._command_pub = node.create_publisher(
            JointState, command_topic, 10)
        self._state_sub = node.create_subscription(
            JointState,
            joint_states_topic,
            self._on_joint_state,
            10,
            callback_group=callback_group,
        )
        self.action_server = ActionServer(
            node,
            GripperCommand,
            action_name,
            self._execute,
            callback_group=callback_group,
        )
        node.get_logger().info(
            "sim gripper bridge serving %s: %s -> leader %s; "
            "paired feedback=%s on %s"
            % (
                action_name,
                command_topic,
                JAW_COMMAND_JOINTS,
                JAW_JOINTS,
                joint_states_topic,
            ))

    def _on_joint_state(self, msg: JointState) -> None:
        try:
            names = list(msg.name)
            indices = [names.index(name) for name in JAW_JOINTS]
            positions = tuple(float(msg.position[index]) for index in indices)
        except (TypeError, ValueError, IndexError):
            return
        if not all(math.isfinite(position) for position in positions):
            return
        efforts = [
            finite_effort_magnitude(msg.effort[index])
            if len(msg.effort) > index else 0.0
            for index in indices
        ]
        with self._condition:
            self._jaw_positions = positions
            self._jaw_position = paired_jaw_position(*positions)
            self._jaw_effort = min(efforts)
            self._condition.notify_all()

    def _publish_target(self, target_m: float) -> None:
        msg = JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(JAW_COMMAND_JOINTS)
        msg.position = [float(target_m)] * len(JAW_COMMAND_JOINTS)
        self._command_pub.publish(msg)

    def _execute(self, goal_handle):
        command = goal_handle.request.command
        target_m, closing = effective_jaw_target(
            command.position, contact_m=self._contact_m)
        started = time.monotonic()
        deadline = started + self._feedback_timeout_sec
        stall_detector = ClosingStallDetector()
        with self._condition:
            ramp_start_m = (
                self._jaw_position
                if self._jaw_position is not None
                else JAW_MAX_OPEN_M
            )

        measured_m = None
        measured_effort_n = None
        stalled = False
        while time.monotonic() < deadline:
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                return self._result(
                    command, measured_m, reached=False, stalled=False)

            drive_target_m = (
                ramped_jaw_target(
                    ramp_start_m,
                    target_m,
                    time.monotonic() - started,
                )
                if closing else target_m
            )
            self._publish_target(drive_target_m)
            with self._condition:
                measured_m = self._jaw_position
                measured_effort_n = self._jaw_effort
                reached = paired_jaw_target_reached(
                    self._jaw_positions, target_m, closing)
                paired_effort_n = paired_drive_effort(
                    self._jaw_positions,
                    target_m,
                    command.max_effort,
                )
                ramp_complete = drive_target_m <= target_m + 1.0e-9
                stalled = closing and contact_stall_detected(
                    stall_detector,
                    measured_m,
                    target_m,
                    command.max_effort,
                    measured_effort_n=measured_effort_n,
                    estimated_effort_n=paired_effort_n,
                    contact_m=self._contact_m,
                    preload_complete=ramp_complete,
                )
                if reached or stalled:
                    break
                remaining = max(0.0, deadline - time.monotonic())
                self._condition.wait(timeout=min(0.1, remaining))
                measured_m = self._jaw_position
                measured_effort_n = self._jaw_effort
                reached = paired_jaw_target_reached(
                    self._jaw_positions, target_m, closing)
                paired_effort_n = paired_drive_effort(
                    self._jaw_positions,
                    target_m,
                    command.max_effort,
                )
                stalled = closing and contact_stall_detected(
                    stall_detector,
                    measured_m,
                    target_m,
                    command.max_effort,
                    measured_effort_n=measured_effort_n,
                    estimated_effort_n=paired_effort_n,
                    contact_m=self._contact_m,
                    preload_complete=ramp_complete,
                )
                if reached or stalled:
                    break
        else:
            measured_m = self._jaw_position
            measured_effort_n = self._jaw_effort

        reached = paired_jaw_target_reached(
            self._jaw_positions, target_m, closing)
        completed = stalled if closing else reached
        effort_n = max(
            finite_effort_magnitude(measured_effort_n),
            paired_drive_effort(
                self._jaw_positions, target_m, command.max_effort)
            if closing else 0.0,
        )
        result = self._result(
            command,
            measured_m,
            effort_n=effort_n,
            reached=reached,
            stalled=stalled,
        )
        if closing and reached:
            self._node.get_logger().error(
                "gripper reached close target %.4f without contact; aborting "
                "pick instead of lifting empty jaws" % target_m)
            goal_handle.abort()
            return result

        if not completed:
            self._node.get_logger().error(
                "gripper command timed out: requested=%.4f target=%.4f "
                "measured=%r paired=%r after %.1fs"
                % (command.position, target_m, self._jaw_positions, measured_m,
                   self._feedback_timeout_sec))
            goal_handle.abort()
            return result

        goal_handle.succeed()
        if closing:
            self._node.get_logger().info(
                "gripper close %.4f stalled on physical contact at %.4f "
                "(drive load %.2f N)"
                % (target_m, measured_m, effort_n))
        else:
            self._node.get_logger().info(
                "gripper open %.4f m/jaw reached (measured %.4f)"
                % (target_m, measured_m))
        return result

    @staticmethod
    def _result(
            command, measured_m, *, effort_n: float = 0.0,
            reached: bool, stalled: bool):
        result = GripperCommand.Result()
        result.position = (
            float(measured_m) if measured_m is not None
            else float(command.position)
        )
        result.effort = float(effort_n)
        result.reached_goal = bool(reached)
        result.stalled = bool(stalled)
        return result
