#!/usr/bin/env python3
"""Send one SINGLE_BIN pick-and-place goal to the reBot multi-object BT.

Run INSIDE the container after the workflow is up (the orchestrator hosts the
`multi_object_pick_and_place` action server). The BT's perception/pick subtree
self-runs, but the DROP pose comes from the goal's target_poses -- an empty goal
leaves UpdateDropPoseFromTargets waiting forever ("Waiting for target_poses from
action call"). This client supplies exactly one target pose inside PLACE_AREA,
so the placed can lands in the validated place rectangle.

DROP POSE
    The default is composed from an upright target object pose and the same
    object-relative side grasp used at the detected bottom-up source. The source
    TCP is rolled 180 degrees by FoundationPose's axial sign; this upright target
    removes that roll so the can's semantic bottom points toward the table.
    base_frame_id for the BT is base_link, and target_poses are consumed as
    base_link TCP poses by the planner.

    The TCP is released 130 mm above the table, leaving the can's semantic
    bottom 119 mm above it for the current 11 mm object-relative grasp. This is
    the lowest upright target validated by cuMotion for the attached can. With
    nvblox enabled, commanding the
    attached can directly onto the reconstructed tabletop is collision-invalid.
    Opening the physical fingers above the surface lets gravity complete the
    upright placement without writing or parenting the object's pose.

The validated place pose remains the default. Run with ``--help`` to override
the drop pose or timeouts without editing source. Object selection, perception,
mesh assignment, and grasps remain workflow-profile settings.
"""

from __future__ import annotations

import sys

import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.action import ActionClient
from rclpy.node import Node

from isaac_ros_manipulation_interfaces.action import MultiObjectPickAndPlace

sys.path.insert(0, "/workspaces/rebot_isaac_ws/src/rebot_b601dm_perception")
sys.path.insert(0, "/workspaces/rebot_isaac_ws/sim")
from rebot_b601dm_perception.grasps import author_grasp_set  # noqa: E402
import pick_area as pa  # noqa: E402
from pick_goal_config import (  # noqa: E402
    DropTarget,
    compose_drop_target,
    parse_goal_options,
)


RELEASE_TCP_HEIGHT_ABOVE_TABLE_M = 0.130
UPRIGHT_OBJECT_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)


def _default_target() -> DropTarget:
    px, py, pz = pa.place_centre()
    grasp = author_grasp_set().grasps[0]
    object_bottom_z = (
        float(pz)
        + RELEASE_TCP_HEIGHT_ABOVE_TABLE_M
        - float(grasp.position[2])
    )
    # Add release clearance to the desired upright semantic bottom, then compose
    # T_world_object_target with T_object_tcp. All supported grasps differ only
    # by yaw, so whichever one cuMotion selected at pick still yields +Z upright.
    return compose_drop_target(
        frame_id="base_link",
        object_position=(
            float(px),
            float(py),
            object_bottom_z,
        ),
        object_quaternion_xyzw=UPRIGHT_OBJECT_QUAT_XYZW,
        grasp_position=grasp.position,
        grasp_quaternion_wxyz=grasp.quat_wxyz,
    )


def main(argv=None) -> int:
    options = parse_goal_options(argv, default_target=_default_target())
    target = options.target

    rclpy.init(args=None)
    node = Node("send_pick_goal")
    try:
        client = ActionClient(
            node, MultiObjectPickAndPlace, "multi_object_pick_and_place"
        )
        node.get_logger().info(
            "waiting for multi_object_pick_and_place action server..."
        )
        if not client.wait_for_server(timeout_sec=options.server_timeout_s):
            node.get_logger().error("action server never appeared")
            return 2

        drop = Pose()
        drop.position.x = target.x
        drop.position.y = target.y
        drop.position.z = target.z
        drop.orientation.x = target.qx
        drop.orientation.y = target.qy
        drop.orientation.z = target.qz
        drop.orientation.w = target.qw

        goal = MultiObjectPickAndPlace.Goal()
        goal.mode = MultiObjectPickAndPlace.Goal.SINGLE_BIN
        pa_msg = PoseArray()
        pa_msg.header.frame_id = target.frame_id
        pa_msg.header.stamp = node.get_clock().now().to_msg()
        pa_msg.poses = [drop]
        goal.target_poses = pa_msg
        goal.class_ids = []

        node.get_logger().info(
            f"sending SINGLE_BIN goal: frame={target.frame_id} "
            f"drop=({target.x:.3f}, {target.y:.3f}, {target.z:.3f}) "
            f"quat_xyzw=({target.qx:.3f},{target.qy:.3f},"
            f"{target.qz:.3f},{target.qw:.3f})"
        )

        send = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(
            node, send, timeout_sec=options.send_timeout_s
        )
        goal_handle = send.result()
        if goal_handle is None or not goal_handle.accepted:
            node.get_logger().error("goal was NOT accepted")
            return 3
        node.get_logger().info(
            f"goal accepted; waiting for result "
            f"(up to {options.result_timeout_s:g} s)..."
        )

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(
            node, result_future, timeout_sec=options.result_timeout_s
        )
        wrapped = result_future.result()
        if wrapped is None:
            node.get_logger().error("no result returned; cancelling timed-out goal")
            cancel = goal_handle.cancel_goal_async()
            rclpy.spin_until_future_complete(node, cancel, timeout_sec=5.0)
            return 4

        result = wrapped.result
        print("=" * 60)
        print(
            f"WORKFLOW RESULT status={result.workflow_status} "
            f"summary={result.workflow_summary!r}"
        )
        print("=" * 60)
        accepted_statuses = (1,) if options.require_complete else (1, 2)
        return 0 if result.workflow_status in accepted_statuses else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
