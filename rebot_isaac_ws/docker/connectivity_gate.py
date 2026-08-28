#!/usr/bin/env python3
"""Stage 0 connectivity gate: host Isaac Sim <-> container ROS 2.

Run INSIDE the container with Isaac Sim playing a scene on the host:

    ./docker/run.sh sim python3 docker/connectivity_gate.py

Host-container DDS discovery is the most likely silent failure in this topology
-- everything "starts fine" and simply never exchanges data -- so it gets its
own gate with numeric thresholds instead of an eyeballed rqt_graph.

Each check prints PASS/FAIL and the measured value. Exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

# Thresholds. Isaac Sim's default bridge publishes /clock at the sim step rate
# (60 Hz at the Stage 3 timestep) and RTX sensors at 30 Hz; these floors are set
# well below nominal so the gate flags "not arriving" rather than jitter.
CLOCK_MIN_RATE_HZ = 10.0
CLOCK_MIN_ADVANCE_S = 0.05
CAMERA_MIN_RATE_HZ = 5.0
TF_MIN_MSGS = 1
DISCOVERY_TIMEOUT_S = 20.0
OBSERVE_WINDOW_S = 5.0

# Latched/transient-local, matching what a publisher of static data uses.
LATCHED = QoSProfile(
    depth=1,
    history=HistoryPolicy.KEEP_LAST,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class Result:
    def __init__(self) -> None:
        self.passes = 0
        self.fails = 0

    def record(self, ok: bool, name: str, detail: str = "") -> None:
        tag = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  {tag}  {name}" + (f" -- {detail}" if detail else ""), flush=True)
        if ok:
            self.passes += 1
        else:
            self.fails += 1

    def info(self, name: str, detail: str = "") -> None:
        print(f"  \033[33mINFO\033[0m  {name}" + (f" -- {detail}" if detail else ""),
              flush=True)


class Gate(Node):
    def __init__(self, camera_topic: str) -> None:
        super().__init__("rebot_connectivity_gate")
        self.clock_msgs: list[float] = []
        self.clock_wall: list[float] = []
        self.tf_count = 0
        self.tf_static_count = 0
        self.joint_count = 0
        self.camera_count = 0

        self.create_subscription(Clock, "/clock", self._on_clock, 50)
        self.create_subscription(TFMessage, "/tf", self._on_tf, 50)
        self.create_subscription(TFMessage, "/tf_static", self._on_tf_static, LATCHED)
        self.create_subscription(JointState, "/isaac_joint_states",
                                 self._on_joints, 20)
        self.create_subscription(Image, camera_topic, self._on_camera, 10)

        # Round-trip probe: publish and subscribe our own topic to prove the
        # container's DDS participant can both send and receive on this domain.
        self.echo_rx = 0
        self.echo_pub = self.create_publisher(String, "/rebot_gate_echo", 10)
        self.create_subscription(String, "/rebot_gate_echo", self._on_echo, 10)

    def _on_clock(self, msg: Clock) -> None:
        self.clock_msgs.append(msg.clock.sec + msg.clock.nanosec * 1e-9)
        self.clock_wall.append(time.monotonic())

    def _on_tf(self, msg: TFMessage) -> None:
        self.tf_count += len(msg.transforms)

    def _on_tf_static(self, msg: TFMessage) -> None:
        self.tf_static_count += len(msg.transforms)

    def _on_joints(self, msg: JointState) -> None:
        self.joint_count += 1

    def _on_camera(self, msg: Image) -> None:
        self.camera_count += 1

    def _on_echo(self, msg: String) -> None:
        self.echo_rx += 1


def spin_for(node: Node, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.05)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera-topic", default="/front_stereo_camera/left/image_raw",
                    help="Isaac Sim RGB topic to rate-check")
    ap.add_argument("--require-camera", action="store_true",
                    help="fail (not INFO) if the camera topic is silent")
    args = ap.parse_args()

    r = Result()
    print("=== 0. Environment agreement ===")
    domain = os.environ.get("ROS_DOMAIN_ID", "<unset>")
    rmw = os.environ.get("RMW_IMPLEMENTATION", "<unset>")
    r.record(domain == "42", "ROS_DOMAIN_ID == 42", domain)
    r.record(rmw == "rmw_cyclonedds_cpp", "RMW == rmw_cyclonedds_cpp", rmw)
    print("  NOTE: Isaac Sim on the host must have the SAME two values exported")
    print("        in the shell that launched it, or discovery silently fails.")

    rclpy.init()
    node = Gate(args.camera_topic)
    try:
        print("=== 1. Discovery ===")
        # Wait for Isaac Sim's participant to appear at all.
        deadline = time.monotonic() + DISCOVERY_TIMEOUT_S
        clock_pubs = 0
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
            clock_pubs = node.count_publishers("/clock")
            if clock_pubs > 0:
                break

        r.record(clock_pubs > 0, "a /clock publisher is visible from the container",
                 f"{clock_pubs} publisher(s)")
        # Exactly one owner. Two publishers means something in the container is
        # also publishing sim time, and consumers will see time jump backwards.
        r.record(clock_pubs == 1, "exactly one /clock publisher",
                 f"{clock_pubs} (expect 1)")

        # Endpoints published by a node that is NOT this one.
        #
        # Counting topic NAMES here is vacuous, which this gate learned the hard
        # way: the original check was `len(get_topic_names_and_types()) > 3`, and
        # it PASSED on a deliberately isolated ROS_DOMAIN_ID. This node creates 8
        # topics of its own (5 subscriptions + the echo pair + /rosout and
        # /parameter_events), so the count was 8 with and without a host to talk
        # to -- it was measuring its own graph.
        #
        # Filtering by name is not enough either: everything Isaac Sim publishes
        # (/clock, /tf) is a topic we also subscribe to, so a name-based
        # exclusion set removes exactly the evidence we want. The publisher's
        # node_name is the only thing that actually separates "the host is
        # reachable" from "I can see myself".
        me = node.get_name()
        remote = sorted({
            f"{t} <- {ep.node_name}"
            for t, _ in node.get_topic_names_and_types()
            for ep in node.get_publishers_info_by_topic(t)
            if ep.node_name != me
        })
        r.record(len(remote) > 0, "host-side publishers visible in container",
                 f"{len(remote)} remote endpoint(s): {remote[:4]}")

        print(f"=== 2. Data flow ({OBSERVE_WINDOW_S:.0f} s window) ===")
        for _ in range(5):
            node.echo_pub.publish(String(data="ping"))
        spin_for(node, OBSERVE_WINDOW_S)

        # /clock advancing, not merely present. A paused sim publishes a
        # constant stamp, which use_sim_time consumers treat as a frozen world.
        n = len(node.clock_msgs)
        if n >= 2:
            sim_adv = node.clock_msgs[-1] - node.clock_msgs[0]
            wall = node.clock_wall[-1] - node.clock_wall[0]
            rate = (n - 1) / wall if wall > 0 else 0.0
            r.record(sim_adv >= CLOCK_MIN_ADVANCE_S,
                     f"/clock advancing (>= {CLOCK_MIN_ADVANCE_S} s)",
                     f"{sim_adv:.3f} s of sim time -- is the sim PLAYING?")
            r.record(rate >= CLOCK_MIN_RATE_HZ,
                     f"/clock rate >= {CLOCK_MIN_RATE_HZ} Hz", f"{rate:.1f} Hz")
        else:
            r.record(False, "/clock advancing", f"only {n} message(s) received")

        r.record(node.echo_rx > 0, "container->container action/topic round-trip",
                 f"{node.echo_rx} echo(es)")

        r.record(node.tf_count + node.tf_static_count >= TF_MIN_MSGS,
                 "TF arriving",
                 f"/tf={node.tf_count} transforms, /tf_static={node.tf_static_count}")

        if node.joint_count > 0:
            r.record(True, "/isaac_joint_states arriving", f"{node.joint_count} msgs")
        else:
            r.info("/isaac_joint_states silent",
                   "expected until the Stage 5 ActionGraph exists")

        cam_rate = node.camera_count / OBSERVE_WINDOW_S
        if node.camera_count > 0 or args.require_camera:
            r.record(cam_rate >= CAMERA_MIN_RATE_HZ,
                     f"camera {args.camera_topic} >= {CAMERA_MIN_RATE_HZ} Hz",
                     f"{cam_rate:.1f} Hz")
        else:
            r.info(f"camera {args.camera_topic} silent",
                   "expected until a scene with an RTX camera is loaded; "
                   "re-run with --require-camera once it is")
    finally:
        node.destroy_node()
        rclpy.shutdown()

    print()
    print("=" * 46)
    print(f"PASS: {r.passes}   FAIL: {r.fails}")
    print("=" * 46)
    return 1 if r.fails else 0


if __name__ == "__main__":
    sys.exit(main())
