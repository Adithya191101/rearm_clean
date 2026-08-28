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
"""Align fixed-camera intrinsics with delayed robot-masked depth for nvblox."""

from __future__ import annotations

import copy

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


CAMERAS = (
    (
        "/scene_cam_0/camera_info",
        "/cumotion/camera_1/world_depth",
        "/nvblox/scene_cam_0/depth/camera_info",
    ),
    (
        "/scene_cam_1/camera_info",
        "/cumotion/camera_2/world_depth",
        "/nvblox/scene_cam_1/depth/camera_info",
    ),
)


class NvbloxCameraInfoRelay(Node):
    """Republish static intrinsics with each processed depth image's header."""

    def __init__(self) -> None:
        super().__init__("nvblox_camera_info_relay")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._camera_info: list[CameraInfo | None] = [None] * len(CAMERAS)
        self._publishers = []
        self._subscriptions = []
        self._published_once = [False] * len(CAMERAS)

        for index, (info_topic, depth_topic, output_topic) in enumerate(CAMERAS):
            publisher = self.create_publisher(CameraInfo, output_topic, qos)
            self._publishers.append(publisher)
            self._subscriptions.append(
                self.create_subscription(
                    CameraInfo,
                    info_topic,
                    lambda message, camera=index: self._on_camera_info(
                        camera, message
                    ),
                    qos,
                )
            )
            self._subscriptions.append(
                self.create_subscription(
                    Image,
                    depth_topic,
                    lambda message, camera=index: self._on_depth(camera, message),
                    qos,
                )
            )

    def _on_camera_info(self, camera: int, message: CameraInfo) -> None:
        self._camera_info[camera] = message

    def _on_depth(self, camera: int, depth: Image) -> None:
        camera_info = self._camera_info[camera]
        if camera_info is None:
            return

        aligned = copy.deepcopy(camera_info)
        aligned.header = copy.deepcopy(depth.header)
        self._publishers[camera].publish(aligned)
        if not self._published_once[camera]:
            self.get_logger().info(
                f"aligned camera {camera} intrinsics to {CAMERAS[camera][1]}"
            )
            self._published_once[camera] = True


def main() -> None:
    rclpy.init()
    node = NvbloxCameraInfoRelay()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
