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
Driver launch file for the reBot B601-DM. SIM ONLY.

Named by ``robot_launch_file_path`` in the workflow params YAML;
``isaac_ros_manipulation_bringup/launch/drivers.launch.py`` includes this file and
forwards every workflow key as a launch argument.

There is no real-hardware profile. The reBot arm is driven through the Seeed
``reBotArm_control_py`` SDK, which is not a ``ros2_control`` hardware interface,
so none of the nodes built here (``TopicBasedSystem`` controller_manager, Isaac
Sim joint parser) have a real-hardware analogue. ``use_sim_time:=false`` therefore
raises immediately with that explanation rather than launching a sim graph that
would command nothing, or half-launching and failing later inside
``ReBotDriverUtils``.
"""

from isaac_ros_launch_utils import GroupAction

from isaac_ros_manipulation_rebot_driver_utils import (
    get_isaac_sim_joint_parser_node, ReBotDriverConfig, ReBotDriverUtils,
)
from isaac_ros_manipulation_ros_python_utils.config import CoreConfig
from isaac_ros_manipulation_ros_python_utils.core import (
    get_visualization_actions
)

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, OpaqueFunction
)
from launch_ros.actions import Node

SIM_ONLY_MESSAGE = (
    'The reBot B601-DM integration in isaac_ros_manipulation_rebot_driver_utils '
    'is SIM ONLY: launched with use_sim_time:=false but no real-hardware '
    'profile is built. Real reBot control goes through the Seeed '
    'reBotArm_control_py SDK, which is not a ros2_control hardware interface, so '
    'there is no real ros2_control graph, no real controller config and no real '
    'driver node in this package. Launch with use_sim_time:=true against Isaac '
    'Sim, or bring the arm up from the upstream Seeed package directly.'
)


def launch_setup(context, *args, **kwargs):
    driver_config = ReBotDriverConfig(context)

    # Single explicit sim-only gate. Checked before any node is constructed so
    # the message names the cause; deeper in, the same condition would surface
    # as a KeyError on driver_config.remapped_joint_states (which is empty when
    # use_sim_time is false) or as one of the NotImplementedErrors in
    # ReBotDriverUtils.
    if not driver_config.use_sim_time:
        raise NotImplementedError(SIM_ONLY_MESSAGE)

    rebot = ReBotDriverUtils(driver_config)

    manipulator_init_nodes = []
    core_config = CoreConfig(context)
    manipulator_init_nodes.append(
        get_isaac_sim_joint_parser_node(driver_config.use_sim_time))
    manipulator_init_nodes.extend([
        Node(
            package='isaac_ros_manipulation_rebot_driver_utils',
            executable='sim_gripper_bridge',
            name='sim_gripper_bridge',
            output='screen',
        ),
        Node(
            package='isaac_ros_manipulation_rebot_driver_utils',
            executable='nvblox_camera_info_relay',
            name='nvblox_camera_info_relay',
            output='screen',
        ),
    ])
    manipulator_init_nodes.append(rebot.get_robot_state_publisher())
    ros2_control_nodes = rebot.get_robot_control_nodes()
    move_group_node, moveit_config = rebot.get_moveit_group_node()
    manipulator_init_nodes.extend(
        get_visualization_actions(
            core_config=core_config,
            moveit_config=moveit_config
        )
    )
    return manipulator_init_nodes + ros2_control_nodes + [move_group_node]


def generate_launch_description():

    # This is the file a workflow YAML names in robot_launch_file_path;
    # drivers.launch.py includes it and forwards every workflow key as a launch
    # argument, so args declared here without a default_value are satisfied by
    # the workflow params file (params/rebot_sim_launch_params.yaml).
    launch_args = [
        DeclareLaunchArgument(
            'log_level',
            description='Log level of the container.',
            choices=['debug', 'info', 'warn', 'error']
        ),
        DeclareLaunchArgument(
            'controller_spawner_timeout',
            description='Timeout used when spawning controllers.',
        ),
        DeclareLaunchArgument(
            'urdf_path',
            description='URDF xacro file path',
        ),
        DeclareLaunchArgument(
            'srdf_path',
            description='SRDF xacro file path',
        ),
        DeclareLaunchArgument(
            'joint_limits_file_path',
            description='Joint limits file path',
        ),
        DeclareLaunchArgument(
            'kinematics_file_path',
            description='Kinematics file path',
        ),
        DeclareLaunchArgument(
            'moveit_controllers_file_path',
            description='MoveIt controller config file path',
        ),
        DeclareLaunchArgument(
            'ros2_controllers_file_path',
            description='ROS2 control controller config file path',
        ),
        DeclareLaunchArgument(
            'gripper_type',
            description='Type of gripper mounted on the reBot arm',
            choices=['rebot_parallel'],
        ),
        DeclareLaunchArgument(
            'workflow_type',
            choices=['POSE_TO_POSE', 'PICK_AND_PLACE',
                     'OBJECT_FOLLOWING', 'GEAR_ASSEMBLY'],
            description='Type of workflow to run',
        ),
        DeclareLaunchArgument(
            'robot_type',
            choices=['UR', 'FLEXIV', 'REBOT'],
            description='Robot family used to drive TF frame prefix and '
                        'arm joint name derivation in shared launch utilities.',
        ),
    ]

    group_action = GroupAction(
        actions=[
            OpaqueFunction(function=launch_setup)
        ],
    )

    return LaunchDescription(launch_args + [group_action])
