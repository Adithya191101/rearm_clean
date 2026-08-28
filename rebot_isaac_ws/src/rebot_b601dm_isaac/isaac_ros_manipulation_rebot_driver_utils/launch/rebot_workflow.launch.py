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
Entry point for the reBot B601-DM Isaac Sim workflow.

    ros2 launch isaac_ros_manipulation_rebot_driver_utils rebot_workflow.launch.py

Thin wrapper over ``isaac_ros_manipulation_bringup/launch/workflows.launch.py``.
Its only job is to resolve this package's installed reBot workflow params file
EXPLICITLY and fail loudly if it cannot, instead of letting
``load_yaml_params`` fall back to the read-only
``isaac_ros_manipulation_bringup/params/sim_launch_params.yaml`` -- which
describes a UR arm with a Robotiq gripper. All validation logic lives in
``isaac_ros_manipulation_rebot_driver_utils.workflow_config`` so it is unit
testable without a ROS install.

SIM ONLY. There is no real-hardware profile in this package. ``use_sim_time`` is
per-profile and comes from the params file (``'true'``); it is never set globally
here.

Override the params file with ``manipulator_workflow_config:=/abs/path.yaml`` to
bypass the environment-variable lookup (useful for tests).
"""

import os

from ament_index_python.packages import get_package_share_directory

from isaac_ros_manipulation_rebot_driver_utils.workflow_config import (
    DRIVER_UTILS_PKG,
    resolve_workflow_config_path,
    validate_workflow_params,
    WORKFLOW_PARAMS_FILENAME,
    WorkflowConfigError,
)
from isaac_ros_manipulation_ros_python_utils.config import load_yaml_params

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

BRINGUP_PKG = 'isaac_ros_manipulation_bringup'


def launch_setup(context, *args, **kwargs):
    """Resolve the params file, validate it, and hand off to bringup."""
    override = context.perform_substitution(
        LaunchConfiguration('manipulator_workflow_config'))

    if override:
        config_path = override
        if not os.path.exists(config_path):
            # An explicit override that does not exist is a typo, not a reason
            # to search elsewhere. load_yaml_params would treat a bare filename
            # as a bringup-relative lookup and could resolve it to the upstream
            # UR params file.
            raise WorkflowConfigError(
                f'manipulator_workflow_config was set to {config_path!r}, which '
                f'does not exist. Pass an absolute path, or leave the argument '
                f'empty to use the profile installed by {DRIVER_UTILS_PKG}.')
    else:
        package_params_dir = os.path.join(
            get_package_share_directory(DRIVER_UTILS_PKG), 'params')
        config_path = resolve_workflow_config_path(
            config_dir=package_params_dir)

    # Validate before including anything: a bad params file should fail here,
    # with the offending key named, rather than three launch files deeper.
    validate_workflow_params(load_yaml_params(config_path))

    workflows_launch = os.path.join(
        get_package_share_directory(BRINGUP_PKG),
        'launch', 'workflows.launch.py')

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource([workflows_launch]),
            launch_arguments={
                'manipulator_workflow_config': config_path,
            }.items(),
        )
    ]


def generate_launch_description():
    launch_args = [
        DeclareLaunchArgument(
            'manipulator_workflow_config',
            default_value='',
            description='Absolute path to a workflow params YAML. Leave empty '
                        f'to use the packaged {WORKFLOW_PARAMS_FILENAME}.',
        ),
    ]
    return LaunchDescription(launch_args + [OpaqueFunction(function=launch_setup)])
