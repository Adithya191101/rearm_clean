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

import os
import shutil
import tempfile
from typing import Any, Dict, List, Tuple

from ament_index_python.packages import get_package_share_directory
from isaac_ros_manipulation_rebot_driver_utils.config import ReBotDriverConfig
from isaac_ros_manipulation_rebot_driver_utils.robot_description import (
    get_robot_description_contents_for_sim,
)
from isaac_ros_manipulation_robot_utils.robot_controller_base import (
    RobotControllerBase,
)

from launch.actions import Shutdown
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile

from moveit_configs_utils import MoveItConfigsBuilder

import yaml


CUMOTION_MOVEIT_PKG = 'isaac_ros_cumotion_moveit'
DRIVER_UTILS_PKG = 'isaac_ros_manipulation_rebot_driver_utils'
ROBOT_DESCRIPTION_PKG = 'isaac_ros_manipulation_rebot_robot_description'

# Named to match both reference packages because the shared test harness
# (isaac_ros_manipulation_ros_python_utils/test_utils.py) and the gear assembly
# orchestrator refer to this controller by this exact string. Only the name is
# shared; the type is plain joint_trajectory_controller/JointTrajectoryController
# (see config/ros2_control_controllers_sim.yaml in the description package).
ARM_CONTROLLER_NAME = 'scaled_joint_trajectory_controller'

# Isaac Sim publishes every joint on /isaac_joint_states, including the mimicked
# jaw. isaac_sim_joint_parser_node.py republishes the arm-only subset here, and
# both robot_state_publisher and move_group consume that instead.
PARSED_JOINT_STATES_TOPIC = '/isaac_parsed_joint_states'


def load_cumotion_config() -> Dict:
    """Load the cuMotion planning pipeline yaml from ``isaac_ros_cumotion_moveit``."""
    config_file_path = os.path.join(
        get_package_share_directory(CUMOTION_MOVEIT_PKG),
        'config', 'isaac_ros_cumotion_planning.yaml',
    )
    with open(config_file_path) as config_file:
        return yaml.safe_load(config_file)


class ReBotDriverUtils(RobotControllerBase):
    """
    reBot B601-DM implementation of :class:`RobotControllerBase`.

    Home for the three ABC methods (``get_robot_state_publisher``,
    ``get_moveit_group_node``, ``get_robot_control_nodes``). Only the Isaac Sim
    code path is implemented: the real reBot arm is driven through the Seeed
    reBotArm_control_py SDK, which is not a ros2_control hardware interface, so
    there is no real-robot ros2_control graph to build here. Each method raises
    ``NotImplementedError`` when ``use_sim_time`` is false rather than silently
    returning sim nodes against real hardware.
    """

    def __init__(self, driver_config: ReBotDriverConfig):
        super().__init__(driver_config)

    # ------------------------------------------------------------------
    # RobotControllerBase implementation
    # ------------------------------------------------------------------

    def get_robot_state_publisher(self) -> Node:
        """
        Return the ``robot_state_publisher`` for the reBot B601-DM.

        Implements :meth:`RobotControllerBase.get_robot_state_publisher`. Uses
        the Isaac-Sim-aware xacro shipped with
        ``isaac_ros_manipulation_rebot_robot_description``, which pulls its
        geometry from the ``rebot_b601dm_description`` macros.

        Returns
        -------
            Node: Configured ``robot_state_publisher`` node.

        Raises
        ------
            NotImplementedError: If ``use_sim_time`` is false.

        """
        driver_config = self.driver_config
        if not driver_config.use_sim_time:
            raise NotImplementedError(
                'Real-robot control for the reBot B601-DM goes through the Seeed '
                'SDK, not ros2_control. Launch with use_sim_time:=true.')
        return self._get_sim_robot_state_publisher()

    def get_moveit_group_node(self) -> Tuple[Node, Any]:
        """
        Return the MoveIt ``move_group`` node and the matching config bundle.

        Implements :meth:`RobotControllerBase.get_moveit_group_node`. Registers
        the cuMotion planning pipeline as the default and returns the builder so
        downstream consumers (e.g. RViz via ``get_visualization_actions``) reuse
        the same robot description.

        Returns
        -------
            Tuple[Node, Any]: The ``move_group`` node and the
            :class:`moveit_configs_utils.MoveItConfigsBuilder` result that
            produced it.

        Raises
        ------
            NotImplementedError: If ``use_sim_time`` is false.

        """
        driver_config = self.driver_config
        if not driver_config.use_sim_time:
            raise NotImplementedError(
                'Real-robot MoveIt bringup for the reBot B601-DM is not provided '
                'by this package. Launch with use_sim_time:=true.')
        return self._get_sim_moveit_group_node()

    def get_robot_control_nodes(self) -> List[Node]:
        """
        Return the ``ros2_control`` node and controller spawners for the reBot.

        Implements :meth:`RobotControllerBase.get_robot_control_nodes`.

        Returns
        -------
            List[Node]: Nodes the launch file should add directly to its
            ``LaunchDescription``.

        Raises
        ------
            NotImplementedError: If ``use_sim_time`` is false.

        """
        driver_config = self.driver_config
        if not driver_config.use_sim_time:
            raise NotImplementedError(
                'The reBot B601-DM has no ros2_control hardware interface for real '
                'hardware. Launch with use_sim_time:=true.')
        return self._get_sim_robot_control_nodes()

    # ------------------------------------------------------------------
    # Sim path implementations
    # ------------------------------------------------------------------

    def _get_sim_robot_state_publisher(self) -> Node:
        driver_config = self.driver_config
        robot_description_contents = get_robot_description_contents_for_sim(
            urdf_xacro_file=driver_config.urdf_path,
            use_sim_time=driver_config.use_sim_time,
        )
        remappings = [
            ('/joint_states',
             driver_config.remapped_joint_states['/joint_states'])
        ]
        return Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'robot_description': robot_description_contents,
                 'use_sim_time': driver_config.use_sim_time}
            ],
            remappings=remappings,
            on_exit=Shutdown(),
        )

    def _get_sim_moveit_group_node(self) -> Tuple[Node, Any]:
        driver_config = self.driver_config
        robot_description_content = get_robot_description_contents_for_sim(
            urdf_xacro_file=driver_config.urdf_path,
            use_sim_time=driver_config.use_sim_time,
        )
        moveit_config = (
            MoveItConfigsBuilder(
                'rebot_with_gripper',
                package_name=ROBOT_DESCRIPTION_PKG)
            .robot_description_semantic(file_path=driver_config.srdf_path)
            .robot_description_kinematics(
                file_path=driver_config.kinematics_file_path)
            .joint_limits(file_path=driver_config.joint_limits_file_path)
            .trajectory_execution(
                file_path=driver_config.moveit_controllers_file_path)
            .planning_pipelines(pipelines=['ompl'])
            .to_moveit_configs()
        )
        cumotion_config = load_cumotion_config()
        moveit_config.planning_pipelines[
            'planning_pipelines'].insert(0, 'isaac_ros_cumotion')
        moveit_config.planning_pipelines['isaac_ros_cumotion'] = cumotion_config
        moveit_config.planning_pipelines[
            'default_planning_pipeline'] = 'isaac_ros_cumotion'
        # MoveItConfigsBuilder's xacro mapping support does not reach our
        # sim_isaac / initial_positions_file args, so the description is
        # overridden with the eagerly resolved string instead.
        moveit_config.robot_description = {
            'robot_description': robot_description_content}
        move_it_dict = moveit_config.to_dict()
        move_it_dict['planning_pipelines'] = {
            'pipeline_names': ['isaac_ros_cumotion'],
        }
        move_group_node = Node(
            package='moveit_ros_move_group',
            executable='move_group',
            output='screen',
            parameters=[
                move_it_dict,
                {'use_sim_time': driver_config.use_sim_time}
            ],
            arguments=['--ros-args', '--log-level', 'info'],
            remappings=[('joint_states', PARSED_JOINT_STATES_TOPIC)],
            on_exit=Shutdown(),
        )
        return move_group_node, moveit_config

    def _get_sim_robot_control_nodes(self) -> List[Node]:
        driver_config = self.driver_config
        # ros2_control_node gets the controllers file at a path whose STRING does
        # not contain the substring "robot_description". controller_manager 4.45
        # (determine_controller_node_options) forwards the manager node's own CLI
        # args to every controller, but drops any arg containing "robot_description"
        # (a heuristic meant to strip the manager's own robot_description param).
        # That filter matches on substring, not on the flag, and does NOT pop a
        # preceding --params-file flag -- so when the controllers YAML lives under
        # our package (isaac_ros_manipulation_rebot_ROBOT_DESCRIPTION/.../
        # ros2_control_controllers_sim.yaml) the manager forwards a dangling
        # "--params-file" with no path, and every controller aborts to load with
        # "Couldn't parse trailing --params-file flag. No file path provided.".
        # The hardware activates regardless, so this presents as spawners timing
        # out on list_controllers, not as a hardware fault. Staging the file at a
        # neutral path sidesteps the substring match without touching the shared
        # controller_manager.
        controllers_file = _stage_controllers_file_at_neutral_path(
            driver_config.ros2_controllers_file_path)
        ros2_control_node = Node(
            package='controller_manager',
            executable='ros2_control_node',
            parameters=[
                ParameterFile(controllers_file, allow_substs=True),
                {'use_sim_time': driver_config.use_sim_time}
            ],
            remappings=[
                (
                    '/controller_manager/robot_description',
                    driver_config.remapped_joint_states[
                        '/controller_manager/robot_description'],
                )
            ],
            arguments=['--ros-args', '--log-level', 'error'],
            output='screen',
            on_exit=Shutdown(),
        )
        arm_controller_spawner = Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                ARM_CONTROLLER_NAME,
                '-c', '/controller_manager',
                '--controller-manager-timeout',
                driver_config.controller_spawner_timeout,
            ],
        )
        joint_state_broadcaster_spawner = Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'joint_state_broadcaster',
                '--controller-manager', '/controller_manager',
                '--controller-manager-timeout',
                driver_config.controller_spawner_timeout,
            ],
        )
        return [
            ros2_control_node,
            arm_controller_spawner,
            joint_state_broadcaster_spawner,
        ]


# Neutral filename for the staged controllers YAML. Deliberately contains no
# "robot_description" substring (see the call site in _get_sim_robot_control_nodes
# for why controller_manager 4.45 mangles paths that do).
_STAGED_CONTROLLERS_BASENAME = 'rebot_sim_controllers.yaml'


def _stage_controllers_file_at_neutral_path(controllers_file_path: str) -> str:
    """
    Copy the controllers YAML to a path with no ``robot_description`` substring.

    controller_manager 4.45's ``determine_controller_node_options`` forwards the
    manager node's own CLI arguments to every controller it loads, but silently
    drops any argument whose string contains ``robot_description`` -- without
    removing the ``--params-file`` flag that precedes it. When the controllers
    file is served from this robot's description package (whose share path is
    ``.../isaac_ros_manipulation_rebot_robot_description/...``) the manager
    therefore forwards a dangling ``--params-file`` and every controller fails to
    parse its node arguments, so no controller ever spawns even though the
    hardware component itself activated cleanly.

    Copying the file verbatim to a neutral location avoids the substring match
    without patching the shared controller_manager. The original file is never
    modified.

    Args
    ----
        controllers_file_path (str): Resolved path to the controllers YAML.

    Returns
    -------
        str: A path to an identical copy that is safe to forward, or the input
        unchanged if it already carries no ``robot_description`` substring (so
        the common case incurs no copy).

    """
    if 'robot_description' not in controllers_file_path:
        return controllers_file_path
    staged_dir = tempfile.mkdtemp(prefix='rebot_ros2_control_')
    staged_path = os.path.join(staged_dir, _STAGED_CONTROLLERS_BASENAME)
    shutil.copyfile(controllers_file_path, staged_path)
    return staged_path


def get_isaac_sim_joint_parser_node(use_sim_time: bool) -> Node:
    """Return Isaac Sim joint parser node for the reBot B601-DM."""
    return Node(
        package=DRIVER_UTILS_PKG,
        executable='isaac_sim_joint_parser_node.py',
        name='joint_parser',
        output='screen',
        parameters=[{'use_sim_time': use_sim_time}],
        on_exit=Shutdown(),
    )
