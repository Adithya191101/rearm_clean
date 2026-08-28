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

from typing import Dict, Optional

from isaac_ros_launch_utils.all_types import LaunchConfiguration
from isaac_ros_manipulation_ros_python_utils.config import (
    _get_optional_str, DriverConfig,
)
from isaac_ros_manipulation_ros_python_utils.launch_utils import (
    get_workflow_type
)
from isaac_ros_manipulation_ros_python_utils.manipulator_types import (
    RobotType, WorkflowType,
)

from launch.launch_context import LaunchContext

# Frames the reBot URDF actually publishes, used to override the base class's
# derived names. See the docstring on ReBotDriverConfig for why.
REBOT_GRIPPER_FRAME = 'gripper_link'
REBOT_GRASP_FRAME = 'gripper_tcp'


class ReBotDriverConfig(DriverConfig):
    """
    Config for reBot B601-DM with parallel-jaw gripper workflows.

    Consolidates the launch args consumed by ``rebot_driver.launch.py``.

    Overrides two frames the base class derives by string concatenation.
    :meth:`DriverConfig.__init__` sets ``gripper_frame`` to
    ``f'{prefix}gripper_frame'`` and ``grasp_frame`` to ``f'{prefix}grasp_frame'``.
    Neither frame exists on this robot: the reBot description publishes
    ``gripper_link`` for the gripper body and ``gripper_tcp`` for the grasp point.
    Upstream robots make those literal names resolve by broadcasting them as
    static TF (see ``flexiv_driver_utils.py:253-268``); we do not, so on this
    robot they are names with nothing behind them.

    HOW MUCH THIS ACTUALLY BUYS, measured rather than assumed. These two
    overrides are currently INERT: ``DriverConfig.gripper_frame`` /
    ``.grasp_frame`` are assigned here and read by nothing outside this
    package's own tests. The frames that reach the behaviours come from the
    orchestration YAML through ``params_loader.from_dict`` (both default to the
    same literal ``'gripper_frame'``/``'grasp_frame'`` when the key is absent),
    and ours sets both to ``gripper_tcp`` explicitly. So this is a correct
    value kept for when something does read it -- not the thing standing
    between us and a TF timeout. Claiming otherwise (as this docstring did)
    invites someone to "verify" the override by breaking it and seeing nothing
    happen.

    The one place the unresolvable literal DOES escape is
    ``core.py:291``: ``getattr(workflow_config, 'gripper_frame',
    'gripper_frame')``. ``gripper_frame`` is a field of ``DriverConfig``, not of
    ``PickAndPlaceConfig``, so the getattr misses and the literal is passed as
    ``cumotion_action_server.tool_frame``. That is survivable and NOT fixed
    here: 4.5's ``RobotManagerImpl`` ctor logs ``Specified tool frame '%s' not
    found in robot description`` and falls back to the XRDF's ``tool_frames``,
    whose first entry is ``gripper_tcp`` -- the value we wanted. Verified by
    disassembling the ctor (the not-found branch reaches ``rcutils_log`` and
    falls through; it does not throw). Worth knowing because the log line is
    the only evidence, and a future XRDF whose ``tool_frames[0]`` is not the
    grasp point would silently plan to the wrong frame.

    ``insertion_frame`` is deliberately NOT overridden. The reBot model has no
    insertion frame and there is no measured offset to author one from, so it is
    left at the base class's nonexistent ``{prefix}insertion_frame``. Only
    GEAR_ASSEMBLY consumes it; that workflow is not supported on this robot.
    """

    # Always-on fields.
    remapped_joint_states: Dict

    # Workflow-level args. Optional: set to '' or None when not declared in the
    # launch file using this config.
    workflow_type: Optional[WorkflowType]
    log_level: str
    controller_spawner_timeout: LaunchConfiguration

    def __init__(self, context: LaunchContext):
        super().__init__(context)

        # Identity guard, matching the Flexiv and UR configs. Requires
        # RobotType.REBOT is supplied by the pinned Isaac ROS patch.
        if self.robot_type is not RobotType.REBOT:
            raise ValueError(
                f'ReBotDriverConfig requires robot_type={RobotType.REBOT}, '
                f'got {self.robot_type}')

        workflow_type_str = _get_optional_str(context, 'workflow_type')
        self.workflow_type = (
            get_workflow_type(workflow_type_str) if workflow_type_str else None
        )
        self.log_level = _get_optional_str(context, 'log_level')
        self.controller_spawner_timeout = LaunchConfiguration(
            'controller_spawner_timeout')

        # Point the grasp-side frames at the links the reBot URDF publishes.
        self.gripper_frame = f'{self.frame_prefix}{REBOT_GRIPPER_FRAME}'
        self.grasp_frame = f'{self.frame_prefix}{REBOT_GRASP_FRAME}'

        if self.use_sim_time:
            self.remapped_joint_states = {
                '/joint_states': '/isaac_parsed_joint_states',
                '/controller_manager/robot_description': '/robot_description',
            }
        else:
            self.remapped_joint_states = {}
