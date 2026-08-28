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

from ament_index_python.packages import get_package_share_directory

import xacro

ROBOT_DESCRIPTION_PACKAGE = 'isaac_ros_manipulation_rebot_robot_description'


def get_robot_description_contents_for_sim(
    urdf_xacro_file: str,
    use_sim_time: bool,
    dump_to_file: bool = False,
    output_file: str = None,
) -> str:
    """Get robot description contents for the reBot B601-DM in Isaac Sim."""
    initial_positions_file = os.path.join(
        get_package_share_directory(ROBOT_DESCRIPTION_PACKAGE),
        'config',
        'initial_positions.yaml'
    )

    mappings = {
        'sim_isaac': 'true' if use_sim_time else 'false',
        'initial_positions_file': initial_positions_file,
        'prefix': '',
        'mesh_package': 'rebot_b601dm_description',
    }

    xacro_processed = xacro.process_file(
        urdf_xacro_file,
        mappings=mappings
    )
    robot_description = xacro_processed.toxml()

    if dump_to_file and output_file:
        with open(output_file, 'w') as file:
            file.write(robot_description)

    return robot_description
