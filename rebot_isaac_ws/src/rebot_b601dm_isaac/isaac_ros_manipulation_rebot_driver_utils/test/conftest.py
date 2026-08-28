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
Make these tests runnable with plain ``python3 -m pytest``, no colcon.

Two obstacles, both worked around here rather than in the tests:

1.  ``isaac_ros_manipulation_ros_python_utils/__init__.py`` star-imports
    ``core.py``, which imports ``moveit_configs_utils`` and ``control_msgs``.
    Neither is installed outside the Isaac ROS container, so a plain
    ``import isaac_ros_manipulation_ros_python_utils.launch_utils`` fails on the
    package ``__init__`` before reaching the module under test. A synthetic
    parent module with an explicit ``__path__`` is installed in ``sys.modules``,
    which lets submodules be imported without ever executing ``__init__.py``.

2.  ``isaac_ros_launch_utils`` (an NVIDIA container-only package) is imported by
    the reBot config module purely for a ``LaunchConfiguration`` re-export. A
    stub module aliasing the real ``launch.substitutions`` symbol stands in for
    it, so nothing is faked: the object the code receives is the genuine
    ``LaunchConfiguration``.

The reBot package's own ``__init__.py`` has the same problem as (1): it
star-imports ``rebot_driver_utils``, which needs ``xacro`` and
``moveit_configs_utils``. It gets the same synthetic-parent treatment, and for the
same reason -- the modules under test (``joint_state_filter``,
``workflow_config``, ``config``) do not need either dependency.

Inside the container every workaround here is inert: the real packages are
importable, so each install function returns early.
"""

import importlib.util
import os
import sys
import types

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))

# .../src/rebot_b601dm_isaac/isaac_ros_manipulation_rebot_driver_utils
_DRIVER_UTILS_PKG_DIR = os.path.dirname(_TEST_DIR)
# .../src
_SRC_DIR = os.path.dirname(os.path.dirname(_DRIVER_UTILS_PKG_DIR))

_PYTHON_UTILS_PKG = 'isaac_ros_manipulation_ros_python_utils'
_PYTHON_UTILS_DIR = os.path.join(
    _SRC_DIR, 'isaac_ros_manipulation', _PYTHON_UTILS_PKG, _PYTHON_UTILS_PKG)

_REBOT_PKG = 'isaac_ros_manipulation_rebot_driver_utils'
_REBOT_PKG_DIR = os.path.join(_DRIVER_UTILS_PKG_DIR, _REBOT_PKG)

# Importable: isaac_ros_manipulation_rebot_driver_utils.*
if _DRIVER_UTILS_PKG_DIR not in sys.path:
    sys.path.insert(0, _DRIVER_UTILS_PKG_DIR)


def _install_parent_without_init(package: str, package_dir: str,
                                 probe_module: str):
    """
    Register ``package`` in ``sys.modules`` so submodules import without its __init__.

    ``probe_module`` is a submodule whose dependencies are only satisfied inside
    the container; if it imports cleanly the real package is usable as-is and
    this function does nothing.
    """
    if package in sys.modules:
        return
    if not os.path.isdir(package_dir):
        return
    try:
        if importlib.util.find_spec(f'{package}.{probe_module}') is not None:
            importlib.import_module(package)
            return
    except Exception:
        pass
    parent = types.ModuleType(package)
    parent.__path__ = [package_dir]
    sys.modules[package] = parent


def _install_python_utils_parent():
    # core.py needs moveit_configs_utils + control_msgs.
    _install_parent_without_init(
        _PYTHON_UTILS_PKG, _PYTHON_UTILS_DIR, 'core')


def _install_rebot_parent():
    # rebot_driver_utils.py needs xacro + moveit_configs_utils.
    _install_parent_without_init(_REBOT_PKG, _REBOT_PKG_DIR, 'rebot_driver_utils')


def _install_isaac_ros_launch_utils_stub():
    """Alias the real LaunchConfiguration under the container-only package name."""
    if importlib.util.find_spec('isaac_ros_launch_utils') is not None:
        return
    from launch.actions import GroupAction
    from launch.substitutions import LaunchConfiguration

    root = types.ModuleType('isaac_ros_launch_utils')
    root.__path__ = []
    root.GroupAction = GroupAction
    all_types = types.ModuleType('isaac_ros_launch_utils.all_types')
    all_types.LaunchConfiguration = LaunchConfiguration
    all_types.GroupAction = GroupAction
    root.all_types = all_types
    sys.modules['isaac_ros_launch_utils'] = root
    sys.modules['isaac_ros_launch_utils.all_types'] = all_types


_install_python_utils_parent()
_install_rebot_parent()
_install_isaac_ros_launch_utils_stub()
