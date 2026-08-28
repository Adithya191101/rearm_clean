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
Tests for workflow-config resolution and for the shipped params YAML.

Two halves:

*  :func:`resolve_workflow_config_path` must resolve the package-owned profile
   and FAIL LOUDLY when the selected file is absent, rather than let
   ``load_yaml_params`` fall back to
   ``isaac_ros_manipulation_bringup/params/sim_launch_params.yaml`` -- which
   describes a UR arm with a Robotiq gripper.
*  the shipped ``params/rebot_sim_launch_params.yaml`` must carry the mandatory
   keys with the right values, and must not have drifted from the upstream seed's
   key set (a missing key becomes an unsatisfied launch argument three launch
   files deep).


The YAML is parsed with ``yaml.safe_load`` rather than ``load_yaml_params`` so no
``$(ros2 pkg prefix ...)`` subshells run; the assertions are about keys and
literal values, not resolved paths.
"""

import os

from isaac_ros_manipulation_rebot_driver_utils.workflow_config import (
    CUMOTION_KEYS,
    DESCRIPTION_PATH_KEYS,
    EXPECTED_CAMERA_TYPE,
    EXPECTED_GRIPPER_TYPE,
    EXPECTED_ROBOT_TYPE,
    MANDATORY_KEYS,
    resolve_workflow_config_path,
    ROBOT_LAUNCH_FILE_KEY,
    validate_workflow_params,
    WORKFLOW_CONFIG_DIR_ENV,
    WORKFLOW_PARAMS_FILENAME,
    WorkflowConfigError,
)

import pytest

import yaml

_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_DIR = os.path.dirname(_TEST_DIR)
_PARAMS_PATH = os.path.join(_PKG_DIR, 'params', WORKFLOW_PARAMS_FILENAME)

# The upstream seed this file was copied from. Its key set is the contract:
# drivers.launch.py forwards every key as a launch argument.
_UPSTREAM_SEED = os.path.join(
    os.path.dirname(os.path.dirname(_PKG_DIR)),
    'isaac_ros_manipulation',
    'isaac_ros_manipulation_bringup',
    'params',
    'sim_launch_params.yaml',
)

# Dropped relative to upstream: declared only by UR launch files.
_INTENTIONALLY_DROPPED_KEYS = {'ur_type'}


@pytest.fixture(scope='module')
def params():
    with open(_PARAMS_PATH) as handle:
        return yaml.safe_load(handle)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------


def test_unset_env_var_raises():
    with pytest.raises(WorkflowConfigError) as excinfo:
        resolve_workflow_config_path(environ={})
    message = str(excinfo.value)
    assert WORKFLOW_CONFIG_DIR_ENV in message
    # The message must be actionable, not just a name.
    assert 'entrypoint.sh' in message
    assert 'mkdir -p' in message


def test_empty_env_var_raises():
    """An exported-but-empty variable is as unusable as an unset one."""
    with pytest.raises(WorkflowConfigError):
        resolve_workflow_config_path(
            environ={WORKFLOW_CONFIG_DIR_ENV: ''})


def test_unset_env_var_does_not_fall_back_to_bringup_params():
    """
    The whole point of the guard.

    A silent fall back would bring up a UR arm and the first symptom would be
    MoveIt failing to find shoulder_pan_joint.
    """
    with pytest.raises(WorkflowConfigError) as excinfo:
        resolve_workflow_config_path(environ={})
    assert 'isaac_ros_manipulation_bringup' in str(excinfo.value)


def test_missing_file_in_a_set_dir_raises_with_the_path():
    with pytest.raises(WorkflowConfigError) as excinfo:
        resolve_workflow_config_path(
            environ={WORKFLOW_CONFIG_DIR_ENV: '/opt/workflows'},
            exists=lambda _path: False)
    message = str(excinfo.value)
    assert '/opt/workflows' in message
    assert WORKFLOW_PARAMS_FILENAME in message


def test_present_file_resolves_to_an_absolute_path():
    resolved = resolve_workflow_config_path(
        environ={WORKFLOW_CONFIG_DIR_ENV: '/opt/workflows'},
        exists=lambda _path: True)
    assert resolved == os.path.join('/opt/workflows', WORKFLOW_PARAMS_FILENAME)


def test_explicit_package_directory_is_preferred_over_environment():
    resolved = resolve_workflow_config_path(
        config_dir='/overlay/share/rebot/params',
        environ={WORKFLOW_CONFIG_DIR_ENV: '/wrong/profile'},
        exists=lambda path: path.startswith('/overlay/'),
    )
    assert resolved == os.path.join(
        '/overlay/share/rebot/params', WORKFLOW_PARAMS_FILENAME)


def test_missing_packaged_file_fails_without_fallback():
    with pytest.raises(WorkflowConfigError, match='Rebuild') as excinfo:
        resolve_workflow_config_path(
            config_dir='/overlay/share/rebot/params',
            environ={WORKFLOW_CONFIG_DIR_ENV: '/existing/wrong/profile'},
            exists=lambda _path: False,
        )
    assert 'UR + Robotiq' in str(excinfo.value)

# ---------------------------------------------------------------------------
# Params validation
# ---------------------------------------------------------------------------


def test_shipped_params_validate(params):
    validate_workflow_params(params)


@pytest.mark.parametrize('key', MANDATORY_KEYS)
def test_removing_a_mandatory_key_is_rejected(params, key):
    broken = dict(params)
    del broken[key]
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert key in str(excinfo.value)


@pytest.mark.parametrize('key', DESCRIPTION_PATH_KEYS)
def test_removing_a_description_path_is_rejected(params, key):
    broken = dict(params)
    del broken[key]
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert key in str(excinfo.value)


@pytest.mark.parametrize('key', CUMOTION_KEYS)
def test_removing_a_cumotion_key_is_rejected(params, key):
    broken = dict(params)
    del broken[key]
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert key in str(excinfo.value)


def test_removing_the_robot_launch_file_path_is_rejected(params):
    broken = dict(params)
    del broken[ROBOT_LAUNCH_FILE_KEY]
    with pytest.raises(WorkflowConfigError):
        validate_workflow_params(broken)


def test_a_ur_robot_type_is_rejected(params):
    broken = dict(params, robot_type='UR')
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert 'robot_type' in str(excinfo.value)


def test_a_robotiq_gripper_type_is_rejected(params):
    broken = dict(params, gripper_type='robotiq_2f_140')
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert 'gripper_type' in str(excinfo.value)


def test_a_realsense_camera_type_is_rejected(params):
    broken = dict(params, camera_type='REALSENSE')
    with pytest.raises(WorkflowConfigError):
        validate_workflow_params(broken)


def test_use_sim_time_false_is_rejected(params):
    """Sim-only build: there is no real profile to route to."""
    broken = dict(params, use_sim_time='false')
    with pytest.raises(WorkflowConfigError) as excinfo:
        validate_workflow_params(broken)
    assert 'sim-only' in str(excinfo.value)

# ---------------------------------------------------------------------------
# The shipped YAML itself
# ---------------------------------------------------------------------------


def test_three_mandatory_keys_have_the_expected_values(params):
    assert params['workflow_type'] == 'PICK_AND_PLACE'
    assert params['camera_type'] == EXPECTED_CAMERA_TYPE
    assert params['robot_type'] == EXPECTED_ROBOT_TYPE


def test_live_perception_and_mapping_values(params):
    assert params['num_cameras'] == 2
    assert params['enable_nvblox'] == 'true'
    assert params['object_detection_type'] == 'GROUNDING_DINO'
    assert params['pose_estimation_type'] == 'FOUNDATION_POSE'
    assert params['use_ground_truth_pose_in_sim'] == 'false'


def test_gripper_type_is_rebot_parallel(params):
    assert params['gripper_type'] == EXPECTED_GRIPPER_TYPE


def test_use_sim_time_is_true(params):
    assert params['use_sim_time'] == 'true'


def test_robot_launch_file_points_at_this_package(params):
    value = params[ROBOT_LAUNCH_FILE_KEY]
    assert 'isaac_ros_manipulation_rebot_driver_utils' in value
    assert value.endswith('rebot_driver.launch.py')


@pytest.mark.parametrize('key', DESCRIPTION_PATH_KEYS)
def test_description_paths_point_at_the_rebot_description_package(params, key):
    assert 'isaac_ros_manipulation_rebot_robot_description' in params[key]


def test_cumotion_urdf_is_a_generated_rebot_urdf(params):
    value = params['cumotion_urdf_file_path']
    assert value.endswith('rebot_b601dm_cumotion_gripper.urdf')
    # NOT $(ros2 pkg prefix --share isaac_ros_cumotion_robot_description)/...
    # like the UR seed: that vendor package is read-only in the container image.
    assert 'isaac_ros_cumotion_robot_description' not in value


def test_cumotion_urdf_exists_on_disk(params):
    """
    Generated by rebot_b601dm_description, so it should already be present.

    Distinct from the XRDF assertion below: this one is expected to hold now.
    """
    ws = os.path.dirname(os.path.dirname(os.path.dirname(_PKG_DIR)))
    relative = params['cumotion_urdf_file_path'].replace(
        '$ISAAC_ROS_WS/', '')
    assert os.path.exists(os.path.join(ws, relative))


def test_cumotion_xrdf_path_is_referenced_but_not_required(params):
    """
    The XRDF is authored by a separate stage and may not exist yet.

    Asserted as a path shape only. Deliberately NOT an existence check: cuMotion
    needs the file at plan time, but sim bringup must not be blocked on a file
    another stage owns.
    """
    value = params['cumotion_xrdf_file_path']
    assert value.endswith('rebot_b601dm_gripper.xrdf')
    assert 'config/xrdf/' in value


def test_cumotion_joint_states_topic_is_the_raw_isaac_stream(params):
    """
    /isaac_joint_states, not /isaac_parsed_joint_states.

    cuMotion tolerates the mimicked jaw that robot_state_publisher and MoveIt
    cannot consume, and wants the unfiltered stream.
    """
    assert params['cumotion_joint_states_topic'] == '/isaac_joint_states'


def test_controller_spawner_timeout_is_ten(params):
    assert params['controller_spawner_timeout'] == 10


def test_key_set_matches_the_upstream_seed(params):
    """
    Guards against a dropped key becoming an unsatisfied launch argument.

    drivers.launch.py forwards the whole dict as launch arguments, and shared
    launch files declare their arguments unconditionally, so a key that upstream
    has and this file lacks fails bringup even when nothing on this robot reads
    it.
    """
    if not os.path.exists(_UPSTREAM_SEED):
        pytest.skip(f'{_UPSTREAM_SEED} not present in this checkout')
    with open(_UPSTREAM_SEED) as handle:
        upstream = yaml.safe_load(handle)
    expected = set(upstream) - _INTENTIONALLY_DROPPED_KEYS
    assert set(params) == expected, (
        f'Workflow schema drift: missing={sorted(expected - set(params))}, '
        f'added={sorted(set(params) - expected)}')


def test_every_upstream_boolean_field_is_preserved(params):
    """Do not replace NVIDIA workflow switches with reBot-only switches."""
    if not os.path.exists(_UPSTREAM_SEED):
        pytest.skip(f'{_UPSTREAM_SEED} not present in this checkout')
    with open(_UPSTREAM_SEED) as handle:
        upstream = yaml.safe_load(handle)

    upstream_booleans = {
        key for key, value in upstream.items()
        if str(value).lower() in {'true', 'false'}
    }
    assert upstream_booleans <= set(params)
