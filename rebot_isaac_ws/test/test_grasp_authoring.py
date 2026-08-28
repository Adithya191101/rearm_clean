"""Focused contracts for the runtime soup-can grasp set."""

from pathlib import Path
import sys

import numpy as np
import pytest


WS = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = WS / "src" / "rebot_b601dm_perception"
sys.path.insert(0, str(PACKAGE_ROOT))

from rebot_b601dm_perception.grasps import (  # noqa: E402
    ATTACHMENT_MESH_POSITION_M,
    GRASP_HEIGHT_M,
    SOURCE_GRASP_HEIGHT_ABOVE_SUPPORT_M,
    author_grasp_set,
    dump_grasp_yaml,
    load_grasp_set,
)


GRASP_FILE = WS / "config" / "rebot_grasps_soup_can.yaml"


def test_committed_grasp_file_matches_the_runtime_author():
    authored = author_grasp_set()

    assert dump_grasp_yaml(authored) == GRASP_FILE.read_text()
    loaded = load_grasp_set(GRASP_FILE)
    assert len(loaded.grasps) == 3
    assert loaded.object_frame == "soup_can"
    assert loaded.gripper_frame == "gripper_tcp"


def test_every_grasp_is_horizontal_and_clears_the_can():
    grasp_set = author_grasp_set()

    for grasp in grasp_set.grasps:
        assert grasp.approach_axis()[2] == pytest.approx(0.0, abs=1e-9)
        assert grasp.required_gap_m(grasp_set.cylinder) == pytest.approx(
            grasp_set.cylinder.diameter_m
        )
        assert grasp.jaw_contact_gap_m > grasp.required_gap_m(
            grasp_set.cylinder
        )


def test_bottom_up_grasp_and_attachment_cover_the_full_can():
    grasp_set = author_grasp_set()
    height = grasp_set.cylinder.height_m

    assert GRASP_HEIGHT_M == pytest.approx(
        height - SOURCE_GRASP_HEIGHT_ABOVE_SUPPORT_M
    )
    assert ATTACHMENT_MESH_POSITION_M == pytest.approx(
        (0.0, 0.0, height / 2.0)
    )
    assert np.linalg.norm(grasp_set.grasps[0].quat_wxyz) == pytest.approx(
        1.0
    )
