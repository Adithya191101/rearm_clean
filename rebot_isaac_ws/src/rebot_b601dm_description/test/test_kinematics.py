#!/usr/bin/env python3
"""Stage 2 model gates for the reBot B601-DM description package.

Run:  colcon test --packages-select rebot_b601dm_description
      (or directly:  pytest test/test_kinematics.py -v)

Design of this suite, and why it is shaped this way:

  * ABSOLUTE golden poses, not just cross-model parity. The full and driver
    products share arm and wrist macros by construction, so a pure parity test
    passes whenever BOTH models are wrong in the same way. The golden values
    below are hand-derived from Seeed's raw URDF numbers by an independent
    transform composition, so they fail if the shared source drifts.

  * An explicit NEGATIVE assertion that driver end_link is NOT 44.3 mm from
    gripper_tcp. That offset is the actual upstream defect this package fixes,
    and "we fixed it" must be a test, not a comment.

  * nq == 6 on the driver model. The Seeed SDK reports get_joint_count() as
    Pinocchio's model.nq and its entire command API assumes 6.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

PKG_DIR = Path(__file__).resolve().parent.parent
URDF_DIR = PKG_DIR / "urdf"

FULL_XACRO = URDF_DIR / "rebot_b601dm_full.urdf.xacro"
DRIVER_XACRO = URDF_DIR / "rebot_b601dm_driver.urdf.xacro"

ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]

# Tolerances. The FK parity requirement is 1e-6 m / 1e-6 rad per the plan.
TOL_M = 1e-6
TOL_RAD = 1e-6
# Golden absolute poses are quoted to 9 decimals from an independent derivation,
# so they are compared a little more loosely to allow float64 ordering effects.
TOL_GOLDEN_M = 1e-9

# ---------------------------------------------------------------------------
# Golden values.
#
# Derived by composing Seeed's raw URDF origins with fixed-axis RPY (URDF
# convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)), all six revolute joints at
# zero. Independent of the code under test.
#
#   T_base_link5 = j1(-8.416e-05,0,0.08465)
#                . j2(0.020084,0.031625,0.05555; roll -1.5708)
#                . j3(-0.264,0,0)
#                . j4(0.2426,-0.054,-0.001625)
#                . j5(0.078308,-0.0375,-0.03; roll -1.5708)
# ---------------------------------------------------------------------------
GOLDEN_ZERO_POSE = {
    "link5":        np.array([0.076907840, 0.000000336, 0.231700116]),
    "link6":        np.array([0.100599840, 0.000000042, 0.191700116]),
    "gripper_link": np.array([0.260309840, 0.000000042, 0.191700703]),
    "gripper_tcp":  np.array([0.216009840, 0.000000042, 0.191700703]),
}

# gripper_link / gripper_tcp orientation at the zero pose: X unchanged, Y and Z
# flipped (a 180-degree rotation about X, up to the 1.5708-vs-pi/2 rounding in
# Seeed's own numbers).
GOLDEN_ZERO_ROT_TCP = np.array([
    [1.0,  0.0,          0.0],
    [0.0, -1.0,         -7.346e-06],
    [0.0,  7.346e-06,   -1.0],
])

# The upstream defect this package fixes: legacy end_link sits 44.296 mm from
# the canonical TCP, with a 180-degree flip. Quoted so the negative test states
# the number it is ruling out.
LEGACY_END_LINK_OFFSET_M = 0.044296


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _xacro(path: Path, **args: str) -> str:
    """Expand a xacro file to URDF text."""
    cmd = ["xacro", str(path)] + [f"{k}:={v}" for k, v in args.items()]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.fail(f"xacro failed for {path.name}:\n{proc.stderr}")
    return proc.stdout


@pytest.fixture(scope="module")
def full_urdf() -> str:
    return _xacro(FULL_XACRO)


@pytest.fixture(scope="module")
def driver_urdf() -> str:
    return _xacro(DRIVER_XACRO)


@pytest.fixture(scope="module")
def full_urdf_file(full_urdf: str):
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(full_urdf)
    yield Path(f.name)
    os.unlink(f.name)


@pytest.fixture(scope="module")
def driver_urdf_file(driver_urdf: str):
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write(driver_urdf)
    yield Path(f.name)
    os.unlink(f.name)


def _rpy_to_R(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _rot_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """Geodesic angle between two rotations, in radians."""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.arccos(np.clip(c, -1.0, 1.0)))


class UrdfFK:
    """Minimal URDF forward kinematics.

    Independent of Pinocchio on purpose: this suite must be able to fail a bad
    model even in an environment where Pinocchio is absent or disagrees, and the
    Pinocchio-specific facts (nq) are asserted separately below.
    """

    def __init__(self, urdf_text: str) -> None:
        root = ET.fromstring(urdf_text)
        self.joints: dict[str, dict] = {}
        self.parent_of: dict[str, tuple[str, str]] = {}   # child -> (joint, parent)
        self.links = {ln.get("name") for ln in root.findall("link")}

        for j in root.findall("joint"):
            name = j.get("name")
            origin = j.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                if origin.get("xyz"):
                    xyz = np.array([float(v) for v in origin.get("xyz").split()])
                if origin.get("rpy"):
                    rpy = np.array([float(v) for v in origin.get("rpy").split()])
            axis_el = j.find("axis")
            axis = (np.array([float(v) for v in axis_el.get("xyz").split()])
                    if axis_el is not None and axis_el.get("xyz") else np.array([0.0, 0.0, 1.0]))
            mimic_el = j.find("mimic")
            self.joints[name] = {
                "type": j.get("type"),
                "xyz": xyz,
                "rpy": rpy,
                "axis": axis,
                "parent": j.find("parent").get("link"),
                "child": j.find("child").get("link"),
                "limit": j.find("limit"),
                "mimic": (
                    {
                        "joint": mimic_el.get("joint"),
                        "multiplier": float(mimic_el.get("multiplier", "1.0")),
                        "offset": float(mimic_el.get("offset", "0.0")),
                    }
                    if mimic_el is not None else None
                ),
            }
            self.parent_of[j.find("child").get("link")] = (name, j.find("parent").get("link"))

    def _joint_T(self, name: str, q: float) -> np.ndarray:
        j = self.joints[name]
        T = np.eye(4)
        T[:3, :3] = _rpy_to_R(*j["rpy"])
        T[:3, 3] = j["xyz"]

        if j["type"] in ("revolute", "continuous"):
            a = j["axis"] / np.linalg.norm(j["axis"])
            K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            Rq = np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)
            M = np.eye(4)
            M[:3, :3] = Rq
            T = T @ M
        elif j["type"] == "prismatic":
            a = j["axis"] / np.linalg.norm(j["axis"])
            M = np.eye(4)
            M[:3, 3] = a * q
            T = T @ M
        return T

    def fk(self, link: str, q: dict[str, float] | None = None) -> np.ndarray:
        """Pose of `link` in base_link."""
        q = q or {}
        T = np.eye(4)
        cur = link
        chain = []
        while cur in self.parent_of:
            jname, parent = self.parent_of[cur]
            chain.append(jname)
            cur = parent
        for jname in reversed(chain):
            T = T @ self._joint_T(jname, q.get(jname, 0.0))
        return T


@pytest.fixture(scope="module")
def fk_full(full_urdf: str) -> UrdfFK:
    return UrdfFK(full_urdf)


@pytest.fixture(scope="module")
def fk_driver(driver_urdf: str) -> UrdfFK:
    return UrdfFK(driver_urdf)


def _sample_configs(n: int, seed: int = 20260804) -> list[dict[str, float]]:
    """n random arm configurations, uniform within joint limits."""
    limits = {
        "joint1": (-2.8, 2.8),
        "joint2": (-3.14, 0.0),
        "joint3": (-3.14, 0.0),
        "joint4": (-1.87, 1.57),
        "joint5": (-1.57, 1.57),
        "joint6": (-3.14, 3.14),
    }
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        out.append({j: float(rng.uniform(lo, hi)) for j, (lo, hi) in limits.items()})
    return out


# ---------------------------------------------------------------------------
# 1. The models parse
# ---------------------------------------------------------------------------
def test_full_model_passes_check_urdf(full_urdf_file: Path):
    proc = subprocess.run(["check_urdf", str(full_urdf_file)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"check_urdf rejected the full model:\n{proc.stderr}"


def test_driver_model_passes_check_urdf(driver_urdf_file: Path):
    proc = subprocess.run(["check_urdf", str(driver_urdf_file)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, f"check_urdf rejected the driver model:\n{proc.stderr}"


# ---------------------------------------------------------------------------
# 2. ABSOLUTE zero-pose FK against hand-derived golden values.
#    Catches "both models wrong identically", which parity cannot.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("link", sorted(GOLDEN_ZERO_POSE))
def test_full_model_absolute_zero_pose_position(fk_full: UrdfFK, link: str):
    got = fk_full.fk(link)[:3, 3]
    want = GOLDEN_ZERO_POSE[link]
    err = float(np.linalg.norm(got - want))
    assert err < TOL_GOLDEN_M, (
        f"{link} at the zero pose is {err:.3e} m from the hand-derived golden "
        f"value.\n  got  {got}\n  want {want}"
    )


def test_full_model_absolute_zero_pose_tcp_orientation(fk_full: UrdfFK):
    got = fk_full.fk("gripper_tcp")[:3, :3]
    err = _rot_angle(got, GOLDEN_ZERO_ROT_TCP)
    assert err < 1e-6, (
        f"gripper_tcp orientation at the zero pose is {math.degrees(err):.6f} deg "
        f"from the golden value.\n{got}"
    )


def test_driver_model_absolute_zero_pose_end_link(fk_driver: UrdfFK):
    """The driver's end_link must land on the canonical TCP, not on the legacy
    end point 44.3 mm away."""
    got = fk_driver.fk("end_link")[:3, 3]
    want = GOLDEN_ZERO_POSE["gripper_tcp"]
    err = float(np.linalg.norm(got - want))
    assert err < TOL_GOLDEN_M, (
        f"driver end_link at the zero pose is {err:.3e} m from the canonical TCP "
        f"golden value.\n  got  {got}\n  want {want}"
    )


# ---------------------------------------------------------------------------
# 3. Cross-model FK parity, 1000 configurations.
# ---------------------------------------------------------------------------
def test_cross_model_fk_parity_1000_configs(fk_full: UrdfFK, fk_driver: UrdfFK):
    """full-model gripper_tcp == driver-model end_link, position and orientation.

    This is the property the Seeed SDK relies on: it plans and reports through
    end_link on the driver model, while MoveIt/cuMotion plan through gripper_tcp
    on the full model. If these disagree, every commanded pose is silently offset.
    """
    worst_pos = 0.0
    worst_rot = 0.0
    worst_cfg: dict[str, float] = {}
    for cfg in _sample_configs(1000):
        A = fk_full.fk("gripper_tcp", cfg)
        B = fk_driver.fk("end_link", cfg)
        dp = float(np.linalg.norm(A[:3, 3] - B[:3, 3]))
        dr = _rot_angle(A[:3, :3], B[:3, :3])
        if dp > worst_pos:
            worst_pos, worst_cfg = dp, cfg
        worst_rot = max(worst_rot, dr)

    assert worst_pos < TOL_M, (
        f"position parity failed: worst {worst_pos:.3e} m > {TOL_M:.0e} m "
        f"at {worst_cfg}"
    )
    assert worst_rot < TOL_RAD, (
        f"orientation parity failed: worst {worst_rot:.3e} rad > {TOL_RAD:.0e} rad"
    )


def test_driver_gripper_tcp_alias_matches_end_link(fk_driver: UrdfFK):
    """Both names must resolve to the same pose on the driver model."""
    for cfg in _sample_configs(100, seed=7):
        A = fk_driver.fk("end_link", cfg)
        B = fk_driver.fk("gripper_tcp", cfg)
        assert np.linalg.norm(A[:3, 3] - B[:3, 3]) < TOL_M
        assert _rot_angle(A[:3, :3], B[:3, :3]) < TOL_RAD


# ---------------------------------------------------------------------------
# 4. NEGATIVE test: the 44.3 mm legacy offset is gone.
# ---------------------------------------------------------------------------
def test_driver_end_link_is_not_the_legacy_frame(fk_driver: UrdfFK, fk_full: UrdfFK):
    """Upstream's end_link sits 44.296 mm from gripper_tcp, with a 180-degree
    flip about X. Assert we are NOT reproducing that.

    Without this test, a regression that restored the legacy placement would
    still pass every parity check (both models would move together) and would
    only show up as grasps closing 44 mm too deep into the object.
    """
    d = float(np.linalg.norm(
        fk_driver.fk("end_link")[:3, 3] - fk_full.fk("gripper_tcp")[:3, 3]))
    assert not (abs(d - LEGACY_END_LINK_OFFSET_M) < 1e-4), (
        f"driver end_link is {d * 1000:.3f} mm from gripper_tcp -- that is the "
        f"legacy fixend placement, not the canonical TCP."
    )
    assert d < TOL_M, f"driver end_link is {d * 1000:.4f} mm from gripper_tcp"


def test_gripper_tcp_is_44mm_from_gripper_link(fk_full: UrdfFK):
    """Positive statement of the geometry: the TCP is 44.3 mm along
    gripper_link -X. If this ever becomes 0, gripper_tcp has collapsed onto the
    gripper body and grasps will be planned inside the object."""
    d = float(np.linalg.norm(
        fk_full.fk("gripper_tcp")[:3, 3] - fk_full.fk("gripper_link")[:3, 3]))
    assert abs(d - 0.0443) < 1e-9, f"gripper_link -> gripper_tcp is {d:.9f} m, want 0.0443"


# ---------------------------------------------------------------------------
# 5. Joint limits: the velocity-limit fix, and bounds.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("joint,want_vel", [
    ("joint1", 5.0), ("joint2", 5.0), ("joint3", 5.0),
    ("joint4", 3.0), ("joint5", 3.0), ("joint6", 3.0),
])
def test_arm_velocity_limits_are_corrected(fk_full: UrdfFK, joint: str, want_vel: float):
    """Upstream exports 50 rad/s (j1-3) and 200 rad/s (j4-6). cuMotion reads the
    URDF, not joint_limits.yaml, so leaving those in place lets the planner emit
    trajectories the hardware cannot track."""
    got = float(fk_full.joints[joint]["limit"].get("velocity"))
    assert got == pytest.approx(want_vel), (
        f"{joint} velocity limit is {got}, want {want_vel} "
        f"(upstream's 50/200 are export artifacts)"
    )


def test_velocity_limits_match_in_both_products(fk_full: UrdfFK, fk_driver: UrdfFK):
    for j in ARM_JOINTS:
        a = float(fk_full.joints[j]["limit"].get("velocity"))
        b = float(fk_driver.joints[j]["limit"].get("velocity"))
        assert a == b, f"{j} velocity limit differs: full={a} driver={b}"


@pytest.mark.parametrize("joint,lower,upper", [
    ("joint1", -2.8, 2.8),
    ("joint2", -3.14, 0.0),
    ("joint3", -3.14, 0.0),
    ("joint4", -1.87, 1.57),
    ("joint5", -1.57, 1.57),
    ("joint6", -3.14, 3.14),
])
def test_arm_position_limits_preserved(fk_full: UrdfFK, joint, lower, upper):
    lim = fk_full.joints[joint]["limit"]
    assert float(lim.get("lower")) == pytest.approx(lower)
    assert float(lim.get("upper")) == pytest.approx(upper)


def test_jaw_limits_and_velocity(fk_full: UrdfFK):
    for j in ("gripper_joint1", "gripper_joint2"):
        lim = fk_full.joints[j]["limit"]
        assert float(lim.get("lower")) == pytest.approx(0.0)
        assert float(lim.get("upper")) == pytest.approx(0.0715)
        # Upstream exports 15 m/s for a 71.5 mm stroke (full travel in 5 ms).
        assert float(lim.get("velocity")) == pytest.approx(0.2), (
            f"{j} velocity is {lim.get('velocity')}, want 0.2 m/s"
        )


# ---------------------------------------------------------------------------
# 6. The authored mimic.
# ---------------------------------------------------------------------------
def test_gripper_mimic_exists_with_plus_one_multiplier(fk_full: UrdfFK):
    """Authored, not corrected -- upstream has no <mimic> at all.

    +1.0 is correct because the two jaw joint origins are antiparallel
    (yaw -1.5708 vs +1.5708) with both axes on local +X, so equal positive joint
    values open the jaws symmetrically. -1.0 would immediately drive one jaw
    outside its [0, 0.0715] limit.
    """
    m = fk_full.joints["gripper_joint2"]["mimic"]
    assert m is not None, "gripper_joint2 has no <mimic>; the gripper is 2 DoF"
    assert m["joint"] == "gripper_joint1"
    assert m["multiplier"] == pytest.approx(1.0)
    assert m["offset"] == pytest.approx(0.0)


def test_mimic_produces_symmetric_jaws(fk_full: UrdfFK):
    """Equal joint values must move the jaws symmetrically apart, and the jaw
    separation must grow with the commanded value."""
    def sep(q: float) -> float:
        cfg = {"gripper_joint1": q, "gripper_joint2": q}
        L = fk_full.fk("gripper_left", cfg)[:3, 3]
        R = fk_full.fk("gripper_right", cfg)[:3, 3]
        return float(np.linalg.norm(L - R))

    closed, mid, wide = sep(0.0), sep(0.03), sep(0.0715)
    assert closed < mid < wide, (
        f"jaw separation is not monotonic in the joint value: "
        f"{closed:.4f} / {mid:.4f} / {wide:.4f} -- check the mimic sign"
    )
    # Symmetry: the jaw midpoint must not translate as the gripper opens.
    def mid_point(q: float) -> np.ndarray:
        cfg = {"gripper_joint1": q, "gripper_joint2": q}
        return (fk_full.fk("gripper_left", cfg)[:3, 3]
                + fk_full.fk("gripper_right", cfg)[:3, 3]) / 2.0
    drift = float(np.linalg.norm(mid_point(0.0715) - mid_point(0.0)))
    assert drift < 1e-4, f"jaw midpoint drifts {drift * 1000:.3f} mm when opening"


# ---------------------------------------------------------------------------
# 7. Frame contract: which links each product must and must not have.
# ---------------------------------------------------------------------------
def test_full_model_frame_contract(fk_full: UrdfFK):
    required = {
        "base_link", "link1", "link2", "link3", "link4", "link5", "link6",
        "gripper_link", "gripper_tcp", "gripper_left", "gripper_right",
        "camera_mount", "camera_link",
        "camera_color_optical_frame", "camera_depth_optical_frame",
    }
    missing = required - fk_full.links
    assert not missing, f"full model is missing frames: {sorted(missing)}"


def test_driver_model_frame_contract(fk_driver: UrdfFK):
    required = {"base_link", "link6", "gripper_link", "end_link", "gripper_tcp"}
    missing = required - fk_driver.links
    assert not missing, f"driver model is missing frames: {sorted(missing)}"

    # The jaws must NOT be here: they would make Pinocchio's nq 8, not 6.
    forbidden = {"gripper_left", "gripper_right"}
    present = forbidden & fk_driver.links
    assert not present, (
        f"driver model contains jaw links {sorted(present)}; this breaks the "
        f"SDK's 6-vector contract"
    )


def test_driver_model_has_no_actuated_gripper_joints(fk_driver: UrdfFK):
    actuated = [n for n, j in fk_driver.joints.items() if j["type"] != "fixed"]
    assert sorted(actuated) == ARM_JOINTS, (
        f"driver model actuated joints are {sorted(actuated)}, want exactly {ARM_JOINTS}"
    )


def test_camera_optical_frame_follows_rep145(fk_full: UrdfFK):
    """The optical frame must be Z-forward / X-right / Y-down relative to the
    camera body frame. Getting this wrong rotates every perceived pose by 90
    degrees, which reads as a calibration error rather than a frame bug."""
    T = np.linalg.inv(fk_full.fk("camera_link")) @ fk_full.fk("camera_color_optical_frame")
    R = T[:3, :3]
    # optical +Z must be the body +X
    assert np.allclose(R @ np.array([0, 0, 1.0]), np.array([1.0, 0, 0]), atol=1e-6), R
    # optical +Y must be the body -Z (down)
    assert np.allclose(R @ np.array([0, 1.0, 0]), np.array([0, 0, -1.0]), atol=1e-6), R


# ---------------------------------------------------------------------------
# 8. Mesh URIs resolve.
# ---------------------------------------------------------------------------
def test_mesh_uris_resolve_to_files(full_urdf: str):
    root = ET.fromstring(full_urdf)
    missing = []
    checked = 0
    for mesh in root.iter("mesh"):
        uri = mesh.get("filename", "")
        assert uri.startswith("package://"), f"non-package:// mesh URI: {uri}"
        rel = uri[len("package://"):]
        pkg, _, tail = rel.partition("/")
        assert pkg == "rebot_b601dm_description", f"unexpected mesh package: {pkg}"
        checked += 1
        if not (PKG_DIR / tail).is_file():
            missing.append(tail)
    assert checked > 0, "no mesh URIs found in the full model"
    assert not missing, (
        f"{len(missing)} mesh file(s) referenced but not present: {sorted(set(missing))}"
    )


def test_both_products_reference_the_same_visual_meshes(full_urdf: str, driver_urdf: str):
    """The driver model is frames-only for the wrist, but its arm links must use
    the same meshes as the full model -- otherwise the SDK's collision picture
    and MoveIt's disagree."""
    def arm_meshes(text: str) -> set[str]:
        root = ET.fromstring(text)
        return {m.get("filename") for m in root.iter("mesh")
                if "gripper" not in (m.get("filename") or "")}
    assert arm_meshes(driver_urdf) <= arm_meshes(full_urdf)


# ---------------------------------------------------------------------------
# 9. Pinocchio: nq == 6 on the driver model.
# ---------------------------------------------------------------------------
def test_driver_model_pinocchio_nq_is_six(driver_urdf_file: Path):
    """robot_model.py reports get_joint_count() as model.nq and the SDK's command
    API assumes exactly 6. Fixed joints do not contribute, which is what lets
    end_link/gripper_tcp exist here without breaking the contract."""
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    model = pin.buildModelFromUrdf(str(driver_urdf_file))
    assert model.nq == 6, (
        f"driver model nq is {model.nq}, want 6 -- the SDK will misinterpret "
        f"every command vector"
    )
    assert model.nv == 6, f"driver model nv is {model.nv}, want 6"


def test_full_model_pinocchio_nq_is_eight(full_urdf_file: Path):
    """Sanity check on the other side: the full model DOES have the two jaws.
    (Pinocchio ignores <mimic>, so it sees 8 even though the real gripper has one
    motor -- which is exactly why the driver product has to exist separately.)"""
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    model = pin.buildModelFromUrdf(str(full_urdf_file))
    assert model.nq == 8, f"full model nq is {model.nq}, want 8 (6 arm + 2 jaws)"


def test_pinocchio_agrees_with_our_fk(driver_urdf_file: Path, fk_driver: UrdfFK):
    """Third opinion on the FK: Pinocchio vs this file's own implementation.
    If they disagree, the golden values above are not trustworthy."""
    pin = pytest.importorskip("pinocchio", reason="pinocchio not installed")
    model = pin.buildModelFromUrdf(str(driver_urdf_file))
    data = model.createData()
    fid = model.getFrameId("end_link")

    worst = 0.0
    for cfg in _sample_configs(50, seed=99):
        q = np.array([cfg[j] for j in ARM_JOINTS])
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        want = fk_driver.fk("end_link", cfg)[:3, 3]
        worst = max(worst, float(np.linalg.norm(data.oMf[fid].translation - want)))
    assert worst < 1e-9, f"Pinocchio and local FK disagree by {worst:.3e} m"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
