"""Author, load and validate the object-relative grasp set.

WHY OBJECT-RELATIVE
The grasp set stores ``T_object_gripper_tcp`` and the runtime composes

    T_cell_tcp = T_cell_object . T_object_tcp

so the same file is valid wherever the can happens to be. This is also what the
upstream consumer already does: ``GraspReader._compute_grasp_poses_from_object_pose``
computes ``world_pose_gripper = world_pose_object @ object_pose_gripper``. Storing
cell-frame grasps instead would silently bind the file to one can position.

FRAME CONVENTIONS, all read from rebot_b601dm_gripper.xacro -- none of these is
a convention this module is free to pick:

  * ``gripper_tcp`` +X is the APPROACH axis. The jaws close on whatever lies
    ahead of the TCP along +X.
  * ``gripper_tcp`` +/-Y is the CLOSING axis. ``gripper_tcp`` is a pure
    translation from ``gripper_link`` (rpy 0,0,0), so it inherits the jaw travel
    direction unchanged.
  * The soup can's object frame has its origin at the CENTRE OF THE BASE and its
    axis along +Z.

WHY A HORIZONTAL SIDE GRASP (and not top-down)
This arm can only SIDE-grasp the can, never top-down. Measured on the GPU at the
reachable can TCP: a HORIZONTAL approach (gripper pointing forward, approach axis
in the ground plane) plans at every roll, while ANY downward tilt -- even 30 deg
off vertical -- fails IK. So the canonical grasp here points the TCP's approach
axis (+X) horizontally INTO the can, closes across the can diameter (+Y), and puts
the TCP on the upper body in the bottom-up source pose. The can's semantic bottom
is facing upward, so a grasp 11 mm from that end is 90 mm above the supporting
table. Its object-relative rotation is the IDENTITY frame; composing it with the
detected bottom-up object pose rolls the world TCP 180 degrees about X. The
upright drop target composes the same stored relation, making the robot visibly
correct the detected orientation instead of relying on a world-fixed grasp.

WHY MORE THAN ONE GRASP (but only a FEW)
A cylinder is rotationally symmetric, so every horizontal azimuth around its axis
grasps the can equally well; they are NOT equally reachable for the arm. Rather
than span the full circle (most of which points the gripper back through the base
and is unreachable), this set offers the one proven azimuth plus two small
neighbours, so cuMotion's goalset has an alternative if the exact centre azimuth
is marginal at some can position. All approaches stay HORIZONTAL and roll is fixed
at 0: rolling about the approach axis would tip the closing axis toward the can's
101 mm height and demand a 101 mm gap -- the error ``cylinder_support_width``
exists to catch, which is why the required width is computed per grasp rather than
compared against 0.066.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import yaml

from .can_geometry import CylinderSpec, SOUP_CAN, cylinder_support_width, point_outside_cylinder
from .errors import GraspSetError
from .jaws import JAW_UPPER_M, command_for_gap, gap_for_command

FORMAT = "isaac_grasp"
FORMAT_VERSION = "1.0"

# Frames. ``gripper_tcp`` and not ``gripper_link``: they are 44.3 mm apart along
# the approach axis (KDR-001), so authoring against the wrong one drives the jaws
# 44 mm into the can.
OBJECT_FRAME = "soup_can"
GRIPPER_FRAME = "gripper_tcp"

# The validated physical TCP is 90 mm above the support surface. The source can
# is bottom-up: its semantic bottom-center is one can height above the table and
# object +Z points down. Therefore the object-relative grasp coordinate is
# height - 90 mm = 11 mm. When composed with the detected 180-degree object pose,
# it lands at the same world z=table+0.09 that was already plan-validated.
SOURCE_GRASP_HEIGHT_ABOVE_SUPPORT_M = 0.09
GRASP_HEIGHT_M = SOUP_CAN.height_m - SOURCE_GRASP_HEIGHT_ABOVE_SUPPORT_M

# The FoundationPose CAD mesh is centered at its origin and the labeled can's
# top axis is +Y. Grasp poses use a base-centered object frame with the cylinder
# axis +Z.
# This is T_object_mesh: move the mesh center half a can height above the base,
# then rotate mesh +Y onto object +Z. AttachObject composes this metadata with
# T_grasp_object so cuMotion's carried collision mesh matches the physical can.
ATTACHMENT_MESH_POSITION_M = (0.0, 0.0, SOUP_CAN.height_m / 2.0)
ATTACHMENT_MESH_QUAT_WXYZ = (
    math.sqrt(0.5),
    math.sqrt(0.5),
    0.0,
    0.0,
)

# Pregrasp standoff back along -approach (TCP -X). For a HORIZONTAL side grasp the
# approach is in the ground plane, so this backs the pregrasp horizontally AWAY
# from the can (radially outward), not upward. 100 mm clears the 33 mm radius with
# 67 mm to spare, so the pregrasp point sits well outside the can.
PREGRASP_OFFSET_M = 0.10

# Jaw commands. BOTH are half-gaps; the gap is recorded next to each because
# reading one as a gap is the bug that already shipped once here.
#
# Contact: chosen as command_for_gap(diameter + 2 mm), i.e. 1 mm of standoff per
# side, then close on force/effort. This is 0.0340 rather than the tempting
# rounded 0.033: 0.033 yields a 65.945 mm gap, 55 um narrower than the 66 mm can.
# That difference is below hardware repeatability, but it does not satisfy a
# strict clearance assertion.
JAW_CONTACT_CLEARANCE_M = 0.002
JAW_PREGRASP_M = 0.045  # gap 0.089945 m

# Azimuths, degrees, about the can's vertical axis. The canonical grasp is
# azimuth 0: approach along object +X (horizontal, straight into the can from the
# reachable side), closing along object +Y. The +/-20 deg neighbours give
# cuMotion's goalset a reachable alternative if the centre azimuth is marginal at
# some can position, WITHOUT sweeping the whole circle -- most of which points the
# gripper back through the arm base and never solves. Kept deliberately small
# (three grasps): fewer orientations means fewer IK failures and a faster,
# more reliable goalset. All approaches stay HORIZONTAL.
SIDE_GRASP_AZIMUTHS_DEG = (0.0, -20.0, 20.0)


@dataclass(frozen=True)
class GraspSpec:
    """One authored grasp, in the object frame.

    ``jaw_contact_m`` and ``jaw_contact_gap_m`` are BOTH stored, and
    :meth:`check` re-derives the gap from the command rather than trusting the
    stored value, so a hand-edited YAML whose two numbers disagree is a failure
    rather than a file that reads plausibly.
    """

    name: str
    azimuth_deg: float
    tilt_deg: float
    position: tuple
    quat_wxyz: tuple
    jaw_contact_m: float
    jaw_contact_gap_m: float
    jaw_pregrasp_m: float
    jaw_pregrasp_gap_m: float
    pregrasp_offset_m: float
    confidence: float = 1.0

    def rotation(self) -> np.ndarray:
        return _quat_wxyz_to_matrix(self.quat_wxyz)

    def approach_axis(self) -> np.ndarray:
        """TCP +X expressed in the object frame."""
        return self.rotation()[:, 0]

    def closing_axis(self) -> np.ndarray:
        """TCP +Y expressed in the object frame."""
        return self.rotation()[:, 1]

    def pregrasp_position(self) -> np.ndarray:
        """Back off along -approach by ``pregrasp_offset_m``."""
        return np.asarray(self.position, dtype=float) - self.approach_axis() * self.pregrasp_offset_m

    def required_gap_m(self, cyl: CylinderSpec) -> float:
        return cylinder_support_width(cyl, self.closing_axis())

    def check(self, cyl: CylinderSpec) -> None:
        """Raise :class:`GraspSetError` if this grasp is not physically sane.

        The interpenetration assertion is the load-bearing one. It compares the
        contact gap against the width the can actually presents along THIS
        grasp's closing direction, not against the diameter -- see the module
        docstring.
        """
        rot = self.rotation()
        # A non-orthonormal rotation reads back as a valid-looking quaternion
        # after normalisation, so check the matrix, not the quaternion norm.
        if not np.allclose(rot.T @ rot, np.eye(3), atol=1e-9):
            raise GraspSetError(f"{self.name}: orientation is not a rotation")
        if float(np.linalg.det(rot)) < 0.0:
            raise GraspSetError(f"{self.name}: orientation is a reflection, not a rotation")

        for label, cmd, stored in (
            ("contact", self.jaw_contact_m, self.jaw_contact_gap_m),
            ("pregrasp", self.jaw_pregrasp_m, self.jaw_pregrasp_gap_m),
        ):
            if not 0.0 <= cmd <= JAW_UPPER_M:
                raise GraspSetError(
                    f"{self.name}: {label} jaw command {cmd} m outside [0, {JAW_UPPER_M}] m"
                )
            derived = gap_for_command(cmd)
            if not math.isclose(derived, stored, abs_tol=1e-9):
                raise GraspSetError(
                    f"{self.name}: {label} gap {stored} m disagrees with the gap "
                    f"{derived} m implied by the command {cmd} m. A jaw command is "
                    f"a HALF-gap; one of these two numbers was computed wrongly."
                )

        required = self.required_gap_m(cyl)

        # THE GATE. Strict: equality is contact, not clearance.
        if self.jaw_contact_gap_m <= required:
            raise GraspSetError(
                f"{self.name}: contact gap {self.jaw_contact_gap_m * 1000:.3f} mm does not "
                f"clear the {cyl.name}, which presents "
                f"{required * 1000:.3f} mm along this grasp's closing axis "
                f"{np.round(self.closing_axis(), 4).tolist()}. The jaws would "
                f"interpenetrate by {(required - self.jaw_contact_gap_m) * 500:.3f} mm per side. "
                f"A jaw command is a HALF-gap: {self.jaw_contact_m} m per jaw opens "
                f"to {self.jaw_contact_gap_m} m, not {self.jaw_contact_m} m."
            )

        if self.jaw_pregrasp_gap_m <= self.jaw_contact_gap_m:
            raise GraspSetError(
                f"{self.name}: pregrasp gap {self.jaw_pregrasp_gap_m} m is not wider than "
                f"the contact gap {self.jaw_contact_gap_m} m -- the jaws would have to "
                f"close on approach"
            )

        if self.pregrasp_offset_m <= 0.0:
            raise GraspSetError(f"{self.name}: pregrasp offset must be positive")

        # A pregrasp inside the can is what an offset applied along +approach
        # instead of -approach produces, and the number still looks plausible.
        if not point_outside_cylinder(cyl, self.pregrasp_position()):
            raise GraspSetError(
                f"{self.name}: pregrasp point "
                f"{np.round(self.pregrasp_position(), 4).tolist()} lies inside the "
                f"{cyl.name}. The offset is applied along -approach; a pregrasp "
                f"inside the object means the sign is flipped."
            )

        if not 0.0 < self.confidence <= 1.0:
            raise GraspSetError(f"{self.name}: confidence {self.confidence} outside (0, 1]")


@dataclass
class GraspSet:
    object_frame: str
    gripper_frame: str
    cylinder: CylinderSpec
    grasps: List[GraspSpec] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.grasps)

    def by_name(self) -> Dict[str, GraspSpec]:
        return {g.name: g for g in self.grasps}


def _quat_wxyz_to_matrix(quat: Sequence[float]) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        raise GraspSetError("zero quaternion")
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _matrix_to_quat_wxyz(rot: np.ndarray) -> tuple:
    """Shepperd's method, canonicalised to w >= 0.

    Same branch-selection reasoning as
    ``rebot_b601dm_verification.kinematics.matrix_to_quat_xyzw``; the order here
    is wxyz because that is what the ``isaac_grasp`` schema stores (``w`` plus an
    ``xyz`` list), NOT because this package prefers it. The two orders are one
    silent 180-degree error apart.
    """
    m = np.asarray(rot, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= np.linalg.norm(q)
    if q[0] < 0.0:
        q = -q
    return tuple(round(float(v), 12) for v in q)


def _grasp_rotation(azimuth_deg: float, tilt_deg: float = 0.0) -> np.ndarray:
    """Build T_object_tcp's rotation for a HORIZONTAL side grasp at an azimuth.

    Columns of the returned matrix are the TCP's X (approach), Y (closing) and Z
    axes expressed in the object frame.

    This is a pure rotation about the can's vertical axis (+Z) by ``azimuth_deg``,
    applied to the IDENTITY frame:

        approach (X) = [cos t,  sin t, 0]   -- HORIZONTAL, into the can
        closing  (Y) = [-sin t, cos t, 0]   -- HORIZONTAL, across the diameter
        third    (Z) = [0,      0,     1]   -- vertical, unchanged

    At azimuth 0 this is exactly the identity: approach = object +X (the proven-
    reachable "gripper pointing forward" direction measured on the GPU), closing =
    object +Y. Because the approach and closing axes both stay in the ground plane
    for every azimuth, the arm keeps a horizontal side grasp (never top-down) and
    the closing axis stays perpendicular to the can axis, so the required jaw gap
    stays at the diameter -- see cylinder_support_width. ``tilt_deg`` is retained
    only for schema/record compatibility and MUST be 0: any lean off horizontal
    fails IK on this arm (measured), and a lean about the approach axis would swing
    the closing direction toward the can's 101 mm height.
    """
    if tilt_deg != 0.0:
        raise GraspSetError(
            f"side grasps must stay horizontal (tilt 0); got tilt {tilt_deg} deg. "
            f"This arm fails IK for any downward tilt (measured on the GPU)."
        )
    t = math.radians(azimuth_deg)
    c, s = math.cos(t), math.sin(t)
    approach = np.array([c, s, 0.0])
    closing = np.array([-s, c, 0.0])
    third = np.cross(approach, closing)  # = [0, 0, 1]
    return np.column_stack([approach, closing, third])


def author_grasp_set(cyl: CylinderSpec = SOUP_CAN) -> GraspSet:
    """Build the grasp set from the geometry. The YAML is generated, not typed.

    Deriving the file means the interpenetration check cannot be satisfied by a
    number that happens to be right; it is satisfied by the arithmetic that
    produced it, which the tests then re-check independently.
    """
    contact_gap = cyl.diameter_m + JAW_CONTACT_CLEARANCE_M
    contact_cmd = round(command_for_gap(contact_gap), 4)

    grasps: List[GraspSpec] = []
    plan = [(az, 0.0) for az in SIDE_GRASP_AZIMUTHS_DEG]

    for index, (azimuth, tilt) in enumerate(plan):
        rot = _grasp_rotation(azimuth, tilt)
        spec = GraspSpec(
            name=f"grasp_{index}",
            azimuth_deg=azimuth,
            tilt_deg=tilt,
            position=(0.0, 0.0, GRASP_HEIGHT_M),
            quat_wxyz=_matrix_to_quat_wxyz(rot),
            jaw_contact_m=contact_cmd,
            jaw_contact_gap_m=gap_for_command(contact_cmd),
            jaw_pregrasp_m=JAW_PREGRASP_M,
            jaw_pregrasp_gap_m=gap_for_command(JAW_PREGRASP_M),
            pregrasp_offset_m=PREGRASP_OFFSET_M,
            confidence=1.0,
        )
        grasps.append(spec)

    grasp_set = GraspSet(
        object_frame=OBJECT_FRAME,
        gripper_frame=GRIPPER_FRAME,
        cylinder=cyl,
        grasps=grasps,
    )
    validate_grasp_set(grasp_set)
    return grasp_set


def validate_grasp_set(grasp_set: GraspSet) -> None:
    """Run every grasp's :meth:`GraspSpec.check`, plus set-level checks."""
    if not grasp_set.grasps:
        raise GraspSetError("grasp set is empty")

    if grasp_set.gripper_frame != GRIPPER_FRAME:
        raise GraspSetError(
            f"grasps must be authored against {GRIPPER_FRAME!r}, not "
            f"{grasp_set.gripper_frame!r}. gripper_link is 44.3 mm away along the "
            f"approach axis (KDR-001) and would drive the jaws into the object."
        )

    names = [g.name for g in grasp_set.grasps]
    if len(set(names)) != len(names):
        raise GraspSetError(f"duplicate grasp names: {names}")

    for grasp in grasp_set.grasps:
        grasp.check(grasp_set.cylinder)

    # Every grasp must approach HORIZONTALLY: this arm fails IK for any downward
    # tilt (measured on the GPU), so a grasp whose approach axis has a vertical
    # component is unreachable and must not reach the planner. This is the gate
    # that would fire if the top-down design crept back in.
    for grasp in grasp_set.grasps:
        approach_z = float(grasp.approach_axis()[2])
        if abs(approach_z) > 1e-9:
            raise GraspSetError(
                f"{grasp.name}: approach axis {np.round(grasp.approach_axis(), 4).tolist()} "
                f"is not horizontal (z={approach_z:.4f}). This arm can only side-grasp; "
                f"any downward tilt fails IK."
            )

    # A set that is nominally multi-grasp but whose orientations are all the same
    # has the single-grasp failure mode while looking like it does not. The side
    # grasp set is deliberately small (a few near azimuths, all horizontal), so
    # require at least 2 distinct closing directions rather than the 4 the old
    # full-circle top-down set could offer.
    axes = np.array([g.closing_axis() for g in grasp_set.grasps])
    # Closing axis is a line: fold antipodal directions together before counting.
    folded = np.array([a if a[0] >= 0 or (a[0] == 0 and a[1] >= 0) else -a for a in axes])
    distinct = 1
    for i in range(1, len(folded)):
        if not any(np.allclose(folded[i], folded[j], atol=1e-6) for j in range(i)):
            distinct += 1
    if distinct < 2:
        raise GraspSetError(
            f"grasp set spans only {distinct} distinct closing directions; a set "
            f"this narrow fails whenever the arm cannot reach that orientation"
        )


def dump_grasp_yaml(grasp_set: GraspSet) -> str:
    """Serialise to the ``isaac_grasp`` schema the existing consumer reads.

    Written by hand rather than through ``yaml.dump`` for two reasons: the
    orientation is a mapping with a bare ``w`` plus an ``xyz`` list, which the
    upstream files write inline and which round-trips through PyYAML as a
    multi-line block; and the per-grasp gap comments are the whole point of the
    file, and a dumper drops comments.
    """
    lines: List[str] = []
    a = lines.append
    cyl = grasp_set.cylinder
    contact = grasp_set.grasps[0]

    a("# Grasp set for the reBot B601-DM parallel gripper on the NVIDIA stock soup can.")
    a("#")
    a("# GENERATED by rebot_b601dm_perception.grasps.author_grasp_set. Do not hand-edit:")
    a("#   python3 -m rebot_b601dm_perception.write_grasps")
    a("# test_grasp_authoring.py regenerates and compares this file exactly.")
    a("#")
    a("# SCHEMA is upstream's `isaac_grasp` 1.0, so the existing consumer")
    a("# (isaac_ros_manipulation_ros_python_utils.grasp_reader.GraspReader) reads this")
    a("# file unchanged. GraspReader consumes only `position` and `orientation` from")
    a("# each entry and composes  world_pose_gripper = world_pose_object @ stored.")
    a("# Everything else below is recorded for humans and for the test suite.")
    a("#")
    a("# The two upstream reference files disagree on the frame key spelling")
    a("# (`object_frame` in robotiq_2f_85_grasps_soup_can.yaml, `object_frame_link` in")
    a("# grav_grasps_soup_can.yaml) and GraspReader reads NEITHER. Both spellings are")
    a("# emitted with identical values so either convention resolves correctly.")
    a("#")
    a("# WHAT IS STORED:  T_object_gripper_tcp.")
    a("# Composed at runtime as  T_cell_tcp = T_cell_object . T_object_tcp,  so the")
    a("# file is valid wherever the can is. gripper_tcp and NOT gripper_link: they are")
    a("# 44.3 mm apart along the approach axis (docs/KDR-001-kinematic-frames.md).")
    a("#")
    a("# FRAMES")
    a(f"#   object: {cyl.name}, origin at the CENTRE OF THE BASE, axis +Z.")
    a(f"#           diameter {cyl.diameter_m} m, height {cyl.height_m} m.")
    a("#   gripper_tcp: +X is the APPROACH axis (jaws close on what lies ahead);")
    a("#                +/-Y is the CLOSING axis (jaw travel).")
    a("#   orientation is {w, xyz} -- w FIRST, per the isaac_grasp schema. The trial")
    a("#                files use xyzw. A silent swap is a 180-degree error that still")
    a("#                normalises to a unit quaternion.")
    a("#")
    a("# JAW COMMANDS ARE HALF-GAPS -- every command below is published with the gap")
    a("# it produces. Commanding 0.030 opens a 0.060 m gap, and on this 0.066 m can")
    a("# that is 3 mm of interpenetration per side. That exact bug has already shipped")
    a("# once in this project's trial data and was caught.")
    a("#")
    a(f"#   contact  {contact.jaw_contact_m} m/jaw -> gap {contact.jaw_contact_gap_m:.6f} m")
    a(f"#            can presents {cyl.diameter_m:.6f} m across -> clearance "
      f"{(contact.jaw_contact_gap_m - cyl.diameter_m) * 1000:.3f} mm total "
      f"({(contact.jaw_contact_gap_m - cyl.diameter_m) * 500:.3f} mm/side).")
    a("#            NOMINAL. The pad inner faces sit inside the collision meshes, which")
    a("#            this generator does not read, so the true gap is smaller by an")
    a("#            unmodelled pad thickness. CLOSE ON FORCE/EFFORT, not on this number.")
    a(f"#   pregrasp {contact.jaw_pregrasp_m} m/jaw -> gap {contact.jaw_pregrasp_gap_m:.6f} m")
    a("#")
    a("# NOTE the contact command is 0.0340, not the tempting rounded 0.033.")
    a("# 0.033 gives a 0.065945 m gap: 55 um narrower than the can because of")
    a("# the residual in the jaw joint origins. That is below hardware repeatability,")
    a("# but it does not satisfy a strict gap > diameter assertion.")
    a("#")
    a("# PREGRASP GEOMETRY")
    a(f"#   Offset {PREGRASP_OFFSET_M} m back along -X of the TCP (i.e. along -approach).")
    a("#   For a HORIZONTAL side grasp the approach is in the ground plane, so this")
    a("#   backs the pregrasp radially OUTWARD from the can (not upward), clear of the")
    a(f"#   {cyl.diameter_m/2:.3f} m radius with {PREGRASP_OFFSET_M - cyl.diameter_m/2:.3f} m to spare.")
    a("#   NOTE the BT's grasp_approach_offset_distance must back off along the TCP")
    a("#   approach axis too: for this robot that is [-0.15, 0, 0] (TCP -X), NOT the")
    a("#   UR/Flexiv [0, 0, -0.15] (their approach is +Z).")
    a("#")
    a("# WHY A FEW HORIZONTAL SIDE GRASPS")
    a("#   This arm can only SIDE-grasp the can, never top-down: measured on the GPU,")
    a("#   a horizontal approach plans at every roll while ANY downward tilt fails IK.")
    a("#   So every grasp here approaches HORIZONTALLY (approach axis in the ground")
    a("#   plane), closing across the can diameter, TCP on the upper body of the")
    a("#   bottom-up source can. The can is a")
    a("#   cylinder so all azimuths grasp it equally; only a FEW near azimuths are")
    a("#   offered (the rest point the gripper back through the arm base and never")
    a("#   solve). cuMotion plans the set as a goalset and picks a reachable one.")
    a(f"#   {len(SIDE_GRASP_AZIMUTHS_DEG)} horizontal grasps at azimuths "
      f"{list(SIDE_GRASP_AZIMUTHS_DEG)} deg about the can axis. Roll is fixed at 0:")
    a("#   rolling about the approach would swing the closing direction toward the")
    a(f"#   can's {cyl.height_m} m height and require a {cyl.height_m} m gap.")
    a("")
    a(f"format: {FORMAT}")
    a(f"format_version: {FORMAT_VERSION}")
    a("")
    a(f"object_frame: /{grasp_set.object_frame}")
    a(f"gripper_frame: /{grasp_set.gripper_frame}")
    a(f"object_frame_link: /{grasp_set.object_frame}")
    a(f"gripper_frame_link: /{grasp_set.gripper_frame}")
    a("")
    a("# Object dimensions, restated so a consumer can re-derive the gap check.")
    a("# test_grasp_authoring.py verifies this file against the canonical geometry.")
    a("object_geometry:")
    a(f"  name: {cyl.name}")
    a(f"  diameter_m: {cyl.diameter_m}")
    a(f"  height_m: {cyl.height_m}")
    a("  origin: base_centre")
    a("  axis: [0.0, 0.0, 1.0]")
    a("  # T_object_mesh for the centered FoundationPose mesh (top axis +Y).")
    a("  mesh_pose_in_object:")
    a(
        "    position: "
        f"[{_num(ATTACHMENT_MESH_POSITION_M[0])}, "
        f"{_num(ATTACHMENT_MESH_POSITION_M[1])}, "
        f"{_num(ATTACHMENT_MESH_POSITION_M[2])}]"
    )
    mesh_w, mesh_x, mesh_y, mesh_z = ATTACHMENT_MESH_QUAT_WXYZ
    a(
        "    orientation: "
        f"{{w: {_num(mesh_w)}, xyz: "
        f"[{_num(mesh_x)}, {_num(mesh_y)}, {_num(mesh_z)}]}}"
    )
    a("")
    a("grasps:")
    for grasp in grasp_set.grasps:
        w, x, y, z = grasp.quat_wxyz
        required = grasp.required_gap_m(cyl)
        pregrasp_pos = grasp.pregrasp_position()
        a(f"  {grasp.name}:")
        a(f"    confidence: {grasp.confidence}")
        a(f"    position: [{_num(grasp.position[0])}, {_num(grasp.position[1])}, "
          f"{_num(grasp.position[2])}]")
        a(f"    orientation: {{w: {_num(w)}, xyz: [{_num(x)}, {_num(y)}, {_num(z)}]}}")
        a("    cspace_position:")
        a(f"      gripper_joint1: {grasp.jaw_contact_m}   # gap "
          f"{grasp.jaw_contact_gap_m:.6f} m")
        a(f"      gripper_joint2: {grasp.jaw_contact_m}   # mimic of gripper_joint1, "
          f"multiplier +1.0")
        a("    pregrasp_cspace_position:")
        a(f"      gripper_joint1: {grasp.jaw_pregrasp_m}   # gap "
          f"{grasp.jaw_pregrasp_gap_m:.6f} m")
        a(f"      gripper_joint2: {grasp.jaw_pregrasp_m}   # mimic of gripper_joint1, "
          f"multiplier +1.0")
        a("    # ---- consumed by rebot_b601dm_perception, ignored by GraspReader ----")
        a("    rebot:")
        a(f"      azimuth_deg: {grasp.azimuth_deg}")
        a(f"      tilt_deg: {grasp.tilt_deg}")
        a(f"      jaw_contact_m: {grasp.jaw_contact_m}")
        a(f"      jaw_contact_gap_m: {_num(grasp.jaw_contact_gap_m)}")
        a(f"      jaw_pregrasp_m: {grasp.jaw_pregrasp_m}")
        a(f"      jaw_pregrasp_gap_m: {_num(grasp.jaw_pregrasp_gap_m)}")
        a(f"      pregrasp_offset_m: {grasp.pregrasp_offset_m}")
        a("      # Width the can presents along THIS grasp's closing axis. Equal to the")
        a("      # diameter only because the closing axis is perpendicular to the can")
        a("      # axis; an axial grasp would need the 0.101 m height instead.")
        a(f"      required_gap_m: {_num(required)}")
        a(f"      gap_clearance_m: {_num(grasp.jaw_contact_gap_m - required)}")
        a(f"      approach_axis_object: [{_num(grasp.approach_axis()[0])}, "
          f"{_num(grasp.approach_axis()[1])}, {_num(grasp.approach_axis()[2])}]")
        a(f"      closing_axis_object: [{_num(grasp.closing_axis()[0])}, "
          f"{_num(grasp.closing_axis()[1])}, {_num(grasp.closing_axis()[2])}]")
        a(f"      pregrasp_position_object: [{_num(pregrasp_pos[0])}, "
          f"{_num(pregrasp_pos[1])}, {_num(pregrasp_pos[2])}]")
        a("")
    return "\n".join(lines).rstrip("\n") + "\n"


def _num(value: float) -> str:
    """Fixed 9-decimal rendering, with -0.0 normalised to 0.0.

    The file is compared byte-for-byte by the test suite, and ``repr`` of a float
    that lands on -0.0 through a cosine of 90 degrees would make the output
    depend on floating-point sign noise.
    """
    out = f"{float(value):.9f}"
    return "0.000000000" if out == "-0.000000000" else out


def load_grasp_set(path, cyl: CylinderSpec = SOUP_CAN) -> GraspSet:
    """Load and fully validate a grasp YAML.

    Validation is not optional on load: an unvalidated loader would let a
    hand-edited narrow gap reach the planner, which is the whole failure this
    module exists to prevent.
    """
    text = Path(path).read_text(encoding="utf-8")
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict):
        raise GraspSetError(f"{path}: not a YAML mapping")
    if doc.get("format") != FORMAT:
        raise GraspSetError(f"{path}: format is {doc.get('format')!r}, expected {FORMAT!r}")

    raw = doc.get("grasps")
    if not isinstance(raw, dict) or not raw:
        raise GraspSetError(f"{path}: no grasps")

    def frame(*keys) -> str:
        for key in keys:
            if key in doc:
                return str(doc[key]).lstrip("/")
        raise GraspSetError(f"{path}: missing any of {keys}")

    grasps: List[GraspSpec] = []
    for name, entry in raw.items():
        extra = entry.get("rebot") or {}
        # Fall back to the cspace block when `rebot` is absent, so a file written
        # to the bare upstream schema still loads -- but derive the gap rather
        # than defaulting it, so a missing gap can never read as "wide enough".
        contact = extra.get("jaw_contact_m")
        if contact is None:
            contact = (entry.get("cspace_position") or {}).get("gripper_joint1")
        pregrasp = extra.get("jaw_pregrasp_m")
        if pregrasp is None:
            pregrasp = (entry.get("pregrasp_cspace_position") or {}).get("gripper_joint1")
        if contact is None or pregrasp is None:
            raise GraspSetError(f"{path}:{name}: missing jaw commands")

        orientation = entry.get("orientation") or {}
        if "w" not in orientation or "xyz" not in orientation:
            raise GraspSetError(f"{path}:{name}: orientation must be {{w, xyz}}")

        grasps.append(GraspSpec(
            name=str(name),
            azimuth_deg=float(extra.get("azimuth_deg", float("nan"))),
            tilt_deg=float(extra.get("tilt_deg", float("nan"))),
            position=tuple(float(v) for v in entry["position"]),
            quat_wxyz=(float(orientation["w"]), *(float(v) for v in orientation["xyz"])),
            jaw_contact_m=float(contact),
            jaw_contact_gap_m=float(extra.get("jaw_contact_gap_m", gap_for_command(float(contact)))),
            jaw_pregrasp_m=float(pregrasp),
            jaw_pregrasp_gap_m=float(
                extra.get("jaw_pregrasp_gap_m", gap_for_command(float(pregrasp)))
            ),
            pregrasp_offset_m=float(extra.get("pregrasp_offset_m", PREGRASP_OFFSET_M)),
            confidence=float(entry.get("confidence", 1.0)),
        ))

    grasp_set = GraspSet(
        object_frame=frame("object_frame", "object_frame_link"),
        gripper_frame=frame("gripper_frame", "gripper_frame_link"),
        cylinder=cyl,
        grasps=grasps,
    )
    validate_grasp_set(grasp_set)
    return grasp_set


def compose_cell_tcp(t_cell_object: np.ndarray, grasp: GraspSpec) -> np.ndarray:
    """T_cell_tcp = T_cell_object . T_object_tcp.

    The same composition ``GraspReader`` performs, reproduced here so the test
    suite can check a composed pose without importing ROS message types.
    """
    t_cell_object = np.asarray(t_cell_object, dtype=float)
    if t_cell_object.shape != (4, 4):
        raise GraspSetError("T_cell_object must be 4x4")
    t_object_tcp = np.eye(4)
    t_object_tcp[:3, :3] = grasp.rotation()
    t_object_tcp[:3, 3] = np.asarray(grasp.position, dtype=float)
    return t_cell_object @ t_object_tcp
