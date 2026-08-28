"""Jaw command <-> gap arithmetic for the B601-DM parallel gripper.

WHY THIS IS A MODULE AND NOT A CONSTANT
A jaw command is a HALF-GAP. Commanding 0.030 opens the jaws to a 0.060 m gap,
not 0.030 m. Reading a command as a gap has already shipped once in this project:
0.030 was published as the contact command for a 0.066 m can, which drives each
jaw 3 mm into the can. It was caught, and the rule that came out of it is that no
jaw command is ever published without its resulting gap alongside it.

So the conversion lives in one place, is derived from the xacro geometry rather
than from the doubling shortcut, and every grasp record carries both numbers.

THE GEOMETRY (from rebot_b601dm_gripper.xacro)
Two prismatic joints on gripper_link, origins antiparallel:

    gripper_joint1  xyz=(-0.042091, +2.7531e-05, -1.3031e-05)  rpy=(0, 0, -1.5708)
    gripper_joint2  xyz=(-0.042091, -2.7531e-05, +1.3031e-05)  rpy=(0, 0, +1.5708)

Both axes are local +X. Rz(-90) maps +X to -Y and Rz(+90) maps +X to +Y, so equal
POSITIVE joint values separate the jaws symmetrically -- which is why the mimic
multiplier is +1.0 even though the jaws move in opposite spatial directions.

The residual offset SUBTRACTS, and this is the part worth being careful about.
Joint 1 starts at y = +27.5 um and travels towards -Y; joint 2 starts at
y = -27.5 um and travels towards +Y. So at v = 0 the two jaw frames are CROSSED
OVER by 55.06 um, and the first 27.5 um of travel each is spent undoing that
crossover rather than opening a gap. Writing this as ``2v + residual`` -- the
obvious reading of "the origins are 61 um apart" -- gets the sign wrong and
over-reports the gap by 116 um at every command:

    2v + 61um    at v = 0.0715  ->  0.143061 m
    exact model  at v = 0.0715  ->  0.142945 m

Separating the Y crossover from the 26.06 um Z offset, which is perpendicular to
the travel and therefore adds in quadrature rather than linearly:

    gap(v) = sqrt( (2v - 55.06um)^2 + (26.06um)^2 )

    gap(0)      = 0.000061 m
    gap(0.0715) = 0.142945 m

The focused grasp-authoring test pins this geometry through the generated grasp
file so the shortcut cannot creep back in.

THE INVERSE IS TWO-BRANCHED, which is a consequence of the crossover and not a
defect. Every gap below 60.9 um is produced by two commands: one with the jaws
still crossed (v < 27.53 um) and one with them opening (v > 27.53 um).
``command_for_gap`` always returns the opening branch, because the crossed branch
is not a grasp. So ``command_for_gap(gap_for_command(v)) == v`` holds for every
v >= 27.53 um -- which is every command any grasp will ever use -- and NOT at
v = 0, where it returns 55.06 um instead. The tests assert the identity on the
opening branch and assert the v = 0 exception separately rather than choosing a
tolerance loose enough to hide it.

CONSEQUENCE, and it matters for the grasp set: gap(0.033) = 0.065945 m, which is
55 um NARROWER than the 0.066 m soup can, not 116 um wider as the shortcut would
claim. A command of 0.033 is a rounded 0.065945 m gap. That 55 um deficit is
inside hardware repeatability, but a sign error here still flips the direction
of the interpenetration check.

NOMINAL, NOT ACTUAL. This is the geometry of the jaw LINK ORIGINS. The pad inner
faces sit somewhere inside the collision meshes, which this module does not read,
so the true contact gap is smaller than gap() reports by an unmodelled pad
thickness. That makes gap() an UPPER bound on free space and therefore the
conservative direction for an interpenetration check: if gap() already says the
jaws are inside the can, they certainly are. Close on force/effort, never on
these numbers.
"""

from __future__ import annotations

import math

# Position limits, verbatim from the xacro's jaw_lower / jaw_upper.
JAW_LOWER_M = 0.0
JAW_UPPER_M = 0.0715

# Jaw joint origins and yaws, verbatim from rebot_b601dm_gripper.xacro.
_JOINT1_ORIGIN = (-0.042091, 2.7531e-05, -1.3031e-05)
_JOINT2_ORIGIN = (-0.042091, -2.7531e-05, 1.3031e-05)
_JOINT1_YAW_RAD = -1.5708
_JOINT2_YAW_RAD = 1.5708

def _travel_direction(yaw_rad: float) -> tuple:
    """Spatial direction of a jaw's local +X axis after its origin's Rz(yaw)."""
    return (math.cos(yaw_rad), math.sin(yaw_rad), 0.0)


# Separation as a function of the command, built vectorially:
#
#     p2(v) - p1(v) = (o2 - o1) + v * (d2 - d1)
#
# Both signs come out of the arithmetic rather than being reasoned about, which is
# the point -- the first attempt at this module wrote the along-travel term by
# hand and got its sign backwards twice.
_D1 = _travel_direction(_JOINT1_YAW_RAD)
_D2 = _travel_direction(_JOINT2_YAW_RAD)
_OFFSET = tuple(_JOINT2_ORIGIN[i] - _JOINT1_ORIGIN[i] for i in range(3))
_OPENING = tuple(_D2[i] - _D1[i] for i in range(3))  # ~= (0, 2, 0)


def _separation(command_m: float) -> float:
    """Exact distance between the two jaw link origins at a given command.

    Derived from the joint origins and axes, because the ``2v + residual``
    shortcut gets the sign of the along-travel term wrong -- see the module
    docstring.
    """
    v = float(command_m)
    return math.dist(
        (0.0, 0.0, 0.0),
        tuple(_OFFSET[i] + v * _OPENING[i] for i in range(3)),
    )


# The same separation decomposed onto the opening direction, for the closed-form
# inverse. ``_ALONG_M`` is NEGATIVE: the jaw frames start crossed over by 55 um
# (joint 1 sits at y = +27.5 um and travels towards -Y, joint 2 the mirror), so
# the first 55 um of total travel undoes the crossover rather than opening a gap.
_OPENING_NORM = math.dist((0.0, 0.0, 0.0), _OPENING)
_OPENING_UNIT = tuple(c / _OPENING_NORM for c in _OPENING)
_ALONG_M = sum(_OFFSET[i] * _OPENING_UNIT[i] for i in range(3))
_PERP_M = math.sqrt(max(0.0, sum(c * c for c in _OFFSET) - _ALONG_M * _ALONG_M))


# Endpoints pinned indirectly by the generated grasp-file contract.
GAP_AT_ZERO_M = _separation(0.0)
GAP_AT_FULL_TRAVEL_M = _separation(JAW_UPPER_M)


def _check_command(command_m: float) -> None:
    if not math.isfinite(command_m):
        raise ValueError(f"jaw command must be finite, got {command_m!r}")
    if command_m < JAW_LOWER_M - 1e-12 or command_m > JAW_UPPER_M + 1e-12:
        raise ValueError(
            f"jaw command {command_m} m is outside the prismatic limit "
            f"[{JAW_LOWER_M}, {JAW_UPPER_M}] m. This is a HALF-GAP, not a gap: "
            f"if you meant a {command_m} m opening, command {command_m / 2.0} m."
        )


def gap_for_command(command_m: float) -> float:
    """Resulting jaw gap, in metres, for a per-jaw HALF-GAP command.

    The whole point of the module. Never publish a command without calling this.
    """
    _check_command(command_m)
    return _separation(command_m)


def command_for_gap(gap_m: float) -> float:
    """Inverse of :func:`gap_for_command`. Exact, including both micron offsets.

    Inverts the quadrature, and takes the OPENING branch of the square root: the
    same gap is reachable at a small negative ``along`` (jaws still crossed) as at
    a positive one, and the crossed branch is not a real grasp.
    """
    if not math.isfinite(gap_m):
        raise ValueError(f"gap must be finite, got {gap_m!r}")
    gap = float(gap_m)
    if gap < abs(_PERP_M):
        raise ValueError(
            f"gap {gap} m is unreachable: the jaw frames are offset by "
            f"{abs(_PERP_M)} m perpendicular to the travel, so no command closes "
            f"them nearer than that."
        )
    along = math.sqrt(gap * gap - _PERP_M * _PERP_M)
    command = (along - _ALONG_M) / _OPENING_NORM
    _check_command(command)
    return command


def describe(command_m: float) -> str:
    """One-line 'command -> gap' string, for logs and YAML comments.

    Exists so that the "always publish the gap with the command" rule is easy to
    obey at every call site instead of being re-derived (or forgotten).
    """
    return f"jaw command {command_m:.4f} m per jaw -> gap {gap_for_command(command_m):.4f} m"
