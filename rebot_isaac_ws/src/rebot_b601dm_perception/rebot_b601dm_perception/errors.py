"""Typed failures for the perception stack.

Distinct types rather than bare ValueError because several of these are the
POINT of a gate: a test that asserts "this is refused" has to be able to say
*which* refusal it expected. ``pytest.raises(ValueError)`` would pass on a typo
in the test's own setup, which is exactly the vacuous-pass failure mode this
package is trying to avoid.
"""

from __future__ import annotations


class PerceptionError(Exception):
    """Base for everything raised by this package."""


class GraspSetError(PerceptionError):
    """A grasp set is malformed, or a grasp would drive the jaws into the object.

    The interpenetration case is the one that has already shipped once: a jaw
    command read as a gap rather than a half-gap.
    """


class MapGateError(PerceptionError):
    """The segmenter/nvblox configuration cannot satisfy the mapping gate."""


class ObjectStateError(PerceptionError):
    """The target object is in both {world, attached} or in neither.

    Both halves matter. In both -> the ESDF blocks the approach to an object the
    planner also thinks it is holding. In neither -> the object has vanished from
    the collision world and the arm will happily plan through it.
    """


class ReachabilityMisuseError(PerceptionError):
    """The IK parity oracle was asked to stand in for collision validation.

    Raised by construction, not by convention -- see ``reachability`` for why a
    comment alone was judged insufficient.
    """


class ExtrinsicSchemaError(PerceptionError):
    """An extrinsics document is missing required keys or has the wrong profile."""


class AssetPathError(PerceptionError):
    """An asset path could not be resolved to a concrete filesystem path.

    Its own type because the failure it guards is specifically NOT "file missing":
    rcl performs no variable expansion in a ``--params-file``, so a value written
    as ``${ISAAC_ROS_WS}/...`` reaches the node as that literal text and fails when
    the node opens it -- at attach time, mid-pick. Raising early, at launch, is the
    whole point.
    """
