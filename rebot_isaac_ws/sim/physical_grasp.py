#!/usr/bin/env python3
"""USD authoring helpers for the contact-driven soup-can grasp.

The module keeps its ``pxr`` imports inside functions so its geometry and
physics constants remain testable with ordinary Python outside Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


@dataclass(frozen=True)
class CanColliderSpec:
    """Canonical collision and mass properties for the YCB soup can."""

    radius_m: float
    height_m: float
    mass_kg: float

    def __post_init__(self) -> None:
        values = (self.radius_m, self.height_m, self.mass_kg)
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("can collider dimensions and mass must be positive")

    @property
    def center_z_m(self) -> float:
        return self.height_m / 2.0


CAN_COLLIDER = CanColliderSpec(
    radius_m=0.033,
    height_m=0.101,
    mass_kg=0.349,
)
CAN_COLLIDER_NAME = "physics_collider"

STATIC_FRICTION = 1.2
DYNAMIC_FRICTION = 1.1
RESTITUTION = 0.0
PHYSICS_MATERIAL_PATH = "/World/physics_materials/grip"

CONTACT_OFFSET_M = 0.001
REST_OFFSET_M = 0.0
PHYSICS_HZ = 120.0
GRIPPER_MAX_EFFORT_N = 60.0
JAW_DRIVE_STIFFNESS_N_PER_M = 5000.0
JAW_DRIVE_DAMPING_N_S_PER_M = 41.28
MIMIC_NATURAL_FREQUENCY_HZ = 1000.0
MIMIC_DAMPING_RATIO = 1.0

SOLVER_POSITION_ITERATIONS = 32
SOLVER_VELOCITY_ITERATIONS = 4

FINGER_LINK_NAMES = ("gripper_left", "gripper_right")
JAW_JOINT_NAMES = ("gripper_joint1", "gripper_joint2")
FOLLOWER_JAW_JOINT = "gripper_joint2"
GRIPPER_COMMAND_TOPIC = "/sim_gripper/joint_command"


def mimic_api_instances(applied_schemas: list[str]) -> tuple[str, ...]:
    """Extract PhysX mimic API instance names from applied schema tokens."""
    prefix = "PhysxMimicJointAPI:"
    return tuple(
        schema[len(prefix):]
        for schema in applied_schemas
        if schema.startswith(prefix) and schema[len(prefix):]
    )


def create_grip_material(stage: Any, path: str = PHYSICS_MATERIAL_PATH) -> Any:
    """Create the shared high-friction PhysX material."""
    from pxr import Sdf, UsdGeom, UsdPhysics, UsdShade

    parent = str(Sdf.Path(path).GetParentPath())
    if parent:
        UsdGeom.Scope.Define(stage, parent)
    material = UsdShade.Material.Define(stage, path)
    physics = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics.CreateStaticFrictionAttr(STATIC_FRICTION)
    physics.CreateDynamicFrictionAttr(DYNAMIC_FRICTION)
    physics.CreateRestitutionAttr(RESTITUTION)
    return material


def bind_physics_material(prim: Any, material: Any) -> None:
    """Bind a physics material strongly enough to override referenced assets."""
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI.Apply(prim)
    binding.Bind(
        material,
        bindingStrength=UsdShade.Tokens.strongerThanDescendants,
        materialPurpose="physics",
    )


def configure_contact_offsets(prim: Any) -> None:
    """Use millimetre-scale contact generation for grasp geometry."""
    from pxr import PhysxSchema

    collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
    collision.CreateContactOffsetAttr().Set(CONTACT_OFFSET_M)
    collision.CreateRestOffsetAttr().Set(REST_OFFSET_M)


def disable_inherited_colliders(stage: Any, root: Any) -> list[str]:
    """Disable every collider composed below a referenced visual hierarchy."""
    from pxr import Usd, UsdPhysics

    disabled = []
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for prim in list(Usd.PrimRange(root, Usd.TraverseInstanceProxies())):
            if not prim.IsInstance():
                continue
            target = stage.GetPrimAtPath(prim.GetPath())
            if target and target.IsValid():
                target.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                continue
            UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr().Set(False)
            disabled.append(str(prim.GetPath()))
    return disabled


def create_can_collider(
    stage: Any,
    can_prim_path: str,
    material: Any,
    spec: CanColliderSpec = CAN_COLLIDER,
) -> Any:
    """Author one hidden analytic cylinder in the can's base-centre frame."""
    from pxr import Gf, UsdGeom, UsdPhysics

    path = f"{can_prim_path}/{CAN_COLLIDER_NAME}"
    cylinder = UsdGeom.Cylinder.Define(stage, path)
    cylinder.CreateAxisAttr(UsdGeom.Tokens.z)
    cylinder.CreateRadiusAttr(spec.radius_m)
    cylinder.CreateHeightAttr(spec.height_m)
    cylinder.CreateVisibilityAttr(UsdGeom.Tokens.invisible)

    xform = UsdGeom.Xformable(cylinder)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, spec.center_z_m))

    collision = UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    collision.CreateCollisionEnabledAttr().Set(True)
    configure_contact_offsets(cylinder.GetPrim())
    bind_physics_material(cylinder.GetPrim(), material)
    return cylinder.GetPrim()


def configure_dynamic_can(root: Any, spec: CanColliderSpec = CAN_COLLIDER) -> None:
    """Make the can a gravity-driven rigid body with no kinematic mode."""
    from pxr import PhysxSchema, UsdPhysics

    rigid_body = UsdPhysics.RigidBodyAPI.Apply(root)
    rigid_body.CreateRigidBodyEnabledAttr().Set(True)
    rigid_body.CreateKinematicEnabledAttr().Set(False)
    UsdPhysics.MassAPI.Apply(root).CreateMassAttr(spec.mass_kg)

    physx = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx.CreateDisableGravityAttr().Set(False)
    physx.CreateEnableCCDAttr().Set(True)


def refine_finger_colliders(
    stage: Any,
    robot_prim_path: str,
    material: Any,
) -> dict[str, Any]:
    """Replace solid finger hulls with decomposed finger geometry.

    Generated URDF USDs have separate ``collisions`` and ``visuals`` meshes. The
    collision STLs have already lost the concave throat, so those are disabled
    and the visible meshes become decomposed colliders. Seeed's official USD
    instead carries the full finger mesh as its authored collider; that collider
    is refined in place.

    Both asset forms use instanceable geometry, so the finger subtrees must be
    de-instanced before session-layer collision opinions can be authored.
    """
    from pxr import Usd, UsdGeom, UsdPhysics

    changed = []
    disabled = []
    deinstanced = []
    robot = stage.GetPrimAtPath(robot_prim_path)
    if not robot or not robot.IsValid():
        raise RuntimeError(f"robot prim does not resolve: {robot_prim_path}")
    matches = {
        name: [
            prim
            for prim in Usd.PrimRange(robot)
            if prim.GetName() == name
            and prim.HasAPI(UsdPhysics.RigidBodyAPI)
        ]
        for name in FINGER_LINK_NAMES
    }
    invalid = {
        name: prims for name, prims in matches.items() if len(prims) != 1
    }
    if invalid:
        raise RuntimeError(
            "expected one prim for each finger link below "
            f"{robot_prim_path}, found "
            f"{ {name: len(prims) for name, prims in invalid.items()} }"
        )
    finger_roots = [matches[name][0] for name in FINGER_LINK_NAMES]

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for root in finger_roots:
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsInstance():
                    continue
                target = stage.GetPrimAtPath(prim.GetPath())
                if target and target.IsValid() and target.SetInstanceable(False):
                    deinstanced.append(str(prim.GetPath()))

        for root in finger_roots:
            descendants = list(Usd.PrimRange(root))
            meshes = [
                prim for prim in descendants if prim.IsA(UsdGeom.Mesh)
            ]
            source_colliders = [
                prim for prim in descendants
                if "/collisions/" in str(prim.GetPath())
                and prim.HasAPI(UsdPhysics.CollisionAPI)
            ]
            visible_meshes = [
                prim for prim in meshes
                if "/visuals/" in str(prim.GetPath())
            ]
            if len(source_colliders) == 1 and len(visible_meshes) == 1:
                source = source_colliders[0]
                source_collision = UsdPhysics.CollisionAPI(source)
                source_collision.CreateCollisionEnabledAttr().Set(False)
                disabled.append(str(source.GetPath()))

                replacements = [(visible_meshes[0], str(source.GetPath()))]
            else:
                # The official Seeed asset has no collisions/visuals split. Its
                # full finger meshes already carry MeshCollisionAPI, so preserve
                # those PBR meshes and refine their collision approximation.
                vendor_colliders = [
                    prim
                    for prim in descendants
                    if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                ]
                if not vendor_colliders:
                    raise RuntimeError(
                        "expected generated collision/visual meshes or an "
                        f"authored vendor mesh collider below {root.GetPath()}"
                    )
                replacements = [
                    (prim, str(prim.GetPath())) for prim in vendor_colliders
                ]

            for replacement, replaces in replacements:
                collision = UsdPhysics.CollisionAPI.Apply(replacement)
                collision.CreateCollisionEnabledAttr().Set(True)
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(replacement)
                approximation = mesh_collision.GetApproximationAttr()
                before = approximation.Get() if approximation else None
                mesh_collision.CreateApproximationAttr().Set(
                    "convexDecomposition"
                )
                configure_contact_offsets(replacement)
                bind_physics_material(replacement, material)
                changed.append({
                    "prim": str(replacement.GetPath()),
                    "before": str(before),
                    "replaces": replaces,
                })

            if len(replacements) < 1:
                raise RuntimeError(
                    f"no finger collider replacement below {root.GetPath()}"
                )

    return {
        "changed_count": len(changed),
        "changed": changed,
        "disabled_count": len(disabled),
        "disabled": disabled,
        "deinstanced_count": len(deinstanced),
    }


def require_runtime_jaw_mimic(
    stage: Any,
    robot_prim_path: str,
    follower_joint: str = FOLLOWER_JAW_JOINT,
) -> dict[str, Any]:
    """Verify the imported hardware-style follower constraint is active.

    The real gripper has one actuator. The generated USD expresses that with a
    PhysX mimic follower, so the runtime must command only the leader and retain
    this constraint. Removing it leaves joint2 present in feedback but unable to
    follow the leader as an independently driven jaw.
    """
    from pxr import Usd

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"robot prim does not resolve: {robot_prim_path}")

    matches = [
        prim for prim in Usd.PrimRange(root)
        if prim.GetName() == follower_joint
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one {follower_joint} prim, found {len(matches)}"
        )

    follower = matches[0]
    instances = mimic_api_instances(list(follower.GetAppliedSchemas()))
    if not instances:
        raise RuntimeError(
            f"{follower.GetPath()} has no applied PhysxMimicJointAPI"
        )

    return {
        "prim": str(follower.GetPath()),
        "instances": instances,
    }


def configure_independent_jaw_drives(
    stage: Any,
    robot_prim_path: str,
) -> dict[str, Any]:
    """Configure both physical prismatic jaws as force-limited drives.

    Generated USDs carry a soft mimic API, which is removed. Seeed's official
    USD already authors two independent joints, so no mimic removal is needed.
    """
    from pxr import PhysxSchema, Usd, UsdPhysics

    root = stage.GetPrimAtPath(robot_prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"robot prim does not resolve: {robot_prim_path}")
    matches = {
        name: [
            prim
            for prim in Usd.PrimRange(root)
            if prim.GetName() == name
        ]
        for name in JAW_JOINT_NAMES
    }
    invalid = {
        name: prims for name, prims in matches.items() if len(prims) != 1
    }
    if invalid:
        raise RuntimeError(
            f"could not resolve both jaw joints below {robot_prim_path}: "
            f"{ {name: len(prims) for name, prims in invalid.items()} }"
        )
    joints = {name: matches[name][0] for name in JAW_JOINT_NAMES}
    follower = joints[FOLLOWER_JAW_JOINT]
    instances = mimic_api_instances(list(follower.GetAppliedSchemas()))

    drive_values = []
    per_jaw_force = GRIPPER_MAX_EFFORT_N / len(JAW_JOINT_NAMES)
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for instance in instances:
            if not follower.RemoveAPI(
                PhysxSchema.PhysxMimicJointAPI, instance
            ):
                raise RuntimeError(
                    f"could not remove mimic API {instance} from "
                    f"{follower.GetPath()}"
                )
        for name in JAW_JOINT_NAMES:
            drive = UsdPhysics.DriveAPI.Apply(joints[name], "linear")
            drive.CreateStiffnessAttr(JAW_DRIVE_STIFFNESS_N_PER_M)
            drive.CreateDampingAttr(JAW_DRIVE_DAMPING_N_S_PER_M)
            drive.CreateMaxForceAttr(per_jaw_force)
            drive_values.append({
                "joint": name,
                "stiffness": float(drive.GetStiffnessAttr().Get()),
                "damping": float(drive.GetDampingAttr().Get()),
                "max_force": float(drive.GetMaxForceAttr().Get()),
            })

    remaining = mimic_api_instances(list(follower.GetAppliedSchemas()))
    if remaining:
        raise RuntimeError(
            f"{follower.GetPath()} still has mimic APIs after removal: "
            f"{remaining}"
        )
    return {
        "prim": str(follower.GetPath()),
        "removed_instances": instances,
        "remaining_instances": remaining,
        "drives": drive_values,
    }


def configure_runtime_jaw_mimic(
    stage: Any,
    robot_prim_path: str,
    *,
    natural_frequency_hz: float = MIMIC_NATURAL_FREQUENCY_HZ,
    damping_ratio: float = MIMIC_DAMPING_RATIO,
) -> dict[str, Any]:
    """Tune and verify the single-actuator jaw's physical mimic constraint."""
    from pxr import PhysxSchema, Usd

    frequency = float(natural_frequency_hz)
    damping = float(damping_ratio)
    if (
        not math.isfinite(frequency)
        or frequency <= 0.0
        or not math.isfinite(damping)
        or damping <= 0.0
    ):
        raise ValueError("mimic frequency and damping must be positive")

    result = require_runtime_jaw_mimic(stage, robot_prim_path)
    if len(result["instances"]) != 1:
        raise RuntimeError(
            "expected exactly one jaw mimic API instance, found "
            f"{result['instances']}"
        )

    instance = result["instances"][0]
    follower = stage.GetPrimAtPath(result["prim"])
    mimic = PhysxSchema.PhysxMimicJointAPI(follower, instance)
    if not mimic:
        raise RuntimeError(
            f"could not construct mimic API {instance} on {result['prim']}"
        )

    prefix = f"physxMimicJoint:{instance}"
    frequency_attr = follower.GetAttribute(f"{prefix}:naturalFrequency")
    damping_attr = follower.GetAttribute(f"{prefix}:dampingRatio")
    if not frequency_attr or not damping_attr:
        raise RuntimeError(
            f"{result['prim']} is missing mimic compliance attributes"
        )
    previous_frequency = float(frequency_attr.Get())
    previous_damping = float(damping_attr.Get())

    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        frequency_attr.Set(frequency)
        damping_attr.Set(damping)

    applied_frequency = float(frequency_attr.Get())
    applied_damping = float(damping_attr.Get())
    if not math.isclose(applied_frequency, frequency) or not math.isclose(
        applied_damping, damping
    ):
        raise RuntimeError(
            "mimic tuning did not apply: "
            f"wanted=({frequency}, {damping}) "
            f"got=({applied_frequency}, {applied_damping})"
        )

    return {
        **result,
        "instance": instance,
        "reference": [
            str(path) for path in mimic.GetReferenceJointRel().GetTargets()
        ],
        "gearing": float(mimic.GetGearingAttr().Get()),
        "offset": float(mimic.GetOffsetAttr().Get()),
        "natural_frequency": applied_frequency,
        "damping_ratio": applied_damping,
        "previous_natural_frequency": previous_frequency,
        "previous_damping_ratio": previous_damping,
    }


def configure_articulation_solver(stage: Any, articulation_root_path: str) -> None:
    """Give finger/object contacts enough solver iterations to remain stable."""
    from pxr import PhysxSchema, Usd

    root = stage.GetPrimAtPath(articulation_root_path)
    if not root or not root.IsValid():
        raise RuntimeError(
            f"articulation root does not resolve: {articulation_root_path}"
        )
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        articulation = PhysxSchema.PhysxArticulationAPI.Apply(root)
        articulation.CreateEnabledSelfCollisionsAttr(False)
        articulation.CreateSolverPositionIterationCountAttr(
            SOLVER_POSITION_ITERATIONS
        )
        articulation.CreateSolverVelocityIterationCountAttr(
            SOLVER_VELOCITY_ITERATIONS
        )
