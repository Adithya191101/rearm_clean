#!/usr/bin/env python3
"""Headless Isaac Sim 5.1 pick-and-place SCENE for the reBot B601-DM. SIM ONLY.

Run on the HOST (Isaac Sim is host-side in this topology):

    source sim/isaac_sim_env.sh
    $ISAACSIM_ROOT/python.sh sim/pick_scene.py --duration 120

then, from the container, verify the topics carry data:

    ./docker/run.sh sim python3 docker/connectivity_gate.py --require-camera
    ./docker/run.sh sim python3 sim/pick_scene_probe.py   # closed-loop + depth

This is the one sim-side piece the end-to-end pick needs: nothing else publishes
the topics the ROS 2 side (driver, segmenter, nvblox, cuMotion, and orchestration
BT) already consumes. This scene loads the canonical Stage 3 robot
USD, drives its articulation from /isaac_joint_commands (the closed loop), and
publishes /clock, /isaac_joint_states, the wrist RGB-D + camera_info, /tf, and
two fixed RGB-D camera streams.

TOPIC CONTRACT (matched EXACTLY -- a wrong name is a silent no-data hang):
  PUBLISH   /clock                                       (sole owner, 60 Hz)
            /isaac_joint_states                          (all 8 joints)
            /front_stereo_camera/left/image_raw          (RGB, 1280x720)
            /front_stereo_camera/left/camera_info        (intrinsics, D435-class)
            /front_stereo_camera/depth/ground_truth      (wrist depth)
            /tf                                           (arm and camera frames)
  SUBSCRIBE /isaac_joint_commands                        (position cmds -> arm)

WHY EACH CHOICE (naming the failure it guards):

  * CUDA MPS bypass -- NOT set here; sim/isaac_sim_env.sh sets
    CUDA_MPS_PIPE_DIRECTORY. Without it Kit hangs FOREVER at device init after
    `[gpu.foundation.plugin]` with no banner/error. This script refuses to start
    if the env was not sourced (ROS_DISTRO check below), which is the same shell
    that must carry the MPS bypass.

  * ROS_DISTRO=jazzy forced -- Isaac's bridge autodetects `humble` on this 22.04
    host and publishes Humble-typed messages the Jazzy container cannot read.
    Nothing errors; the container just never receives anything. Fail loud here.

  * step(render=True) every tick -- OnPlaybackTick fires from the render loop and
    the RTX depth/RGB sensors only produce a frame on a RENDERED step. A
    physics-only step would advance the articulation while publishing nothing on
    the wire and a stale (or empty) camera frame: alive locally, silent remotely.

  * GPU dynamics OFF -- inherited from the USD's PhysicsScene (import_robot_usd.py
    sets enableGPUDynamics=False, 60 Hz). With GPU dynamics on, contact reporting
    and articulation readback go through the GPU pipeline and the jaw contact the
    gripper adapter's stall detection needs becomes unreliable. We do NOT re-author
    it -- the USD is SHA-hashed and owned by another stage -- we only assert it.

  * exact topic names -- every one is what the installed ROS-side config already
    subscribes to. `/front_stereo_camera/left/...` (not a renamed convenience
    topic) is what the Isaac ROS robot segmenter consumes.

  * rclpy is NOT imported -- Isaac Sim loads its own rclpy for the Kit interpreter
    (python 3.11). The OmniGraph nodes publish/subscribe through the C++ bridge, so
    keeping Python out of the wire path removes a whole class of "which rclpy did I
    get" failures. The joint-command TEST client lives in a separate file
    (sim/pick_scene_probe.py) that runs in the container, not here.

``--duration`` bounds the run for scripting and ``--gui`` opens a viewport. The
startup banner logs every published/subscribed topic with ``flush=True`` so it
can be diffed against the contract.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

from physical_grasp import (
    CAN_COLLIDER,
    CAN_COLLIDER_NAME,
    CONTACT_OFFSET_M,
    GRIPPER_MAX_EFFORT_N,
    GRIPPER_COMMAND_TOPIC,
    JAW_JOINT_NAMES,
    PHYSICS_HZ,
    REST_OFFSET_M,
    bind_physics_material,
    configure_articulation_solver,
    configure_dynamic_can,
    configure_independent_jaw_drives,
    create_can_collider,
    create_grip_material,
    disable_inherited_colliders,
    refine_finger_colliders,
)
from presentation_views import (
    CAMERA_HORIZONTAL_APERTURE_MM,
    MAIN_CAMERA_EYE,
    MAIN_CAMERA_EYE_OFFSET,
    MAIN_CAMERA_FOCAL_LENGTH_MM,
    MAIN_CAMERA_TARGET,
    MAIN_RECORD_SIZE,
    PERCEPTION_CAMERA_EYE,
    PERCEPTION_CAMERA_TARGET,
    WIDE_CAMERA_EYE,
    WIDE_CAMERA_FOCAL_LENGTH_MM,
    WIDE_CAMERA_TARGET,
    WIDE_RECORD_SIZE,
)
from transfer_obstacle import (
    TRANSFER_WALL,
    WALL_SAFETY_SCHEMA_VERSION,
    atomic_write_json,
    create_transfer_wall,
    finite_cylinder_aabb_clearance_bounds,
    load_planner_collision_spheres,
    minimum_sphere_aabb_clearance,
    quaternion_xyzw_rotation_matrix,
    transform_collision_spheres,
)
import vendor_robot as vendor

# ---------------------------------------------------------------------------
# Argument parsing BEFORE SimulationApp: SimulationApp consumes sys.argv and the
# Kit config is fixed once the app is up. (Same ordering constraint as the other
# sim/ scenes.)
# ---------------------------------------------------------------------------
_ap = argparse.ArgumentParser(
    description=__doc__.split("\n\n")[0],
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
_ap.add_argument("--duration", type=float, default=120.0,
                 help="seconds to run before a clean exit (scriptable)")
_ap.add_argument("--gui", action="store_true",
                 help="open a viewport (default headless)")
_ap.add_argument("--usd", default=None, help="override the robot USD path")
# Establishing-camera video capture of the whole workcell. OFF by default (adds
# render cost + disk); --record turns it on for an end-to-end run. A
# passive observer: a second world camera + rgb annotator, does NOT touch physics
# or the wrist-camera perception path.
_ap.add_argument("--record", action="store_true",
                 help="capture establishing-camera JPEG frames of the full run")
_ap.add_argument("--record-dir", default="/tmp/rebot_pick_place_frames",
                 help="directory for captured JPEG frames")
_ap.add_argument("--record-wide-dir", default="",
                 help="optional directory for the independent high-wide view")
_ap.add_argument("--record-every", type=int, default=2,
                 help="capture one frame every N sim steps (2 => ~30 fps at 60 Hz)")
_ap.add_argument("--record-width", type=int, default=MAIN_RECORD_SIZE[0],
                 help="observer frame width (default: 1920)")
_ap.add_argument("--record-height", type=int, default=MAIN_RECORD_SIZE[1],
                 help="observer frame height (default: 1080)")
_ap.add_argument("--record-wide-width", type=int, default=WIDE_RECORD_SIZE[0],
                 help="high-wide observer frame width (default: 640)")
_ap.add_argument("--record-wide-height", type=int, default=WIDE_RECORD_SIZE[1],
                 help="high-wide observer frame height (default: 603)")
_ap.add_argument("--record-jpeg-quality", type=int, default=92,
                 help="observer JPEG quality in [1, 100] (default: 92)")
_ap.add_argument(
    "--record-state-file",
    default="",
    help="optional atomic JSON output with live transfer-wall safety telemetry",
)
_ap.add_argument("--record-focus-scene-cam", action="store_true",
                 help="aim the recording observer at the visible scene_cam_0 "
                      "fixture instead of the full manipulation workcell")
_ap.add_argument(
    "--transfer-wall-command-file",
    default="",
    help="optional JSON command file with {'wall_visible': bool}; visibility "
         "changes affect the RGB-D map but leave the PhysX collider enabled",
)
_args, _unknown = _ap.parse_known_args()

# ---------------------------------------------------------------------------
# Fail fast on the one misconfiguration that silently breaks everything: the
# bridge falling back to Humble on this 22.04 host. sim/isaac_sim_env.sh explains
# why, and sets the CUDA MPS bypass in the same breath -- so this check doubles as
# a "did you source the env" gate.
# ---------------------------------------------------------------------------
if os.environ.get("ROS_DISTRO") != "jazzy":
    sys.exit(
        "ERROR: ROS_DISTRO is %r, expected 'jazzy'.\n"
        "       Isaac Sim would autodetect 'humble' from this host's Ubuntu 22.04\n"
        "       and publish Humble-typed messages the Jazzy container cannot read.\n"
        "       (That same env also sets the CUDA_MPS_PIPE_DIRECTORY bypass without\n"
        "        which Kit hangs forever at device init.)\n"
        "       Run: source sim/isaac_sim_env.sh"
        % os.environ.get("ROS_DISTRO", "<unset>")
    )

WS_DIR = Path(__file__).resolve().parent.parent
PLANNER_XRDF_PATH = WS_DIR / "config" / "xrdf" / "rebot_b601dm_gripper.xrdf"
VENDOR_ASSET_PATH = vendor.ASSET_PATH
USD_PATH = _args.usd or str(VENDOR_ASSET_PATH)
USING_VENDOR_ASSET = (
    Path(USD_PATH).expanduser().resolve() == VENDOR_ASSET_PATH.resolve()
)
#: NVIDIA's real textured YCB tomato-soup can (005), streamed directly from the
#: Isaac 5.1 asset S3 bucket -- gives a photo-real red-and-white Campbell's-style
#: label instead of the bare grey .obj. Same physical extent (0.068 x 0.102 x
#: 0.068 m, Y-up in its own space) so the grasp gap budget is unchanged.
CAN_YCB_URL = ("https://omniverse-content-production.s3-us-west-2.amazonaws.com"
               "/Assets/Isaac/5.1/Isaac/Props/YCB/Axis_Aligned_Physics/"
               "005_tomato_soup_can.usd")
# Official Isaac Sim Intel RealSense D455. The complete default prim is
# referenced so its geometry and sibling material library compose together.
# Its embedded sensors and inherited physics are disabled after registering the
# housing to the authored color imager. The custom Camera prims below retain the
# already-validated intrinsics, topics, frame ids, and optical poses.
REALSENSE_D455_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com"
    "/Assets/Isaac/5.1/Isaac/Sensors/Intel/RealSense/rsd455.usd"
)
REALSENSE_D455_MODEL_PRIM = "RSD455"
REALSENSE_D455_COLOR_CAMERA = "Camera_OmniVision_OV9782_Color"
REALSENSE_D455_EMBEDDED_CAMERAS = (
    "Camera_Pseudo_Depth",
    REALSENSE_D455_COLOR_CAMERA,
    "Camera_OmniVision_OV9782_Left",
    "Camera_OmniVision_OV9782_Right",
)

if not Path(USD_PATH).is_file():
    sys.exit("ERROR: robot USD not found at %s" % USD_PATH)
if USING_VENDOR_ASSET:
    _asset_sha256 = vendor.asset_sha256(Path(USD_PATH))
    if _asset_sha256 != vendor.OFFICIAL_ROOT_LAYER_SHA256:
        sys.exit(
            "ERROR: vendor robot USD checksum mismatch at %s\n"
            "       actual:   %s\n"
            "       expected: %s"
            % (
                USD_PATH,
                _asset_sha256,
                vendor.OFFICIAL_ROOT_LAYER_SHA256,
            )
        )

# ---------------------------------------------------------------------------
# Prim paths and the topic contract, in one place so the banner and the graph
# cannot drift from each other.
# ---------------------------------------------------------------------------
ROBOT_PRIM = (
    vendor.ROBOT_PRIM_PATH if USING_VENDOR_ASSET else "/rebot_b601dm"
)
# The production camera chain is authored below the vendor's moving link6 using
# the same extrinsics_sim.yaml contract as the URDF. A custom --usd remains a
# diagnostics-only path and is expected to carry the legacy camera prim.
CAMERA_PRIM = (
    vendor.CAMERA_PRIM_PATH
    if USING_VENDOR_ASSET
    else "%s/camera_color_optical_frame/rgbd_camera" % ROBOT_PRIM
)
CAN_PRIM = "/World/soup_can"
WORKTOP_PRIM = "/World/worktop"

TOPIC_CLOCK = "/clock"
TOPIC_JOINT_STATES = "/isaac_joint_states"
TOPIC_JOINT_COMMANDS = "/isaac_joint_commands"
TOPIC_GRIPPER_COMMAND = GRIPPER_COMMAND_TOPIC
TOPIC_RGB = "/front_stereo_camera/left/image_raw"
TOPIC_CAMERA_INFO = "/front_stereo_camera/left/camera_info"
TOPIC_DEPTH = "/front_stereo_camera/depth/ground_truth"

# --- static tabletop RGB-D cameras --------------------------------------------
# Two fixed cameras at diagonally-opposite corners of the workspace, looking in
# at the pick/place area. These are the two active mapping streams and provide a
# continuous view of the object for Grounding DINO + FoundationPose while the
# wrist camera swings away during approach.
# Each publishes rgb + depth + camera_info under its own namespace and a TF frame.
# frame == the prim LEAF name: ROS2PublishTransformTree names the child frame
# after the prim leaf, and the CameraHelper tags images with `frame`, so making
# them identical means every published image has a resolvable TF (the same
# construction the wrist camera uses: prim leaf `rgbd_camera` under
# `camera_color_optical_frame`, frame id `camera_color_optical_frame`). Here the
# prim leaf IS the frame, so images and TF agree by name.
SCENE_CAMS = [
    {  # near-right corner, looking back at the workspace
        "name": "scene_cam_0",
        "prim": "/World/scene_cam_0",
        "frame": "scene_cam_0",
        "eye": PERCEPTION_CAMERA_EYE,
        "target": PERCEPTION_CAMERA_TARGET,
        "ns": "/scene_cam_0",
        "visible_fixture": True,
    },
    {  # far-left corner, diagonally opposite
        "name": "scene_cam_1",
        "prim": "/World/scene_cam_1",
        "frame": "scene_cam_1",
        "eye": (0.05, 0.55, 0.70),
        "target": (0.37, 0.06, 0.15),
        "ns": "/scene_cam_1",
        "visible_fixture": True,
    },
]
SCENE_CAM_W, SCENE_CAM_H = 640, 480
SCENE_CAM_FRAME_SKIP = 3

RECORD_CAMERA_TARGET = MAIN_CAMERA_TARGET
RECORD_CAMERA_EYE_OFFSET = MAIN_CAMERA_EYE_OFFSET
RECORD_CAMERA_FOCAL_LENGTH_MM = MAIN_CAMERA_FOCAL_LENGTH_MM

# The frame the perception side publishes poses/depth against. The wrist depth is
# ground-truth and carries no CameraInfo of its own, so the segmenter pairs
# `depth/ground_truth` with `left/camera_info`; both therefore use the optical
# frame id the USD camera is parented under.
CAMERA_FRAME_ID = "camera_color_optical_frame"

# ---------------------------------------------------------------------------
# The joint state and command joint ORDER. Isaac Sim's articulation order comes
# from the USD and need not match; we publish ALL eight joints (6 arm + both
# mimic jaws) because the parser node downsamples /isaac_joint_states, and the
# ros2_control TopicBasedSystem commands arrive on /isaac_joint_commands for the
# six arm joints. Both force-limited jaw drives receive the same target. This
# replaces the imported compliant mimic, which tracks unloaded but fails to
# transmit bilateral contact force under the can's load.
# ---------------------------------------------------------------------------
ARM_JOINTS = ["joint%d" % i for i in range(1, 7)]
JAW_JOINTS = ["gripper_joint1", "gripper_joint2"]
ALL_JOINTS = ARM_JOINTS + JAW_JOINTS

#: Jaw command position (metres per jaw). Open is the URDF upper limit. Close
#: commands come from the ROS bridge and intentionally pass the can's nominal
#: contact width; PhysX contact, not a command clamp, stops the fingers.
JAW_OPEN_M = 0.0715
JAW_COMMAND_JOINTS = JAW_JOINT_NAMES

# Horizontal-gripper startup pose (radians, arm joints). The base is yawed away
# from the can and zero final roll keeps the open gripper rail parallel to the
# worktop. FK places the TCP at approximately (0.237, -0.162, 0.372): 62 mm
# above the wall top and safely away from the can.
START_Q = {
    "joint1": -0.6000, "joint2": -0.7500, "joint3": -0.7500,
    "joint4": 0.0000, "joint5": 0.0000, "joint6": 0.0000,
}
# The imported arm drives have enough stiffness but too little damping. The
# leader-jaw gains are the values validated by sim/vendor_pick.py for a physical
# squeeze; the mimic follower remains constraint-driven with its authored gains.
ARM_DAMPING = [1000.0, 1800.0, 1400.0, 300.0, 180.0, 120.0]
JAW_STIFFNESS = 5000.0
JAW_DAMPING = 41.28
# Pick and place areas, carved from the MEASURED reachable band and shared with
# the grasp-reachability harness via sim/pick_area.py (single source of truth, no
# Isaac import there so the container-side verifier can import it too). Re-exported
# here so a later random-in-area spawn wires to the SAME bounds that were plan-
# verified corner-by-corner by the wall-safety tests.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pick_area import (  # noqa: E402
    PICK_AREA, PLACE_AREA, pick_centre, place_centre,
)

# Can placement, chosen for REACHABILITY (measured on the GPU, not assumed).
# The earlier (-0.413, 0.0) at floor z=0 was BEHIND the base and on the floor --
# cuMotion could not plan to it from any orientation. The arm can only SIDE-grasp
# (horizontal approach; any downward tilt fails IK), so the reachable front zone
# for the grasp TCP is x in [0.30,0.45], z in [0.20,0.30]. The can starts
# bottom-up and the source-relative grasp composes to table + 0.09 = world
# z=0.24, clear of the mapped worktop collision envelope. The can is spawned at
# the PICK_AREA centre (fixed for this first run; PICK_AREA is ready for random
# sampling later). All three authored grasps + all four PICK_AREA corners plan
# here (verified on the GPU).
CAN_XY = (pick_centre()[0], pick_centre()[1])
CAN_BASE_Z = PICK_AREA["z"]  # support surface at world z=0.15 (see add_worktop)

# ---------------------------------------------------------------------------
# Launch Kit. Offline/hermetic flags via sys.argv (SimulationApp forwards unknown
# args to Kit), same as the other sim/ scenes: every extension is already in
# extscache/, so the remote registries are only latency + a network dependency.
# These are NOT the fix for the MPS hang -- that is CUDA_MPS_PIPE_DIRECTORY in the
# sourced env.
# ---------------------------------------------------------------------------
sys.argv.extend([
    "--/app/extensions/registryEnabled=false",
    "--/persistent/app/omniverse/hubEnabled=false",
    "--/structuredLog/enable=false",
])

from isaacsim import SimulationApp  # noqa: E402  (must follow argv handling)

simulation_app = SimulationApp({"headless": not _args.gui})

import numpy as np  # noqa: E402
import omni.graph.core as og  # noqa: E402
import omni.usd  # noqa: E402
import usdrt.Sdf  # noqa: E402
from omni.physx import get_physx_simulation_interface  # noqa: E402
from omni.physx.bindings._physx import ContactEventType  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402
from isaacsim.core.utils.extensions import enable_extension  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import (  # noqa: E402
    Gf,
    PhysicsSchemaTools,
    PhysxSchema,
    Sdf,
    Usd,
    UsdGeom,
    UsdPhysics,
)

enable_extension("isaacsim.ros2.bridge")
simulation_app.update()


class TransferWallContactMonitor:
    """Track real PhysX contacts between the wall and the robot or can."""

    def __init__(self, wall_path: str, tracked_prefixes: tuple[str, ...]) -> None:
        self.wall_path = wall_path
        self.tracked_prefixes = tracked_prefixes
        self.active_pairs: set[tuple[str, str]] = set()
        self.ever = False
        self.events = 0
        self.minimum_separation_m: float | None = None
        self.last_pair: tuple[str, str] | None = None

    @staticmethod
    def _path(path_id) -> str:
        return str(PhysicsSchemaTools.intToSdfPath(path_id))

    def _is_wall(self, path: str) -> bool:
        return path == self.wall_path or path.startswith(self.wall_path + "/")

    def _is_tracked(self, path: str) -> bool:
        return any(
            path == prefix or path.startswith(prefix + "/")
            for prefix in self.tracked_prefixes
        )

    def on_contact_report(self, headers, contact_data) -> None:
        for header in headers:
            side_0 = (
                self._path(header.actor0),
                self._path(header.collider0),
            )
            side_1 = (
                self._path(header.actor1),
                self._path(header.collider1),
            )
            if not (
                (any(self._is_wall(path) for path in side_0)
                 and any(self._is_tracked(path) for path in side_1))
                or
                (any(self._is_wall(path) for path in side_1)
                 and any(self._is_tracked(path) for path in side_0))
            ):
                continue

            collider_0 = side_0[1] or side_0[0]
            collider_1 = side_1[1] or side_1[0]
            pair = tuple(sorted((collider_0, collider_1)))
            if header.type in (
                ContactEventType.CONTACT_FOUND,
                ContactEventType.CONTACT_PERSIST,
            ):
                if header.type == ContactEventType.CONTACT_FOUND:
                    self.events += 1
                self.active_pairs.add(pair)
                self.ever = True
                self.last_pair = pair
                start = int(header.contact_data_offset)
                stop = start + int(header.num_contact_data)
                for index in range(start, stop):
                    contact = contact_data[index]
                    separation = float(contact.separation)
                    if (
                        self.minimum_separation_m is None
                        or separation < self.minimum_separation_m
                    ):
                        self.minimum_separation_m = separation
            elif header.type == ContactEventType.CONTACT_LOST:
                self.active_pairs.discard(pair)

    def snapshot(self) -> dict:
        return {
            "current": bool(self.active_pairs),
            "ever": self.ever,
            "events": self.events,
            "active_pairs": [list(pair) for pair in sorted(self.active_pairs)],
            "last_pair": (
                list(self.last_pair) if self.last_pair is not None else None
            ),
            "minimum_separation_m": self.minimum_separation_m,
        }


# ---------------------------------------------------------------------------
# Stage assembly
# ---------------------------------------------------------------------------
def build_stage() -> tuple[Usd.Stage, list[str]]:
    """Reference the robot USD read-only and add scene content around it.

    The official Seeed USD is REFERENCED, never edited. Its nested rigid-body
    repair and the ROS-only TCP/camera frames are session-stage opinions applied
    before physics initializes. ``--usd`` retains the old generated asset as an
    explicit diagnostics override.
    """
    add_reference_to_stage(usd_path=USD_PATH, prim_path=ROBOT_PRIM)
    stage = omni.usd.get_context().get_stage()
    UsdGeom.Xform.Define(stage, "/World")
    repaired_body_paths: list[str] = []
    if USING_VENDOR_ASSET:
        repair = vendor.repair_vendor_stage(stage)
        repaired_body_paths = list(repair["repaired_body_paths"])
        frame_paths = vendor.author_ros_frames_and_camera(stage)
        print(
            "VENDOR ASSET: official Seeed USD sha256=%s, repaired=%d "
            "session-only links, ROS frames=%s"
            % (
                vendor.OFFICIAL_ROOT_LAYER_SHA256,
                len(repaired_body_paths),
                sorted(frame_paths),
            ),
            flush=True,
        )
    return stage, repaired_body_paths


def find_articulation_root(stage: Usd.Stage) -> str:
    """Return the prim path that carries UsdPhysics.ArticulationRootAPI.

    THIS IS NOT /rebot_b601dm. The URDF importer (fix_base=True) applies
    ArticulationRootAPI to the generated `root_joint`, so the articulation root is
    /rebot_b601dm/root_joint (see usd/rebot_b601dm_checks.json: "exactly one
    articulation root"). It matters because the OmniGraph ROS2PublishJointState
    and IsaacArticulationController nodes resolve the articulation by EXACT prim
    path -- passing /rebot_b601dm makes them log "Prim /rebot_b601dm is not an
    articulation" every tick, and in the "execution" evaluator that failing node
    aborts the whole graph tick, so even /clock goes silent. (SingleArticulation
    searches downward and does not hit this, which is why the DOFs still resolve.)

    Discovered rather than hardcoded so a future re-import that moves the API does
    not silently re-break the wire.
    """
    roots = [str(p.GetPath()) for p in stage.Traverse()
             if p.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if len(roots) == 1:
        return roots[0]
    # Fall back to the robot prim with a loud note; the graph error above will
    # then say exactly which prim was wrong.
    print("WARN: expected exactly one ArticulationRootAPI prim, found %r; "
          "falling back to %s" % (roots, ROBOT_PRIM), flush=True)
    return roots[0] if roots else ROBOT_PRIM


def enable_robot_gravity_compensation(stage: Usd.Stage) -> None:
    """Model the arm's actuator gravity compensation on robot links only."""
    robot = stage.GetPrimAtPath(ROBOT_PRIM)
    compensated = []
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for prim in Usd.PrimRange(robot):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            api.CreateDisableGravityAttr().Set(True)
            compensated.append(str(prim.GetPath()))
    if not compensated:
        raise RuntimeError("no robot rigid bodies found for gravity compensation")
    print("enabled ideal gravity compensation on %d robot rigid bodies"
          % len(compensated), flush=True)


def add_lights(stage: Usd.Stage) -> None:
    """Use the exposure proven with the official Seeed PBR materials.

    These are the cousin reference run's measured settings. They expose the
    vendor robot's green, black, and metallic surfaces while retaining enough
    workcell contrast for RGB perception.
    """
    import isaacsim.core.utils.prims as prim_utils

    prim_utils.create_prim(
        "/World/dome_light",
        "DomeLight",
        attributes={
            "inputs:intensity": 800.0,
            "inputs:color": (1.0, 1.0, 1.0),
        },
    )
    prim_utils.create_prim(
        "/World/key_light",
        "DistantLight",
        attributes={"inputs:intensity": 2200.0, "inputs:angle": 2.0},
        orientation=np.array([0.9239, 0.0, 0.3827, 0.0]),
    )


def add_worktop(stage: Usd.Stage, grip_material) -> None:
    """A thin static worktop at z=0 so the depth image has a non-degenerate
    surface behind the can and nvblox has a floor to map.

    Static (no RigidBodyAPI): it is the ground the can rests on and the arm plans
    above, not a dynamic object. A large thin cuboid keeps it out of the arm's
    swept volume while filling the wrist camera's downward view.
    """
    top = UsdGeom.Cube.Define(stage, WORKTOP_PRIM)
    top.CreateSizeAttr(1.0)
    xf = UsdGeom.Xformable(top)
    xf.ClearXformOpOrder()
    # A TABLE in front of the arm: 0.5 x 0.8 m top, 2 cm thick, top face flush
    # with z = CAN_BASE_Z (0.15 m) so the can rests on a reachable tabletop rather
    # than the floor. Centred at x=0.40 (under the can), NOT at the base -- a slab
    # under the base would swallow the arm's lower links. Kept thin and offset so
    # it stays clear of the arm's swept volume while giving the can a surface.
    xf.AddTranslateOp().Set(Gf.Vec3d(0.40, 0.0, CAN_BASE_Z - 0.01))
    xf.AddScaleOp().Set(Gf.Vec3f(0.5, 0.8, 0.02))
    UsdPhysics.CollisionAPI.Apply(top.GetPrim())
    bind_physics_material(top.GetPrim(), grip_material)


def add_area_markers(stage: Usd.Stage) -> None:
    """Draw the PICK area and PLACE drop sheet as flat worktop decals.

    Purely cosmetic: thin, un-collidable slabs so the operator can SEE which
    region is the pick area (green) and which is the drop area (blue) in the
    viewport. They carry no physics -- no CollisionAPI, no RigidBodyAPI -- so they
    cannot deflect the arm, the can, or nvblox mapping. The blue sheet is larger
    than the exact target bounds so it remains visible around the placed can.
    """
    # 1 mm proud of the worktop top face (z = CAN_BASE_Z) so they read as painted
    # regions without z-fighting the table.
    marker_z = CAN_BASE_Z + 0.001
    place_sheet_size_m = (0.12, 0.12)
    for name, area, rgb, minimum_size_m in (
        (
            "pick_area_marker",
            PICK_AREA,
            (0.10, 0.70, 0.20),
            (0.0, 0.0),
        ),
        (
            "place_area_marker",
            PLACE_AREA,
            (0.15, 0.35, 0.90),
            place_sheet_size_m,
        ),
    ):
        x0, x1 = area["x"]
        y0, y1 = area["y"]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        sx = max(x1 - x0, minimum_size_m[0])
        sy = max(y1 - y0, minimum_size_m[1])
        cube = UsdGeom.Cube.Define(stage, "/World/%s" % name)
        cube.CreateSizeAttr(1.0)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        xf = UsdGeom.Xformable(cube)
        xf.ClearXformOpOrder()
        xf.AddTranslateOp().Set(Gf.Vec3d(cx, cy, marker_z))
        xf.AddScaleOp().Set(Gf.Vec3f(float(sx), float(sy), 0.002))
        # guide purpose keeps it out of any collision/planning traversal while
        # RTX still renders it (it is not 'render'-only geometry we hide).
        UsdGeom.Imageable(cube.GetPrim()).CreatePurposeAttr(UsdGeom.Tokens.default_)
        print(
            "MARKER: %s target=x[%.2f,%.2f] y[%.2f,%.2f] "
            "visual_size=(%.2f,%.2f) rgb=%s"
            % (name, x0, x1, y0, y1, sx, sy, rgb),
            flush=True,
        )


def _box(stage: Usd.Stage, path: str, center, scale, rgb, *, collision: bool):
    """A colored box prim. Optionally collidable; env dressing is visual-only."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
    xf = UsdGeom.Xformable(cube)
    xf.ClearXformOpOrder()
    xf.AddTranslateOp().Set(Gf.Vec3d(*center))
    xf.AddScaleOp().Set(Gf.Vec3f(*scale))
    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return cube


def add_environment(stage: Usd.Stage) -> None:
    """Dress the scene as a room with the arm on a workbench -- VISUAL ONLY.

    Everything here is cosmetic and deliberately kept away from the working
    stack, on three independent guards so it can never turn into a phantom
    obstacle or a moved-robot bug:

      1. NO CollisionAPI on any prop -- cuMotion's collision path never sees them.
      2. Walls/floor sit OUTSIDE the nvblox mapped box (|x|,|y| > 0.85, floor
         below z = -0.05), so even the depth->ESDF path (which maps what the
         camera SEES, not USD colliders) clips them out.
      3. The workbench top is at z = 0 -- the arm's MOUNT plane -- well below the
         grasp band (TCP z in [0.20,0.30]) and below the can worktop (z = 0.15),
         so if a sliver ever were mapped it is ground, not an obstacle.

    The arm's base_link is NOT moved: it stays at the world origin (all grasps and
    planning are relative to it). The bench top merely renders under the mount.
    """
    FLOOR_TOP = -0.75            # a table-height floor below the bench
    BENCH_TOP = 0.0              # arm base_link mount plane
    # --- floor: a big slab whose top is at FLOOR_TOP, entirely below the box ---
    _box(stage, "/World/room_floor",
         center=(0.4, 0.0, FLOOR_TOP - 0.05), scale=(10.0, 10.0, 0.10),
         rgb=(0.28, 0.26, 0.24), collision=False)
    # --- four walls, from the floor up, all beyond |1.8| m (outside the box) ---
    wall_rgb = (0.62, 0.63, 0.60)
    wall_h = 3.0
    wall_cz = FLOOR_TOP + wall_h / 2.0
    _box(stage, "/World/room_wall_xneg", (-1.8, 0.0, wall_cz), (0.1, 5.0, wall_h), wall_rgb, collision=False)
    _box(stage, "/World/room_wall_xpos", (2.2, 0.0, wall_cz), (0.1, 5.0, wall_h), wall_rgb, collision=False)
    _box(stage, "/World/room_wall_yneg", (0.2, -2.2, wall_cz), (5.0, 0.1, wall_h), wall_rgb, collision=False)
    _box(stage, "/World/room_wall_ypos", (0.2, 2.2, wall_cz), (5.0, 0.1, wall_h), wall_rgb, collision=False)
    # --- workbench: top face flush with the arm mount plane (z = 0) ----------
    bench_rgb = (0.48, 0.34, 0.19)      # wood
    leg_rgb = (0.20, 0.15, 0.10)
    bx0, bx1, by0, by1 = -0.40, 0.80, -0.55, 0.55
    top_thk = 0.04
    _box(stage, "/World/bench_top",
         center=((bx0 + bx1) / 2.0, 0.0, BENCH_TOP - top_thk / 2.0),
         scale=(bx1 - bx0, by1 - by0, top_thk), rgb=bench_rgb, collision=False)
    leg_top = BENCH_TOP - top_thk
    leg_h = leg_top - FLOOR_TOP
    leg_cz = FLOOR_TOP + leg_h / 2.0
    leg_i = 0
    for lx in (bx0 + 0.06, bx1 - 0.06):
        for ly in (by0 + 0.06, by1 - 0.06):
            # USD prim names must be valid identifiers -- no '-', so index them
            # rather than embed signed coordinates.
            _box(stage, "/World/bench_leg_%d" % leg_i,
                 center=(lx, ly, leg_cz), scale=(0.05, 0.05, leg_h),
                 rgb=leg_rgb, collision=False)
            leg_i += 1
    # --- pedestal so the raised can-worktop (z=0.15) does not float ----------
    # The worktop top is at CAN_BASE_Z; fill from the bench top up to its bottom.
    ped_top = CAN_BASE_Z - 0.02      # worktop bottom face
    _box(stage, "/World/worktop_pedestal",
         center=(0.40, 0.0, (BENCH_TOP + ped_top) / 2.0),
         scale=(0.34, 0.60, ped_top - BENCH_TOP), rgb=(0.30, 0.31, 0.33),
         collision=False)
    # --- ceiling: a slab above the box top (z=0.95), so it caps the room -----
    CEIL_Z = 2.2
    _box(stage, "/World/room_ceiling",
         center=(0.2, 0.0, CEIL_Z + 0.05), scale=(5.0, 5.0, 0.10),
         rgb=(0.82, 0.82, 0.84), collision=False)
    # --- indoor ceiling lights: rectangular panels + emissive fixture boxes --
    # SphereLights (Isaac supports them well) hung just under the ceiling. They
    # light the whole room; the existing dome/key still expose the RGB stream the
    # perception path needs. Placed high (z~2.1) and central so shadows fall
    # naturally on the bench.
    from pxr import UsdLux
    for i, (lx, ly) in enumerate([(-0.3, -0.6), (-0.3, 0.6), (0.7, -0.6), (0.7, 0.6)]):
        panel = _box(stage, "/World/ceil_fixture_%d" % i,
                     center=(lx, ly, CEIL_Z - 0.03), scale=(0.30, 0.30, 0.04),
                     rgb=(1.0, 0.98, 0.92), collision=False)
        light = UsdLux.SphereLight.Define(stage, "/World/ceil_light_%d" % i)
        light.CreateRadiusAttr(0.15)
        light.CreateIntensityAttr(20000.0)
        light.CreateColorAttr(Gf.Vec3f(1.0, 0.97, 0.90))
        lxf = UsdGeom.Xformable(light)
        lxf.ClearXformOpOrder()
        lxf.AddTranslateOp().Set(Gf.Vec3d(lx, ly, CEIL_Z - 0.10))
    # recolor the bare-grey worktop so the work surface reads as a fixture
    wt = stage.GetPrimAtPath(WORKTOP_PRIM)
    if wt and wt.IsValid():
        UsdGeom.Gprim(wt).CreateDisplayColorAttr([Gf.Vec3f(0.38, 0.39, 0.42)])
    print("ENVIRONMENT: room (floor z=%.2f, 4 walls, ceiling z=%.2f, 4 lights) + "
          "workbench (top z=%.2f), visual-only, outside nvblox box"
          % (FLOOR_TOP, CEIL_Z, BENCH_TOP), flush=True)


#: The staged mesh's half-height along its Y axis (extent 0.102 m, centred on the
#: mesh origin), so the base sits 0.051 m below the mesh origin. Used to lift the
#: mesh so its base lands on the can's own frame origin.
CAN_HALF_HEIGHT = 0.051
CAN_INITIAL_ROOT_Z = CAN_BASE_Z + CAN_COLLIDER.height_m
CAN_INITIAL_ROTATION_DEG = (180.0, 0.0, 0.0)


def _remove_referenced_rigid_bodies(stage: Usd.Stage, mesh_prim: Usd.Prim) -> None:
    """Strip rigid-body ownership from a referenced can hierarchy.

    ``rigidBodyEnabled=false`` is insufficient: PhysX still sees the nested
    RigidBodyAPI and gives the visible child an independent transform. Author a
    session-layer API deletion so CAN_PRIM is the only body in the composition.
    """
    removed = []
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for prim in list(Usd.PrimRange(mesh_prim)):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            if not prim.RemoveAPI(UsdPhysics.RigidBodyAPI):
                raise RuntimeError(
                    "could not remove referenced RigidBodyAPI from %s"
                    % prim.GetPath())
            removed.append(str(prim.GetPath()))

    remaining = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(mesh_prim)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if remaining:
        raise RuntimeError(
            "referenced can still owns nested rigid bodies: %s" % remaining)
    if removed:
        print("CAN: removed referenced RigidBodyAPI from %s" % removed, flush=True)


def _assert_single_can_rigid_body(root: Usd.Prim) -> None:
    owners = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI)
    ]
    if owners != [CAN_PRIM]:
        raise RuntimeError(
            "soup can must have exactly one rigid-body owner at %s, found %s"
            % (CAN_PRIM, owners))
    print("CAN: single rigid-body owner verified at %s" % CAN_PRIM, flush=True)


def add_soup_can(stage: Usd.Stage, grip_material) -> tuple[float, float, float]:
    """Place ONE soup can as a rigid dynamic prim on the worktop, in reach.

    Frame convention is load-bearing. The grasp set
    (config/rebot_grasps_soup_can.yaml) is authored object-relative with the
    object origin at the CENTRE OF THE BASE and +Z along the can axis. The mesh
    is authored Y-UP and CENTRED on its own origin (extent ~0.068 x 0.102 x
    0.068 m).

    Two-prim layout to reconcile those:
      CAN_PRIM  -- the rigid body at the object's semantic bottom centre. It
                   starts one can height above the table and rotated 180 degrees
                   about X, so object +Z points down and the bottom faces upward.
      CAN_PRIM/mesh -- the visual-only referenced mesh, rotated -90 deg about X
                   (mesh -Y, the labeled can's top axis, -> world +Z) and lifted
                   +CAN_HALF_HEIGHT so the mesh base coincides with CAN_PRIM's
                   origin.
      CAN_PRIM/physics_collider -- a hidden analytic cylinder with the canonical
                   66 mm diameter and 101 mm height.

    Getting this wrong is silent: a mesh centred on CAN_PRIM (the earlier form)
    places the collider and rendered object 50 mm above their intended pose.
    """
    x, y = CAN_XY
    # The semantic bottom-center starts above the can with +Z pointing down.
    # T * Rx(pi) keeps the physical center at table + half-height.
    can = UsdGeom.Xform.Define(stage, CAN_PRIM)
    root = can.GetPrim()
    rxf = UsdGeom.Xformable(root)
    rxf.ClearXformOpOrder()
    rxf.AddTranslateOp().Set(Gf.Vec3d(x, y, CAN_INITIAL_ROOT_Z))
    rxf.AddRotateXYZOp().Set(Gf.Vec3f(*CAN_INITIAL_ROTATION_DEG))

    # The mesh, stood upright and lifted so its base sits on the frame origin.
    mesh_xf = UsdGeom.Xform.Define(stage, "%s/mesh" % CAN_PRIM)
    mesh_prim = mesh_xf.GetPrim()
    mesh_prim.GetReferences().AddReference(CAN_YCB_URL)
    mxf = UsdGeom.Xformable(mesh_prim)
    mxf.ClearXformOpOrder()
    mxf.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CAN_HALF_HEIGHT))
    mxf.AddRotateXYZOp().Set(Gf.Vec3f(-90.0, 0.0, 0.0))
    # The YCB asset ships its OWN RigidBodyAPI; nested under our frame's rigid
    # body (applied below) it receives an independent PhysX transform and leaves
    # the visible mesh behind when CAN_PRIM moves. Remove the composed API,
    # rather than merely disabling it, so our frame is the sole body owner.
    _remove_referenced_rigid_bodies(stage, mesh_prim)

    # The referenced YCB collider is substantially wider than the rendered can.
    # Keep the textured hierarchy visual-only and author deterministic local
    # physics geometry instead. This order is important: the local collider must
    # be created only after inherited colliders have been disabled.
    disabled_colliders = disable_inherited_colliders(stage, mesh_prim)
    if disabled_colliders:
        print("CAN: disabled oversized inherited colliders at %s"
              % disabled_colliders, flush=True)
    configure_dynamic_can(root)
    collider = create_can_collider(
        stage, CAN_PRIM, grip_material, spec=CAN_COLLIDER)
    print(
        "CAN: local cylinder collider radius=%.3f height=%.3f center_z=%.4f "
        "mass=%.3f kinematic=false"
        % (
            CAN_COLLIDER.radius_m,
            CAN_COLLIDER.height_m,
            CAN_COLLIDER.center_z_m,
            CAN_COLLIDER.mass_kg,
        ),
        flush=True,
    )
    if not UsdPhysics.CollisionAPI(collider).GetCollisionEnabledAttr().Get():
        raise RuntimeError("local soup-can collider is not enabled")
    _assert_single_can_rigid_body(root)

    print("CAN: YCB textured tomato-soup can (S3)", flush=True)
    return (x, y, CAN_INITIAL_ROOT_Z)


def assert_physics(stage: Usd.Stage) -> None:
    """Assert the composed PhysicsScene uses CPU dynamics.

    The referenced USD is hashed and owned elsewhere. SimulationContext applies
    this scene's runtime timestep on the composed stage. We verify CPU dynamics
    because GPU dynamics makes the jaw-contact feedback used by stall detection
    unreliable.
    """
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if not scenes:
        print("WARN: no PhysicsScene found on the composed stage", flush=True)
        return
    physx = PhysxSchema.PhysxSceneAPI(scenes[0])
    gpu = physx.GetEnableGPUDynamicsAttr().Get()
    steps = physx.GetTimeStepsPerSecondAttr().Get()
    print("physics: scene=%s enableGPUDynamics=%r timeStepsPerSecond=%r"
          % (scenes[0].GetPath(), gpu, steps), flush=True)
    if gpu is not False:
        print("WARN: GPU dynamics is not OFF -- jaw contact reporting may be "
              "unreliable (see import_robot_usd.py configure_physics)", flush=True)


# ---------------------------------------------------------------------------
# ActionGraph: the wiring the whole contract flows through.
# ---------------------------------------------------------------------------
def build_action_graph(articulation_root: str) -> None:
    """Build the ROS 2 publisher/subscriber ActionGraph.

    *articulation_root* is the prim carrying ArticulationRootAPI (see
    find_articulation_root) -- NOT the robot xform. The joint-state publisher and
    the articulation controller resolve the articulation by this exact path.

    Shape mirrors Isaac's own moveit.py sample (the reference for the closed
    loop) and connectivity_scene.py (the /clock + /tf reference), driven by
    OnPlaybackTick so every node ticks on the render loop:

      OnPlaybackTick ─┬─> PublishClock         (/clock, sole owner)
                      ├─> PublishJointState     (/isaac_joint_states, all 8)
                      ├─> SubscribeJointState   (/isaac_joint_commands) ─┐
                      ├─> ArticulationController <───────────────────────┘ (closed loop)
                      ├─> SubscribeGripperCommand (leader jaw command) ────────┐
                      ├─> JawArticulationController <───────────────────────────┘
                      └─> PublishTf             (/tf, arm articulation)
      IsaacReadSimulationTime ─> timeStamp on Clock/JointState/Tf

    The camera pipeline (RGB/depth/camera_info) is a SEPARATE push graph built in
    build_camera_graph(), because the RTX sensor publishers must be generated in
    the SDG pipeline via CameraHelper.

    CLOSED LOOP -- the single most important thing here: SubscribeJointState's
    jointNames/positionCommand/velocityCommand/effortCommand feed the
    IsaacArticulationController on /rebot_b601dm. A command on
    /isaac_joint_commands therefore MOVES the articulation, which then reports the
    new state on /isaac_joint_states. A scene that publishes state but ignores
    commands looks alive and cannot be driven.
    """
    connect = [
        ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "PublishJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "SubscribeJointState.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "ArticulationController.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "SubscribeGripperCommand.inputs:execIn"),
        ("OnPlaybackTick.outputs:tick", "JawArticulationController.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
        ("ReadSimTime.outputs:simulationTime", "PublishJointState.inputs:timeStamp"),
        # The subscribed command fields drive the articulation controller. This is
        # the closed loop.
        ("SubscribeJointState.outputs:jointNames",
         "ArticulationController.inputs:jointNames"),
        ("SubscribeJointState.outputs:positionCommand",
         "ArticulationController.inputs:positionCommand"),
        ("SubscribeJointState.outputs:velocityCommand",
         "ArticulationController.inputs:velocityCommand"),
        ("SubscribeJointState.outputs:effortCommand",
         "ArticulationController.inputs:effortCommand"),
        ("SubscribeGripperCommand.outputs:jointNames",
         "JawArticulationController.inputs:jointNames"),
        ("SubscribeGripperCommand.outputs:positionCommand",
         "JawArticulationController.inputs:positionCommand"),
    ]
    create = [
        ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
        ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
        ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
        ("PublishJointState", "isaacsim.ros2.bridge.ROS2PublishJointState"),
        ("SubscribeJointState", "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("ArticulationController",
         "isaacsim.core.nodes.IsaacArticulationController"),
        ("SubscribeGripperCommand",
         "isaacsim.ros2.bridge.ROS2SubscribeJointState"),
        ("JawArticulationController",
         "isaacsim.core.nodes.IsaacArticulationController"),
    ]
    set_values = [
        # "/clock" stated explicitly though it is the node default: the
        # connectivity gate asserts exactly one publisher of that exact name.
        ("PublishClock.inputs:topicName", TOPIC_CLOCK),
        # Publish the WHOLE articulation's joint state. targetPrim is the robot
        # root; the bridge walks the articulation and emits every joint, so all 6
        # arm joints + both jaws appear on the wire.
        ("PublishJointState.inputs:topicName", TOPIC_JOINT_STATES),
        # targetPrim is the ARTICULATION ROOT prim, not /rebot_b601dm. The bridge
        # walks the articulation from here and emits every joint (6 arm + 2 jaws).
        ("PublishJointState.inputs:targetPrim",
         [usdrt.Sdf.Path(articulation_root)]),
        ("SubscribeJointState.inputs:topicName", TOPIC_JOINT_COMMANDS),
        # The articulation the commands drive. robotPath is the string form the
        # IsaacArticulationController expects; it must be the articulation-root
        # prim path or the controller reports "not an articulation" every tick and
        # (in the execution evaluator) aborts the whole graph tick.
        ("ArticulationController.inputs:robotPath", articulation_root),
        ("SubscribeGripperCommand.inputs:topicName", TOPIC_GRIPPER_COMMAND),
        ("JawArticulationController.inputs:robotPath", articulation_root),
    ]

    create.append(
        ("PublishTf", "isaacsim.ros2.bridge.ROS2PublishTransformTree"))
    connect += [
        ("OnPlaybackTick.outputs:tick", "PublishTf.inputs:execIn"),
        ("ReadSimTime.outputs:simulationTime", "PublishTf.inputs:timeStamp"),
    ]
    # targetPrims is the articulation ROOT. ROS2PublishTransformTree walks the
    # whole articulation tree from an articulation prim and emits one frame per
    # link. The dynamic can is deliberately not a target: FoundationPose owns
    # object-pose estimation for the live pipeline.
    set_values += [
        ("PublishTf.inputs:topicName", "/tf"),
        ("PublishTf.inputs:targetPrims",
         [usdrt.Sdf.Path(articulation_root)]),
    ]

    # Static TF for each scene camera prim, so nvblox/perception can locate the
    # depth streams in the world frame (base_link == World here).
    for _spec in SCENE_CAMS:
        node = "PublishTf_%s" % _spec["name"]
        create.append((node, "isaacsim.ros2.bridge.ROS2PublishTransformTree"))
        connect += [
            ("OnPlaybackTick.outputs:tick", "%s.inputs:execIn" % node),
            ("ReadSimTime.outputs:simulationTime", "%s.inputs:timeStamp" % node),
        ]
        set_values += [
            ("%s.inputs:topicName" % node, "/tf"),
            ("%s.inputs:targetPrims" % node, [usdrt.Sdf.Path(_spec["prim"])]),
        ]

    og.Controller.edit(
        {"graph_path": "/PickSceneGraph", "evaluator_name": "execution"},
        {
            og.Controller.Keys.CREATE_NODES: create,
            og.Controller.Keys.SET_VALUES: set_values,
            og.Controller.Keys.CONNECT: connect,
        },
    )


def build_camera_graph() -> None:
    """Build the wrist RGB-D + camera_info publishers on the USD camera prim.

    Headless (no viewport), so a render product is created DIRECTLY on the
    existing wrist camera prim via IsaacCreateRenderProduct -- the pattern from
    Isaac's own test_camera_tf_delay.py -- rather than the viewport route the
    camera_periodic.py sample uses. Using the stage's authored camera (not a new
    one) is deliberate: its pose and 1280x720 D435-class intrinsics are baked into
    the USD at the extrinsics_sim.yaml transform, and the 180-deg optical-frame
    flip is already applied, so the published image agrees with
    camera_color_optical_frame.

    Three CameraHelper/CameraInfoHelper nodes share one render product:
      * RGB   -> /front_stereo_camera/left/image_raw      (type rgb)
      * depth -> /front_stereo_camera/depth/ground_truth  (type depth)
      * info  -> /front_stereo_camera/left/camera_info
    The helpers derive CameraInfo intrinsics from the render product's resolution
    and the camera prim's aperture/focal length, so camera_info matches the USD
    camera by construction. `depth` = DistanceToImagePlane, which is the
    ground-truth range image the segmenter's depth_image_topics expects.

    A push/on-demand graph, evaluated once, so the CameraHelper nodes generate
    their SDG-pipeline publishers; thereafter each render step drives them.
    """
    keys = og.Controller.Keys
    (cam_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/PickSceneCameraGraph",
            "evaluator_name": "push",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND,
        },
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("CreateRenderProduct",
                 "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CameraHelperRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CameraHelperInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("CameraHelperDepth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                # Render product on the wrist camera at the contract resolution.
                ("CreateRenderProduct.inputs:cameraPrim",
                 [usdrt.Sdf.Path(CAMERA_PRIM)]),
                ("CreateRenderProduct.inputs:width", 1280),
                ("CreateRenderProduct.inputs:height", 720),
                # RGB
                ("CameraHelperRgb.inputs:frameId", CAMERA_FRAME_ID),
                ("CameraHelperRgb.inputs:topicName", TOPIC_RGB),
                ("CameraHelperRgb.inputs:type", "rgb"),
                # CameraInfo (paired with both RGB and the GT depth on the ROS
                # side; there is one wrist camera so one info topic).
                ("CameraHelperInfo.inputs:frameId", CAMERA_FRAME_ID),
                ("CameraHelperInfo.inputs:topicName", TOPIC_CAMERA_INFO),
                # Depth (ground-truth range image = DistanceToImagePlane).
                ("CameraHelperDepth.inputs:frameId", CAMERA_FRAME_ID),
                ("CameraHelperDepth.inputs:topicName", TOPIC_DEPTH),
                ("CameraHelperDepth.inputs:type", "depth"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut",
                 "CameraHelperRgb.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut",
                 "CameraHelperInfo.inputs:execIn"),
                ("CreateRenderProduct.outputs:execOut",
                 "CameraHelperDepth.inputs:execIn"),
                ("CreateRenderProduct.outputs:renderProductPath",
                 "CameraHelperRgb.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath",
                 "CameraHelperInfo.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath",
                 "CameraHelperDepth.inputs:renderProductPath"),
            ],
        },
    )
    # Evaluate once so the SDG-pipeline ROS publishers are generated before the
    # main loop starts stepping.
    og.Controller.evaluate_sync(cam_graph)


def camera_look_at_transform(eye_value, target_value):
    """Return a USD camera transform and normalized world-space view direction."""
    eye = np.asarray(eye_value, dtype=float)
    target = np.asarray(target_value, dtype=float)
    fwd = target - eye
    fwd /= np.linalg.norm(fwd)
    transform = Gf.Matrix4d(1.0).SetLookAt(
        Gf.Vec3d(*[float(v) for v in eye]),
        Gf.Vec3d(*[float(v) for v in target]),
        Gf.Vec3d(0.0, 0.0, 1.0),
    ).GetInverse()
    return transform, fwd


def make_scene_camera_asset_visual_only(
        stage: Usd.Stage, asset_model: Usd.Prim) -> tuple[list[str], list[str]]:
    """Remove inherited PhysX ownership from a referenced camera housing."""
    removed_rigid_bodies = []
    removed_colliders = []
    with Usd.EditContext(stage, Usd.EditTarget(stage.GetSessionLayer())):
        for prim in list(Usd.PrimRange(asset_model)):
            path = str(prim.GetPath())
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                if not prim.RemoveAPI(UsdPhysics.RigidBodyAPI):
                    raise RuntimeError(
                        "could not remove D455 RigidBodyAPI from %s" % path)
                removed_rigid_bodies.append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                if not prim.RemoveAPI(UsdPhysics.CollisionAPI):
                    raise RuntimeError(
                        "could not remove D455 CollisionAPI from %s" % path)
                removed_colliders.append(path)

    remaining_physics = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(asset_model)
        if (
            prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.HasAPI(UsdPhysics.CollisionAPI)
        )
    ]
    if remaining_physics:
        raise RuntimeError(
            "D455 fixture must be visual-only, remaining physics=%s"
            % remaining_physics
        )
    return removed_rigid_bodies, removed_colliders


def add_scene_camera_fixture(
        stage: Usd.Stage, spec: dict, camera_back: np.ndarray,
        camera_transform: Gf.Matrix4d) -> None:
    """Add a visible D455 and bench mount without changing the sensor pose.

    A USD Camera prim is only an optical origin, so it is invisible in the
    viewport. The official D455 model is registered from its authored color
    camera transform: applying that transform's inverse to the model root makes
    the referenced color imager coincide exactly with the validated custom
    Camera prim. This preserves the D455's native visual orientation without
    guessing axis rotations or lens offsets. The full model subtree is needed
    for its sibling material library. Its inherited dynamic body is removed so
    the housing cannot fall away during physics warmup, and all embedded sensor
    prims are deactivated. Only the custom Camera prims have render products and
    ROS publishers.

    The support terminates at the bench mount plane, z=0. Everything is
    visual-only: camera fixtures must not become unmodeled PhysX obstacles in
    this manipulation POC.
    """
    eye = np.asarray(spec["eye"], dtype=float)
    rear_attachment = eye + 0.075 * np.asarray(camera_back, dtype=float)
    mount_path = "/World/%s_mount" % spec["name"]
    UsdGeom.Xform.Define(stage, mount_path)

    # Keep render geometry out from under the Camera schema prim. Some Hydra
    # delegates treat descendants of sensor prims as camera internals and omit
    # them from unrelated render products. A sibling visual Xform with the exact
    # same world transform remains visibly and mechanically aligned.
    housing_frame_path = "%s/camera" % mount_path
    housing_frame = UsdGeom.Xform.Define(stage, housing_frame_path)
    housing_frame_xf = UsdGeom.Xformable(housing_frame)
    housing_frame_xf.ClearXformOpOrder()
    housing_frame_xf.AddTransformOp().Set(camera_transform)

    asset_root_path = "%s/realsense_d455" % housing_frame_path
    asset_root = UsdGeom.Xform.Define(stage, asset_root_path)
    asset_root.GetPrim().GetReferences().AddReference(REALSENSE_D455_URL)
    asset_model_path = "%s/%s" % (
        asset_root_path, REALSENSE_D455_MODEL_PRIM)
    asset_model = stage.GetPrimAtPath(asset_model_path)
    if not asset_model.IsValid():
        raise RuntimeError(
            "D455 asset model not found at %s" % asset_model_path)
    color_camera = stage.GetPrimAtPath(
        "%s/%s" % (asset_model_path, REALSENSE_D455_COLOR_CAMERA))
    if not color_camera.IsValid():
        raise RuntimeError(
            "D455 asset color camera not found at %s"
            % color_camera.GetPath()
        )
    color_camera_transform = (
        UsdGeom.Xformable(color_camera).GetLocalTransformation()
    )
    asset_model_xf = UsdGeom.Xformable(asset_model)
    asset_model_xf.ClearXformOpOrder()
    asset_model_xf.AddTransformOp().Set(color_camera_transform.GetInverse())

    removed_bodies, removed_colliders = make_scene_camera_asset_visual_only(
        stage, asset_model)
    disabled_cameras = []
    for camera_name in REALSENSE_D455_EMBEDDED_CAMERAS:
        embedded_camera = stage.GetPrimAtPath(
            "%s/%s" % (asset_model_path, camera_name))
        if embedded_camera.IsValid():
            embedded_camera.SetActive(False)
            disabled_cameras.append(camera_name)
    if len(disabled_cameras) != len(REALSENSE_D455_EMBEDDED_CAMERAS):
        raise RuntimeError(
            "D455 asset camera hierarchy changed: disabled=%s expected=%s"
            % (disabled_cameras, REALSENSE_D455_EMBEDDED_CAMERAS)
        )
    template_render_products = stage.GetPrimAtPath(
        "%s/TemplateRenderProducts" % asset_model_path)
    if not template_render_products.IsValid():
        raise RuntimeError("D455 asset template render products not found")
    template_render_products.SetActive(False)
    # A narrow camera-local adapter joins the real sensor body to the rear mount.
    _box(
        stage, "%s/mount_adapter" % housing_frame_path,
        center=(0.0, 0.0, 0.0505), scale=(0.018, 0.018, 0.049),
        rgb=(0.20, 0.23, 0.26), collision=False,
    )

    # Put the post directly under the rear housing attachment, rather than under
    # the optical origin. This both reads as a real mount and keeps the support
    # outside the camera's forward frustum.
    post_x, post_y, post_top = [float(v) for v in rear_attachment]
    clamp_height = 0.07
    _box(
        stage, "%s/bench_clamp" % mount_path,
        center=(post_x, post_y, -0.005),
        scale=(0.075, 0.075, clamp_height),
        rgb=(0.24, 0.27, 0.30), collision=False,
    )
    post_bottom = 0.025
    _box(
        stage, "%s/post" % mount_path,
        center=(post_x, post_y, 0.5 * (post_bottom + post_top)),
        scale=(0.025, 0.025, post_top - post_bottom),
        rgb=(0.55, 0.58, 0.62), collision=False,
    )
    mount_ball = UsdGeom.Sphere.Define(stage, "%s/mount_ball" % mount_path)
    mount_ball.CreateRadiusAttr(0.018)
    mount_ball.CreateDisplayColorAttr([Gf.Vec3f(0.20, 0.23, 0.26)])
    mount_ball_xf = UsdGeom.Xformable(mount_ball)
    mount_ball_xf.ClearXformOpOrder()
    mount_ball_xf.AddTranslateOp().Set(
        Gf.Vec3d(post_x, post_y, post_top))
    print(
        "SCENE CAM MOUNT: %s Intel RealSense D455 + bench clamp at "
        "(%.2f, %.2f), head z=%.2f, visual-only "
        "(removed rigid bodies=%d colliders=%d)"
        % (
            spec["name"],
            post_x,
            post_y,
            post_top,
            len(removed_bodies),
            len(removed_colliders),
        ),
        flush=True,
    )


def create_scene_camera(stage: Usd.Stage, spec: dict) -> None:
    """Author a fixed Camera prim and visible fixture at a look-at pose.

    The camera looks from ``eye`` toward ``target``. USD cameras look down their
    local -Z with +Y up; we build the rotation so the optical axis points at the
    target, matching the ROS optical frame (+Z forward is handled by the ROS
    CameraHelper's frame). The sensor and its visual-only bench fixture are
    static and world-fixed.
    """
    from pxr import Gf, UsdGeom
    eye = np.asarray(spec["eye"], dtype=float)
    target = np.asarray(spec["target"], dtype=float)
    transform, fwd = camera_look_at_transform(eye, target)
    cam = UsdGeom.Camera.Define(stage, spec["prim"])
    cam.CreateFocalLengthAttr(18.15)              # ~69 deg HFOV on a 24mm aperture
    cam.CreateHorizontalApertureAttr(24.0)
    cam.CreateVerticalApertureAttr(18.0)          # 4:3 sensor, keeps fx == fy
    cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    x = UsdGeom.Xformable(cam)
    x.ClearXformOpOrder()
    x.AddTransformOp().Set(transform)
    if spec.get("visible_fixture", False):
        add_scene_camera_fixture(stage, spec, -fwd, transform)
    print("SCENE CAM: %s eye=%s -> target=%s" %
          (spec["name"], [round(v, 2) for v in eye], [round(v, 2) for v in target]),
          flush=True)


def build_scene_camera_graph(spec: dict) -> None:
    """RGB + depth + camera_info publishers + TF for one static scene camera.

    Mirrors build_camera_graph (wrist) but on a world-fixed camera prim and under
    the camera's own namespace, so all three streams (2 scene + 1 wrist) coexist
    on the bus and can be fed to nvblox as separate depth inputs.
    """
    ns = spec["ns"]
    frame = spec["frame"]
    graph_path = "/SceneCamGraph_%s" % spec["name"]
    keys = og.Controller.Keys
    (g, _, _, _) = og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "push",
         "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_ONDEMAND},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnTick"),
                ("RP", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("Rgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("Info", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
                ("Depth", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("RP.inputs:cameraPrim", [usdrt.Sdf.Path(spec["prim"])]),
                ("RP.inputs:width", SCENE_CAM_W),
                ("RP.inputs:height", SCENE_CAM_H),
                ("Rgb.inputs:frameId", frame),
                ("Rgb.inputs:topicName", "%s/rgb/image_raw" % ns),
                ("Rgb.inputs:type", "rgb"),
                ("Rgb.inputs:frameSkipCount", SCENE_CAM_FRAME_SKIP),
                ("Info.inputs:frameId", frame),
                ("Info.inputs:topicName", "%s/camera_info" % ns),
                ("Info.inputs:frameSkipCount", SCENE_CAM_FRAME_SKIP),
                ("Depth.inputs:frameId", frame),
                ("Depth.inputs:topicName", "%s/depth/image_raw" % ns),
                ("Depth.inputs:type", "depth"),
                ("Depth.inputs:frameSkipCount", SCENE_CAM_FRAME_SKIP),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "RP.inputs:execIn"),
                ("RP.outputs:execOut", "Rgb.inputs:execIn"),
                ("RP.outputs:execOut", "Info.inputs:execIn"),
                ("RP.outputs:execOut", "Depth.inputs:execIn"),
                ("RP.outputs:renderProductPath", "Rgb.inputs:renderProductPath"),
                ("RP.outputs:renderProductPath", "Info.inputs:renderProductPath"),
                ("RP.outputs:renderProductPath", "Depth.inputs:renderProductPath"),
            ],
        },
    )
    og.Controller.evaluate_sync(g)
    print("SCENE CAM GRAPH: %s -> %s/{rgb,depth,camera_info}" % (spec["name"], ns),
          flush=True)


# ---------------------------------------------------------------------------
# Establishing camera for the pick->place video (passive observer).
# ---------------------------------------------------------------------------
def build_record_camera(
    *,
    camera_path: str,
    eye,
    target,
    focal_length: float,
    width: int,
    height: int,
    label: str,
):
    """Build one passive world observer and return its RGB annotator."""
    import omni.replicator.core as rep  # noqa: E402  (Kit ext, import on demand)

    stage = omni.usd.get_context().get_stage()
    camera = UsdGeom.Camera.Define(stage, camera_path)
    camera.CreateFocalLengthAttr(focal_length)
    camera.CreateHorizontalApertureAttr(CAMERA_HORIZONTAL_APERTURE_MM)
    camera.CreateVerticalApertureAttr(
        CAMERA_HORIZONTAL_APERTURE_MM * height / width)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    camera_xf = UsdGeom.Xformable(camera)
    camera_xf.ClearXformOpOrder()
    transform, _ = camera_look_at_transform(eye, target)
    camera_xf.AddTransformOp().Set(transform)

    rp = rep.create.render_product(
        camera_path,
        (width, height),
    )
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    rgb.attach(rp)
    print(
        "RECORD CAMERA %s: eye=%s target=%s focal=%.1fmm size=%dx%d"
        % (label, eye, target, focal_length, width, height),
        flush=True,
    )
    return rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def log_contract() -> None:
    """Log exactly which topics are published and subscribed.

    flush=True so a verifier can diff this against the contract while the sim
    runs.
    """
    print("=" * 70, flush=True)
    print("reBot pick_scene up (ROS_DISTRO=%s RMW=%s DOMAIN=%s) headless=%s"
          % (os.environ.get("ROS_DISTRO"), os.environ.get("RMW_IMPLEMENTATION"),
             os.environ.get("ROS_DOMAIN_ID"), not _args.gui), flush=True)
    print("PUBLISH:", flush=True)
    print("  %-42s clock, sole owner, 60 Hz" % TOPIC_CLOCK, flush=True)
    print("  %-42s all %d joints %s"
          % (TOPIC_JOINT_STATES, len(ALL_JOINTS), ALL_JOINTS), flush=True)
    print("  %-42s RGB 1280x720" % TOPIC_RGB, flush=True)
    print("  %-42s CameraInfo (fx=fy=923.6 cx=640 cy=360)"
          % TOPIC_CAMERA_INFO, flush=True)
    print("  %-42s wrist depth" % TOPIC_DEPTH, flush=True)
    print("  %-42s arm articulation and scene-camera TF" % "/tf", flush=True)
    print("SUBSCRIBE:", flush=True)
    print("  %-42s joint position commands -> articulation (CLOSED LOOP)"
          % TOPIC_JOINT_COMMANDS, flush=True)
    print("  %-42s paired physical jaw command -> %s (feedback %s)"
          % (TOPIC_GRIPPER_COMMAND, JAW_COMMAND_JOINTS, JAW_JOINTS), flush=True)
    print("=" * 70, flush=True)


def main() -> int:
    if _args.record_width <= 0 or _args.record_height <= 0:
        raise ValueError("record dimensions must be positive")
    if (
        _args.record_wide_dir
        and (_args.record_wide_width <= 0 or _args.record_wide_height <= 0)
    ):
        raise ValueError("high-wide record dimensions must be positive")
    if not 1 <= _args.record_jpeg_quality <= 100:
        raise ValueError("record JPEG quality must be in [1, 100]")

    planner_spheres = load_planner_collision_spheres(PLANNER_XRDF_PATH)
    record_state_path = (
        Path(_args.record_state_file).expanduser().resolve()
        if _args.record_state_file
        else None
    )
    stage, repaired_body_paths = build_stage()
    grip_material = create_grip_material(stage)
    add_lights(stage)
    add_worktop(stage, grip_material)
    wall = create_transfer_wall(stage)
    bind_physics_material(wall, grip_material)
    print(
        "OBSTACLE: static transfer wall center=%s size=%s top_z=%.3f; "
        "visible to scene_cam_0 and scene_cam_1"
        % (
            TRANSFER_WALL.center_xyz_m,
            TRANSFER_WALL.size_xyz_m,
            TRANSFER_WALL.top_z_m,
        ),
        flush=True,
    )

    wall_command_path = (
        Path(_args.transfer_wall_command_file).expanduser().resolve()
        if _args.transfer_wall_command_file
        else None
    )
    wall_command_mtime_ns = None
    wall_visible = True

    def poll_transfer_wall_command() -> None:
        nonlocal wall_command_mtime_ns, wall_visible
        if wall_command_path is None or not wall_command_path.is_file():
            return
        mtime_ns = wall_command_path.stat().st_mtime_ns
        if mtime_ns == wall_command_mtime_ns:
            return
        wall_command_mtime_ns = mtime_ns
        try:
            command = json.loads(wall_command_path.read_text(encoding="utf-8"))
            requested = command["wall_visible"]
            if not isinstance(requested, bool):
                raise ValueError("wall_visible must be a JSON boolean")
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            print("WARN: invalid transfer-wall command %s (%s)"
                  % (wall_command_path, exc), flush=True)
            return
        imageable = UsdGeom.Imageable(wall)
        if requested:
            imageable.MakeVisible()
        else:
            imageable.MakeInvisible()
        wall_visible = requested
        print(
            "OBSTACLE: wall_visible=%s (PhysX collider remains enabled)"
            % wall_visible,
            flush=True,
        )

    poll_transfer_wall_command()
    add_environment(stage)
    add_area_markers(stage)

    try:
        add_soup_can(stage, grip_material)
    except Exception as _e:  # noqa: BLE001 -- Kit can swallow the real cause
        import traceback
        sys.stderr.write("FATAL add_soup_can: %s\n%s\n"
                         % (_e, traceback.format_exc()))
        sys.stderr.flush()
        raise

    finger_fix = refine_finger_colliders(stage, ROBOT_PRIM, grip_material)
    if finger_fix["changed_count"] < 2:
        raise RuntimeError(
            "expected at least one collider per finger, refined %d: %s"
            % (finger_fix["changed_count"], finger_fix["changed"])
        )
    print(
        "GRIPPER: refined %d finger colliders to convexDecomposition "
        "(deinstanced=%d)"
        % (finger_fix["changed_count"], finger_fix["deinstanced_count"]),
        flush=True,
    )
    jaw_model = configure_independent_jaw_drives(stage, ROBOT_PRIM)
    print(
        "GRIPPER: independent jaws on %s removed_mimic_instances=%s; "
        "paired force-limited drives=%s"
        % (
            jaw_model["prim"],
            jaw_model["removed_instances"],
            jaw_model["drives"],
        ),
        flush=True,
    )

    # Static scene cameras (prims authored now; their ROS graphs built after play).
    for _spec in SCENE_CAMS:
        try:
            create_scene_camera(stage, _spec)
        except Exception as _e:  # noqa: BLE001
            import traceback
            sys.stderr.write("FATAL create_scene_camera %s: %s\n%s\n"
                             % (_spec.get("name"), _e, traceback.format_exc()))
            sys.stderr.flush()
            raise

    # The validated contact probe runs physics at 120 Hz. Rendering remains at
    # 60 Hz so perception bandwidth does not double with the contact substeps.
    # stage_units_in_meters=1.0 matches the metric USD.
    sim = SimulationContext(physics_dt=1.0 / PHYSICS_HZ, rendering_dt=1.0 / 60.0,
                            stage_units_in_meters=1.0)

    assert_physics(stage)

    articulation_root = find_articulation_root(stage)
    print("articulation root: %s" % articulation_root, flush=True)
    contact_monitor = TransferWallContactMonitor(
        TRANSFER_WALL.prim_path,
        (ROBOT_PRIM, CAN_PRIM),
    )
    for report_prim in (
        wall,
        stage.GetPrimAtPath(WORKTOP_PRIM),
        stage.GetPrimAtPath(CAN_PRIM),
        stage.GetPrimAtPath(articulation_root),
    ):
        report_api = PhysxSchema.PhysxContactReportAPI.Apply(report_prim)
        report_api.CreateThresholdAttr().Set(0.0)
    contact_subscription = (
        get_physx_simulation_interface().subscribe_contact_report_events(
            contact_monitor.on_contact_report
        )
    )
    print(
        "WALL SAFETY: %d buffered XRDF spheres; PhysX contact reporting active"
        % len(planner_spheres),
        flush=True,
    )
    configure_articulation_solver(stage, articulation_root)
    print(
        "physics: %.0f Hz, contact_offset=%.4f rest_offset=%.4f, "
        "articulation solver iterations position=32 velocity=4"
        % (PHYSICS_HZ, CONTACT_OFFSET_M, REST_OFFSET_M),
        flush=True,
    )
    enable_robot_gravity_compensation(stage)
    build_action_graph(articulation_root)
    build_camera_graph()
    for _spec in SCENE_CAMS:
        build_scene_camera_graph(_spec)
    simulation_app.update()

    sim.initialize_physics()
    sim.play()

    # Set a good default viewport framing so --gui opens looking AT the work,
    # not at Kit's default origin view (which needs manual zoom every time). Eye
    # is opposite scene_cam_0, so its visible body is in the workcell view rather
    # than between the viewport and the table.
    if _args.gui:
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            cam_target = RECORD_CAMERA_TARGET
            cam_eye = MAIN_CAMERA_EYE
            set_camera_view(eye=cam_eye, target=cam_target)
            print("VIEWPORT: framed eye=%s target=%s" % (cam_eye, cam_target),
                  flush=True)
        except Exception as e:  # noqa: BLE001 -- framing is cosmetic
            print("WARN: could not set viewport camera (%s)" % e, flush=True)

    # Bring the articulation up and command the elevated start pose. The articulation
    # must be initialized AFTER play() so the physics view exists;
    # set_joint_positions seeds the state so the first
    # /isaac_joint_states starts at the pose we asked for rather than an
    # uninitialized articulation state.
    from isaacsim.core.prims import SingleArticulation  # noqa: E402
    arm = SingleArticulation(prim_path=articulation_root, name="rebot")
    arm.initialize()
    dof_names = list(arm.dof_names)
    link_sync = None
    if repaired_body_paths:
        link_sync = vendor.LinkUsdSync(stage, arm, repaired_body_paths)
        if link_sync.link_count != vendor.EXPECTED_NESTED_RIGID_BODY_COUNT:
            raise RuntimeError(
                "tensor-to-USD sync resolved %d repaired links, expected %d: %s"
                % (
                    link_sync.link_count,
                    vendor.EXPECTED_NESTED_RIGID_BODY_COUNT,
                    link_sync.link_names,
                )
            )
        print(
            "RENDER SYNC: PhysX tensor -> %d vendor repair transforms"
            % link_sync.link_count,
            flush=True,
        )

    target_q = {jn: 0.0 for jn in dof_names}
    target_q.update(START_Q)
    # Seed the jaws OPEN, not at 0.0 (=closed). A closed gripper on approach is
    # exactly why the arm appeared to drive its shut fingers through the can; the
    # gripper action bridge drives them shut only at the grasp.
    for jj in JAW_JOINTS:
        target_q[jj] = JAW_OPEN_M
    q = np.array([target_q.get(n, 0.0) for n in dof_names])
    # Seed both the STATE (so the first /isaac_joint_states is the requested pose)
    # and the drive TARGET (so there is no startup control transient before
    # /isaac_joint_commands takes over). apply_action(ArticulationAction) is the
    # drive-target path SingleArticulation exposes across 5.x.
    from isaacsim.core.utils.types import ArticulationAction  # noqa: E402

    expected_dof_order = ARM_JOINTS + JAW_JOINTS
    if dof_names != expected_dof_order:
        raise RuntimeError(
            "runtime gains require DOF order %s, got %s"
            % (expected_dof_order, dof_names))
    drive_controller = arm.get_articulation_controller()
    authored_kp, authored_kd = drive_controller.get_gains()
    print("authored articulation PD gains: kp=%s kd=%s"
          % (np.round(authored_kp, 3).tolist(),
             np.round(authored_kd, 3).tolist()), flush=True)
    if USING_VENDOR_ASSET:
        # The official converter authors force limits but zero PD gains. These
        # are the exact eight gains validated against the vendor asset.
        tuned_kp = vendor.RUNTIME_KP.copy()
        damped_kd = vendor.RUNTIME_KD.copy()
    else:
        tuned_kp = np.asarray(authored_kp, dtype=float).reshape(-1).copy()
        damped_kd = np.asarray(authored_kd, dtype=float).reshape(-1).copy()
        damped_kd[:len(ARM_JOINTS)] = ARM_DAMPING
    jaw_indices = [dof_names.index(name) for name in JAW_JOINTS]
    jaw_drive_indices = [
        dof_names.index(name) for name in JAW_COMMAND_JOINTS
    ]
    for index in jaw_drive_indices:
        tuned_kp[index] = JAW_STIFFNESS
        damped_kd[index] = JAW_DAMPING
    drive_controller.set_gains(
        kps=tuned_kp,
        kds=damped_kd,
    )
    applied_kp, applied_kd = drive_controller.get_gains()
    expected_arm_kp = (
        vendor.RUNTIME_KP[:len(ARM_JOINTS)]
        if USING_VENDOR_ASSET
        else np.asarray(authored_kp).reshape(-1)[:len(ARM_JOINTS)]
    )
    expected_arm_kd = (
        vendor.RUNTIME_KD[:len(ARM_JOINTS)]
        if USING_VENDOR_ASSET
        else np.asarray(ARM_DAMPING)
    )
    if not np.allclose(
            np.asarray(applied_kp).reshape(-1)[:len(ARM_JOINTS)],
            expected_arm_kp):
        raise RuntimeError("arm stiffness gains were not applied")
    if not np.allclose(
            np.asarray(applied_kd).reshape(-1)[:len(ARM_JOINTS)],
            expected_arm_kd):
        raise RuntimeError("arm damping was not applied")
    if not np.allclose(
            np.asarray(applied_kp).reshape(-1)[jaw_drive_indices],
            JAW_STIFFNESS):
        raise RuntimeError("jaw stiffness was not applied")
    if not np.allclose(
            np.asarray(applied_kd).reshape(-1)[jaw_drive_indices],
            JAW_DAMPING):
        raise RuntimeError("jaw damping was not applied")
    print("stabilized articulation PD gains: kp=%s kd=%s"
          % (np.round(applied_kp, 3).tolist(),
             np.round(applied_kd, 3).tolist()), flush=True)

    # The imported 27 N.m arm limits clip even the slow planned trajectories and
    # create large tracking errors. Raise only the six arm limits for this POC;
    # gravity is compensated separately above, so this budget is used for motion
    # rather than for a hidden static position error. This is a runtime
    # physics-view write; the hashed USD remains untouched. Split the 40 N action
    # budget across the two jaws, matching the physical can probe's 20 N per-jaw
    # limit and remaining below the URDF's 100 N per-joint limit.
    try:
        av = arm._articulation_view
        me = np.asarray(av.get_max_efforts(), dtype=float).ravel().copy()
        arm_idx = [dof_names.index(j) for j in ARM_JOINTS if j in dof_names]
        for i in arm_idx:
            me[i] = max(me[i], 1000.0)
        per_jaw_effort = GRIPPER_MAX_EFFORT_N / len(jaw_drive_indices)
        for i in jaw_drive_indices:
            me[i] = per_jaw_effort
        av.set_max_efforts(np.expand_dims(me, 0))
        print(
            "max_efforts: arm=%s N.m leader_jaw=%s N"
            % (
                [round(me[i], 0) for i in arm_idx],
                [round(me[i], 1) for i in jaw_drive_indices],
            ),
            flush=True,
        )
    except Exception as e:  # noqa: BLE001 -- non-fatal; tracking may clip
        print("WARN: could not raise arm max_efforts (%s); motion may clip" % e,
              flush=True)

    arm.set_joint_positions(q)
    arm.apply_action(ArticulationAction(joint_positions=q))

    def step_rendered() -> None:
        # Push the previous PhysX state before the next render. This gives USD,
        # RTX cameras, the viewport, and TF the moving vendor links instead of
        # the frozen transforms left by resetXformStack.
        if link_sync is not None:
            link_sync.push()
        sim.step(render=True)

    print("DOF order (Isaac articulation): %s" % dof_names, flush=True)
    print("initial joint targets: %s"
          % {n: round(float(v), 3) for n, v in zip(dof_names, q)}, flush=True)
    log_contract()

    # Warm up: let physics settle and the first RTX frame render before the
    # bounded loop. Re-assert the drive target on each warmup step so the arm
    # arrives AT the elevated start pose and the drive target is registered before
    # ROS commands can take ownership.
    for _ in range(30):
        arm.apply_action(ArticulationAction(joint_positions=q))
        step_rendered()
    settled = arm.get_joint_positions()
    print("settled joint positions: %s"
          % {n: round(float(v), 3) for n, v in zip(dof_names, settled)}, flush=True)

    # Read-only object monitoring. There is deliberately no scene-side grasp
    # state and no can pose writer: lift, hold, and release are all PhysX results.
    can_xform_cache = UsdGeom.XformCache()
    can_root = stage.GetPrimAtPath(CAN_PRIM)
    can_collider = stage.GetPrimAtPath(
        "%s/%s" % (CAN_PRIM, CAN_COLLIDER_NAME))

    def _can_state():
        if not can_root.IsValid():
            return None
        can_xform_cache.Clear()
        root_world = can_xform_cache.GetLocalToWorldTransform(can_root)
        root_xyz = np.array(root_world.ExtractTranslation(), dtype=float)
        center_xyz = np.array(
            can_xform_cache.GetLocalToWorldTransform(
                can_collider).ExtractTranslation(),
            dtype=float,
        )
        upright = np.array(
            root_world.TransformDir(Gf.Vec3d(0.0, 0.0, 1.0)),
            dtype=float,
        )
        upright /= np.linalg.norm(upright)
        return root_xyz, center_xyz, float(upright[2]), upright

    minimum_observed = {
        "robot_clearance_m": float("inf"),
        "can_clearance_m": float("inf"),
        "clearance_m": float("inf"),
        "sim_time_s": 0.0,
        "kind": "",
    }
    wall_state_samples = 0

    def _live_frame_transforms() -> dict[str, np.ndarray]:
        if link_sync is None:
            raise RuntimeError(
                "wall telemetry requires live PhysX link transforms"
            )
        transforms = link_sync.link_transforms()
        frames = {
            name: transforms[index].copy()
            for index, name in enumerate(link_sync.body_names)
        }
        if "link6" not in frames:
            raise RuntimeError("PhysX body transforms are missing link6")
        link6 = frames["link6"]
        camera_mount = link6.copy()
        camera_mount[:3] = (
            link6[:3]
            + quaternion_xyzw_rotation_matrix(link6[3:])
            @ np.asarray([0.05, 0.0, 0.06], dtype=float)
        )
        frames["camera_mount"] = camera_mount
        return frames

    def write_wall_state() -> dict | None:
        nonlocal wall_state_samples
        if record_state_path is None:
            return None
        can_state = _can_state()
        if can_state is None:
            raise RuntimeError("wall telemetry cannot read the soup-can pose")

        robot_spheres = transform_collision_spheres(
            planner_spheres,
            _live_frame_transforms(),
        )
        robot_clearance, robot_index = minimum_sphere_aabb_clearance(
            robot_spheres,
            TRANSFER_WALL.min_xyz_m,
            TRANSFER_WALL.max_xyz_m,
        )
        can_clearance_result = finite_cylinder_aabb_clearance_bounds(
            can_state[1],
            can_state[3],
            radius_m=CAN_COLLIDER.radius_m,
            height_m=CAN_COLLIDER.height_m,
            minimum_xyz_m=TRANSFER_WALL.min_xyz_m,
            maximum_xyz_m=TRANSFER_WALL.max_xyz_m,
        )
        can_clearance = can_clearance_result.lower_bound_m
        can_witness = can_clearance_result.cylinder_witness_m
        wall_witness = can_clearance_result.box_witness_m
        clearance = min(robot_clearance, can_clearance)
        kind = "robot" if robot_clearance <= can_clearance else "can"
        sim_time_s = float(sim.current_time)
        minimum_observed["robot_clearance_m"] = min(
            minimum_observed["robot_clearance_m"],
            robot_clearance,
        )
        minimum_observed["can_clearance_m"] = min(
            minimum_observed["can_clearance_m"],
            can_clearance,
        )
        if clearance < minimum_observed["clearance_m"]:
            minimum_observed["clearance_m"] = clearance
            minimum_observed["sim_time_s"] = sim_time_s
            minimum_observed["kind"] = kind
        wall_state_samples += 1

        robot_sphere = robot_spheres[robot_index]
        payload = {
            "schema_version": WALL_SAFETY_SCHEMA_VERSION,
            "sim_time_s": sim_time_s,
            "samples": wall_state_samples,
            "wall": {
                "prim_path": TRANSFER_WALL.prim_path,
                "minimum_xyz_m": list(TRANSFER_WALL.min_xyz_m),
                "maximum_xyz_m": list(TRANSFER_WALL.max_xyz_m),
            },
            "sample": {
                "robot_clearance_m": robot_clearance,
                "robot_link": planner_spheres[robot_index].link_name,
                "robot_sphere_center_m": robot_sphere[:3].tolist(),
                "robot_sphere_radius_m": float(robot_sphere[3]),
                "can_clearance_m": can_clearance,
                "can_clearance_upper_bound_m": (
                    can_clearance_result.upper_bound_m
                ),
                "can_clearance_uncertainty_m": (
                    can_clearance_result.uncertainty_m
                ),
                "can_clearance_iterations": can_clearance_result.iterations,
                "can_clearance_converged": can_clearance_result.converged,
                "can_witness_m": can_witness.tolist(),
                "wall_witness_m": wall_witness.tolist(),
                "clearance_m": clearance,
                "clearance_kind": kind,
                "can_center_m": can_state[1].tolist(),
                "can_axis_xyz": can_state[3].tolist(),
                "can_upright_z": can_state[2],
            },
            "minimum_observed": dict(minimum_observed),
            "contact": contact_monitor.snapshot(),
            "planner_sphere_count": len(planner_spheres),
            "can_clearance_method": (
                "finite_cylinder_aabb_certified_lower_bound"
            ),
        }
        atomic_write_json(record_state_path, payload)
        return payload

    settled_state = _can_state()
    if settled_state is not None:
        (
            settled_root,
            settled_center,
            settled_upright,
            _settled_axis,
        ) = settled_state
        settle_error = float(abs(settled_root[2] - CAN_INITIAL_ROOT_Z))
        if settle_error > 0.01:
            raise RuntimeError(
                "bottom-up can did not settle on the worktop: pose=%s "
                "expected_root_z=%.3f"
                % (np.round(settled_root, 4).tolist(), CAN_INITIAL_ROOT_Z)
            )
        if settled_upright > -0.95:
            raise RuntimeError(
                "can did not remain bottom-up: upright_z=%.3f"
                % settled_upright
            )
        print(
            "GRASP: physical contact mode; bottom-up can settled root=%s "
            "center=%s upright_z=%.3f "
            "(no attachment or pose writes)"
            % (
                np.round(settled_root, 4).tolist(),
                np.round(settled_center, 4).tolist(),
                settled_upright,
            ),
            flush=True,
        )
    initial_wall_state = write_wall_state()
    if initial_wall_state is not None:
        print(
            "WALL SAFETY: initial clearance=%.1f mm "
            "(robot=%.1f mm can=%.1f mm) contact=%s -> %s"
            % (
                1000.0 * initial_wall_state["sample"]["clearance_m"],
                1000.0 * initial_wall_state["sample"]["robot_clearance_m"],
                1000.0 * initial_wall_state["sample"]["can_clearance_m"],
                initial_wall_state["contact"]["ever"],
                record_state_path,
            ),
            flush=True,
        )

    # -- establishing-camera capture state -----------------------------------
    recorders = []
    rec_saved = 0
    if _args.record:
        try:
            from PIL import Image  # noqa: E402
            main_eye = MAIN_CAMERA_EYE
            main_target = MAIN_CAMERA_TARGET
            main_focal = MAIN_CAMERA_FOCAL_LENGTH_MM
            if _args.record_focus_scene_cam:
                main_eye = (0.32, 0.10, 0.55)
                main_target = SCENE_CAMS[0]["eye"]
                main_focal = 45.0
            main_dir = Path(_args.record_dir)
            main_dir.mkdir(parents=True, exist_ok=True)
            recorders.append({
                "label": "main",
                "dir": main_dir,
                "rgb": build_record_camera(
                    camera_path="/World/record_camera",
                    eye=main_eye,
                    target=main_target,
                    focal_length=main_focal,
                    width=_args.record_width,
                    height=_args.record_height,
                    label="main",
                ),
            })
            if _args.record_wide_dir:
                wide_dir = Path(_args.record_wide_dir)
                wide_dir.mkdir(parents=True, exist_ok=True)
                recorders.append({
                    "label": "high-wide",
                    "dir": wide_dir,
                    "rgb": build_record_camera(
                        camera_path="/World/record_camera_high_wide",
                        eye=WIDE_CAMERA_EYE,
                        target=WIDE_CAMERA_TARGET,
                        focal_length=WIDE_CAMERA_FOCAL_LENGTH_MM,
                        width=_args.record_wide_width,
                        height=_args.record_wide_height,
                        label="high-wide",
                    ),
                })
            # Let both new render products warm up before the first synchronized
            # pair is read.
            for _ in range(5):
                step_rendered()
            print(
                "RECORD: observers up -> %s"
                % [str(recorder["dir"]) for recorder in recorders],
                flush=True,
            )
        except Exception as e:  # noqa: BLE001 -- recording is non-fatal
            print("WARN: could not set up recording (%s); continuing" % e, flush=True)
            recorders = []

    def capture_frame():
        nonlocal rec_saved
        if not recorders:
            return
        images = []
        for recorder in recorders:
            data = recorder["rgb"].get_data()
            if data is None or getattr(data, "size", 0) == 0:
                if recorder["label"] == "main":
                    return
                images.append((recorder, None))
                continue
            images.append((
                recorder,
                np.asarray(data)[:, :, :3].astype(np.uint8),
            ))
        if not images:
            return
        from PIL import Image  # noqa: E402
        # Default PNG compression stalls the simulation long enough to trip the
        # perception action's wall-clock timeout. JPEG is appropriate for the
        # final H.264 video and keeps this passive observer from changing runtime
        # behavior.
        for recorder, image in images:
            if image is None:
                continue
            Image.fromarray(image).save(
                recorder["dir"] / ("frame_%05d.jpg" % rec_saved),
                quality=_args.record_jpeg_quality,
            )
        rec_saved += 1

    # step(render=True) every tick: OnPlaybackTick fires from the render loop and
    # the RTX sensors only produce a frame on a rendered step. A physics-only loop
    # would publish nothing on the wire and a stale image.
    #
    # HOLD THE POSE, WITHOUT FIGHTING COMMANDS. In the normal workflow the drive
    # target set during warmup persists in the gravity-compensated PhysX drive,
    # so the main loop does NOT need to re-issue it, and MUST NOT: the graph's
    # IsaacArticulationController is the sole path /isaac_joint_commands reaches the
    # articulation, and a Python apply_action every step overwrites the graph's
    # command in the same step (measured: with a per-step hold, commanding
    # joint1 -> 0.3 rad moved it 0.000 -- the closed loop was silently dead). So
    # the graph controller owns the articulation from here; the persisted target
    # holds until a real command arrives, and a command then moves the arm
    # (verified: joint1 0 -> 0.3 rad, and it stays).

    deadline = time.monotonic() + _args.duration
    next_report = time.monotonic() + 10.0
    step_i = 0
    print("LOOP: entering main loop, is_running=%s duration=%.1f"
          % (simulation_app.is_running(), _args.duration), flush=True)
    while simulation_app.is_running() and time.monotonic() < deadline:
        step_rendered()
        step_i += 1
        if step_i % 30 == 0:
            poll_transfer_wall_command()
        state_period = max(1, _args.record_every if recorders else 6)
        if record_state_path is not None and step_i % state_period == 0:
            write_wall_state()
        if recorders and (step_i % max(1, _args.record_every) == 0):
            capture_frame()
        if time.monotonic() >= next_report:
            can_state = _can_state()
            state = (
                " can_root=%s can_center=%s upright_z=%.3f"
                % (
                    np.round(can_state[0], 3).tolist(),
                    np.round(can_state[1], 3).tolist(),
                    can_state[2],
                )
                if can_state is not None
                else ""
            )
            print("  ... sim time %.2f s%s" % (sim.current_time, state), flush=True)
            next_report += 10.0
    if recorders:
        print(
            "RECORD: saved %d synchronized frames to %s"
            % (rec_saved, [str(recorder["dir"]) for recorder in recorders]),
            flush=True,
        )
    final_wall_state = write_wall_state()
    if final_wall_state is not None:
        print(
            "WALL SAFETY: min=%.1f mm kind=%s at sim=%.2fs "
            "contacts=%d ever=%s samples=%d"
            % (
                1000.0
                * final_wall_state["minimum_observed"]["clearance_m"],
                final_wall_state["minimum_observed"]["kind"],
                final_wall_state["minimum_observed"]["sim_time_s"],
                final_wall_state["contact"]["events"],
                final_wall_state["contact"]["ever"],
                final_wall_state["samples"],
            ),
            flush=True,
        )
    print("pick_scene done, sim time %.2f s" % sim.current_time, flush=True)
    sim.stop()
    return 0


if __name__ == "__main__":
    _rc = 1
    try:
        _rc = main()
    except Exception as _exc:  # noqa: BLE001 -- report before Kit swallows it
        import traceback

        sys.stderr.write("FATAL pick_scene: %s\n%s\n"
                         % (_exc, traceback.format_exc()))
        sys.stderr.flush()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        simulation_app.close()
    sys.exit(_rc)
