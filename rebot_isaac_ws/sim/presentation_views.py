"""Camera specifications shared by Isaac recording and video composition."""

CAMERA_HORIZONTAL_APERTURE_MM = 24.0

# Place the primary observer just inside and below the scene_cam_0 fixture so
# the mount does not occlude the workcell while the pick remains close enough
# to inspect.
MAIN_CAMERA_EYE = (0.84, -0.39, 0.61)
MAIN_CAMERA_TARGET = (0.25, 0.09, 0.21)
MAIN_CAMERA_EYE_OFFSET = tuple(
    MAIN_CAMERA_EYE[index] - MAIN_CAMERA_TARGET[index]
    for index in range(3)
)
MAIN_CAMERA_FOCAL_LENGTH_MM = 18.0
MAIN_RECORD_SIZE = (1920, 1080)

# Independent high observer from the far-right side of the room. Its render
# aspect matches the presentation tile, avoiding a crop that would hide the
# room edges.
WIDE_CAMERA_EYE = (1.55, 1.30, 1.65)
WIDE_CAMERA_TARGET = (0.25, 0.05, 0.20)
WIDE_CAMERA_FOCAL_LENGTH_MM = 16.0
WIDE_RECORD_SIZE = (640, 603)

# The perception source is scene_cam_0. These values are the static camera pose
# authored by pick_scene.py and convert ROS optical-frame detections to world.
PERCEPTION_CAMERA_EYE = (0.75, -0.45, 0.70)
PERCEPTION_CAMERA_TARGET = (0.37, 0.06, 0.15)
