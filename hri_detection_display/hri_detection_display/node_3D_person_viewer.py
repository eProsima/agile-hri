# Copyright 2024 Proyectos y Sistemas de Mantenimiento SL (eProsima).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
import rclpy
import rclpy.duration

from geometry_msgs.msg import Point
from hri_msgs.msg import Skeleton3D, Skeleton3DList, Face2DList, Expression
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
import rclpy.parameter

from scipy.spatial.transform import Rotation as R
import cv2 as cv
import numpy as np
from threading import Lock
from typing import List, Optional

from hri_detection_display.PersonDetectionTracker import PersonDetection

try:
    from cv_bridge import CvBridge
except ImportError:
    CvBridge = None


# Max number of calls to the timer callback that a body/face can miss before being removed
MAX_ITERATIONS_RETENTION = 15

# Time margin to consider a body/face as detected again
TIME_MARGIN_DETECTION = 0.2

# Anatomy constants (from node_3D_skel_display.py)
NECK_LENGTH = 0.1
SHOULDERS_LENGTH = 0.35
HIPS_LENGTH = 0.22
MAX_HIP_WIDTH = 0.35
MIN_HIP_WIDTH = 0.14
MAX_SHOULDER_WIDTH = 0.5
MIN_SHOULDER_WIDTH = 0.2
TRUNK_LENGTH = 0.6
MAX_BODY_EXTENSION = 0.8
MAX_LENGTH_CYLINDER = 1.5

# Padding (pixels) around projected skeleton points for bounding box
BBOX_PADDING = 20

# BGR colors constants
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
BGR_TEAL = (0, 128, 128)
BGR_GREEN = (0, 255, 0)
BGR_BLACK = (0, 0, 0)
BGR_WHITE = (255, 255, 255)
BGR_GREY = (180, 180, 180)
BGR_DARK_GREEN = (0, 100, 50)
BGR_LIGHT_ORANGE = (90, 155, 255)
BGR_ORANGE = (0, 130, 255)
BGR_YELLOW = (0, 225, 255)

# Skeleton connections using Skeleton3D constants
_connections = [
    (Skeleton3D.LEFT_WRIST, Skeleton3D.LEFT_ELBOW),
    (Skeleton3D.RIGHT_WRIST, Skeleton3D.RIGHT_ELBOW),
    (Skeleton3D.LEFT_ELBOW, Skeleton3D.LEFT_SHOULDER),
    (Skeleton3D.RIGHT_ELBOW, Skeleton3D.RIGHT_SHOULDER),
    (Skeleton3D.LEFT_SHOULDER, Skeleton3D.RIGHT_SHOULDER),
    (Skeleton3D.RIGHT_SHOULDER, Skeleton3D.RIGHT_HIP),
    (Skeleton3D.LEFT_SHOULDER, Skeleton3D.LEFT_HIP),
    (Skeleton3D.RIGHT_HIP, Skeleton3D.LEFT_HIP),
    (Skeleton3D.RIGHT_HIP, Skeleton3D.RIGHT_KNEE),
    (Skeleton3D.LEFT_HIP, Skeleton3D.LEFT_KNEE),
    (Skeleton3D.RIGHT_KNEE, Skeleton3D.RIGHT_ANKLE),
    (Skeleton3D.LEFT_KNEE, Skeleton3D.LEFT_ANKLE),
]

_face_landmarks = [
    Skeleton3D.NOSE,
    Skeleton3D.LEFT_EYE,
    Skeleton3D.RIGHT_EYE,
    Skeleton3D.LEFT_EAR,
    Skeleton3D.RIGHT_EAR,
]

_body_landmarks = [
    Skeleton3D.NECK,
    Skeleton3D.LEFT_SHOULDER,
    Skeleton3D.RIGHT_SHOULDER,
    Skeleton3D.LEFT_HIP,
    Skeleton3D.RIGHT_HIP,
    Skeleton3D.NOSE,
    Skeleton3D.LEFT_EYE,
    Skeleton3D.RIGHT_EYE,
    Skeleton3D.LEFT_EAR,
    Skeleton3D.RIGHT_EAR,
]

_markers_dict = {
    "head": 0,
    "body": 1,
    "shoulders": 2,
    "hips": 3,
    "left_botarm": 4,
    "left_uparm": 5,
    "right_botarm": 6,
    "right_uparm": 7,
    "left_upleg": 8,
    "left_botleg": 9,
    "right_upleg": 10,
    "right_botleg": 11,
    "hinge_left_shoulder": 12,
    "hinge_right_shoulder": 13,
    "hinge_left_hip": 14,
    "hinge_right_hip": 15,
    "hinge_neck": 16,
    "hinge_left_elbow": 17,
    "hinge_right_elbow": 18,
    "hinge_left_knee": 19,
    "hinge_right_knee": 20,
    "hinge_left_wrist": 21,
    "hinge_right_wrist": 22,
    "hinge_left_ankle": 23,
    "hinge_right_ankle": 24,
    "left_eye": 41,
    "right_eye": 42,
}


def bound(val, min_val, max_val):
    """Bound a value between lower and upper limits."""
    return max(min_val, min(val, max_val))


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates [0..1] to bounded image pixel coordinates."""
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


def point_not_null(point):
    """Return True when a 3D skeleton point contains non-zero coordinates."""
    return point.x != 0 and point.y != 0 and point.z != 0


def color_to_msg(color):
    """Convert a BGR color to a ROS 2 ColorRGBA message."""
    msg = ColorRGBA()
    msg.r = color[2] / 255.0
    msg.g = color[1] / 255.0
    msg.b = color[0] / 255.0
    msg.a = 1.0
    return msg


class Node3DPersonViewer(Node):
    """ROS 2 node that uses 3D skeleton data for both 2D image overlay and 3D MarkerArray publishing."""

    def __init__(self):
        """Initialize parameters, subscriptions, publishers, timer loop, and the viewer window."""
        super().__init__('hri_3d_person_viewer')

        # -- Parameters --
        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing and rendering display frames.'))
        self.declare_parameter(
            'display_mode', 'all', ParameterDescriptor(
                description='Display mode. Options: "body", "face", "both", "all". Default: "all".'))
        self.declare_parameter(
            'allow_half_body', True, ParameterDescriptor(
                description='Allow displaying bodies that are not entirely visible.'))
        self.declare_parameter(
            'allow_back_turned', True, ParameterDescriptor(
                description='Allow displaying bodies that are not facing the camera.'))
        self.declare_parameter(
            'image_topic', '/color/image_raw', ParameterDescriptor(
                description='Input sensor_msgs/Image topic to visualize.'))
        self.declare_parameter(
            'camera_info_topic', '/color/camera_info', ParameterDescriptor(
                description='CameraInfo topic for 3D-to-pixel projection.'))
        self.declare_parameter(
            'skel3d_topic', '/humans/bodies/skel3D', ParameterDescriptor(
                description='Input Skeleton3DList topic.'))
        self.declare_parameter(
            'marker_topic', '/humans/detection/skel3D', ParameterDescriptor(
                description='Output MarkerArray topic for 3D skeleton visualization.'))
        self.declare_parameter(
            'display_hinges', True, ParameterDescriptor(
                description='Display joint hinge spheres in 3D markers.'))
        self.declare_parameter(
            'visual_style', 'cylinder', ParameterDescriptor(
                description='3D visual style. Options: "stripes", "cylinder".'))
        self.declare_parameter(
            'window_name', 'HRI 3D Person Viewer', ParameterDescriptor(
                description='OpenCV window title.'))
        self.declare_parameter(
            'window_x', 400, ParameterDescriptor(
                description='Initial window X position in pixels.'))
        self.declare_parameter(
            'window_y', 400, ParameterDescriptor(
                description='Initial window Y position in pixels.'))
        self.declare_parameter(
            'window_width', 1280, ParameterDescriptor(
                description='Initial window width in pixels.'))
        self.declare_parameter(
            'window_height', 720, ParameterDescriptor(
                description='Initial window height in pixels.'))
        self.declare_parameter(
            'window_move_step', 50, ParameterDescriptor(
                description='Keyboard move step in pixels for WASD controls.'))
        self.declare_parameter(
            'keep_aspect_ratio', True, ParameterDescriptor(
                description='Preserve image aspect ratio while resizing the window.'))
        self.declare_parameter(
            'no_signal_timeout', 2.0, ParameterDescriptor(
                description='Seconds without image frames before rendering "No signal".'))

        self.param_change_callback = self.add_on_set_parameters_callback(self.parameter_callback)

        self.dict_lock = Lock()
        self.image_lock = Lock()

        self.processing_rate = int(self.get_parameter('processing_rate').value)
        self.display_mode = self.get_parameter('display_mode').value
        self.allow_half_body = self.get_parameter('allow_half_body').value
        self.allow_back_turned = self.get_parameter('allow_back_turned').value
        self.image_topic = self.get_parameter('image_topic').value
        self.camera_info_topic = self.get_parameter('camera_info_topic').value
        self.skel3d_topic = self.get_parameter('skel3d_topic').value
        self.marker_topic = self.get_parameter('marker_topic').value
        self.display_hinges = self.get_parameter('display_hinges').value
        self.visual_style = self.get_parameter('visual_style').value
        self.window_name = self.get_parameter('window_name').value
        self.window_x = int(self.get_parameter('window_x').value)
        self.window_y = int(self.get_parameter('window_y').value)
        self.window_width = int(self.get_parameter('window_width').value)
        self.window_height = int(self.get_parameter('window_height').value)
        self.window_move_step = int(self.get_parameter('window_move_step').value)
        self.keep_aspect_ratio = self.get_parameter('keep_aspect_ratio').value
        self.no_signal_timeout = float(self.get_parameter('no_signal_timeout').value)

        self.persons_ = {}
        self.image_width = 0
        self.image_height = 0
        self.reception_start_proc_time = self.get_clock().now()
        self.start_time = self.get_clock().now()
        self.last_image_time = None
        self.cv_image_raw: Optional[np.ndarray] = None
        self.cv_image_marks: Optional[np.ndarray] = None
        self.no_signal_active = False
        self.window_initialized = False
        self.window_visible = True
        self.warned_topmost_unsupported = False

        # Camera intrinsics for 3D→2D projection
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        # Header from last 3D skeleton message (for marker frame_id)
        self.skel3d_header = None

        if CvBridge is not None:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warning(
                'cv_bridge is not installed. Falling back to basic numpy conversion for bgr8/rgb8/mono8.')

        # -- Subscriptions --
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub_ = self.create_subscription(Image, self.image_topic, self.image_callback, qos_sensor_data)
        self.camera_info_sub_ = self.create_subscription(CameraInfo, self.camera_info_topic, self.camera_info_callback, qos_sensor_data)
        self.pose_sub_ = self.create_subscription(Skeleton3DList, self.skel3d_topic, self.bodies_callback, 1)
        self.face_sub_ = self.create_subscription(Face2DList, '/humans/faces', self.faces_callback, 1)
        self.emotion_sub_ = self.create_subscription(Expression, '/humans/faces/emotion', self.expression_callback, 1)

        # -- Publisher --
        self.marker_pub_ = self.create_publisher(MarkerArray, self.marker_topic, 1)

        # -- Timer --
        self.proc_timer = self.create_timer(1 / max(1, self.processing_rate), self.main_callback)

        self._init_window()

        self.get_logger().info(
            f"Node3DPersonViewer initialized. Image: {self.image_sub_.topic_name}, "
            f"Skel3D: {self.pose_sub_.topic_name}, Markers: {self.marker_topic}, "
            f"CameraInfo: {self.camera_info_topic}. "
            'Keys: [q|Esc]=quit.')

    # ------------------------------------------------------------------ #
    #  Window management (same as node_person_viewer.py)
    # ------------------------------------------------------------------ #

    def _init_window(self):
        """Create and configure the OpenCV window using current node parameters."""
        if self.window_initialized:
            return

        flags = cv.WINDOW_NORMAL
        if hasattr(cv, 'WINDOW_GUI_NORMAL'):
            flags |= cv.WINDOW_GUI_NORMAL
        if self.keep_aspect_ratio and hasattr(cv, 'WINDOW_KEEPRATIO'):
            flags |= cv.WINDOW_KEEPRATIO
        elif hasattr(cv, 'WINDOW_FREERATIO'):
            flags |= cv.WINDOW_FREERATIO

        try:
            cv.namedWindow(self.window_name, flags)
            cv.resizeWindow(self.window_name, max(1, self.window_width), max(1, self.window_height))
            cv.moveWindow(self.window_name, self.window_x, self.window_y)
            self.window_initialized = True
            self._apply_window_behaviors()
        except cv.error as exc:
            self.get_logger().error(
                f"Failed to create OpenCV window '{self.window_name}': {exc}. "
                'Check DISPLAY and session type.')
            raise RuntimeError('Failed to initialize viewer window.') from exc

    def _apply_window_behaviors(self):
        """Apply top-most state and optional front-focus request to the viewer window."""
        if not self.window_initialized:
            return

    def move_window(self, dx: int, dy: int):
        """Move the viewer window by `dx, dy` pixels and persist the new coordinates."""
        self.window_x += int(dx)
        self.window_y += int(dy)
        if self.window_initialized:
            try:
                cv.moveWindow(self.window_name, self.window_x, self.window_y)
            except cv.error as exc:
                self.get_logger().warning(f'Failed to move window: {exc}')

    def destroy_window(self):
        """Destroy the viewer window if it exists."""
        if self.window_initialized:
            try:
                cv.destroyWindow(self.window_name)
            except cv.error:
                pass
            self.window_initialized = False

    # ------------------------------------------------------------------ #
    #  Image and CameraInfo callbacks
    # ------------------------------------------------------------------ #

    def image_callback(self, msg: Image):
        """Convert and cache the latest input image frame for rendering."""
        frame = self._to_bgr_image(msg)
        if frame is None:
            return

        with self.image_lock:
            self.cv_image_raw = frame
            self.image_height, self.image_width = frame.shape[:2]
            self.last_image_time = self.get_clock().now()

    def _to_bgr_image(self, msg: Image) -> Optional[np.ndarray]:
        """Convert an incoming ROS image to BGR OpenCV format."""
        encoding = (msg.encoding or '').lower()

        try:
            if self.bridge is not None:
                if encoding == 'bgr8':
                    return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                if encoding == 'rgb8':
                    rgb_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
                    return cv.cvtColor(rgb_frame, cv.COLOR_RGB2BGR)
                if encoding == 'mono8':
                    gray_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                    return cv.cvtColor(gray_frame, cv.COLOR_GRAY2BGR)
                self.get_logger().warning(
                    f"Unsupported encoding '{msg.encoding}', attempting conversion to bgr8.")
                return self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            return self._fallback_to_bgr(msg, encoding)
        except Exception as exc:
            self.get_logger().warning(f"Could not convert image with encoding '{msg.encoding}': {exc}")
            return None

    def _fallback_to_bgr(self, msg: Image, encoding: str) -> Optional[np.ndarray]:
        """Convert bgr8/rgb8/mono8 images to BGR without cv_bridge."""
        if encoding not in ['bgr8', 'rgb8', 'mono8']:
            self.get_logger().warning(
                f"Unsupported encoding '{msg.encoding}' without cv_bridge. Supported: bgr8/rgb8/mono8.")
            return None

        channels = 1 if encoding == 'mono8' else 3
        row_data_size = msg.width * channels
        if msg.step < row_data_size:
            self.get_logger().warning(
                f"Invalid image step for encoding '{msg.encoding}': step={msg.step}, expected>={row_data_size}.")
            return None

        data = np.frombuffer(msg.data, dtype=np.uint8)
        if data.size < msg.height * msg.step:
            self.get_logger().warning('Received truncated image buffer.')
            return None

        image_rows = data.reshape((msg.height, msg.step))
        compact = image_rows[:, :row_data_size]

        if encoding == 'mono8':
            gray = compact.reshape((msg.height, msg.width))
            return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)

        frame = compact.reshape((msg.height, msg.width, 3))
        if encoding == 'rgb8':
            return cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        return frame

    def camera_info_callback(self, msg: CameraInfo):
        """Store camera intrinsics for 3D-to-pixel projection."""
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    # ------------------------------------------------------------------ #
    #  3D --> 2D projection
    # ------------------------------------------------------------------ #

    def project_3d_to_px(self, pt, w, h):
        """Project a 3D point to pixel coordinates using camera intrinsics."""
        if self.fx is None or pt.z <= 0.0:
            return None
        px = int(self.fx * pt.x / pt.z + self.cx)
        py = int(self.fy * pt.y / pt.z + self.cy)
        if 0 <= px < w and 0 <= py < h:
            return (px, py)
        return None

    # ------------------------------------------------------------------ #
    #  Skeleton3D body callback
    # ------------------------------------------------------------------ #

    def bodies_callback(self, msg: Skeleton3DList):
        """Update tracked person body data from Skeleton3DList."""
        self.skel3d_header = msg.header

        for skel, roi in zip(msg.skeletons, msg.bboxes):
            if skel.key == '':
                continue
            if roi.key != skel.key:
                self.get_logger().error(f'Body id mismatch: [{roi.key}] != [{skel.key}]')
                continue

            key = skel.key
            with self.dict_lock:
                if key not in self.persons_:
                    self.get_logger().debug(f'Adding person with key {key}.')
                    self.persons_[key] = PersonDetection()
                    self.persons_[key].shoulders_distance = SHOULDERS_LENGTH
                    self.persons_[key].hips_distance = HIPS_LENGTH
                    self.persons_[key].landmarks = skel.skeleton
                else:
                    self._update_landmarks_3d(key, skel)

                # Compute median depth
                median_depth = self.get_median_depth(self.persons_[key].landmarks)
                self.persons_[key].depth = median_depth

                # Update shoulders/hips distances
                self._update_shoulders_hips_length(key)

                # Compute whole_body, facing, raise_hand from 3D data
                self._whole_body_3d(key, self.persons_[key].landmarks)
                self._facing_3d(key, self.persons_[key].landmarks)
                self._raise_hand_3d(key, self.persons_[key].landmarks)

                # Update ROI
                position = [roi.xmin, roi.ymin, roi.xmax, roi.ymax]
                self.persons_[key].body_position = position

                # Update body score from skeleton confidence
                self.persons_[key].body_score = skel.confidence

                # Update body timestamp
                self.persons_[key].times['body'] = self.get_clock().now()

    def _update_landmarks_3d(self, id, skel_msg):
        """Update 3D landmarks with depth filtering."""
        stored_depth = self.persons_[id].depth
        for idx, point in enumerate(skel_msg.skeleton):
            if not point_not_null(point):
                self.persons_[id].landmarks[idx] = point
            else:
                self.persons_[id].landmarks[idx].x = point.x
                self.persons_[id].landmarks[idx].y = point.y
                if point.z != 0:
                    if stored_depth == 0 or abs(point.z - stored_depth) < MAX_BODY_EXTENSION:
                        self.persons_[id].landmarks[idx].z = point.z
                    else:
                        self.get_logger().debug(
                            f"Point [{idx}] out of range with {point.z} and median: "
                            f"{stored_depth} for body [{id}].")

    def _whole_body_3d(self, body_id, landmarks):
        """Compute if enough joints are visible to consider the body as whole (3D version)."""
        head_seen = False
        shoulder_seen = False
        hips_seen = self.allow_half_body

        if point_not_null(landmarks[Skeleton3D.NOSE]) or \
                point_not_null(landmarks[Skeleton3D.LEFT_EAR]) or \
                point_not_null(landmarks[Skeleton3D.RIGHT_EAR]) or \
                point_not_null(landmarks[Skeleton3D.LEFT_EYE]) or \
                point_not_null(landmarks[Skeleton3D.RIGHT_EYE]):
            head_seen = True
        if point_not_null(landmarks[Skeleton3D.LEFT_SHOULDER]) or \
                point_not_null(landmarks[Skeleton3D.RIGHT_SHOULDER]):
            shoulder_seen = True
        if point_not_null(landmarks[Skeleton3D.LEFT_HIP]) or \
                point_not_null(landmarks[Skeleton3D.RIGHT_HIP]):
            hips_seen = True

        self.persons_[body_id].whole_body = head_seen and shoulder_seen and hips_seen

    def _facing_3d(self, body_id, landmarks):
        """Estimate whether the person is facing the camera using visible facial points (3D version)."""
        face_points = 0
        if point_not_null(landmarks[Skeleton3D.NOSE]):
            face_points += 1
        if point_not_null(landmarks[Skeleton3D.LEFT_EAR]):
            face_points += 1
        if point_not_null(landmarks[Skeleton3D.RIGHT_EAR]):
            face_points += 1
        if point_not_null(landmarks[Skeleton3D.LEFT_EYE]):
            face_points += 1
        if point_not_null(landmarks[Skeleton3D.RIGHT_EYE]):
            face_points += 1
        self.persons_[body_id].facing = face_points

    def _raise_hand_3d(self, body_id, landmarks):
        """Infer whether a person is raising the left hand (3D version)."""
        if self.persons_[body_id].facing == 0:
            self.persons_[body_id].hand_raised = False
            return

        face_ref = None
        for face_point in _face_landmarks:
            if point_not_null(landmarks[face_point]):
                face_ref = face_point
                break

        if face_ref is not None and point_not_null(landmarks[Skeleton3D.LEFT_WRIST]) and \
                landmarks[Skeleton3D.LEFT_WRIST].y < landmarks[face_ref].y:
            self.persons_[body_id].hand_raised = True
        else:
            self.persons_[body_id].hand_raised = False

    def _update_shoulders_hips_length(self, person_id):
        """Update the measured shoulders and hips distance for a person."""
        person = self.persons_[person_id]
        # Shoulders distance is updated only once
        if person.shoulders_distance == SHOULDERS_LENGTH:
            left_sh = person.landmarks[Skeleton3D.LEFT_SHOULDER]
            right_sh = person.landmarks[Skeleton3D.RIGHT_SHOULDER]
            # Check if shoulders are visible
            if point_not_null(left_sh) and point_not_null(right_sh):
                # Check if depths are similar and x_dist has a valid value
                x_dist = abs(left_sh.x - right_sh.x)
                if abs(left_sh.z - right_sh.z) < 0.005 and x_dist > SHOULDERS_LENGTH - 0.1:
                    # Calculate distance
                    person.shoulders_distance = x_dist - 0.03
                    self.get_logger().info(
                        f"Shoulders distance for body {person_id} updated to: {person.shoulders_distance}.")
        # Hips distance is updated only once
        if person.hips_distance == HIPS_LENGTH:
            # Check if hips are visible
            left_hip = person.landmarks[Skeleton3D.LEFT_HIP]
            right_hip = person.landmarks[Skeleton3D.RIGHT_HIP]
            if point_not_null(left_hip) and point_not_null(right_hip):
                # Check if depths are similar and x_dist has a valid value
                x_dist = abs(left_hip.x - right_hip.x)
                if abs(left_hip.z - right_hip.z) < 0.005 and x_dist > HIPS_LENGTH - 0.06:
                    person.hips_distance = x_dist

    # ------------------------------------------------------------------ #
    #  Face and emotion callbacks (same as node_person_viewer.py)
    # ------------------------------------------------------------------ #

    def faces_callback(self, msg: Face2DList):
        """Update tracked person face data from `/humans/faces`."""
        for roi_msg, ldmks in zip(msg.bboxes, msg.landmarks):
            if roi_msg.key != ldmks.key:
                self.get_logger().error(f'Face id mismatch: [{roi_msg.key}] != [{ldmks.key}]')
                continue
            if roi_msg.key == '':
                continue

            key = roi_msg.key
            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            with self.dict_lock:
                if key not in self.persons_:
                    self.get_logger().debug(f'Adding person with key {key}.')
                    self.persons_[key] = PersonDetection()
                self.persons_[key].face_position = position
                self.persons_[key].face_score = roi_msg.c
                self.persons_[key].face_landmarks = ldmks.landmarks
                self.persons_[key].times['face'] = self.get_clock().now()

    def expression_callback(self, msg: Expression):
        """Update tracked emotions from `/humans/faces/emotion`."""
        if msg.key not in self.persons_:
            self.get_logger().warning(f'Face id [{msg.key}] not found when assigning expression.')
            return
        self.persons_[msg.key].emotion = msg.expression
        self.persons_[msg.key].times['emotion'] = self.get_clock().now()

    # ------------------------------------------------------------------ #
    #  Main loop
    # ------------------------------------------------------------------ #

    def main_callback(self):
        """Main timer loop: update state, render overlays, present the frame, publish markers."""
        self.reception_start_proc_time = self.get_clock().now()
        self.update_tracking_status()

        timed_out = self.signal_timed_out()
        if timed_out:
            if not self.no_signal_active:
                self.no_signal_active = True
                self.get_logger().warning(
                    f"No frames received on '{self.image_topic}' for {self.no_signal_timeout:.2f}s.")
            frame = self.no_signal_frame()
        else:
            if self.no_signal_active:
                self.no_signal_active = False
                self.get_logger().info('Image signal restored.')
            frame = self.render_detection_frame()
            if frame is None:
                frame = self.no_signal_frame()

        self.show_frame(frame)

        # Publish 3D markers
        if self.skel3d_header is not None:
            self.publish_3d_markers()

    def signal_timed_out(self) -> bool:
        """Return True when no image has been received within `no_signal_timeout`."""
        if self.no_signal_timeout <= 0.0:
            return False
        now = self.get_clock().now().nanoseconds
        reference = self.start_time if self.last_image_time is None else self.last_image_time
        return (now - reference.nanoseconds) > self.no_signal_timeout * 1e9

    def no_signal_frame(self) -> np.ndarray:
        """Build a fallback frame when no messages are received."""
        with self.image_lock:
            base = None if self.cv_image_raw is None else self.cv_image_raw.copy()

        if base is None:
            height = max(240, self.window_height)
            width = max(320, self.window_width)
            base = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            cv.rectangle(base, (0, 0), (base.shape[1] - 1, base.shape[0] - 1), BGR_RED, 2)

        h = base.shape[0]
        cv.putText(base, 'No signal', (10, h - 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, BGR_RED, 1)
        cv.putText(base, f'{self.image_topic}', (10, h - 12), cv.FONT_HERSHEY_SIMPLEX, 0.35, BGR_WHITE, 1)
        return base

    def render_detection_frame(self) -> Optional[np.ndarray]:
        """Render person overlays over the latest frame according to display filters."""
        with self.image_lock:
            if self.cv_image_raw is None:
                return None
            self.cv_image_marks = self.cv_image_raw.copy()
            self.image_height, self.image_width = self.cv_image_marks.shape[:2]

        ids_print = ''
        with self.dict_lock:
            for person_id, person in self.persons_.items():
                if person.online and self.should_display_person(person):
                    has_body = person.landmarks is not None and any(
                        point_not_null(p) for p in person.landmarks if p is not None)
                    has_face = person.face_position != [0, 0, 0, 0]

                    if person.matched and (self.display_mode == 'both' or self.display_mode == 'all'):
                        self.draw_body(person_id, c_no_hand=BGR_DARK_GREEN, c_hand_raised=BGR_RED, c_ske=BGR_GREY)
                        self.draw_face(person_id, BGR_TEAL, matched=True)
                    elif has_body and (self.display_mode == 'body' or self.display_mode == 'all'):
                        self.draw_body(person_id, c_no_hand=BGR_BLUE, c_hand_raised=BGR_RED, c_ske=BGR_GREY)
                    elif has_face and (self.display_mode == 'face' or self.display_mode == 'all'):
                        self.draw_face(person_id, BGR_BLUE)

                    ids_print += f'[{person_id}] | '

        processing_duration_ms = (self.get_clock().now() - self.reception_start_proc_time).nanoseconds / 1e6
        self.get_logger().debug(f'Displaying: {ids_print}in {processing_duration_ms} ms.')
        return self.cv_image_marks

    def show_frame(self, frame: np.ndarray):
        """Display one frame and process keyboard/window interactions."""
        if not self.window_initialized:
            self._init_window()

        display_frame = self.prepare_display_frame(frame)
        try:
            cv.imshow(self.window_name, display_frame)
            key = cv.waitKey(1) & 0xFF
        except cv.error as exc:
            self.get_logger().error(f'OpenCV viewer error: {exc}')
            rclpy.shutdown()
            return

        if key in [27, ord('q')]:
            self.get_logger().info('Exit requested from viewer window (Esc/q).')
            rclpy.shutdown()
            return
        if key == ord('w'):
            self.move_window(0, -self.window_move_step)
            return
        if key == ord('s'):
            self.move_window(0, self.window_move_step)
            return
        if key == ord('a'):
            self.move_window(-self.window_move_step, 0)
            return
        if key == ord('d'):
            self.move_window(self.window_move_step, 0)
            return

        if hasattr(cv, 'WND_PROP_VISIBLE'):
            try:
                self.window_visible = cv.getWindowProperty(self.window_name, cv.WND_PROP_VISIBLE) >= 1
                if not self.window_visible:
                    self.get_logger().info('Viewer window closed by user.')
                    rclpy.shutdown()
            except cv.error:
                pass

    def prepare_display_frame(self, frame: np.ndarray) -> np.ndarray:
        """Resize the frame to the configured viewer size with optional aspect preservation."""
        window_w = max(1, int(self.window_width))
        window_h = max(1, int(self.window_height))

        img_h, img_w = frame.shape[:2]
        if img_h <= 0 or img_w <= 0:
            return frame

        if not self.keep_aspect_ratio:
            return cv.resize(frame, (window_w, window_h), interpolation=cv.INTER_LINEAR)

        scale = min(window_w / img_w, window_h / img_h)
        new_w = max(1, int(img_w * scale))
        new_h = max(1, int(img_h * scale))
        interpolation = cv.INTER_AREA if scale < 1.0 else cv.INTER_LINEAR
        resized = cv.resize(frame, (new_w, new_h), interpolation=interpolation)

        canvas = np.zeros((window_h, window_w, 3), dtype=np.uint8)
        x_offset = (window_w - new_w) // 2
        y_offset = (window_h - new_h) // 2
        canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
        return canvas

    # ------------------------------------------------------------------ #
    #  Tracking status
    # ------------------------------------------------------------------ #

    def update_tracking_status(self):
        """Update online/matched status and remove stale tracked persons."""
        time_check = self.get_clock().now().nanoseconds
        should_delete = []
        with self.dict_lock:
            for id, person in self.persons_.items():
                body_detected = time_check - person.times['body'].nanoseconds <= TIME_MARGIN_DETECTION * 1e9
                face_detected = time_check - person.times['face'].nanoseconds <= TIME_MARGIN_DETECTION * 1e9

                if not face_detected:
                    person.face_position = [0, 0, 0, 0]

                person.matched = body_detected and face_detected
                if body_detected or face_detected:
                    person.online = True
                    if person.frames_since_last_detection > 0:
                        person.frames_since_last_detection = 0
                else:
                    person.online = False
                    person.frames_since_last_detection += 1
                    if person.frames_since_last_detection > MAX_ITERATIONS_RETENTION:
                        should_delete.append(id)

                if time_check - person.times['emotion'].nanoseconds > TIME_MARGIN_DETECTION * 1e9:
                    person.emotion = ''

            for id in should_delete:
                self.get_logger().debug(f'Removing person {id}.')
                del self.persons_[id]

    def should_display_person(self, person):
        """Apply configured body visibility and facing filters for display."""
        half_body = self.allow_half_body or person.whole_body
        facing = self.allow_back_turned or person.facing >= 4
        return half_body and facing

    # ------------------------------------------------------------------ #
    #  2D drawing methods (adapted for 3D skeleton projection)
    # ------------------------------------------------------------------ #

    def should_draw_ske_line(self, start_point, end_point, landmarks):
        """Return True when a skeleton line segment has valid 3D endpoints."""
        return (landmarks[start_point] is not None and landmarks[end_point] is not None
                and point_not_null(landmarks[start_point]) and point_not_null(landmarks[end_point]))

    def draw_skeleton(self, image, landmarks, color=BGR_GREY):
        """Draw skeleton connections on the image by projecting 3D points to 2D pixels."""
        if self.fx is None:
            return
        for start_point, end_point in _connections:
            if self.should_draw_ske_line(start_point, end_point, landmarks):
                start_px = self.project_3d_to_px(landmarks[start_point], self.image_width, self.image_height)
                end_px = self.project_3d_to_px(landmarks[end_point], self.image_width, self.image_height)
                if start_px is not None and end_px is not None:
                    cv.line(image, start_px, end_px, color, 2)

    def _compute_body_bbox(self, person_id):
        """Compute bounding box from projected 3D skeleton points or ROI data if available. Returns (pt1, pt2) or None."""
        # ROI has priority if available
        if self.persons_[person_id].body_position != [0, 0, 0, 0]:
            pt1 = normalized_to_pixel_coordinates(
                self.persons_[person_id].body_position[0], self.persons_[person_id].body_position[1],
                self.image_width, self.image_height)
            pt2 = normalized_to_pixel_coordinates(
                self.persons_[person_id].body_position[2], self.persons_[person_id].body_position[3],
                self.image_width, self.image_height)
            if pt1 is not None and pt2 is not None:
                return pt1, pt2

        # Fallback method using projected 3D landmarks
        if self.fx is None:
            return None
        landmarks = self.persons_[person_id].landmarks
        if landmarks is None:
            return None

        xs, ys = [], []
        for lm in landmarks:
            if lm is not None and point_not_null(lm):
                px = self.project_3d_to_px(lm, self.image_width, self.image_height)
                if px is not None:
                    xs.append(px[0])
                    ys.append(px[1])

        if len(xs) < 2:
            return None

        xmin = max(0, min(xs) - BBOX_PADDING)
        ymin = max(0, min(ys) - BBOX_PADDING)
        xmax = min(self.image_width - 1, max(xs) + BBOX_PADDING)
        ymax = min(self.image_height - 1, max(ys) + BBOX_PADDING)
        return (xmin, ymin), (xmax, ymax)

    def draw_body(self, person_id, c_ske=BGR_GREY, c_no_hand=BGR_BLUE, c_hand_raised=BGR_RED):
        """Draw body bounding box (computed from projected 3D joints), score/depth label, and skeleton."""
        bbox = self._compute_body_bbox(person_id)
        if bbox is None:
            # No bbox but still try to draw skeleton
            if self.persons_[person_id].landmarks is not None:
                self.draw_skeleton(self.cv_image_marks, self.persons_[person_id].landmarks, c_ske)
            return

        pt1, pt2 = bbox
        score = str('{:.2f}'.format(self.persons_[person_id].body_score * 100))

        if self.persons_[person_id].hand_raised:
            cv.rectangle(self.cv_image_marks, pt1, pt2, c_hand_raised, 2)
        else:
            cv.rectangle(self.cv_image_marks, pt1, pt2, c_no_hand, 2)

        text = 'ID: ' + person_id + ' (' + score + '%)'
        if self.persons_[person_id].depth != 0:
            text += ' Dist: ' + str(round(self.persons_[person_id].depth, 2))

        cv.putText(self.cv_image_marks, text, (pt1[0], pt1[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, BGR_BLACK)

        if self.persons_[person_id].landmarks is not None:
            self.draw_skeleton(self.cv_image_marks, self.persons_[person_id].landmarks, c_ske)

    def draw_face(self, person_id, color=BGR_BLUE, matched=False):
        """Draw face ROI and text labels for one person."""
        pt1 = normalized_to_pixel_coordinates(
            self.persons_[person_id].face_position[0], self.persons_[person_id].face_position[1],
            self.image_width, self.image_height)
        pt2 = normalized_to_pixel_coordinates(
            self.persons_[person_id].face_position[2], self.persons_[person_id].face_position[3],
            self.image_width, self.image_height)
        score = str('{:.2f}'.format(self.persons_[person_id].face_score * 100))

        cv.rectangle(self.cv_image_marks, pt1, pt2, color, 2)

        if matched:
            emotion_offset = 5
        else:
            emotion_offset = 20
            cv.putText(self.cv_image_marks, 'ID: ' + person_id + ' (' + score + '%)',
                       (pt1[0], pt1[1] - 5), cv.FONT_HERSHEY_SIMPLEX, 0.5, BGR_BLACK)

        if self.persons_[person_id].emotion != '':
            cv.putText(self.cv_image_marks, 'Feeling: ' + self.persons_[person_id].emotion,
                       (pt1[0], pt1[1] - emotion_offset), cv.FONT_HERSHEY_SIMPLEX, 0.5, BGR_BLACK)

    # ------------------------------------------------------------------ #
    #  3D marker creation methods (from node_3D_skel_display.py)
    # ------------------------------------------------------------------ #

    def get_median_depth(self, landmarks):
        """Get the median depth of the body points."""
        valid_points = [landmarks[point] for point in _body_landmarks if point_not_null(landmarks[point])]
        if len(valid_points) == 0:
            return 0
        return np.median([point.z for point in valid_points])

    def create_base_marker(self, header, ns, id, color, marker_type):
        """Create a base marker msg."""
        marker = Marker()
        marker.header = header
        if marker.header.frame_id == "_depth_optical_frame" or marker.header.frame_id == "_color_optical_frame":
            marker.header.frame_id = "camera_depth_optical_frame"
        marker.ns = ns
        marker.id = id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.scale.y = 0.23
        marker.scale.z = 0.2
        marker.color = color_to_msg(color)
        marker.lifetime = rclpy.duration.Duration(nanoseconds=1e8).to_msg()
        return marker

    def get_mid_head(self, skeleton):
        """Get midpoint of face landmarks."""
        tot_x, tot_y, tot_points = 0, 0, 0
        for point in _face_landmarks:
            if point_not_null(skeleton[point]):
                tot_x += skeleton[point].x
                tot_y += skeleton[point].y
                tot_points += 1

        if tot_points != 0:
            return [(tot_x / tot_points), (tot_y / tot_points)]
        return None

    def fill_head(self, header, id, skeleton, median_depth, orientation, trunk=None):
        """Create head and eye markers."""
        if trunk is not None:
            mid_point = [0, 0]
            mid_point[0] = trunk.pose.position.x
            mid_point[1] = trunk.pose.position.y - trunk.scale.z / 2 - NECK_LENGTH
        else:
            mid_point = self.get_mid_head(skeleton)

        quaternion = R.from_euler('y', -orientation - np.pi / 2).as_quat()

        if mid_point is not None:
            head = self.create_base_marker(header, str(id), _markers_dict['head'], BGR_YELLOW, Marker.SPHERE)
            head.scale.x = 0.2
            head.pose.position.x = mid_point[0]
            head.pose.position.y = mid_point[1]
            head.pose.position.z = median_depth
            head.pose.orientation.x = quaternion[0]
            head.pose.orientation.y = quaternion[1]
            head.pose.orientation.z = quaternion[2]
            head.pose.orientation.w = quaternion[3]

            base_orientation = R.from_quat([head.pose.orientation.x,
                                            head.pose.orientation.y,
                                            head.pose.orientation.z,
                                            head.pose.orientation.w])
            relative_position_1 = np.array([-0.05, -0.04, -0.06])
            relative_position_2 = np.array([0.05, -0.04, -0.06])
            global_position_1 = base_orientation.apply(relative_position_1) + np.array([head.pose.position.x,
                                                                                        head.pose.position.y,
                                                                                        head.pose.position.z])
            global_position_2 = base_orientation.apply(relative_position_2) + np.array([head.pose.position.x,
                                                                                        head.pose.position.y,
                                                                                        head.pose.position.z])

            left_eye = self.create_base_marker(header, str(id), _markers_dict['left_eye'], BGR_BLACK, Marker.SPHERE)
            left_eye.scale.x = 0.05
            left_eye.scale.y = 0.05
            left_eye.scale.z = 0.05
            left_eye.pose.position.x = global_position_1[0]
            left_eye.pose.position.y = global_position_1[1]
            left_eye.pose.position.z = global_position_1[2]

            right_eye = self.create_base_marker(header, str(id), _markers_dict['right_eye'], BGR_BLACK, Marker.SPHERE)
            right_eye.scale.x = 0.05
            right_eye.scale.y = 0.05
            right_eye.scale.z = 0.05
            right_eye.pose.position.x = global_position_2[0]
            right_eye.pose.position.y = global_position_2[1]
            right_eye.pose.position.z = global_position_2[2]

            return [head, left_eye, right_eye]
        return None

    def cylinder_from_points(self, header, ns, id, p1, p2):
        """Create a cylinder marker between two 3D points."""
        cylinder = self.create_base_marker(header, ns, id, BGR_ORANGE, Marker.CYLINDER)
        cylinder.pose.position.x = (p1.x + p2.x) / 2
        cylinder.pose.position.y = (p1.y + p2.y) / 2
        cylinder.pose.position.z = (p1.z + p2.z) / 2
        cylinder.scale.x = 0.06
        cylinder.scale.y = 0.06

        n_p1 = np.array([p1.x, p1.y, p1.z])
        n_p2 = np.array([p2.x, p2.y, p2.z])
        direction = n_p2 - n_p1
        length = np.linalg.norm(direction)
        # Limit max length to avoid extreme scaling from outliers
        length = min(length, MAX_LENGTH_CYLINDER)
        direction_normalized = direction / length

        z_axis = np.array([0.0, 0.0, 1.0])
        dot_product = np.dot(z_axis, direction_normalized)

        if np.isclose(dot_product, 1.0):
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        elif np.isclose(dot_product, -1.0):
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])
        else:
            axis = np.cross(z_axis, direction_normalized)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(dot_product)
            rotation = R.from_rotvec(axis * angle)
            quaternion = rotation.as_quat()

        cylinder.scale.z = length
        cylinder.pose.orientation.x = quaternion[0]
        cylinder.pose.orientation.y = quaternion[1]
        cylinder.pose.orientation.z = quaternion[2]
        cylinder.pose.orientation.w = quaternion[3]

        return cylinder

    def get_orientation(self, skeleton, last_orientation, sh_dist, hips_dist):
        """Get the orientation of the body in the XZ plane."""
        try_use_shoulder, try_use_hips = False, False
        if point_not_null(skeleton[Skeleton3D.RIGHT_SHOULDER]) and point_not_null(skeleton[Skeleton3D.LEFT_SHOULDER]):
            try_use_shoulder = True
        if point_not_null(skeleton[Skeleton3D.RIGHT_HIP]) and point_not_null(skeleton[Skeleton3D.LEFT_HIP]):
            try_use_hips = True

        if try_use_shoulder:
            x_dist = (skeleton[Skeleton3D.LEFT_SHOULDER].x - skeleton[Skeleton3D.RIGHT_SHOULDER].x)

            facing_positive_x = True
            if abs(x_dist) < 0.08 and last_orientation != -1.57:
                # Nearly sideways - use last_orientation to decide direction
                # Skip this shortcut if last_orientation is still the default (-1.57),
                # so we fall through to the full arccos computation.
                if abs(abs(last_orientation) - (np.pi)) < abs(last_orientation):
                    return -np.pi
                else:
                    return 0
            else:
                if skeleton[Skeleton3D.LEFT_SHOULDER].z < skeleton[Skeleton3D.RIGHT_SHOULDER].z:
                    facing_positive_x = False

            cos_theta = x_dist / sh_dist
            cos_theta = bound(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)

            theta = theta if facing_positive_x else -theta
            return (theta - np.pi / 2)

        if try_use_hips:
            x_dist = (skeleton[Skeleton3D.LEFT_HIP].x - skeleton[Skeleton3D.RIGHT_HIP].x)

            facing_positive_x = True
            if abs(x_dist) < 0.08 and last_orientation != -1.57:
                # Nearly sideways - use last_orientation to decide direction
                # Skip this shortcut if last_orientation is still the default (-1.57).
                if abs(abs(last_orientation) - (np.pi)) < abs(last_orientation):
                    return -np.pi
                else:
                    return 0
            else:
                if skeleton[Skeleton3D.LEFT_HIP].z < skeleton[Skeleton3D.RIGHT_HIP].z:
                    facing_positive_x = False

            cos_theta = x_dist / hips_dist
            cos_theta = bound(cos_theta, -1, 1)
            theta = np.arccos(cos_theta)

            theta = theta if facing_positive_x else -theta
            return (theta - np.pi / 2)

        return None

    def check_valid_orientation(self, orientation, last_orientation):
        """Check if the orientation is too different from the last one. Returns True if valid."""
        if orientation is not None:
            if last_orientation == -1.57:
                return True
            diff = 0
            if orientation >= 0 and last_orientation >= 0:
                diff = abs(orientation - last_orientation)
            elif orientation <= 0 and last_orientation <= 0:
                diff = abs(orientation - last_orientation)
            else:
                if orientation < 0:
                    diff = abs(2 * np.pi + orientation - last_orientation)
                else:
                    diff = abs(2 * np.pi + last_orientation - orientation)
            if diff < 1.0:
                return True
        return False

    def create_fix_body(self, header, id, skeleton, median_depth, orientation):
        """Create trunk, shoulders, and hips markers."""
        trunk = self.create_neck_spine_cyl(header, id, skeleton, median_depth)
        if trunk is None:
            return None
        quaternion = R.from_euler('y', -orientation).as_quat()
        shoulders = self.create_shoulders_cyl(header, id, skeleton, median_depth, quaternion, trunk.pose.position)
        hip = self.create_hips_cyl(header, id, skeleton, median_depth, quaternion, trunk.pose.position)
        return (trunk, shoulders, hip)

    def create_neck_spine_cyl(self, header, id, skeleton, median_depth):
        """Create a cylinder for neck and spine."""
        head_mid_point = self.get_mid_head(skeleton)
        if head_mid_point is None:
            # No face landmarks visible; cannot construct spine without a head reference
            return None
        head_mp = Point(x=head_mid_point[0], y=head_mid_point[1], z=median_depth)
        neck_visible = point_not_null(skeleton[Skeleton3D.NECK])
        hips_visible = point_not_null(skeleton[Skeleton3D.LEFT_HIP]) and \
            point_not_null(skeleton[Skeleton3D.RIGHT_HIP])

        if not hips_visible:
            if point_not_null(skeleton[Skeleton3D.LEFT_HIP]):
                hip_visible = skeleton[Skeleton3D.LEFT_HIP]
                hip_visible.z = median_depth
            elif point_not_null(skeleton[Skeleton3D.RIGHT_HIP]):
                hip_visible = skeleton[Skeleton3D.RIGHT_HIP]
                hip_visible.z = median_depth
            else:
                if neck_visible:
                    hip_x = skeleton[Skeleton3D.NECK].x
                    hip_y = skeleton[Skeleton3D.NECK].y + TRUNK_LENGTH - NECK_LENGTH
                    hip_z = median_depth
                    hip_visible = Point(x=hip_x, y=hip_y, z=hip_z)
                else:
                    hip_visible = Point(x=head_mp.x, y=head_mp.y + TRUNK_LENGTH, z=median_depth)
        else:
            mid_hip_x = (skeleton[Skeleton3D.LEFT_HIP].x + skeleton[Skeleton3D.RIGHT_HIP].x) / 2
            mid_hip_y = (skeleton[Skeleton3D.LEFT_HIP].y + skeleton[Skeleton3D.RIGHT_HIP].y) / 2
            mid_hip_z = median_depth
            hip_ep = Point(x=mid_hip_x, y=mid_hip_y, z=mid_hip_z)

        if neck_visible and hips_visible:
            neck_ep = Point(x=skeleton[Skeleton3D.NECK].x, y=skeleton[Skeleton3D.NECK].y - NECK_LENGTH, z=median_depth)
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], neck_ep, hip_ep)
        elif not neck_visible and hips_visible:
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], head_mp, hip_ep)
        elif neck_visible and not hips_visible:
            hip_visible.x = skeleton[Skeleton3D.NECK].x
            neck_ep = Point(x=skeleton[Skeleton3D.NECK].x, y=skeleton[Skeleton3D.NECK].y - NECK_LENGTH, z=median_depth)
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], neck_ep, hip_visible)
        elif not neck_visible and not hips_visible:
            hip_visible.x = head_mp.x
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], head_mp, hip_visible)

        return spine

    def create_shoulders_cyl(self, header, id, skeleton, median_depth, quaternion, trunk_pos):
        """Create the shoulders cylinder."""
        if point_not_null(skeleton[Skeleton3D.NECK]):
            pos = Point(x=trunk_pos.x, y=skeleton[Skeleton3D.NECK].y, z=median_depth)
        elif point_not_null(skeleton[Skeleton3D.RIGHT_SHOULDER]):
            pos = Point(x=trunk_pos.x, y=skeleton[Skeleton3D.RIGHT_SHOULDER].y, z=median_depth)
        elif point_not_null(skeleton[Skeleton3D.LEFT_SHOULDER]):
            pos = Point(x=trunk_pos.x, y=skeleton[Skeleton3D.LEFT_SHOULDER].y, z=median_depth)
        else:
            self.get_logger().error("Shoulders not added. Missing data.")
            return None

        shoulders = self.create_base_marker(header, id, _markers_dict['shoulders'], BGR_ORANGE, Marker.CYLINDER)
        shoulders.pose.position.x = pos.x
        shoulders.pose.position.y = pos.y
        shoulders.pose.position.z = pos.z
        shoulders.scale.x = 0.06
        shoulders.scale.y = 0.06
        shoulders.scale.z = SHOULDERS_LENGTH
        shoulders.pose.orientation.x = quaternion[0]
        shoulders.pose.orientation.y = quaternion[1]
        shoulders.pose.orientation.z = quaternion[2]
        shoulders.pose.orientation.w = quaternion[3]

        return shoulders

    def create_hips_cyl(self, header, id, skeleton, median_depth, quaternion, trunk_pos):
        """Create the hips cylinder."""
        color = BGR_ORANGE
        if point_not_null(skeleton[Skeleton3D.LEFT_HIP]) and point_not_null(skeleton[Skeleton3D.RIGHT_HIP]):
            mid_hip_x = (skeleton[Skeleton3D.LEFT_HIP].x + skeleton[Skeleton3D.RIGHT_HIP].x) / 2
            mid_hip_y = (skeleton[Skeleton3D.LEFT_HIP].y + skeleton[Skeleton3D.RIGHT_HIP].y) / 2
            mid_hip_z = median_depth
            pos = Point(x=mid_hip_x, y=mid_hip_y, z=mid_hip_z)
        elif point_not_null(skeleton[Skeleton3D.LEFT_HIP]):
            pos = Point(x=trunk_pos.x, y=skeleton[Skeleton3D.LEFT_HIP].y, z=median_depth)
        elif point_not_null(skeleton[Skeleton3D.RIGHT_HIP]):
            pos = Point(x=trunk_pos.x, y=skeleton[Skeleton3D.RIGHT_HIP].y, z=median_depth)
        else:
            pos = Point(x=trunk_pos.x, y=trunk_pos.y + TRUNK_LENGTH / 2, z=median_depth)
            color = BGR_GREY

        hips = self.create_base_marker(header, id, _markers_dict['hips'], color, Marker.CYLINDER)
        hips.pose.position.x = pos.x
        hips.pose.position.y = pos.y
        hips.pose.position.z = pos.z
        hips.scale.x = 0.06
        hips.scale.y = 0.06
        hips.scale.z = HIPS_LENGTH
        hips.pose.orientation.x = quaternion[0]
        hips.pose.orientation.y = quaternion[1]
        hips.pose.orientation.z = quaternion[2]
        hips.pose.orientation.w = quaternion[3]

        return hips

    def create_arms(self, header, id, skeleton, shoulders):
        """Create the arms cylinders from the shoulders position."""
        arms = []
        displacement = np.array([0, 0, shoulders.scale.z / 2])
        rotation = R.from_quat([shoulders.pose.orientation.x,
                                shoulders.pose.orientation.y,
                                shoulders.pose.orientation.z,
                                shoulders.pose.orientation.w])
        # Left arm
        if point_not_null(skeleton[Skeleton3D.LEFT_ELBOW]):
            displacement_rotated = rotation.apply(displacement)
            final_position = np.array([shoulders.pose.position.x,
                                       shoulders.pose.position.y,
                                       shoulders.pose.position.z]) + displacement_rotated
            l_shoulder_ep = Point(x=final_position[0], y=final_position[1], z=final_position[2])
            l_top_arm = self.cylinder_from_points(header, id, _markers_dict['left_uparm'], l_shoulder_ep, skeleton[Skeleton3D.LEFT_ELBOW])
            arms.append(l_top_arm)
            if point_not_null(skeleton[Skeleton3D.LEFT_WRIST]):
                l_bot_arm = self.cylinder_from_points(header, id, _markers_dict['left_botarm'], skeleton[Skeleton3D.LEFT_ELBOW], skeleton[Skeleton3D.LEFT_WRIST])
                arms.append(l_bot_arm)

        # Right arm
        if point_not_null(skeleton[Skeleton3D.RIGHT_ELBOW]):
            displacement_rotated = rotation.apply(-displacement)
            final_position = np.array([shoulders.pose.position.x,
                                       shoulders.pose.position.y,
                                       shoulders.pose.position.z]) + displacement_rotated
            r_shoulder_ep = Point(x=final_position[0], y=final_position[1], z=final_position[2])
            r_top_arm = self.cylinder_from_points(header, id, _markers_dict['right_uparm'], r_shoulder_ep, skeleton[Skeleton3D.RIGHT_ELBOW])
            arms.append(r_top_arm)
            if point_not_null(skeleton[Skeleton3D.RIGHT_WRIST]):
                r_bot_arm = self.cylinder_from_points(header, id, _markers_dict['right_botarm'], skeleton[Skeleton3D.RIGHT_ELBOW], skeleton[Skeleton3D.RIGHT_WRIST])
                arms.append(r_bot_arm)

        # Hinges
        if self.display_hinges:
            if point_not_null(skeleton[Skeleton3D.LEFT_ELBOW]):
                l_hinge_sh = self.create_base_marker(header, str(id), _markers_dict['hinge_left_shoulder'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                l_hinge_sh.scale.x = 0.06
                l_hinge_sh.scale.z = 0.06
                l_hinge_sh.scale.y = 0.06
                l_hinge_sh.pose.position.x = l_shoulder_ep.x
                l_hinge_sh.pose.position.y = l_shoulder_ep.y
                l_hinge_sh.pose.position.z = l_shoulder_ep.z
                arms.append(l_hinge_sh)

                if point_not_null(skeleton[Skeleton3D.LEFT_WRIST]):
                    l_hinge_el = self.create_base_marker(header, str(id), _markers_dict['hinge_left_elbow'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                    l_hinge_el.scale.x = 0.06
                    l_hinge_el.scale.z = 0.06
                    l_hinge_el.scale.y = 0.06
                    l_hinge_el.pose.position.x = skeleton[Skeleton3D.LEFT_ELBOW].x
                    l_hinge_el.pose.position.y = skeleton[Skeleton3D.LEFT_ELBOW].y
                    l_hinge_el.pose.position.z = skeleton[Skeleton3D.LEFT_ELBOW].z
                    arms.append(l_hinge_el)

            if point_not_null(skeleton[Skeleton3D.RIGHT_ELBOW]):
                r_hinge_sh = self.create_base_marker(header, str(id), _markers_dict['hinge_right_shoulder'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                r_hinge_sh.scale.x = 0.06
                r_hinge_sh.scale.z = 0.06
                r_hinge_sh.scale.y = 0.06
                r_hinge_sh.pose.position.x = r_shoulder_ep.x
                r_hinge_sh.pose.position.y = r_shoulder_ep.y
                r_hinge_sh.pose.position.z = r_shoulder_ep.z
                arms.append(r_hinge_sh)

                if point_not_null(skeleton[Skeleton3D.RIGHT_WRIST]):
                    r_hinge_el = self.create_base_marker(header, str(id), _markers_dict['hinge_right_elbow'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                    r_hinge_el.scale.x = 0.06
                    r_hinge_el.scale.z = 0.06
                    r_hinge_el.scale.y = 0.06
                    r_hinge_el.pose.position.x = skeleton[Skeleton3D.RIGHT_ELBOW].x
                    r_hinge_el.pose.position.y = skeleton[Skeleton3D.RIGHT_ELBOW].y
                    r_hinge_el.pose.position.z = skeleton[Skeleton3D.RIGHT_ELBOW].z
                    arms.append(r_hinge_el)

        return arms

    def create_legs(self, header, id, skeleton, hips):
        """Create the legs cylinders from the hips position."""
        legs = []
        displacement = np.array([0, 0, hips.scale.z / 2])
        rotation = R.from_quat([hips.pose.orientation.x,
                                hips.pose.orientation.y,
                                hips.pose.orientation.z,
                                hips.pose.orientation.w])
        # Left leg
        if point_not_null(skeleton[Skeleton3D.LEFT_KNEE]):
            displacement_rotated = rotation.apply(displacement)
            final_position = np.array([hips.pose.position.x,
                                       hips.pose.position.y,
                                       hips.pose.position.z]) + displacement_rotated
            l_hip_ep = Point(x=final_position[0], y=final_position[1], z=final_position[2])
            l_top_leg = self.cylinder_from_points(header, id, _markers_dict['left_upleg'], l_hip_ep, skeleton[Skeleton3D.LEFT_KNEE])
            legs.append(l_top_leg)
            if point_not_null(skeleton[Skeleton3D.LEFT_ANKLE]):
                l_bot_leg = self.cylinder_from_points(header, id, _markers_dict['left_botleg'], skeleton[Skeleton3D.LEFT_KNEE], skeleton[Skeleton3D.LEFT_ANKLE])
                legs.append(l_bot_leg)

        # Right leg
        if point_not_null(skeleton[Skeleton3D.RIGHT_KNEE]):
            displacement_rotated = rotation.apply(-displacement)
            final_position = np.array([hips.pose.position.x,
                                       hips.pose.position.y,
                                       hips.pose.position.z]) + displacement_rotated
            r_hip_ep = Point(x=final_position[0], y=final_position[1], z=final_position[2])
            r_top_leg = self.cylinder_from_points(header, id, _markers_dict['right_upleg'], r_hip_ep, skeleton[Skeleton3D.RIGHT_KNEE])
            legs.append(r_top_leg)
            if point_not_null(skeleton[Skeleton3D.RIGHT_ANKLE]):
                r_bot_leg = self.cylinder_from_points(header, id, _markers_dict['right_botleg'], skeleton[Skeleton3D.RIGHT_KNEE], skeleton[Skeleton3D.RIGHT_ANKLE])
                legs.append(r_bot_leg)

        # Hinges
        if self.display_hinges:
            if point_not_null(skeleton[Skeleton3D.LEFT_KNEE]):
                l_hinge_sh = self.create_base_marker(header, str(id), _markers_dict['hinge_left_hip'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                l_hinge_sh.scale.x = 0.06
                l_hinge_sh.scale.z = 0.06
                l_hinge_sh.scale.y = 0.06
                l_hinge_sh.pose.position.x = l_hip_ep.x
                l_hinge_sh.pose.position.y = l_hip_ep.y
                l_hinge_sh.pose.position.z = l_hip_ep.z
                legs.append(l_hinge_sh)

                if point_not_null(skeleton[Skeleton3D.LEFT_ANKLE]):
                    l_hinge_el = self.create_base_marker(header, str(id), _markers_dict['hinge_left_knee'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                    l_hinge_el.scale.x = 0.06
                    l_hinge_el.scale.z = 0.06
                    l_hinge_el.scale.y = 0.06
                    l_hinge_el.pose.position.x = skeleton[Skeleton3D.LEFT_KNEE].x
                    l_hinge_el.pose.position.y = skeleton[Skeleton3D.LEFT_KNEE].y
                    l_hinge_el.pose.position.z = skeleton[Skeleton3D.LEFT_KNEE].z
                    legs.append(l_hinge_el)

            if point_not_null(skeleton[Skeleton3D.RIGHT_KNEE]):
                r_hinge_sh = self.create_base_marker(header, str(id), _markers_dict['hinge_right_hip'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                r_hinge_sh.scale.x = 0.06
                r_hinge_sh.scale.z = 0.06
                r_hinge_sh.scale.y = 0.06
                r_hinge_sh.pose.position.x = r_hip_ep.x
                r_hinge_sh.pose.position.y = r_hip_ep.y
                r_hinge_sh.pose.position.z = r_hip_ep.z
                legs.append(r_hinge_sh)

                if point_not_null(skeleton[Skeleton3D.RIGHT_ANKLE]):
                    r_hinge_el = self.create_base_marker(header, str(id), _markers_dict['hinge_right_knee'], BGR_LIGHT_ORANGE, Marker.SPHERE)
                    r_hinge_el.scale.x = 0.06
                    r_hinge_el.scale.z = 0.06
                    r_hinge_el.scale.y = 0.06
                    r_hinge_el.pose.position.x = skeleton[Skeleton3D.RIGHT_KNEE].x
                    r_hinge_el.pose.position.y = skeleton[Skeleton3D.RIGHT_KNEE].y
                    r_hinge_el.pose.position.z = skeleton[Skeleton3D.RIGHT_KNEE].z
                    legs.append(r_hinge_el)

        return legs

    # ------------------------------------------------------------------ #
    #  3D marker publishing
    # ------------------------------------------------------------------ #

    def publish_3d_markers(self):
        """Generate and publish 3D skeleton MarkerArray for all online persons."""
        final_msg = MarkerArray()

        with self.dict_lock:
            for id, person in self.persons_.items():
                if not person.online or person.landmarks is None:
                    continue

                has_3d = any(point_not_null(p) for p in person.landmarks if p is not None)
                if not has_3d:
                    continue

                # Compute orientation
                orientation = self.get_orientation(
                    person.landmarks, person.theta,
                    person.shoulders_distance, person.hips_distance)

                if not self.check_valid_orientation(orientation, person.theta):
                    orientation = person.theta
                else:
                    person.theta = orientation

                # Create body markers
                body_parts = self.create_fix_body(
                    self.skel3d_header, id, person.landmarks, person.depth, orientation)
                if body_parts is not None:
                    trunk, shoulders, hip = body_parts
                    final_msg.markers.extend([m for m in body_parts if m is not None])

                    # Head
                    head_result = self.fill_head(
                        self.skel3d_header, id, person.landmarks, person.depth, orientation, trunk)
                    if head_result is not None:
                        final_msg.markers.extend(head_result)

                    # Arms
                    if shoulders is not None:
                        arms = self.create_arms(
                            self.skel3d_header, id, person.landmarks, shoulders)
                        if arms is not None:
                            final_msg.markers.extend(arms)

                    # Legs
                    if hip is not None:
                        legs = self.create_legs(
                            self.skel3d_header, id, person.landmarks, hip)
                        if legs is not None:
                            final_msg.markers.extend(legs)

        self.marker_pub_.publish(final_msg)

    # ------------------------------------------------------------------ #
    #  Parameter callback
    # ------------------------------------------------------------------ #

    def parameter_callback(self, params):
        """Handle runtime parameter updates with explicit validation and errors."""
        result = SetParametersResult()
        result.successful = True
        errors = []

        for param in params:
            if param.name == 'processing_rate':
                errors.append("'processing_rate' cannot be changed at runtime")
            elif param.name == 'image_topic':
                errors.append("'image_topic' cannot be changed at runtime")
            elif param.name == 'camera_info_topic':
                errors.append("'camera_info_topic' cannot be changed at runtime")
            elif param.name == 'skel3d_topic':
                errors.append("'skel3d_topic' cannot be changed at runtime")
            elif param.name == 'marker_topic':
                errors.append("'marker_topic' cannot be changed at runtime")
            elif param.name == 'window_name':
                errors.append("'window_name' cannot be changed at runtime")
            elif param.name == 'visual_style':
                errors.append("'visual_style' cannot be changed at runtime")
            elif param.name == 'display_mode':
                if param.value in ['body', 'face', 'both', 'all']:
                    self.display_mode = param.value
                    self.get_logger().info(f'Display mode set to: {self.display_mode}.')
                else:
                    errors.append("'display_mode' must be one of: body, face, both, all")
            elif param.name == 'allow_half_body' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.allow_half_body = param.value
                self.get_logger().info(f'Allow_half_body set to: {self.allow_half_body}.')
            elif param.name == 'allow_back_turned' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.allow_back_turned = param.value
                self.get_logger().info(f'Allow_back_turned set to: {self.allow_back_turned}.')
            elif param.name == 'display_hinges' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.display_hinges = param.value
                self.get_logger().info(f'display_hinges set to: {self.display_hinges}.')
            elif param.name == 'keep_aspect_ratio' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.keep_aspect_ratio = param.value
            elif param.name == 'window_move_step' and param.type_ == rclpy.Parameter.Type.INTEGER:
                self.window_move_step = max(1, int(param.value))
            elif param.name == 'no_signal_timeout' and param.type_ in [rclpy.Parameter.Type.DOUBLE, rclpy.Parameter.Type.INTEGER]:
                self.no_signal_timeout = max(0.0, float(param.value))
            elif param.name in ['window_x', 'window_y', 'window_width', 'window_height'] and param.type_ == rclpy.Parameter.Type.INTEGER:
                setattr(self, param.name, int(param.value))
                if self.window_initialized:
                    try:
                        cv.resizeWindow(self.window_name, max(1, self.window_width), max(1, self.window_height))
                        cv.moveWindow(self.window_name, self.window_x, self.window_y)
                    except cv.error as exc:
                        errors.append(f'Failed applying geometry change for {param.name}: {exc}')
            else:
                errors.append(f'Parameter {param.name} not recognized or incorrect type')

        if errors:
            result.successful = False
            result.reason = '; '.join(errors)

        return result


def main(args=None):
    rclpy.init(args=args)
    node = Node3DPersonViewer()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_window()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
