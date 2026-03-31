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

from hri_msgs.msg import Skeleton2D, Skeleton2DList, Face2DList, Expression
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from sensor_msgs.msg import Image

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

# BGR colors constants
BGR_RED = (0, 0, 255)
BGR_BLUE = (255, 0, 0)
BGR_TEAL = (0, 128, 128)
BGR_GREEN = (0, 255, 0)
BGR_BLACK = (0, 0, 0)
BGR_WHITE = (255, 255, 255)
BGR_GREY = (180, 180, 180)
BGR_DARK_GREEN = (0, 100, 50)

_connections = [
        (Skeleton2D.LEFT_WRIST, Skeleton2D.LEFT_ELBOW),
        (Skeleton2D.RIGHT_WRIST, Skeleton2D.RIGHT_ELBOW),
        (Skeleton2D.LEFT_ELBOW, Skeleton2D.LEFT_SHOULDER),
        (Skeleton2D.RIGHT_ELBOW, Skeleton2D.RIGHT_SHOULDER),
        (Skeleton2D.LEFT_SHOULDER, Skeleton2D.RIGHT_SHOULDER),
        (Skeleton2D.RIGHT_SHOULDER, Skeleton2D.RIGHT_HIP),
        (Skeleton2D.LEFT_SHOULDER, Skeleton2D.LEFT_HIP),
        (Skeleton2D.RIGHT_HIP, Skeleton2D.LEFT_HIP),
        (Skeleton2D.RIGHT_HIP, Skeleton2D.RIGHT_KNEE),
        (Skeleton2D.LEFT_HIP, Skeleton2D.LEFT_KNEE),
        (Skeleton2D.RIGHT_KNEE, Skeleton2D.RIGHT_ANKLE),
        (Skeleton2D.LEFT_KNEE, Skeleton2D.LEFT_ANKLE),
    ]

_face_landmarks = [
        Skeleton2D.NOSE,
        Skeleton2D.LEFT_EYE,
        Skeleton2D.RIGHT_EYE,
        Skeleton2D.LEFT_EAR,
        Skeleton2D.RIGHT_EAR,
]


def bound(val, min_val, max_val):
    """Bound a value between lower and upper limits."""
    return max(min_val, min(val, max_val))


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates [0..1] to bounded image pixel coordinates."""
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


def normalized_to_pixel_coordinates_list(
        coords: List[float], image_width: int, image_height: int) -> (int, int):
    """Convert a coordinate list `[x_norm, y_norm]` into pixel coordinates."""
    return normalized_to_pixel_coordinates(coords[0], coords[1], image_width, image_height)


def point_not_null(point):
    """Return True when a skeleton point contains non-zero coordinates."""
    return point.x != 0 and point.y != 0


class NodePersonViewer(Node):
    """ROS 2 node that overlays HRI detections and renders them in an OpenCV window."""

    def __init__(self):
        """Initialize parameters, subscriptions, timer loop, and the viewer window."""
        super().__init__('hri_person_viewer')

        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing and rendering display frames.'))
        self.declare_parameter(
            'display_mode', 'all', ParameterDescriptor(
                description='Display mode. Options: "body", "face", "both", "all". Default: "all".'))
        self.declare_parameter(
            'allow_half_body', True, ParameterDescriptor(
                description='Allow displaying bodies that are not entirely visible. '
                            'A body is considered whole if at least the head and one shoulder, hip and knee are visible.'))
        self.declare_parameter(
            'allow_back_turned', True, ParameterDescriptor(
                description='Allow displaying bodies that are not facing the camera.'))
        self.declare_parameter(
            'image_topic', '/color/image_raw', ParameterDescriptor(
                description='Input sensor_msgs/Image topic to visualize.'))
        self.declare_parameter(
            'window_name', 'HRI Person Viewer', ParameterDescriptor(
                description='OpenCV window title. Cannot be changed at runtime.'))
        self.declare_parameter(
            'window_x', 400, ParameterDescriptor(
                description='Initial window X position in pixels. Can be changed at runtime.'))
        self.declare_parameter(
            'window_y', 400, ParameterDescriptor(
                description='Initial window Y position in pixels. Can be changed at runtime.'))
        self.declare_parameter(
            'window_width', 1280, ParameterDescriptor(
                description='Initial window width in pixels. Can be changed at runtime.'))
        self.declare_parameter(
            'window_height', 720, ParameterDescriptor(
                description='Initial window height in pixels. Can be changed at runtime.'))
        self.declare_parameter(
            'window_move_step', 50, ParameterDescriptor(
                description='Keyboard move step in pixels for WASD controls.'))
        self.declare_parameter(
            'always_on_top', False, ParameterDescriptor(
                description='Best-effort request for top-most window behavior (X11 backend dependent).'))
        self.declare_parameter(
            'bring_to_front', True, ParameterDescriptor(
                description='Best-effort request to focus the viewer window on startup.'))
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
        self.window_name = self.get_parameter('window_name').value
        self.window_x = int(self.get_parameter('window_x').value)
        self.window_y = int(self.get_parameter('window_y').value)
        self.window_width = int(self.get_parameter('window_width').value)
        self.window_height = int(self.get_parameter('window_height').value)
        self.window_move_step = int(self.get_parameter('window_move_step').value)
        self.always_on_top = self.get_parameter('always_on_top').value
        self.bring_to_front = self.get_parameter('bring_to_front').value
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
        self.window_topmost_supported = hasattr(cv, 'WND_PROP_TOPMOST')
        self.warned_topmost_unsupported = False

        if CvBridge is not None:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warning(
                'cv_bridge is not installed. Falling back to basic numpy conversion for bgr8/rgb8/mono8.')

        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub_ = self.create_subscription(Image, self.image_topic, self.image_callback, qos_sensor_data)
        self.pose_sub_ = self.create_subscription(Skeleton2DList, '/humans/bodies', self.bodies_callback, 1)
        self.face_sub_ = self.create_subscription(Face2DList, '/humans/faces', self.faces_callback, 1)
        self.emotion_sub_ = self.create_subscription(Expression, '/humans/faces/emotion', self.expression_callback, 1)

        self.proc_timer = self.create_timer(1 / max(1, self.processing_rate), self.main_callback)

        self._init_window()

        self.get_logger().info(
            f"NodePersonViewer initialized. Image topic: {self.image_sub_.topic_name}, "
            f"window: '{self.window_name}' ({self.window_x},{self.window_y},{self.window_width},{self.window_height}). "
            'Keys: [q|Esc]=quit, [f]=bring front, [t]=toggle topmost, [WASD]=move window.')

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
            self._apply_window_behaviors(self.bring_to_front)
        except cv.error as exc:
            self.get_logger().error(
                f"Failed to create OpenCV window '{self.window_name}': {exc}. "
                'Check DISPLAY and session type.')
            raise RuntimeError('Failed to initialize viewer window.') from exc

    def _apply_window_behaviors(self, bring_to_front=False):
        """Apply top-most state and optional front-focus request to the viewer window."""
        if not self.window_initialized:
            return

        if self.window_topmost_supported:
            try:
                cv.setWindowProperty(self.window_name, cv.WND_PROP_TOPMOST, 1 if self.always_on_top else 0)
            except cv.error as exc:
                self.get_logger().warning(f'Failed to set always_on_top: {exc}')
        elif self.always_on_top and not self.warned_topmost_unsupported:
            self.warned_topmost_unsupported = True
            self.get_logger().warning(
                'always_on_top requested, but OpenCV backend does not expose WND_PROP_TOPMOST.')

        if bring_to_front:
            if self.window_topmost_supported:
                try:
                    cv.setWindowProperty(self.window_name, cv.WND_PROP_TOPMOST, 1)
                    cv.waitKey(1)
                    if not self.always_on_top:
                        cv.setWindowProperty(self.window_name, cv.WND_PROP_TOPMOST, 0)
                except cv.error as exc:
                    self.get_logger().warning(f'Failed to bring window to front: {exc}')
            else:
                self.get_logger().warning(
                    'bring_to_front requested, but backend cannot force focus. '
                    'Using moveWindow as best effort.')
                cv.moveWindow(self.window_name, self.window_x, self.window_y)

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

    def bodies_callback(self, msg: Skeleton2DList):
        """Update tracked person body data from `/humans/bodies`."""
        for roi_msg, ske_msg, depth_msg in zip(msg.bboxes, msg.skeletons, msg.depths):
            if roi_msg.key != ske_msg.key:
                self.get_logger().error(f'Body id mismatch: [{roi_msg.key}] != [{ske_msg.key}]')
                continue
            if roi_msg.key == '':
                continue

            key = roi_msg.key
            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            with self.dict_lock:
                self.update_position(key, position, body=True)
                self.update_score(key, roi_msg.c, body=True)
                self.update_depth(key, depth_msg)
                self.update_landmarks(key, ske_msg, body=True)
                self.raise_hand(key, ske_msg.skeleton)
                self.whole_body(key, ske_msg.skeleton)
                self.facing(key, ske_msg.skeleton)
                self.update_times(key, body=True)

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
                self.update_position(key, position, face=True)
                self.update_score(key, roi_msg.c, face=True)
                self.update_landmarks(key, ldmks, face=True)
                self.update_times(key, face=True)

    def expression_callback(self, msg: Expression):
        """Update tracked emotions from `/humans/faces/emotion`."""
        self.update_expression(msg.key, msg.expression)
        self.update_times(msg.key, emotion=True)

    def main_callback(self):
        """Main timer loop: update state, render overlays, and present the frame."""
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

        cv.putText(base, 'No msgs received', (30, 60), cv.FONT_HERSHEY_SIMPLEX, 1.5, BGR_RED, 3)
        cv.putText(base, f'Topic: {self.image_topic}', (30, 105), cv.FONT_HERSHEY_SIMPLEX, 0.8, BGR_WHITE, 2)
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
                    if person.matched and (self.display_mode == 'both' or self.display_mode == 'all'):
                        self.draw_body(person_id, c_no_hand=BGR_DARK_GREEN, c_hand_raised=BGR_RED, c_ske=BGR_GREY)
                        self.draw_face(person_id, BGR_TEAL, matched=True)
                    elif person.body_position != [0, 0, 0, 0] and (self.display_mode == 'body' or self.display_mode == 'all'):
                        self.draw_body(person_id, c_no_hand=BGR_BLUE, c_hand_raised=BGR_RED, c_ske=BGR_GREY)
                    elif person.face_position != [0, 0, 0, 0] and (self.display_mode == 'face' or self.display_mode == 'all'):
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
        if key == ord('f'):
            self._apply_window_behaviors(True)
            return
        if key == ord('t'):
            self.always_on_top = not self.always_on_top
            self._apply_window_behaviors(False)
            self.get_logger().info(f'always_on_top toggled to: {self.always_on_top}')
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

    def update_tracking_status(self):
        """Update online/matched status and remove stale tracked persons."""
        time_check = self.get_clock().now().nanoseconds
        should_delete = []
        with self.dict_lock:
            for person_id, person in self.persons_.items():
                body_detected = time_check - person.times['body'].nanoseconds <= TIME_MARGIN_DETECTION * 1e9
                face_detected = time_check - person.times['face'].nanoseconds <= TIME_MARGIN_DETECTION * 1e9

                if not body_detected:
                    person.body_position = [0, 0, 0, 0]
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
                        should_delete.append(person_id)

                if time_check - person.times['emotion'].nanoseconds > TIME_MARGIN_DETECTION * 1e9:
                    person.emotion = ''

            for person_id in should_delete:
                self.get_logger().debug(f'Removing person {person_id}.')
                del self.persons_[person_id]

    def update_position(self, person_id, position, body=False, face=False):
        """Update body or face ROI for a person id."""
        if person_id not in self.persons_:
            self.get_logger().debug(f'Adding person with key {person_id}.')
            self.persons_[person_id] = PersonDetection()

        if body:
            self.persons_[person_id].body_position = position
        elif face:
            self.persons_[person_id].face_position = position

    def update_score(self, person_id, score, body=False, face=False):
        """Update body or face confidence score for a person id."""
        if person_id not in self.persons_:
            self.get_logger().error(f'Body id [{person_id}] not found when assigning score.')
            return
        if body:
            self.persons_[person_id].body_score = score
        elif face:
            self.persons_[person_id].face_score = score

    def update_expression(self, face_id, expression):
        """Update expression label for a tracked face id."""
        if face_id not in self.persons_:
            self.get_logger().warning(f'Face id [{face_id}] not found when assigning expression.')
            return
        self.persons_[face_id].emotion = expression

    def update_depth(self, body_id, depth):
        """Update depth value for a tracked body id."""
        if body_id not in self.persons_:
            self.get_logger().warning(f'Body id [{body_id}] not found when assigning depth.')
            return
        self.persons_[body_id].depth = depth

    def update_landmarks(self, person_id, msg, body=False, face=False):
        """Update body skeleton landmarks or face landmarks for a person id."""
        if body and face:
            self.get_logger().error('Cannot update both body and face landmarks at the same time.')
            return
        if body:
            self.persons_[person_id].landmarks = msg.skeleton
        elif face:
            self.persons_[person_id].face_landmarks = msg.landmarks

    def update_times(self, person_id, body=False, face=False, emotion=False, voice=False):
        """Update per-signal timestamps used for stale detection and cleanup."""
        if body:
            self.persons_[person_id].times['body'] = self.get_clock().now()
        if face:
            self.persons_[person_id].times['face'] = self.get_clock().now()
        if emotion:
            if person_id not in self.persons_:
                self.get_logger().warning(f'ID [{person_id}] not found when assigning expression.')
                return
            self.persons_[person_id].times['emotion'] = self.get_clock().now()
        if voice:
            if person_id not in self.persons_:
                self.get_logger().warning(f'ID [{person_id}] not found when assigning voice.')
                return
            self.persons_[person_id].times['voice'] = self.get_clock().now()

    def raise_hand(self, body_id, skeleton):
        """Infer whether a person is raising the left hand."""
        if self.allow_back_turned:
            self.get_logger().warning(
                'Back turned is allowed. A hand will not be considered raised if no face is visible.', once=True)
        if self.persons_[body_id].facing == 0:
            self.persons_[body_id].hand_raised = False
            return

        face_ref = None
        for face_point in _face_landmarks:
            if point_not_null(skeleton[face_point]):
                face_ref = face_point
                break

        if face_ref is not None and skeleton[Skeleton2D.LEFT_WRIST].y != 0 and \
                skeleton[Skeleton2D.LEFT_WRIST].y < skeleton[face_ref].y:
            self.persons_[body_id].hand_raised = True
        else:
            self.persons_[body_id].hand_raised = False

    def whole_body(self, body_id, skeleton):
        """Compute if enough joints are visible to consider the body as whole."""
        head_seen = False
        shoulder_seen = False
        hips_seen = False
        knees_seen = False

        if (skeleton[Skeleton2D.NOSE].x != 0 and skeleton[Skeleton2D.NOSE].y != 0) or \
                (skeleton[Skeleton2D.LEFT_EAR].x != 0 and skeleton[Skeleton2D.LEFT_EAR].y != 0) or \
                (skeleton[Skeleton2D.RIGHT_EAR].x != 0 and skeleton[Skeleton2D.RIGHT_EAR].y != 0) or \
                (skeleton[Skeleton2D.LEFT_EYE].x != 0 and skeleton[Skeleton2D.LEFT_EYE].y != 0) or \
                (skeleton[Skeleton2D.RIGHT_EYE].x != 0 and skeleton[Skeleton2D.RIGHT_EYE].y != 0):
            head_seen = True
        if (skeleton[Skeleton2D.LEFT_SHOULDER].x != 0 and skeleton[Skeleton2D.LEFT_SHOULDER].y != 0) or \
                (skeleton[Skeleton2D.RIGHT_SHOULDER].x != 0 and skeleton[Skeleton2D.RIGHT_SHOULDER].y != 0):
            shoulder_seen = True
        if (skeleton[Skeleton2D.LEFT_HIP].x != 0 and skeleton[Skeleton2D.LEFT_HIP].y != 0) or \
                (skeleton[Skeleton2D.RIGHT_HIP].x != 0 and skeleton[Skeleton2D.RIGHT_HIP].y != 0):
            hips_seen = True
        if (skeleton[Skeleton2D.LEFT_KNEE].x != 0 and skeleton[Skeleton2D.LEFT_KNEE].y != 0) or \
                (skeleton[Skeleton2D.RIGHT_KNEE].x != 0 and skeleton[Skeleton2D.RIGHT_KNEE].y != 0):
            knees_seen = True

        self.persons_[body_id].whole_body = head_seen and shoulder_seen and hips_seen and knees_seen

    def facing(self, body_id, skeleton):
        """Estimate whether the person is facing the camera using visible facial points."""
        face_points = 0
        if skeleton[Skeleton2D.NOSE].x != 0 and skeleton[Skeleton2D.NOSE].y != 0:
            face_points += 1
        if skeleton[Skeleton2D.LEFT_EAR].x != 0 and skeleton[Skeleton2D.LEFT_EAR].y != 0:
            face_points += 1
        if skeleton[Skeleton2D.RIGHT_EAR].x != 0 and skeleton[Skeleton2D.RIGHT_EAR].y != 0:
            face_points += 1
        if skeleton[Skeleton2D.LEFT_EYE].x != 0 and skeleton[Skeleton2D.LEFT_EYE].y != 0:
            face_points += 1
        if skeleton[Skeleton2D.RIGHT_EYE].x != 0 and skeleton[Skeleton2D.RIGHT_EYE].y != 0:
            face_points += 1

        self.persons_[body_id].facing = face_points

    def should_draw_ske_line(self, start_point, end_point, landmarks):
        """Return True when a skeleton line segment has valid endpoints."""
        return (landmarks[start_point] is not None and landmarks[end_point] is not None
                and landmarks[start_point].x != 0 and landmarks[end_point].x != 0
                and landmarks[start_point].y != 0 and landmarks[end_point].y != 0
                and landmarks[start_point].x is not None and landmarks[end_point].x is not None
                and landmarks[start_point].y is not None and landmarks[end_point].y is not None)

    def draw_skeleton(self, image, landmarks, color=BGR_GREY):
        """Draw skeleton connections on the destination image."""
        for start_point, end_point in _connections:
            if self.should_draw_ske_line(start_point, end_point, landmarks):
                start_coords = [landmarks[start_point].x, landmarks[start_point].y]
                end_coords = [landmarks[end_point].x, landmarks[end_point].y]

                start_pixel = normalized_to_pixel_coordinates_list(start_coords, self.image_width, self.image_height)
                end_pixel = normalized_to_pixel_coordinates_list(end_coords, self.image_width, self.image_height)

                cv.line(image, start_pixel, end_pixel, color, 2)

    def draw_body(self, person_id, c_ske=BGR_GREY, c_no_hand=BGR_BLUE, c_hand_raised=BGR_RED):
        """Draw body ROI, score/depth label, and skeleton for one person."""
        pt1 = normalized_to_pixel_coordinates(
            self.persons_[person_id].body_position[0], self.persons_[person_id].body_position[1],
            self.image_width, self.image_height)
        pt2 = normalized_to_pixel_coordinates(
            self.persons_[person_id].body_position[2], self.persons_[person_id].body_position[3],
            self.image_width, self.image_height)
        score = str('{:.2f}'.format(self.persons_[person_id].body_score * 100))

        if self.persons_[person_id].hand_raised:
            cv.rectangle(self.cv_image_marks, pt1, pt2, c_hand_raised, 2)
        else:
            cv.rectangle(self.cv_image_marks, pt1, pt2, c_no_hand, 2)

        text = 'ID: ' + person_id + ' (' + score + '%)'
        if self.persons_[person_id].depth != 0:
            text += ' Dist: ' + str(self.persons_[person_id].depth)

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

    def should_display_person(self, person):
        """Apply configured body visibility and facing filters for display."""
        half_body = self.allow_half_body or person.whole_body
        facing = self.allow_back_turned or person.facing >= 4
        return half_body and facing

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
            elif param.name == 'window_name':
                errors.append("'window_name' cannot be changed at runtime")
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
            elif param.name == 'always_on_top' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.always_on_top = param.value
                self._apply_window_behaviors(False)
            elif param.name == 'bring_to_front' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.bring_to_front = param.value
                if self.bring_to_front:
                    self._apply_window_behaviors(True)
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
    node = NodePersonViewer()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_window()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
