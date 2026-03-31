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


class NodePersonDisplay(Node):
    """
    ROS 2 Node managing the detection display. It publishes the detections of both bodies and faces.
    If a person has a matched body and face (same ID), it will be displayed in a different color.
    """
    def __init__(self):
        """Initialize parameters, subscriptions, publishers and the timer loop."""
        super().__init__('hri_person_display')

        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing input images.'))
        self.declare_parameter(
            'image_topic', '/color/image_raw', ParameterDescriptor(
                description='Input sensor_msgs/Image topic to visualize.'))
        # Displays modes:
        # - "body": Display only bodies
        # - "face": Display only faces
        # - "both": Display only persons which have a matching body and face
        # - "all": Display all detections
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
            'no_signal_timeout', 2.0, ParameterDescriptor(
                description='Seconds without image frames before rendering "No signal".'))

        self.param_change_callback = self.add_on_set_parameters_callback(self.parameter_callback)

        # Initialize variables
        self.dict_lock = Lock()  # Lock to protect the persons_ dictionary (persons detections)
        self.image_lock = Lock()  # Lock to protect the cv_image_marks variable (image to be displayed)

        self.processing_rate = self.get_parameter('processing_rate').value
        self.image_topic = self.get_parameter('image_topic').value
        self.display_mode = self.get_parameter('display_mode').value
        self.allow_half_body = self.get_parameter('allow_half_body').value
        self.allow_back_turned = self.get_parameter('allow_back_turned').value
        self.no_signal_timeout = float(self.get_parameter('no_signal_timeout').value)
        self.persons_ = {}
        self.image_header = None
        self.image_width = 0
        self.image_height = 0
        self.start_time = self.get_clock().now()
        self.no_signal_active = False
        self.last_image_time = None
        self.cv_image_raw: Optional[np.ndarray] = None
        self.cv_image_marks: Optional[np.ndarray] = None

        if CvBridge is not None:
            self.bridge = CvBridge()
        else:
            self.bridge = None
            self.get_logger().warning(
                'cv_bridge is not installed. Falling back to basic numpy conversion for bgr8/rgb8/mono8.')

        # We want to plot both bodies and faces, so we cannot rely on TimeSynchronizers because:
        # - Running the display with just one detector will block the other one

        # Subscribe to image
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub_ = self.create_subscription(Image, self.image_topic, self.image_callback, qos_sensor_data)

        # Subscribe to body positions
        self.pose_sub_ = self.create_subscription(Skeleton2DList, '/humans/bodies', self.bodies_callback, 1)

        # Subscribe to face positions
        self.face_sub_ = self.create_subscription(Face2DList, '/humans/faces', self.faces_callback, 1)

        # Subscribe to face expression
        self.emotion_sub_ = self.create_subscription(Expression, '/humans/faces/emotion', self.expression_callback, 1)

        # Create publisher for detections
        self.detection_pub_ = self.create_publisher(Image, '/humans/detection', 1)

        # Create timer for publishing detections
        self.proc_timer = self.create_timer(1/self.processing_rate, self.main_callback)

        self.get_logger().info(f"NodePersonDisplay initialized and listening on topic: {self.image_sub_.topic_name}")

    def image_callback(self, msg):
        """Callback to save the raw image to be used for display."""
        self.image_header = msg.header
        self.image_width = msg.width
        self.image_height = msg.height
        self.last_image_time = self.get_clock().now()

        # Convert ROS Image message to OpenCV image
        with self.image_lock:
            self.cv_image_raw = CvBridge().imgmsg_to_cv2(msg, 'bgr8')

    def bodies_callback(self, msg: Skeleton2DList):
        """Update tracked person body data from `/humans/bodies`."""
        for roi_msg, ske_msg, depth_msg in zip(msg.bboxes, msg.skeletons, msg.depths):
            if roi_msg.key != ske_msg.key:
                self.get_logger().error(f'Body id mismatch: [{roi_msg.key}] != [{ske_msg.key}]')
                continue
            if roi_msg.key == '':
                continue

            key = roi_msg.key
            # Handle ROIs
            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            # If body is not known it will always be added here
            with self.dict_lock:
                self.update_position(key, position, body=True)
                self.update_score(key, roi_msg.c, body=True)
                self.update_depth(key, depth_msg)

                # Handle Landmarks
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
            # Handle ROIs
            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            with self.dict_lock:
                # If face is not known it will always be added here
                self.update_position(key, position, face=True)
                self.update_score(key, roi_msg.c, face=True)
                self.update_landmarks(key, ldmks, face=True)

                self.update_times(key, face=True)

    def expression_callback(self, msg):
        """Callback to save the emotion detections data."""
        self.update_expression(msg.key, msg.expression)

        self.update_times(msg.key, emotion=True)

    def main_callback(self):
        """Callback to process and publish the display. It is called by a timer running at the processing rate (parameter)."""
        if self.image_header is None:
            return

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

        self.publish_detection(frame, self.image_header)

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

    # Publish detection for a given person
    def publish_detection(self, frame: np.ndarray, header):
        """Publish the detections of the persons in the image."""
        ids_print = ''

        # Convert OpenCV image back to ROS Image message
        image_marks = CvBridge().cv2_to_imgmsg(frame, "bgr8")
        image_marks.header = header
        self.detection_pub_.publish(image_marks)

        processing_duration_ms = (
            self.get_clock().now() - self.reception_start_proc_time).nanoseconds / 1e6
        self.get_logger().debug(f"Displaying: {ids_print}in {processing_duration_ms}.")


    def update_tracking_status(self):
        """Update online/matched status and remove stale tracked persons."""
        time_check = self.get_clock().now().nanoseconds
        should_delete = []
        # Check times
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
                    self.get_logger().debug(f"Person {id} is online.")
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
        # Ensure body_pos is in the correct format
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

        # Draw skeleton
        if self.persons_[person_id].landmarks is not None:
            self.draw_skeleton(self.cv_image_marks, self.persons_[person_id].landmarks, c_ske)

    def draw_face(self, person_id, color=BGR_BLUE, matched=False):
        """Draw face ROI and text labels for one person."""
        # Ensure body_pos is in the correct format
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
            elif param.name == 'no_signal_timeout' and param.type_ in [rclpy.Parameter.Type.DOUBLE, rclpy.Parameter.Type.INTEGER]:
                self.no_signal_timeout = max(0.0, float(param.value))
            else:
                errors.append(f'Parameter {param.name} not recognized or incorrect type')

        if errors:
            result.successful = False
            result.reason = '; '.join(errors)

        return result


def main(args=None):
    rclpy.init(args=args)
    node = NodePersonDisplay()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
