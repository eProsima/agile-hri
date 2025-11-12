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
import rclpy
import rclpy.duration
import rclpy.parameter

from geometry_msgs.msg import Point
from hri_msgs.msg import Skeleton3D, Skeleton3DList
from rcl_interfaces.msg import ParameterDescriptor, SetParametersResult
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from scipy.spatial.transform import Rotation as R
from threading import Lock
from typing import List
import numpy as np

from hri_detection_display.PersonDetectionTracker import PersonDetection

# Max number of calls to the main_process (timer_loop) that a body/face can be without updates
# before it is not tracked anymore
MAX_ITERATIONS_RETENTION = 15

# Time margin to consider a body/face as detected again
TIME_MARGIN_DETECTION = 0.2

# Anatomy constants
NECK_LENGTH = 0.1
SHOULDERS_LENGTH = 0.35
HIPS_LENGTH = 0.22
# Limits for hips and shoulders
MAX_HIP_WIDTH = 0.35
MIN_HIP_WIDTH = 0.14
MAX_SHOULDER_WIDTH = 0.5
MIN_SHOULDER_WIDTH = 0.2
TRUNK_LENGTH = 0.6
MAX_BODY_EXTENSION = 0.8

# BGR colors
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


_connections = [
        (Skeleton3D.LEFT_WRIST, Skeleton3D.LEFT_ELBOW, 'left_botarm_cylinder'),
        (Skeleton3D.RIGHT_WRIST, Skeleton3D.RIGHT_ELBOW, 'right_botarm_cylinder'),
        (Skeleton3D.LEFT_ELBOW, Skeleton3D.LEFT_SHOULDER, 'left_uparm_cylinder'),
        (Skeleton3D.RIGHT_ELBOW, Skeleton3D.RIGHT_SHOULDER, 'right_uparm_cylinder'),
        (Skeleton3D.LEFT_SHOULDER, Skeleton3D.RIGHT_SHOULDER, 'shoulders_cylinder'),
        (Skeleton3D.RIGHT_HIP, Skeleton3D.LEFT_HIP, 'hips_cylinder'),
        (Skeleton3D.RIGHT_HIP, Skeleton3D.RIGHT_KNEE, 'right_upleg_cylinder'),
        (Skeleton3D.LEFT_HIP, Skeleton3D.LEFT_KNEE, 'left_upleg_cylinder'),
        (Skeleton3D.RIGHT_KNEE, Skeleton3D.RIGHT_ANKLE, 'right_botleg_cylinder'),
        (Skeleton3D.LEFT_KNEE, Skeleton3D.LEFT_ANKLE, 'left_botleg_cylinder'),
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
    "body": 1,  # Neck and spine
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
    """Bound a value between min_val and max_val."""
    return max(min_val, min(val, max_val))


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates to pixel coordinates."""
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


def point_not_null(point):
    """Check if a point is not null."""
    if point.x != 0 and point.y != 0 and point.z != 0:
        return True
    return False


def color_to_msg(color):
    """Convert a BGR color to a ROS 2 ColorRGBA message."""
    msg = ColorRGBA()
    msg.r = color[2] / 255.0
    msg.g = color[1] / 255.0
    msg.b = color[0] / 255.0
    msg.a = 1.0
    return msg


class NodeSkeleton3DDisplay(Node):
    """
    ROS 2 Node managing the 3D skeleton display. It publishes a Marker array containing all the body
    elements of each person. It uses the Skeleton3DList message to get the 3D skeleton data.
    """
    def __init__(self):

        # Initialize node
        super().__init__('hri_skel3D_display')

        self.declare_parameter(
            'allow_half_body', True, ParameterDescriptor(
                description='Allow displaying bodies that are not entirely visible. \
                      A body is considered whole if at least the head and one shoulder, hip and knee are visible.'))
        # TODO Carlos: Parameter not yet implemented
        self.declare_parameter(
            'allow_back_turned', True, ParameterDescriptor(
                description='Allow displaying bodies that are not facing the camera.'))
        self.declare_parameter(
            'visual_style', 'cylinder', ParameterDescriptor(
                description='Allow changing the visual style of the skeletons. Options: "stripes", "cylinder".'))
        self.declare_parameter(
            'processing_rate', 20, ParameterDescriptor(
                description='Best effort frequency for processing input images.'))
        self.declare_parameter(
            'display_hinges', True, ParameterDescriptor(
                description='Display hinges for each joint.'))

        if self.get_parameter('visual_style').value not in ['stripes', 'cylinder']:
            self.get_logger().warning(f"Value {self.get_parameter('visual_style').value} for parameter 'visual_style' \
                                      not recognized. It should be 'stripes' or 'cylinder'.")
            self.get_parameter('visual_style').set_parameter('cylinder')

        self.param_change_callback = self.add_on_set_parameters_callback(self.parameter_callback)

        # Initialize variables
        self.allow_half_body = self.get_parameter('allow_half_body').value
        # TODO Carlos: Parameter not yet implemented
        self.allow_back_turned = self.get_parameter('allow_back_turned').value
        self.visual_style = self.get_parameter('visual_style').value
        self.processing_rate = self.get_parameter('processing_rate').value
        self.display_hinges = self.get_parameter('display_hinges').value
        self.persons_ = {}
        self.dict_lock = Lock()
        self.header = None

        # Subscribe to body positions
        self.pose_sub_ = \
            self.create_subscription(Skeleton3DList, '/humans/bodies/skel3D', self.skel3D_callback_cylinder, 1)

        # Create publisher for detections
        self.marker_pub_ = self.create_publisher(MarkerArray, '/humans/detection/skel3D', 1)

        # Create timer for publishing detections
        self.proc_timer = self.create_timer(1/self.processing_rate, self.main_callback)

        self.get_logger().info(f"NodeSkeleton3DDisplay initialized and listening on topic: {self.pose_sub_.topic_name}")

    def create_base_marker(self, header, ns, id, color, type):
        """Creates a base marker msg."""
        marker = Marker()
        marker.header = header
        if marker.header.frame_id == "_depth_optical_frame" or marker.header.frame_id == "_color_optical_frame":
            # Use frame_id specified in URDF
            marker.header.frame_id = "camera_depth_optical_frame"
        marker.ns = ns
        marker.id = id
        marker.type = type
        marker.action = Marker.ADD
        marker.scale.x = 0.06
        marker.scale.y = 0.23
        marker.scale.z = 0.2
        marker.color = color_to_msg(color)
        # 0.5 seconds of lifetime
        marker.lifetime = rclpy.duration.Duration(nanoseconds=1e8).to_msg()

        return marker

    def get_mid_head(self, skeleton):
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
        if trunk is not None:
            mid_point = [0, 0]
            mid_point[0] = trunk.pose.position.x
            mid_point[1] = trunk.pose.position.y - trunk.scale.z/2 - NECK_LENGTH
        else:
            mid_point = self.get_mid_head(skeleton)

        # Create quaternion from euler angles
        quaternion = R.from_euler('y', -orientation-np.pi/2).as_quat()

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
            # Eyes
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
            self.get_logger().debug(f"Adding head: {head.ns}, {head.id}, [{head.pose.position.x}, {head.pose.position.y}, {head.pose.position.z}]")
            return [head, left_eye, right_eye]
        return None

    def skel3D_callback_cylinder(self, msg: Skeleton3DList):
        """Callback to save data of last persons."""

        for skel in msg.skeletons:
            # Skip empty skeletons or those with only half of the body visible
            if skel.key == '' or not self.whole_body(skel.skeleton):
                continue
            self.header = msg.header
            # Get median depth of body points
            with self.dict_lock:
                self.update_landmarks(skel.key, skel)
                median_depth = self.get_median_depth(skel.skeleton)
                self.update_median_depth(skel.key, median_depth)
                self.update_shoulders_hips_length(skel.key)

                self.update_times(skel.key)

    def main_callback(self):
        """Callback to process and publish the skeletons with cylinder."""
        if self.header is None:
            return

        self.reception_start_proc_time = self.get_clock().now()
        time_check = self.get_clock().now().nanoseconds
        should_delete = []
        final_msg = MarkerArray()

        with self.dict_lock:
            for id, person in self.persons_.items():
                self.get_logger().debug(f"Processing skeleton: {id}")

                if time_check - person.times["body"].nanoseconds > TIME_MARGIN_DETECTION * 1e9:
                    person.online = False
                    person.frames_since_last_detection += 1
                    self.get_logger().debug(f"Person {id} not seen for: {self.persons_[id].frames_since_last_detection} frames.")
                    if self.persons_[id].frames_since_last_detection > MAX_ITERATIONS_RETENTION:
                        should_delete.append(id)
                else:
                    person.online = True
                    self.get_logger().debug(f"Person {id} is online.")
                    if person.frames_since_last_detection > 0:
                        person.frames_since_last_detection = 0

            for id in should_delete:
                self.get_logger().debug(f"Removing person {id}.")
                del self.persons_[id]

            # Local dict to free the lock as soon as possible
            local_dict = self.persons_

        for id, person in local_dict.items():
            if person.online:
                # Create body
                orientation = self.get_orientation(person.landmarks,
                                                   self.persons_[id].theta,
                                                   self.persons_[id].shoulders_distance,
                                                   self.persons_[id].hips_distance)
                if not self.check_valid_orientation(orientation, self.persons_[id].theta):
                    if orientation is None:
                        self.get_logger().debug(f"Using last orientation for body {id}: {orientation}||{self.persons_[id].theta}")
                    else:
                        self.get_logger().debug(f"Using last orientation for body {id}: {orientation}||{self.persons_[id].theta}= {abs(orientation - self.persons_[id].theta)}")
                    # Orientation not computed or too different from last one (probably an occlusion occured)
                    orientation = self.persons_[id].theta
                else:
                    self.get_logger().debug(f"Using computed orientation for body {id}: {orientation}||{self.persons_[id].theta}= {abs(orientation - self.persons_[id].theta)}")
                    self.persons_[id].theta = orientation

                trunk, shoulders, hip = self.create_fix_body(self.header, id, person.landmarks, person.depth, orientation)
                final_msg.markers.extend([trunk, shoulders, hip])
                # Use sphere for head and cylinders for each arm, each leg and the body
                head, left_eye, right_eye = self.fill_head(self.header, id, person.landmarks, person.depth, orientation, trunk)
                if head is not None:
                    final_msg.markers.extend([head, left_eye, right_eye])
                arms = self.create_arms(self.header, id, person.landmarks, shoulders)
                if arms is not None:
                    final_msg.markers.extend(arms)
                legs = self.create_legs(self.header, id, person.landmarks, hip)
                if legs is not None:
                    final_msg.markers.extend(legs)

        self.marker_pub_.publish(final_msg)

    def check_valid_orientation(self, orientation, last_orientation):
        """Check if the orientation is too different from the last one. Returns 'True' if it is valid."""
        if orientation is not None:
            if last_orientation == -1.57:
                # This is the first orientation computed for the body. Assume it is valid
                return True
            diff = 0
            # Both orientations are positive (positive Z axis)
            if orientation >= 0 and last_orientation >= 0:
                diff = abs(orientation - last_orientation)
            # Both orientations are negative (negative Z axis)
            elif orientation <= 0 and last_orientation <= 0:
                diff = abs(orientation - last_orientation)
            # Orientations have different sign (one positive Z axis and one negative Z axis)
            else:
                if orientation < 0:
                    # diff = abs(orientation + np.pi - last_orientation)
                    diff = abs(2 * np.pi + orientation - last_orientation)
                else:
                    # diff = abs(orientation - np.pi - last_orientation)
                    diff = abs(2 * np.pi + last_orientation - orientation)
            # TODO Carlos: fine tune this value so we can avoid occlusions. A lower value will restrict more but it might not
            # work in a full rotation, specially at -pi and pi. If there is a big difference between the real distance of shoulders in X axis
            # and the constant SHOULDERS_LENGTH, this value should be increased, because there will exist jumps between the
            # limit angles computed.
            if diff < 1.0:
                self.get_logger().debug(f"Orientation is valid with diff: {diff}.")
                return True
            else:
                self.get_logger().debug(f"Orientation is not valid with diff: {diff} because: {orientation} and last is: {last_orientation}.")
        # Orientation not computed or too different from last one (probably an occlusion occured)
        return False

    def whole_body(self, landmarks):
        """Returns 'True' if a full view of the body is available. 'False' otherwise."""
        head_seen = False
        shoulder_seen = False
        if self.allow_half_body:
            hips_seen = True
        else:
            hips_seen = False

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

        if head_seen and shoulder_seen and hips_seen:
            return True
        return False

    def update_landmarks(self, id, msg):
        """Update the landmarks of the given body/face."""
        # New person
        if id not in self.persons_:
            self.get_logger().info(f"Adding person with key {id}.")
            self.persons_[id] = PersonDetection()
            self.persons_[id].shoulders_distance = SHOULDERS_LENGTH
            self.persons_[id].hips_distance = HIPS_LENGTH
            self.persons_[id].landmarks = msg.skeleton
            return
        # Person already known
        for idx, point in enumerate(msg.skeleton):
            if not point_not_null(point):
                # Null point means that the point is not visible, need to update to stop displaying it
                self.persons_[id].landmarks[idx] = point
            else:
                # Always update x and y
                self.persons_[id].landmarks[idx].x, self.persons_[id].landmarks[idx].y = point.x, point.y
                # Check if depth is not null
                if point.z != 0:
                    # Avoid assigning a point to a body if it is too far from the median depth (YOLO marks a point out of the body boundary)
                    if abs(point.z - self.persons_[id].depth) < MAX_BODY_EXTENSION:
                        self.persons_[id].landmarks[idx].z = point.z
                    else:
                        self.get_logger().debug(f"Point [{idx}] out of range with {point.z} and median: {self.persons_[id].depth} for body [{id}].")
                else:
                    self.get_logger().debug(f"Point [{idx}] with null depth for body [{id}].")

    def update_median_depth(self, id, depth):
        """Updates the median_depth of the given body"""
        if id not in self.persons_:
            self.get_logger().warning(f"Body id [{id}] not found when assigning depth.")
            return
        self.persons_[id].depth = depth

    def update_times(self, id):
        """Updates the last time update of the given body"""
        if id not in self.persons_:
            self.get_logger().warning(f"Body id [{id}] not found when assigning depth.")
            return
        self.persons_[id].times["body"] = self.get_clock().now()

    def update_shoulders_hips_length(self, id):
        """Updates the shoulders distance of the given body"""
        if id not in self.persons_:
            self.get_logger().warning(f"Body id [{id}] not found when updating shoulders distance.")
            return
        # Shoulders distance is updated only once
        if self.persons_[id].shoulders_distance == SHOULDERS_LENGTH:
            # Check if shoulders are visible
            left_sh = self.persons_[id].landmarks[Skeleton3D.LEFT_SHOULDER]
            right_sh = self.persons_[id].landmarks[Skeleton3D.RIGHT_SHOULDER]
            if point_not_null(left_sh) and \
                    point_not_null(right_sh):
                # Check if depths are similar and x_dist has a valid value
                x_dist = abs(left_sh.x - right_sh.x)
                # TODO Carlos: fine tune these values with experimental tests
                if abs(left_sh.z - right_sh.z) < 0.005 and x_dist > SHOULDERS_LENGTH - 0.1:
                    # Calculate distance
                    self.persons_[id].shoulders_distance = x_dist - 0.03
                    self.get_logger().info(f"Shoulders distance for body {id} updated to: {self.persons_[id].shoulders_distance}.")
        # Hips distance is updated only once
        if self.persons_[id].hips_distance == HIPS_LENGTH:
            # Check if shoulders are visible
            left_hip = self.persons_[id].landmarks[Skeleton3D.LEFT_HIP]
            right_hip = self.persons_[id].landmarks[Skeleton3D.RIGHT_HIP]
            if point_not_null(left_hip) and \
                    point_not_null(right_hip):
                # Check if depths are similar and x_dist has a valid value
                x_dist = abs(left_hip.x - right_hip.x)
                # TODO Carlos: fine tune these values with experimental tests
                if abs(left_hip.z - right_hip.z) < 0.005 and x_dist > HIPS_LENGTH - 0.06:
                    # Calculate distance
                    self.persons_[id].hips_distance = x_dist
                    # self.get_logger().info(f"Hips distance for body {id} updated to: {self.persons_[id].hips_distance}.")

    def get_median_depth(self, landmarks):
        """Get the median depth of the body points."""
        valid_points = [landmarks[point] for point in _body_landmarks if point_not_null(landmarks[point])]
        if len(valid_points) == 0:
            return 0
        return np.median([point.z for point in valid_points])

    def create_fix_body(self, header, id, skeleton, median_depth, orientation):
        """Create a body marker."""
        trunk = self.create_neck_spine_cyl(header, id, skeleton, median_depth)
        # Create quaternion from euler angles
        quaternion = R.from_euler('y', -orientation).as_quat()
        shoulders = self.create_shoulders_cyl(header, id, skeleton, median_depth, quaternion, trunk.pose.position)
        hip = self.create_hips_cyl(header, id, skeleton, median_depth, quaternion, trunk.pose.position)
        return [trunk, shoulders, hip]

    def cylinder_from_points(self, header, ns, id, p1, p2):
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
        direction_normalized = direction / length

        # Create a quaternion for the orientation
        z_axis = np.array([0.0, 0.0, 1.0])
        dot_product = np.dot(z_axis, direction_normalized)

        if np.isclose(dot_product, 1.0):
            # The direction is aligned with z_axis (no rotation needed)
            quaternion = np.array([0.0, 0.0, 0.0, 1.0])
        elif np.isclose(dot_product, -1.0):
            # The direction is anti-aligned with z_axis (180 degrees rotation around any perpendicular axis)
            quaternion = np.array([1.0, 0.0, 0.0, 0.0])  # Rotate 180 degrees around x-axis
        else:
            # General case: direction and z_axis are not aligned nor anti-aligned
            axis = np.cross(z_axis, direction_normalized)
            axis = axis / np.linalg.norm(axis)  # Normalize the axis of rotation
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
            left_shoulder = np.array([skeleton[Skeleton3D.LEFT_SHOULDER].x, skeleton[Skeleton3D.LEFT_SHOULDER].y, skeleton[Skeleton3D.LEFT_SHOULDER].z])
            right_shoulder = np.array([skeleton[Skeleton3D.RIGHT_SHOULDER].x, skeleton[Skeleton3D.RIGHT_SHOULDER].y, skeleton[Skeleton3D.RIGHT_SHOULDER].z])

            x_dist = (skeleton[Skeleton3D.LEFT_SHOULDER].x - skeleton[Skeleton3D.RIGHT_SHOULDER].x)

            facing_positive_x = True
            if abs(x_dist) < 0.08:
                # This means that we are very close to being sideways, we can assume orthogonal rotation
                if abs(abs(last_orientation) - (np.pi)) < abs(last_orientation):
                    self.get_logger().debug(f"Returning sh ortho -pi/2 because shoulders x_dist is: {x_dist}.")
                    return -np.pi
                else:
                    self.get_logger().debug(f"Returning sh ortho +0 because shoulders x_dist is: {x_dist}.")
                    return 0
            else:
                # We are not close to being sideways, so we can rely on depth of shoulders to calculate
                # which side we are facing
                if skeleton[Skeleton3D.LEFT_SHOULDER].z < skeleton[Skeleton3D.RIGHT_SHOULDER].z:
                    # Left shoulder is closer to the camera
                    self.get_logger().debug(f"Right shoulder farther than left shoulder and x_dist is: {x_dist}.")
                    facing_positive_x = False
                else:
                    self.get_logger().debug(f"Left shoulder farther than right shoulder and x_dist is: {x_dist}.")

            cos_theta = x_dist / sh_dist
            cos_theta = bound(cos_theta, -1, 1)

            theta = np.arccos(cos_theta)

            if left_shoulder[0] > right_shoulder[0]:
                facing_camera = True
            else:
                facing_camera = False

            theta = theta if facing_positive_x else -theta

            self.get_logger().debug(f"Shoulder Orientation: {theta}, x_dist: {x_dist} facing camera: {facing_camera} from: {left_shoulder} to {right_shoulder}")
            return (theta - np.pi/2)
            # return (theta - np.pi/2) if theta > 0 else (theta + np.pi/2)

        # Try to use hips to compute orientation
        if try_use_hips:
            left_hip = np.array([skeleton[Skeleton3D.LEFT_HIP].x, skeleton[Skeleton3D.LEFT_HIP].y, skeleton[Skeleton3D.LEFT_HIP].z])
            right_hip = np.array([skeleton[Skeleton3D.RIGHT_HIP].x, skeleton[Skeleton3D.RIGHT_HIP].y, skeleton[Skeleton3D.RIGHT_HIP].z])

            x_dist = (skeleton[Skeleton3D.LEFT_HIP].x - skeleton[Skeleton3D.RIGHT_HIP].x)

            facing_positive_x = True
            if abs(x_dist) < 0.08:
                # This means that we are very close to being sideways, we can assume orthogonal rotation
                if abs(abs(last_orientation) - (np.pi)) < abs(last_orientation):
                    self.get_logger().debug(f"Returning hip ortho -pi/2 because hip x_dist is: {x_dist}.")
                    return -np.pi
                else:
                    self.get_logger().debug(f"Returning hip ortho +0 because hip x_dist is: {x_dist}.")
                    return 0
            else:
                # We are not close to being sideways, so we can rely on depth of hips to calculate
                # which side we are facing
                if skeleton[Skeleton3D.LEFT_HIP].z < skeleton[Skeleton3D.RIGHT_HIP].z:
                    # Left hip is closer to the camera
                    self.get_logger().debug("Right hip farther than left hip.")
                    facing_positive_x = False
                else:
                    self.get_logger().debug("Left hip farther than right hip.")

            cos_theta = x_dist / hips_dist
            cos_theta = bound(cos_theta, -1, 1)

            theta = np.arccos(cos_theta)

            if left_hip[0] > right_hip[0]:
                facing_camera = True
            else:
                facing_camera = False

            theta = theta if facing_positive_x else -theta

            self.get_logger().debug(f"Hips Orientation: {theta}, facing camera: {facing_camera} from: {left_hip} to {right_hip}")
            return (theta - np.pi/2)
        # Not enough data to compute orientation, using last orientation saved or default orientation
        return None

    def create_neck_spine_cyl(self, header, id, skeleton, median_depth):
        """Create a cylinder for neck and spine."""
        head_mid_point = self.get_mid_head(skeleton)
        head_mp = Point(x=head_mid_point[0], y=head_mid_point[1], z=median_depth)
        # Neck visible implies shoulders are visible
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

        # Neck and hips are visible
        if neck_visible and hips_visible:
            # Offset for the neck in the y-axis (vertical)
            neck_ep = Point(x=skeleton[Skeleton3D.NECK].x, y=skeleton[Skeleton3D.NECK].y - NECK_LENGTH, z=median_depth)
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], neck_ep, hip_ep)
            self.get_logger().debug(f"Adding spine from neck and hips: {spine.ns}, {spine.id}, [{neck_ep} || {hip_ep}]")
        # Neck is not visible but both hips are visible (one shoulder missing)
        elif not neck_visible and hips_visible:
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], head_mp, hip_ep)
            self.get_logger().debug(f"Adding spine from head and hips: {spine.ns}, {spine.id}, [{skeleton[Skeleton3D.NECK]} || {hip_ep}]")
        # Neck is visible but one hip is missing
        elif neck_visible and not hips_visible:
            hip_visible.x = skeleton[Skeleton3D.NECK].x  # Align hip with neck
            neck_ep = Point(x=skeleton[Skeleton3D.NECK].x, y=skeleton[Skeleton3D.NECK].y - NECK_LENGTH, z=median_depth)
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], neck_ep, hip_visible)
            self.get_logger().debug(f"Adding spine from neck and hip: {spine.ns}, {spine.id}, [{neck_ep} || {hip_visible}]")
        # Neck is not visible and one hip is missing
        elif not neck_visible and not hips_visible:
            hip_visible.x = head_mp.x  # Align hip with head
            spine = self.cylinder_from_points(header, id, _markers_dict['body'], head_mp, hip_visible)
            self.get_logger().debug(f"Adding spine from head and hip: {spine.ns}, {spine.id}, [{head_mp} || {hip_visible}]")

        return spine

    def create_shoulders_cyl(self, header, id, skeleton, median_depth, quaternion, trunk_pos):
        """Create the shoulders cylinder from the neck/shoulders position with the body orientation."""
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
        """Create the shoulders cylinder from the hips position with the body orientation."""
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
            # Always add hips but with another color
            pos = Point(x=trunk_pos.x, y=trunk_pos.y + TRUNK_LENGTH/2, z=median_depth)
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
            self.get_logger().debug(f"Adding left up arm from {l_shoulder_ep} to {skeleton[Skeleton3D.LEFT_ELBOW]}")
            if point_not_null(skeleton[Skeleton3D.LEFT_WRIST]):
                l_bot_arm = self.cylinder_from_points(header, id, _markers_dict['left_botarm'], skeleton[Skeleton3D.LEFT_ELBOW], skeleton[Skeleton3D.LEFT_WRIST])
                arms.append(l_bot_arm)
                self.get_logger().debug(f"Adding left bot arm from {skeleton[Skeleton3D.LEFT_ELBOW]} to {skeleton[Skeleton3D.LEFT_WRIST]}")

        # Right arm
        if point_not_null(skeleton[Skeleton3D.RIGHT_ELBOW]):
            displacement_rotated = rotation.apply(-displacement)
            final_position = np.array([shoulders.pose.position.x,
                                       shoulders.pose.position.y,
                                       shoulders.pose.position.z]) + displacement_rotated
            r_shoulder_ep = Point(x=final_position[0], y=final_position[1], z=final_position[2])
            r_top_arm = self.cylinder_from_points(header, id, _markers_dict['right_uparm'], r_shoulder_ep, skeleton[Skeleton3D.RIGHT_ELBOW])
            arms.append(r_top_arm)
            self.get_logger().debug(f"Adding right up arm from {r_shoulder_ep} to {skeleton[Skeleton3D.RIGHT_ELBOW]}")
            if point_not_null(skeleton[Skeleton3D.RIGHT_WRIST]):
                r_bot_arm = self.cylinder_from_points(header, id, _markers_dict['right_botarm'], skeleton[Skeleton3D.RIGHT_ELBOW], skeleton[Skeleton3D.RIGHT_WRIST])
                arms.append(r_bot_arm)
                self.get_logger().debug(f"Adding right bot arm from {skeleton[Skeleton3D.RIGHT_ELBOW]} to {skeleton[Skeleton3D.RIGHT_WRIST]}")

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

    def should_draw_cyl(self, start_point, end_point, landmarks):
        """Returns 'True' if the cylinder between two points should be drawn. 'False' otherwise."""
        return (landmarks[start_point] is not None and landmarks[end_point] is not None
                and point_not_null(landmarks[start_point]) and point_not_null(landmarks[end_point]))

    def parameter_callback(self, params):
        """Callback to update parameters."""
        result = SetParametersResult()
        result.successful = False

        for param in params:
            if param.name == 'processing_rate':
                self.get_logger().error("Parameter 'processing_rate' cannot be changed at runtime.")
            elif param.name == 'allow_half_body' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.allow_half_body = param.value
                result.successful = True
                self.get_logger().warning(f"Allow_half_body set to: {self.allow_half_body}.")
            elif param.name == 'allow_back_turned' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.allow_back_turned = param.value
                result.successful = True
                self.get_logger().warning(f"Allow_back_turned set to: {self.allow_back_turned} but param is not yet implemented.")
            elif param.name == 'display_hinges' and param.type_ == rclpy.Parameter.Type.BOOL:
                self.display_hinges = param.value
                result.successful = True
                self.get_logger().info(f"Display_hinges set to: {self.display_hinges}.")
            else:
                self.get_logger().warning(f"Parameter {param.name} not recognized OR incorrect type.")

        return result


def main(args=None):
    rclpy.init(args=args)
    node = NodeSkeleton3DDisplay()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
