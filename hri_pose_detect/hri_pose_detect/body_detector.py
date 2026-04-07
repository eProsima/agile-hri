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

# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
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

from rclpy.node import Node
from rclpy.time import Time

from hri_msgs.msg import Skeleton2D, Skeleton3D, NormalizedRegionOfInterest2D
from std_msgs.msg import Header

from dataclasses import dataclass, InitVar
from image_geometry import PinholeCameraModel
from typing import Dict, List, TypeAlias
from ultralytics import YOLO
import cv2
import math
import numpy as np
import time

ColorType: TypeAlias = 'numpy.dtype[numpy.uint8]'
ThreeD: TypeAlias = tuple[int, int, int]
RGBMat: TypeAlias = 'numpy.ndarray[ThreeD, ColorType]'

# YOLO skeleton indexing
YOLO_NOSE = 0
# YOLO_NECK is not present in YOLO. Compute it as the average of left and right shoulders
YOLO_LEFT_EYE = 1
YOLO_RIGHT_EYE = 2
YOLO_LEFT_EAR = 3
YOLO_RIGHT_EAR = 4
YOLO_LEFT_SHOULDER = 5
YOLO_RIGHT_SHOULDER = 6
YOLO_LEFT_ELBOW = 7
YOLO_RIGHT_ELBOW = 8
YOLO_LEFT_WRIST = 9
YOLO_RIGHT_WRIST = 10
YOLO_LEFT_HIP = 11
YOLO_RIGHT_HIP = 12
YOLO_LEFT_KNEE = 13
YOLO_RIGHT_KNEE = 14
YOLO_LEFT_ANKLE = 15
YOLO_RIGHT_ANKLE = 16

ROS4HRI_YOLO_LANDMARKS_N = 17

# ROS4HRI to YOLO skeleton indexing conversion table
ros4hri_to_yolo = [None] * (ROS4HRI_YOLO_LANDMARKS_N + 1)

ros4hri_to_yolo[Skeleton2D.NOSE] = YOLO_NOSE
ros4hri_to_yolo[Skeleton2D.LEFT_EYE] = YOLO_LEFT_EYE
ros4hri_to_yolo[Skeleton2D.LEFT_EAR] = YOLO_LEFT_EAR
ros4hri_to_yolo[Skeleton2D.LEFT_SHOULDER] = YOLO_LEFT_SHOULDER
ros4hri_to_yolo[Skeleton2D.LEFT_ELBOW] = YOLO_LEFT_ELBOW
ros4hri_to_yolo[Skeleton2D.LEFT_WRIST] = YOLO_LEFT_WRIST
ros4hri_to_yolo[Skeleton2D.LEFT_HIP] = YOLO_LEFT_HIP
ros4hri_to_yolo[Skeleton2D.LEFT_KNEE] = YOLO_LEFT_KNEE
ros4hri_to_yolo[Skeleton2D.LEFT_ANKLE] = YOLO_LEFT_ANKLE
ros4hri_to_yolo[Skeleton2D.RIGHT_EYE] = YOLO_RIGHT_EYE
ros4hri_to_yolo[Skeleton2D.RIGHT_EAR] = YOLO_RIGHT_EAR
ros4hri_to_yolo[Skeleton2D.RIGHT_SHOULDER] = YOLO_RIGHT_SHOULDER
ros4hri_to_yolo[Skeleton2D.RIGHT_ELBOW] = YOLO_RIGHT_ELBOW
ros4hri_to_yolo[Skeleton2D.RIGHT_WRIST] = YOLO_RIGHT_WRIST
ros4hri_to_yolo[Skeleton2D.RIGHT_HIP] = YOLO_RIGHT_HIP
ros4hri_to_yolo[Skeleton2D.RIGHT_KNEE] = YOLO_RIGHT_KNEE
ros4hri_to_yolo[Skeleton2D.RIGHT_ANKLE] = YOLO_RIGHT_ANKLE

# YOLO to ROS4HRI skeleton indexing conversion table
yolo_to_ros4hri = {value: key for key, value in enumerate(ros4hri_to_yolo) if value is not None}

_face_landmarks = [
        Skeleton2D.NOSE,
        Skeleton2D.LEFT_EYE,
        Skeleton2D.RIGHT_EYE,
        Skeleton2D.LEFT_EAR,
        Skeleton2D.RIGHT_EAR,
]

# Max distance, relative to RoI diagonal, between the RoI centers of two
# successive detections to consider they belong to the same person
MAX_ROIS_REL_DISTANCE = 0.3

# Max scale factor between two successive ROIs to consider they belong to the same person
MAX_SCALING_ROIS = 1.5

# Max time allowed between frames to consider they belong to the same person
MAX_TIME_BETWEEN_FRAMES = 1.5


def _bound(val, min_val, max_val):
    """Bound a value between min_val and max_val."""
    return (max(min_val, min(val, max_val)))


def _pixel_to_normalized_coordinates(
        x_px: int, y_px: int, image_width: int, image_height: int) -> (float, float):
    """Convert pixel coordinates to normalized coordinates."""
    x_norm = _bound(x_px / image_width, 0., 1.)
    y_norm = _bound(y_px / image_height, 0., 1.)
    return x_norm, y_norm


def _normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates to pixel coordinates."""
    x_px = _bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = _bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


@dataclass
class ImagePoint:
    """
    Class representing a point in an image with coordinates in pixels.
    """
    x: int
    y: int
    image_width: InitVar(int)
    image_height: InitVar(int)

    def __post_init__(self, image_width, image_height):
        self.x = _bound(self.x, 0, image_width - 1)
        self.y = _bound(self.y, 0, image_height - 1)


@dataclass
class ImagePoint_norm:
    """
    Class representing a point in an image with normalized coordinates in the range [0, 1].
    """
    x: float
    y: float

    def __post_init__(self):
        self.x = _bound(self.x, 0., 1.)
        self.y = _bound(self.y, 0., 1.)


@dataclass
class BoundingBox:
    """
    Class representing a boundary box in an image.
    """
    xmin: int
    ymin: int
    width: int
    height: int
    image_width: InitVar(int)
    image_height: InitVar(int)

    def __post_init__(self, image_width, image_height):
        xmax = self.xmin + self.width
        ymax = self.ymin + self.height
        self.xmin = _bound(self.xmin, 0, image_width - 1)
        self.ymin = _bound(self.ymin, 0, image_height - 1)
        self.width = _bound(xmax - self.xmin, 0, image_width - 1 - self.xmin)
        self.height = _bound(ymax - self.ymin, 0, image_height - 1 - self.ymin)
        self.image_height = image_height
        self.image_width = image_width

    def __str__(self):
        return (f'BoundingBox(xmin={self.xmin}, ymin={self.ymin}, width={self.width}, height={self.height}, '
                f'image_width={self.image_width}, image_height={self.image_height})')

    def diag_length(self):
        """Return the diagonal length of the bounding box."""
        return math.dist((0, 0), (self.width, self.height))

    def get_coords(self):
        """Return the bounding box coordinates: xmin, ymin, xmax, ymax."""
        return (self.xmin, self.ymin, self.xmin + self.width, self.ymin + self.height)

    def get_norm_coords(self):
        """Return the normalized bounding box coordinates: xmin, ymin, xmax, ymax."""
        xmin, ymin = _pixel_to_normalized_coordinates(self.xmin, self.ymin, self.image_width, self.image_height)
        xmax, ymax = _pixel_to_normalized_coordinates(self.xmin + self.width, self.ymin + self.height, self.image_width, self.image_height)
        return (xmin, ymin, xmax, ymax)


@dataclass
class BodyDetection:
    """
    Class representing a detected body in an image.
    """
    score: float
    bb: BoundingBox
    landmarks: Dict[int, ImagePoint_norm]  # BodyLandmarks to ImagePoint

    def __post_init__(self):
        self.score = _bound(self.score, 0., 1.)


def distance_rois(bb1: BoundingBox, bb2: BoundingBox) -> float:
    """Compute the distance between the centers of two bounding boxes."""
    x1, y1 = bb1.xmin + bb1.width / 2, bb1.ymin + bb1.height / 2
    x2, y2 = bb2.xmin + bb2.width / 2, bb2.ymin + bb2.height / 2
    return math.dist((x1, y1), (x2, y2))


def bbs_match(bb1: BoundingBox, bb2: BoundingBox) -> bool:
    """Check if two bounding boxes match, so they can be considered the same person."""
    return (
        (distance_rois(bb1, bb2) / bb1.diag_length()) < MAX_ROIS_REL_DISTANCE
        and (1 / MAX_SCALING_ROIS) < (bb1.width / bb2.width) < MAX_SCALING_ROIS
        and (1 / MAX_SCALING_ROIS) < (bb1.height / bb2.height) < MAX_SCALING_ROIS)


def time_match(t1, t2, proc_time, offset) -> bool:
    """
    Check if the register time of two bounding boxes is low enough to consider them of the same person.
    @proc_time is the image processing time in milliseconds.
    """
    return (t1 - t2).nanoseconds - offset.nanoseconds < (MAX_TIME_BETWEEN_FRAMES * 1e9 + proc_time * 1e6)


class Body:
    """
    Class representing a body detected by the NodePoseDetector.
    """

    def __init__(self,
                 node: Node,
                 id: str):

        self.id = id
        self.temp_id = True

        self.initial_detection_time: Time = None
        self.last_detection_time: Time = None
        self.nb_frames_visible = 0
        self.nb_frames_since_last_detection = 0

        self.score = 0.
        self.bb: BoundingBox = None
        self.landmarks: Dict[Skeleton2D, ImagePoint_norm] = dict()

        self.node = node

        self.do_publish = False

        self.node.get_logger().debug(f'New body: {self.id}')

    def create_msgs(self, src_image: RGBMat, image_msg_header: Header):
        """Return the ROI and skeleton messages for this body."""
        return self.generate_roi_msg(src_image, image_msg_header), self.generate_skeleton_msg(image_msg_header)

    def generate_roi_msg(self, src_image: RGBMat, image_msg_header: Header):
        """Generate a NormalizedRegionOfInterest2D message for this body to be used as ROI."""
        img_height, img_width, _ = src_image.shape
        msg = NormalizedRegionOfInterest2D()
        msg.key = self.id
        msg.header = image_msg_header
        msg.xmin, msg.ymin = _pixel_to_normalized_coordinates(
            self.bb.xmin, self.bb.ymin, img_width, img_height)
        msg.xmax, msg.ymax = _pixel_to_normalized_coordinates(
            self.bb.xmin + self.bb.width, self.bb.ymin + self.bb.height, img_width, img_height)
        msg.c = self.score

        return msg

    def _estimate_neck(self):
        """Estimate the neck landmark as the midpoint of both shoulders, if visible."""
        # Skip if already estimated or if any of the shoulders is not visible
        if self.landmarks.get(Skeleton2D.NECK) is not None:
            return
        if self.landmarks[Skeleton2D.LEFT_SHOULDER].x != 0 and self.landmarks[Skeleton2D.RIGHT_SHOULDER].x != 0:
            self.landmarks[Skeleton2D.NECK] = ImagePoint_norm(0., 0.)
            self.landmarks[Skeleton2D.NECK].x = (
                self.landmarks[Skeleton2D.LEFT_SHOULDER].x
                + self.landmarks[Skeleton2D.RIGHT_SHOULDER].x
            )/2
            self.landmarks[Skeleton2D.NECK].y = (
                self.landmarks[Skeleton2D.LEFT_SHOULDER].y
                + self.landmarks[Skeleton2D.RIGHT_SHOULDER].y
            )/2

    def generate_skeleton_msg(self, image_msg_header: Header):
        """Generate a Skeleton2D message for this body."""
        msg = Skeleton2D()
        msg.key = self.id
        msg.header = image_msg_header
        msg.confidence = self.score

        self._estimate_neck()

        for dict_key, landmark in self.landmarks.items():
            x, y = landmark.x, landmark.y
            msg.skeleton[dict_key].x = x
            msg.skeleton[dict_key].y = y

        return msg

    def create_depth_msg(self, depth, depth_msg_header, encoding, rgb_camera_info, depth_camera_info):
        """Generate a Skeleton3D message for this body."""
        if depth is None or rgb_camera_info is None or depth_camera_info is None:
            self.node.get_logger().info("Cannot create Skeleton3D msg because parameters are missing.")
            return None

        # Ensure NECK is estimated before building the 3D skeleton
        self._estimate_neck()

        depth_model = PinholeCameraModel()
        rgb_model = PinholeCameraModel()
        depth_model.fromCameraInfo(depth_camera_info)
        rgb_model.fromCameraInfo(rgb_camera_info)

        msg = Skeleton3D()
        msg.key = self.id
        msg.header = depth_msg_header
        msg.confidence = self.score
        for dict_key, landmark in self.landmarks.items():
            if landmark.x == 0 and landmark.y == 0:
                continue

            x, y = _normalized_to_pixel_coordinates(landmark.x, landmark.y, rgb_camera_info.width, rgb_camera_info.height)
            x_d = int(((x - rgb_model.cx())
                      * depth_model.fx()
                      / rgb_model.fx())
                      + depth_model.cx())
            y_d = int(((y - rgb_model.cy())
                      * depth_model.fy()
                      / rgb_model.fy())
                      + depth_model.cy())

            # Ensure the coordinates are within the depth image
            x_d = _bound(x_d, 0, depth_model.width - 1)
            y_d = _bound(y_d, 0, depth_model.height - 1)

            if encoding == '32FC1':
                # Get depth data encoded as 32bit/m
                z = depth[y_d][x_d]
            elif encoding == '16UC1':
                # Convert depth data encoded as 16bit/mm to m
                z = depth[y_d][x_d]/1000

            if z != 0 and not np.isnan(z):
                uv_rectified = rgb_model.rectifyPoint((x, y))
                position = rgb_model.projectPixelTo3dRay(uv_rectified)
                position = np.array(position) * z

                msg.skeleton[dict_key].x = position[0]
                msg.skeleton[dict_key].y = position[1]
                msg.skeleton[dict_key].z = position[2]

        return msg

    def extract_body_depth_of_interest(self, depth, skeleton_msg, encoding, rgb_camera_info, depth_camera_info):
        """Extract the depth point of interest from the depth image."""
        if depth is None:
            self.node.get_logger().info("Returning None because depth is None.")
            return None

        x, y = 0, 0
        # This could also be calculated as the average/median between all torso and head points.
        if skeleton_msg.skeleton[Skeleton2D.NECK].x != 0 and skeleton_msg.skeleton[Skeleton2D.NECK].y != 0:
            x, y = skeleton_msg.skeleton[Skeleton2D.NECK].x, skeleton_msg.skeleton[Skeleton2D.NECK].y
        elif skeleton_msg.skeleton[Skeleton2D.NOSE].x != 0 and skeleton_msg.skeleton[Skeleton2D.NOSE].y != 0:
            x, y = skeleton_msg.skeleton[Skeleton2D.NOSE].x, skeleton_msg.skeleton[Skeleton2D.NOSE].y
        elif skeleton_msg.skeleton[Skeleton2D.LEFT_SHOULDER].x != 0 and skeleton_msg.skeleton[Skeleton2D.LEFT_SHOULDER].y != 0:
            x, y = skeleton_msg.skeleton[Skeleton2D.LEFT_SHOULDER].x, skeleton_msg.skeleton[Skeleton2D.LEFT_SHOULDER].y
        elif skeleton_msg.skeleton[Skeleton2D.RIGHT_SHOULDER].x != 0 and skeleton_msg.skeleton[Skeleton2D.RIGHT_SHOULDER].y != 0:
            x, y = skeleton_msg.skeleton[Skeleton2D.RIGHT_SHOULDER].x, skeleton_msg.skeleton[Skeleton2D.RIGHT_SHOULDER].y
        else:
            self.node.get_logger().debug("No valid point to extract depth of interest from, returning zero.")
            return 0

        depth_model = PinholeCameraModel()
        rgb_model = PinholeCameraModel()
        depth_model.fromCameraInfo(depth_camera_info)
        rgb_model.fromCameraInfo(rgb_camera_info)

        x, y = _normalized_to_pixel_coordinates(x, y, depth_camera_info.width, depth_camera_info.height)

        x_d = int(((x - rgb_model.cx())
                  * depth_model.fx()
                  / rgb_model.fx())
                  + depth_model.cx())
        y_d = int(((y - rgb_model.cy())
                  * depth_model.fy()
                  / rgb_model.fy())
                  + depth_model.cy())

        # Ensure the coordinates are within the depth image
        x_d = _bound(x_d, 0, depth_model.width - 1)
        y_d = _bound(y_d, 0, depth_model.height - 1)

        if encoding == '32FC1':
            # Get depth data encoded as 32bit/m
            z = depth[y_d][x_d]
        elif encoding == '16UC1':
            # Convert depth data encoded as 16bit/mm to m
            z = depth[y_d][x_d]/1000

        return z

    def change_id(self, new_id: str):
        """Change the temporary id of the body."""
        self.node.get_logger().debug(f'Body [{self.id}] changed id to {new_id}')
        self.id = new_id
        self.temp_id = False

    def set_publish(self, bool: bool):
        """Set the flag to publish the body only if it is not a temp body."""
        if not self.temp_id:
            self.do_publish = bool

    def ref_face_point(self):
        """Return the reference face point for this body."""
        tot_x, tot_y, tot_points = 0, 0, 0
        for point in _face_landmarks:
            if self.landmarks[point] is not None and self.landmarks[point].x != 0 and self.landmarks[point].y != 0:
                tot_x += self.landmarks[point].x
                tot_y += self.landmarks[point].y
                tot_points += 1

        ref_point = [0., 0.]
        if tot_points != 0:
            ref_point = [(tot_x / tot_points), (tot_y / tot_points)]

        self.node.get_logger().debug(f'Body [{self.id}] reference point: {ref_point}')
        return ref_point

    def __del__(self):
        detect_time = (self.node.get_clock().now() - self.initial_detection_time).nanoseconds / 1e9
        self.node.get_logger().debug(
            f'Body [{self.id}] lost. It remained visible for {detect_time:.2f}sec')


class BodyDetector:
    """
    Class for detecting bodies in an image using YOLO. It is called by the NodePoseDetect.
    """
    def __init__(self, model_path: str, confidence_threshold: float, image_scale: float, node: Node):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.image_scale = image_scale
        self.node = node
        self.detector = None

    def load_model(self):
        """Load the YOLO model. Returns True if the model is loaded successfully."""
        if self.detector is None:
            try:
                self.node.get_logger().info(f'Loading YOLO model from {self.model_path}')
                self.detector = YOLO(self.model_path)
            except Exception as e:
                self.node.get_logger().error(f'Error loading YOLO model: {str(e)}')
                self.detector = None
                return False
        else:
            self.node.get_logger().warning("YOLO model already loaded.")
        return True

    def unload_model(self):
        """Unload the YOLO model."""
        if self.detector is not None:
            del self.detector
        self.detector = None

    @staticmethod
    def _extract_body_detections(
            boxes, kps, image_width: int, image_height: int, th: float
            ) -> List[BodyDetection]:
        """Extract a BodyDetection from the raw output of the model."""
        body_detections = []
        for box_coords, box_conf, keypoints in zip(boxes.xyxy, boxes.conf, kps.xyn):
            if box_conf < th:
                continue
            # Box coordinates are NOT normalized
            bbx1, bby1, bbx2, bby2 = [int(coord) for coord in box_coords]
            bb = BoundingBox(bbx1, bby1, bbx2 - bbx1, bby2 - bby1, image_width, image_height)

            # Landmarks are normalized
            landmarks: Dict[Skeleton2D, ImagePoint_norm] = dict()
            for idx, keypoint in enumerate(keypoints):
                kpx, kpy = [float(coord) for coord in keypoint]
                landmarks[yolo_to_ros4hri[idx]] = ImagePoint_norm(kpx, kpy)

            body_detections.append(BodyDetection(float(box_conf), bb, landmarks))
        return body_detections

    def _detect(self, img: RGBMat) -> List[BodyDetection]:
        """
        Main detection method.
        It returns a list of BodyDetection objects with the
        boundary boxes and lanmarks of each body in the image.
        """
        img_height, img_width, _ = img.shape
        scaled_img = cv2.resize(
            img, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA)

        raw_body_results = self.detector(scaled_img)

        body_detections = self._extract_body_detections(
                raw_body_results[0].boxes,
                raw_body_results[0].keypoints,
                img_width,
                img_height,
                self.confidence_threshold,
            )

        valid_body_detections = [
            d for d in body_detections if d.bb.width > 0 and d.bb.height > 0]

        return valid_body_detections

    def run(self, queue_in, queue_out):
        """
        Main loop for the subprocess running YOLO.
        It calls the @detect method whenever is possible.
        """
        self.node.get_logger().info("Starting YOLO running process.")
        if not self.load_model():
            return
        try:
            while True:
                if self.detector is None:
                    self.node.get_logger().warning("In RUN YOLO model not loaded.")
                    return

                if queue_in.empty():
                    # Aprox 30 Hz
                    time.sleep(0.033)
                    continue
                img = queue_in.get()
                if isinstance(img, str) and img == 'stop':
                    break

                result = self._detect(img)

                queue_out.put(result)
        except KeyboardInterrupt:
            self.node.get_logger().info("YOLO running process interrupted by user.")
        except Exception as e:
            self.node.get_logger().error(f'Error in YOLO running process: {str(e)}')

        self.unload_model()
