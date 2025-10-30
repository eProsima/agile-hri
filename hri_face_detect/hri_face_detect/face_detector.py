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

# Copyright (c) 2023 PAL Robotics S.L. All rights reserved.
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

from rclpy.lifecycle import Node
from rclpy.time import Time

from hri_msgs.msg import FacialLandmarks, Face2D, NormalizedPointOfInterest2D, NormalizedRegionOfInterest2D
from std_msgs.msg import Header

from dataclasses import dataclass, InitVar
from hri_face_detect import YuNetDetector
from mediapipe.python.solutions.face_mesh import FaceMesh
from typing import Dict, List, TypeAlias
import cv2
import math

ColorType: TypeAlias = 'numpy.dtype[numpy.uint8]'
ThreeD: TypeAlias = tuple[int, int, int]
RGBMat: TypeAlias = 'numpy.ndarray[ThreeD, ColorType]'

# Max number of face meshes detected in the same frame
MAX_FACE_MESH_N = 4

# Max distance, relative to RoI diagonal, between the RoI centers of two
# successive detections to consider they belong to the same person
MAX_ROIS_REL_DISTANCE = 0.5

# Max scale factor between two successive regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.5

# Max time allowed between frames to consider they belong to the same person
MAX_TIME_BETWEEN_FRAMES = 1.5

# Default size in pixels for the re-published faces can be changed via the ROS parameters
# /humans/faces/width and /humans/faces/height
cropped_face_width = 128
cropped_face_height = 128

# ROS4HRI to mediapipe landmarks mapping
# ROS4HRI FacialLandmarks ref:
# https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/.github/media/keypoints_face.png  # noqa
# Mediapipe Landmarks refs:
# https://i.stack.imgur.com/5Mohl.jpg
# https://developers.google.com/static/mediapipe/images/solutions/face_landmarker_keypoints.png
ros4hri_to_mp_landmarks_mapping = {
    FacialLandmarks.RIGHT_EAR: 34,
    FacialLandmarks.RIGHT_PROFILE_1: 227,
    FacialLandmarks.RIGHT_PROFILE_2: 137,
    FacialLandmarks.RIGHT_PROFILE_3: 177,
    FacialLandmarks.RIGHT_PROFILE_4: 215,
    FacialLandmarks.RIGHT_PROFILE_5: 135,
    FacialLandmarks.RIGHT_PROFILE_6: 170,
    FacialLandmarks.RIGHT_PROFILE_7: 171,
    FacialLandmarks.MENTON: 175,
    FacialLandmarks.LEFT_EAR: 264,
    FacialLandmarks.LEFT_PROFILE_1: 447,
    FacialLandmarks.LEFT_PROFILE_2: 366,
    FacialLandmarks.LEFT_PROFILE_3: 401,
    FacialLandmarks.LEFT_PROFILE_4: 435,
    FacialLandmarks.LEFT_PROFILE_5: 364,
    FacialLandmarks.LEFT_PROFILE_6: 395,
    FacialLandmarks.LEFT_PROFILE_7: 396,
    FacialLandmarks.RIGHT_EYEBROW_OUTSIDE: 70,
    FacialLandmarks.RIGHT_EYEBROW_1: 63,
    FacialLandmarks.RIGHT_EYEBROW_2: 105,
    FacialLandmarks.RIGHT_EYEBROW_3: 66,
    FacialLandmarks.RIGHT_EYEBROW_INSIDE: 107,
    FacialLandmarks.LEFT_EYEBROW_OUTSIDE: 300,
    FacialLandmarks.LEFT_EYEBROW_1: 293,
    FacialLandmarks.LEFT_EYEBROW_2: 334,
    FacialLandmarks.LEFT_EYEBROW_3: 296,
    FacialLandmarks.LEFT_EYEBROW_INSIDE: 336,
    FacialLandmarks.RIGHT_EYE_OUTSIDE: 130,
    FacialLandmarks.RIGHT_EYE_TOP_1: 29,
    FacialLandmarks.RIGHT_EYE_TOP_2: 28,
    FacialLandmarks.RIGHT_EYE_INSIDE: 243,
    FacialLandmarks.RIGHT_EYE_BOTTOM_1: 24,
    FacialLandmarks.RIGHT_EYE_BOTTOM_2: 22,
    FacialLandmarks.LEFT_EYE_OUTSIDE: 359,
    FacialLandmarks.LEFT_EYE_TOP_1: 259,
    FacialLandmarks.LEFT_EYE_TOP_2: 258,
    FacialLandmarks.LEFT_EYE_INSIDE: 463,
    FacialLandmarks.LEFT_EYE_BOTTOM_1: 254,
    FacialLandmarks.LEFT_EYE_BOTTOM_2: 252,
    FacialLandmarks.SELLION: 6,
    FacialLandmarks.NOSE_1: 197,
    FacialLandmarks.NOSE_2: 4,
    FacialLandmarks.NOSE: 1,
    FacialLandmarks.NOSTRIL_1: 242,
    FacialLandmarks.NOSTRIL_2: 141,
    FacialLandmarks.NOSTRIL_3: 94,
    FacialLandmarks.NOSTRIL_4: 370,
    FacialLandmarks.NOSTRIL_5: 462,
    FacialLandmarks.MOUTH_OUTER_RIGHT: 61,
    FacialLandmarks.MOUTH_OUTER_TOP_1: 40,
    FacialLandmarks.MOUTH_OUTER_TOP_2: 37,
    FacialLandmarks.MOUTH_OUTER_TOP_3: 0,
    FacialLandmarks.MOUTH_OUTER_TOP_4: 267,
    FacialLandmarks.MOUTH_OUTER_TOP_5: 270,
    FacialLandmarks.MOUTH_OUTER_LEFT: 291,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_1: 321,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_2: 314,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_3: 17,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_4: 84,
    FacialLandmarks.MOUTH_OUTER_BOTTOM_5: 91,
    FacialLandmarks.MOUTH_INNER_RIGHT: 62,
    FacialLandmarks.MOUTH_INNER_TOP_1: 41,
    FacialLandmarks.MOUTH_INNER_TOP_2: 12,
    FacialLandmarks.MOUTH_INNER_TOP_3: 271,
    FacialLandmarks.MOUTH_INNER_LEFT: 292,
    FacialLandmarks.MOUTH_INNER_BOTTOM_1: 403,
    FacialLandmarks.MOUTH_INNER_BOTTOM_2: 15,
    FacialLandmarks.MOUTH_INNER_BOTTOM_3: 179,
    FacialLandmarks.RIGHT_PUPIL: 468,
    FacialLandmarks.LEFT_PUPIL: 473
}

# Total number of mediapipe landmarks
ROS4HRI_LANDMARKS_N = 70

# Useful landmarks for the emotion detection
useful_points = [
    FacialLandmarks.RIGHT_PUPIL,
    FacialLandmarks.LEFT_PUPIL,
    FacialLandmarks.NOSE,
    FacialLandmarks.MOUTH_OUTER_RIGHT,
    FacialLandmarks.MOUTH_OUTER_LEFT,
]

# Total number of useful landmarks for the emotion detection
ROS4HRI_EMOTION_LANDMARKS_N = 5


def bound(val, min_val, max_val):
    """Bound a value between a minimum and maximum value."""
    return max(min_val, min(val, max_val))


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates to pixel coordinates."""
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


def pixel_to_normalized_coordinates(
        x_px: int, y_px: int, image_width: int, image_height: int) -> (float, float):
    """Convert pixel coordinates to normalized coordinates."""
    x_norm = bound(x_px / image_width, 0., 1.)
    y_norm = bound(y_px / image_height, 0., 1.)
    return x_norm, y_norm


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
        self.x = bound(self.x, 0, image_width - 1)
        self.y = bound(self.y, 0, image_height - 1)


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
        self.xmin = bound(self.xmin, 0, image_width - 1)
        self.ymin = bound(self.ymin, 0, image_height - 1)
        self.width = bound(xmax - self.xmin, 0, image_width - 1 - self.xmin)
        self.height = bound(ymax - self.ymin, 0, image_height - 1 - self.ymin)
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
        xmin, ymin = pixel_to_normalized_coordinates(self.xmin, self.ymin, self.image_width, self.image_height)
        xmax, ymax = pixel_to_normalized_coordinates(self.xmin + self.width, self.ymin + self.height, self.image_width, self.image_height)
        return (xmin, ymin, xmax, ymax)


@dataclass
class FaceDetection:
    """
    Class representing a detected face in an image.
    """
    score: float
    bb: BoundingBox
    landmarks: Dict[int, ImagePoint]  # FacialLandmarks to ImagePoint

    def __post_init__(self):
        self.score = bound(self.score, 0., 1.)


def distance_rois(bb1: BoundingBox, bb2: BoundingBox) -> float:
    """Returns the Euclidean distance between the center points of two @BoundingBox."""
    x1, y1 = bb1.xmin + bb1.width / 2, bb1.ymin + bb1.height / 2
    x2, y2 = bb2.xmin + bb2.width / 2, bb2.ymin + bb2.height / 2
    return math.dist((x1, y1), (x2, y2))


def bbs_match(bb1: BoundingBox, bb2: BoundingBox) -> bool:
    """Returns whether two @BoundingBox of two consecutive frames should be considered of the same person."""
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


class Face:
    """
    Class representing a face detected by the FacePoseDetector.
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
        self.landmarks: Dict[FacialLandmarks, ImagePoint] = dict()

        self.node = node

        self.do_publish = False
        self.node.get_logger().debug(f'New face: {self.id}')

    def create_msgs(self, src_image: RGBMat, image_msg_header: Header):
        """Return the ROI and Face2D messages for this face."""
        return self.generate_roi_msg(src_image, image_msg_header), self.generate_face2D_msg(src_image, image_msg_header)

    def generate_roi_msg(self, src_image: RGBMat, image_msg_header: Header):
        """Generate a NormalizedRegionOfInterest2D message for this face to be used as ROI."""
        img_height, img_width, _ = src_image.shape
        msg = NormalizedRegionOfInterest2D()
        msg.key = self.id
        msg.header = image_msg_header
        msg.xmin, msg.ymin = pixel_to_normalized_coordinates(
            self.bb.xmin, self.bb.ymin, img_width, img_height)
        msg.xmax, msg.ymax = pixel_to_normalized_coordinates(
            self.bb.xmin + self.bb.width, self.bb.ymin + self.bb.height, img_width, img_height)
        msg.c = self.score
        return msg

    def generate_face2D_msg(self, src_image: RGBMat, image_msg_header: Header):
        """Generate a Face2D message for this face to be used as landmarks."""
        img_height, img_width, _ = src_image.shape
        msg = Face2D()
        msg.key = self.id
        msg.header = image_msg_header
        msg.confidence = self.score
        msg.landmarks = [NormalizedPointOfInterest2D() for _ in range(ROS4HRI_EMOTION_LANDMARKS_N)]
        for idx, point in enumerate(useful_points):
            x, y = pixel_to_normalized_coordinates(self.landmarks[point].x, self.landmarks[point].y, img_width, img_height)
            msg.landmarks[idx].x = x
            msg.landmarks[idx].y = y

        return msg

    def create_mesh_msg(self, src_image: RGBMat, image_msg_header: Header) -> FacialLandmarks:
        """Generate a FacialLandmarks message for this face to be used as landmarks."""
        img_height, img_width, _ = src_image.shape
        msg = FacialLandmarks()
        msg.key = self.id
        msg.header = image_msg_header
        msg.landmarks = [NormalizedPointOfInterest2D() for _ in range(ROS4HRI_LANDMARKS_N)]
        for point in range(ROS4HRI_LANDMARKS_N):
            x, y = pixel_to_normalized_coordinates(self.landmarks[point].x, self.landmarks[point].y, img_width, img_height)
            self.node.get_logger().info(f'Face [{self.id}] landmark {point} at ({x}, {y})')
            msg.landmarks[point].x = x
            msg.landmarks[point].y = y
            msg.landmarks[point].c = self.score

        return msg

    def change_id(self, new_id: str):
        """Change the temporary id of the face."""
        self.node.get_logger().debug(f'Face [{self.id}] changed id to {new_id}')
        self.id = new_id
        self.temp_id = False

    def set_publish(self, should_publish: bool):
        """Set the flag to publish the face only if it is not a temp face."""
        if not self.temp_id:
            self.do_publish = should_publish

    def ref_face_point(self):
        """Return the reference face point for this face: Center of the bounding box."""
        xmin, ymin, xmax, ymax = self.bb.get_norm_coords()
        ref_point = [(xmax + xmin) / 2, (ymax + ymin) / 2]
        self.node.get_logger().debug(f'Face [{self.id}] reference point: {ref_point} with bbox: {self.bb} because norm coords are: {xmin, ymin, xmax, ymax}')
        return ref_point

    def __del__(self):
        detect_time = (self.node.get_clock().now() - self.initial_detection_time).nanoseconds / 1e9
        self.node.get_logger().debug(
            f'Face [{self}] lost. It remained visible for {detect_time:.2f}sec')

    def __repr__(self):
        return self.id


class FaceDetector:
    """
    Class containing the CNN and its methods to detect faces. It is called by the NodeFaceDetector
    """
    def __init__(self, confidence_threshold: float, image_scale: float):
        self.confidence_threshold = confidence_threshold
        self.image_scale = image_scale
        self.detector = YuNetDetector()

    @staticmethod
    def _extract_face_detection(
            raw_detection: List, scale: float, image_width: int, image_height: int
            ) -> FaceDetection:
        """Extract a FaceDetection from the raw output of the model."""
        score = float(raw_detection[0]) / 100.
        scaled_raw_coords = [int(x*scale) for x in raw_detection[1:15]]
        bb = BoundingBox(*scaled_raw_coords[0:4], image_width, image_height)

        landmarks: Dict[FacialLandmarks, ImagePoint] = dict()
        landmarks[FacialLandmarks.RIGHT_PUPIL] = ImagePoint(
            *scaled_raw_coords[4:6], image_width, image_height)
        landmarks[FacialLandmarks.LEFT_PUPIL] = ImagePoint(
            *scaled_raw_coords[6:8], image_width, image_height)
        landmarks[FacialLandmarks.NOSE] = ImagePoint(
            *scaled_raw_coords[8:10], image_width, image_height)
        landmarks[FacialLandmarks.MOUTH_OUTER_RIGHT] = ImagePoint(
            *scaled_raw_coords[10:12], image_width, image_height)
        landmarks[FacialLandmarks.MOUTH_OUTER_LEFT] = ImagePoint(
            *scaled_raw_coords[12:14], image_width, image_height)

        return FaceDetection(score, bb, landmarks)

    def detect(self, img: RGBMat) -> List[FaceDetection]:
        """Run the inference on the input image and return the detected faces."""
        img_height, img_width, _ = img.shape

        scaled_img = cv2.resize(
            img, None, fx=self.image_scale, fy=self.image_scale, interpolation=cv2.INTER_AREA)
        scaled_img_height, scaled_img_width, _ = scaled_img.shape

        raw_face_detections = self.detector.detect(
            scaled_img, scaled_img_width, scaled_img_height, scaled_img.strides[0])
        face_detections = [
            self._extract_face_detection(d, 1./self.image_scale, img_width, img_height)
            for d in raw_face_detections]
        valid_face_detections = [
            d for d in face_detections
            if d.score > self.confidence_threshold and d.bb.width > 0 and d.bb.height > 0]

        return valid_face_detections


class MeshDetector:
    """
    Class containing the MediaPipe model and its methods to detect faces.
    """
    def __init__(self):
        self.detector = FaceMesh(
            static_image_mode=False, max_num_faces=MAX_FACE_MESH_N, refine_landmarks=True)

    @staticmethod
    def _extract_mesh_detection(
            raw_landmarks: List, image_width: int, image_height: int) -> FaceDetection:
        """Extract a FaceDetection from the raw output of the model."""
        xmin = image_width - 1
        ymin = image_height - 1
        xmax = 0
        ymax = 0

        landmarks: Dict[FacialLandmarks, ImagePoint] = dict()
        for ros4hri_idx, mp_idx in ros4hri_to_mp_landmarks_mapping.items():
            landmark_norm = raw_landmarks.landmark[mp_idx]
            x, y = normalized_to_pixel_coordinates(
                landmark_norm.x, landmark_norm.y, image_width, image_height
            )
            landmarks[ros4hri_idx] = ImagePoint(x, y, image_width, image_height)
            xmin = min(x, xmin)
            ymin = min(y, ymin)
            xmax = max(x, xmax)
            ymax = max(y, ymax)

        bb = BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin, image_width, image_height)

        return FaceDetection(1.0, bb, landmarks)

    def detect(self, img: RGBMat) -> List[FaceDetection]:
        """Run the inference on the input image and return the detected faces."""
        img_height, img_width, _ = img.shape

        mesh_detections: List[FaceDetection] = list()
        mesh_results = self.detector.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if mesh_results.multi_face_landmarks:
            mesh_detections = [
                self._extract_mesh_detection(raw_landmarks, img_width, img_height)
                for raw_landmarks in mesh_results.multi_face_landmarks]

        return mesh_detections
