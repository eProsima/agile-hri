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

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

from hri_msgs.msg import Expression, Face2D, Face2DList, NormalizedRegionOfInterest2D
from sensor_msgs.msg import Image
from rcl_interfaces.msg import ParameterDescriptor

from ament_index_python.packages import get_package_share_directory
from cv_bridge import CvBridge
import cv2 as cv
import numpy as np
import os

from hri_emotion_detect.facial_fer_model import FacialExpressionRecog
from hri_emotion_detect.PersonExpressionTracker import PersonExpression

cropped_face_width = 128
cropped_face_height = 128


def bound(val, min_val, max_val):
    """Bound a value between a minimum and maximum value."""
    return max(min_val, min(val, max_val))


def normalized_to_pixel_coordinates(
        x_norm: float, y_norm: float, image_width: int, image_height: int) -> (int, int):
    """Convert normalized coordinates to pixel coordinates."""
    x_px = bound(int(x_norm * image_width), 0, image_width - 1)
    y_px = bound(int(y_norm * image_height), 0, image_height - 1)
    return x_px, y_px


class NodeEmotionDetect(Node):
    """
    ROS 2 Node managing the emotion detection from given faces in an image.
    """
    def __init__(self):

        # Initialize node
        super().__init__('hri_emotion_detect')

        self.get_logger().info("Starting emotion detection node")

        # Get package directory
        hri_emotions_dir = get_package_share_directory('hri_emotion_detect')

        # Declare parameters
        self.declare_parameter(
            'model_expression_detection',
            os.path.join(hri_emotions_dir, 'models', 'facial_expression_recognition_mobilefacenet_2022july.onnx'),
            ParameterDescriptor(description='Path to the facial expression recognition model'))

        self.declare_parameter(
            'backend_id', cv.dnn.DNN_BACKEND_OPENCV,
            ParameterDescriptor(description='Backend computation id'))

        self.declare_parameter(
            'target_id', cv.dnn.DNN_TARGET_CPU, ParameterDescriptor(description='Target computation id'))

        # Initialize variables
        self.persons_ = {}
        self.image = None
        self.width = 0
        self.height = 0
        self.detection_proc_duration_ms = 0.
        self.detection_start_proc_time = self.get_clock().now()

        # Create publisher for emotions
        self.emotion_pub_ = self.create_publisher(Expression, '/humans/faces/emotion', 1)

        # Subscribe to original image
        self.img_sub_ = self.create_subscription(Image, '/image', self.image_callback, 1)

        self.face_sub_ = self.create_subscription(Face2DList, '/humans/faces', self.faces_callback, 1)

        # Load facial expression recognition model
        self.fer_model_ = FacialExpressionRecog(modelPath=self.get_parameter('model_expression_detection').value,
                                                backendId=self.get_parameter('backend_id').value,
                                                targetId=self.get_parameter('target_id').value)

    def image_callback(self, msg: Image):
        """
        Callback to save the image and its header.
        It runs a simple logic to check if the processing is too slow.
        """
        # Convert image to OpenCV format
        if self.image is None:
            self.width = msg.width
            self.height = msg.height
        self.image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.image_msg_header = msg.header

    def faces_callback(self, msg: Face2DList):
        """Callback to process the faces detected in the image"""
        self.detection_start_proc_time = self.get_clock().now()
        ids_print = ''
        for roi_msg, face_msg in zip(msg.bboxes, msg.landmarks):
            if roi_msg.key != face_msg.key:
                self.get_logger().error(f"Face id mismatch: [{roi_msg.key}] != [{face_msg.key}]")
                continue
            elif roi_msg.key == "":
                continue
            ids_print += f"[{roi_msg.key}] | "
            self.single_face_processing(roi_msg, face_msg)

        self.detection_proc_duration_ms = (
            self.get_clock().now() - self.detection_start_proc_time).nanoseconds / 1e6
        self.get_logger().debug(f"Pub Emotion: {ids_print}in {self.detection_proc_duration_ms} (ms).")

    def single_face_processing(self, roi: NormalizedRegionOfInterest2D, face: Face2D):
        """Processes the emotion of a single face."""
        if self.image is None:
            self.get_logger().error("No image received yet")
            return
        key = roi.key

        xmin, ymin = normalized_to_pixel_coordinates(roi.xmin,
                                                     roi.ymin,
                                                     self.width,
                                                     self.height)
        xmax, ymax = normalized_to_pixel_coordinates(roi.xmax,
                                                     roi.ymax,
                                                     self.width,
                                                     self.height)
        roi = self.image[ymin:ymax, xmin:xmax]
        header = self.image_msg_header

        #######
        # Uncomment and change roi to output in infer call to crop faces. Need to scale also the landmarks
        # original_bb_width, original_bb_height = (roi.xmax - roi.xmin)*self.width, (roi.ymax - roi.ymin)*self.height
        # sx = cropped_face_width * 1.0 / original_bb_width
        # sy = cropped_face_height * 1.0 / original_bb_height
        # scale = min(sx, sy)
        # scaled = cv.resize(roi, (int(original_bb_width * scale), int(original_bb_height * scale)))
        # scaled_h, scaled_w = scaled.shape[:2]
        # self.get_logger().debug(f"Scaled dims for scale {scale}: {original_bb_width} to {scaled_w}:{original_bb_height} to {scaled_h}")
        # output = np.zeros((cropped_face_width, cropped_face_height, 3), np.uint8)
        # x_offset = int((cropped_face_width - scaled_w) / 2)
        # y_offset = int((cropped_face_height - scaled_h) / 2)
        # output[y_offset:(y_offset+scaled_h), x_offset:(x_offset + scaled_w)] = scaled
        ########

        facial_landmarks = []
        for point in face.landmarks:
            x_norm, y_norm = normalized_to_pixel_coordinates(point.x,
                                                             point.y,
                                                             self.width,
                                                             self.height)
            # x, y = (x_norm - xmin)*scale, (y_norm - ymin)*scale
            x, y = (x_norm - xmin), (y_norm - ymin)
            facial_landmarks.append(x)
            facial_landmarks.append(y)

        facial_landmarks = np.array(facial_landmarks)

        infer_res = self.fer_model_.infer(roi, facial_landmarks)
        self.get_logger().info(f"Face id {key} FER result: {infer_res}")

        # Get emotion type. Infer result is a list of one element with the label index
        infer_type = FacialExpressionRecog.getDesc(infer_res[0])
        self.get_logger().info(f"Face id {key} emotion: {infer_type}")

        self.update_emotion(key, infer_type)

        # Publish emotion for a given face if it is in the list
        if key in self.persons_:
            # Publish emotion
            self.publish_emotion(header, key)

    # Update emotion for a given face
    def update_emotion(self, face_id, emotion):
        """Update the emotion of the given face."""
        if face_id not in self.persons_:
            self.get_logger().debug(f"New face detected with id {face_id}")
            self.persons_[face_id] = PersonExpression()
            self.persons_[face_id].id = len(self.persons_)

        self.persons_[face_id].add_expression(emotion)

    # Publish emotion for a given face
    def publish_emotion(self, header, face_id):
        """Publishes the emotion of the given face."""
        emotion_msg = Expression()
        emotion_msg.header.frame_id = header.frame_id
        emotion_msg.header.stamp = header.stamp

        emotion_msg.key = face_id

        emotion_msg.expression = self.persons_[face_id].get_expression()

        self.emotion_pub_.publish(emotion_msg)


def main(args=None):
    rclpy.init(args=args)
    node = NodeEmotionDetect()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
