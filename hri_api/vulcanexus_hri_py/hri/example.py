# Copyright 2025 Proyectos y Sistemas de Mantenimiento SL (eProsima).
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

import cv2
from cv_bridge import CvBridge

from hri import HRIListener

import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image as ImageCamera


class NodeShowFaces(Node):
    def __init__(self):
        super().__init__('vulcanexus_hri_py_example')
        self.hri_listener = HRIListener('vulcanexus_hri_py_example_hri_listener')
        self.timer = self.create_timer(0.5, self.timer_callback)

        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub = self.create_subscription(
            ImageCamera, '/test_image', self.image_callback, qos_sensor_data)
        self.image = None

    def image_callback(self, msg: ImageCamera):
        """Save the image."""
        self.image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def timer_callback(self):
        """Timer callback to show the faces."""
        if self.image is None:
            self.get_logger().info('Waiting for image to be published on /test_image topic...')
            return
        for face_id, face in self.hri_listener.faces.items():
            # Cropped image is not published to avoid publishing multiple images
            if (face.cropped is not None):
                cv2.imshow(f'Cropped face {face_id}', face.cropped)

            # Draw Region of Interest on the original image
            roi = face.roi
            if roi is not None:
                x = int(round(roi[0] * self.image.shape[1]))
                y = int(round(roi[1] * self.image.shape[0]))
                w = int(round(roi[2] * self.image.shape[1]))
                h = int(round(roi[3] * self.image.shape[0]))
                cv2.rectangle(self.image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Show Face landmarks
            landmarks = face.facial_landmarks
            if landmarks is not None:
                for key, value in landmarks.items():
                    x = int(round(value[0] * self.image.shape[1]))
                    y = int(round(value[1] * self.image.shape[0]))
                    cv2.circle(self.image, (x, y), 2, (0, 0, 255), -1)

        cv2.imshow('Python HRI Faces', self.image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = NodeShowFaces()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
