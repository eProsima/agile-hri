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

# test_hri_pose_detect_launch.py

# Integration tests for the hri_pose_detect node. Simulating the launch_testing framework

import signal
import subprocess
import time
import os
import unittest
import threading
import rclpy

import cv2
from cv_bridge import CvBridge
from hri_msgs.msg import Skeleton2DList, Skeleton2D
from sensor_msgs.msg import CameraInfo, Image

minimal_landmarks = [
    Skeleton2D.NOSE,
]

upper_landmarks = [
    Skeleton2D.NOSE,
    Skeleton2D.NECK,
    Skeleton2D.RIGHT_SHOULDER,
    Skeleton2D.LEFT_SHOULDER,
    Skeleton2D.LEFT_EYE,
    Skeleton2D.RIGHT_EYE,
    Skeleton2D.LEFT_EAR,
    Skeleton2D.RIGHT_EAR,
]

def make_cam_info(width, height) -> CameraInfo:
    msg = CameraInfo()
    msg.width = width
    msg.height = height
    msg.k = [
        580.0, 0.0, 320.0,  # fx,  0, cx
        0.0, 580.0, 240.0,  # 0,  fy, cy
        0.0, 0.0, 1.0,      # 0,  0,  1
    ]
    return msg


class TestHRIPoseDetectIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ROS_DOMAIN_ID"] = "116"
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        rclpy.init()
        cls.bridge = CvBridge()

        cls.hz = 30  # Hz for publishing images

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):

        self.proc_id_manager = subprocess.Popen(
            [
                "ros2", "launch", "hri_id_manager",
                "id_manager.launch.py",
                "log-level:=debug"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),  # Inherit + our overrides
            preexec_fn=os.setsid
        )

        self.proc = subprocess.Popen(
            [
                "ros2", "launch", "hri_pose_detect",
                "pose_detect.launch.py",
                "rgb_camera_topic:=test_image",
                "rgb_camera_info:=test_cam_info",
                "log-level:=debug"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),  # Inherit + our overrides
            preexec_fn=os.setsid
        )

        time.sleep(3)  # Wait for the nodes to start

        print("Setting up test client node")
        self.node = rclpy.create_node('test_client')

        # Publisher for camera topics expected by NodePoseDetect
        self.pub_img = self.node.create_publisher(Image, "test_image", 10)
        self.pub_info = self.node.create_publisher(CameraInfo, "test_cam_info", 10)

        self.msg_received = threading.Event()
        self.msg_received.clear()
        self.num_msgs = 0
        self.skeletons_msg = None
        def poses_cb(msg: Skeleton2DList):
            print(f"Received Skeleton2DList with stamp {(msg.header.stamp)}.")
            self.skeletons_msg = msg
            self.num_msgs += 1
            if self.num_msgs > 3:
                self.msg_received.set()
        self.sub_bodies = self.node.create_subscription(Skeleton2DList, "/humans/bodies", poses_cb, 10)

        time.sleep(2)  # Wait for the subscriptions to be established

    def tearDown(self):
        print("Tearing down test client node")
        self.node.destroy_subscription(self.sub_bodies)
        self.node.destroy_publisher(self.pub_img)
        self.node.destroy_publisher(self.pub_info)
        self.node.destroy_node()

        for proc in (self.proc, self.proc_id_manager):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            try:
                ret = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                ret = proc.wait()
            finally:
                print(f"Process {proc.pid} terminated with return code {ret}")

        print("OUTPUT:")
        stdout, stderr = self.proc.communicate()
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))

        time.sleep(2)  # Wait for processes to clean up (YOLO thread might take a while)

    def pub_loop(self, img_msg: Image):
        print("Starting image publishing loop")
        # Publish the image once to proc the YOLO loading in the node
        self.pub_img.publish(img_msg)
        time.sleep(3)
        pub_times = 0
        # Publish the image for 10 seconds at 30 Hz
        while rclpy.ok() and pub_times < (10 * self.hz):
            self.node.get_logger().info(f"Publishing image {pub_times + 1} at {self.hz} Hz")
            img_msg.header.stamp = self.node.get_clock().now().to_msg()
            self.pub_img.publish(img_msg)
            pub_times += 1
            time.sleep(1 / self.hz)  # Sleep for 1/hz seconds

    def test_single_pose_detection(self):
        """
        Test that after publishing an image, a body is published in the corresponding topic.
        """
        # Path to test image
        img_path = os.path.join(os.path.dirname(__file__), "images/test_image_person.png")
        self.assertTrue(os.path.exists(img_path), f"Test image not found at {img_path}")
        # Load image from file
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.node.get_clock().now().to_msg()

        # Create and publish matching CameraInfo
        info_msg = make_cam_info(640, 480)
        info_msg.header = img_msg.header
        # Publish cam info once
        self.pub_info.publish(info_msg)

        # Start publishing the image for 5 seconds at 30 Hz in a separate thread
        pub_thread = threading.Thread(target=self.pub_loop, args=(img_msg,))
        pub_thread.start()

        # Spin until messages arrives or timeout
        timeout = time.time() + 15.0
        while rclpy.ok() and time.time() < timeout and not self.msg_received.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Wait for the publishing thread to finish
        if pub_thread.is_alive():
            pub_thread.join()

        self.assertTrue(self.msg_received.is_set(), "No Skeleton2DList received after publishing an image.")
        self.assertGreaterEqual(len(self.skeletons_msg.bboxes), 1,
            "Skeleton2DList received but contains zero bounding boxes.")  # Always contains 10
        # Check that the first bounding box has valid coordinates
        self.assertGreater(self.skeletons_msg.bboxes[0].xmin, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].ymin, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].xmax, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].ymax, 0)
        # Check that the first Skeleton2D has landmarks and valid coordinates
        self.assertGreaterEqual(len(self.skeletons_msg.skeletons), 1,
            "Skeleton2DList received but contains zero bounding boxes.")   # Always contains 10
        # There might be partial bodies, so we cannot expect all landmarks to be greater than 0
        # Testing image only has upper landmarks
        for landmark_idx in upper_landmarks:
            self.assertGreater(self.skeletons_msg.skeletons[0].skeleton[landmark_idx].x, 0)
            self.assertGreater(self.skeletons_msg.skeletons[0].skeleton[landmark_idx].y, 0)

    def test_single_pose_detection_fullbody(self):
        """
        Test that after publishing an image, a full body is published in the corresponding topic.
        """
        # Path to test image
        img_path = os.path.join(os.path.dirname(__file__), "images/test_image_fullbody.png")
        self.assertTrue(os.path.exists(img_path), f"Test image not found at {img_path}")
        # Load image from file
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.node.get_clock().now().to_msg()

        # Create and publish matching CameraInfo
        info_msg = make_cam_info(640, 480)
        info_msg.header = img_msg.header
        # Publish cam info once
        self.pub_info.publish(info_msg)

        # Start publishing the image for 5 seconds at 30 Hz in a separate thread
        pub_thread = threading.Thread(target=self.pub_loop, args=(img_msg,))
        pub_thread.start()

        # Spin until messages arrives or timeout
        timeout = time.time() + 15.0
        while rclpy.ok() and time.time() < timeout and not self.msg_received.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Wait for the publishing thread to finish
        if pub_thread.is_alive():
            pub_thread.join()

        self.assertTrue(self.msg_received.is_set(), "No Skeleton2DList received after publishing an image.")
        self.assertGreaterEqual(len(self.skeletons_msg.bboxes), 1,
            "Skeleton2DList received but contains zero bounding boxes.")  # Always contains 10
        # Check that the first bounding box has valid coordinates
        self.assertGreater(self.skeletons_msg.bboxes[0].xmin, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].ymin, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].xmax, 0)
        self.assertGreater(self.skeletons_msg.bboxes[0].ymax, 0)
        # Check that the first Skeleton2D has landmarks and valid coordinates
        self.assertGreaterEqual(len(self.skeletons_msg.skeletons), 1,
            "Skeleton2DList received but contains zero bounding boxes.")   # Always contains 10
        # Check for full body: 18 landmarks
        for landmark_idx in range(18):
            self.assertGreater(self.skeletons_msg.skeletons[0].skeleton[landmark_idx].x, 0)
            self.assertGreater(self.skeletons_msg.skeletons[0].skeleton[landmark_idx].y, 0)

    def test_three_poses_detection(self):
        """
        Test that after publishing an image, multiples poses are published in the corresponding topic.
        """
        # Path to test image
        img_path = os.path.join(os.path.dirname(__file__), "images/test_image_three_persons.png")
        self.assertTrue(os.path.exists(img_path), f"Test image not found at {img_path}")
        # Load image from file
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.node.get_clock().now().to_msg()

        # Create and publish matching CameraInfo
        info_msg = make_cam_info(640, 480)
        info_msg.header = img_msg.header
        # Publish cam info once
        self.pub_info.publish(info_msg)

        # Start publishing the image for 5 seconds at 30 Hz in a separate thread
        pub_thread = threading.Thread(target=self.pub_loop, args=(img_msg,))
        pub_thread.start()

        # Spin until messages arrives or timeout
        timeout = time.time() + 15.0
        while rclpy.ok() and time.time() < timeout and not self.msg_received.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Wait for the publishing thread to finish
        if pub_thread.is_alive():
            pub_thread.join()

        self.assertTrue(self.msg_received.is_set(), "No Skeleton2DList received after publishing an image.")
        self.assertGreaterEqual(len(self.skeletons_msg.bboxes), 1,
            "Skeleton2DList received but contains zero bounding boxes.")  # Always contains 10
        # Check that the first bounding box has valid coordinates
        for i in range(3):
            # There is a person in the left, center and right of the image
            # Person on the left might have xmin = 0
            self.assertGreater(self.skeletons_msg.bboxes[i].ymin, 0)
            self.assertGreater(self.skeletons_msg.bboxes[i].xmax, 0)
            self.assertGreater(self.skeletons_msg.bboxes[i].ymax, 0)
        # Check that the first Skeleton2D has landmarks and valid coordinates
        self.assertGreaterEqual(len(self.skeletons_msg.skeletons), 1,
            "Skeleton2DList received but contains zero bounding boxes.")   # Always contains 10
        for i in range(3):
            # There might be partial bodies, so we cannot expect all landmarks to be greater than 0
            # Testing image only has upper landmarks
            for landmark_idx in minimal_landmarks:
                print(f"Checking landmark {landmark_idx} for skeleton {i}")
                self.assertGreater(self.skeletons_msg.skeletons[i].skeleton[landmark_idx].x, 0)
                self.assertGreater(self.skeletons_msg.skeletons[i].skeleton[landmark_idx].y, 0)

    def test_no_poses_on_empty_image(self):
        """
        Test that publishing an empty image does not produce any poses.
        """
        # Path to test image
        img_path = os.path.join(os.path.dirname(__file__), "images/test_image_tree.png")
        self.assertTrue(os.path.exists(img_path), f"Test image not found at {img_path}")
        # Load image from file
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        self.assertIsNotNone(img, f"Failed to read image at {img_path}")
        height, width = img.shape[:2]
        # Convert to ROS Image message
        img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        img_msg.header.stamp = self.node.get_clock().now().to_msg()

        # Create and publish matching CameraInfo
        info_msg = make_cam_info(width, height)
        info_msg.header = img_msg.header
        # Publish cam info once
        self.pub_info.publish(info_msg)

        # Start publishing the image for 5 seconds at 30 Hz in a separate thread
        pub_thread = threading.Thread(target=self.pub_loop, args=(img_msg,))
        pub_thread.start()

        # Spin until messages arrives or timeout
        timeout = time.time() + 15.0
        while rclpy.ok() and time.time() < timeout and not self.msg_received.is_set():
            rclpy.spin_once(self.node, timeout_sec=0.1)

        # Wait for the publishing thread to finish
        if pub_thread.is_alive():
            pub_thread.join()

        self.assertFalse(self.msg_received.is_set(), "Skeleton2DList received after publishing emtpy image.")
        self.assertEqual(self.skeletons_msg, None)


if __name__ == '__main__':
    unittest.main()
