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

# test_hri_id_manager_launch.py

# Integration tests for the hri_id_manager node. Simulating the launch_testing framework

import signal
import subprocess
import time
import os
import unittest
import rclpy

from hri_msgs.msg import Skeleton2DList, Skeleton2D, Face2D, Face2DList, NormalizedRegionOfInterest2D
from hri_msgs.srv import PersonID


class TestHRIIDManagerIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ROS_DOMAIN_ID"] = "116"
        os.environ["ROS_AUTOMATIC_DISCOVERY_RANGE"] = "LOCALHOST"
        rclpy.init()

    @classmethod
    def tearDownClass(cls):
        rclpy.shutdown()

    def setUp(self):
        self.proc = subprocess.Popen(
            ["ros2", "launch", "hri_id_manager", "id_manager.launch.py", "log-level:=debug"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=os.environ.copy(),        # Inherit + our overrides
        )

        time.sleep(3)  # Wait for the node to start

        print("Setting up test client node")
        self.node = rclpy.create_node('test_client')

    def tearDown(self):
        print("Tearing down test client node")
        self.node.destroy_node()

        self.proc.send_signal(signal.SIGINT)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        print("OUTPUT:")
        stdout, stderr = self.proc.communicate()
        if stdout:
            print(stdout.decode('utf-8'))
        if stderr:
            print(stderr.decode('utf-8'))

    def _call_person_id(self, client, req):
        """Helper function to call the PersonID service."""
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=2.0)
        return future.result()

    def test_service_id_assignment_face(self):
        """
        Test that the service /assign_id responds with a valid ID:
        - First, it should assign an ID to a body request.
        - Then, it should assign the same ID to a face request matching the same ROI.
        - Finally, it should assign a different ID to a subsequent face request matching the same ROI
        """
        client = self.node.create_client(PersonID, 'assign_id')
        # Wait for service
        assert client.wait_for_service(timeout_sec=5.0)

        # Prepare a "body" request
        req = PersonID.Request()
        req.type = 'body'
        req.xmin = 0.1; req.ymin = 0.1; req.xmax = 0.4; req.ymax = 0.4
        req.xref = 0.25; req.yref = 0.25
        resp = self._call_person_id(client, req)
        # Should return a non-empty string
        self.assertTrue(isinstance(resp.id, str) and len(resp.id) > 0)

        # Now send a "face" matching the same ROI:
        req2 = PersonID.Request()
        req2.type = 'face'
        req2.xmin = 0.2; req2.ymin = 0.2; req2.xmax = 0.3; req2.ymax = 0.3
        req2.xref = 0.25; req2.yref = 0.25
        resp2 = self._call_person_id(client, req2)
        # ID should match the previous one
        self.assertTrue(isinstance(resp2.id, str) and len(resp2.id) > 0)
        self.assertEqual(resp.id, resp2.id)

        # Now send another "face" matching the same ROI, it should return different ID:
        resp3 = self._call_person_id(client, req2)
        # ID should match the previous one
        self.assertTrue(isinstance(resp3.id, str) and len(resp3.id) > 0)
        self.assertNotEqual(resp.id, resp3.id)

    def test_service_id_assignment_body(self):
        """
        Test that the service /assign_id responds with a valid ID:
        - First, it should assign an ID to a face request.
        - Then, it should assign the same ID to a body request matching the same ROI.
        - Finally, it should assign a different ID to a subsequent body request matching the same ROI
        """
        client = self.node.create_client(PersonID, 'assign_id')
        # Wait for service
        assert client.wait_for_service(timeout_sec=5.0)

        # Prepare a "face" request
        req = PersonID.Request()
        req.type = 'face'
        req.xmin = 0.2; req.ymin = 0.2; req.xmax = 0.3; req.ymax = 0.3
        req.xref = 0.25; req.yref = 0.25
        resp = self._call_person_id(client, req)
        # Should return a non-empty string
        self.assertTrue(isinstance(resp.id, str) and len(resp.id) > 0)

        # Now send a "body" matching the same ROI:
        req2 = PersonID.Request()
        req2.type = 'body'
        req2.xmin = 0.1; req2.ymin = 0.1; req2.xmax = 0.4; req2.ymax = 0.4
        req2.xref = 0.25; req2.yref = 0.25
        resp2 = self._call_person_id(client, req2)
        # ID should match the previous one
        self.assertTrue(isinstance(resp2.id, str) and len(resp2.id) > 0)
        self.assertEqual(resp.id, resp2.id)

        # Now send another "body" matching the same ROI, it should return different ID:
        resp3 = self._call_person_id(client, req2)
        # ID should match the previous one
        self.assertTrue(isinstance(resp3.id, str) and len(resp3.id) > 0)
        self.assertNotEqual(resp.id, resp3.id)

    def test_two_candidates_for_face(self):
        """
        Test that the service /assign_id responds with a valid ID:
        - First, it should assign an ID to a body request.
        - Then, it should assign the same ID to a face request matching the same ROI.
        - Finally, it should assign a different ID to a subsequent face request matching the same ROI
        """
        client = self.node.create_client(PersonID, 'assign_id')
        # Wait for service
        assert client.wait_for_service(timeout_sec=5.0)

        # Prepare a "body" request
        req = PersonID.Request()
        req.type = 'body'
        req.xmin = 0.1; req.ymin = 0.1; req.xmax = 0.4; req.ymax = 0.4
        req.xref = 0.25; req.yref = 0.3
        resp = self._call_person_id(client, req)
        self.assertTrue(isinstance(resp.id, str) and len(resp.id) > 0)

        # Prepare another "body" request with a similar ROI:
        req.xmin = 0.2; req.ymin = 0.1; req.xmax = 0.5; req.ymax = 0.4
        req.xref = 0.35; req.yref = 0.3
        resp2 = self._call_person_id(client, req)
        self.assertTrue(isinstance(resp2.id, str) and len(resp2.id) > 0)
        self.assertNotEqual(resp.id, resp2.id)

        # Now send a "face" matching both ROIs but closes to second body:
        req2 = PersonID.Request()
        req2.type = 'face'
        req2.xmin = 0.3; req2.ymin = 0.25; req2.xmax = 0.4; req2.ymax = 0.35
        req2.xref = 0.35; req2.yref = 0.3
        resp3 = self._call_person_id(client, req2)
        # ID should match the second body
        self.assertTrue(isinstance(resp3.id, str) and len(resp3.id) > 0)
        self.assertEqual(resp2.id, resp3.id)
        self.assertNotEqual(resp.id, resp3.id)

    def test_face_movement_and_body_id_matching(self):
        """
        Simulate a face moving by publishing a Face2DList message with an updated position.
        Then check that a new body request close to the new position returns the same ID as the face.
        """
        client = self.node.create_client(PersonID, 'assign_id')
        # Wait for service
        assert client.wait_for_service(timeout_sec=5.0)

        # Initial face request
        req = PersonID.Request()
        req.type = 'face'
        req.xmin = 0.2; req.ymin = 0.2; req.xmax = 0.3; req.ymax = 0.3
        req.xref = 0.25; req.yref = 0.25
        resp = self._call_person_id(client, req)
        self.assertTrue(isinstance(resp.id, str) and len(resp.id) > 0)
        face_id = resp.id

        # Now send a body request where the the new face position will be
        # This should return a new ID because the face has not moved yet
        req2 = PersonID.Request()
        req2.type = 'body'
        req2.xmin = 0.35; req2.ymin = 0.35; req2.xmax = 0.55; req2.ymax = 0.55
        req2.xref = 0.45; req2.yref = 0.45
        resp2 = self._call_person_id(client, req2)
        self.assertTrue(isinstance(resp2.id, str) and len(resp2.id) > 0)
        self.assertNotEqual(face_id, resp2.id)

        # Create publisher for Face2DList
        face_pub = self.node.create_publisher(Face2DList, '/humans/faces', 1)
        time.sleep(0.5)  # Allow publisher to be ready

        # Publish updated face position with same key
        face_msg = Face2DList()
        face = Face2D()
        face_roi = NormalizedRegionOfInterest2D()
        face.key = face_id
        # face landmarks are not used in this test
        face_roi.key = face_id
        face_roi.xmin = 0.4
        face_roi.ymin = 0.4
        face_roi.xmax = 0.5
        face_roi.ymax = 0.5
        self.assertNotEqual(len(face_msg.landmarks), 0)
        self.assertNotEqual(len(face_msg.bboxes), 0)
        face_msg.landmarks[0] = face
        face_msg.bboxes[0] = face_roi
        face_pub.publish(face_msg)
        rclpy.spin_once(self.node, timeout_sec=0.5)
        # Give time for the node to process the update but less than the TIME_MARGIN_DETECTION
        # or face will be discarded for old detection
        time.sleep(0.1)

        # Now send the same body request close to the new face position
        resp3 = self._call_person_id(client, req2)
        self.assertTrue(isinstance(resp3.id, str) and len(resp3.id) > 0)
        self.assertEqual(face_id, resp3.id)
        self.assertNotEqual(resp2.id, resp3.id)

        # Clean up publisher
        face_pub.destroy()

    def test_body_movement_and_face_id_matching(self):
        """
        Simulate a body moving by publishing a Skeleton2DList message with an updated position.
        Then check that a new body request close to the new position returns the same ID as the body.
        """
        client = self.node.create_client(PersonID, 'assign_id')
        # Wait for service
        assert client.wait_for_service(timeout_sec=5.0)

        # Initial body request
        req = PersonID.Request()
        req.type = 'body'
        req.xmin = 0.1; req.ymin = 0.2; req.xmax = 0.3; req.ymax = 0.5
        req.xref = 0.25; req.yref = 0.25
        resp = self._call_person_id(client, req)
        self.assertTrue(isinstance(resp.id, str) and len(resp.id) > 0)
        body_id = resp.id

        # Now send a face request where the the new body position will be
        # This should return a new ID because the body has not moved yet
        req2 = PersonID.Request()
        req2.type = 'face'
        req2.xmin = 0.45; req2.ymin = 0.2; req2.xmax = 0.55; req2.ymax = 0.3
        req2.xref = 0.5; req2.yref = 0.25
        resp2 = self._call_person_id(client, req2)
        self.assertTrue(isinstance(resp2.id, str) and len(resp2.id) > 0)
        self.assertNotEqual(body_id, resp2.id)

        # Create publisher for Skeleton2DList
        body_pub = self.node.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        time.sleep(0.5)  # Allow publisher to be ready

        # Publish updated face position with same key
        body_msg = Skeleton2DList()
        body = Skeleton2D()
        body_roi = NormalizedRegionOfInterest2D()
        body.key = body_id
        # body skeleton are not used in this test
        body_roi.key = body_id
        body_roi.xmin = 0.4
        body_roi.ymin = 0.2  # Only X movement
        body_roi.xmax = 0.6
        body_roi.ymax = 0.5
        self.assertNotEqual(len(body_msg.skeletons), 0)
        self.assertNotEqual(len(body_msg.bboxes), 0)
        body_msg.skeletons[0] = body
        body_msg.bboxes[0] = body_roi
        body_pub.publish(body_msg)
        rclpy.spin_once(self.node, timeout_sec=0.5)
        # Give time for the node to process the update but less than the TIME_MARGIN_DETECTION
        # or body will be discarded for old detection
        time.sleep(0.1)

        # Now send the same face request close to the new body position
        resp3 = self._call_person_id(client, req2)
        self.assertTrue(isinstance(resp3.id, str) and len(resp3.id) > 0)
        self.assertEqual(body_id, resp3.id)
        self.assertNotEqual(resp2.id, resp3.id)

        # Clean up publisher
        body_pub.destroy()


if __name__ == '__main__':
    unittest.main()
