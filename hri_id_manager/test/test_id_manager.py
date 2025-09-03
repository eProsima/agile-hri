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

# tests/test_id_manager.py

# Unit tests for the IDManager class in hri_id_manager

import unittest
import rclpy
from hri_id_manager.node_id_manager import IDManager, _face_landmarks


class TestIDManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize ROS 2 client library once for all tests
        rclpy.init()
        # Create the node under test
        cls.node = IDManager()
        # Force deterministic IDs for reproducibility
        cls.node.deterministic_ids = True
        IDManager.last_id = 0

    @classmethod
    def tearDownClass(cls):
        # Clean up ROS 2 node and shutdown
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_generate_id_deterministic(self):
        """Test that deterministic IDs increment with leading zeros."""
        id1 = self.node.generate_id()
        id2 = self.node.generate_id()
        id3 = self.node.generate_id()
        self.assertEqual(id1, 'f00000')
        self.assertEqual(id2, 'f00001')
        self.assertEqual(id3, 'f00002')
        self.node.deterministic_ids = False  # Reset to random IDs
        id4 = self.node.generate_id()
        self.assertNotEqual(id4, 'f00003')

    def test_generate_id_random(self):
        """Test random ID format (5 lowercase letters)."""
        self.node.deterministic_ids = False
        idr = self.node.generate_id()
        self.assertTrue(len(idr) == 5)
        self.assertTrue(idr.islower())

    def test_calculate_overlapping_full(self):
        """Test overlapping when face_roi fully inside body_roi."""
        face = [0.2, 0.2, 0.4, 0.4]
        body = [0.0, 0.0, 1.0, 1.0]
        ov = self.node.calculate_overlapping(face, body)
        self.assertEqual(ov, 1.0)

    def test_calculate_overlapping_partial(self):
        """Test overlapping partial area."""
        face = [0.0, 0.0, 0.5, 0.5]   # Area = 0.25
        body = [0.25, 0.25, 0.75, 0.75]  # Intersection area = 0.25 * (1/4) = 0.0625
        ov = self.node.calculate_overlapping(face, body)
        expected = (0.25 * 0.25) / 0.25
        self.assertEqual(ov, expected)

    def test_calculate_face_from_body_with_landmarks(self):
        """Test face reference point from skeleton landmarks."""
        # Prepare a fake skeleton with five landmarks at known coords
        class FakeLandmark:
            def __init__(self, x, y):
                self.x = x
                self.y = y
        skeleton = [FakeLandmark(0.42, 0.42)] * 18  # Assuming 18 landmarks for a full body skeleton
        # Put face landmarks at (0.1, 0.2), (0.2, 0.4), (0.3, 0.6), (0.4, 0.8), (0.5, 1.0)
        for idx, point in enumerate(_face_landmarks):
            skeleton[point] = FakeLandmark(0.1 * (idx + 1), 0.2 * (idx + 1))
        roi = [0.0, 0.0, 1.0, 1.0]
        ref = self.node.calculate_face_from_body(skeleton, roi)
        # Average x = (0.1 + 0.2 + 0.3 + 0.4 + 0.5)/5 = 0.3 ; average y = (0.2 + 0.4 + 0.6 + 0.8 + 1.0)/5 = 0.6
        self.assertEqual(ref[0], 0.3)
        self.assertEqual(ref[1], 0.6)

    def test_calculate_face_from_body_without_landmarks(self):
        """Test fallback ref point when no landmarks present."""
        skeleton = [None] * 18  # No landmarks
        roi = [0.0, 0.0, 2.0, 1.0]
        ref = self.node.calculate_face_from_body(skeleton, roi)
        # center x = 1.0 ; y = 0 * 0.75 + 1.0 * 0.25 = 0.25
        self.assertAlmostEqual(ref[0], 1.0)
        self.assertAlmostEqual(ref[1], 0.25)

if __name__ == '__main__':
    unittest.main()
