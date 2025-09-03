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

# tests/test_pose_detect.py

# Unit tests for the NodePoseDetect class in hri_pose_detect

import unittest
import rclpy
from hri_pose_detect.node_pose_detect import NodePoseDetect


class TestNodePoseDetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize ROS 2 client library once for all tests
        rclpy.init()
        # Create the node under test
        cls.node = NodePoseDetect()

    @classmethod
    def tearDownClass(cls):
        # Clean up ROS 2 node and shutdown
        cls.node.destroy_node()
        rclpy.shutdown()

    def test_generate_temp_id_format(self):
        """Test that generate_temp_id returns 'temp_' + 5 lowercase letters."""
        temp_id = self.node.generate_temp_id()
        # Must start with 'temp_'
        self.assertTrue(temp_id.startswith('temp_'))
        # The rest must be 5 lowercase letters
        suffix = temp_id[len('temp_'):]
        self.assertEqual(len(suffix), 5)
        self.assertTrue(suffix.islower())
        self.assertTrue(suffix.isalpha())


if __name__ == '__main__':
    unittest.main()
