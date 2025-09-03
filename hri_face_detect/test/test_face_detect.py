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

# tests/test_face_detect.py

# Unit tests for the NodeFaceDetect class in hri_face_detect

import unittest
import rclpy
from rclpy.parameter import Parameter
from hri_face_detect.node_face_detect import NodeFaceDetect
from hri_face_detect.face_detector import FaceDetector, MeshDetector


class TestNodeFaceDetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize ROS 2 client library once for all tests
        rclpy.init()
        # Create the node under test
        cls.node = NodeFaceDetect()

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

    def test_reset_faces(self):
        """Test that reset_faces clears the detected_faces dict."""
        # Populate with dummy entries
        self.node.detected_faces = {'a': object(), 'b': object()}
        self.assertTrue(self.node.detected_faces)
        # Call reset
        self.node.reset_faces()
        # Check dict
        self.assertEqual(self.node.detected_faces, {})

    def test_on_configure_default(self):
        """Test on_configure with face_mesh=False creates FaceDetector and faces_pub only."""
        # Call configure; ignore return value or exceptions from superclass
        try:
            self.node.on_configure(None)
        except Exception:
            pass

        # face_detector should be set
        self.assertIsInstance(self.node.face_detector, FaceDetector)

        # mesh_detector remains None
        self.assertIsNone(self.node.mesh_detector)

        # faces_pub publisher must exist
        self.assertTrue(hasattr(self.node, 'faces_pub'))

        # Should NOT have mesh_pub
        self.assertFalse(hasattr(self.node, 'mesh_pub'))

    def test_on_configure_with_mesh(self):
        """Test on_configure with face_mesh=True creates MeshDetector and mesh_pub."""
        # Set the face_mesh parameter to True
        self.node.set_parameters([
            Parameter('face_mesh', Parameter.Type.BOOL, True)
        ])

        # Call configure again
        try:
            self.node.on_configure(None)
        except Exception:
            pass

        # mesh_detector should be an instance of MeshDetector
        self.assertIsInstance(self.node.mesh_detector, MeshDetector)

        # mesh_pub publisher must exist
        self.assertTrue(hasattr(self.node, 'mesh_pub'))

        # faces_pub still exists
        self.assertTrue(hasattr(self.node, 'faces_pub'))


if __name__ == '__main__':
    unittest.main()
