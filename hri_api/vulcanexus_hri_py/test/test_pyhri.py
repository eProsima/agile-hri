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

import copy
from datetime import timedelta
import time
import unittest

from geometry_msgs.msg import TransformStamped
from hri import (
    Expression, FacialLandmark, HRIListener, SkeletalKeypoint)
from hri_msgs.msg import (
    Audio, Face2D, Face2DList, NormalizedRegionOfInterest2D,
    Skeleton2D, Skeleton2DList, Skeleton3D, Skeleton3DList)
from hri_msgs.msg import Expression as ExpressionMsg

import rclpy
from rclpy.context import Context
from rclpy.executors import SingleThreadedExecutor
from rclpy.qos import QoSDurabilityPolicy, QoSProfile, QoSReliabilityPolicy
from tf2_ros import StaticTransformBroadcaster


class TestHRI(unittest.TestCase):
    latching_qos = QoSProfile(
        depth=1,
        durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        reliability=QoSReliabilityPolicy.RELIABLE)

    @classmethod
    def setUpClass(cls) -> None:
        cls.context = Context()
        rclpy.init(context=cls.context)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        rclpy.shutdown(context=cls.context)
        return super().tearDownClass()

    def setUp(self) -> None:
        self.tester_node = rclpy.create_node('tester_node', context=self.context)
        self.hri_listener = HRIListener('hri_node', False)
        return super().setUp()

    def tearDown(self) -> None:
        try:
            del self.hri_listener
        except AttributeError:
            pass
        self.tester_node.destroy_node()
        return super().tearDown()

    def spin(self, hri_timeout_ms=10000):
        time.sleep(0.5)
        self.hri_listener.spin_all(timedelta(milliseconds=hri_timeout_ms))

    def test_get_faces(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        empty_msg = Face2DList()
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'
        face_b = Face2D()
        face_b.key = 'B'

        self.assertEqual(faces_pub.get_subscription_count(), 1)
        self.assertEqual(len(self.hri_listener.faces), 0)

        face_list_msg.landmarks[0] = face_a
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 1)
        self.assertIn('A', faces)
        self.assertEqual(faces['A'].id, 'A')

        faces_pub.publish(face_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.faces), 1)

        face_list_msg.landmarks[1] = face_b
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 2)
        self.assertIn('A', faces)
        self.assertIn('B', faces)

        face_list_msg.landmarks = copy.deepcopy(empty_msg.landmarks)
        face_list_msg.landmarks[0] = face_b
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 1)
        self.assertNotIn('A', faces)
        self.assertIn('B', faces)

        face_list_msg.landmarks = copy.deepcopy(empty_msg.landmarks)
        faces_pub.publish(face_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.faces), 0)

        del self.hri_listener
        self.assertEqual(faces_pub.get_subscription_count(), 0)

    def test_get_faces_roi(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'
        face_b = Face2D()
        face_b.key = 'B'
        roi_msg = NormalizedRegionOfInterest2D(xmin=0.1, ymin=0.0, xmax=1.0, ymax=1.0)

        # Face A with ROI
        face_list_msg.landmarks[0] = face_a
        face_list_msg.bboxes[0] = roi_msg
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 1)
        self.assertIn('A', faces)
        self.assertEqual(faces['A'].id, 'A')
        self.assertIsNotNone(faces['A'].roi)
        self.assertAlmostEqual(faces['A'].roi[0], 0.1)
        self.assertAlmostEqual(faces['A'].roi[1], 0.0)
        self.assertAlmostEqual(faces['A'].roi[2], 0.9)
        self.assertAlmostEqual(faces['A'].roi[3], 1.0)

        # Only Face B, Face A should be removed
        face_list_msg.landmarks[0] = face_b
        roi_msg.xmax = 0.4
        roi_msg.ymax = 0.2
        face_list_msg.bboxes[0] = roi_msg
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 1)
        self.assertNotIn('A', faces)
        self.assertIn('B', faces)
        self.assertAlmostEqual(faces['B'].roi[0], 0.1)
        self.assertAlmostEqual(faces['B'].roi[1], 0.0)
        self.assertAlmostEqual(faces['B'].roi[2], 0.3)
        self.assertAlmostEqual(faces['B'].roi[3], 0.2)

        # Both A and B
        roi_msg.xmax = 0.5
        face_list_msg.bboxes[0] = copy.deepcopy(roi_msg)  # B updated
        roi_msg.xmin = 0.0
        roi_msg.ymin = 0.0
        roi_msg.xmax = 0.6
        roi_msg.ymax = 0.7
        face_list_msg.landmarks[1] = face_a
        face_list_msg.bboxes[1] = copy.deepcopy(roi_msg)
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 2)
        self.assertIsNotNone(faces['A'].roi)
        self.assertAlmostEqual(faces['A'].roi[0], 0.0)
        self.assertAlmostEqual(faces['A'].roi[1], 0.0)
        self.assertAlmostEqual(faces['A'].roi[2], 0.6)
        self.assertAlmostEqual(faces['A'].roi[3], 0.7)
        self.assertIsNotNone(faces['B'].roi)
        self.assertAlmostEqual(faces['B'].roi[0], 0.1)
        self.assertAlmostEqual(faces['B'].roi[1], 0.0)
        self.assertAlmostEqual(faces['B'].roi[2], 0.4)
        self.assertAlmostEqual(faces['B'].roi[3], 0.2)

    def test_get_bodies(self):
        bodies_pub = self.tester_node.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        empty_msg = Skeleton2DList()
        body_list_msg = Skeleton2DList()
        body_a = Skeleton2D()
        body_a.key = 'A'
        body_b = Skeleton2D()
        body_b.key = 'B'

        self.assertEqual(bodies_pub.get_subscription_count(), 1)
        self.assertEqual(len(self.hri_listener.bodies), 0)

        body_list_msg.skeletons[0] = body_a
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 1)
        self.assertIn('A', bodies)
        self.assertEqual(bodies['A'].id, 'A')

        bodies_pub.publish(body_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.bodies), 1)

        body_list_msg.skeletons[1] = body_b
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 2)
        self.assertIn('A', bodies)
        self.assertIn('B', bodies)

        body_list_msg.skeletons = copy.deepcopy(empty_msg.skeletons)
        body_list_msg.skeletons[0] = body_b
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 1)
        self.assertNotIn('A', bodies)
        self.assertIn('B', bodies)

        body_list_msg.skeletons = copy.deepcopy(empty_msg.skeletons)
        bodies_pub.publish(body_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.bodies), 0)

        del self.hri_listener
        self.assertEqual(bodies_pub.get_subscription_count(), 0)

    def test_get_bodies_3d(self):
        bodies_pub = self.tester_node.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        bodies3d_pub = \
            self.tester_node.create_publisher(Skeleton3DList, '/humans/bodies/skel3D', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        empty_msg = Skeleton2DList()
        body_list_msg = Skeleton2DList()
        body3d_list_msg = Skeleton3DList()
        body3d_a = Skeleton3D()
        body3d_a.key = 'A'
        body_a = Skeleton2D()
        body_a.key = 'A'
        body3d_b = Skeleton3D()
        body3d_b.key = 'B'
        body_b = Skeleton2D()
        body_b.key = 'B'

        self.assertEqual(bodies3d_pub.get_subscription_count(), 1)
        self.assertEqual(len(self.hri_listener.bodies), 0)

        body3d_list_msg.skeletons[0] = body3d_a
        body_list_msg.skeletons[0] = body_a
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 1)
        self.assertIn('A', bodies)
        self.assertEqual(bodies['A'].id, 'A')

        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.bodies), 1)

        body3d_list_msg.skeletons[1] = body3d_b
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 1)
        self.assertIn('A', bodies)
        self.assertNotIn('B', bodies)

        body_list_msg.skeletons[1] = body_b
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 2)
        self.assertIn('A', bodies)
        self.assertIn('B', bodies)

        body_list_msg.skeletons = copy.deepcopy(empty_msg.skeletons)
        body_list_msg.skeletons[0] = body_b
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertEqual(len(bodies), 1)
        self.assertNotIn('A', bodies)
        self.assertIn('B', bodies)

        body_list_msg.skeletons = copy.deepcopy(empty_msg.skeletons)
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.bodies), 0)

        del self.hri_listener
        self.assertEqual(bodies3d_pub.get_subscription_count(), 0)

    def test_face_callbacks(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        empty_msg = Face2DList()
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'
        face_b = Face2D()
        face_b.key = 'B'

        cb_triggered = 0

        def face_cb(_):
            nonlocal cb_triggered
            cb_triggered += 1

        def face_lost_cb(_):
            nonlocal cb_triggered
            cb_triggered -= 1

        self.hri_listener.on_face(face_cb)
        self.hri_listener.on_face_lost(face_lost_cb)

        self.assertEqual(cb_triggered, 0)
        face_list_msg.landmarks[0] = face_a
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 1)
        self.assertIn('A', faces)
        self.assertEqual(faces['A'].id, 'A')
        self.assertEqual(cb_triggered, 1)

        face_list_msg.landmarks[1] = face_b
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertEqual(len(faces), 2)
        self.assertIn('A', faces)
        self.assertIn('B', faces)
        self.assertEqual(cb_triggered, 2)

        face_list_msg.landmarks = copy.deepcopy(empty_msg.landmarks)
        faces_pub.publish(face_list_msg)
        self.spin()
        self.assertEqual(len(self.hri_listener.faces), 0)
        self.assertEqual(cb_triggered, 0)

    def test_get_faces_expression(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        expression_pub = \
            self.tester_node.create_publisher(ExpressionMsg, '/humans/faces/emotion', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'
        face_b = Face2D()
        face_b.key = 'B'

        face_list_msg.landmarks[0] = face_a
        face_list_msg.landmarks[1] = face_b
        faces_pub.publish(face_list_msg)
        self.spin()

        expression_msg = ExpressionMsg()
        expression_msg.key = 'A'
        expression_msg.expression = ExpressionMsg.HAPPY
        expression_pub.publish(expression_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertIsNotNone(faces['A'].expression)
        self.assertEqual(faces['A'].expression, Expression.HAPPY)

        expression_msg.expression = ExpressionMsg.SAD
        expression_pub.publish(expression_msg)
        self.spin()
        self.assertIsNotNone(faces['A'].expression)
        self.assertEqual(faces['A'].expression, Expression.SAD)

        expression_msg.key = 'B'
        expression_msg.expression = ExpressionMsg.ANGRY
        expression_pub.publish(expression_msg)
        self.spin()
        self.assertIsNotNone(faces['B'].expression)
        self.assertEqual(faces['B'].expression, Expression.ANGRY)
        self.assertEqual(faces['A'].expression, Expression.SAD)

    def test_image(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'

        face_list_msg.landmarks[0] = face_a
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertIn('A', faces)

        # Cropped and aligned images are not published -> expect None
        self.assertIsNone(faces['A'].cropped)

    def test_facial_action_units(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'

        face_list_msg.landmarks[0] = face_a
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertIn('A', faces)

        # FAU not published -> expect None
        self.assertIsNone(faces['A'].facial_action_units)

    def test_facial_landmarks(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'A'
        norm_poi_nose = face_a.landmarks[Face2D.NOSE]
        norm_poi_nose.x = 0.4
        norm_poi_nose.y = 0.2
        face_a.landmarks[Face2D.NOSE] = norm_poi_nose
        norm_poi_le = face_a.landmarks[Face2D.LEFT_EYE]
        norm_poi_le.x = 0.6
        norm_poi_le.y = 0.8
        face_a.landmarks[Face2D.LEFT_EYE] = norm_poi_le

        face_list_msg.landmarks[0] = face_a
        faces_pub.publish(face_list_msg)
        self.spin()
        faces = self.hri_listener.faces
        self.assertIn('A', faces)
        self.assertIsNotNone(faces['A'].facial_landmarks)
        point = faces['A'].facial_landmarks[FacialLandmark.NOSE]
        self.assertAlmostEqual(point[0], norm_poi_nose.x)
        self.assertAlmostEqual(point[1], norm_poi_nose.y)
        point = faces['A'].facial_landmarks[FacialLandmark.LEFT_EYE_INSIDE]
        self.assertAlmostEqual(point[0], norm_poi_le.x)
        self.assertAlmostEqual(point[1], norm_poi_le.y)

    def test_skeletal_keypoints(self):
        bodies_pub = self.tester_node.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        body_list_msg = Skeleton2DList()
        body_a = Skeleton2D()
        body_a.key = 'A'

        poi_nose = body_a.skeleton[Skeleton2D.NOSE]
        poi_nose.x = 0.3
        poi_nose.y = 0.5
        poi_nose.c = 0.8
        body_a.skeleton[Skeleton2D.NOSE] = poi_nose
        poi_le = body_a.skeleton[Skeleton2D.LEFT_EYE]
        poi_le.x = 0.6
        poi_le.y = 0.8
        poi_le.c = 0.9
        body_a.skeleton[Skeleton2D.LEFT_EYE] = poi_le

        body_list_msg.skeletons[0] = body_a
        bodies_pub.publish(body_list_msg)
        self.spin()
        bodies = self.hri_listener.bodies
        self.assertIn('A', bodies)
        self.assertIsNotNone(bodies['A'].skeleton)

        point = bodies['A'].skeleton[SkeletalKeypoint.NOSE]
        self.assertAlmostEqual(point[0], poi_nose.x)
        self.assertAlmostEqual(point[1], poi_nose.y)
        self.assertAlmostEqual(point[2], poi_nose.c)

        point = bodies['A'].skeleton[SkeletalKeypoint.LEFT_EYE]
        self.assertAlmostEqual(point[0], poi_le.x)
        self.assertAlmostEqual(point[1], poi_le.y)
        self.assertAlmostEqual(point[2], poi_le.c)

    def test_skeletal_keypoints_3d(self):
        bodies2d_pub = self.tester_node.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        bodies3d_pub = \
            self.tester_node.create_publisher(Skeleton3DList, '/humans/bodies/skel3D', 1)
        # Give some time for the entities to be created
        time.sleep(3)

        body2d_list_msg = Skeleton2DList()
        body3d_list_msg = Skeleton3DList()
        body2d_a = Skeleton2D()
        body2d_a.key = 'A'
        body3d_a = Skeleton3D()
        body3d_a.key = 'A'

        body3d_a.skeleton[Skeleton3D.NOSE].x = 0.1
        body3d_a.skeleton[Skeleton3D.NOSE].y = 0.2
        body3d_a.skeleton[Skeleton3D.NOSE].z = 1.0

        body3d_a.skeleton[Skeleton3D.LEFT_EYE].x = -0.3
        body3d_a.skeleton[Skeleton3D.LEFT_EYE].y = 0.4
        body3d_a.skeleton[Skeleton3D.LEFT_EYE].z = 0.9

        body2d_list_msg.skeletons[0] = body2d_a
        bodies2d_pub.publish(body2d_list_msg)
        self.spin()

        body3d_list_msg.skeletons[0] = body3d_a
        bodies3d_pub.publish(body3d_list_msg)
        self.spin()

        bodies = self.hri_listener.bodies
        self.assertIn('A', bodies)
        self.assertIsNotNone(bodies['A'].skeleton3d)
        p3 = bodies['A'].skeleton3d[SkeletalKeypoint.NOSE]
        self.assertAlmostEqual(p3[0], 0.1)
        self.assertAlmostEqual(p3[1], 0.2)
        self.assertAlmostEqual(p3[2], 1.0)
        p3 = bodies['A'].skeleton3d[SkeletalKeypoint.LEFT_EYE]
        self.assertAlmostEqual(p3[0], -0.3)
        self.assertAlmostEqual(p3[1], 0.4)
        self.assertAlmostEqual(p3[2], 0.9)

    def test_get_voices(self):
        voices_pub = self.tester_node.create_publisher(Audio, '/humans/voices/tracked', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        # No one should subscribe to this topic in the new API
        self.assertEqual(voices_pub.get_subscription_count(), 0)
        # Listener should expose no voices
        self.assertEqual(len(self.hri_listener.voices), 0)

    def test_get_known_persons(self):
        # Persons are not used in the new API
        self.assertEqual(len(self.hri_listener.persons), 0)

    def test_get_tracked_persons(self):
        # Tracked persons are not used in the new API
        self.assertEqual(len(self.hri_listener.tracked_persons), 0)

    def test_gaze_transform(self):
        faces_pub = self.tester_node.create_publisher(Face2DList, '/humans/faces', 1)
        # Give some time for the entities to be created
        time.sleep(3)
        face_list_msg = Face2DList()
        face_a = Face2D()
        face_a.key = 'f1'
        face_list_msg.landmarks[0] = face_a

        static_broadcaster = StaticTransformBroadcaster(self.tester_node)
        transform_msg = TransformStamped()

        self.hri_listener.set_reference_frame('base_link')
        transform_msg.header.stamp = self.tester_node.get_clock().now().to_msg()
        transform_msg.header.frame_id = 'world'
        transform_msg.child_frame_id = 'base_link'
        transform_msg.transform.translation.x = -1.0
        transform_msg.transform.translation.y = 0.0
        transform_msg.transform.translation.z = 0.0
        transform_msg.transform.rotation.w = 1.0
        static_broadcaster.sendTransform(transform_msg)
        # Ensure TF is processed
        exec_ = SingleThreadedExecutor(context=self.context)
        exec_.add_node(self.tester_node)
        exec_.spin_once(0.1)
        self.spin()

        faces_pub.publish(face_list_msg)
        self.spin()
        f1 = self.hri_listener.faces['f1']
        self.assertIsNone(f1.gaze_transform, 'no gaze transform should be available')

        transform_msg.header.stamp = self.tester_node.get_clock().now().to_msg()
        transform_msg.header.frame_id = 'base_link'
        transform_msg.child_frame_id = 'gaze_f1'
        transform_msg.transform.translation.x = 2.0
        transform_msg.transform.translation.y = 0.0
        transform_msg.transform.translation.z = 0.0
        transform_msg.transform.rotation.w = 1.0
        static_broadcaster.sendTransform(transform_msg)
        exec_.spin_once(0.1)
        self.spin()
        self.assertIsNotNone(f1.gaze_transform, 'the gaze transform should be available')
        t = f1.gaze_transform
        self.assertEqual(t.child_frame_id, 'gaze_f1')
        self.assertEqual(t.header.frame_id, 'base_link')
        self.assertAlmostEqual(t.transform.translation.x, 2.0)

        self.hri_listener.set_reference_frame('gaze_f1')
        self.assertIsNotNone(f1.gaze_transform)
        t = f1.gaze_transform
        self.assertEqual(t.child_frame_id, 'gaze_f1')
        self.assertEqual(t.header.frame_id, 'gaze_f1')
        self.assertAlmostEqual(t.transform.translation.x, 0.0)


if __name__ == '__main__':
    unittest.main()
