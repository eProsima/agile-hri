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

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor, ExternalShutdownException
from rclpy.lifecycle import Node, TransitionCallbackReturn
from rclpy.lifecycle.node import LifecycleState
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, qos_profile_system_default
from rclpy.time import Time

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_msgs.msg import Face2DList, FacialLandmarks
from hri_msgs.srv import PersonID
from lifecycle_msgs.msg import State
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageCamera
from std_msgs.msg import Header

from copy import deepcopy
from cv_bridge import CvBridge
from scipy.optimize import linear_sum_assignment
from threading import Lock
from typing import Dict
import numpy as np
import random
import time

from hri_face_detect.face_detector import Face, FaceDetector, MeshDetector, BoundingBox, bbs_match, time_match, distance_rois

# Minimum image fraction of a face detection bounding box to trigger face mesh detection
MIN_FACE_IMAGE_FRAC_FOR_MESH = 0.02

# Min number of frames a face is detected before it is tracked or considered a new face
# That is, the face must be detected in at least this many consecutive frames
MIN_FRAMES_FACE_TRACKING = 4

# Max number of frames a face is not detected before it is not tracked anymore
# That is, the face must NOT be detected this many consecutive frames to stop being tracked
MAX_FRAMES_FACE_RETENTION = 5

# Face detection processing time in ms triggering a diagnostic warning
FACE_DETECTION_PROC_TIME_WARN = 500.
# Face detection processing time in ms triggering a diagnostic error
FACE_DETECTION_PROC_TIME_ERROR = 2000.


class NodeFaceDetect(Node):
    """
    ROS 2 LifecyleNode managing the face detection from an image stream.
    """
    def __init__(self):
        super().__init__('hri_face_detect')
        self.image_lock = Lock()
        self.proc_lock = Lock()

        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing input images'))
        self.declare_parameter(
            'confidence_threshold', 0.75, ParameterDescriptor(
                description='Face detection confidence threshold'))
        self.declare_parameter(
            'image_scale', 0.25, ParameterDescriptor(
                description='Input scale for the image processing pipeline wrt 640x480 pixels'))
        self.declare_parameter(
            'face_mesh', False, ParameterDescriptor(
                description='Enable face mesh output for near faces'))
        self.declare_parameter(
            'use_diagnosis', False, ParameterDescriptor(
                description='Enable additional topic for diagnosis'))
        self.declare_parameter(
            "diagnostic_period", 5., ParameterDescriptor(
                description="Diagnostic period"))
        self.declare_parameter(
            'id_timeout', 7., ParameterDescriptor(
                description='Timeout in seconds for the ID manager service'))
        self.declare_parameter(
            "use_time_offset", False, ParameterDescriptor(
                description="Use first image timestamp as offset to compute time differences"))

        # Processing image loop runs in a separate thread, as well as the client
        self.client_cbg = MutuallyExclusiveCallbackGroup()
        self.timer_cbg = MutuallyExclusiveCallbackGroup()

        self.get_logger().info('State: Unconfigured.')

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.face_detector = None
        self.mesh_detector = None
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.processing_rate = self.get_parameter('processing_rate').value
        self.use_diagnosis = self.get_parameter('use_diagnosis').value
        self.diag_period = self.get_parameter("diagnostic_period").value
        self.client_id_timeout = self.get_parameter('id_timeout').value
        self.use_time_offset = self.get_parameter('use_time_offset').value

        self.face_detector = FaceDetector(
            self.get_parameter('confidence_threshold').value,
            self.get_parameter('image_scale').value)
        if self.get_parameter('face_mesh').value:
            self.mesh_detector = MeshDetector()
            self.mesh_pub = self.create_publisher(FacialLandmarks, '/humans/faces/mesh', 1)
            self.get_logger().info('MeshDetector initialized.')
        else:
            self.mesh_detector = None
            self.mesh_pub = None
            self.get_logger().info('MeshDetector disabled.')

        self.faces_pub = self.create_publisher(Face2DList, '/humans/faces', 1)

        self.get_logger().info('State: Configured and Inactive.')
        return super().on_configure(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.reset_faces()
        self.destroy_ros_interfaces()
        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image = None
        self.image_msg_header = None
        self.new_image = False
        self.skipped_images = 0
        self.last_id = 0
        # Dict used to store the current faces(@Face) in the image
        self.detected_faces: Dict[str, Face] = dict()

        self.start_skipping_ts = self.get_clock().now()
        self.detection_start_proc_time = self.get_clock().now()
        self.detection_proc_duration_ms = 0.

        # Subscribers
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        self.image_sub = self.create_subscription(ImageCamera, 'image', self.image_callback, qos_sensor_data)
        self.image_info_sub = self.create_subscription(
            CameraInfo, 'camera_info', self.info_callback, qos_sensor_data)

        # ID manager Client
        self.client_id = self.create_client(PersonID, 'assign_id', callback_group=self.client_cbg)
        self.get_logger().info('Waiting for ID manager service...')
        while not self.client_id.wait_for_service(timeout_sec=self.client_id_timeout):
            self.get_logger().warning('ID manager service not available, keep waiting...')
        self.get_logger().info(f"Timeout for ID assignation set to: {self.client_id_timeout} sec.")

        # Main processing loop
        self.proc_timer = self.create_timer(1/self.processing_rate, self.process_image, callback_group=self.timer_cbg)

        # Diagnosis
        if self.use_diagnosis:
            self.diag_pub = self.create_publisher(DiagnosticArray, '/diagnostics', 1)
            self.diag_timer = self.create_timer(1/self.diag_period, self.do_diagnosis)

        self.get_logger().info(
            f'Waiting for images to be published on {self.image_sub.topic_name} .')
        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.id == State.PRIMARY_STATE_ACTIVE:
            self.reset_faces()
            self.destroy_ros_interfaces()
        self.face_detector = None
        self.mesh_detector = None
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def reset_faces(self):
        """Reset the face detection data."""
        self.detected_faces.clear()

    def destroy_ros_interfaces(self):
        """Destroy all ROS interfaces."""
        if self.use_diagnosis:
            self.destroy_timer(self.diag_timer)
            self.destroy_publisher(self.diag_pub)

        self.destroy_timer(self.proc_timer)
        self.destroy_subscription(self.image_sub)
        if self.image_info_sub:
            self.destroy_subscription(self.image_info_sub)

        self.destroy_publisher(self.faces_pub)

    def do_diagnosis(self):
        """Perform diagnostic operations and publish data."""
        now = self.get_clock().now()
        arr = DiagnosticArray(header=Header(stamp=now.to_msg()))
        msg = DiagnosticStatus(
            name='Social perception: Face analysis: Detection', hardware_id='none')

        current_proc_duration = (now - self.detection_start_proc_time).nanoseconds / 1e9
        if ((current_proc_duration > FACE_DETECTION_PROC_TIME_ERROR) and self.image_lock.locked()):
            msg.level = DiagnosticStatus.ERROR
            msg.message = 'Face detection process not responding'
        elif self.detection_proc_duration_ms > FACE_DETECTION_PROC_TIME_WARN:
            msg.level = DiagnosticStatus.WARN
            msg.message = 'Face detection processing is slow'
        else:
            msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key='Package name', value='hri_face_detect'),
            KeyValue(key='Currently detected faces', value=str(len(self.detected_faces))),
            KeyValue(key='Last detected face ID', value=str(self.last_id)),
            KeyValue(
                key='Detection processing time', value=f'{self.detection_proc_duration_ms:.2f}ms')]

        arr.status = [msg]
        self.diag_pub.publish(arr)

    def info_callback(self, msg: CameraInfo):
        """Callback to save metadata about the camera. Can be used for calibration"""
        if not hasattr(self, 'cam_info_msg'):
            self.cam_info_msg = msg
            self.k = np.zeros((3, 3), np.float32)
            self.k[0][0:3] = self.cam_info_msg.k[0:3]
            self.k[1][0:3] = self.cam_info_msg.k[3:6]
            self.k[2][0:3] = self.cam_info_msg.k[6:9]

        self.destroy_subscription(self.image_info_sub)

    def image_callback(self, msg: ImageCamera):
        """
        Callback to save the image and its header.
        It runs a simple logic to check if the processing is too slow.
        """
        with self.image_lock:
            self.image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_msg_header = msg.header

            if not hasattr(self, 'first_image'):
                self.offset = Time().from_msg(msg.header.stamp) if self.use_time_offset else Time()
                self.get_logger().info(f"Offset set to {self.offset}")
                self.first_image = True

            if self.new_image:
                self.skipped_images += 1
                if self.skipped_images > 100:
                    now = self.get_clock().now()
                    skip_time = (now - self.start_skipping_ts).nanoseconds / 1e9
                    self.get_logger().warn(
                        "Face_detect's processing too slow. "
                        f'Skipped 100 new incoming image over the last {skip_time:.1f}sec')
                    self.start_skipping_ts = now
                    self.skipped_images = 0
            self.new_image = True

    def generate_temp_id(self):
        """Generate a temporary ID for a face."""
        return 'temp_' + "".join(random.sample("abcdefghijklmnopqrstuvwxyz", 5))

    def request_id(self, bb: BoundingBox, point: list):
        """Request a new ID for a face to the ID manager."""
        id_req = PersonID.Request()
        id_req.xmin, id_req.ymin, id_req.xmax, id_req.ymax = bb.get_norm_coords()
        id_req.xref, id_req.yref = point[0], point[1]
        id_req.type = PersonID.Request.FACE
        self.get_logger().debug(f"Requesting ID for face at {bb.get_norm_coords()} and ref point: {point}")
        future = self.client_id.call_async(id_req)

        # Manually lock the thread until the future is done or timeout
        now = self.get_clock().now().nanoseconds
        while rclpy.ok() and not future.done() and not future.cancelled():
            if self.get_clock().now().nanoseconds - now > (self.client_id_timeout * 1e9):
                self.get_logger().warning("Timeout for ID assignation response")
                future.cancel()
            time.sleep(0.1)
        if future.done():
            try:
                result = future.result()
            except Exception as e:
                self.get_logger().error(f"Service call failed {str(e)}")
                return None
            else:
                self.get_logger().debug(f"Return {result.id}")
                return result.id
        return None

    def process_image(self):
        """Processing loop for the image. It runs the detection and publishes the results."""
        if (not self.new_image) or (not self.proc_lock.acquire(blocking=False)):
            return

        try:
            with self.image_lock:
                self.detection_start_proc_time = self.get_clock().now()
                image = deepcopy(self.image)
                image_msg_header = deepcopy(self.image_msg_header)
                self.new_image = False

            # Copy the list of Faces ID before iterating over detection, so that we
            # can delete non-existant faces at the end.
            knownIds = list(self.detected_faces.keys())

            # Run model inference
            face_detections = self.face_detector.detect(image)

            # Run fase mesh detection only if it is enabled, at least one face is detected and it is sufficiently big
            image_area_px = image.shape[0] * image.shape[1]
            if (
                self.mesh_detector
                and len(face_detections)
                and any((d.bb.width * d.bb.height) / image_area_px > MIN_FACE_IMAGE_FRAC_FOR_MESH
                        for d in face_detections)
            ):
                mesh_detections = self.mesh_detector.detect(image)

                if len(face_detections) and len(mesh_detections):
                    # Find best association between faces and meshes using as cost the bounding boxes distances.
                    # In the cost matrix the rows are ordered by face detection and columns by meshes.
                    cost_matrix = np.array([
                        [distance_rois(fd.bb, md.bb) for md in mesh_detections]
                        for fd in face_detections
                    ])
                    fd_indices, md_indices = linear_sum_assignment(cost_matrix)
                    for fd_idx, md_idx in zip(fd_indices, md_indices):
                        # Substitute the landmarks of the matching face detections with the mesh detection ones
                        face_detections[fd_idx].landmarks = mesh_detections[md_idx].landmarks

            # Set of current IDs to avoid duplicates if two detections are too close
            currentIds = set()

            for detection in face_detections:

                # Have we seen this face before? -> check whether or not bounding boxes overlap
                face = None
                min_dist = float('inf')

                for prev_face in self.detected_faces.values():
                    if bbs_match(prev_face.bb, detection.bb) and time_match(self.get_clock().now(), prev_face.last_detection_time, self.detection_proc_duration_ms, self.offset):
                        dist = distance_rois(prev_face.bb, detection.bb)
                        if dist < min_dist:
                            min_dist = dist
                            face = prev_face

                if not face:
                    face = Face(self, self.generate_temp_id())
                    face.initial_detection_time = Time().from_msg(image_msg_header.stamp)
                    self.detected_faces[face.id] = face
                    self.last_id = face.id

                # Update the face with its current position and landmarks
                face.score = detection.score
                face.bb = detection.bb
                face.landmarks = detection.landmarks
                face.last_detection_time = Time().from_msg(image_msg_header.stamp)

                face.nb_frames_visible += 1

                # If the face has been detected in MIN_FRAMES_FACE_TRACKING consecutive frames, we assign the final ID.
                if face.nb_frames_visible == MIN_FRAMES_FACE_TRACKING:
                    # Ask for definitive ID
                    id = self.request_id(face.bb, face.ref_face_point())
                    if id is None:
                        self.get_logger().error("Could not get a new ID for the face.")
                        continue
                    # Replace the temporary ID with the definitive one
                    self.detected_faces[id] = self.detected_faces.pop(face.id)
                    # In case there were two detections associated to the same body and the new ID is assigned in
                    # the second detection, we need to remove the old ID from the set of currentIDs
                    currentIds.discard(face.id)
                    knownIds.remove(face.id)
                    knownIds.append(id)
                    face.change_id(id)
                    face.set_publish(True)
                    self.get_logger().info(f"Started publishing face {face.id}.")

                # Reset the number of frames since last detection
                if face.nb_frames_since_last_detection > 0:
                    self.get_logger().debug(f"Face {face.id} detected again.")
                    face.nb_frames_since_last_detection = 0
                    face.set_publish(True)

                currentIds.add(face.id)

            # Iterate over faces not seen anymore, and unregister corresponding publishers
            for id in knownIds:
                if id not in currentIds:
                    face = self.detected_faces[id]
                    face.nb_frames_since_last_detection += 1
                    face.set_publish(False)
                    if face.nb_frames_since_last_detection > MAX_FRAMES_FACE_RETENTION:
                        self.get_logger().debug(f"Deleting face {id}.")
                        del self.detected_faces[id]

            # Create msg and create new data
            main_msg = Face2DList()
            # This is the maximum number of faces that can be published in a single message.
            # It depends on the Face2DList definition in hri_msgs/msg/Face2DList.idl
            max_len_face2dlist = len(main_msg.bboxes)
            main_msg.header = image_msg_header
            num_faces = 0
            ids_print = ''
            pub = False
            for id in currentIds:
                face = self.detected_faces[id]
                if face.do_publish:
                    pub = True
                    # Create ROI and Face2D submsgs and add them to the main msg
                    roi_msg, face_msg = face.create_msgs(image, image_msg_header)
                    main_msg.bboxes[num_faces] = roi_msg
                    main_msg.landmarks[num_faces] = face_msg
                    num_faces += 1
                    ids_print += f"[{id}] | "
                    if self.mesh_detector:
                        # Publish an extra message with the face mesh
                        mesh_msg = face.create_mesh_msg(image, image_msg_header)
                        mesh_msg.height = image.shape[0]
                        mesh_msg.width = image.shape[1]
                        self.mesh_pub.publish(mesh_msg)
                if num_faces >= max_len_face2dlist:
                    self.get_logger().warn(f"Too many faces detected. Only the first {max_len_face2dlist} will be published.")
                    break

            if pub:
                self.faces_pub.publish(main_msg)

            self.detection_proc_duration_ms = (
                self.get_clock().now() - self.detection_start_proc_time).nanoseconds / 1e6
            self.get_logger().debug(f"Pub Face: {ids_print}in {self.detection_proc_duration_ms} (ms).")
        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")
            raise e
        finally:
            self.proc_lock.release()


def main(args=None):
    rclpy.init(args=args)
    node = NodeFaceDetect()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
