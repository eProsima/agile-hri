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

# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
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
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from message_filters import ApproximateTimeSynchronizer, Subscriber
from ament_index_python.packages import get_package_share_directory

from builtin_interfaces.msg import Time as TimeInterface
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from hri_msgs.msg import Skeleton2DList, Skeleton3DList
from hri_msgs.srv import PersonID, Reload
from lifecycle_msgs.msg import State
from rcl_interfaces.msg import ParameterDescriptor
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ImageCamera

from collections import OrderedDict
from copy import deepcopy
from cv_bridge import CvBridge
from threading import Lock
import multiprocessing as mp
import random
import time
import os

from hri_pose_detect.body_detector import Body, BodyDetector, BoundingBox, bbs_match, time_match, distance_rois

# Min number of frames a body is detected before it is tracked or considered a new body
# That is, the body must be detected in at least this many consecutive frames
MIN_FRAMES_BODY_TRACKING = 4

# Max number of frames a body is not detected before it is not tracked anymore
# That is, the body must NOT be detected this many consecutive frames to stop being tracked
# This parameter affects how keys are recycled after occlusions. It has a different purpose than MAX_TIME_BETWEEN_FRAMES,
# which focuses on avoid reusing keys if the image is freezed (and frames are not being received).
MAX_FRAMES_BODY_RETENTION = 5

# Body detection processing time in ms triggering a diagnostic warning
BODY_DETECTION_PROC_TIME_WARN = 500.
# Body detection processing time in ms signalling a timeout error
BODY_DETECTION_PROC_TIME_ERROR = 2000.


def _builtin_time_to_nanosecs(time: TimeInterface) -> int:
    """Transform builtin_interface.Time to int."""
    return (time.sec * 1e9) + time.nanosec


class NodePoseDetect(Node):
    """
    ROS 2 Node performing the pose detection. It can detect multiple bodies in the image.
    It is a Lifecycle node.
    """
    def __init__(self):
        super().__init__("hri_pose_detect")
        self.image_lock = Lock()
        self.proc_lock = Lock()

        hri_body_dir = get_package_share_directory("hri_pose_detect")
        model_path = os.path.join(hri_body_dir, 'models', 'yolov8n-pose.pt')

        self.declare_parameter(
            'yolo_model_path', model_path, ParameterDescriptor(
                description='Path to the YOLOv8 pose detection model'))
        self.declare_parameter(
            'processing_rate', 30, ParameterDescriptor(
                description='Best effort frequency for processing input images'))
        self.declare_parameter(
            'confidence_threshold', 0.55, ParameterDescriptor(
                description='Pose detection confidence threshold'))
        self.declare_parameter(
            'image_scale', 1.0, ParameterDescriptor(
                description='Input scale for the image processing pipeline wrt 640x480 pixels'))
        self.declare_parameter(
            'use_diagnosis', False, ParameterDescriptor(
                description='Enable additional topic for diagnostics'))
        self.declare_parameter(
            "diagnostic_period", 1., ParameterDescriptor(
                description="Diagnostic period"))
        self.declare_parameter(
            'id_timeout', 7., ParameterDescriptor(
                description='Timeout in seconds for the ID manager service'))
        self.declare_parameter(
            "use_depth", False, ParameterDescriptor(
                description="Use depth info from camera"))
        self.declare_parameter(
            "sync_margin", 0.05, ParameterDescriptor(
                description="Margin for RGB and Depth images sync (in seconds)"))
        self.declare_parameter(
            "use_time_offset", False, ParameterDescriptor(
                description="Use first image timestamp as offset to compute time differences"))

        # Processing image loop runs in a separate thread, as well as the client
        self.id_client_cbg = MutuallyExclusiveCallbackGroup()
        self.timer_cbg = MutuallyExclusiveCallbackGroup()
        self.yolo_reload_cbg = MutuallyExclusiveCallbackGroup()

        self.get_logger().info('State: Unconfigured.')

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.body_detector = None
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.processing_rate = self.get_parameter('processing_rate').value
        self.use_diagnosis = self.get_parameter('use_diagnosis').value
        self.diag_period = self.get_parameter("diagnostic_period").value
        self.client_id_timeout = self.get_parameter('id_timeout').value
        self.use_depth = self.get_parameter('use_depth').value
        self.sync_margin = self.get_parameter('sync_margin').value
        self.use_time_offset = self.get_parameter('use_time_offset').value

        self.body_detector = BodyDetector(
            self.get_parameter('yolo_model_path').value,
            self.get_parameter('confidence_threshold').value,
            self.get_parameter('image_scale').value,
            self
        )

        self.bodies_pub = self.create_publisher(Skeleton2DList, '/humans/bodies', 1)
        if self.use_depth:
            self.bodies3D_pub = self.create_publisher(Skeleton3DList, '/humans/bodies/skel3D', 1)

        self.get_logger().info('State: Inactive.')
        return super().on_configure(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        # Clean up detected bodies
        self.detected_bodies.clear()
        # Clean up ROS interfaces
        self.destroy_ros_interfaces()

        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.image = None
        self.image_msg_header = None
        self.new_image = False
        self.skipped_images = 0
        self.last_id = 0
        self.image_depth = None
        self.depth_msg_header = None
        # Dict used to store the current bodies(@Body) in the image
        self.detected_bodies: OrderedDict[str, Body] = OrderedDict()

        # Multiprocessing YOLO
        self.queue_in = mp.Queue()
        self.queue_out = mp.Queue()
        self.init_detection_process()

        self.first_image = False
        self.offset = None

        self.start_skipping_ts = self.get_clock().now()
        self.detection_start_proc_time = self.get_clock().now()
        self.detection_proc_duration_ms = 0.

        # Subscribers
        self.get_logger().info(f'Depth activated: {self.use_depth}')
        qos_sensor_data = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5
        )
        if self.use_depth:
            self.ats = ApproximateTimeSynchronizer(
                [
                    # Subscribe to image
                    Subscriber(self, ImageCamera, "/image", qos_profile=qos_sensor_data),
                    # Subscribe to body landmarks
                    Subscriber(self, ImageCamera, "/depth/image_raw", qos_profile=qos_sensor_data),
                ],
                # Queue size
                10,
                # Synchronization threshold (seconds)
                self.sync_margin,
                allow_headerless=False
            )
            self.ats.registerCallback(self.img_and_depth_callback)

            self.depth_info_sub = self.create_subscription(
                    CameraInfo, 'depth/camera_info', self.depth_info_callback, qos_sensor_data)
        else:
            self.image_sub = self.create_subscription(ImageCamera, 'image', self.image_callback, qos_sensor_data)

        self.image_info_sub = self.create_subscription(
                    CameraInfo, 'camera_info', self.info_callback, qos_sensor_data)

        # ID manager Client
        self.get_logger().info('Creating ID manager client...')
        self.client_id = self.create_client(PersonID, 'assign_id', callback_group=self.id_client_cbg)
        self.get_logger().info('Waiting for ID manager service...')
        while not self.client_id.wait_for_service(timeout_sec=self.client_id_timeout):
            self.get_logger().warning('ID manager service not available, keep waiting...')
        self.get_logger().info(f"Timeout for ID assignation set to: {self.client_id_timeout} sec.")

        # Main processing loop
        self.proc_timer = self.create_timer(
            1/self.get_parameter('processing_rate').value, self.process_image, callback_group=self.timer_cbg)

        # Diagnosis
        if self.use_diagnosis:
            self.diag_timer = self.create_timer(self.diag_period, self.do_diagnostics)
            self.diag_pub = self.create_publisher(DiagnosticArray, "/diagnostics_pose", 1)

        if self.use_depth:
            self.get_logger().info('Waiting for images to be published on /depth/image_raw and /image.')
        else:
            self.get_logger().info(
                f'Waiting for images to be published on "rgb_camera_topic" and "depth_camera_topic" launch parameters.')

        # Service to provide YOLO reload in GPU
        self.reload_srv = self.create_service(Reload, 'reload_yolo', self.reload_yolo_in_gpu, callback_group=self.yolo_reload_cbg)

        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def init_detection_process(self):
        self.detection_process = mp.Process(target=self.body_detector.run, args=(self.queue_in, self.queue_out))
        self.detection_process.start()

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.id == State.PRIMARY_STATE_ACTIVE:
            self.detected_bodies.clear()
            self.destroy_ros_interfaces()
        self.body_detector = None
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def destroy_ros_interfaces(self):
        """Destroy all ROS interfaces."""
        if self.use_diagnosis:
            self.destroy_timer(self.diag_timer)
            self.destroy_publisher(self.diag_pub)

        self.destroy_timer(self.proc_timer)
        self.destroy_subscription(self.image_sub)
        if self.image_info_sub:
            self.destroy_subscription(self.image_info_sub)
        if self.depth_info_sub:
            self.destroy_subscription(self.depth_info_sub)

        self.destroy_publisher(self.bodies_pub)
        self.destroy_publisher(self.bodies3D_pub)

    def info_callback(self, msg: CameraInfo):
        """Callback to save metadata about the camera. Can be used for calibration"""
        self.camera_info = msg
        self.destroy_subscription(self.image_info_sub)

    def depth_info_callback(self, msg: CameraInfo):
        """Callback to save metadata about the depth camera. Can be used for calibration"""
        self.depth_info = msg
        self.destroy_subscription(self.depth_info_sub)

    def image_callback(self, msg: ImageCamera):
        """
        Callback to save the image and its header.
        It runs a simple logic to check if the processing is too slow.
        """
        with self.image_lock:
            self.image = CvBridge().imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_msg_header = msg.header

            if not self.first_image:
                self.offset = Time().from_msg(msg.header.stamp) if self.use_time_offset else Time()
                self.get_logger().info(f"Offset set to {self.offset}")
                self.first_image = True

            if self.new_image:
                self.skipped_images += 1
                if self.skipped_images > 100:
                    now = self.get_clock().now()
                    skip_time = (now - self.start_skipping_ts).nanoseconds / 1e9
                    self.get_logger().warning(
                        "Pose_detect's processing too slow. "
                        f'Skipped 100 new incoming images over the last {skip_time:.1f}sec')
                    self.start_skipping_ts = now
                    self.skipped_images = 0
            self.new_image = True

    def depth_callback(self, msg: ImageCamera):
        """Callback to save the depth image."""
        with self.image_lock:
            self.set_depth_image(msg)

    def img_and_depth_callback(self, img_msg: ImageCamera, depth_msg: ImageCamera):
        """Callback to save the rgb and depth images, along with their headers."""
        self.image_callback(img_msg)
        self.depth_callback(depth_msg)

    def set_depth_image(self, depth_img):
        if not hasattr(self, 'depth_encoding'):
            self.depth_encoding = depth_img.encoding
            if self.depth_encoding != '32FC1' and self.depth_encoding != '16UC1':
                raise ValueError('Unexpected encoding {}. '.format(self.depth_encoding) +
                                 'Depth encoding should be 16UC1 or `32FC1`.')

        self.image_depth = CvBridge().imgmsg_to_cv2(depth_img, desired_encoding=self.depth_encoding)
        self.depth_msg_header = depth_img.header

    def do_diagnostics(self):
        """Perform diagnostic operations and publish data."""
        now = self.get_clock().now()
        arr = DiagnosticArray()
        arr.header.stamp = now.to_msg()

        msg = DiagnosticStatus(name="Social perception: Body analysis: Skeleton extraction",
                               hardware_id="none")

        current_proc_duration = (now - self.detection_start_proc_time).nanoseconds / 1e9
        if ((current_proc_duration > BODY_DETECTION_PROC_TIME_ERROR) and self.image_lock.locked()):
            msg.level = DiagnosticStatus.ERROR
            msg.message = 'Body detection process not responding'
        elif self.detection_proc_duration_ms > BODY_DETECTION_PROC_TIME_WARN:
            msg.level = DiagnosticStatus.WARN
            msg.message = 'Body detection processing is slow'
        else:
            msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key="Package name", value='hri_pose_detect'),
            KeyValue(key="Currently detected bodies", value=str(len(self.detected_bodies))),
            KeyValue(key="Last detected body ID", value=str(self.last_id)),
            KeyValue(
                key='Detection processing time', value=f'{self.detection_proc_duration_ms:.2f}ms')
        ]

        arr.status = [msg]
        self.diag_pub.publish(arr)

    def generate_temp_id(self):
        """Generate a temporary ID for a face."""
        return 'temp_' + "".join(random.sample("abcdefghijklmnopqrstuvwxyz", 5))

    def request_id(self, bb: BoundingBox, point: list):
        """Request a new ID for a body to the ID manager."""
        id_req = PersonID.Request()
        id_req.xmin, id_req.ymin, id_req.xmax, id_req.ymax = bb.get_norm_coords()
        id_req.xref, id_req.yref = point
        id_req.type = PersonID.Request.BODY
        self.get_logger().debug(f"Requesting ID for body at {bb.get_norm_coords()} and ref_point: {point}")
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

        with self.image_lock:
            self.detection_start_proc_time = self.get_clock().now()
            image = deepcopy(self.image)
            image_msg_header = deepcopy(self.image_msg_header)
            self.new_image = False

            if self.use_depth:
                depth = deepcopy(self.image_depth)
                depth_msg_header = deepcopy(self.depth_msg_header)

        if self.use_depth and self.depth_msg_header is not None:
            if abs(_builtin_time_to_nanosecs(image_msg_header.stamp) - _builtin_time_to_nanosecs(depth_msg_header.stamp)) \
                    > (self.sync_margin * 1e9):
                self.get_logger().error(f"Depth and RGB timestamps are not synchronized: {self.sync_margin} seconds")
                self.proc_lock.release()
                return

        # Copy the list of Bodies ID before iterating over detection, so that we
        # can delete non-existant bodies at the end.
        knownIds = list(self.detected_bodies.keys())

        # Run model inference
        self.queue_in.put(image)
        body_detections = self.queue_out.get()

        # Set of current IDs to avoid duplicates if two detections are too close
        currentIds = set()

        for detection in body_detections:

            # Have we seen this body before? -> check whether or not bounding boxes overlap and the time since last detection
            body = None
            min_dist = float('inf')

            for prev_body in self.detected_bodies.values():
                if bbs_match(prev_body.bb, detection.bb) and time_match(self.get_clock().now(), prev_body.last_detection_time, self.detection_proc_duration_ms, self.offset):
                    dist = distance_rois(prev_body.bb, detection.bb)
                    if dist < min_dist:
                        min_dist = dist
                        body = prev_body
                elif not time_match(self.get_clock().now(), prev_body.last_detection_time, self.detection_proc_duration_ms, self.offset):
                    self.get_logger().warning(f"Body {prev_body.id} does not match in time. Creating new temp_id.")

            if not body:
                body = Body(self, self.generate_temp_id())
                body.initial_detection_time = Time().from_msg(image_msg_header.stamp)
                self.detected_bodies[body.id] = body
                self.last_id = body.id

            # Update the body with its current position and landmarks
            body.score = detection.score
            body.bb = detection.bb
            body.landmarks = detection.landmarks

            body.last_detection_time = Time().from_msg(image_msg_header.stamp)
            body.nb_frames_visible += 1

            # If the body has been detected in MIN_FRAMES_BODY_TRACKING consecutive frames, we assign the final ID.
            if body.nb_frames_visible == MIN_FRAMES_BODY_TRACKING:
                # Ask for definitive ID
                id = self.request_id(body.bb, body.ref_face_point())
                if id is None:
                    self.get_logger().error("Could not get a new ID for the body.")
                    continue
                # Replace the temporary ID with the definitive one
                self.detected_bodies[id] = self.detected_bodies.pop(body.id)
                # In case there were two detections associated to the same body and the new ID is assigned in
                # the second detection, we need to remove the old ID from the set of currentIDs
                currentIds.discard(body.id)
                knownIds.remove(body.id)
                knownIds.append(id)
                body.change_id(id)
                body.set_publish(True)
                self.get_logger().info(f"Started publishing body {body.id}.")

            # Reset the number of frames since last detection
            if body.nb_frames_since_last_detection > 0:
                self.get_logger().debug(f"Body {body.id} detected again.")
                body.nb_frames_since_last_detection = 0
                body.set_publish(True)

            currentIds.add(body.id)

        # Iterate over bodies not seen anymore and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                body = self.detected_bodies[id]
                body.nb_frames_since_last_detection += 1
                body.set_publish(False)
                if body.nb_frames_since_last_detection > MAX_FRAMES_BODY_RETENTION:
                    self.get_logger().debug(f"Deleting body {id}.")
                    del self.detected_bodies[id]

        # Create msg and create new data
        main_msg = Skeleton2DList()
        main_msg.header = image_msg_header
        num_bodies = 0
        ids_print = ''
        pub = False
        if self.use_depth:
            msg_3D = Skeleton3DList()
            msg_3D.header = image_msg_header

        for id in currentIds:
            body = self.detected_bodies[id]
            if body.do_publish:
                pub = True
                # Create ROI and Skeleton submsgs and add them to the main msg
                roi_msg, skeleton_msg = body.create_msgs(image, image_msg_header)
                main_msg.bboxes[num_bodies] = roi_msg
                main_msg.skeletons[num_bodies] = skeleton_msg
                if self.use_depth:
                    msg_3D.skeletons[num_bodies] = \
                        body.create_depth_msg(depth, depth_msg_header, self.depth_encoding, self.camera_info, self.depth_info)
                    main_msg.depths[num_bodies] = \
                        body.extract_body_depth_of_interest(depth, skeleton_msg, self.depth_encoding, self.camera_info, self.depth_info)
                num_bodies += 1
                ids_print += f"[{id}] | "
            if num_bodies >= 10:
                self.get_logger().warn("Too many bodies detected. Only the first 10 will be published.")
                break

        if pub:
            self.bodies_pub.publish(main_msg)
            if self.use_depth:
                self.bodies3D_pub.publish(msg_3D)

        self.detection_proc_duration_ms = (
            self.get_clock().now() - self.detection_start_proc_time).nanoseconds / 1e6
        self.get_logger().debug(f"Publishing: {ids_print}in {self.detection_proc_duration_ms} (ms).")

        self.proc_lock.release()

    def reload_yolo_in_gpu(self, request, response):
        """Callback for the YOLO reload model. This allows to unload the GPU with the model and free its VRAM."""
        if request.load:
            # Load the model
            self.init_detection_process()
            response.state = True
        else:
            # Unload the model
            self.queue_in.put('stop')
            self.detection_process.join()
            response.state = False
        self.get_logger().info(f"YOLO model state after request {response.state}.")
        return response

    def shutdown_detection_process(self):
        """Stop the detection process."""
        try:
            self.queue_in.put('stop')
            time.sleep(1) # Give time to process the stop command and unload de model
            self.queue_out.close()
            self.queue_in.close()
            self.queue_in.cancel_join_thread()
            self.queue_out.cancel_join_thread()
        except:
            pass
        if self.detection_process.is_alive():
            self.detection_process.join(timeout=5)

        if self.detection_process.is_alive():
            self.get_logger().error("Detection process could not be stopped.")
            self.detection_process.terminate()
            self.detection_process.join()

        if self.detection_process.is_alive():
            try:
                import signal
                self.get_logger().error("Sending kill process.")
                os.kill(self.detection_process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.detection_process.join(timeout=2)
        self.detection_process = None


def main(args=None):
    rclpy.init(args=args)

    node = NodePoseDetect()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.shutdown_detection_process()
        node.destroy_node()


if __name__ == "__main__":
    main()
