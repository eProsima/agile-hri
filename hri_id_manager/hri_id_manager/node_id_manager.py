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
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.node import Node

from hri_msgs.msg import Skeleton2DList, Skeleton2D, Face2DList
from hri_msgs.srv import PersonID
from rcl_interfaces.msg import ParameterDescriptor

from threading import Lock
import math
import numpy as np
import random

from hri_id_manager.PersonEntity import PersonEntity

# Max number of frames a face/body is not detected before it is not tracked anymore
MAX_FRAMES_RETENTION = 5
# Time margin to consider a face/body obsolete and to avoid use it to match
TIME_MARGIN_DETECTION = 0.2
# Relative distance between face and body to consider them as the same person
MAX_FACE_TO_BODY_REL_DIST = 0.15
# Minimum % of overlapping between face and body to consider them as the same person, relative to face ROI
MIN_OVERLAPPING_REQUIRED = 0.8

_face_landmarks = [
        Skeleton2D.NOSE,
        Skeleton2D.LEFT_EYE,
        Skeleton2D.RIGHT_EYE,
        Skeleton2D.LEFT_EAR,
        Skeleton2D.RIGHT_EAR,
]


class IDManager(Node):
    """
    ROS 2 Node managing the runtime ID assignation for both face and body detection.
    """
    last_id = 0

    def __init__(self):
        # Initialize node
        super().__init__("hri_id_manager")

        self.declare_parameter(
            'deterministic_ids', False, ParameterDescriptor(
                description='Enable use of non-random increasing body IDs'))

        # Initialize variables
        self.deterministic_ids = self.get_parameter('deterministic_ids').value

        self.dict_lock = Lock()
        self.persons_ = {}

        # Subscribe to bodies detections
        self.bodies_sub = self.create_subscription(Skeleton2DList, '/humans/bodies', self.bodies_callback, 1)
        # Subscribe to faces detections
        self.faces_sub = self.create_subscription(Face2DList, '/humans/faces', self.faces_callback, 1)

        # Service to provide IDs
        self.srv = self.create_service(PersonID, 'assign_id', self.assign_id_callback)

    def assign_id_callback(self, request, response):
        """Callback for the ID assignment service."""
        # Normalized coords for the img, as we compare to the face and body detections which use NormalizedRegionOfInterest
        roi = [request.xmin, request.ymin, request.xmax, request.ymax]
        ref_point = [request.xref, request.yref]
        self.get_logger().debug(f"Received request [{request.type}]. ROI: {roi}. Ref: {ref_point}.")
        if request.type == PersonID.Request.BODY:
            response.id = self.check_existing_face(roi, ref_point)
        elif request.type == PersonID.Request.FACE:
            response.id = self.check_existing_body(roi, ref_point)
        else:
            self.get_logger().error(f"Unknown type {request.type}.")
            return response

        with self.dict_lock:
            if response.id not in self.persons_:
                self.persons_[response.id] = PersonEntity()
                if request.type == PersonID.Request.BODY:
                    self.persons_[response.id].body_position = roi
                    self.persons_[response.id].face_from_body = ref_point
                    self.persons_[response.id].times["body"] = self.get_clock().now()
                elif request.type == PersonID.Request.FACE:
                    self.persons_[response.id].face_position = roi
                    # No need to save face_from_body because it is calculated from body landmarks
                    self.persons_[response.id].times["face"] = self.get_clock().now()
            else:
                # If the ID was already known, it means that the face/body was matched to another one
                self.persons_[response.id].matched = True

        self.get_logger().info(f"Assigned ID {response.id} to {request.type}.")
        return response

    def check_existing_face(self, roi, ref_point) -> str:
        """
        Check if there is a face without a body associated to it that could match to this body.
        roi: [xmin, ymin, xmax, ymax] of the body to be matched.
        ref_point: [x, y] of the center point of the face (calculated with YOLO) inside the body to be matched.
        """
        candidates = []
        self.get_logger().debug(f"Persons already registered prior check_face {(self.persons_.keys())}.")
        for id, person in self.persons_.items():
            if person.matched:
                # Skip already matched faces
                self.get_logger().debug(f"Face [{id}] already matched.")
                continue
            face = person.face_position
            if face != [0, 0, 0, 0]:
                time_check = self.get_clock().now().nanoseconds
                if time_check - person.times["face"].nanoseconds < TIME_MARGIN_DETECTION * 1e9:
                    mid_face_bbox = [(face[0] + face[2]) / 2, (face[1] + face[3]) / 2]
                    self.get_logger().debug(f"Distance between {ref_point} and {mid_face_bbox}")
                    dist = math.dist((ref_point[0], ref_point[1]), (mid_face_bbox[0], mid_face_bbox[1]))
                    body_roi_diag = math.dist((roi[0], roi[1]), (roi[2], roi[3]))
                    if dist / body_roi_diag < MAX_FACE_TO_BODY_REL_DIST:
                        self.get_logger().debug(f"Face [{id}] has a valid dist: {dist}/{body_roi_diag}={dist/body_roi_diag}!")
                        overlap = self.calculate_overlapping(face_roi=face, body_roi=roi)
                        if overlap > MIN_OVERLAPPING_REQUIRED:
                            self.get_logger().debug(f"Face [{id}] is a candidate with overlap: {overlap}")
                            candidates.append(id)
                        else:
                            self.get_logger().debug(f"Face [{id}] does NOT overlap enough: {overlap}")
                    else:
                        self.get_logger().debug(f"Face [{id}] is too far from the body. {dist}/{body_roi_diag}={dist/body_roi_diag}.")
                else:
                    subs = time_check - person.times["face"].nanoseconds
                    self.get_logger().debug(f"Face [{id}] is too old: {time_check} - {subs}.")
            else:
                self.get_logger().debug(f"Face of [{id}] not detected.")

        if len(candidates) == 1:
            self.get_logger().debug(f"Face [{candidates[0]}] is the only match.")
            return candidates[0]
        elif len(candidates) > 1:
            # If there are multiple candidates, select the closest one (distance is more relevant than overlapping)
            min_dist = np.inf
            closest_id = None
            for id in candidates:
                face = self.persons_[id].face_position
                mid_face_bbox = [(face[0] + face[2]) / 2, (face[1] + face[3]) / 2]
                dist = math.dist((mid_face_bbox[0], mid_face_bbox[1]), (ref_point[0], ref_point[1]))
                if dist < min_dist:
                    min_dist = dist
                    closest_id = id
            self.get_logger().debug(f"Face [{closest_id}] is the better match out of: {candidates}.")
            return closest_id

        return self.generate_id()

    def check_existing_body(self, roi, ref_point) -> str:
        """
        Check if there is a body without a face associated to it that could match to this face.
        roi: [xmin, ymin, xmax, ymax] of the face to be matched.
        ref_point: [x, y] of the reference point of the face (center of the bounding box)
        """
        candidates = []
        self.get_logger().debug(f"Persons already registered prior check_body {(self.persons_.keys())}.")
        for id, person in self.persons_.items():
            if person.matched:
                # Skip already matched faces
                self.get_logger().debug(f"Body [{id}] already matched.")
                continue
            body = person.body_position
            if body != [0, 0, 0, 0]:
                time_check = self.get_clock().now().nanoseconds
                if time_check - person.times["body"].nanoseconds < TIME_MARGIN_DETECTION * 1e9:
                    body_face_point = person.face_from_body
                    self.get_logger().debug(f"Distance between {ref_point} and {body_face_point}")
                    dist = math.dist((ref_point[0], ref_point[1]), (body_face_point[0], body_face_point[1]))
                    body_roi_diag = math.dist((body[0], body[1]), (body[2], body[3]))
                    if dist / body_roi_diag < MAX_FACE_TO_BODY_REL_DIST:
                        self.get_logger().debug(f"Body [{id}] has a valid dist: {dist}/{body_roi_diag}={dist/body_roi_diag}!")
                        overlap = self.calculate_overlapping(face_roi=roi, body_roi=body)
                        if overlap > MIN_OVERLAPPING_REQUIRED:
                            self.get_logger().debug(f"Body [{id}] is a candidate with overlap: {overlap}")
                            candidates.append(id)
                        else:
                            self.get_logger().debug(f"Body [{id}] does NOT overlap enough: {overlap}")
                    else:
                        self.get_logger().debug(f"Body [{id}] is too far from the body. {dist}/{body_roi_diag}={dist/body_roi_diag}.")
                else:
                    subs = time_check - person.times["body"].nanoseconds
                    self.get_logger().debug(f"Body [{id}] is too old: {time_check} - {subs}.")
            else:
                self.get_logger().debug(f"Body of [{id}] not detected.")

        if len(candidates) == 1:
            self.get_logger().debug(f"Body [{candidates[0]}] is the only match.")
            return candidates[0]
        elif len(candidates) > 1:
            # If there are multiple candidates, select the closest one (distance is more relevant than overlapping)
            min_dist = np.inf
            closest_id = None
            for id in candidates:
                body_face_point = self.persons_[id].face_from_body
                self.get_logger().debug(f"Distance between {ref_point} and {body_face_point}")
                dist = math.dist((ref_point[0], ref_point[1]), (body_face_point[0], body_face_point[1]))
                if dist < min_dist:
                    min_dist = dist
                    closest_id = id
            self.get_logger().debug(f"Body [{closest_id}] is the better match out of: {candidates}.")
            return closest_id

        return self.generate_id()

    def generate_id(self) -> str:
        """Generate a new ID for a person."""
        if self.deterministic_ids:
            id = 'f%05d' % IDManager.last_id
            IDManager.last_id = (IDManager.last_id + 1) % 10000
        else:
            id = "".join(random.sample("abcdefghijklmnopqrstuvwxyz", 5))
        self.get_logger().debug(f"Generated new ID: [{id}].")
        return id

    def calculate_overlapping(self, face_roi, body_roi) -> float:
        """Calculate the overlapping between two regions of interest, relative to @face_roi."""
        xA = max(face_roi[0], body_roi[0])
        yA = max(face_roi[1], body_roi[1])
        xB = min(face_roi[2], body_roi[2])
        yB = min(face_roi[3], body_roi[3])

        # Compute the area of intersection rectangle
        boxAArea = (face_roi[2] - face_roi[0]) * (face_roi[3] - face_roi[1])
        if boxAArea <= 0:
            self.get_logger().warn(f"Face ROI area is zero or negative ({boxAArea}); returning 0 overlap. face_roi={face_roi}")
            return 0.0
        interArea = max(0, xB - xA) * max(0, yB - yA)

        return interArea / boxAArea

    def bodies_callback(self, msg: Skeleton2DList):
        """Callback for body detection."""
        # Copy the list of Bodies ID before iterating over detection, so that we
        # can delete non-existant bodies at the end.
        knownIds = list(self.persons_.keys())
        currentIds = []

        for roi_msg, ske_msg in zip(msg.bboxes, msg.skeletons):
            if roi_msg.key != ske_msg.key:
                self.get_logger().error(f"Body id mismatch: [{roi_msg.key}] != [{ske_msg.key}]")
                continue
            elif roi_msg.key == "":
                continue
            key = roi_msg.key

            # Check ID correctness
            if key not in self.persons_:
                self.get_logger().error(f"Body ID {key} not registered with this IDManager.")
                return

            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            self.update_position(key, position, body=True)
            self.update_landmarks(key, ske_msg, body=True, body_roi=position)
            self.update_times(key, body=True)

            currentIds.append(key)
            # Reset the number of frames if it was lost or if it was not detected before
            frames_count = self.persons_[key].body_frames_since_last_detection
            if frames_count > 0 or \
                    frames_count == -1:
                if frames_count == -1:
                    self.get_logger().debug(f"Body {key} detected for the first time: {frames_count}.")
                else:
                    self.get_logger().debug(f"Body {key} detected again: {frames_count}.")
                self.persons_[key].body_frames_since_last_detection = 0
                self.persons_[key].online = True
                # Match again if there is a face in the same key that was detected
                if self.persons_[key].face_frames_since_last_detection == 0:
                    self.persons_[key].matched = True

        # Iterate over bodies not seen anymore and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                if self.persons_[id].body_frames_since_last_detection != -1:
                    # Only add missing frames if it was detected before
                    self.persons_[id].body_frames_since_last_detection += 1
                self.persons_[id].online = False
                if self.persons_[id].matched:
                    self.get_logger().info(f"Body [{id}] unmatched.")
                self.persons_[id].matched = False
                if self.persons_[id].body_frames_since_last_detection > MAX_FRAMES_RETENTION and \
                        (self.persons_[id].face_frames_since_last_detection > MAX_FRAMES_RETENTION or
                            self.persons_[id].face_frames_since_last_detection == -1):
                    # Delete the person if it has not been detected for a long time and (no face was detected or
                    # if the face has not been detected for a long time)
                    with self.dict_lock:
                        self.get_logger().info(f"Deleting Body [{id}].")
                        del self.persons_[id]

    def faces_callback(self, msg: Face2DList):
        """Callback for face detection."""
        # Copy the list of Bodies ID before iterating over detection, so that we
        # can delete non-existant bodies at the end.
        knownIds = list(self.persons_.keys())
        currentIds = []

        for roi_msg, face_msg in zip(msg.bboxes, msg.landmarks):
            if roi_msg.key != face_msg.key:
                self.get_logger().error(f"Face id mismatch: [{roi_msg.key}] != [{face_msg.key}]")
                continue
            elif roi_msg.key == "":
                continue
            key = roi_msg.key

            # Check ID correctness
            if key not in self.persons_:
                self.get_logger().error(f"Face ID {key} not registered with this IDManager.")
                return

            position = [roi_msg.xmin, roi_msg.ymin, roi_msg.xmax, roi_msg.ymax]
            self.update_position(key, position, face=True)
            self.update_landmarks(key, face_msg, face=True)
            self.update_times(key, face=True)

            currentIds.append(key)
            # Reset the number of frames if it was lost or if it was not detected before
            frames_count = self.persons_[key].face_frames_since_last_detection
            if frames_count > 0 or \
                    frames_count == -1:
                if frames_count == -1:
                    self.get_logger().debug(f"Face {key} detected for the first time: {frames_count}.")
                else:
                    self.get_logger().debug(f"Face {key} detected again: {frames_count}.")
                self.persons_[key].face_frames_since_last_detection = 0
                self.persons_[key].online = True
                # Match again if there is a body in the same key that was detected
                if self.persons_[key].body_frames_since_last_detection == 0:
                    self.persons_[key].matched = True

        # Iterate over faces not seen anymore and unregister corresponding publishers
        for id in knownIds:
            if id not in currentIds:
                if self.persons_[id].face_frames_since_last_detection != -1:
                    # Only add missing frames if it was detected before
                    self.persons_[id].face_frames_since_last_detection += 1
                self.persons_[id].online = False
                if self.persons_[id].matched:
                    self.get_logger().debug(f"Face [{id}] unmatched.")
                self.persons_[id].matched = False
                if self.persons_[id].face_frames_since_last_detection > MAX_FRAMES_RETENTION and \
                        (self.persons_[id].body_frames_since_last_detection > MAX_FRAMES_RETENTION or
                            self.persons_[id].body_frames_since_last_detection == -1):
                    # Delete the person if it has not been detected for a long time and (no face was detected or
                    # if the face has not been detected for a long time)
                    with self.dict_lock:
                        self.get_logger().debug(f"Deleting Face [{id}].")
                        del self.persons_[id]

    def update_position(self, id, position, body=False, face=False):
        """Update the position of the given body/face."""
        if body:
            self.persons_[id].body_position = position
        elif face:
            self.persons_[id].face_position = position

    def update_landmarks(self, id, msg, body=False, face=False, body_roi=None):
        """Update the landmarks of the given body/face."""
        if body and face:
            self.get_logger().error("Cannot update both body and face landmarks at the same time.")
            return
        elif body:
            self.persons_[id].skeleton = msg.skeleton
            self.persons_[id].face_from_body = self.calculate_face_from_body(msg.skeleton, body_roi)
        elif face:
            self.persons_[id].face_landmarks = msg.landmarks

    def update_times(self, id, body=False, face=False):
        """Update the detection times of the given body/face."""
        if body:
            self.persons_[id].times["body"] = self.get_clock().now()
        if face:
            self.persons_[id].times["face"] = self.get_clock().now()

    def calculate_face_from_body(self, skeleton, roi):
        """Return the reference face point computed from body landmarks (Skeleton2D)."""
        tot_x, tot_y, tot_points = 0, 0, 0
        for point in _face_landmarks:
            if skeleton[point] is not None and skeleton[point].x != 0 and skeleton[point].y != 0:
                tot_x += skeleton[point].x
                tot_y += skeleton[point].y
                tot_points += 1

        if tot_points != 0:
            ref_point = [(tot_x / tot_points), (tot_y / tot_points)]
        else:
            # If there are no face landmarks in Skeleton, we return an estimate of the face position based on body ROI.
            ref_point = [(roi[0] + roi[2]) / 2, roi[1] * 0.75 + roi[3] * 0.25]

        return ref_point


def main(args=None):
    rclpy.init(args=args)

    node = IDManager()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == "__main__":
    main()
