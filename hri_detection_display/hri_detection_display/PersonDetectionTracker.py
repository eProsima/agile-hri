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

from rclpy.time import Time

initial_time = Time(seconds=0, nanoseconds=0)


class PersonDetection:
    """
    Class containing the meta data of a person detection
    """
    def __init__(self):
        # Whether the person should be displayed or not
        self.online = False
        # Frames since last detection
        self.frames_since_last_detection = 0
        # Boundary boxes
        self.body_position = [0, 0, 0, 0] # xmin ymin xmax ymax
        self.face_position = [0, 0, 0, 0] # xmin ymin xmax ymax
        # Emotion
        self.emotion = ""
        # Scores
        self.body_score = 0.0
        self.face_score = 0.0
        # Skeleton
        self.landmarks = [None] * (18)
        self.face_landmarks = [None] * (5)
        # Whether left hand is raised or not
        self.hand_raised = False
        # Whether a full view of the body is available. That is, at least the head and one shoulder, hip and knee are visible.
        self.whole_body = False
        # Whether the person is facing the camera or not. >= 4 means facing the camera. < 4 means not facing the camera.
        self.facing = 0
        # Times with the last detection of each field
        self.times = {"body": initial_time, "face": initial_time, "emotion": initial_time, "voice": initial_time}
        # Both body and face are identified and recognized as the same person
        self.matched = False
        # Depth
        self.depth = 0.0
        # Orientation (only used by 3D visualizator and goal generator)
        self.theta = -1.57
        # Shoulders and hips distance
        self.shoulders_distance = 0.0
        self.hips_distance = 0.0
        # Frames with hand raised (selectable to be tracked)
        self.frames_for_selection = 0
