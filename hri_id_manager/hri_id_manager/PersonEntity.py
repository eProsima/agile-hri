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


class PersonEntity:
    """
    Class containing the meta data of a person entity.
    """
    def __init__(self):
        # Whether the person will be displayed or not
        self.online = False
        # Frames since last detection
        self.body_frames_since_last_detection = -1
        self.face_frames_since_last_detection = -1
        # Boundary boxes
        self.body_position = [0, 0, 0, 0]  # xmin ymin xmax ymax
        self.face_position = [0, 0, 0, 0]  # xmin ymin xmax ymax
        # Skeleton
        self.skeleton = [None] * (18)
        # Reference point for face position calculated from body landmarks
        self.face_from_body = [0, 0]
        # Face landmarks obtained from the face NN
        self.face_landmarks = [None] * (5)
        # Times with the last detection of each field
        self.times = {"body": initial_time, "face": initial_time}
        # Both body and face are identified and recognized as the same person
        self.matched = False
