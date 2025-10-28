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

# Face2DList.msg
# This message contains all info required to pinpoint all faces in a single image frame.
# Faces are limited to a maximum of 10. Any scenario requiring a higher number
# of face detections requires to modify this message and `Skeleton2DList.msg`

std_msgs/Header header                   # Header for meta data

Face2D[10] landmarks                     # Array of landmarks of each Face
                                         # Each element represents a face of the image
NormalizedRegionOfInterest2D[10] bboxes  # Array of ROIs of each Face
