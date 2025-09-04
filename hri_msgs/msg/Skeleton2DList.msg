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

# Skeleton2DList.msg
# This message includes a list of Skeleton2D msgs and ROIs to represent all the
# bodies of a single image

std_msgs/Header header

Skeleton2D[10] skeletons                 # Array of landmarks of each Skeleton
                                         # Each element represents a face of the image
NormalizedRegionOfInterest2D[10] bboxes  # Array of ROIs of each Skeleton
float32[10] depths                       # [Optional] Average distance of the body to the camera
