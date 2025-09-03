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

import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node

import cv2 as cv


def generate_launch_description():

    hri_emotions_dir = get_package_share_directory("hri_emotion_detect")

    rgb_camera_arg = DeclareLaunchArgument(
        'rgb_camera', default_value='color',
        description='The input camera namespace'
    )
    rgb_camera_topic_arg = DeclareLaunchArgument(
        'rgb_camera_topic', default_value=[LaunchConfiguration('rgb_camera'), '/image_raw'],
        description='The input camera image topic'
    )
    declare_model_expresion_detection = DeclareLaunchArgument(
        "model_expresion_detection",
        default_value=os.path.join(hri_emotions_dir, "models", "facial_expression_recognition_mobilefacenet_2022july.onnx"),
        description="Path to the face detection model"
    )
    declare_backend_id = DeclareLaunchArgument(
        "backend_id",
        default_value=str(cv.dnn.DNN_BACKEND_OPENCV),
        description="Backend computation id"
    )
    declare_target_id = DeclareLaunchArgument(
        "target_id",
        default_value=str(cv.dnn.DNN_TARGET_CPU),
        description="Target computation id"
    )
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level",
    )
    log_level = LaunchConfiguration("log-level")

    emotion_detect_node = Node(
        package="hri_emotion_detect",
        executable="node_emotion_detect",
        name="node_emotion_detect",
        parameters=[
            {"model_expresion_detection": LaunchConfiguration("model_expresion_detection")},
            {"backend_id": LaunchConfiguration("backend_id")},
            {"target_id": LaunchConfiguration("target_id")}
        ],
        arguments=['--ros-args', '--log-level', ['node_emotion_detect:=', log_level]],
        output="screen",
        remappings=[
            ('image', LaunchConfiguration('rgb_camera_topic'))]
    )

    return LaunchDescription([
        rgb_camera_arg,
        rgb_camera_topic_arg,
        declare_model_expresion_detection,
        declare_backend_id,
        declare_target_id,
        log_level_arg,
        emotion_detect_node])
