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
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


def generate_launch_description():

    hri_detections_dir = get_package_share_directory('hri_detection_display')

    rviz_config_file = LaunchConfiguration('rviz_config_file')

    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/color/image_raw',
        description='Input sensor_msgs/Image topic to visualize.'
    )
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level"
    )
    log_level = LaunchConfiguration("log-level")
    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value='30',
        description='Best effort frequency for processing and rendering display frames.'
    )
    display_mode_arg = DeclareLaunchArgument(
        'display_mode',
        default_value=['all'],
        description='Display mode to be used.',
        choices=[
            "all",   # Display all detections
            "both",  # Display only persons which have a matching body and face
            "body",  # Display only bodies
            "face",  # Display only faces
        ],
    )
    allow_half_body_arg = DeclareLaunchArgument(
        'allow_half_body',
        default_value='True',
        description='Allow displaying bodies that are not entirely visible. \
                      A body is considered whole if at least the head and one shoulder, hip and knee are visible.'
    )
    allow_back_turned_arg = DeclareLaunchArgument(
        'allow_back_turned',
        default_value='True',
        description='Allow displaying bodies that are not facing the camera.'
    )
    no_signal_timeout_arg = DeclareLaunchArgument(
        'no_signal_timeout',
        default_value='2.0',
        description='Seconds without image frames before rendering "No signal".'
    )
    declare_rviz_config_file = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(hri_detections_dir, 'rviz', 'person_display.rviz'),
        description='Full path to the RVIZ config file to use'
    )
    launch_rviz = DeclareLaunchArgument(
        'rviz',
        default_value='True',
        description='Whether to launch Rviz2 node'
    )

    rviz_node = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("rviz"), "' == 'True'"])),
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    detection_display_node = Node(
        package='hri_detection_display',
        executable='node_person_publisher',
        name='node_person_publisher',
        output='screen',
        parameters=[{'processing_rate': LaunchConfiguration('processing_rate'),
                     'display_mode': LaunchConfiguration('display_mode'),
                     'allow_half_body': LaunchConfiguration('allow_half_body'),
                     'allow_back_turned': LaunchConfiguration('allow_back_turned'),
                     'image_topic': LaunchConfiguration('image_topic'),
                     'no_signal_timeout': LaunchConfiguration('no_signal_timeout')}],
        arguments=['--ros-args', '--log-level', ['node_person_publisher:=', log_level]],
    )

    return LaunchDescription([
        image_topic_arg,
        log_level_arg,
        processing_rate_arg,
        display_mode_arg,
        allow_half_body_arg,
        allow_back_turned_arg,
        no_signal_timeout_arg,
        declare_rviz_config_file,
        launch_rviz,
        rviz_node,
        detection_display_node])
