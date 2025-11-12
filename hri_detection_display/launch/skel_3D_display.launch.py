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

    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level"
    )
    log_level = LaunchConfiguration("log-level")
    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value='30',
        description='Best effort frequency for processing input images.'
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
    visual_style_arg = DeclareLaunchArgument(
        'visual_style',
        default_value=['cylinder'],
        description='Allow changing the visual style of the skeletons.',
        choices=[
            "cylinder",  # Use cylinders to represent the limbs
            "stripes",   # Use stripes to represent the limbs
        ],
    )
    display_hinges_arg = DeclareLaunchArgument(
        'display_hinges',
        default_value='True',
        description='Display hinges for each joint.'
    )
    declare_rviz_config_file = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(hri_detections_dir, 'rviz', 'skel_display.rviz'),
        description='Full path to the RVIZ config file to use'
    )
    declare_pub_static_tf = DeclareLaunchArgument(
        "pub_static_tf",
        default_value="True",
        description="Specifies if an additional static tf should be published to link the camera and the map frame.",
    )

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    detection_display_node = Node(
        package='hri_detection_display',
        executable='node_3D_skel_display',
        name='node_3D_skel_display',
        output='screen',
        parameters=[{'processing_rate': LaunchConfiguration('processing_rate'),
                     'display_hinges': LaunchConfiguration('display_hinges'),
                     'allow_half_body': LaunchConfiguration('allow_half_body'),
                     'allow_back_turned': LaunchConfiguration('allow_back_turned'),
                     'visual_style': LaunchConfiguration('visual_style')}],
        arguments=['--ros-args', '--log-level', ['node_3D_skel_display:=', log_level]],
    )

    # Static transform for link map tf with camera tf. It is only needed if the navigation stack is not used.
    static_tf = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("pub_static_tf"), "' == 'True'"])),
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '1', '0', '0', '-1.5708', 'map', 'camera_depth_optical_frame'],
        output='screen'
    )

    return LaunchDescription([
        log_level_arg,
        processing_rate_arg,
        allow_half_body_arg,
        allow_back_turned_arg,
        visual_style_arg,
        display_hinges_arg,
        declare_rviz_config_file,
        declare_pub_static_tf,
        rviz_node,
        detection_display_node,
        static_tf])
