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

    # -- Launch arguments --

    rgb_camera_arg = DeclareLaunchArgument(
        'rgb_camera',
        default_value='color',
        description='The input rgb camera namespace'
    )
    rgb_camera_topic_arg = DeclareLaunchArgument(
        'rgb_camera_topic',
        default_value=[LaunchConfiguration('rgb_camera'), '/image_raw'],
        description='The input rgb camera image topic'
    )
    camera_info_topic_arg = DeclareLaunchArgument(
        'camera_info_topic',
        default_value=[LaunchConfiguration('rgb_camera'), '/camera_info'],
        description='CameraInfo topic for 3D-to-pixel projection'
    )
    skel3d_topic_arg = DeclareLaunchArgument(
        'skel3d_topic',
        default_value='/humans/bodies/skel3D',
        description='Input Skeleton3DList topic'
    )
    marker_topic_arg = DeclareLaunchArgument(
        'marker_topic',
        default_value='/humans/detection/skel3D',
        description='Output MarkerArray topic for 3D skeleton visualization'
    )
    log_level_arg = DeclareLaunchArgument(
        'log-level',
        default_value=['info'],
        description='Logging level'
    )
    log_level = LaunchConfiguration('log-level')
    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value='30',
        description='Best effort frequency for processing and rendering display frames.'
    )
    display_mode_arg = DeclareLaunchArgument(
        'display_mode',
        default_value=['all'],
        description='Display mode to be used.',
        choices=['all', 'both', 'body', 'face'],
    )
    allow_half_body_arg = DeclareLaunchArgument(
        'allow_half_body',
        default_value='True',
        description='Allow displaying bodies that are not entirely visible. '
                    'A body is considered whole if at least the head and one shoulder, hip and knee are visible.'
    )
    allow_back_turned_arg = DeclareLaunchArgument(
        'allow_back_turned',
        default_value='True',
        description='Allow displaying bodies that are not facing the camera.'
    )
    display_hinges_arg = DeclareLaunchArgument(
        'display_hinges',
        default_value='True',
        description='Display joint hinge spheres in 3D markers.'
    )
    visual_style_arg = DeclareLaunchArgument(
        'visual_style',
        default_value=['cylinder'],
        description='3D visual style of the skeletons.',
        choices=['cylinder', 'stripes'],
    )
    declare_rviz_config_file = DeclareLaunchArgument(
        'rviz_config_file',
        default_value=os.path.join(hri_detections_dir, 'rviz', 'node_viewer.rviz'),
        description='Full path to the RVIZ config file to use'
    )
    launch_rviz_arg = DeclareLaunchArgument(
        'rviz',
        default_value='True',
        description='Whether to launch Rviz2 node'
    )
    declare_pub_static_tf = DeclareLaunchArgument(
        'pub_static_tf',
        default_value='True',
        description='Publish a static tf to link the map and camera_depth_optical_frame frames. '
                    'Only needed if the navigation stack is not used.',
    )

    # -- Nodes --

    viewer_node = Node(
        package='hri_detection_display',
        executable='node_3D_person_viewer',
        name='node_3D_person_viewer',
        output='screen',
        parameters=[{
            'processing_rate': LaunchConfiguration('processing_rate'),
            'image_topic': LaunchConfiguration('rgb_camera_topic'),
            'camera_info_topic': LaunchConfiguration('camera_info_topic'),
            'skel3d_topic': LaunchConfiguration('skel3d_topic'),
            'marker_topic': LaunchConfiguration('marker_topic'),
            'display_mode': LaunchConfiguration('display_mode'),
            'allow_half_body': LaunchConfiguration('allow_half_body'),
            'allow_back_turned': LaunchConfiguration('allow_back_turned'),
            'display_hinges': LaunchConfiguration('display_hinges'),
            'visual_style': LaunchConfiguration('visual_style'),
        }],
        arguments=['--ros-args', '--log-level', ['node_3D_person_viewer:=', log_level]],
    )

    rviz_node = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('rviz'), "' == 'True'"])),
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config_file],
        output='screen'
    )

    # Static transform to link map frame with camera frame.
    # Only needed if the navigation stack is not used.
    static_tf = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('pub_static_tf'), "' == 'True'"])),
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['0', '0', '1', '0', '0', '-1.5708', 'map', 'camera_depth_optical_frame'],
        output='screen'
    )

    return LaunchDescription([
        # Arguments
        rgb_camera_arg,
        rgb_camera_topic_arg,
        camera_info_topic_arg,
        skel3d_topic_arg,
        marker_topic_arg,
        log_level_arg,
        processing_rate_arg,
        display_mode_arg,
        allow_half_body_arg,
        allow_back_turned_arg,
        display_hinges_arg,
        visual_style_arg,
        declare_rviz_config_file,
        launch_rviz_arg,
        declare_pub_static_tf,
        # Nodes
        viewer_node,
        rviz_node,
        static_tf,
    ])
