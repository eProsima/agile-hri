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


from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

from launch_ros.actions import Node


def generate_launch_description():

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
    image_topic_arg = DeclareLaunchArgument(
        'image_topic',
        default_value='/color/image_raw',
        description='Input sensor_msgs/Image topic to visualize.'
    )
    window_name_arg = DeclareLaunchArgument(
        'window_name',
        default_value='HRI Person Viewer',
        description='OpenCV window title. Cannot be changed at runtime.'
    )
    window_x_arg = DeclareLaunchArgument(
        'window_x',
        default_value='400',
        description='Initial window X position in pixels. Can be changed at runtime.'
    )
    window_y_arg = DeclareLaunchArgument(
        'window_y',
        default_value='400',
        description='Initial window Y position in pixels. Can be changed at runtime.'
    )
    window_width_arg = DeclareLaunchArgument(
        'window_width',
        default_value='1280',
        description='Initial window width in pixels. Can be changed at runtime.'
    )
    window_height_arg = DeclareLaunchArgument(
        'window_height',
        default_value='720',
        description='Initial window height in pixels. Can be changed at runtime.'
    )
    window_move_step_arg = DeclareLaunchArgument(
        'window_move_step',
        default_value='50',
        description='Keyboard move step in pixels for WASD controls.'
    )
    always_on_top_arg = DeclareLaunchArgument(
        'always_on_top',
        default_value='False',
        description='Best-effort request for top-most window behavior (X11 backend dependent).'
    )
    bring_to_front_arg = DeclareLaunchArgument(
        'bring_to_front',
        default_value='True',
        description='Best-effort request to focus the viewer window on startup.'
    )
    keep_aspect_ratio_arg = DeclareLaunchArgument(
        'keep_aspect_ratio',
        default_value='True',
        description='Preserve image aspect ratio while resizing the window.'
    )
    no_signal_timeout_arg = DeclareLaunchArgument(
        'no_signal_timeout',
        default_value='2.0',
        description='Seconds without image frames before rendering "No signal".'
    )

    detection_display_node = Node(
        package='hri_detection_display',
        executable='node_person_display',
        name='node_person_display',
        output='screen',
        parameters=[{'processing_rate': LaunchConfiguration('processing_rate'),
                     'display_mode': LaunchConfiguration('display_mode'),
                     'allow_half_body': LaunchConfiguration('allow_half_body'),
                     'allow_back_turned': LaunchConfiguration('allow_back_turned'),
                     'image_topic': LaunchConfiguration('image_topic'),
                     'window_name': LaunchConfiguration('window_name'),
                     'window_x': LaunchConfiguration('window_x'),
                     'window_y': LaunchConfiguration('window_y'),
                     'window_width': LaunchConfiguration('window_width'),
                     'window_height': LaunchConfiguration('window_height'),
                     'window_move_step': LaunchConfiguration('window_move_step'),
                     'always_on_top': LaunchConfiguration('always_on_top'),
                     'bring_to_front': LaunchConfiguration('bring_to_front'),
                     'keep_aspect_ratio': LaunchConfiguration('keep_aspect_ratio'),
                     'no_signal_timeout': LaunchConfiguration('no_signal_timeout')}],
        arguments=['--ros-args', '--log-level', ['node_person_display:=', log_level]],
    )

    return LaunchDescription([
        log_level_arg,
        processing_rate_arg,
        display_mode_arg,
        allow_half_body_arg,
        allow_back_turned_arg,
        image_topic_arg,
        window_name_arg,
        window_x_arg,
        window_y_arg,
        window_width_arg,
        window_height_arg,
        window_move_step_arg,
        always_on_top_arg,
        bring_to_front_arg,
        keep_aspect_ratio_arg,
        no_signal_timeout_arg,
        detection_display_node])
