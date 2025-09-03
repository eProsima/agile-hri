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
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.events import matches_action
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import LifecycleNode
from launch_ros.events.lifecycle import ChangeState
from launch_ros.event_handlers import OnStateTransition
from lifecycle_msgs.msg import Transition


def generate_launch_description():
    rgb_camera_arg = DeclareLaunchArgument(
        'rgb_camera', default_value='color',
        description='The input camera namespace'
    )
    rgb_camera_topic_arg = DeclareLaunchArgument(
        'rgb_camera_topic', default_value=[LaunchConfiguration('rgb_camera'), '/image_raw'],
        description='The input camera image topic'
    )
    rgb_camera_info_arg = DeclareLaunchArgument(
        'rgb_camera_info', default_value=[LaunchConfiguration('rgb_camera'), '/camera_info'],
        description='The input camera info topic'
    )
    processing_rate_arg = DeclareLaunchArgument(
        'processing_rate',
        default_value='30',
        description='Best effort frequency for processing input images'
    )
    confidence_threshold_arg = DeclareLaunchArgument(
        'confidence_threshold',
        default_value='0.75',
        description='Pose detection confidence threshold'
    )
    image_scale_arg = DeclareLaunchArgument(
        'image_scale',
        default_value='0.25',
        description='Input scale for the image processing pipeline wrt 640x480 pixels'
    )
    use_diag_arg = DeclareLaunchArgument(
        'use_diagnosis',
        default_value='False',
        description='Enable additional topic for diagnosis'
    )
    diagnostic_period_arg = DeclareLaunchArgument(
        'diagnostic_period',
        default_value='5.0',
        description='Diagnostic period'
    )
    face_mesh_arg = DeclareLaunchArgument(
        'face_mesh',
        default_value='False',
        description='Enable face mesh output for near faces'
    )
    id_timeout_arg = DeclareLaunchArgument(
        'id_timeout',
        default_value='7.0',
        description='Timeout in seconds for the ID manager service'
    )
    use_time_offset_arg = DeclareLaunchArgument(
        'use_time_offset',
        default_value=['False'],
        description='Use first image timestamp as offset\
                     to compute time differences'
    )
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level",
    )
    log_level = LaunchConfiguration("log-level")

    face_detect_node = LifecycleNode(
        package='hri_face_detect', executable='face_detect', namespace='', name='hri_face_detect',
        parameters=[{'processing_rate': LaunchConfiguration('processing_rate'),
                     'confidence_threshold': LaunchConfiguration('confidence_threshold'),
                     'image_scale': LaunchConfiguration('image_scale'),
                     'face_mesh': LaunchConfiguration('face_mesh'),
                     'use_diagnosis': LaunchConfiguration('use_diagnosis'),
                     'diagnostic_period': LaunchConfiguration('diagnostic_period'),
                     'id_timeout': LaunchConfiguration('id_timeout'),
                     'use_time_offset': LaunchConfiguration('use_time_offset')}],
        arguments=['--ros-args', '--log-level', ['hri_face_detect:=', log_level]],
        remappings=[
            ('image', LaunchConfiguration('rgb_camera_topic')),
            ('camera_info', LaunchConfiguration('rgb_camera_info'))])

    configure_event = EmitEvent(event=ChangeState(
        lifecycle_node_matcher=matches_action(face_detect_node),
        transition_id=Transition.TRANSITION_CONFIGURE))

    activate_event = RegisterEventHandler(OnStateTransition(
        target_lifecycle_node=face_detect_node, goal_state='inactive',
        entities=[EmitEvent(event=ChangeState(
            lifecycle_node_matcher=matches_action(face_detect_node),
            transition_id=Transition.TRANSITION_ACTIVATE))]))

    return LaunchDescription([
        rgb_camera_arg,
        rgb_camera_topic_arg,
        rgb_camera_info_arg,
        processing_rate_arg,
        confidence_threshold_arg,
        image_scale_arg,
        diagnostic_period_arg,
        face_mesh_arg,
        id_timeout_arg,
        use_time_offset_arg,
        use_diag_arg,
        log_level_arg,
        face_detect_node,
        configure_event,
        activate_event])
