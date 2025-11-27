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
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression

from launch_ros.actions import Node


def generate_launch_description():

    # Publisher launch arguments
    model_arg = DeclareLaunchArgument(
        'model',
        default_value='tts_models/en/ljspeech/tacotron2-DDC',
        description='TTS model to use for text-to-speech synthesis.',
    )
    wait_for_finished_arg = DeclareLaunchArgument(
        'wait_for_finished',
        default_value='True',
        description='Whether the action should wait for the /hri_tts/finished message before completing (wait for audio playback).',
    )
    launch_pub_arg = DeclareLaunchArgument(
        'launch_pub',
        default_value='True',
        description='Whether to launch the TTSPub node to publish audio messages.',
    )
    # Subscriber launch arguments
    launch_sub_arg = DeclareLaunchArgument(
        'launch_sub',
        default_value='False',
        description='Whether to launch the TTSSub node play audio messages.',
    )
    card_number_arg = DeclareLaunchArgument(
        'card_number',
        default_value='-1',
        description='Card number used by aplay to find the sound card device in the system.',
    )
    device_index_arg = DeclareLaunchArgument(
        'device_index',
        default_value='0',
        description='Speaker index used by aplay to find the device in the system if there are multiple.',
    )
    volume_arg = DeclareLaunchArgument(
        'volume',
        default_value='70',
        description='Volume level (%)',
    )
    pub_face_exp_arg = DeclareLaunchArgument(
        'publish_face_expression',
        default_value='True',
        description='Whether to publish face expressions.',
    )

    # Common launch arguments
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level"
    )
    log_level = LaunchConfiguration("log-level")

    tts_pub = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("launch_pub"), "' == 'True'"])),
        package="hri_tts",
        executable="node_tts_pub",
        name='hri_tts_pub',
        parameters=[{'model': LaunchConfiguration('model'),
                     'wait_for_finished': LaunchConfiguration('wait_for_finished')}],
        arguments=['--ros-args', '--log-level', ['hri_tts_pub:=', log_level]],
        output="screen",
    )

    tts_sub = Node(
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration("launch_sub"), "' == 'True'"])),
        package="hri_tts",
        executable="node_tts_sub",
        name='hri_tts_sub',
        parameters=[{'card_number': LaunchConfiguration('card_number'),
                     'device_index': LaunchConfiguration('device_index'),
                     'volume': LaunchConfiguration('volume'),
                     'publish_face_expression': LaunchConfiguration('publish_face_expression')}],
        arguments=['--ros-args', '--log-level', ['hri_tts_sub:=', log_level]],
        output="screen",
    )

    return LaunchDescription([
        model_arg,
        wait_for_finished_arg,
        launch_pub_arg,
        launch_sub_arg,
        card_number_arg,
        device_index_arg,
        volume_arg,
        pub_face_exp_arg,
        log_level_arg,
        tts_pub,
        tts_sub
    ])
