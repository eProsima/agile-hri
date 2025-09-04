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


def generate_launch_description():

    hri_stt_dir = get_package_share_directory("hri_stt")

    microphone_arg = DeclareLaunchArgument(
        'microphone',
        default_value=[''],
        description='Microphone name used by PyAudio to find the microphone device in the system.',
    )
    vad_arg = DeclareLaunchArgument(
        'vad',
        default_value=['silero'],
        description='VAD selection. Options: silero, mic.',
        choices=[
            "silero",  # Use Silero VAD
            "mic",     # Use integrated microphone VAD
        ],
    )
    config_file_arg = DeclareLaunchArgument(
        "config_file",
        default_value=os.path.join(hri_stt_dir, 'config', 'noise_config.json'),
        description="Path to the Microphone JSON configuration file",
    )
    whisper_arg = DeclareLaunchArgument(
        'whisper_model',
        default_value=['medium.en'],
        description='Whisper model selection.',
    )
    gpu_load_arg = DeclareLaunchArgument(
        'gpu_load',
        default_value='persist',
        description='If the models should persist in the GPU when they are inactive.',
        choices=[
            "persist",  # Persist in the GPU
            "expire",   # Expire from the GPU. This will require reloading the model
        ],
    )
    max_audio_recording_arg = DeclareLaunchArgument(
        'max_audio_recording',
        default_value=['30'],
        description='Maximum audio recording duration in seconds.',
    )
    pub_face_exp_arg = DeclareLaunchArgument(
        'publish_face_expression',
        default_value=['true'],
        description='Whether to publish face expressions.',
    )
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level"
    )
    log_level = LaunchConfiguration("log-level")

    stt_action_server = Node(
        package="hri_stt",
        executable="node_stt",
        name='hri_stt',
        parameters=[{'config_file': LaunchConfiguration('config_file'),
                     'vad': LaunchConfiguration('vad'),
                     'whisper_model': LaunchConfiguration('whisper_model'),
                     'microphone': LaunchConfiguration('microphone'),
                     'gpu_load': LaunchConfiguration('gpu_load'),
                     'max_audio_recording': LaunchConfiguration('max_audio_recording'),
                     'publish_face_expression': LaunchConfiguration('publish_face_expression')}],
        arguments=['--ros-args', '--log-level', ['hri_stt:=', log_level]],
        output="screen",
    )

    return LaunchDescription([
        config_file_arg,
        vad_arg,
        whisper_arg,
        microphone_arg,
        gpu_load_arg,
        max_audio_recording_arg,
        pub_face_exp_arg,
        log_level_arg,
        stt_action_server])
