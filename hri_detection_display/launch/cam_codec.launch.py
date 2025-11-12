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
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression, PathJoinSubstitution

from launch_ros.actions import Node


def generate_launch_description():

    hri_display_dir = get_package_share_directory("hri_detection_display")
    usb_cam_dir = get_package_share_directory('usb_cam')

    camera_model = LaunchConfiguration("camera_model")
    astra_params_file = LaunchConfiguration("astra_params_file")
    webcam_params_file = LaunchConfiguration("webcam_params_file")

    declare_camera_model = DeclareLaunchArgument(
        "camera_model",
        default_value="orbbec_astra",
        description="Specifies the camera model to use.",
        choices=[
            "None",
            "webcam",
            "orbbec_astra",
        ],
    )
    color_plugins_arg = DeclareLaunchArgument(
        'color_plugins',
        default_value="['image_transport/compressed', 'image_transport/raw']",
        description='List of plugins for color image transport',
    )
    depth_plugins_arg = DeclareLaunchArgument(
        'depth_plugins',
        default_value="['image_transport/compressedDepth', 'image_transport/raw']",
        description='List of plugins for depth image transport',
    )

    declare_astra_params_file = DeclareLaunchArgument(
        "astra_params_file",
        default_value=os.path.join(hri_display_dir, 'params', 'astra_params.yaml'),
        description="Path to the Astra camera parameters file",
    )
    compression_jpg_arg = DeclareLaunchArgument(
        "compression_jpg",
        default_value='15',
        description="Compression rate for thhe JPG RGB image compression, when using the compressed transport",
    )
    declare_webcam_params_file = DeclareLaunchArgument(
        "webcam_params_file",
        default_value=os.path.join(usb_cam_dir, 'config', 'params_1.yaml'),
        description="Path to the default webcam parameters file",
    )
    xml_file_arg = DeclareLaunchArgument(
        'xml_file',
        default_value=['image.xml'],
        description='The path to the xml file with the Fast DDS profiles'
    )

    webcam_node = Node(
        condition=IfCondition(PythonExpression(["'", camera_model, "' == 'webcam'"])),
        package="usb_cam",
        executable="usb_cam_node_exe",
        output="screen",
        parameters=[
                webcam_params_file,
                {
                    'ffmpeg_image_transport.gop': 15,

                    # Params for CPU
                    # 'ffmpeg_image_transport.encoding': 'libx264',
                    # 'ffmpeg_image_transport.preset': 'ultrafast',
                    # 'ffmpeg_image_transport.tune': 'zerolatency',

                    # Params for Nvidia GPU
                    'ffmpeg_image_transport.encoding': 'hevc_nvenc',
                    'ffmpeg_image_transport.profile': 'main',
                    'ffmpeg_image_transport.preset': 'll',

                    # 'image_raw.enable_pub_plugins': LaunchConfiguration('color_plugins'),
                },
            ],
        remappings=[
            ('/camera_info', '/test_camera_info'),
            ('/image_raw', '/test_image'),
            ('/image_raw/compressed', '/color/image_raw/compressed'),
        ],
    )

    astracamera_node = Node(
        condition=IfCondition(PythonExpression(["'", camera_model, "' == 'orbbec_astra'"])),
        package="astra_camera",
        executable="astra_camera_node",
        parameters=[
                astra_params_file,
                {
                    'ffmpeg_image_transport.gop': 15,
                    # 'ffmpeg_image_transport.bit_rate': 20000,
                    'ffmpeg_image_transport.preset': 'ultrafast',
                    'ffmpeg_image_transport.tune': 'zerolatency',
                    # 'ffmpeg_image_transport.profile': 'main',

                    # Use CPU
                    'ffmpeg_image_transport.encoding': 'libx264',
                    # Use Nvidia GPU
                    # 'ffmpeg_image_transport.encoding': 'hevc_nvenc',

                    'color.image_raw.format': 'jpeg',
                    'color.image_raw.jpeg_quality': LaunchConfiguration('compression_jpg'),
                    'color.image_raw.png_level': 1,

                    # Enable raw images
                    'color.image_raw.enable_pub_plugins': LaunchConfiguration('color_plugins'),
                    'depth.image_raw.enable_pub_plugins': LaunchConfiguration('depth_plugins'),
                },
            ],
        output="screen",
    )

    xml_file_value = PathJoinSubstitution([hri_display_dir, 'config', LaunchConfiguration('xml_file')])
    log_info = LogInfo(msg=['The XML file loaded is: ', xml_file_value])

    xml_env_var = SetEnvironmentVariable(name='FASTDDS_DEFAULT_PROFILES_FILE', value=xml_file_value)

    return LaunchDescription([
        xml_file_arg,
        xml_env_var,
        declare_camera_model,
        color_plugins_arg,
        depth_plugins_arg,
        declare_astra_params_file,
        compression_jpg_arg,
        declare_webcam_params_file,
        webcam_node,
        astracamera_node,
        log_info])
