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
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    deterministic_ids_arg = DeclareLaunchArgument(
        'deterministic_ids',
        default_value='False',
        description='Enable use of non-random increasing body IDs'
    )
    log_level_arg = DeclareLaunchArgument(
        "log-level",
        default_value=["info"],
        description="Logging level",
    )
    log_level = LaunchConfiguration("log-level")

    detection_display_node = Node(
        package='hri_id_manager',
        executable='node_id_manager',
        name='node_id_manager',
        output='screen',
        parameters=[{'deterministic_ids': LaunchConfiguration('deterministic_ids')}],
        arguments=['--ros-args', '--log-level', ['node_id_manager:=', log_level]],
    )

    return LaunchDescription([
        deterministic_ids_arg,
        log_level_arg,
        detection_display_node])
