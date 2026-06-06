#!/usr/bin/env python3
"""Launch the GraspObject action server with grasp_poses.yaml parameters.

Usage:
    ros2 launch jetank_manipulation grasp.launch.py
    ros2 launch jetank_manipulation grasp.launch.py use_sim_time:=true

Requires move_group to be running (launched separately via
jetank_moveit_config/launch/moveit_sim.launch.py or sim_demo.launch.py).
"""

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

import os


def generate_launch_description():
    pkg_share = get_package_share_directory("jetank_manipulation")
    grasp_poses_yaml = os.path.join(pkg_share, "config", "grasp_poses.yaml")

    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="true",
        description="Use simulation (Gazebo) clock. Set to false for real robot.",
    )

    grasp_server_node = Node(
        package="jetank_manipulation",
        executable="grasp_server",
        name="grasp_server",
        output="screen",
        parameters=[
            grasp_poses_yaml,
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    return LaunchDescription([
        use_sim_time_arg,
        grasp_server_node,
    ])
