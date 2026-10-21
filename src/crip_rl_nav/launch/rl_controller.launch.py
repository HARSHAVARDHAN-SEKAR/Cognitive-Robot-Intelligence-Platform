"""Bring up the RL local planner inference node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('policy_path', default_value='',
                              description='Absolute path to an ONNX policy; '
                                          'empty uses the heuristic fallback'),
        Node(
            package='crip_rl_nav',
            executable='rl_controller',
            name='rl_controller',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'policy_path': LaunchConfiguration('policy_path'),
            }],
        ),
    ])
