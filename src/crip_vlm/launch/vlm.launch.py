"""Bring up the CRIP VLM node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('auto_period', default_value='0.0',
                              description='Seconds between auto analyses; 0 = on-demand only'),
        Node(
            package='crip_vlm',
            executable='vlm',
            name='vlm',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'auto_period': LaunchConfiguration('auto_period')}],
        ),
    ])
