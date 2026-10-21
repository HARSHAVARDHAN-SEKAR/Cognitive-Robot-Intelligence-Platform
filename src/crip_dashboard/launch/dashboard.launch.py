"""Bring up the web dashboard bridge."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('port', default_value='8088'),
        Node(
            package='crip_dashboard',
            executable='dashboard',
            name='dashboard',
            output='screen',
            parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                         'port': LaunchConfiguration('port')}],
        ),
    ])
