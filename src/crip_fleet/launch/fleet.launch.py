"""Bring up the fleet-intelligence node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('robot_id', default_value='robot_alpha'),
        Node(package='crip_fleet', executable='fleet', name='fleet',
             output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'robot_id': LaunchConfiguration('robot_id')}]),
    ])
