"""Bring up the trajectory-prediction node."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('model_path', default_value='',
                              description='ONNX forecaster path; empty = constant-velocity'),
        Node(
            package='crip_prediction',
            executable='prediction',
            name='prediction',
            output='screen',
            parameters=[{
                'use_sim_time': LaunchConfiguration('use_sim_time'),
                'model_path': LaunchConfiguration('model_path'),
            }],
        ),
    ])
