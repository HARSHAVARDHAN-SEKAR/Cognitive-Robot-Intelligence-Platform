"""Bring up the self-learning loop: experience logger + scheduled trainer."""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    home = os.path.expanduser('~')
    log_path = os.path.join(home, '.crip', 'experience.jsonl')
    registry = os.path.join(home, '.crip', 'registry')
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('interval_sec', default_value='3600.0'),
        DeclareLaunchArgument('epochs', default_value='15'),
        Node(package='crip_learning', executable='experience_logger',
             name='experience_logger', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'log_path': log_path}]),
        Node(package='crip_learning', executable='learning_trainer',
             name='learning_trainer', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('use_sim_time'),
                          'log_path': log_path,
                          'registry_dir': registry,
                          'interval_sec': LaunchConfiguration('interval_sec'),
                          'epochs': LaunchConfiguration('epochs'),
                          'run_on_start': False}]),
    ])
