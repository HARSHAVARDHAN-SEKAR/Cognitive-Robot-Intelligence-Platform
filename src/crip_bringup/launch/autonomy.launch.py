"""
Full autonomy demo: simulation + perception + reactive safety pilot.

The robot will wander the arena, stopping and turning away from walls and
obstacles, narrating its decisions on /brain/explanation.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    bringup_pkg = get_package_share_directory('crip_bringup')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'simulation.launch.py')),
        launch_arguments={'rviz': LaunchConfiguration('rviz')}.items(),
    )

    perception = Node(
        package='crip_perception',
        executable='obstacle_monitor',
        output='screen',
        parameters=[{'use_sim_time': True, 'warn_distance': 0.8}],
    )

    # BUGFIX (robot escaped the arena, x=-12 when walls are at +/-5):
    # safety_pilot only stops for obstacles it SEES ahead. Nothing enforced the
    # arena bounds, so it drove straight through a wall and cruised forever.
    #
    # Fix: safety_pilot now publishes to /cmd_vel_raw, and the REAL safety
    # supervisor (crip_safety, M11) gates it -> /cmd_vel. Its geofence rule
    # vetoes any forward motion that would leave the arena, its collision rule
    # stops before obstacles, and a watchdog stops the robot if commands stall.
    brain = Node(
        package='crip_brain',
        executable='safety_pilot',
        name='safety_pilot',
        output='screen',
        parameters=[{'use_sim_time': True, 'stop_distance': 1.1,
                     'cruise_speed': 0.22, 'turn_speed': 0.9}],
        remappings=[('cmd_vel', 'cmd_vel_raw')],
    )

    # The veto layer: nothing reaches the wheels without passing this.
    safety_supervisor = Node(
        package='crip_safety',
        executable='safety_supervisor',
        name='safety_supervisor',
        output='screen',
        parameters=[{'use_sim_time': True,
                     'max_v': 0.26,
                     'stop_distance': 0.4,
                     'arena_half': 7.5}],
    )

    world_model = Node(
        package='crip_world_model',
        executable='world_model',
        name='world_model',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    llm_brain = Node(
        package='crip_llm_brain',
        executable='llm_brain',
        name='llm_brain',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    vlm = Node(
        package='crip_vlm',
        executable='vlm',
        name='vlm',
        output='screen',
        parameters=[{'use_sim_time': True, 'auto_period': 0.0}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        sim,
        perception,
        brain,
        safety_supervisor,
        world_model,
        llm_brain,
        vlm,
    ])
