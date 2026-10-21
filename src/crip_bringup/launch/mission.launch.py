"""
Full CRIP mission stack (Milestone 5+).

Brings up everything needed to take a natural-language mission and physically
execute it:
    simulation (Gazebo + bridge + RViz)
    perception (obstacle monitor / detections)
    world model (M2)
    LLM brain (M3)
    VLM (M4)
    Nav2 (M5)  -- owns /cmd_vel

Note: unlike autonomy.launch.py this does NOT start the reactive safety_pilot,
because Nav2 is the velocity source here. Use autonomy.launch.py for the
lightweight reactive demo without Nav2.
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
    nav_pkg = get_package_share_directory('crip_navigation')

    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(bringup_pkg, 'launch', 'simulation.launch.py')),
        launch_arguments={'rviz': LaunchConfiguration('rviz')}.items(),
    )

    # Nav2 publishes velocity; remap it to /cmd_vel_raw so the safety
    # supervisor (M11) gates it before it reaches the robot's /cmd_vel.
    # The CRIP-specific Nav2 launch explicitly wires its velocity smoother
    # output to /cmd_vel_raw. The safety supervisor then publishes /cmd_vel.
    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav_pkg, 'launch', 'nav2.launch.py')),
        launch_arguments={
            'use_sim_time': 'true',
            'autostart': 'true',
        }.items(),
    )

    supervisor = Node(package='crip_safety', executable='safety_supervisor',
                      name='safety_supervisor', output='screen',
                      parameters=[{'use_sim_time': True}])

    perception = Node(package='crip_perception', executable='obstacle_monitor',
                      output='screen',
                      parameters=[{'use_sim_time': True, 'warn_distance': 0.8}])
    world_model = Node(package='crip_world_model', executable='world_model',
                       name='world_model', output='screen',
                       parameters=[{'use_sim_time': True}])
    llm_brain = Node(package='crip_llm_brain', executable='llm_brain',
                     name='llm_brain', output='screen',
                     parameters=[{'use_sim_time': True}])
    vlm = Node(package='crip_vlm', executable='vlm', name='vlm',
               output='screen',
               parameters=[{'use_sim_time': True, 'auto_period': 0.0}])

    prediction = Node(package='crip_prediction', executable='prediction',
                      name='prediction', output='screen',
                      parameters=[{'use_sim_time': True,
                                   'registry_dir': os.path.join(
                                       os.path.expanduser('~'), '.crip', 'registry')}])

    memory = Node(package='crip_memory', executable='memory', name='memory',
                  output='screen', parameters=[{'use_sim_time': True}])

    _crip_home = os.path.join(os.path.expanduser('~'), '.crip')
    logger = Node(package='crip_learning', executable='experience_logger',
                  name='experience_logger', output='screen',
                  parameters=[{'use_sim_time': True,
                               'log_path': os.path.join(_crip_home, 'experience.jsonl')}])
    trainer = Node(package='crip_learning', executable='learning_trainer',
                   name='learning_trainer', output='screen',
                   parameters=[{'use_sim_time': True,
                                'log_path': os.path.join(_crip_home, 'experience.jsonl'),
                                'registry_dir': os.path.join(_crip_home, 'registry'),
                                'interval_sec': 3600.0,
                                'run_on_start': False}])

    dashboard = Node(package='crip_dashboard', executable='dashboard',
                     name='dashboard', output='screen',
                     parameters=[{'use_sim_time': True, 'port': 8088}])

    explain = Node(package='crip_explain', executable='explain',
                   name='explain', output='screen',
                   parameters=[{'use_sim_time': True}])

    fleet = Node(package='crip_fleet', executable='fleet', name='fleet',
                 output='screen',
                 parameters=[{'use_sim_time': True, 'robot_id': 'robot_alpha'}])

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        sim,
        nav2,
        supervisor,
        perception,
        world_model,
        llm_brain,
        vlm,
        prediction,
        memory,
        logger,
        trainer,
        dashboard,
        explain,
        fleet,
    ])
