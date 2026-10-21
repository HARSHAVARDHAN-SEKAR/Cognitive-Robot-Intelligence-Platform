"""
CRIP full simulation bringup.

Launches:
  * Gazebo (gz-sim) with the CRIP world
  * robot_state_publisher (publishes /robot_description and TF)
  * spawns the robot into Gazebo
  * ros_gz parameter bridge (cmd_vel, odom, scan, camera, clock, tf, joint_states)
  * RViz2 (optional)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, IncludeLaunchDescription,
                            ExecuteProcess, SetEnvironmentVariable)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (Command, LaunchConfiguration,
                                   PathJoinSubstitution, PythonExpression)
from launch_ros.parameter_descriptions import ParameterValue
from launch.conditions import IfCondition
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    desc_pkg = get_package_share_directory('crip_description')
    bringup_pkg = get_package_share_directory('crip_bringup')

    xacro_file = os.path.join(desc_pkg, 'urdf', 'crip_robot.urdf.xacro')
    world_file = os.path.join(bringup_pkg, 'worlds', 'crip_world.sdf')
    bridge_cfg = os.path.join(bringup_pkg, 'config', 'gz_bridge.yaml')
    rviz_cfg = os.path.join(bringup_pkg, 'rviz', 'crip.rviz')

    use_rviz = LaunchConfiguration('rviz')
    headless = LaunchConfiguration('headless')

    # BUGFIX: the xacro output MUST be wrapped in ParameterValue(value_type=str).
    # Without it, ROS 2 tries to parse the URDF XML as YAML and fails with
    # "Unable to parse the value of parameter robot_description as yaml".
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file]), value_type=str),
        'use_sim_time': True,
    }

    # Make package meshes/models findable by Gazebo
    set_gz_resource = SetEnvironmentVariable(
        name='GZ_SIM_RESOURCE_PATH',
        value=os.path.join(desc_pkg, '..'))

    # Launch gz sim. -r runs immediately, -v 3 sets verbosity.
    # headless:=true adds -s (server only, no GUI window) -- required when
    # there is no X display (headless server, CI, or a container without
    # xhost access). Without it Gazebo aborts on "could not connect to display".
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('ros_gz_sim'),
                         'launch', 'gz_sim.launch.py')),
        launch_arguments={
            'gz_args': [
                PythonExpression(
                    ["'-s -r -v 3 ' if '",
                     LaunchConfiguration('headless'),
                     "' == 'true' else '-r -v 3 '"]),
                world_file,
            ],
        }.items(),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    spawn_robot = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=[
            '-topic', 'robot_description',
            '-name', 'crip_robot',
            '-x', '0', '-y', '0', '-z', '0.2',
        ],
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        output='screen',
        parameters=[{'config_file': bridge_cfg, 'use_sim_time': True}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_cfg],
        condition=IfCondition(use_rviz),
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('headless', default_value='false'),
        set_gz_resource,
        gz_sim,
        robot_state_publisher,
        spawn_robot,
        bridge,
        rviz,
    ])
