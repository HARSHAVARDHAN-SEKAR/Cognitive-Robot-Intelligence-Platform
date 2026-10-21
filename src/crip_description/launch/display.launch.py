"""Display the CRIP robot in RViz (no Gazebo)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = FindPackageShare('crip_description')
    xacro_file = PathJoinSubstitution([pkg, 'urdf', 'crip_robot.urdf.xacro'])
    use_gui = LaunchConfiguration('gui')

    # BUGFIX: must wrap in ParameterValue(value_type=str) or ROS tries to
    # parse the URDF as YAML and fails.
    robot_description = {'robot_description': ParameterValue(
        Command(['xacro ', xacro_file]), value_type=str)}

    return LaunchDescription([
        DeclareLaunchArgument('gui', default_value='true',
                              description='Start joint_state_publisher_gui'),
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            parameters=[robot_description],
        ),
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            condition=IfCondition(use_gui),
        ),
        Node(
            package='joint_state_publisher',
            executable='joint_state_publisher',
            condition=UnlessCondition(use_gui),
        ),
    ])
