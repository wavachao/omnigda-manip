"""Launch ManiSkill 3 sim bridge + GDA registry + 2D/3D perception (+ optional RViz2)."""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    backend = LaunchConfiguration("backend")
    seed = LaunchConfiguration("seed")
    num_objects = LaunchConfiguration("num_objects")
    include_novel = LaunchConfiguration("include_novel")
    use_rviz = LaunchConfiguration("use_rviz")
    memory_dir = LaunchConfiguration("memory_dir")

    rviz_config = os.path.join(
        get_package_share_directory("omnigda_sim"),
        "rviz",
        "omnigda.rviz",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "backend",
                default_value="pytorch",
                description="2D backbone inference backend: pytorch | onnx | tensorrt",
            ),
            DeclareLaunchArgument(
                "seed",
                default_value="42",
                description="Random seed for ManiSkill 3 cluttered tabletop episode",
            ),
            DeclareLaunchArgument(
                "num_objects",
                default_value="6",
                description="Number of catalog objects spawned on the tabletop",
            ),
            DeclareLaunchArgument(
                "include_novel",
                default_value="false",
                description="Include held-out novel OOD objects in the tabletop scene",
            ),
            DeclareLaunchArgument(
                "use_rviz",
                default_value="false",
                description="Launch RViz2 with the preconfigured OmniGDA display layout",
            ),
            DeclareLaunchArgument(
                "memory_dir",
                default_value="/tmp/omnigda_memory",
                description="Shared directory for persisted 2D/3D GDA prototype memories",
            ),
            Node(
                package="omnigda_sim",
                executable="sim_bridge_node",
                name="sim_bridge_node",
                output="screen",
                parameters=[
                    {
                        "seed": ParameterValue(seed, value_type=int),
                        "num_objects": ParameterValue(num_objects, value_type=int),
                        "include_novel": ParameterValue(include_novel, value_type=bool),
                        "init_controller": False,
                    }
                ],
            ),
            Node(
                package="omnigda_perception",
                executable="registry_service_node",
                name="registry_service_node",
                output="screen",
                parameters=[
                    {
                        "backend": backend,
                        "memory_dir": memory_dir,
                    }
                ],
            ),
            Node(
                package="omnigda_perception",
                executable="perception_2d_node",
                name="perception_2d_node",
                output="screen",
                parameters=[
                    {
                        "backend": backend,
                        "memory_dir": memory_dir,
                    }
                ],
            ),
            Node(
                package="omnigda_perception",
                executable="perception_3d_node",
                name="perception_3d_node",
                output="screen",
                parameters=[
                    {
                        "backend": backend,
                        "memory_dir": memory_dir,
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
                condition=IfCondition(use_rviz),
            ),
        ]
    )
