from setuptools import find_packages, setup

package_name = "omnigda_manipulation"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="huachaowu",
    maintainer_email="huachaowu@qq.com",
    description="ROS 2 grasp planner and manipulation coordinator for OmniGDA-Manip",
    license="MIT",
    entry_points={
        "console_scripts": [
            "grasp_planner_node = omnigda_manipulation.grasp_planner_node:main",
            "coordinator_node = omnigda_manipulation.coordinator_node:main",
        ],
    },
)
