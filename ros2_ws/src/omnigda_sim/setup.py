import os
from glob import glob

from setuptools import find_packages, setup

package_name = "omnigda_sim"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "rviz"), glob("rviz/*.rviz")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="huachaowu",
    maintainer_email="huachaowu@qq.com",
    description="ManiSkill 3 simulation bridge and ROS 2 conversions for OmniGDA-Manip",
    license="MIT",
    entry_points={
        "console_scripts": [
            "sim_bridge_node = omnigda_sim.sim_bridge_node:main",
        ],
    },
)
