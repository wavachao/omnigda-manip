from setuptools import find_packages, setup

package_name = "omnigda_perception"

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
    description="ROS 2 perception nodes (Registry, SegmentGDA-Plus 2D, Partial-Aware 3D GMDA) for OmniGDA-Manip",
    license="MIT",
    entry_points={
        "console_scripts": [
            "registry_service_node = omnigda_perception.registry_service_node:main",
            "perception_2d_node = omnigda_perception.perception_2d_node:main",
            "perception_3d_node = omnigda_perception.perception_3d_node:main",
        ],
    },
)
