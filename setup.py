from glob import glob

from setuptools import find_packages, setup

package_name = "fleetlink"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools", "numpy", "msgpack", "zstandard"],
    zip_safe=True,
    maintainer="harelb",
    maintainer_email="harelb@mit.edu",
    description="QoS-aware robot to base-station scene-graph bridge",
    license="BSD-3-Clause",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fleetlink-agent = fleetlink.agent:main",
            "fleetlink-bridge = fleetlink.relay_ros:main",
            "fleetlink-image-pack = fleetlink.image_pack:main",
        ]
    },
)
