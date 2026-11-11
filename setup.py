from setuptools import find_packages, setup

package_name = "fleetlink"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/config",
            [
                "config/fleetlink.yaml",
                "config/fleetlink_router.json5",
                "config/relay.example.yaml",
            ],
        ),
        # bash launcher for the base's dedicated zenoh router (ros2 run fleetlink fleetlink-router)
        ("lib/" + package_name, ["scripts/fleetlink-router"]),
    ],
    python_requires=">=3.10",
    install_requires=["setuptools", "numpy", "msgpack", "zstandard", "PyYAML"],
    extras_require={
        # the zenoh transport (agent, bridge, console receiver); must match rmw_zenoh's zenoh 1.8
        "zenoh": ["eclipse-zenoh==1.8.0"],
        # camera frames (C4) and the image-store codec: OpenCV or Pillow, either works
        "images": ["opencv-python-headless", "Pillow"],
        # base-side mesh deformation (mesh_source: odometric)
        "deform": ["scipy"],
        "test": ["pytest"],
    },
    zip_safe=True,
    maintainer="harelb",
    maintainer_email="harelb@mit.edu",
    description="QoS-aware robot to base-station link over zenoh for lossy radios",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "fleetlink-agent = fleetlink.launcher:agent",
            "fleetlink-bridge = fleetlink.launcher:bridge",
            "fleetlink-image-pack = fleetlink.image_pack:main",
        ]
    },
)
