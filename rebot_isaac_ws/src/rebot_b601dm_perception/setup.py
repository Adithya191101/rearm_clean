from setuptools import find_packages, setup

package_name = "rebot_b601dm_perception"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="reBot Isaac integration",
    maintainer_email="rebot-dev@example.invalid",
    description=(
        "ROS-free soup-can grasp geometry for the reBot B601-DM workflow."
    ),
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "write_grasps = rebot_b601dm_perception.write_grasps:main",
        ],
    },
)
