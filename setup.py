import os
from glob import glob
from setuptools import setup

package_name = 'offboard_py'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name, package_name + '.latency_tools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml')))
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='usrg',
    maintainer_email='seungwook1024@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'offboard_control = offboard_py.offboard_control:main',
            'thrust_and_rate_control = offboard_py.thrust_and_rate_control_ros2:main',
            'sysid_node = offboard_py.sysid_node:main',
            'sysid_analyze = offboard_py.sysid_analyze:main',
            'imu_latency = offboard_py.latency_tools.imu_latency:main',
            'detector_latency_bench = offboard_py.latency_tools.detector_latency_bench:main',
            'image_transport_latency = offboard_py.latency_tools.image_transport_latency:main',
            'datalink_latency_ping = offboard_py.latency_tools.datalink_latency_ping:main',
            'flight_test_velocity_step = offboard_py.flight_test_velocity_step:main',
        ],
    },
)
