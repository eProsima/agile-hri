import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hri_pose_detect'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include all launch files.
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
        # Include all models files.
        (os.path.join('share', package_name, 'models'), glob(os.path.join('models', '*.pt'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Carlos Ferreira',
    author_email='carlosferreira@eprosima.com',
    maintainer='Carlos Ferreira',
    maintainer_email='carlosferreira@eprosima.com',
    description='ROS 2 node implementing pose estimation, using YOLO.\
                 Part of ROS4HRI.',
    license='Apache License 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node_pose_detect = hri_pose_detect.node_pose_detect:main'
        ],
    },
)
