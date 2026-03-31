import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hri_detection_display'

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
        # Include all RViz config files.
        (os.path.join('share', package_name, 'rviz'), glob(os.path.join('rviz', '*.rviz'))),
        # Include all XML profiles config files.
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.xml'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Carlos Ferreira',
    author_email='carlosferreira@eprosima.com',
    maintainer='Carlos Ferreira',
    maintainer_email='carlosferreira@eprosima.com',
    description='Contains display to visualize HRI detections and camera launch files.',
    license='Apache License, Version 2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'node_person_publisher = hri_detection_display.node_person_publisher:main',
            'node_person_viewer = hri_detection_display.node_person_viewer:main',
            'node_3D_skel_display = hri_detection_display.node_3D_skel_display:main',
        ],
    },
)
