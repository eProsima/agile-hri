import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'hri_tts'

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
        # Install all audio files.
        (os.path.join('share', package_name, 'audio_files'), glob(os.path.join('audio_files', '*.wav'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    author='Eugenio Collado',
    author_email='eugeniocollado@eprosima.com',
    maintainer='Carlos Ferreira',
    maintainer_email='carlosferreira@eprosima.com',
    description='ROS 2 Action server implementing Text to Speech features',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'node_tts_sub = hri_tts.tts_sub:main',
            'node_tts_pub = hri_tts.tts_pub:main',
            'tts_gen = hri_tts.tts_gen:main',
            'list_speakers = hri_tts.list_speakers:main',
            'download_tt_models = hri_tts.download_models:main',
        ],
    },
)
