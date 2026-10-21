from setuptools import setup
import os
from glob import glob

package_name = 'crip_memory'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CRIP',
    maintainer_email='you@example.com',
    description='Semantic mission memory for CRIP.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'memory = crip_memory.memory_node:main',
        ],
    },
)
