from setuptools import setup
import os
from glob import glob

package_name = 'crip_vlm'

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
    description='Vision-Language scene reasoning for CRIP.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'vlm = crip_vlm.vlm_node:main',
        ],
    },
)
