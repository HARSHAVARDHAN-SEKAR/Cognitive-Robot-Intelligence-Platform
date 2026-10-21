from setuptools import setup
import os
from glob import glob

package_name = 'crip_llm_brain'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='CRIP',
    maintainer_email='you@example.com',
    description='LLM task planner for CRIP.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'llm_brain = crip_llm_brain.llm_brain_node:main',
        ],
    },
)
