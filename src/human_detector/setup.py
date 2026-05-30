from setuptools import find_packages, setup
# import os
# from glob import glob

package_name = 'human_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='localdawsonj2',
    maintainer_email='localdawsonj2@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            # 'h_detector = human_detector.human_detector_node:main',
            'human_detector_exec = human_detector.main:main',
        ],
    },
)
