from setuptools import find_packages, setup
# import os
# from glob import glob

package_name = 'human_following_robot'

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
    maintainer='Awa Cisse, Jeremiah Dawson, Mai Tran, Mason Moses',
    maintainer_email='awacisse@gmail.com, jeremiahdawson556@gmail.com, mai.tran1714@gmail.com, masonsmoses@gmail.com',
    description='ROS 2 Node designed to detect, track, and physically follow a person using a camera and a pre-trained YOLO object detection model.',
    license='MIT License',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'h_follower = human_following_robot.human_follower_node:main',
        ],
    },
)
