# Human Follower ROS 2 Node
**Carleton College, Introduction to Robotics**

**Awa Cisse, Mai Tran, Jeremiah Dawson, Mason Moses**

A ROS 2 package that enables a mobile robot, TurtleBot 4, to track and autonomously follow a human using real-time computer vision. 

The system 
- subscribes to an RGB camera feed, 
- processes the frames through a **YOLO** object detection model to detect a person, 
- calculates tracking errors, 
- outputs velocity commands to steer the robot towards the target, 
- features a "lost-target search" routine, and 
- publishes a visual debugging stream with bounding boxes.

---

## Project Architecture & Control Logic

The node implements a simple closed-loop visual servoing routine:
* **Angular Control (Steering):** Calculates the horizontal pixel error ($e_x$) between the bounding box center ($x_{center}$) and the true center of the image ($x_{img\_center}$). The robot uses a proportional control factor to steer and keep the human centered in its field of view.
* **Linear Control (Distance):** Monitors the **height** of the bounding box. Because the human appears larger as they get closer, a bounding box height exceeding `200.0` pixels acts as a safety threshold, causing the robot to stop to avoid collision. If the human is further away (smaller box height), the robot drives forward at `0.2 m/s`.
* **Target Recovery Memory:** If the human momentarily walks out of frame, the node remembers the last known direction the human was moving and rotates in place ($0.4 \text{ rad/s}$) towards that side to re-acquire the target.

---

## Installation and Setup

Execute the following steps from the root directory of the workspace.

1. Source ROS 2 Environment

Initialize the base ROS 2 environment variable setup (replace <ros-distro> with the installed version, e.g., jazzy):
    
    source /opt/ros/<ros-distro>/setup.bash

2. Install Python Packages

Install the required machine learning, object detection, and numeric processing packages using pip:
    
    pip install -r requirements.txt

3. Install System ROS 2 Dependencies

Ensure all foundational ROS 2 messaging packages and vision bridging tools are completely resolved using rosdep:

    rosdep install --from-paths src --ignore-src -r -y

4. Compile the Package

Build the workspace utilizing the colcon build tool specifically for the tracking package:

    colcon build --packages-select human_detector

---

## How to Run the Node
1. Environment Activation

Open a new terminal shell execution instance and source the base ROS 2 installation along with the newly compiled workspace install files:

    source /opt/ros/<ros-distro>/setup.bash
    source install/setup.bash

2. Execute the Node

Launch the human tracking script using the standard ROS 2 command-line interface:
    
    ros2 run human_detector h_detector

3. Verification & Topics

While the node is running, you can open another terminal window to verify data streams:

- Verify Velocity Output Commands:

        ros2 topic echo /don/cmd_vel

- View the Visual Bounding Box Stream:

        ros2 run rqt_image_view rqt_image_view
