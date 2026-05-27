# Human Follower ROS 2 Node

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