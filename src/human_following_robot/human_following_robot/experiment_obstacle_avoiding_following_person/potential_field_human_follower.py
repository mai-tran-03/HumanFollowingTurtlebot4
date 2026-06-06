from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
import cv2
import math
import numpy as np
from .histogram_visual_tracker import HistogramVisualTracker

class PotentialFieldHumanFollower(Node):
    def __init__(self):
        super().__init__('reactive_human_follower')
        
        self.declare_parameter('robot_name', 'don')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.get_logger().info(f"Initializing Potential Field Follower for: '{self.robot_name}'")

        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = HistogramVisualTracker()

        # Dynamic Topics
        image_topic = f'/{self.robot_name}/oakd/rgb/preview/image_raw'
        scan_topic = f'/{self.robot_name}/scan'
        vel_topic = f'/{self.robot_name}/cmd_vel'
        viz_topic = f'/yolo/{self.robot_name}/visualization'

        # Subscriptions & Publishers
        self.img_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.vel_pub = self.create_publisher(Twist, vel_topic, 10)
        self.viz_pub = self.create_publisher(Image, viz_topic, 10)
        
        # State and Sensor Variables
        self.target_person_id = None
        self.latest_scan = None
        
        # Potential Field Hyperparameters
        self.K_ATTRACTIVE = 0.5   # Gain pulling towards human
        self.K_REPULSIVE = 1.2    # Gain pushing away from obstacles
        self.OBSTACLE_THRESH = 1.0 # Distance (meters) where obstacles start pushing the robot
        self.MIN_FOLLOW_DIST = 1.2 # Stop moving forward if closer than this to human

    def scan_callback(self, msg):
        """Store the latest LiDAR scan data"""
        self.latest_scan = msg

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]
        
        results = self.model.track(cv_image, classes=[0], persist=True, verbose=False)
        target_box = self.evaluate_tracking_states(results, cv_image)
        
        # Vector Initialization (x = forward/backward, y = left/right)
        f_total_x = 0.0
        f_total_y = 0.0

        if target_box is not None:
            box_coords = target_box.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)

            # 1. Calculate Attractive Forces (Target Location)
            distance, angle = self.estimate_human_relative_pos(center_x, box_coords, img_width)
            
            if distance is not None and angle is not None:
                # If target is further than safety margin, pull forward
                target_range = max(0.0, distance - self.MIN_FOLLOW_DIST)
                f_attr_x = target_range * math.cos(angle) * self.K_ATTRACTIVE
                f_attr_y = target_range * math.sin(angle) * self.K_ATTRACTIVE
                
                f_total_x += f_attr_x
                f_total_y += f_attr_y

        # 2. Calculate Repulsive Forces (LiDAR Obstacles)
        if self.latest_scan is not None:
            f_rep_x, f_rep_y = self.calculate_repulsive_force()
            f_total_x += f_rep_x
            f_total_y += f_rep_y

        # 3. Convert Resultant Vector into Robot Velocities (Twist)
        twist_msg = Twist()
        
        if target_box is not None or (abs(f_total_x) > 0.05 or abs(f_total_y) > 0.05):
            # Map forward vector component directly to linear speed
            twist_msg.linear.x = max(-0.2, min(0.5, f_total_x)) 
            
            # Use the angle of the resultant vector to steer the robot
            resultant_angle = math.atan2(f_total_y, f_total_x)
            twist_msg.angular.z = max(-1.0, min(1.0, resultant_angle * 1.5))
        else:
            twist_msg.linear.x = 0.0
            twist_msg.angular.z = 0.0

        # Publish instantly at frame-rate
        self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)

    def calculate_repulsive_force(self):
        """Iterates over LiDAR points to compute a net push-away vector."""
        f_rep_x = 0.0
        f_rep_y = 0.0
        
        scan = self.latest_scan
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        for i, r in enumerate(scan.ranges):
            # Skip invalid, phantom, or out-of-range readings
            if math.isnan(r) or math.isinf(r) or r < scan.range_min or r > self.OBSTACLE_THRESH:
                continue
                
            # Angle of specific LiDAR beam relative to base_link center line
            angle = angle_min + (i * angle_increment)
            
            # Force magnitude increases quadratically as you get closer to an obstacle
            # Formula: Force = K * (1/dist - 1/threshold)^2
            magnitude = self.K_REPULSIVE * ((1.0 / r) - (1.0 / self.OBSTACLE_THRESH)) ** 2
            
            # The force pushes AWAY from the obstacle, hence the negative signs
            f_rep_x -= magnitude * math.cos(angle)
            f_rep_y -= magnitude * math.sin(angle)
            
        return f_rep_x, f_rep_y

    def estimate_human_relative_pos(self, center_x, box_coords, img_width):
        HFOV = 60.0 
        error_x = center_x - (img_width / 2.0)
        # ROS Angle Map: Left is positive, Right is negative
        angle_rad = -(error_x / (img_width / 2.0)) * (math.radians(HFOV) / 2.0)
        
        box_height = float(box_coords[3] - box_coords[1])
        if box_height == 0: 
            return None, None
        
        distance = 300.0 / box_height 
        return distance, angle_rad

    def evaluate_tracking_states(self, results, cv_image):
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return None
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()
        
        if self.target_person_id in ids:
            target_idx = ids.index(self.target_person_id)
            coords = boxes[target_idx].xyxy[0].cpu().numpy()
            self.v_tracker.update_profile(cv_image, coords)
            return boxes[target_idx]
        
        for idx, current_id in enumerate(ids):
            coords = boxes[idx].xyxy[0].cpu().numpy()
            if self.v_tracker.matches_profile(cv_image, coords):
                self.target_person_id = current_id
                return boxes[idx]

        if self.target_person_id is None and len(ids) > 0:
            self.target_person_id = ids[0]
            coords = boxes[0].xyxy[0].cpu().numpy()
            self.v_tracker.update_profile(cv_image, coords)
            return boxes[0]

        return None
    
    def publish_visualization(self, results):
        if results and len(results) > 0 and results[0].boxes is not None:
            annotated_img = results[0].plot()
            msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
            self.viz_pub.publish(msg_out)