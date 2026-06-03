from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from irobot_create_msgs.action import NavigateToPosition
import cv2
import math
import time
from .visual_tracker import VisualTracker

class Nav2HumanFollower(Node):
    def __init__(self):
        super().__init__('nav2_human_follower')
        
        self.declare_parameter('robot_name', 'don')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.get_logger().info(f"Initializing Nav2 Human Follower with Obstacle Avoidance for: '{self.robot_name}'")

        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = VisualTracker()

        # Dynamic Topics
        image_topic = f'/{self.robot_name}/oakd/rgb/preview/image_raw'
        viz_topic = f'/yolo/{self.robot_name}/visualization'
        nav_pos_topic = f'/{self.robot_name}/navigate_to_position'

        self.img_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.viz_pub = self.create_publisher(Image, viz_topic, 10)
        
        # Action Client for Nav2 Stack (handles obstacle costmaps automatically)
        self.nav_client = ActionClient(self, NavigateToPosition, nav_pos_topic)

        # State tracking
        self.target_person_id = None
        self.goal_handle = None
        
        # Rate limiting variables to fix the delay/stutter
        self.last_goal_sent_time = 0.0
        self.GOAL_UPDATE_INTERVAL = 0.5  # Only update Nav2 goal every 0.5 seconds (2 Hz)

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]
        
        results = self.model.track(
            cv_image, 
            classes=[0], 
            persist=True, 
            verbose=False, 
        )

        target_box = self.evaluate_tracking_states(results, cv_image)
        current_time = time.time()
        
        if target_box is not None:
            box_coords = target_box.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)

            # Estimate position
            distance, angle = self.estimate_human_relative_pos(center_x, box_coords, img_width)

            if distance is not None and angle is not None:
                # Rate limit the Nav2 requests so we don't choke the action server
                if (current_time - self.last_goal_sent_time) > self.GOAL_UPDATE_INTERVAL:
                    self.send_nav2_goal(distance, angle)
                    self.last_goal_sent_time = current_time
        else:
            self.get_logger().info("No person detected in frame.", throttle_duration_sec=2.0)
        
        self.publish_visualization(results)

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
    
    def estimate_human_relative_pos(self, center_x, box_coords, img_width):
        HFOV = 60.0 
        error_x = center_x - (img_width / 2.0)
        
        # Fixed: accurately mapped angles to ROS convention (Positive is Left, Negative is Right)
        angle_rad = -(error_x / (img_width / 2.0)) * (math.radians(HFOV) / 2.0)
        
        box_height = float(box_coords[3] - box_coords[1])
        if box_height == 0: return None, None
        
        distance = 300.0 / box_height 
        
        # Stop tracking/moving if human gets closer than 1.2 meters
        if distance < 1.2:
            return None, None
            
        return distance, angle_rad
    
    def send_nav2_goal(self, distance, angle):
        if not self.nav_client.wait_for_server(timeout_sec=0.1):
            return
        
        # Target distance keeps the robot safely behind the person
        target_range = max(0.2, distance - 1.0)
        goal_x = target_range * math.cos(angle)
        goal_y = target_range * math.sin(angle)

        goal_msg = NavigateToPosition.Goal()
        goal_msg.goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal_pose.header.frame_id = f'{self.robot_name}/base_link'
        
        goal_msg.goal_pose.pose.position.x = goal_x
        goal_msg.goal_pose.pose.position.y = goal_y
        
        # Fix orientation calculation to avoid permanent right turning
        goal_msg.goal_pose.pose.orientation.z = math.sin(angle / 2.0)
        goal_msg.goal_pose.pose.orientation.w = math.cos(angle / 2.0)
        goal_msg.achieve_goal_heading = True

        self.get_logger().info(f"Updating Nav2 target: ({goal_x:.2f}m, {goal_y:.2f}m)")
        
        # Send the goal asynchronously without blocking future camera frames
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)
    
    def goal_response_callback(self, future):
        new_goal_handle = future.result()
        if not new_goal_handle.accepted:
            return
        
        # If an older goal was running, cancel it so the robot seamlessly updates to the new human path
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass
                
        self.goal_handle = new_goal_handle

    def publish_visualization(self, results):
        if results and len(results) > 0 and results[0].boxes is not None:
            annotated_img = results[0].plot()
            msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
            self.viz_pub.publish(msg_out)
