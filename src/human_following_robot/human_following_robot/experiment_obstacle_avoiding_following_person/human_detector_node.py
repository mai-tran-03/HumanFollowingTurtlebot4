from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
import cv2
from .visual_tracker import VisualTracker


class HumanDetector(Node):
    def __init__(self):
        super().__init__('human_follower')
        
        # Get runtime parameter (defaults to 'don' if not specified)
        # Run specific robot name, e.g.,
        #   ros2 run human_detector human_detector_exec --ros-args -p robot_name:=don
        self.declare_parameter('robot_name', 'don')
        self.robot_name = self.get_parameter('robot_name').get_parameter_value().string_value
        self.get_logger().info(f"Initializing HumanDetector for robot: '{self.robot_name}'")

        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = VisualTracker()

        # Construct topics dynamically using f-strings
        image_topic = f'/{self.robot_name}/oakd/rgb/preview/image_raw'
        vel_topic = f'/{self.robot_name}/cmd_vel'
        viz_topic = f'/yolo/{self.robot_name}/visualization'
        scan_topic = f'/{self.robot_name}/scan'

        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = VisualTracker()

        # ROS infrastructure:
        #   subscribe to camera feed
        #   subscribe to lidar
        #   publish robot movement
        #   publish YOLO processed images
        self.img_sub = self.create_subscription(
            Image, image_topic, self.image_callback, 10)
        self.vel_pub = self.create_publisher(TwistStamped, vel_topic, 10)
        self.viz_pub = self.create_publisher(Image, viz_topic, 10)
        
        # Persistent state variables
        self.target_person_id = None
        self.last_direction_person_detected = 1.0 # default left turn
        self.last_time_person_detected = None
        
        # Constant parameters
        self.SEARCH_SPEED = 0.4
        self.SPIN_DURATION = 16
        self.GRACE_PERIOD = 1.5
        self.STOP_HEIGHT_THRESH = 240.0

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV BGR frame
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]
        
        results = self.model.track(
            cv_image, 
            classes=[0], # Process only the person class (0) from the frame
            persist=True, 
            verbose=False, 
        )

        twist_msg = self.create_empty_twist()
        target_box = self.evaluate_tracking_states(results, cv_image)
        
        if target_box is not None:
            self.execute_tracking_behavior(target_box, img_width, twist_msg)
        else:
            self.execute_searching_behavior(twist_msg)
        
        self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)

    def create_empty_twist(self):
        """Initialize stamped twist message"""
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        return twist_msg

    def evaluate_tracking_states(self, results, cv_image):
        """Process YOLO trackers and handle target confirmation"""
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return None
        
        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()
        
        # Target ID exists, look for it in current detections
        if self.target_person_id in ids:
            target_idx = ids.index(self.target_person_id)
            coords = boxes[target_idx].xyxy[0].cpu().numpy()
            self.v_tracker.update_profile(cv_image, coords)
            return boxes[target_idx]
        
        # No target person, try to Re-Identify based on visual profile
        for idx, current_id in enumerate(ids):
            coords = boxes[idx].xyxy[0].cpu().numpy()
            if self.v_tracker.matches_profile(cv_image, coords):
                self.target_person_id = current_id
                return boxes[idx]

        # No target person and someone new show up
        if self.target_person_id is None and len(ids) > 0:
            self.target_person_id = ids[0]
            coords = boxes[0].xyxy[0].cpu().numpy()
            self.v_tracker.update_profile(cv_image, coords)
            return boxes[0]

        return None

    def execute_tracking_behavior(self, target_box, img_width, twist_msg):
        if target_box is not None:
            self.last_time_person_detected = None # Reset timer if person found

            # Extract coordinates (xyxy format)
            box_coords = target_box.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)
            box_height = float(box_coords[3] - box_coords[1])

            # Control angular velocity, steer based on error from image center
            error_x = center_x - (img_width / 2.0)
            twist_msg.twist.angular.z = float(-error_x / 200.0)

            # Control linear velocity, stop if person too close
            twist_msg.twist.linear.x = 0.0 if box_height >= self.STOP_HEIGHT_THRESH else 0.2
            
            # Save last known angular velocity direction before person leave frame
            if abs(twist_msg.twist.angular.z) > 0.05:
                self.last_direction_person_detected = 1.0 if twist_msg.twist.angular.z > 0 else -1.0

    def execute_searching_behavior(self, twist_msg):
        twist_msg.twist.linear.x = 0.0
        curr_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
        
        elapsed_search_time = curr_time - self.last_time_person_detected

        # Stand still for a few seconds to let YOLO recover target ID
        if elapsed_search_time < self.GRACE_PERIOD:
            twist_msg.twist.angular.z = 0.0
        
        # Spin in direction person last seen
        elif elapsed_search_time < (self.GRACE_PERIOD + self.SPIN_DURATION):
            twist_msg.twist.angular.z = self.last_direction_person_detected * self.SEARCH_SPEED
            
        else:
            twist_msg.twist.angular.z = 0.0
            self.target_person_id = None
    
    def publish_visualization(self, results):
        annotated_img = results[0].plot()
        msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
        self.viz_pub.publish(msg_out)