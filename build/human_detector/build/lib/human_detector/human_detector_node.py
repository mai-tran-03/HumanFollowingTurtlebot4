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
        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = VisualTracker()

        # ROS infrastructure:
        #   subscribe to camera feed
        #   publish robot movement
        #   publish YOLO processed images
        self.img_sub = self.create_subscription(
            Image, '/raph/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.vel_pub = self.create_publisher(TwistStamped, '/raph/cmd_vel', 10)
        self.viz_pub = self.create_publisher(Image, '/yolo/visualization', 10)
        
        # Persistent state variables
        self.target_person_id = None
        self.last_direction_person_detected = 0.0
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
            self.execute_tracking_behavior(target_box, cv_image, img_width, twist_msg)
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
            return boxes[target_idx]
        
        # No target person, try to Re-Identify based on visual profile
        for idx, current_id in enumerate(ids):
            coords = boxes[idx].xyxy[0].cpu().numpy()
            if self.v_tracker.matches_profile(cv_image, coords):
                self.target_person_id = current_id  # Re-lock to new ID
                self.get_logger().info(f"Re-ID Success! Re-locked to person under new ID: {current_id}")
                return boxes[idx]

        # No target person and someone new show up
        if self.target_person_id is None and len(ids) > 0:
            self.target_person_id = ids[0]
            self.get_logger().info(f"Fresh Target Lock on ID: {self.target_person_id}")
            return boxes[0]

        return None

    def execute_tracking_behavior(self, target_box, cv_image, img_width, twist_msg):
        if target_box is not None:
            self.last_time_person_detected = None # Reset timer if person found

            # Extract coordinates (xyxy format)
            box_coords = target_box.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)
            box_height = float(box_coords[3] - box_coords[1])

            # Update visual fingerprint profile continuously while tracking
            self.v_tracker.update_profile(cv_image, box_coords)
            img_width
            # Control angular velocity, steer based on error from image center
            error_x = center_x - (img_width / 2.0)
            twist_msg.twist.angular.z = float(-error_x / 200.0)

            # Control linear velocity, stop if person too close
            twist_msg.twist.linear.x = 0.0 if box_height >= self.STOP_HEIGHT_THRESH else 0.2
            
            # Save last known angular velocity direction before person leave frame
            if abs(twist_msg.twist.angular.z) > 0.05:
                self.last_direction_person_detected = 1.0 if twist_msg.twist.angular.z > 0 else -1.0
                direction_text = "LEFT (Counter-Clockwise)" if self.last_direction_person_detected > 0 else "RIGHT (Clockwise)"
                self.get_logger().info(
                    f"Saved escape trajectory: {direction_text} | Vel: {twist_msg.twist.angular.z:.2f} rad/s"
                )


    def execute_searching_behavior(self, twist_msg):
        twist_msg.twist.linear.x = 0.0
        curr_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
            self.get_logger().warning("Target lost, rotate 360")
        
        elapsed_search_time = curr_time - self.last_time_person_detected

        # Stand still for a few seconds to let YOLO recover target ID
        if elapsed_search_time < self.GRACE_PERIOD:
            twist_msg.twist.angular.z = 0.0
            self.get_logger().info("Target missing: Waiting to see if ID re-appears...")
        
        # Spin in direction person last seen
        elif elapsed_search_time < (self.GRACE_PERIOD + self.SPIN_DURATION):
            twist_msg.twist.angular.z = self.last_direction_person_detected * self.SEARCH_SPEED
            self.get_logger().info(f"Searching... 360 spin")
            
        else:
            twist_msg.twist.angular.z = 0.0
            self.target_person_id = None
            self.get_logger().error("Stop searching. Target completely lost. Restart")
    
    def publish_visualization(self, results):
        """Annotate YOLO outputs and add on histogram fingerprint"""
        annotated_img = results[0].plot()

        hist_overlay = self.v_tracker.get_histogram_image()
        
        # Inject histogram picture onto top-left corner of visual stream output
        h, w, _ = hist_overlay.shape
        annotated_img[20:20+h, 20:20+w] = hist_overlay
        
        # Draw clear label border around fingerprint overlay
        cv2.putText(annotated_img, "Target Identity Profile", (20, 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.rectangle(annotated_img, (20, 20), (20+w, 20+h), (0, 255, 0), 2)

        msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
        self.viz_pub.publish(msg_out)
