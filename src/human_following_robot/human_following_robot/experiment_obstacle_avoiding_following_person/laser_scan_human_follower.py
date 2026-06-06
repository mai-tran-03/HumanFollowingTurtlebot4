import rclpy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped

from ..human_follower_node import HumanFollower
from .laser_processor import LaserProcessor


class LaserScanHumanFollower(HumanFollower):
    def __init__(self):
        # 1. Initialize the base HumanFollower class
        super().__init__()
        self.get_logger().info("Extending node with LaserScan collision avoidance.")

        # 2. Composition: Inject the laser processor component
        self.laser_processor = LaserProcessor(avoid_distance=0.6, hard_stop_distance=0.35)

        # Construct the scan topic dynamically using the base node's namespace variable
        scan_topic = f'/{self.robot_ns}/scan'
        
        # 3. New ROS Subscription for LiDAR data
        self.scan_sub = self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        
        # New persistent obstacle tracking states
        self.obstacle_detected = False
        self.obstacle_in_way = False
        self.avoidance_steering_bias = 0.0

    def scan_callback(self, msg):
        """Processes LiDAR fields to extract obstacle flags and safety paths."""
        self.obstacle_detected, self.obstacle_in_way, self.avoidance_steering_bias = \
            self.laser_processor.process_scan(msg, logger=self.get_logger())

    def image_callback(self, msg):
        """Overrides base tracking pipeline to interlace YOLO tracking with collision parameters."""
        # Convert raw ROS image to OpenCV format using the base CvBridge instance
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]
        
        # Run the base YOLO tracking model
        results = self.model.track(
            cv_image, 
            classes=[0], 
            persist=True, 
            verbose=False, 
        )

        # Initialize the stamped twist message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = f'{self.robot_ns}/base_link'

        # Look for the first confident person box using base threshold parameters
        valid_target = None
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    valid_target = box
                    break
        
        # Merge vision decisions with laser obstacle data
        if valid_target is not None:
            self.calculate_tracking_with_avoidance(valid_target, img_width, twist_msg)
        else:
            self.calculate_searching_with_avoidance(twist_msg)
        
        # Actuate movement and show standard YOLO visualization from base code
        self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)

    def calculate_tracking_with_avoidance(self, target_box, img_width, twist_msg):
        """Calculates tracking maneuvers while prioritizing LiDAR collision safeguards."""
        self.last_time_person_detected = None 

        # Extract bounding box boundaries
        x1, y1, x2, y2 = target_box.xyxy[0].cpu().numpy()
        box_center_x = float((x1 + x2) / 2.0)
        box_height_y = y2 - y1

        # Turning error calculation
        positional_error_x = box_center_x - (img_width / 2.0)
        tracking_steering = float(-positional_error_x / 200.0)
        
        # Save last known steering direction before target escapes frame
        if abs(tracking_steering) > 0.05:
            self.last_direction_person_detected = 1.0 if positional_error_x > 0 else -1.0
        
        # Control Law mixing Vision and Laser Obstacle Avoidance
        if self.obstacle_in_way and abs(positional_error_x) > 120.0:
            twist_msg.twist.angular.z = self.avoidance_steering_bias
            twist_msg.twist.linear.x = 0.1
        else:
            twist_msg.twist.angular.z = tracking_steering
            # Use variables from base code ($HEIGHT_THRESHOLD and $FORWARD_SPEED)
            twist_msg.twist.linear.x = 0.0 if box_height_y >= self.HEIGHT_THRESHOLD else self.FORWARD_SPEED
        
        # Critical Override: Emergency Brake
        if self.obstacle_detected:
            self.get_logger().warn("EMERGENCY BRAKE: Obstacle blocking path!", throttle_duration_sec=2.0)
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.linear.y = 0.0
            twist_msg.twist.angular.z = 0.0

    def calculate_searching_with_avoidance(self, twist_msg):
        """Gracefully spins looking for target unless blocked by obstacles."""
        twist_msg.twist.linear.x = 0.0
        curr_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
        
        elapsed_search_time = curr_time - self.last_time_person_detected

        # Spin loop using search time metrics from base configuration ($SPIN_DURATION)
        if elapsed_search_time < self.SPIN_DURATION:
            if self.obstacle_in_way:
                twist_msg.twist.angular.z = self.avoidance_steering_bias
            else:
                twist_msg.twist.angular.z = self.last_direction_person_detected * self.SEARCH_SPEED
        else:
            twist_msg.twist.angular.z = 0.0
            self.get_logger().error("Stop searching. Target completely lost.", throttle_duration_sec=5.0)


def main(args=None):
    rclpy.init(args=args)
    node = LaserScanHumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()