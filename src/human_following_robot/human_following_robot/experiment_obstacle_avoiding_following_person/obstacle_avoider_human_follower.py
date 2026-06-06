import cv2
import rclpy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped

from ..human_follower_node import HumanFollower
from .obstacle_avoidance import ObstacleAvoider


class ObstacleAvoidanceFollower(HumanFollower):
    def __init__(self):
        # 1. Initialize the base HumanFollower class
        # This auto-sets self.bridge, self.model, and dynamically sets topics using self.robot_ns
        super().__init__()
        self.get_logger().info("Extending HumanFollower with specific ObstacleAvoider logic.")

        # 2. Composition: Inject the customized obstacle avoider component
        self.obstacle_avoider = ObstacleAvoider(logger=self.get_logger())

        # Construct the scan topic using the namespace declared in the base node
        scan_topic = f'/{self.robot_ns}/scan'
        
        # 3. Handle additional LiDAR Subscriptions
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10
        )
        
        # Override specific tracking parameter overrides from base defaults if necessary
        self.SPIN_DURATION = 16.0

    def scan_callback(self, msg: LaserScan):
        """Feeds the latest LiDAR scan straight to the obstacle avoider module."""
        self.obstacle_avoider.update_scan(msg)

    def image_callback(self, msg: Image):
        """Overrides base camera pipeline to link YOLO vision streams with ObstacleAvoider logic."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]

        # Use base YOLO tracking models
        results = self.model.track(
            cv_image,
            classes=[0],   
            persist=True,
            verbose=False,
        )

        # Initialize base twist stamped frame outputs
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = f'{self.robot_ns}/base_link'

        # Simplified Target Strategy: Find first confident person detection frame
        valid_target = None
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    valid_target = box
                    break

        if valid_target is not None:
            self.last_time_person_detected = None  # Reset loss timestamp tracking

            # Extract tracking coordinates
            box_coords = valid_target.xyxy[0].cpu().numpy()
            center_x   = float((box_coords[0] + box_coords[2]) / 2.0)
            box_height = float(box_coords[3] - box_coords[1])

            # Calculate error angle relative to camera center axis (~60 deg HFOV mapping)
            error_x = center_x - (img_width / 2.0)
            human_angle_deg = float(-error_x / (img_width / 2.0)) * 30.0

            # Store directional tracking context for search behaviors
            if abs(error_x) > 15:
                self.last_direction_person_detected = -1.0 if error_x > 0 else 1.0

            # --- Core Obstacle Logic Branching ---
            if self.obstacle_avoider.is_path_blocked():
                self.get_logger().warning("Path blocked! Executing footprint gap routing.", throttle_duration_sec=2.0)
                
                # Query custom component for reactive path adjustments
                linear_x, angular_z = self.obstacle_avoider.get_avoidance_velocities(target_angle_deg=human_angle_deg)
                twist_msg.twist.linear.x  = linear_x
                twist_msg.twist.angular.z = angular_z
            else:
                # Use parent parameter metrics ($HEIGHT_THRESHOLD and $FORWARD_SPEED)
                twist_msg.twist.angular.z = float(-error_x / 200.0)
                twist_msg.twist.linear.x  = 0.0 if box_height >= self.HEIGHT_THRESHOLD else self.FORWARD_SPEED
        else:
            # Revert to standard search patterns if human drops from vision
            self.execute_searching_behavior(twist_msg)

        # Execute final motor actuations and publish visual diagnostics
        self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()