import math
import rclpy
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import TwistStamped

from ..human_follower_node import HumanFollower


class PotentialFieldHumanFollower(HumanFollower):

    def __init__(self):
        # 1. Initialize the base HumanFollower class
        super().__init__()
        self.get_logger().info(
            f"Extending HumanFollower with Artificial Potential Field logic for: '{self.robot_ns}'"
        )

        # 2. Add Potential Field Hyperparameters
        self.K_ATTRACTIVE = 0.5  # Gain pulling towards human
        self.K_REPULSIVE = 1.2  # Gain pushing away from obstacles
        self.OBSTACLE_THRESH = (
            1.0  # Distance (meters) where obstacles start pushing
        )
        self.MIN_FOLLOW_DIST = (
            1.2  # Stop forward motion if closer than this to human
        )

        # 3. Sensor Tracking State
        self.latest_scan = None

        # 4. Subscribe to the LiDAR scan topic (mapped to the base namespace)
        scan_topic = f"/{self.robot_ns}/scan"
        self.scan_sub = self.create_subscription(
            LaserScan, scan_topic, self.scan_callback, 10
        )

    def scan_callback(self, msg):
        """Stores the latest incoming LiDAR data array."""
        self.latest_scan = msg

    def image_callback(self, msg):
        """Overrides base camera pipeline to calculate combined vector forces."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]

        # Process frame using base tracking pipeline configuration
        results = self.model.track(
            cv_image, classes=[0], persist=True, verbose=False
        )

        # Simply isolate the first confident target person frame from base configuration
        valid_target = None
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    valid_target = box
                    break

        # Vector Initialization (x = forward/backward, y = left/right)
        f_total_x = 0.0
        f_total_y = 0.0

        if valid_target is not None:
            box_coords = valid_target.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)

            # 1. Calculate Attractive Forces (Pull toward Target Location)
            distance, angle = self.estimate_human_relative_pos(
                center_x, box_coords, img_width
            )

            if distance is not None and angle is not None:
                target_range = max(0.0, distance - self.MIN_FOLLOW_DIST)
                f_total_x += target_range * math.cos(angle) * self.K_ATTRACTIVE
                f_total_y += target_range * math.sin(angle) * self.K_ATTRACTIVE

        # 2. Calculate Repulsive Forces (Push away from LiDAR Obstacles)
        if self.latest_scan is not None:
            f_rep_x, f_rep_y = self.calculate_repulsive_force()
            f_total_x += f_rep_x
            f_total_y += f_rep_y

        # 3. Build Stamped Twist frame using base configuration settings
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = f"{self.robot_ns}/base_link"

        if valid_target is not None or (
            abs(f_total_x) > 0.05 or abs(f_total_y) > 0.05
        ):
            # Translate local force vector components straight to motor outputs
            twist_msg.twist.linear.x = max(-0.2, min(0.5, f_total_x))

            resultant_angle = math.atan2(f_total_y, f_total_x)
            twist_msg.twist.angular.z = max(-1.0, min(1.0, resultant_angle * 1.5))
        else:
            twist_msg.twist.linear.x = 0.0
            twist_msg.twist.angular.z = 0.0

        # Run motor actuation frames and display standard parent vision windows
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
            if (
                math.isnan(r)
                or math.isinf(r)
                or r < scan.range_min
                or r > self.OBSTACLE_THRESH
            ):
                continue

            angle = angle_min + (i * angle_increment)

            # Force magnitude equation scaling quadratically upon closer proximity
            magnitude = (
                self.K_REPULSIVE
                * ((1.0 / r) - (1.0 / self.OBSTACLE_THRESH)) ** 2
            )

            # Direct negative push components away from obstacle coordinate sources
            f_rep_x -= magnitude * math.cos(angle)
            f_rep_y -= magnitude * math.sin(angle)

        return f_rep_x, f_rep_y

    def estimate_human_relative_pos(self, center_x, box_coords, img_width):
        """Calculates distance and relative radians angle using image frame scaling."""
        HFOV = 60.0
        error_x = center_x - (img_width / 2.0)
        angle_rad = -(error_x / (img_width / 2.0)) * (
            math.radians(HFOV) / 2.0
        )

        box_height = float(box_coords[3] - box_coords[1])
        if box_height == 0:
            return None, None

        distance = 300.0 / box_height
        return distance, angle_rad


def main(args=None):
    rclpy.init(args=args)
    node = PotentialFieldHumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()