import math
import time
import rclpy
from rclpy.action import ActionClient
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from irobot_create_msgs.action import NavigateToPosition

from ..human_follower_node import HumanFollower


class Nav2HumanFollower(HumanFollower):

    def __init__(self):
        # 1. Initialize the base HumanFollower class
        super().__init__()
        self.get_logger().info(
            f"Extending HumanFollower with Nav2 Action Stack for: '{self.robot_ns}'"
        )

        # 2. Setup the Nav2 Action Client using base parameters
        nav_pos_topic = f'/{self.robot_ns}/navigate_to_position'
        self.nav_client = ActionClient(self, NavigateToPosition, nav_pos_topic)

        # State tracking management
        self.goal_handle = None

        # Rate limiting properties to resolve execution delay/stuttering
        self.last_goal_sent_time = 0.0
        self.GOAL_UPDATE_INTERVAL = (
            0.5  # Update Nav2 target layout every 0.5 seconds (2 Hz)
        )

    def image_callback(self, msg):
        """Overrides base image processing pipeline to forward coordinates to Nav2."""
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]

        # Use base YOLO tracking configurations
        results = self.model.track(cv_image, classes=[0], persist=True, verbose=False)

        # Standard clean extraction for the first confident bounding box detection
        valid_target = None
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    valid_target = box
                    break

        current_time = time.time()

        if valid_target is not None:
            box_coords = valid_target.xyxy[0].cpu().numpy()
            center_x = float((box_coords[0] + box_coords[2]) / 2.0)

            # Convert 2D pixel coordinates into spatial estimation offsets
            distance, angle = self.estimate_human_relative_pos(
                center_x, box_coords, img_width
            )

            if distance is not None and angle is not None:
                # Enforce rate limit constraints to optimize action server loads
                if (
                    current_time - self.last_goal_sent_time
                ) > self.GOAL_UPDATE_INTERVAL:
                    self.send_nav2_goal(distance, angle)
                    self.last_goal_sent_time = current_time
        else:
            self.get_logger().info(
                "No person detected in frame.", throttle_duration_sec=2.0
            )

        # Render visual streaming tracking layouts utilizing base definitions
        self.publish_visualization(results)

    def estimate_human_relative_pos(self, center_x, box_coords, img_width):
        """Estimates target distance and heading angle from camera frames."""
        HFOV = 60.0
        error_x = center_x - (img_width / 2.0)

        # Map pixel error to spatial heading angle (Positive is Left, Negative is Right)
        angle_rad = -(error_x / (img_width / 2.0)) * (
            math.radians(HFOV) / 2.0
        )

        box_height = float(box_coords[3] - box_coords[1])
        if box_height == 0:
            return None, None

        distance = 300.0 / box_height

        # Threshold safety check: stop targeting if the person is too close
        if distance < 1.2:
            return None, None

        return distance, angle_rad

    def send_nav2_goal(self, distance, angle):
        """Dispatches an asynchronous relative translation position to Nav2."""
        if not self.nav_client.wait_for_server(timeout_sec=0.1):
            return

        # Computes goal translations keeping the platform trailing behind target
        target_range = max(0.2, distance - 1.0)
        goal_x = target_range * math.cos(angle)
        goal_y = target_range * math.sin(angle)

        goal_msg = NavigateToPosition.Goal()
        goal_msg.goal_pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.goal_pose.header.frame_id = f"{self.robot_ns}/base_link"

        goal_msg.goal_pose.pose.position.x = goal_x
        goal_msg.goal_pose.pose.position.y = goal_y

        # Target orientation quaternion transformations
        goal_msg.goal_pose.pose.orientation.z = math.sin(angle / 2.0)
        goal_msg.goal_pose.pose.orientation.w = math.cos(angle / 2.0)
        goal_msg.achieve_goal_heading = True

        self.get_logger().info(
            f"Updating Nav2 target: ({goal_x:.2f}m, {goal_y:.2f}m)"
        )

        # Dispatches goal handles asynchronously preserving image subscription loops
        send_goal_future = self.nav_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Handles cancellation of stale tracking goals when a fresh path arrives."""
        new_goal_handle = future.result()
        if not new_goal_handle.accepted:
            return

        # Preempt active tracking plans to clear updates seamlessly
        if self.goal_handle is not None:
            try:
                self.goal_handle.cancel_goal_async()
            except Exception:
                pass

        self.goal_handle = new_goal_handle


def main(args=None):
    rclpy.init(args=args)
    node = Nav2HumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()