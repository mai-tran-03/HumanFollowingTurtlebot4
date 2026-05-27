from ultralytics import YOLO

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped


class HumanFollower(Node):
    def __init__(self):
        super().__init__('human_follower')
        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')

        # Subscriber for TurtleBot4 RGB camera feed
        self.img_sub = self.create_subscription(
        Image, '/don/oakd/rgb/preview/image_raw', self.image_callback, 10)

        # Publisher for the annotated YOLO-processed images
        self.viz_pub = self.create_publisher(Image, '/yolo/visualization', 10)

        # Publisher for robot movement
        self.vel_pub = self.create_publisher(TwistStamped, '/don/cmd_vel', 10)
        
        self.last_linear_twist = 0.0
        self.last_angular_twist = 0.0

    def image_callback(self, msg):
        # Convert ROS Image to OpenCV BGR frame
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Process only the person class (0) from the frame
        results = self.model(cv_image, classes=[0], verbose=False)

        # Draw bounding boxes and publish visualization
        annotated = results[0].plot()
        msg_out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        self.viz_pub.publish(msg_out)

        # Initialize the stamped twist message
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'

        STOP_HEIGHT_THRESH = 200.0  # Stop if the box height is exceeds this
        img_width = cv_image.shape[1]
        img_center = img_width / 2.0

        if len(results[0].boxes) > 0:
            # Detected the first person, extract tracking data
            box = results[0].boxes[0].xyxy[0]
            center_x = float((box[0] + box[2]) / 2.0)
            box_height = float(box[3] - box[1])
            self.get_logger().info(f"Height - Center X: {box_height:.2f}, {center_x:.2f}")
        
            # Adjusted angular velocity, steer based on error from image center
            error_x = center_x - img_center
            twist_msg.twist.angular.z = float(-error_x / 200.0)

            # Adjusted linear velocity, move forward if person is far, stop if close by
            if box_height >= STOP_HEIGHT_THRESH:
                twist_msg.twist.linear.x = 0.0
            else:
                twist_msg.twist.linear.x = 0.2

            # Save current speeds to memory for future frames if person is lost
            self.last_linear_twist = twist_msg.twist.linear.x
            self.last_angular_twist = twist_msg.twist.angular.z

            self.get_logger().info(f"Linear - Angular: {self.last_linear_twist:.2f} - {self.last_angular_twist:.2f}")

        else:
            twist_msg.twist.linear.x = 0.0

            # Keep spinning right/left when person is lost if saved speed exists
            if self.last_angular_twist != 0.0:
                search_direction = 1.0 if self.last_angular_twist > 0 else -1.0
                twist_msg.twist.angular.z = search_direction * 0.4
            else:
                twist_msg.twist.angular.z = 0.4
        
        self.vel_pub.publish(twist_msg)


def main(args=None):
    rclpy.init(args=args)
    node = HumanFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

