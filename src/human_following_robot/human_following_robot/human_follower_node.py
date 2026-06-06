import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import TwistStamped
from ultralytics import YOLO


class HumanFollower(Node):
    def __init__(self):
        """
            ROS 2 Node designed to detect, track, and physically follow a person 
            using a camera and a pre-trained YOLO object detection model.
        """
        super().__init__('human_follower')
        
        # Get runtime parameter (default robot namespace 'don')
        # Run specific robot name, e.g.,
        #   ros2 run human_following_robot follower --ros-args -p robot_ns:=don
        self.declare_parameter('robot_ns', 'don')
        self.robot_ns = self.get_parameter('robot_ns').get_parameter_value().string_value
        self.get_logger().info(f"Initializing HumanFollower for robot: '{self.robot_ns}'")

        # Using CvBridge to convert raw image data
        self.bridge = CvBridge()

        # Using pre-trained YOLO model for computer vision work
        self.model = YOLO('yolo26n.pt')

        # Construct topics dynamically using f-strings
        image_topic = f'/{self.robot_ns}/oakd/rgb/preview/image_raw'
        velocity_topic = f'/{self.robot_ns}/cmd_vel'
        visualization_topic = f'/yolo/visualization'

        # ROS infrastructure:
        #   subscribe to camera feed
        #   publish robot movement
        #   publish YOLO-processed image
        self.img_sub = self.create_subscription(Image, image_topic, self.image_callback, 10)
        self.vel_pub = self.create_publisher(TwistStamped, velocity_topic, 10)
        self.viz_pub = self.create_publisher(Image, visualization_topic, 10)
        
        # Persistent state variables
        self.target_person_id = None
        self.last_direction_person_detected = 1.0   # default left turn
        self.last_time_person_detected = None

        # Constant parameters
        self.FORWARD_SPEED = 0.2            # m/s
        self.SEARCH_SPEED = 0.4             # rad/s
        self.SPIN_DURATION = 16             # seconds
        self.HEIGHT_THRESHOLD = 240.0       # pixels
        self.CONFIDENCE_THRESHOLD = 0.5     # percentage

    def image_callback(self, msg):
        # Convert a raw image message into a useful NumPy array
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # A tuple containing dimensions of an image array
        img_height, img_width, color_channels = cv_image.shape
        
        results = self.model.track(
            cv_image,       # Input to analyze
            classes=[0],    # Filter for 'person' class
            persist=True,   # Assign a unique tracking ID
            verbose=False,  # Disable default console printouts
        )

        valid_target = None

        # Check if results exist and have boxes
        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                # Iterate through detections to find first high-confidence person
                confidence = float(box.conf[0].cpu().item()) # box.conf is a tensor
                if confidence >= self.CONFIDENCE_THRESHOLD:
                    valid_target = box
                    break
        
        # Execute tracking if a confident match is found, otherwise search
        if valid_target is not None:
            self.execute_tracking_behavior(valid_target, img_width)
        else:
            self.execute_searching_behavior()
        
        self.publish_visualization(results)

    def publish_velocity(self, linear_vel: float, angular_vel: float):
        """Translates abstract target velocities into ROS 2 network command."""
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'

        twist_msg.twist.linear.x = linear_vel
        twist_msg.twist.angular.z = angular_vel

        self.vel_pub.publish(twist_msg)

    def execute_tracking_behavior(self, target_box, img_width):
        """
            Calculates and updates movement commands (linear and angular velocities) 
            required for robot to follow a detected person, and also saves 
            their last known direction in case they disappear.
        """
        # Person is found, reset timer
        self.last_time_person_detected = None

        # Extract bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = target_box.xyxy[0].cpu().numpy()

        # Steer forward or stop based on height threshold
        target_linear = 0.0 if (y2-y1) >= self.HEIGHT_THRESHOLD else self.FORWARD_SPEED

        # Steer left or right based on angular error
        #   (difference between robot and target orientation)
        # and scaled by an appropriate angular proportional gain
        #   (how aggressive robot should turn)
        anuglar_error = (img_width / 2.0) - ((x1 + x2) / 2.0)   # image center - box center
        target_angular = anuglar_error * 0.005
        
        self.get_logger().info(
            f"TRACKING - Linear: {target_linear:.2f}, Angular: {target_angular:.2f}", 
            throttle_duration_sec=3.0
        )

        # Save last known angular velocity direction before person leave frame
        if abs(target_angular) > 0.05:
            self.last_direction_person_detected = 1.0 if target_angular > 0.0 else -1.0
        
        self.publish_velocity(target_linear, target_angular)
    
    def execute_searching_behavior(self):
        """
            Adjusts angular velocity to spin in direction person exit camera view, and 
            robot would rotate 360 degrees until person is found, otherwise stop completely.
        """
        # Start timing when person is lost
        curr_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
        elapsed_search_time = curr_time - self.last_time_person_detected

        # Spin in place in last known direction person is detected
        if elapsed_search_time < self.SPIN_DURATION:
            target_linear = 0.0
            target_angular = self.last_direction_person_detected * self.SEARCH_SPEED
            status = f"SPINNING ({'LEFT' if self.last_direction_person_detected > 0 else 'RIGHT'})"
        else: # Done searching, completely stop
            target_linear = 0.0
            target_angular = 0.0
            status = "SEARCH TIMEOUT (STOPPED)"
        
        self.get_logger().info(
            f"SEARCHING - State: {status}, Elapsed: {elapsed_search_time:.1f}s", 
            throttle_duration_sec=3.0
        )

        self.publish_velocity(target_linear, target_angular)
    
    def publish_visualization(self, results):
        """Publishes YOLO-processed images to camera view to see what robot sees."""
        annotated_img = results[0].plot()
        msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
        self.viz_pub.publish(msg_out)

def main(args=None):
    rclpy.init(args=args)
    node = HumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()