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
        self.curr_linear_vel = 0.0
        self.curr_angular_vel = 0.0
        
        # Constant parameters
        self.FORWARD_SPEED = 0.2            # rad/s
        self.SEARCH_SPEED = 0.4             # rad/s
        self.SPIN_DURATION = 16             # seconds
        self.GRACE_PERIOD = 1.5             # seconds
        self.HEIGHT_THRESHOLD = 240.0       # pixels

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

        # twist_msg = self.create_empty_twist()
        # target_box = self.evaluate_tracking_states(results, cv_image)
        
        # if target_box is not None:
        if results and results[0].boxes:
            self.execute_tracking_behavior(results[0].boxes[0], img_width)
        else:
            self.execute_searching_behavior()
        
        # self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)

    # def create_empty_twist(self):
    #     """Returns a pre-configured, zero-velocity TwistStamped ROS 2 message."""
    #     twist_msg = TwistStamped()
    #     twist_msg.header.stamp = self.get_clock().now().to_msg()
    #     twist_msg.header.frame_id = 'base_link'

    #     self.linear_velocity = twist_msg.twist.linear.x
    #     self.angular_velocity = twist_msg.twist.angular.z
    #     return twist_msg
    
    def publish_velocityy(self, linear_vel: float, angular_vel: float):
        """Updates internal velocity tracking state, builds and publishes message."""
        self.curr_linear_vel = linear_vel
        self.curr_angular_vel = angular_vel
        
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'

        twist_msg.twist.linear.x = linear_vel
        twist_msg.twist.angular.z = angular_vel

        self.vel_pub.publish(twist_msg)

    # def evaluate_tracking_states(self, results, cv_image):
    #     """Process YOLO trackers and handle target confirmation"""
    #     if not results or results[0].boxes is None or results[0].boxes.id is None:
    #         return None
        
    #     boxes = results[0].boxes
    #     ids = boxes.id.int().cpu().tolist()
        
    #     # Target ID exists, look for it in current detections
    #     if self.target_person_id in ids:
    #         target_idx = ids.index(self.target_person_id)
    #         coords = boxes[target_idx].xyxy[0].cpu().numpy()
    #         self.v_tracker.update_profile(cv_image, coords)
    #         return boxes[target_idx]
        
    #     # No target person, try to Re-Identify based on visual profile
    #     for idx, current_id in enumerate(ids):
    #         coords = boxes[idx].xyxy[0].cpu().numpy()
    #         if self.v_tracker.matches_profile(cv_image, coords):
    #             self.target_person_id = current_id
    #             return boxes[idx]

    #     # No target person and someone new show up
    #     if self.target_person_id is None and len(ids) > 0:
    #         self.target_person_id = ids[0]
    #         coords = boxes[0].xyxy[0].cpu().numpy()
    #         self.v_tracker.update_profile(cv_image, coords)
    #         return boxes[0]

    #     return None

    def execute_tracking_behavior(self, target_box, img_width, twist_msg):
        # Person is found, reset timer
        self.last_time_person_detected = None

        # Extract bounding box coordinates (xyxy format)
        x1, y1, x2, y2 = target_box.xyxy[0].cpu().numpy()

        # Steer forward or stop based on height threshold
        # if (y2 - y1) >= self.HEIGHT_THRESHOLD:
        #     # twist_msg.twist.linear.x = 0.0 
        #     self.linear_velocity = 0.0
        # else: 
        #     # self.linear_velocity  = self.FORWARD_SPEED
        #     self.linear_velocity = self.FORWARD_SPEED

        if (y2-y1) >= self.HEIGHT_THRESHOLD:
            self.publish_velocity(0.0, self.curr_angular_vel)
        else:
            self.publish_velocityy(self.FORWARD_SPEED, self.curr_angular_vel)

        # Steer left or right based on angular error
        #   (difference between robot and target orientation)
        # and scaled by an appropriate angular proportional gain
        #   (how aggressive robot should turn)
        anuglar_error = (img_width / 2.0) - (x1 + x2 / 2.0)
        # twist_msg.twist.angular.z = anuglar_error * 0.005
        new_angular_vel = anuglar_error * 0.005
        self.publish_velocityy(self.curr_linear_vel, new_angular_vel)
        
        # Save last known angular velocity direction before person leave frame
        # if abs(twist_msg.twist.angular.z) > 0.05:
        #     if twist_msg.twist.angular.z > 0:
        if abs(self.curr_angular_vel > 0.05):
            if self.curr_angular_vel > 0.0:
                self.last_direction_person_detected = 1.0
            else:
                self.last_direction_person_detected = -1.0

    def execute_searching_behavior(self, twist_msg):
        # twist_msg.twist.linear.x = 0.0
        # Person is lost, robot stops
        self.publish_velocity(0.0, self.curr_angular_vel)
        
        # Start timing when person is lost
        curr_time = self.get_clock().now().nanoseconds / 1e9
        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
        elapsed_search_time = curr_time - self.last_time_person_detected

        # Stand still for a few seconds to let YOLO recover target ID
        if elapsed_search_time < self.GRACE_PERIOD:
            self.publish_velocity(self.curr_linear_vel, 0.0)
        
        # Spin in direction person last seen
        elif elapsed_search_time < (self.GRACE_PERIOD + self.SPIN_DURATION):
            new_angular_vel = self.last_direction_person_detected * self.SEARCH_SPEED
            self.publish_velocityy(self.curr_linear_vel, new_angular_vel)
        
        else:
            self.publish_velocityy(0.0, 0.0)
            # self.target_person_id = None
    
    def publish_visualization(self, results):
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