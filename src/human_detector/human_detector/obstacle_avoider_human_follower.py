from ultralytics import YOLO
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
import cv2
import math
from .visual_tracker import VisualTracker
from .obstacle_avoidance import ObstacleAvoider


class ObstacleAvoiderHumanFollower(Node):
    def __init__(self):
        super().__init__('human_follower')
        self.bridge = CvBridge()
        self.model = YOLO('yolo26n.pt')
        self.v_tracker = VisualTracker()
        self.obstacle_avoider = ObstacleAvoider(logger=self.get_logger())

    
        # Inputs
        self.img_sub = self.create_subscription(
            Image, '/don/oakd/rgb/preview/image_raw', self.image_callback, 10)
        self.scan_sub = self.create_subscription(
            LaserScan, '/don/scan', self.scan_callback, 10)

        # Outputs
        self.vel_pub = self.create_publisher(TwistStamped, '/don/cmd_vel', 10)
        self.viz_pub = self.create_publisher(Image, '/yolo/visualization', 10)

   
        self.target_person_id = None
        self.target_acquired = False   
        self.last_direction_person_detected = 0.0
        self.last_time_person_detected = None


        self.SEARCH_SPEED       = 0.4
        self.SPIN_DURATION      = 16
        self.GRACE_PERIOD       = 1.5
        self.STOP_HEIGHT_THRESH = 240.0

    # ---------------------------------------------------------------------- #
    # Callbacks                                                                #
    # ---------------------------------------------------------------------- #

    def scan_callback(self, msg: LaserScan):
        """Feed the latest LiDAR scan to the obstacle avoider."""
        self.obstacle_avoider.update_scan(msg)

    def image_callback(self, msg: Image):
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_width = cv_image.shape[1]

        results = self.model.track(
            cv_image,
            classes=[0],   # person class only
            persist=True,
            verbose=False,
        )

        twist_msg = self.create_empty_twist()
        target_box = self.evaluate_tracking_states(results, cv_image)

        if target_box is not None:
            self.last_time_person_detected = None  # reset loss timer

            # 1. Extract Target Position Data
            box_coords = target_box.xyxy[0].cpu().numpy()
            center_x   = float((box_coords[0] + box_coords[2]) / 2.0)
            box_height = float(box_coords[3] - box_coords[1])

            # Keep visual profile memory running every frame
            self.v_tracker.update_profile(cv_image, box_coords)

            # Calculate error angle relative to camera center axis
            error_x = center_x - (img_width / 2.0)
            
            # Assuming ~60 degree horizontal camera FOV field for calculations
            human_angle_deg = float(-error_x / (img_width / 2.0)) * 30.0

            # Save direction fallback for searching behaviors
            if abs(error_x) > 15:
                self.last_direction_person_detected = -1.0 if error_x > 0 else 1.0

            # 2. Coordinate Navigation Decision via Target-Aware States
            if self.obstacle_avoider.is_path_blocked():
                self.get_logger().warning("Path blocked! Running physical footprint gap routing.")
                linear_x, angular_z = self.obstacle_avoider.get_avoidance_velocities(target_angle_deg=human_angle_deg)
                twist_msg.twist.linear.x  = linear_x
                twist_msg.twist.angular.z = angular_z
            else:
                # Clear path tracking behavior
                twist_msg.twist.angular.z = float(-error_x / 200.0)
                twist_msg.twist.linear.x  = 0.0 if box_height >= self.STOP_HEIGHT_THRESH else 0.2
        else:
            # Human out of frame entirely -> Search behavior
            self.execute_searching_behavior(twist_msg)

        self.vel_pub.publish(twist_msg)
        self.publish_visualization(results)

    # ---------------------------------------------------------------------- #
    # Helpers                                                                  #
    # ---------------------------------------------------------------------- #

    def create_empty_twist(self) -> TwistStamped:
        twist_msg = TwistStamped()
        twist_msg.header.stamp = self.get_clock().now().to_msg()
        twist_msg.header.frame_id = 'base_link'
        return twist_msg

    def evaluate_tracking_states(self, results, cv_image):
        if not results or results[0].boxes is None or results[0].boxes.id is None:
            return None

        boxes = results[0].boxes
        ids = boxes.id.int().cpu().tolist()

        if not self.target_acquired:
            if len(ids) > 0:
                self.target_person_id = ids[0]
                self.target_acquired = True
                self.get_logger().info(f"Fresh Target Lock on ID: {self.target_person_id}")
                return boxes[0]
            return None

        if self.target_person_id in ids:
            target_idx = ids.index(self.target_person_id)
            return boxes[target_idx]

        for idx, current_id in enumerate(ids):
            coords = boxes[idx].xyxy[0].cpu().numpy()
            if self.v_tracker.matches_profile(cv_image, coords):
                self.target_person_id = current_id
                self.get_logger().info(f"Re-ID Success! Re-locked to original target ID: {current_id}")
                return boxes[idx]
        return None

    def execute_searching_behavior(self, twist_msg):
        twist_msg.twist.linear.x = 0.0
        curr_time = self.get_clock().now().nanoseconds / 1e9

        if self.last_time_person_detected is None:
            self.last_time_person_detected = curr_time
            self.get_logger().warning("Target lost, starting 360° search")

        elapsed = curr_time - self.last_time_person_detected

        if elapsed < self.GRACE_PERIOD:
            twist_msg.twist.angular.z = 0.0
        elif elapsed < (self.GRACE_PERIOD + self.SPIN_DURATION):
            if self.last_direction_person_detected == 0.0:
                self.last_direction_person_detected = 1.0  # Default left spin fallback
            twist_msg.twist.angular.z = self.last_direction_person_detected * self.SEARCH_SPEED
            self.get_logger().info("Searching… 360° spin")
        else:
            twist_msg.twist.angular.z = 0.0
            self.target_person_id = None
            self.target_acquired = False   
            self.get_logger().error("Stop searching. Target completely lost. Restart.")

    def publish_visualization(self, results):
        annotated_img = results[0].plot()
        hist_overlay = self.v_tracker.get_histogram_image()
        h, w, _ = hist_overlay.shape
        annotated_img[20:20 + h, 20:20 + w] = hist_overlay

        cv2.putText(annotated_img, "Target Identity Profile", (20, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        cv2.rectangle(annotated_img, (20, 20), (20 + w, 20 + h), (0, 255, 0), 2)

        msg_out = self.bridge.cv2_to_imgmsg(annotated_img, 'bgr8')
        self.viz_pub.publish(msg_out)
