import rclpy
from sensor_msgs.msg import Image

from ..human_follower_node import HumanFollower
from .histogram_visual_tracker import HistogramVisualTracker


class OneHumanFollower(HumanFollower):

    def __init__(self):
        # Initialize the base ROS 2 HumanFollower node
        super().__init__()
        self.get_logger().info(
            "Extending node with HistogramVisualTracker..."
        )

        # 1. Compose the helper tracker into this class
        self.tracker = HistogramVisualTracker()

        # 2. Setup an additional ROS publisher for the histogram heatmap
        hist_viz_topic = f'/yolo/histogram_profile'
        self.hist_viz_pub = self.create_publisher(Image, hist_viz_topic, 10)

    def image_callback(self, msg):
        """Overrides the base image callback to add color-profile filtering."""
        # Convert raw ROS image to OpenCV format
        cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        img_height, img_width, _ = cv_image.shape

        # Run the base YOLO tracking model
        results = self.model.track(cv_image, classes=[0], persist=True, verbose=False)

        valid_target = None

        if results and len(results[0].boxes) > 0:
            for box in results[0].boxes:
                confidence = float(box.conf[0].cpu().item())
                if confidence < self.CONFIDENCE_THRESHOLD:
                    continue

                # Extract bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_coords = [x1, y1, x2, y2]

                # Extension Logic:
                # No profile yet, lock onto the first person seen
                if self.tracker.saved_profile is None:
                    self.tracker.update_profile(cv_image, box_coords)
                    valid_target = box
                    self.get_logger().info("Initialized color profile lock!")
                    break

                # Person profile exist, verify if this YOLO box matches it
                elif self.tracker.matches_profile(
                    cv_image, box_coords, threshold=0.45
                ):
                    # Update profile dynamically to handle slight lighting changes
                    self.tracker.update_profile(cv_image, box_coords)
                    valid_target = box
                    break

        # Execute behaviors based on our advanced filtering
        if valid_target is not None:
            self.execute_tracking_behavior(valid_target, img_width)
        else:
            # If the specific person is lost (even if YOLO sees other people), search!
            self.execute_searching_behavior()

        self.publish_visualization(results)

        # Extension Logic: 
        # Publish the color histogram heatmap stream
        self.publish_histogram_visualization()

    def publish_histogram_visualization(self):
        """Generates and publishes the 8x8 color map as a ROS 2 Image."""
        hist_img = self.tracker.get_histogram_image()
        msg_out = self.bridge.cv2_to_imgmsg(hist_img, "bgr8")
        self.hist_viz_pub.publish(msg_out)


def main(args=None):
    rclpy.init(args=args)
    node = OneHumanFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()