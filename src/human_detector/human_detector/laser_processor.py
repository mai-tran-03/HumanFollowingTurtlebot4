import numpy as np

class LaserProcessor:
    def __init__(self, avoid_distance=0.6, hard_stop_distance=0.35):
        self.AVOID_DISTANCE_THRESH = avoid_distance
        self.HARD_STOP_DISTANCE = hard_stop_distance
        
        # State outputs updated per scan cycle
        self.obstacle_detected = False
        self.obstacle_in_way = False
        self.avoidance_steering_bias = 0.0

    def process_scan(self, msg, logger=None):
        """
        Parses raw LaserScan messages to determine proximity risks and calculation biases.
        """
        # Split front beams (approx 15-20 degrees left and right)
        front_left_beams = msg.ranges[:20]
        front_right_beams = msg.ranges[-20:]
        
        # Filter out infinite/error/out-of-bound readings
        valid_left = [r for r in front_left_beams if msg.range_min < r < msg.range_max and not np.isnan(r)]
        valid_right = [r for r in front_right_beams if msg.range_min < r < msg.range_max and not np.isnan(r)]        

        # Reset states for the current processing sweep
        self.obstacle_detected = False
        self.obstacle_in_way = False
        self.avoidance_steering_bias = 0.0

        all_front_beams = valid_left + valid_right
        
        # 1. Emergency Hard Stop Evaluation
        if all_front_beams and min(all_front_beams) < self.HARD_STOP_DISTANCE:
            self.obstacle_detected = True
            if logger:
                logger.warn("LaserProcessor: Hard stop triggered!")
            return self.obstacle_detected, self.obstacle_in_way, self.avoidance_steering_bias

        # 2. Smart Steering Proximity Sweep
        min_left = min(valid_left) if valid_left else float('inf')
        min_right = min(valid_right) if valid_right else float('inf')

        if min_left < self.AVOID_DISTANCE_THRESH or min_right < self.AVOID_DISTANCE_THRESH:
            self.obstacle_in_way = True
            
            if min_left < min_right:
                # Obstacle closer on LEFT -> Steer RIGHT (Negative Angular Z)
                self.avoidance_steering_bias = -0.5 * (1.0 - (min_left / self.AVOID_DISTANCE_THRESH))
                if logger:
                    logger.info("LaserProcessor: Obstacle LEFT -> Nudging Right")
            else:
                # Obstacle closer on RIGHT -> Steer LEFT (Positive Angular Z)
                self.avoidance_steering_bias = 0.5 * (1.0 - (min_right / self.AVOID_DISTANCE_THRESH))
                if logger:
                    logger.info("LaserProcessor: Obstacle RIGHT -> Nudging Left")
        else:
            if logger:
                logger.info("LaserProcessor: Paths Clear. Flags Reset.")

        return self.obstacle_detected, self.obstacle_in_way, self.avoidance_steering_bias