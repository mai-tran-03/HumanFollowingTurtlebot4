"""
obstacle_avoidance.py
---------------------
LiDAR-based obstacle avoidance for the human-following robot.
Fixed with physical width clearance calculations to prevent wedging into small spaces.
"""

import math
import numpy as np
from sensor_msgs.msg import LaserScan


class ObstacleAvoider:
    # ------------------------------------------------------------------ #
    #  Tunable parameters                                                  #
    # ------------------------------------------------------------------ #
    OBSTACLE_DIST_THRESH  = 0.50   # metres – stop/avoid if object closer than this
    FORWARD_HALF_ARC_DEG  = 30.0   # degrees each side of dead-ahead to check
    SIDE_SCAN_HALF_ARC    = 75.0   # degrees each side to search for a free gap
    
    # --- PHYSICAL WIDTH SAFETY CHECK PARAMETERS ---
    ROBOT_DIAMETER_METERS = 0.35   # TurtleBot 4 width is ~34cm
    SAFETY_BUFFER_METERS  = 0.15   # Extra padding so it doesn't scrape edges
    
    AVOIDANCE_ANGULAR_SPD = 0.35   # rad/s used while steering around obstacle
    AVOIDANCE_LINEAR_SPD  = 0.10   # m/s crept forward during avoidance steering
    # ------------------------------------------------------------------ #

    def __init__(self, logger=None):
        self._scan: LaserScan | None = None
        self._log = logger

    def update_scan(self, scan_msg: LaserScan):
        """Call from a LaserScan subscription callback to feed new data."""
        self._scan = scan_msg

    def is_path_blocked(self) -> bool:
        """Return True if an obstacle sits inside the forward arc."""
        if self._scan is None:
            return False
        ranges = self._get_arc_ranges(
            -self.FORWARD_HALF_ARC_DEG, self.FORWARD_HALF_ARC_DEG
        )
        return self._arc_has_obstacle(ranges)

    def get_avoidance_velocities(self, target_angle_deg: float = 0.0) -> tuple[float, float]:
        """
        Return (linear_x, angular_z) to steer around obstacles toward a safe 
        gap that physically accommodates the robot's footprint.
        """
        if self._scan is None:
            return 0.0, 0.0

        best_angle_deg = self._find_best_gap_angle(target_heading=target_angle_deg)

        if best_angle_deg is None:
            if self._log:
                self._log.warning("ObstacleAvoider: NO PHYSICALLY VIABLE GAP FOUND! Safe spin in place.")
            return 0.0, self.AVOIDANCE_ANGULAR_SPD

        angular_z = self._angle_to_angular_cmd(best_angle_deg)
        # Creep forward only when we are roughly pointing at the gap opening
        linear_x = self.AVOIDANCE_LINEAR_SPD if abs(angular_z) < 0.20 else 0.0

        if self._log:
            self._log.info(
                f"ObstacleAvoider: Steering to gap at {best_angle_deg:.1f}° | "
                f"lin={linear_x:.2f} ang={angular_z:.2f}"
            )
        return linear_x, angular_z

    def _get_arc_ranges(self, start_deg: float, end_deg: float) -> list[float]:
        scan = self._scan
        angle_min = math.degrees(scan.angle_min)
        angle_increment = math.degrees(scan.angle_increment)

        result = []
        for i, r in enumerate(scan.ranges):
            angle = angle_min + i * angle_increment
            angle = (angle + 180.0) % 360.0 - 180.0  # Normalise to (-180, 180]
            if start_deg <= angle <= end_deg:
                if math.isfinite(r) and scan.range_min < r < scan.range_max:
                    result.append(r)
        return result

    def _arc_has_obstacle(self, ranges: list[float]) -> bool:
        if not ranges:
            return False
        return min(ranges) < self.OBSTACLE_DIST_THRESH

    def _find_best_gap_angle(self, target_heading: float) -> float | None:
        """
        Finds continuous clear areas and filters them by real physical metric distance width
        rather than just relying on raw angular sizes. 
        """
        scan = self._scan
        angle_min_deg = math.degrees(scan.angle_min)
        angle_inc_deg = math.degrees(scan.angle_increment)
        
        search_start = -self.SIDE_SCAN_HALF_ARC
        search_end = self.SIDE_SCAN_HALF_ARC

        valid_indices = []
        angles = []
        
        for i, r in enumerate(scan.ranges):
            angle = angle_min_deg + i * angle_inc_deg
            angle = (angle + 180.0) % 360.0 - 180.0  
            if search_start <= angle <= search_end:
                valid_indices.append(i)
                angles.append(angle)

        if not valid_indices:
            return None

        # Build true/false blocked mask
        is_blocked = []
        for idx in valid_indices:
            r = scan.ranges[idx]
            if math.isfinite(r) and scan.range_min < r < self.OBSTACLE_DIST_THRESH:
                is_blocked.append(True)
            else:
                is_blocked.append(False)

        # Segment continuous free windows
        gaps = []
        in_gap = False
        gap_start_idx = 0

        for idx, blocked in enumerate(is_blocked):
            if not blocked and not in_gap:
                in_gap = True
                gap_start_idx = idx
            elif blocked and in_gap:
                in_gap = False
                gaps.append((gap_start_idx, idx - 1))
        
        if in_gap:
            gaps.append((gap_start_idx, len(is_blocked) - 1))

        # Evaluate physical clearance size for each window
        usable_gaps = []
        required_clearance = self.ROBOT_DIAMETER_METERS + self.SAFETY_BUFFER_METERS

        for start_idx, end_idx in gaps:
            start_angle = angles[start_idx]
            end_angle = angles[end_idx]
            angular_width_deg = end_angle - start_angle
            
            # Find the closest bounding obstacle distance in or around this gap
            # to calculate the worst-case narrowest physical pass-through corridor.
            gap_range_readings = []
            for k in range(start_idx, end_idx + 1):
                r_val = scan.ranges[valid_indices[k]]
                if math.isfinite(r_val) and r_val < scan.range_max:
                    gap_range_readings.append(r_val)
            
            # If no ranges present or it's completely wide open, assume maximum visibility distance
            avg_distance = min(gap_range_readings) if gap_range_readings else self.OBSTACLE_DIST_THRESH
            
            # Chord length physical opening width math equation: 2 * D * sin(angular_width / 2)
            physical_width_meters = 2.0 * avg_distance * math.sin(math.radians(angular_width_deg / 2.0))
            
            if self._log and physical_width_meters < required_clearance:
                # Debug log showing why small gaps are discarded
                pass 

            if physical_width_meters >= required_clearance:
                centre = (start_angle + end_angle) / 2.0
                usable_gaps.append((physical_width_meters, centre))

        if not usable_gaps:
            return None

        # Sort and return the open window closest to the human target vector path
        best_gap = min(usable_gaps, key=lambda g: abs(g[1] - target_heading))
        return best_gap[1]

    def _angle_to_angular_cmd(self, target_angle_deg: float) -> float:
        """Simple proportional controller converting heading error to rad/s."""
        Kp = self.AVOIDANCE_ANGULAR_SPD / self.FORWARD_HALF_ARC_DEG
        cmd = Kp * target_angle_deg
        return float(np.clip(cmd, -self.AVOIDANCE_ANGULAR_SPD, self.AVOIDANCE_ANGULAR_SPD))