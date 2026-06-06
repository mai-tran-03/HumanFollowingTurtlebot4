import cv2
import numpy as np


class HistogramVisualTracker:
    """Helper class to manage human visual profiles using color histograms"""
    def __init__(self):
        self.saved_profile = None

    def _calculate_histogram(self, cv_image, box_coords):
        """Helper to crop image and compute normalized 2D HSV histogram"""
        x1, y1, x2, y2 = map(int, box_coords)
        crop = cv_image[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        # Convert to HSV (Hue/Saturation are resilient to shadows)
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # 8 bins for Hue (colors), 8 bins for Saturation (intensity)
        hist = cv2.calcHist([hsv_crop], [0, 1], None, [8, 8], [0, 180, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def update_profile(self, cv_image, box_coords):
        """Continuously update stored visual fingerprint"""
        try:
            profile = self._calculate_histogram(cv_image, box_coords)
            if profile is not None:
                self.saved_profile = profile
        except Exception: pass

    def matches_profile(self, cv_image, box_coords, threshold=0.4):
        """Compare new bounding box to saved profile using Bhattacharyya distance"""
        if self.saved_profile is None: 
            return False
        try:
            current_profile = self._calculate_histogram(cv_image, box_coords)
            if current_profile is None: 
                return False

            # Lower score = closer match in Bhattacharyya distance (0 to 1)
            distance = cv2.compareHist(
                self.saved_profile, current_profile, cv2.HISTCMP_BHATTACHARYYA)
            return distance < threshold
        except Exception: 
            return False

    def get_histogram_image(self):
        """Generates visual 128x128 pixel image representation of stored histogram"""
        if self.saved_profile is None:
            return np.zeros((128, 128, 3), dtype=np.uint8) # Output empty dark square
        
        # Reshape flat array back to 8x8 grid
        hist_grid = self.saved_profile.reshape(8, 8)
        
        # Normalize to 0-255 scaling for display
        hist_img = np.uint8(np.clip(hist_grid * 255 * 5, 0, 255)) 
        
        # Resize to visible block size (128x128 pixels)
        resized_hist = cv2.resize(hist_img, (128, 128), interpolation=cv2.INTER_NEAREST)
        
        # Convert grayscale heatmap to a colorized layout
        color_mapped = cv2.applyColorMap(resized_hist, cv2.COLORMAP_JET)

        return color_mapped