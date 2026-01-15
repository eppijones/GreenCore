"""
Color-based ball tracker module for standard webcams.

Detects and tracks a white golf ball using RGB/HSV color detection.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple
from collections import deque

import numpy as np
import cv2

from .calibration import Calibration


@dataclass
class TrackingResult:
    """Result of ball tracking for a single frame."""
    # Detection status
    detected: bool = False
    
    # Position in pixel coordinates (within full frame)
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    
    # Position in world coordinates (meters, relative to ROI center)
    world_x: float = 0.0
    world_y: float = 0.0
    
    # Depth in meters (estimated or fixed for webcam)
    depth: float = 0.5  # Default estimate for webcam
    
    # Detection quality
    circularity: float = 0.0
    area: float = 0.0
    
    # Timing
    timestamp: float = 0.0
    frame_number: int = 0
    
    # Velocity (computed from trajectory, meters per second)
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    @property
    def speed(self) -> float:
        """Get speed magnitude in m/s."""
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    @property
    def velocity_angle_rad(self) -> float:
        """Get velocity direction in radians."""
        return math.atan2(self.velocity_y, self.velocity_x)


@dataclass
class ColorTrackerConfig:
    """Configuration for color-based ball tracker."""
    # Ball detection parameters - tuned for golf ball (~43mm diameter)
    min_area: int = 300  # Minimum contour area in pixels
    max_area: int = 8000  # Maximum contour area in pixels
    min_circularity: float = 0.65  # Minimum circularity (1.0 = perfect circle)
    
    # Golf ball should be roughly circular (aspect ratio close to 1.0)
    max_aspect_ratio: float = 1.5  # Maximum width/height ratio
    
    # Color detection (HSV ranges for white golf ball)
    # White ball appears as very low saturation and high value
    white_h_range: Tuple[int, int] = (0, 180)     # All hues (white is neutral)
    white_s_range: Tuple[int, int] = (0, 50)      # Low saturation (white)
    white_v_range: Tuple[int, int] = (200, 255)   # High value (bright)
    
    # Alternative: brightness threshold for simple detection
    brightness_threshold: int = 210  # Minimum brightness for ball detection
    
    # Image processing
    blur_kernel: int = 5  # Gaussian blur kernel size
    morph_kernel_size: int = 5  # Morphological operation kernel size
    
    # Tracking
    max_trajectory_length: int = 60  # Maximum frames to keep in trajectory
    velocity_smoothing_window: int = 5  # Frames to smooth velocity over
    max_jump_distance: float = 0.5  # Maximum allowed jump between frames (meters)
    
    # Debug
    debug_visualization: bool = True


class ColorBallTracker:
    """
    Color-based golf ball tracker for standard webcams.
    
    Uses color/brightness detection to find white golf balls.
    Supports manual ball selection via click.
    """
    
    def __init__(
        self, 
        calibration: Calibration,
        config: Optional[ColorTrackerConfig] = None
    ):
        """
        Initialize ball tracker.
        
        Args:
            calibration: Calibration data for coordinate conversion.
            config: Tracker configuration.
        """
        self.calibration = calibration
        self.config = config or ColorTrackerConfig()
        
        # Trajectory buffer
        self._trajectory: deque = deque(maxlen=self.config.max_trajectory_length)
        
        # Last valid detection for continuity
        self._last_detection: Optional[TrackingResult] = None
        
        # Manual ball position (set by clicking)
        self._manual_ball_pos: Optional[Tuple[int, int]] = None
        self._tracking_active: bool = False
        
        # Search radius around manual position
        self._search_radius: int = 80
        
        # Debug visualization frame
        self.debug_frame: Optional[np.ndarray] = None
        
        # Processing stats
        self._process_times: deque = deque(maxlen=30)
    
    def set_ball_position(self, x: int, y: int):
        """
        Manually set the ball position (from mouse click).
        
        Args:
            x: X coordinate in full frame.
            y: Y coordinate in full frame.
        """
        self._manual_ball_pos = (x, y)
        self._tracking_active = True
        self._trajectory.clear()
        self._last_detection = None
        print(f"[Tracker] Ball position set to ({x}, {y}) - tracking active")
    
    def stop_tracking(self):
        """Stop tracking and clear ball position."""
        self._tracking_active = False
        self._manual_ball_pos = None
        self._trajectory.clear()
        self._last_detection = None
        print("[Tracker] Tracking stopped")
    
    @property
    def is_tracking(self) -> bool:
        """Check if tracking is active."""
        return self._tracking_active
    
    def update(
        self, 
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int = 0
    ) -> Optional[TrackingResult]:
        """
        Process a frame and update tracking.
        
        Args:
            color_frame: BGR color image.
            gray_frame: Grayscale image.
            timestamp: Frame timestamp in seconds.
            frame_number: Frame sequence number.
            
        Returns:
            TrackingResult if ball detected, None otherwise.
        """
        start_time = time.time()
        
        # If tracking not active, return no detection
        if not self._tracking_active:
            result = TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
            if self.config.debug_visualization:
                self.debug_frame = color_frame[:200, :200].copy() if color_frame.size > 0 else None
            return result
        
        frame_h, frame_w = gray_frame.shape[:2]
        
        # Determine search area - either around manual position or last detection
        if self._manual_ball_pos:
            search_center = self._manual_ball_pos
        elif self._last_detection and self._last_detection.detected:
            search_center = (int(self._last_detection.pixel_x), int(self._last_detection.pixel_y))
        else:
            # No position known - can't track
            result = TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
            return result
        
        # Create search ROI around expected position
        cx, cy = search_center
        r = self._search_radius
        
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(frame_w, cx + r)
        y2 = min(frame_h, cy + r)
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            result = TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
            return result
        
        # Crop search area
        color_roi = color_frame[y1:y2, x1:x2].copy()
        gray_roi = gray_frame[y1:y2, x1:x2].copy()
        roi_offset = (x1, y1)
        
        # Detect ball candidates within search area
        candidates = self._detect_candidates(color_roi, gray_roi)
        
        # Filter and select best candidate
        result = self._select_best_candidate(
            candidates, roi_offset, timestamp, frame_number
        )
        
        # If we found the ball, clear the manual position (now tracking automatically)
        if result and result.detected:
            self._manual_ball_pos = None  # Now track from last detection
            self._compute_velocity(result)
            self._trajectory.append(result)
            self._last_detection = result
        else:
            # Lost tracking - expand search radius temporarily
            if self._search_radius < 150:
                self._search_radius = min(150, self._search_radius + 10)
        
        # Reset search radius when tracking well
        if result and result.detected and self._search_radius > 80:
            self._search_radius = max(80, self._search_radius - 5)
        
        # Track processing time
        process_time = time.time() - start_time
        self._process_times.append(process_time)
        
        # Create debug visualization
        if self.config.debug_visualization:
            self._create_debug_frame(color_roi, gray_roi, candidates, result, roi_offset)
        
        return result
    
    def _detect_candidates(
        self, 
        color_roi: np.ndarray, 
        gray_roi: np.ndarray
    ) -> List[dict]:
        """
        Detect ball candidates in the ROI.
        
        Returns list of candidate dictionaries with position and metrics.
        """
        candidates = []
        
        if gray_roi.size == 0:
            return candidates
        
        # Blur first to reduce noise
        blurred = cv2.GaussianBlur(
            gray_roi, 
            (self.config.blur_kernel, self.config.blur_kernel), 
            0
        )
        
        # Method 1: Simple brightness threshold - most reliable for white ball
        # Use adaptive threshold based on image brightness
        mean_brightness = np.mean(blurred)
        threshold = max(150, min(230, mean_brightness + 40))
        
        _, mask_bright = cv2.threshold(
            blurred, 
            int(threshold), 
            255, 
            cv2.THRESH_BINARY
        )
        
        # Method 2: HSV color detection (white ball)
        if color_roi.size > 0:
            hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
            
            # Create mask for white color (low saturation, high value)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 60, 255])
            mask_hsv = cv2.inRange(hsv, lower_white, upper_white)
            
            # Combine masks - either white OR very bright
            combined_mask = cv2.bitwise_or(mask_hsv, mask_bright)
        else:
            combined_mask = mask_bright
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours from mask
        contours, _ = cv2.findContours(
            combined_mask, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (be more lenient)
            if area < 100 or area > 15000:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Filter by circularity (be more lenient - ball might have shadows)
            if circularity < 0.4:
                continue
            
            # Get centroid using moments
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check aspect ratio (golf ball should be roughly circular)
            aspect_ratio = max(w, h) / max(min(w, h), 1)
            if aspect_ratio > 2.0:
                continue
            
            # Calculate average brightness in the detected region
            mask_region = np.zeros(gray_roi.shape, dtype=np.uint8)
            cv2.drawContours(mask_region, [contour], -1, 255, -1)
            avg_brightness = cv2.mean(gray_roi, mask=mask_region)[0]
            
            candidates.append({
                'cx': cx,
                'cy': cy,
                'area': area,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'contour': contour,
                'bbox': (x, y, w, h),
                'brightness': avg_brightness,
                'hough_match': False,
            })
        
        return candidates
    
    def _select_best_candidate(
        self,
        candidates: List[dict],
        roi_offset: Tuple[int, int],
        timestamp: float,
        frame_number: int
    ) -> Optional[TrackingResult]:
        """
        Select the best candidate based on tracking continuity and quality.
        """
        if not candidates:
            return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
        
        best_candidate = None
        best_score = -1.0
        
        for candidate in candidates:
            score = self._score_candidate(candidate, roi_offset)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        if best_candidate is None:
            return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
        
        # Convert to full frame coordinates
        pixel_x = best_candidate['cx'] + roi_offset[0]
        pixel_y = best_candidate['cy'] + roi_offset[1]
        
        # Convert to world coordinates (meters)
        # Origin at ROI center, +X right, +Y down in image (forward in world)
        roi = self.calibration.data.roi
        if roi:
            roi_center_x = roi[0] + roi[2] / 2
            roi_center_y = roi[1] + roi[3] / 2
        else:
            roi_center_x = pixel_x
            roi_center_y = pixel_y
        
        world_x = self.calibration.pixels_to_meters(pixel_x - roi_center_x)
        world_y = self.calibration.pixels_to_meters(pixel_y - roi_center_y)
        
        return TrackingResult(
            detected=True,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            world_x=world_x,
            world_y=world_y,
            depth=self.calibration.data.ground_depth,  # Use calibrated depth
            circularity=best_candidate['circularity'],
            area=best_candidate['area'],
            timestamp=timestamp,
            frame_number=frame_number,
        )
    
    def _score_candidate(self, candidate: dict, roi_offset: Tuple[int, int]) -> float:
        """
        Score a candidate based on quality and tracking continuity.
        
        Higher score = better candidate.
        """
        score = 0.0
        
        # Circularity score (0-1) - most important for ball detection
        score += candidate['circularity'] * 3.0
        
        # Brightness score (brighter = more likely golf ball)
        brightness_score = min(1.0, candidate['brightness'] / 255.0)
        score += brightness_score * 1.0
        
        # Aspect ratio - closer to 1.0 is better (circular)
        aspect_score = 1.0 / max(candidate.get('aspect_ratio', 1.0), 1.0)
        score += aspect_score * 2.0
        
        # Hough circle match bonus
        if candidate.get('hough_match', False):
            score += 2.0
        
        # Prefer candidates near last detection (if we have one)
        if self._last_detection and self._last_detection.detected:
            cx = candidate['cx'] + roi_offset[0]
            cy = candidate['cy'] + roi_offset[1]
            
            dx = cx - self._last_detection.pixel_x
            dy = cy - self._last_detection.pixel_y
            distance = math.sqrt(dx*dx + dy*dy)
            
            # Convert to meters
            distance_m = self.calibration.pixels_to_meters(distance)
            
            # Reject if too far (unlikely to be same ball)
            if distance_m > self.config.max_jump_distance:
                return -1.0
            
            # Closer = better (exponential decay)
            proximity_score = math.exp(-distance_m * 5)
            score += proximity_score * 3.0
        
        return score
    
    def _compute_velocity(self, result: TrackingResult):
        """
        Compute velocity from trajectory history.
        
        Uses linear regression over recent frames for smoothness.
        """
        if len(self._trajectory) < 2:
            return
        
        # Get recent trajectory points
        window = min(self.config.velocity_smoothing_window, len(self._trajectory))
        recent = list(self._trajectory)[-window:]
        
        if len(recent) < 2:
            return
        
        # Calculate velocity using first and last point in window
        first = recent[0]
        last = recent[-1]
        
        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return
        
        dx = last.world_x - first.world_x
        dy = last.world_y - first.world_y
        
        result.velocity_x = dx / dt
        result.velocity_y = dy / dt
    
    def _create_debug_frame(
        self,
        color_roi: np.ndarray,
        gray_roi: np.ndarray,
        candidates: List[dict],
        result: Optional[TrackingResult],
        roi_offset: Tuple[int, int]
    ):
        """Create debug visualization frame."""
        # Use color frame for visualization
        debug = color_roi.copy()
        
        # Draw all candidates in yellow
        for candidate in candidates:
            cv2.drawContours(debug, [candidate['contour']], -1, (0, 255, 255), 1)
            cx, cy = candidate['cx'], candidate['cy']
            cv2.circle(debug, (cx, cy), 3, (0, 255, 255), -1)
        
        # Draw selected detection in green
        if result and result.detected:
            # Convert back to ROI coordinates
            rx = int(result.pixel_x - roi_offset[0])
            ry = int(result.pixel_y - roi_offset[1])
            
            cv2.circle(debug, (rx, ry), 15, (0, 255, 0), 2)
            cv2.circle(debug, (rx, ry), 3, (0, 255, 0), -1)
            
            # Draw velocity vector
            if result.speed > 0.01:
                vx = result.velocity_x * 50  # Scale for visibility
                vy = result.velocity_y * 50
                end_x = int(rx + vx)
                end_y = int(ry + vy)
                cv2.arrowedLine(debug, (rx, ry), (end_x, end_y), (255, 0, 0), 2)
        
        # Draw trajectory
        if len(self._trajectory) > 1:
            points = []
            for tr in self._trajectory:
                if tr.detected:
                    px = int(tr.pixel_x - roi_offset[0])
                    py = int(tr.pixel_y - roi_offset[1])
                    points.append((px, py))
            
            for i in range(1, len(points)):
                cv2.line(debug, points[i-1], points[i], (255, 0, 255), 1)
        
        self.debug_frame = debug
    
    def get_trajectory(self) -> List[TrackingResult]:
        """Get current trajectory buffer."""
        return list(self._trajectory)
    
    def reset(self):
        """Reset tracker state."""
        self._trajectory.clear()
        self._last_detection = None
        self._manual_ball_pos = None
        self._tracking_active = False
        self._search_radius = 80
        self.debug_frame = None
        print("[Tracker] Reset - click on ball to start tracking")
    
    @property
    def has_detection(self) -> bool:
        """Check if we have a current detection."""
        if not self._trajectory:
            return False
        return self._trajectory[-1].detected
    
    @property
    def current_speed(self) -> float:
        """Get current ball speed in m/s."""
        if self._trajectory and self._trajectory[-1].detected:
            return self._trajectory[-1].speed
        return 0.0
    
    @property
    def average_process_time(self) -> float:
        """Get average processing time in ms."""
        if self._process_times:
            return sum(self._process_times) / len(self._process_times) * 1000
        return 0.0
