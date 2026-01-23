"""
Ball tracker module using classical computer vision.

Detects and tracks a golf ball using IR imagery with depth validation.
Supports ArUco-based homography for accurate pixel→world coordinate mapping.
"""

import math
import time
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING
from collections import deque

import numpy as np
import cv2

from .calibration import Calibration

# Import ArucoCalibration for type hints (avoid circular imports)
if TYPE_CHECKING:
    from .calibration import ArucoCalibration


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
    
    # Depth in meters
    depth: float = 0.0
    
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
class TrackerConfig:
    """Configuration for ball tracker."""
    # Ball detection parameters
    min_area: int = 100  # Minimum contour area in pixels
    max_area: int = 5000  # Maximum contour area in pixels
    min_circularity: float = 0.6  # Minimum circularity (1.0 = perfect circle)
    
    # Image processing
    blur_kernel: int = 5  # Gaussian blur kernel size
    threshold_block_size: int = 21  # Adaptive threshold block size
    threshold_c: int = 5  # Adaptive threshold constant
    morph_kernel_size: int = 5  # Morphological operation kernel size
    
    # Tracking
    max_trajectory_length: int = 60  # Maximum frames to keep in trajectory
    velocity_smoothing_window: int = 5  # Frames to smooth velocity over
    max_jump_distance: float = 0.3  # Maximum allowed jump between frames (meters)
    
    # Depth validation
    validate_depth: bool = True
    
    # Debug
    debug_visualization: bool = True


class BallTracker:
    """
    Classical CV-based golf ball tracker.
    
    Uses IR imagery for detection due to high contrast with golf balls.
    Pipeline:
    1. Crop to ROI
    2. Gaussian blur
    3. Adaptive thresholding
    4. Morphological operations
    5. Contour detection
    6. Filter by area, circularity, and depth
    7. Track over time with trajectory buffer
    
    Coordinate Transformation:
    - Uses ArUco homography when available (accurate across entire mat)
    - Falls back to simple pixels_per_meter scaling if no homography
    """
    
    def __init__(
        self, 
        calibration: Calibration,
        config: Optional[TrackerConfig] = None,
        aruco_calibration: Optional['ArucoCalibration'] = None
    ):
        """
        Initialize ball tracker.
        
        Args:
            calibration: Calibration data for coordinate conversion.
            config: Tracker configuration.
            aruco_calibration: Optional ArucoCalibration for homography-based transforms.
        """
        self.calibration = calibration
        self.config = config or TrackerConfig()
        self._aruco_calibration = aruco_calibration
        
        # Trajectory buffer
        self._trajectory: deque = deque(maxlen=self.config.max_trajectory_length)
        
        # Last valid detection for continuity
        self._last_detection: Optional[TrackingResult] = None
        
        # Debug visualization frame
        self.debug_frame: Optional[np.ndarray] = None
        
        # Processing stats
        self._process_times: deque = deque(maxlen=30)
    
    def set_aruco_calibration(self, aruco_calibration: 'ArucoCalibration'):
        """
        Set the ArUco calibration for homography-based coordinate transforms.
        
        Args:
            aruco_calibration: ArucoCalibration instance with valid homography.
        """
        self._aruco_calibration = aruco_calibration
        if aruco_calibration and aruco_calibration.is_ready:
            print("[BallTracker] Using ArUco homography for coordinate transforms")
        else:
            print("[BallTracker] ArUco calibration set but not ready, using legacy scaling")
    
    def _pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to world coordinates.
        
        Uses homography if available, otherwise falls back to legacy scaling.
        
        Args:
            pixel_x: X coordinate in pixels (full frame).
            pixel_y: Y coordinate in pixels (full frame).
            
        Returns:
            (world_x, world_y) in meters.
        """
        # Use ArUco homography if available
        if self._aruco_calibration and self._aruco_calibration.is_ready:
            return self._aruco_calibration.pixel_to_world(pixel_x, pixel_y)
        
        # Fallback to legacy ROI-center based scaling
        roi = self.calibration.data.roi
        if roi:
            roi_center_x = roi[0] + roi[2] / 2
            roi_center_y = roi[1] + roi[3] / 2
        else:
            roi_center_x = pixel_x
            roi_center_y = pixel_y
        
        world_x = self.calibration.pixels_to_meters(pixel_x - roi_center_x)
        world_y = self.calibration.pixels_to_meters(pixel_y - roi_center_y)
        
        return (world_x, world_y)
    
    def _pixel_distance_to_meters(self, pixel_distance: float, pixel_x: float = 0, pixel_y: float = 0) -> float:
        """
        Convert a pixel distance to meters.
        
        For homography, this is approximate since scale varies across the image.
        Uses the local Jacobian at the given position for accuracy.
        
        Args:
            pixel_distance: Distance in pixels.
            pixel_x: Reference X position for local scale (optional).
            pixel_y: Reference Y position for local scale (optional).
            
        Returns:
            Approximate distance in meters.
        """
        # Use ArUco homography if available - compute local scale from Jacobian
        if self._aruco_calibration and self._aruco_calibration.is_ready:
            # Sample nearby points to estimate local scale
            delta = max(1.0, pixel_distance / 10)  # Small delta for numerical gradient
            
            # Get world coords at reference point and offset
            w0 = self._aruco_calibration.pixel_to_world(pixel_x, pixel_y)
            w1 = self._aruco_calibration.pixel_to_world(pixel_x + delta, pixel_y)
            w2 = self._aruco_calibration.pixel_to_world(pixel_x, pixel_y + delta)
            
            # Compute local scale (average of x and y directions)
            scale_x = math.sqrt((w1[0] - w0[0])**2 + (w1[1] - w0[1])**2) / delta
            scale_y = math.sqrt((w2[0] - w0[0])**2 + (w2[1] - w0[1])**2) / delta
            local_scale = (scale_x + scale_y) / 2
            
            return pixel_distance * local_scale
        
        # Fallback to legacy scaling
        return self.calibration.pixels_to_meters(pixel_distance)
    
    def update(
        self, 
        ir_frame: np.ndarray, 
        depth_frame: np.ndarray,
        timestamp: float,
        frame_number: int = 0,
        depth_scale: float = 0.001
    ) -> Optional[TrackingResult]:
        """
        Process a frame and update tracking.
        
        Args:
            ir_frame: Grayscale IR image (uint8).
            depth_frame: Depth image (uint16).
            timestamp: Frame timestamp in seconds.
            frame_number: Frame sequence number.
            depth_scale: Depth unit to meters conversion.
            
        Returns:
            TrackingResult if ball detected, None otherwise.
        """
        start_time = time.time()
        
        # Get ROI
        roi = self.calibration.data.roi
        if roi:
            x, y, w, h = roi
            ir_roi = ir_frame[y:y+h, x:x+w]
            depth_roi = depth_frame[y:y+h, x:x+w]
            roi_offset = (x, y)
        else:
            ir_roi = ir_frame
            depth_roi = depth_frame
            roi_offset = (0, 0)
        
        # Detect ball candidates
        candidates = self._detect_candidates(ir_roi, depth_roi, depth_scale)
        
        # Filter and select best candidate
        result = self._select_best_candidate(
            candidates, roi_offset, depth_scale, timestamp, frame_number
        )
        
        # Compute velocity if we have enough trajectory
        if result and result.detected:
            self._compute_velocity(result)
            self._trajectory.append(result)
            self._last_detection = result
        
        # Track processing time
        process_time = time.time() - start_time
        self._process_times.append(process_time)
        
        # Create debug visualization
        if self.config.debug_visualization:
            self._create_debug_frame(ir_roi, candidates, result, roi_offset)
        
        return result
    
    def _detect_candidates(
        self, 
        ir_roi: np.ndarray, 
        depth_roi: np.ndarray,
        depth_scale: float
    ) -> List[dict]:
        """
        Detect ball candidates in the ROI.
        
        Returns list of candidate dictionaries with position and metrics.
        """
        candidates = []
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(
            ir_roi, 
            (self.config.blur_kernel, self.config.blur_kernel), 
            0
        )
        
        # Adaptive threshold - golf balls appear bright in IR
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            self.config.threshold_block_size,
            -self.config.threshold_c  # Negative = detect bright objects
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (self.config.morph_kernel_size, self.config.morph_kernel_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            binary, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Analyze each contour
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area
            if area < self.config.min_area or area > self.config.max_area:
                continue
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            # Filter by circularity
            if circularity < self.config.min_circularity:
                continue
            
            # Get centroid using moments
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Get depth at centroid (use average in small region)
            depth_value = self._get_median_depth(depth_roi, cx, cy, radius=5)
            depth_meters = depth_value * depth_scale
            
            # Validate depth if enabled
            if self.config.validate_depth:
                if not self.calibration.is_valid_depth(depth_meters):
                    continue
            
            # Compute bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            candidates.append({
                'cx': cx,
                'cy': cy,
                'area': area,
                'circularity': circularity,
                'depth': depth_meters,
                'contour': contour,
                'bbox': (x, y, w, h),
            })
        
        return candidates
    
    def _get_median_depth(
        self, 
        depth_frame: np.ndarray, 
        x: int, 
        y: int, 
        radius: int = 3
    ) -> float:
        """Get median depth value in a small region."""
        h, w = depth_frame.shape
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        
        region = depth_frame[y1:y2, x1:x2]
        valid = region[region > 0]
        
        if len(valid) > 0:
            return float(np.median(valid))
        return 0.0
    
    def _select_best_candidate(
        self,
        candidates: List[dict],
        roi_offset: Tuple[int, int],
        depth_scale: float,
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
        
        # Convert to world coordinates using homography or legacy scaling
        world_x, world_y = self._pixel_to_world(pixel_x, pixel_y)
        
        return TrackingResult(
            detected=True,
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            world_x=world_x,
            world_y=world_y,
            depth=best_candidate['depth'],
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
        
        # Circularity score (0-1)
        score += candidate['circularity'] * 2.0
        
        # Prefer candidates near last detection (if we have one)
        if self._last_detection and self._last_detection.detected:
            cx = candidate['cx'] + roi_offset[0]
            cy = candidate['cy'] + roi_offset[1]
            
            dx = cx - self._last_detection.pixel_x
            dy = cy - self._last_detection.pixel_y
            distance_px = math.sqrt(dx*dx + dy*dy)
            
            # Convert to meters using homography-aware method
            distance_m = self._pixel_distance_to_meters(
                distance_px, 
                self._last_detection.pixel_x, 
                self._last_detection.pixel_y
            )
            
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
        ir_roi: np.ndarray,
        candidates: List[dict],
        result: Optional[TrackingResult],
        roi_offset: Tuple[int, int]
    ):
        """Create debug visualization frame."""
        # Convert to color for visualization
        debug = cv2.cvtColor(ir_roi, cv2.COLOR_GRAY2BGR)
        
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
            
            cv2.circle(debug, (rx, ry), 10, (0, 255, 0), 2)
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
        self.debug_frame = None
        print("[Tracker] Reset")
    
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
