"""
Auto Ball Tracker - Improved golf ball tracking with auto-detection.

Features:
- Auto-detects golf ball on startup (no click required)
- Stops tracking when ball exits frame
- Waits for virtual simulation before resuming
- Uses full frame (no ROI required)
- Click-to-track as fallback
"""

import math
import time
from dataclasses import dataclass
from typing import Optional, List, Tuple
from collections import deque
from enum import Enum

import numpy as np
import cv2

from .calibration import Calibration


class TrackerState(Enum):
    """Tracker state machine."""
    SEARCHING = "searching"      # Looking for ball (auto-detect)
    TRACKING = "tracking"        # Actively tracking ball
    SHOT_DETECTED = "shot"       # Shot in progress
    WAITING = "waiting"          # Waiting for virtual ball to stop
    LOST = "lost"                # Ball exited frame


@dataclass
class TrackingResult:
    """Result of ball tracking for a single frame."""
    detected: bool = False
    pixel_x: float = 0.0
    pixel_y: float = 0.0
    world_x: float = 0.0
    world_y: float = 0.0
    depth: float = 0.5
    circularity: float = 0.0
    area: float = 0.0
    timestamp: float = 0.0
    frame_number: int = 0
    velocity_x: float = 0.0
    velocity_y: float = 0.0
    
    @property
    def speed(self) -> float:
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
    
    @property
    def velocity_angle_rad(self) -> float:
        return math.atan2(self.velocity_y, self.velocity_x)


@dataclass
class AutoTrackerConfig:
    """Configuration for auto ball tracker."""
    # Ball detection
    min_area: int = 200
    max_area: int = 10000
    min_circularity: float = 0.5
    min_brightness: int = 180
    
    # Auto-detection
    auto_detect_enabled: bool = True
    auto_detect_interval: float = 0.5  # Seconds between auto-detect attempts
    
    # Frame boundaries (margin in pixels to consider "exited")
    edge_margin: int = 20
    
    # Tracking
    max_trajectory_length: int = 100
    velocity_smoothing_window: int = 5
    max_jump_distance_m: float = 0.3  # Max jump between frames
    
    # Shot detection integration
    wait_after_shot_ms: float = 3000  # Wait 3 seconds for virtual ball
    
    # Debug
    debug_visualization: bool = True


class AutoBallTracker:
    """
    Automatic golf ball tracker.
    
    State machine:
    SEARCHING -> TRACKING (ball found)
    TRACKING -> SHOT_DETECTED (ball moving fast)
    SHOT_DETECTED -> WAITING (ball exits frame)
    WAITING -> SEARCHING (after timeout)
    TRACKING -> LOST (ball exits, no shot)
    LOST -> SEARCHING (after brief pause)
    """
    
    def __init__(
        self,
        calibration: Calibration,
        config: Optional[AutoTrackerConfig] = None
    ):
        self.calibration = calibration
        self.config = config or AutoTrackerConfig()
        
        # State
        self._state = TrackerState.SEARCHING
        self._state_start_time = time.time()
        
        # Ball tracking
        self._trajectory: deque = deque(maxlen=self.config.max_trajectory_length)
        self._last_detection: Optional[TrackingResult] = None
        self._ball_position: Optional[Tuple[int, int]] = None
        self._search_radius: int = 100
        
        # Auto-detection
        self._last_auto_detect_time = 0.0
        self._auto_detected_ball: Optional[Tuple[int, int]] = None
        
        # Frame info
        self._frame_width = 0
        self._frame_height = 0
        
        # Shot tracking
        self._shot_in_progress = False
        self._shot_start_time = 0.0
        
        # Debug
        self.debug_frame: Optional[np.ndarray] = None
        
        # Callbacks
        self._on_shot_complete: Optional[callable] = None
    
    def set_shot_complete_callback(self, callback):
        """Set callback for when we should resume tracking."""
        self._on_shot_complete = callback
    
    def notify_shot_detected(self):
        """Called by shot detector when shot is detected."""
        self._shot_in_progress = True
        self._shot_start_time = time.time()
        self._state = TrackerState.SHOT_DETECTED
        print("[AutoTracker] Shot detected - tracking impact")
    
    def notify_simulation_complete(self):
        """Called when virtual ball simulation is complete."""
        print("[AutoTracker] Simulation complete - resuming search")
        self._state = TrackerState.SEARCHING
        self._shot_in_progress = False
        self._trajectory.clear()
        self._last_detection = None
        self._ball_position = None
    
    def manual_set_ball(self, x: int, y: int):
        """Manually set ball position (fallback click-to-track)."""
        self._ball_position = (x, y)
        self._state = TrackerState.TRACKING
        self._trajectory.clear()
        self._last_detection = None
        print(f"[AutoTracker] Manual ball position: ({x}, {y})")
    
    def update(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int = 0
    ) -> Optional[TrackingResult]:
        """Process frame and update tracking."""
        
        self._frame_height, self._frame_width = gray_frame.shape[:2]
        
        # State machine
        if self._state == TrackerState.SEARCHING:
            return self._handle_searching(color_frame, gray_frame, timestamp, frame_number)
        
        elif self._state == TrackerState.TRACKING:
            return self._handle_tracking(color_frame, gray_frame, timestamp, frame_number)
        
        elif self._state == TrackerState.SHOT_DETECTED:
            return self._handle_shot_detected(color_frame, gray_frame, timestamp, frame_number)
        
        elif self._state == TrackerState.WAITING:
            return self._handle_waiting(color_frame, gray_frame, timestamp, frame_number)
        
        elif self._state == TrackerState.LOST:
            return self._handle_lost(timestamp, frame_number)
        
        return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
    
    def _handle_searching(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int
    ) -> TrackingResult:
        """Search for golf ball in full frame."""
        
        # Try auto-detection periodically
        if self.config.auto_detect_enabled:
            if timestamp - self._last_auto_detect_time >= self.config.auto_detect_interval:
                self._last_auto_detect_time = timestamp
                
                ball_pos = self._auto_detect_ball(color_frame, gray_frame)
                
                if ball_pos:
                    self._ball_position = ball_pos
                    self._state = TrackerState.TRACKING
                    print(f"[AutoTracker] Ball auto-detected at {ball_pos}")
                    
                    # Create initial result
                    return self._create_result(ball_pos, timestamp, frame_number, gray_frame)
        
        # Update debug frame
        if self.config.debug_visualization:
            self._create_search_debug(color_frame)
        
        return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
    
    def _handle_tracking(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int
    ) -> TrackingResult:
        """Track ball from known position."""
        
        if self._ball_position is None and self._last_detection is None:
            self._state = TrackerState.SEARCHING
            return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
        
        # Get search center
        if self._ball_position:
            search_center = self._ball_position
        else:
            search_center = (int(self._last_detection.pixel_x), int(self._last_detection.pixel_y))
        
        # Search for ball near expected position
        result = self._track_near_position(
            color_frame, gray_frame, search_center, timestamp, frame_number
        )
        
        if result and result.detected:
            # Clear manual position, now tracking automatically
            self._ball_position = None
            
            # Compute velocity
            self._compute_velocity(result)
            self._trajectory.append(result)
            self._last_detection = result
            
            # Check if ball exited frame
            if self._is_near_edge(result.pixel_x, result.pixel_y):
                if self._shot_in_progress:
                    # Ball exited during shot - go to waiting
                    self._state = TrackerState.WAITING
                    self._state_start_time = timestamp
                    print("[AutoTracker] Ball exited frame during shot")
                else:
                    # Ball drifted out - lost
                    self._state = TrackerState.LOST
                    self._state_start_time = timestamp
        else:
            # Lost tracking
            if self._shot_in_progress:
                self._state = TrackerState.WAITING
                self._state_start_time = timestamp
                print("[AutoTracker] Lost ball during shot - waiting")
            else:
                # Expand search
                self._search_radius = min(200, self._search_radius + 20)
                
                if self._search_radius >= 200:
                    self._state = TrackerState.LOST
                    self._state_start_time = timestamp
        
        # Shrink search radius when tracking well
        if result and result.detected:
            self._search_radius = max(60, self._search_radius - 5)
        
        return result or TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
    
    def _handle_shot_detected(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int
    ) -> TrackingResult:
        """Continue tracking during shot."""
        # Same as tracking, but we know a shot is in progress
        return self._handle_tracking(color_frame, gray_frame, timestamp, frame_number)
    
    def _handle_waiting(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        timestamp: float,
        frame_number: int
    ) -> TrackingResult:
        """Wait for virtual simulation to complete."""
        
        elapsed = (timestamp - self._state_start_time) * 1000
        
        # Timeout - go back to searching
        if elapsed >= self.config.wait_after_shot_ms:
            print("[AutoTracker] Wait timeout - resuming search")
            self._state = TrackerState.SEARCHING
            self._shot_in_progress = False
            self._trajectory.clear()
            self._last_detection = None
        
        # Update debug frame with waiting message
        if self.config.debug_visualization:
            self._create_waiting_debug(color_frame, elapsed)
        
        return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
    
    def _handle_lost(self, timestamp: float, frame_number: int) -> TrackingResult:
        """Handle lost ball state."""
        
        elapsed = (timestamp - self._state_start_time) * 1000
        
        # After brief pause, go back to searching
        if elapsed >= 500:  # 500ms pause
            self._state = TrackerState.SEARCHING
            self._trajectory.clear()
            self._last_detection = None
            self._ball_position = None
            self._search_radius = 100
        
        return TrackingResult(detected=False, timestamp=timestamp, frame_number=frame_number)
    
    def _auto_detect_ball(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Auto-detect golf ball in full frame."""
        
        # Blur to reduce noise
        blurred = cv2.GaussianBlur(gray_frame, (5, 5), 0)
        
        # Threshold for bright objects
        _, mask = cv2.threshold(blurred, self.config.min_brightness, 255, cv2.THRESH_BINARY)
        
        # Also use HSV for white detection
        hsv = cv2.cvtColor(color_frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine masks
        combined = cv2.bitwise_or(mask, mask_white)
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Find best ball candidate
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.config.min_area or area > self.config.max_area:
                continue
            
            # Circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity < self.config.min_circularity:
                continue
            
            # Get centroid
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Skip if too close to edge
            margin = self.config.edge_margin * 2
            if cx < margin or cx > self._frame_width - margin:
                continue
            if cy < margin or cy > self._frame_height - margin:
                continue
            
            # Score based on circularity and area (prefer larger, rounder)
            score = circularity * 2 + (area / self.config.max_area)
            
            if score > best_score:
                best_score = score
                best_candidate = (cx, cy)
        
        return best_candidate
    
    def _track_near_position(
        self,
        color_frame: np.ndarray,
        gray_frame: np.ndarray,
        center: Tuple[int, int],
        timestamp: float,
        frame_number: int
    ) -> Optional[TrackingResult]:
        """Track ball near expected position."""
        
        cx, cy = center
        r = self._search_radius
        
        # Clamp ROI to frame bounds
        x1 = max(0, cx - r)
        y1 = max(0, cy - r)
        x2 = min(self._frame_width, cx + r)
        y2 = min(self._frame_height, cy + r)
        
        if x2 - x1 < 20 or y2 - y1 < 20:
            return None
        
        # Extract ROI
        color_roi = color_frame[y1:y2, x1:x2]
        gray_roi = gray_frame[y1:y2, x1:x2]
        
        # Detect ball in ROI
        ball_pos = self._detect_ball_in_roi(color_roi, gray_roi)
        
        if ball_pos:
            # Convert to full frame coordinates
            px = ball_pos[0] + x1
            py = ball_pos[1] + y1
            
            # Create debug frame
            if self.config.debug_visualization:
                self._create_tracking_debug(color_roi, ball_pos, (x1, y1))
            
            return self._create_result((px, py), timestamp, frame_number, gray_frame)
        
        return None
    
    def _detect_ball_in_roi(
        self,
        color_roi: np.ndarray,
        gray_roi: np.ndarray
    ) -> Optional[Tuple[int, int]]:
        """Detect ball in ROI."""
        
        if gray_roi.size == 0:
            return None
        
        # Blur
        blurred = cv2.GaussianBlur(gray_roi, (5, 5), 0)
        
        # Adaptive threshold based on ROI brightness
        mean_brightness = np.mean(blurred)
        threshold = max(150, min(230, mean_brightness + 30))
        
        _, mask = cv2.threshold(blurred, int(threshold), 255, cv2.THRESH_BINARY)
        
        # HSV white detection
        hsv = cv2.cvtColor(color_roi, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 170])
        upper_white = np.array([180, 60, 255])
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        combined = cv2.bitwise_or(mask, mask_white)
        
        # Cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        best_candidate = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < 100 or area > 15000:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * math.pi * area / (perimeter * perimeter)
            
            if circularity < 0.4:
                continue
            
            moments = cv2.moments(contour)
            if moments['m00'] == 0:
                continue
            
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Score
            score = circularity * 2 + (area / 5000)
            
            # Prefer center of ROI (closer to expected position)
            roi_h, roi_w = gray_roi.shape[:2]
            dist_to_center = math.sqrt((cx - roi_w/2)**2 + (cy - roi_h/2)**2)
            max_dist = math.sqrt((roi_w/2)**2 + (roi_h/2)**2)
            center_score = 1.0 - (dist_to_center / max_dist)
            score += center_score * 2
            
            if score > best_score:
                best_score = score
                best_candidate = (cx, cy)
        
        return best_candidate
    
    def _create_result(
        self,
        pixel_pos: Tuple[int, int],
        timestamp: float,
        frame_number: int,
        gray_frame: np.ndarray
    ) -> TrackingResult:
        """Create tracking result from pixel position."""
        
        px, py = pixel_pos
        
        # Convert to world coordinates
        # Use frame center as origin if no ROI
        roi = self.calibration.data.roi
        if roi:
            roi_center_x = roi[0] + roi[2] / 2
            roi_center_y = roi[1] + roi[3] / 2
        else:
            roi_center_x = self._frame_width / 2
            roi_center_y = self._frame_height / 2
        
        world_x = self.calibration.pixels_to_meters(px - roi_center_x)
        world_y = self.calibration.pixels_to_meters(py - roi_center_y)
        
        return TrackingResult(
            detected=True,
            pixel_x=px,
            pixel_y=py,
            world_x=world_x,
            world_y=world_y,
            depth=self.calibration.data.ground_depth,
            timestamp=timestamp,
            frame_number=frame_number,
        )
    
    def _compute_velocity(self, result: TrackingResult):
        """Compute velocity from trajectory."""
        if len(self._trajectory) < 2:
            return
        
        window = min(self.config.velocity_smoothing_window, len(self._trajectory))
        recent = list(self._trajectory)[-window:]
        
        if len(recent) < 2:
            return
        
        first = recent[0]
        last = recent[-1]
        
        dt = last.timestamp - first.timestamp
        if dt <= 0:
            return
        
        dx = last.world_x - first.world_x
        dy = last.world_y - first.world_y
        
        result.velocity_x = dx / dt
        result.velocity_y = dy / dt
    
    def _is_near_edge(self, x: float, y: float) -> bool:
        """Check if position is near frame edge."""
        margin = self.config.edge_margin
        
        if x < margin or x > self._frame_width - margin:
            return True
        if y < margin or y > self._frame_height - margin:
            return True
        
        return False
    
    def _create_search_debug(self, color_frame: np.ndarray):
        """Create debug frame for search mode."""
        h, w = color_frame.shape[:2]
        debug = color_frame[:min(200, h), :min(200, w)].copy()
        cv2.putText(debug, "SEARCHING", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        self.debug_frame = debug
    
    def _create_tracking_debug(
        self,
        color_roi: np.ndarray,
        ball_pos: Tuple[int, int],
        roi_offset: Tuple[int, int]
    ):
        """Create debug frame for tracking mode."""
        debug = color_roi.copy()
        
        # Draw ball position
        cv2.circle(debug, ball_pos, 15, (0, 255, 0), 2)
        cv2.circle(debug, ball_pos, 3, (0, 255, 0), -1)
        
        # Draw state
        cv2.putText(debug, self._state.value.upper(), (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        self.debug_frame = debug
    
    def _create_waiting_debug(self, color_frame: np.ndarray, elapsed_ms: float):
        """Create debug frame for waiting mode."""
        h, w = color_frame.shape[:2]
        debug = color_frame[:min(200, h), :min(200, w)].copy()
        
        remaining = max(0, self.config.wait_after_shot_ms - elapsed_ms) / 1000
        cv2.putText(debug, f"WAITING {remaining:.1f}s", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
        
        self.debug_frame = debug
    
    def reset(self):
        """Reset tracker."""
        self._state = TrackerState.SEARCHING
        self._trajectory.clear()
        self._last_detection = None
        self._ball_position = None
        self._shot_in_progress = False
        self._search_radius = 100
        print("[AutoTracker] Reset - searching for ball")
    
    @property
    def state(self) -> TrackerState:
        return self._state
    
    @property
    def is_tracking(self) -> bool:
        return self._state in (TrackerState.TRACKING, TrackerState.SHOT_DETECTED)
    
    @property
    def has_detection(self) -> bool:
        return self._last_detection is not None and self._last_detection.detected
    
    @property
    def current_speed(self) -> float:
        if self._trajectory and self._trajectory[-1].detected:
            return self._trajectory[-1].speed
        return 0.0
    
    def get_trajectory(self) -> List[TrackingResult]:
        return list(self._trajectory)
