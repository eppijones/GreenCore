#!/usr/bin/env python3
"""
Ball Detector - Unified Golf Ball Detection Module

High-accuracy detection optimized for:
- White golf ball on dark putting surface
- 78-80cm camera height
- 120fps Arducam global shutter
- Left-to-right ball tracking

This module is shared between camera_alignment.py and main.py
to ensure consistent detection and tracking.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List
import time


@dataclass
class BallDetection:
    """Result of ball detection."""
    detected: bool
    x: int = 0              # Center X in pixels
    y: int = 0              # Center Y in pixels
    radius: int = 0         # Radius in pixels
    confidence: float = 0.0  # Detection confidence 0-1
    timestamp: float = 0.0   # Detection timestamp
    
    @property
    def center(self) -> Tuple[int, int]:
        return (self.x, self.y)
    
    @property
    def diameter(self) -> int:
        return self.radius * 2


@dataclass
class DetectorConfig:
    """Ball detector configuration."""
    # Scale calibration
    scale_px_per_cm: float = 11.7  # Pixels per cm at 78cm height
    
    # Ball physical size
    ball_diameter_cm: float = 4.27  # Standard golf ball
    
    # Detection tolerances
    size_tolerance: float = 0.4     # Allow ±40% from expected size
    min_circularity: float = 0.65   # Minimum circularity score
    
    # Brightness threshold for white ball detection
    brightness_threshold: int = 180  # Threshold for white ball (0-255)
    
    # ROI (optional - None means full frame)
    roi: Optional[Tuple[int, int, int, int]] = None  # (x, y, w, h)
    
    @property
    def expected_ball_px(self) -> int:
        """Expected ball diameter in pixels."""
        return int(self.ball_diameter_cm * self.scale_px_per_cm)
    
    @property
    def min_radius(self) -> int:
        """Minimum detection radius."""
        return max(10, int(self.expected_ball_px * (1 - self.size_tolerance) / 2))
    
    @property
    def max_radius(self) -> int:
        """Maximum detection radius."""
        return int(self.expected_ball_px * (1 + self.size_tolerance) / 2)


class BallDetector:
    """
    High-accuracy golf ball detector.
    
    Uses multiple detection methods with voting for reliability:
    1. Threshold + contour analysis (best for white ball on dark)
    2. Hough circles (backup method)
    3. Template matching (for tracking continuity)
    """
    
    def __init__(self, config: Optional[DetectorConfig] = None):
        self.config = config or DetectorConfig()
        
        # Tracking state
        self._last_detection: Optional[BallDetection] = None
        self._detection_history: List[BallDetection] = []
        self._history_max = 10
        
        # Motion tracking
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._last_time: float = 0.0
    
    def detect(
        self,
        frame: np.ndarray,
        gray: Optional[np.ndarray] = None,
        timestamp: Optional[float] = None
    ) -> BallDetection:
        """
        Detect golf ball in frame.
        
        Args:
            frame: BGR color frame
            gray: Grayscale frame (computed if not provided)
            timestamp: Frame timestamp (uses current time if not provided)
            
        Returns:
            BallDetection result
        """
        if timestamp is None:
            timestamp = time.time()
        
        if gray is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        cfg = self.config
        
        # Apply ROI if configured
        if cfg.roi:
            rx, ry, rw, rh = cfg.roi
            roi_gray = gray[ry:ry+rh, rx:rx+rw]
            offset_x, offset_y = rx, ry
        else:
            roi_gray = gray
            offset_x, offset_y = 0, 0
        
        # Method 1: Threshold-based detection
        detection = self._detect_threshold(roi_gray, offset_x, offset_y, timestamp)
        
        if detection.detected:
            self._update_tracking(detection, timestamp)
            return detection
        
        # Method 2: Hough circles fallback
        detection = self._detect_hough(roi_gray, offset_x, offset_y, timestamp)
        
        if detection.detected:
            self._update_tracking(detection, timestamp)
            return detection
        
        # Method 3: Predictive tracking (if we had recent detection)
        if self._last_detection and self._last_detection.detected:
            elapsed = timestamp - self._last_time
            if elapsed < 0.1:  # Within 100ms
                # Predict position based on velocity
                predicted = self._predict_position(elapsed)
                if predicted:
                    detection = self._detect_near_point(
                        roi_gray, predicted, offset_x, offset_y, timestamp
                    )
                    if detection.detected:
                        self._update_tracking(detection, timestamp)
                        return detection
        
        # No detection
        return BallDetection(detected=False, timestamp=timestamp)
    
    def _detect_threshold(
        self,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
        timestamp: float
    ) -> BallDetection:
        """Detect ball using threshold + contour analysis."""
        cfg = self.config
        
        # Threshold for white ball
        _, thresh = cv2.threshold(
            gray, cfg.brightness_threshold, 255, cv2.THRESH_BINARY
        )
        
        # Morphological cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Size constraints
        min_area = np.pi * cfg.min_radius**2 * 0.5
        max_area = np.pi * cfg.max_radius**2 * 1.5
        
        best_match = None
        best_score = 0
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            
            # Circularity score
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < cfg.min_circularity:
                continue
            
            # Get enclosing circle
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            # Verify radius
            if not (cfg.min_radius <= radius <= cfg.max_radius):
                continue
            
            # Score based on circularity and size match
            expected_r = cfg.expected_ball_px / 2
            size_match = 1.0 - abs(radius - expected_r) / expected_r
            score = circularity * 0.6 + size_match * 0.4
            
            if score > best_score:
                best_score = score
                best_match = (int(cx), int(cy), int(radius), score)
        
        if best_match:
            cx, cy, radius, score = best_match
            return BallDetection(
                detected=True,
                x=offset_x + cx,
                y=offset_y + cy,
                radius=radius,
                confidence=min(1.0, score),
                timestamp=timestamp
            )
        
        return BallDetection(detected=False, timestamp=timestamp)
    
    def _detect_hough(
        self,
        gray: np.ndarray,
        offset_x: int,
        offset_y: int,
        timestamp: float
    ) -> BallDetection:
        """Detect ball using Hough circles."""
        cfg = self.config
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Hough circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=cfg.expected_ball_px,
            param1=50,
            param2=25,
            minRadius=cfg.min_radius,
            maxRadius=cfg.max_radius
        )
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            # Take best match (first one, sorted by accumulator)
            x, y, r = circles[0][0]
            return BallDetection(
                detected=True,
                x=offset_x + int(x),
                y=offset_y + int(y),
                radius=int(r),
                confidence=0.7,  # Lower confidence for Hough
                timestamp=timestamp
            )
        
        return BallDetection(detected=False, timestamp=timestamp)
    
    def _detect_near_point(
        self,
        gray: np.ndarray,
        point: Tuple[int, int],
        offset_x: int,
        offset_y: int,
        timestamp: float
    ) -> BallDetection:
        """Search for ball near a predicted point."""
        cfg = self.config
        search_radius = cfg.max_radius * 3
        
        px, py = point
        px -= offset_x
        py -= offset_y
        
        # Extract search region
        h, w = gray.shape[:2]
        x1 = max(0, px - search_radius)
        y1 = max(0, py - search_radius)
        x2 = min(w, px + search_radius)
        y2 = min(h, py + search_radius)
        
        if x2 <= x1 or y2 <= y1:
            return BallDetection(detected=False, timestamp=timestamp)
        
        roi = gray[y1:y2, x1:x2]
        
        # Use threshold detection in small region
        result = self._detect_threshold(roi, offset_x + x1, offset_y + y1, timestamp)
        
        if result.detected:
            result.confidence *= 0.9  # Slightly lower for predicted
        
        return result
    
    def _predict_position(self, elapsed: float) -> Optional[Tuple[int, int]]:
        """Predict ball position based on velocity."""
        if not self._last_detection or not self._last_detection.detected:
            return None
        
        vx, vy = self._velocity
        if abs(vx) < 1 and abs(vy) < 1:
            # Not moving, return last position
            return (self._last_detection.x, self._last_detection.y)
        
        # Predict new position
        new_x = int(self._last_detection.x + vx * elapsed)
        new_y = int(self._last_detection.y + vy * elapsed)
        
        return (new_x, new_y)
    
    def _update_tracking(self, detection: BallDetection, timestamp: float):
        """Update tracking state with new detection."""
        if self._last_detection and self._last_detection.detected:
            dt = timestamp - self._last_time
            if dt > 0.001:  # Avoid division by zero
                vx = (detection.x - self._last_detection.x) / dt
                vy = (detection.y - self._last_detection.y) / dt
                
                # Smooth velocity with exponential filter
                alpha = 0.3
                self._velocity = (
                    alpha * vx + (1 - alpha) * self._velocity[0],
                    alpha * vy + (1 - alpha) * self._velocity[1]
                )
        
        self._last_detection = detection
        self._last_time = timestamp
        
        # Update history
        self._detection_history.append(detection)
        if len(self._detection_history) > self._history_max:
            self._detection_history.pop(0)
    
    def get_velocity(self) -> Tuple[float, float]:
        """Get current ball velocity in pixels/second."""
        return self._velocity
    
    def get_speed_mps(self, pixels_per_meter: float) -> float:
        """Get ball speed in meters per second."""
        vx, vy = self._velocity
        speed_px = np.sqrt(vx**2 + vy**2)
        return speed_px / pixels_per_meter
    
    def reset(self):
        """Reset tracking state."""
        self._last_detection = None
        self._detection_history.clear()
        self._velocity = (0.0, 0.0)
        self._last_time = 0.0


def create_detector_from_config(config_path: str = "config.json") -> BallDetector:
    """Create a BallDetector from config file."""
    import json
    from pathlib import Path
    
    cfg = DetectorConfig()
    
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            # Load alignment settings
            if 'alignment' in data:
                align = data['alignment']
                cfg.scale_px_per_cm = align.get('scale_px_per_cm', cfg.scale_px_per_cm)
                
                # Set ROI if defined
                if all(k in align for k in ['roi_x', 'roi_y', 'roi_width', 'roi_height']):
                    cfg.roi = (
                        align['roi_x'],
                        align['roi_y'],
                        align['roi_width'],
                        align['roi_height']
                    )
            
            print(f"[BallDetector] Loaded config: scale={cfg.scale_px_per_cm:.1f} px/cm")
            
        except Exception as e:
            print(f"[BallDetector] Could not load config: {e}")
    
    return BallDetector(cfg)
