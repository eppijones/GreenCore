"""
Calibration module for the putting launch monitor.

Handles ROI selection, target line calibration, scale calibration,
and ground plane depth detection.
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Callable
from enum import Enum

import numpy as np
import cv2


class CalibrationMode(Enum):
    """Current calibration mode."""
    NONE = "none"
    ROI = "roi"
    TARGET_LINE = "target_line"
    SCALE = "scale"


@dataclass
class CalibrationData:
    """Calibration data container."""
    # ROI: (x, y, width, height) in pixels
    roi: Optional[Tuple[int, int, int, int]] = None
    
    # Target line: two points defining forward direction
    target_line_p1: Optional[Tuple[int, int]] = None
    target_line_p2: Optional[Tuple[int, int]] = None
    
    # Scale: pixels per meter
    pixels_per_meter: float = 500.0  # Default estimate
    
    # Ground plane depth in meters
    ground_depth: float = 0.5  # Default estimate
    
    # Depth tolerance for ball detection (meters)
    depth_tolerance: float = 0.05  # 5cm
    
    # Forward direction angle (computed from target line)
    forward_angle_rad: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'roi': list(self.roi) if self.roi else None,
            'target_line_p1': list(self.target_line_p1) if self.target_line_p1 else None,
            'target_line_p2': list(self.target_line_p2) if self.target_line_p2 else None,
            'pixels_per_meter': self.pixels_per_meter,
            'ground_depth': self.ground_depth,
            'depth_tolerance': self.depth_tolerance,
            'forward_angle_rad': self.forward_angle_rad,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CalibrationData':
        """Create from dictionary."""
        cal = cls()
        if data.get('roi'):
            cal.roi = tuple(data['roi'])
        if data.get('target_line_p1'):
            cal.target_line_p1 = tuple(data['target_line_p1'])
        if data.get('target_line_p2'):
            cal.target_line_p2 = tuple(data['target_line_p2'])
        cal.pixels_per_meter = data.get('pixels_per_meter', 500.0)
        cal.ground_depth = data.get('ground_depth', 0.5)
        cal.depth_tolerance = data.get('depth_tolerance', 0.05)
        cal.forward_angle_rad = data.get('forward_angle_rad', 0.0)
        return cal


class Calibration:
    """
    Handles all calibration operations for the putting monitor.
    
    Calibration modes:
    - ROI: User draws a rectangle defining the tracking region
    - Target Line: User clicks two points to define forward direction
    - Scale: User clicks two points exactly 1 meter apart
    - Ground Plane: Automatic depth calibration from static frames
    """
    
    def __init__(self, config_path: str = "config.json"):
        """
        Initialize calibration manager.
        
        Args:
            config_path: Path to save/load configuration.
        """
        self.config_path = config_path
        self.data = CalibrationData()
        
        # Current calibration mode
        self.mode = CalibrationMode.NONE
        
        # Temporary points for calibration
        self._temp_points: List[Tuple[int, int]] = []
        self._roi_start: Optional[Tuple[int, int]] = None
        self._roi_end: Optional[Tuple[int, int]] = None
        self._is_drawing_roi = False
        
        # Callbacks
        self._on_calibration_complete: Optional[Callable] = None
        
        # Load existing calibration if available
        self.load()
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save calibration to JSON file.
        
        Args:
            path: Optional path override.
            
        Returns:
            True if saved successfully.
        """
        save_path = path or self.config_path
        try:
            with open(save_path, 'w') as f:
                json.dump(self.data.to_dict(), f, indent=2)
            print(f"[Calibration] Saved to {save_path}")
            return True
        except Exception as e:
            print(f"[Calibration] Failed to save: {e}")
            return False
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load calibration from JSON file.
        
        Args:
            path: Optional path override.
            
        Returns:
            True if loaded successfully.
        """
        load_path = path or self.config_path
        try:
            with open(load_path, 'r') as f:
                data = json.load(f)
            self.data = CalibrationData.from_dict(data)
            print(f"[Calibration] Loaded from {load_path}")
            self._compute_forward_angle()
            return True
        except FileNotFoundError:
            print(f"[Calibration] No config file found at {load_path}")
            return False
        except Exception as e:
            print(f"[Calibration] Failed to load: {e}")
            return False
    
    def start_roi_calibration(self):
        """Start ROI selection mode."""
        self.mode = CalibrationMode.ROI
        self._roi_start = None
        self._roi_end = None
        self._is_drawing_roi = False
        print("[Calibration] ROI mode: Click and drag to select region")
    
    def start_target_line_calibration(self):
        """Start target line calibration mode."""
        self.mode = CalibrationMode.TARGET_LINE
        self._temp_points = []
        print("[Calibration] Target line mode: Click two points (start → target direction)")
    
    def start_scale_calibration(self):
        """Start scale calibration mode."""
        self.mode = CalibrationMode.SCALE
        self._temp_points = []
        print("[Calibration] Scale mode: Click two points exactly 1 meter apart")
    
    def cancel_calibration(self):
        """Cancel current calibration mode."""
        self.mode = CalibrationMode.NONE
        self._temp_points = []
        self._roi_start = None
        self._roi_end = None
        self._is_drawing_roi = False
        print("[Calibration] Cancelled")
    
    def handle_mouse_event(
        self, 
        event: int, 
        x: int, 
        y: int, 
        flags: int
    ) -> bool:
        """
        Handle mouse events for calibration.
        
        Args:
            event: OpenCV mouse event type.
            x: X coordinate.
            y: Y coordinate.
            flags: OpenCV event flags.
            
        Returns:
            True if event was handled.
        """
        if self.mode == CalibrationMode.NONE:
            return False
        
        if self.mode == CalibrationMode.ROI:
            return self._handle_roi_mouse(event, x, y)
        elif self.mode == CalibrationMode.TARGET_LINE:
            return self._handle_point_click(event, x, y, 2, self._finish_target_line)
        elif self.mode == CalibrationMode.SCALE:
            return self._handle_point_click(event, x, y, 2, self._finish_scale)
        
        return False
    
    def _handle_roi_mouse(self, event: int, x: int, y: int) -> bool:
        """Handle mouse events for ROI selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._roi_start = (x, y)
            self._roi_end = (x, y)
            self._is_drawing_roi = True
            return True
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self._is_drawing_roi:
                self._roi_end = (x, y)
                return True
        
        elif event == cv2.EVENT_LBUTTONUP:
            if self._is_drawing_roi:
                self._roi_end = (x, y)
                self._is_drawing_roi = False
                self._finish_roi()
                return True
        
        return False
    
    def _handle_point_click(
        self, 
        event: int, 
        x: int, 
        y: int, 
        required_points: int,
        on_complete: Callable
    ) -> bool:
        """Handle point clicking for calibration."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._temp_points.append((x, y))
            print(f"[Calibration] Point {len(self._temp_points)}: ({x}, {y})")
            
            if len(self._temp_points) >= required_points:
                on_complete()
            
            return True
        
        return False
    
    def _finish_roi(self):
        """Finish ROI calibration."""
        if self._roi_start and self._roi_end:
            x1, y1 = self._roi_start
            x2, y2 = self._roi_end
            
            # Normalize coordinates
            x = min(x1, x2)
            y = min(y1, y2)
            w = abs(x2 - x1)
            h = abs(y2 - y1)
            
            if w > 10 and h > 10:
                self.data.roi = (x, y, w, h)
                print(f"[Calibration] ROI set: {self.data.roi}")
                self.save()
            else:
                print("[Calibration] ROI too small, try again")
        
        self.mode = CalibrationMode.NONE
    
    def _finish_target_line(self):
        """Finish target line calibration."""
        if len(self._temp_points) >= 2:
            self.data.target_line_p1 = self._temp_points[0]
            self.data.target_line_p2 = self._temp_points[1]
            self._compute_forward_angle()
            print(f"[Calibration] Target line set, angle: {math.degrees(self.data.forward_angle_rad):.1f}°")
            self.save()
        
        self._temp_points = []
        self.mode = CalibrationMode.NONE
    
    def _finish_scale(self):
        """Finish scale calibration."""
        if len(self._temp_points) >= 2:
            p1 = self._temp_points[0]
            p2 = self._temp_points[1]
            
            # Calculate pixel distance
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            pixel_distance = math.sqrt(dx*dx + dy*dy)
            
            if pixel_distance > 10:
                # Distance is 1 meter
                self.data.pixels_per_meter = pixel_distance
                print(f"[Calibration] Scale set: {self.data.pixels_per_meter:.1f} pixels/meter")
                self.save()
            else:
                print("[Calibration] Points too close, try again")
        
        self._temp_points = []
        self.mode = CalibrationMode.NONE
    
    def _compute_forward_angle(self):
        """Compute forward direction angle from target line."""
        if self.data.target_line_p1 and self.data.target_line_p2:
            dx = self.data.target_line_p2[0] - self.data.target_line_p1[0]
            dy = self.data.target_line_p2[1] - self.data.target_line_p1[1]
            self.data.forward_angle_rad = math.atan2(dy, dx)
    
    def calibrate_ground(self, depth_frames: List[np.ndarray], depth_scale: float = 0.001):
        """
        Calibrate ground plane depth from static frames.
        
        Uses the median depth value within the ROI from multiple frames.
        
        Args:
            depth_frames: List of depth frames (at least 5 recommended).
            depth_scale: Depth unit scale (default 0.001 for millimeters).
        """
        if len(depth_frames) < 1:
            print("[Calibration] Need at least one depth frame for ground calibration")
            return
        
        all_depths = []
        
        for frame in depth_frames:
            if self.data.roi:
                x, y, w, h = self.data.roi
                region = frame[y:y+h, x:x+w]
            else:
                region = frame
            
            # Get valid depth values
            valid = region[region > 0]
            if len(valid) > 0:
                all_depths.extend(valid.flatten().tolist())
        
        if all_depths:
            # Use median for robustness
            median_depth = np.median(all_depths)
            self.data.ground_depth = float(median_depth) * depth_scale
            
            # Set tolerance based on depth variation
            std_depth = np.std(all_depths) * depth_scale
            self.data.depth_tolerance = max(0.03, min(0.10, std_depth * 3))
            
            print(f"[Calibration] Ground depth: {self.data.ground_depth:.3f}m ± {self.data.depth_tolerance:.3f}m")
            self.save()
        else:
            print("[Calibration] No valid depth data for ground calibration")
    
    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw calibration overlays on frame.
        
        Args:
            frame: Input frame (will be modified).
            
        Returns:
            Frame with overlays.
        """
        # Ensure frame is color
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw ROI
        if self.data.roi:
            x, y, w, h = self.data.roi
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw target line
        if self.data.target_line_p1 and self.data.target_line_p2:
            p1 = self.data.target_line_p1
            p2 = self.data.target_line_p2
            cv2.line(frame, p1, p2, (255, 0, 0), 2)
            cv2.circle(frame, p1, 5, (255, 0, 0), -1)
            cv2.circle(frame, p2, 5, (0, 0, 255), -1)
            
            # Draw arrow for direction
            arrow_len = 30
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx*dx + dy*dy)
            if length > 0:
                dx /= length
                dy /= length
                mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
                end = (int(mid[0] + dx * arrow_len), int(mid[1] + dy * arrow_len))
                cv2.arrowedLine(frame, mid, end, (0, 255, 255), 2)
        
        # Draw current calibration mode
        if self.mode != CalibrationMode.NONE:
            text = f"CALIBRATING: {self.mode.value.upper()}"
            cv2.putText(frame, text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Draw temporary ROI
            if self.mode == CalibrationMode.ROI and self._roi_start and self._roi_end:
                cv2.rectangle(frame, self._roi_start, self._roi_end, (0, 255, 255), 2)
            
            # Draw temporary points
            for i, pt in enumerate(self._temp_points):
                color = (0, 255, 0) if i == 0 else (0, 0, 255)
                cv2.circle(frame, pt, 8, color, -1)
                cv2.putText(frame, str(i+1), (pt[0]+10, pt[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def pixels_to_meters(self, pixels: float) -> float:
        """Convert pixel distance to meters."""
        if self.data.pixels_per_meter > 0:
            return pixels / self.data.pixels_per_meter
        return pixels / 500.0  # Fallback
    
    def meters_to_pixels(self, meters: float) -> float:
        """Convert meters to pixel distance."""
        return meters * self.data.pixels_per_meter
    
    def get_direction_relative_to_target(self, velocity_angle_rad: float) -> float:
        """
        Get direction angle relative to calibrated target line.
        
        Args:
            velocity_angle_rad: Velocity vector angle in radians.
            
        Returns:
            Angle in degrees relative to target line.
            Positive = right of target, negative = left.
        """
        # Calculate relative angle
        rel_angle = velocity_angle_rad - self.data.forward_angle_rad
        
        # Normalize to -180 to 180 degrees
        rel_deg = math.degrees(rel_angle)
        while rel_deg > 180:
            rel_deg -= 360
        while rel_deg < -180:
            rel_deg += 360
        
        return rel_deg
    
    def is_in_roi(self, x: int, y: int) -> bool:
        """Check if a point is inside the ROI."""
        if not self.data.roi:
            return True  # No ROI = everywhere is valid
        
        rx, ry, rw, rh = self.data.roi
        return rx <= x < rx + rw and ry <= y < ry + rh
    
    def is_valid_depth(self, depth_meters: float) -> bool:
        """Check if a depth value is within the valid range for ground plane."""
        if depth_meters <= 0:
            return False
        
        min_depth = self.data.ground_depth - self.data.depth_tolerance
        max_depth = self.data.ground_depth + self.data.depth_tolerance
        
        return min_depth <= depth_meters <= max_depth
    
    def get_roi_offset(self) -> Tuple[int, int]:
        """Get ROI top-left offset for coordinate conversion."""
        if self.data.roi:
            return (self.data.roi[0], self.data.roi[1])
        return (0, 0)
    
    @property
    def is_calibrated(self) -> bool:
        """Check if all required calibrations are complete."""
        return (
            self.data.roi is not None and
            self.data.target_line_p1 is not None and
            self.data.target_line_p2 is not None and
            self.data.pixels_per_meter > 0 and
            self.data.ground_depth > 0
        )
    
    @property
    def status_text(self) -> str:
        """Get calibration status as text."""
        status = []
        status.append(f"ROI: {'OK' if self.data.roi else 'NO'}")
        status.append(f"Target: {'OK' if self.data.target_line_p1 else 'NO'}")
        status.append(f"Scale: {self.data.pixels_per_meter:.0f} px/m")
        return " | ".join(status)
