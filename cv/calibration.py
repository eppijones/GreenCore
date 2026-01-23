"""
Calibration module for the putting launch monitor.

Handles ROI selection, target line calibration, scale calibration,
ground plane depth detection, and ArUco-based homography calibration.

ArUco Calibration:
- Uses 4 ArUco markers (DICT_4X4_50, IDs 0-3) at putting mat corners
- Computes homography matrix for accurate pixel→world coordinate mapping
- Supports optional lens undistortion via ChArUco intrinsics calibration
"""

import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, List, Callable, Dict, Any
from enum import Enum
from pathlib import Path

import numpy as np
import cv2

# ArUco dictionary and detector setup
try:
    from cv2 import aruco
    ARUCO_DICT = aruco.DICT_4X4_50
except AttributeError:
    # OpenCV 4.7+ moved ArUco
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    aruco = cv2.aruco


class CalibrationMode(Enum):
    """Current calibration mode."""
    NONE = "none"
    ROI = "roi"
    TARGET_LINE = "target_line"
    SCALE = "scale"
    ARUCO = "aruco"  # ArUco marker-based homography calibration
    INTRINSICS = "intrinsics"  # ChArUco lens calibration


@dataclass
class CalibrationData:
    """Calibration data container."""
    # ROI: (x, y, width, height) in pixels
    roi: Optional[Tuple[int, int, int, int]] = None
    
    # Target line: two points defining forward direction
    target_line_p1: Optional[Tuple[int, int]] = None
    target_line_p2: Optional[Tuple[int, int]] = None
    
    # Scale: pixels per meter (legacy, replaced by homography when available)
    pixels_per_meter: float = 500.0  # Default estimate
    
    # Ground plane depth in meters
    ground_depth: float = 0.5  # Default estimate
    
    # Depth tolerance for ball detection (meters)
    depth_tolerance: float = 0.05  # 5cm
    
    # Forward direction angle (computed from target line)
    forward_angle_rad: float = 0.0
    
    # === NEW: ArUco Homography Calibration ===
    
    # Mat physical dimensions in meters
    mat_width_m: float = 0.70  # 70cm default
    mat_height_m: float = 1.00  # 1m visible area default
    
    # Homography matrix (3x3) - maps pixel coords to world coords (meters)
    # Stored as flat list for JSON serialization
    homography_matrix: Optional[List[float]] = None
    
    # Detected marker corners in pixel coordinates
    # Dict of marker_id -> corner_pixel_position (center of marker)
    marker_corners_px: Optional[Dict[int, Tuple[float, float]]] = None
    
    # World coordinates for each marker (in meters, relative to origin marker 0)
    marker_corners_world: Optional[Dict[int, Tuple[float, float]]] = None
    
    # ArUco marker size in meters (for pose estimation)
    aruco_marker_size_m: float = 0.05  # 5cm default
    
    # === NEW: Lens Undistortion (Optional) ===
    
    # Camera intrinsic matrix (3x3)
    camera_matrix: Optional[List[float]] = None
    
    # Distortion coefficients (5 or 8 values)
    dist_coeffs: Optional[List[float]] = None
    
    # Whether undistortion is enabled
    undistortion_enabled: bool = False
    
    # Undistortion map dimensions (for validation)
    undistort_map_size: Optional[Tuple[int, int]] = None
    
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
            # Homography fields
            'mat_width_m': self.mat_width_m,
            'mat_height_m': self.mat_height_m,
            'homography_matrix': self.homography_matrix,
            'marker_corners_px': {str(k): list(v) for k, v in self.marker_corners_px.items()} if self.marker_corners_px else None,
            'marker_corners_world': {str(k): list(v) for k, v in self.marker_corners_world.items()} if self.marker_corners_world else None,
            'aruco_marker_size_m': self.aruco_marker_size_m,
            # Intrinsics fields
            'camera_matrix': self.camera_matrix,
            'dist_coeffs': self.dist_coeffs,
            'undistortion_enabled': self.undistortion_enabled,
            'undistort_map_size': list(self.undistort_map_size) if self.undistort_map_size else None,
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
        
        # Homography fields
        cal.mat_width_m = data.get('mat_width_m', 0.70)
        cal.mat_height_m = data.get('mat_height_m', 1.00)
        cal.homography_matrix = data.get('homography_matrix')
        if data.get('marker_corners_px'):
            cal.marker_corners_px = {int(k): tuple(v) for k, v in data['marker_corners_px'].items()}
        if data.get('marker_corners_world'):
            cal.marker_corners_world = {int(k): tuple(v) for k, v in data['marker_corners_world'].items()}
        cal.aruco_marker_size_m = data.get('aruco_marker_size_m', 0.05)
        
        # Intrinsics fields
        cal.camera_matrix = data.get('camera_matrix')
        cal.dist_coeffs = data.get('dist_coeffs')
        cal.undistortion_enabled = data.get('undistortion_enabled', False)
        if data.get('undistort_map_size'):
            cal.undistort_map_size = tuple(data['undistort_map_size'])
        
        return cal
    
    def get_homography_matrix(self) -> Optional[np.ndarray]:
        """Get homography matrix as numpy array."""
        if self.homography_matrix is None:
            return None
        return np.array(self.homography_matrix).reshape(3, 3)
    
    def set_homography_matrix(self, H: np.ndarray):
        """Set homography matrix from numpy array."""
        self.homography_matrix = H.flatten().tolist()
    
    def get_camera_matrix(self) -> Optional[np.ndarray]:
        """Get camera intrinsic matrix as numpy array."""
        if self.camera_matrix is None:
            return None
        return np.array(self.camera_matrix).reshape(3, 3)
    
    def set_camera_matrix(self, K: np.ndarray):
        """Set camera intrinsic matrix from numpy array."""
        self.camera_matrix = K.flatten().tolist()
    
    def get_dist_coeffs(self) -> Optional[np.ndarray]:
        """Get distortion coefficients as numpy array."""
        if self.dist_coeffs is None:
            return None
        return np.array(self.dist_coeffs)
    
    def set_dist_coeffs(self, coeffs: np.ndarray):
        """Set distortion coefficients from numpy array."""
        self.dist_coeffs = coeffs.flatten().tolist()
    
    @property
    def has_homography(self) -> bool:
        """Check if valid homography calibration exists."""
        return (
            self.homography_matrix is not None and
            len(self.homography_matrix) == 9
        )
    
    @property
    def has_intrinsics(self) -> bool:
        """Check if valid camera intrinsics exist."""
        return (
            self.camera_matrix is not None and
            len(self.camera_matrix) == 9
        )


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
            if self.data.target_line_p1 and self.data.target_line_p2:
                print(f"[Calibration] Using custom target line, angle: {math.degrees(self.data.forward_angle_rad):.1f}°")
            else:
                print(f"[Calibration] Using default direction: RIGHT (0°)")
            return True
        except FileNotFoundError:
            print(f"[Calibration] No config file found at {load_path}, using defaults")
            self._compute_forward_angle()  # Ensure default forward angle is set
            print(f"[Calibration] Using default direction: RIGHT (0°)")
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
        else:
            # Default: forward is RIGHT in the camera frame (positive X)
            # This assumes putting left-to-right on camera
            self.data.forward_angle_rad = 0.0
    
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
        
        # Draw target line or default direction indicator
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
        else:
            # Draw default "forward is RIGHT" indicator
            h, w = frame.shape[:2]
            # Draw arrow in bottom-left corner showing direction
            arrow_start = (40, h - 40)
            arrow_end = (100, h - 40)
            cv2.arrowedLine(frame, arrow_start, arrow_end, (0, 255, 255), 2, tipLength=0.3)
            cv2.putText(frame, "TARGET", (35, h - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
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
        """
        Check if basic calibrations are complete.
        
        ROI and target line are now optional:
        - ROI defaults to full frame
        - Target direction defaults to RIGHT
        """
        return (
            self.data.pixels_per_meter > 0 and
            self.data.ground_depth > 0
        )
    
    @property
    def status_text(self) -> str:
        """Get calibration status as text."""
        status = []
        status.append(f"ROI: {'OK' if self.data.roi else 'Full'}")
        status.append(f"Target: {'Custom' if self.data.target_line_p1 else 'Right'}")
        if self.data.has_homography:
            status.append("Homography: OK")
        else:
            status.append(f"Scale: {self.data.pixels_per_meter:.0f} px/m")
        return " | ".join(status)


# =============================================================================
# ArUco-Based Homography Calibration
# =============================================================================

class ArucoCalibration:
    """
    ArUco marker-based homography calibration for accurate pixel→world mapping.
    
    AUTO-DETECTS 4 corner markers from any visible ArUco markers by finding
    the 4 markers closest to the image corners (convex hull approach).
    
    Features:
    - Real-time marker detection (any DICT_4X4_50 markers)
    - Auto-corner detection (no specific IDs required)
    - Homography computation from 4+ points
    - pixel_to_world coordinate transformation
    - Distance verification with known-distance markers
    - Optional lens undistortion support
    - Test Lab Mode visualization
    """
    
    # Corner marker IDs (auto-detected, but can be overridden)
    # These are updated when compute_homography_auto() is called
    CORNER_MARKER_IDS = [0, 1, 2, 3]
    
    # Distance verification marker IDs and their distances from origin (meters)
    # User has markers at 30cm, 60cm, 80cm
    DISTANCE_MARKERS = {
        10: 0.30,  # 30cm
        11: 0.60,  # 60cm  
        12: 0.80,  # 80cm
    }
    
    def __init__(self, calibration_data: CalibrationData):
        """
        Initialize ArUco calibration.
        
        Args:
            calibration_data: CalibrationData instance to store results.
        """
        self.data = calibration_data
        
        # ArUco detector setup
        self._aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
        
        # Detector parameters - optimized for detection reliability
        try:
            # OpenCV 4.7+ API
            self._detector_params = aruco.DetectorParameters()
            self._detector = aruco.ArucoDetector(self._aruco_dict, self._detector_params)
            self._use_new_api = True
        except AttributeError:
            # Older OpenCV API
            self._detector_params = aruco.DetectorParameters_create()
            self._detector = None
            self._use_new_api = False
        
        # Optimize detector parameters for speed and reliability
        self._detector_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self._detector_params.adaptiveThreshWinSizeMin = 3
        self._detector_params.adaptiveThreshWinSizeMax = 23
        self._detector_params.adaptiveThreshWinSizeStep = 10
        
        # Cached homography matrix (numpy)
        self._H: Optional[np.ndarray] = None
        self._H_inv: Optional[np.ndarray] = None
        
        # Undistortion maps (precomputed for speed)
        self._undistort_map1: Optional[np.ndarray] = None
        self._undistort_map2: Optional[np.ndarray] = None
        
        # Last detection results (for visualization)
        self._last_detected_markers: Dict[int, np.ndarray] = {}
        self._last_detection_frame_size: Optional[Tuple[int, int]] = None
        
        # Load existing homography if available
        self._load_homography()
    
    def _load_homography(self):
        """Load homography from calibration data."""
        H = self.data.get_homography_matrix()
        if H is not None:
            self._H = H
            try:
                self._H_inv = np.linalg.inv(H)
            except np.linalg.LinAlgError:
                print("[ArucoCalibration] Warning: Homography matrix is singular")
                self._H_inv = None
            print(f"[ArucoCalibration] Loaded homography for {self.data.mat_width_m}x{self.data.mat_height_m}m mat")
    
    def detect_markers(self, frame: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Detect ArUco markers in frame.
        
        Args:
            frame: BGR or grayscale image.
            
        Returns:
            Dict mapping marker_id -> corner points (4x2 array).
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self._last_detection_frame_size = (frame.shape[1], frame.shape[0])
        
        # Detect markers
        if self._use_new_api:
            corners, ids, rejected = self._detector.detectMarkers(gray)
        else:
            corners, ids, rejected = aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._detector_params
            )
        
        # Build result dict
        detected = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                detected[int(marker_id)] = corners[i][0]  # 4x2 array of corner points
        
        self._last_detected_markers = detected
        return detected
    
    def get_marker_centers(self, detected: Dict[int, np.ndarray]) -> Dict[int, Tuple[float, float]]:
        """
        Get center points of detected markers.
        
        Args:
            detected: Dict from detect_markers().
            
        Returns:
            Dict mapping marker_id -> (center_x, center_y).
        """
        centers = {}
        for marker_id, corners in detected.items():
            center = corners.mean(axis=0)
            centers[marker_id] = (float(center[0]), float(center[1]))
        return centers
    
    def compute_homography(
        self,
        detected_markers: Dict[int, np.ndarray],
        mat_width_m: Optional[float] = None,
        mat_height_m: Optional[float] = None,
        corner_ids: Optional[List[int]] = None
    ) -> bool:
        """
        Compute homography matrix from detected corner markers.
        
        Args:
            detected_markers: Dict from detect_markers().
            mat_width_m: Override mat width (meters).
            mat_height_m: Override mat height (meters).
            corner_ids: Optional list of 4 marker IDs to use as corners.
                       If None, uses auto-detection.
            
        Returns:
            True if homography was computed successfully.
        """
        # Use provided dimensions or fall back to stored values
        width_m = mat_width_m if mat_width_m is not None else self.data.mat_width_m
        height_m = mat_height_m if mat_height_m is not None else self.data.mat_height_m
        
        # Auto-detect corners if not specified
        if corner_ids is None:
            corner_ids = self.auto_detect_corners(detected_markers)
            if len(corner_ids) < 4:
                print(f"[ArucoCalibration] Only {len(corner_ids)} markers detected, need 4")
                return False
            print(f"[ArucoCalibration] Auto-detected corners: {corner_ids}")
        
        # Update class-level corner IDs
        self.CORNER_MARKER_IDS = corner_ids[:4]
        
        # Check if all 4 corner markers are detected
        missing = [mid for mid in corner_ids[:4] if mid not in detected_markers]
        if missing:
            print(f"[ArucoCalibration] Missing markers: {missing}")
            return False
        
        # Get marker centers in pixel coordinates
        pixel_centers = self.get_marker_centers(detected_markers)
        corner_pixels = [pixel_centers[mid] for mid in corner_ids[:4]]
        
        # Order corners by position: BL, BR, TR, TL
        # Sort by y first (bottom = high y), then by x
        sorted_corners = self._order_corners_bl_br_tr_tl(corner_pixels, corner_ids[:4])
        
        # Define world coordinates: BL=(0,0), BR=(w,0), TR=(w,h), TL=(0,h)
        world_coords_ordered = [
            (0.0, 0.0),           # BL
            (width_m, 0.0),       # BR
            (width_m, height_m),  # TR
            (0.0, height_m),      # TL
        ]
        
        pixel_pts = np.array([p for _, p in sorted_corners], dtype=np.float32)
        world_pts = np.array(world_coords_ordered, dtype=np.float32)
        
        # Compute homography: world = H @ pixel (with homogeneous coords)
        H, mask = cv2.findHomography(pixel_pts, world_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            print("[ArucoCalibration] Failed to compute homography")
            return False
        
        # Validate homography by checking reprojection error
        errors = []
        for i, (px, world) in enumerate(zip(pixel_pts, world_pts)):
            transformed = self._apply_homography(H, px[0], px[1])
            if transformed:
                error = np.sqrt((transformed[0] - world[0])**2 + (transformed[1] - world[1])**2)
                errors.append(error)
        
        if errors:
            max_error = max(errors)
            avg_error = sum(errors) / len(errors)
            print(f"[ArucoCalibration] Homography reprojection error: avg={avg_error*1000:.1f}mm, max={max_error*1000:.1f}mm")
            
            if max_error > 0.02:  # More than 2cm error
                print("[ArucoCalibration] Warning: High reprojection error, check marker placement")
        
        # Store results
        self._H = H
        try:
            self._H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("[ArucoCalibration] Warning: Homography matrix is singular")
            self._H_inv = None
        
        # Build marker corner dicts
        marker_corners_px = {mid: pixel_centers[mid] for mid in corner_ids[:4]}
        marker_corners_world = {
            sorted_corners[0][0]: (0.0, 0.0),
            sorted_corners[1][0]: (width_m, 0.0),
            sorted_corners[2][0]: (width_m, height_m),
            sorted_corners[3][0]: (0.0, height_m),
        }
        
        self.data.set_homography_matrix(H)
        self.data.mat_width_m = width_m
        self.data.mat_height_m = height_m
        self.data.marker_corners_px = marker_corners_px
        self.data.marker_corners_world = marker_corners_world
        
        print(f"[ArucoCalibration] Homography computed for {width_m}x{height_m}m mat")
        print(f"[ArucoCalibration] Corner markers: {[mid for mid, _ in sorted_corners]}")
        return True
    
    def _order_corners_bl_br_tr_tl(
        self, 
        corners: List[Tuple[float, float]], 
        ids: List[int]
    ) -> List[Tuple[int, Tuple[float, float]]]:
        """
        Order corner points as: bottom-left, bottom-right, top-right, top-left.
        
        In image coordinates, y increases downward, so "bottom" has higher y.
        """
        paired = list(zip(ids, corners))
        
        # Sort by y descending (bottom first), then by x for tie-breaking
        by_y = sorted(paired, key=lambda x: -x[1][1])  # Highest y first (bottom)
        
        # Bottom two (highest y)
        bottom = sorted(by_y[:2], key=lambda x: x[1][0])  # Sort by x: left, right
        # Top two (lowest y)
        top = sorted(by_y[2:], key=lambda x: -x[1][0])  # Sort by x desc: right, left
        
        # Order: BL, BR, TR, TL
        return [bottom[0], bottom[1], top[0], top[1]]
    
    def _apply_homography(self, H: np.ndarray, px: float, py: float) -> Optional[Tuple[float, float]]:
        """Apply homography transformation to a point."""
        pt = np.array([px, py, 1.0])
        transformed = H @ pt
        if abs(transformed[2]) < 1e-10:
            return None
        return (transformed[0] / transformed[2], transformed[1] / transformed[2])
    
    def pixel_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float]:
        """
        Transform pixel coordinates to world coordinates (meters).
        
        Uses homography if available, otherwise falls back to simple scaling.
        
        Args:
            pixel_x: X coordinate in pixels.
            pixel_y: Y coordinate in pixels.
            
        Returns:
            (world_x, world_y) in meters relative to marker 0 origin.
        """
        if self._H is not None:
            result = self._apply_homography(self._H, pixel_x, pixel_y)
            if result:
                return result
        
        # Fallback to legacy pixels_per_meter scaling
        roi = self.data.roi
        if roi:
            roi_center_x = roi[0] + roi[2] / 2
            roi_center_y = roi[1] + roi[3] / 2
        else:
            roi_center_x = pixel_x
            roi_center_y = pixel_y
        
        world_x = (pixel_x - roi_center_x) / self.data.pixels_per_meter
        world_y = (pixel_y - roi_center_y) / self.data.pixels_per_meter
        
        return (world_x, world_y)
    
    def world_to_pixel(self, world_x: float, world_y: float) -> Optional[Tuple[float, float]]:
        """
        Transform world coordinates to pixel coordinates.
        
        Args:
            world_x: X coordinate in meters.
            world_y: Y coordinate in meters.
            
        Returns:
            (pixel_x, pixel_y) or None if transformation failed.
        """
        if self._H_inv is not None:
            return self._apply_homography(self._H_inv, world_x, world_y)
        return None
    
    def test_distance(
        self,
        point1_px: Tuple[float, float],
        point2_px: Tuple[float, float]
    ) -> Tuple[float, float, float]:
        """
        Measure distance between two pixel points in world coordinates.
        
        Args:
            point1_px: First point (pixel_x, pixel_y).
            point2_px: Second point (pixel_x, pixel_y).
            
        Returns:
            (distance_m, dx_m, dy_m) - total distance and components in meters.
        """
        world1 = self.pixel_to_world(*point1_px)
        world2 = self.pixel_to_world(*point2_px)
        
        dx = world2[0] - world1[0]
        dy = world2[1] - world1[1]
        distance = math.sqrt(dx*dx + dy*dy)
        
        return (distance, dx, dy)
    
    def run_distance_tests(self, detected_markers: Dict[int, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """
        Run distance verification tests using distance markers.
        
        Uses DISTANCE_MARKERS dict for expected distances.
        Measures from the closest corner marker to each distance marker.
        
        Args:
            detected_markers: Dict from detect_markers().
            
        Returns:
            Dict of test results: {"0.30m": {"expected": 0.3, "measured": 0.302, "error_mm": 2}, ...}
        """
        results = {}
        
        # Find origin - use first corner marker or closest to bottom-left
        origin_marker_id = None
        origin_center = None
        
        for mid in self.CORNER_MARKER_IDS:
            if mid in detected_markers:
                origin_marker_id = mid
                origin_center = self.get_marker_centers({mid: detected_markers[mid]})[mid]
                break
        
        if origin_center is None:
            # Fallback: use any marker with lowest y coordinate (bottom of frame)
            if detected_markers:
                centers = self.get_marker_centers(detected_markers)
                # Find marker closest to bottom-left
                bottom_markers = sorted(centers.items(), key=lambda x: (x[1][1], -x[1][0]))  # Sort by y desc, x asc
                if bottom_markers:
                    origin_marker_id = bottom_markers[-1][0]  # Highest y (bottom)
                    origin_center = bottom_markers[-1][1]
        
        if origin_center is None:
            return {"error": {"message": "No markers detected for distance tests"}}
        
        results["origin_marker_id"] = origin_marker_id
        
        # Test each distance marker
        for marker_id, expected_m in self.DISTANCE_MARKERS.items():
            if marker_id in detected_markers:
                marker_center = self.get_marker_centers({marker_id: detected_markers[marker_id]})[marker_id]
                
                measured_m, dx, dy = self.test_distance(origin_center, marker_center)
                error_mm = (measured_m - expected_m) * 1000
                
                results[f"{expected_m:.2f}m"] = {
                    "expected": expected_m,
                    "measured": round(measured_m, 4),
                    "error_mm": round(error_mm, 1),
                    "dx_m": round(dx, 4),
                    "dy_m": round(dy, 4),
                    "marker_id": marker_id,
                }
        
        return results
    
    def auto_detect_corners(self, detected_markers: Dict[int, np.ndarray]) -> List[int]:
        """
        Auto-detect which 4 markers are at the mat corners.
        
        Uses convex hull / bounding box approach to find the 4 markers
        furthest from the centroid (most likely to be corners).
        
        Args:
            detected_markers: Dict from detect_markers().
            
        Returns:
            List of 4 marker IDs representing corners, ordered:
            [bottom-left, bottom-right, top-right, top-left]
        """
        if len(detected_markers) < 4:
            return []
        
        # Get marker centers
        centers = self.get_marker_centers(detected_markers)
        
        # Compute centroid
        all_points = np.array(list(centers.values()))
        centroid = all_points.mean(axis=0)
        
        # Find 4 markers furthest from centroid
        distances = []
        for marker_id, center in centers.items():
            dist = np.sqrt((center[0] - centroid[0])**2 + (center[1] - centroid[1])**2)
            distances.append((marker_id, center, dist))
        
        # Sort by distance, take top 4
        distances.sort(key=lambda x: x[2], reverse=True)
        corner_candidates = distances[:4]
        
        # Order corners: bottom-left, bottom-right, top-right, top-left
        # (assuming image coordinates: y increases downward)
        corners = [(mid, c) for mid, c, _ in corner_candidates]
        
        # Sort by angle from centroid
        def angle_from_centroid(item):
            mid, c = item
            return math.atan2(c[1] - centroid[1], c[0] - centroid[0])
        
        corners.sort(key=angle_from_centroid)
        
        # Reorder to [BL, BR, TR, TL] convention
        # Find bottom-left (max y, min x quadrant)
        ordered = []
        for mid, c in corners:
            ordered.append(mid)
        
        return ordered
    
    def draw_overlays(
        self,
        frame: np.ndarray,
        show_grid: bool = True,
        show_markers: bool = True,
        show_labels: bool = True
    ) -> np.ndarray:
        """
        Draw ArUco calibration overlays on frame.
        
        Args:
            frame: Input frame (will be modified).
            show_grid: Draw world coordinate grid.
            show_markers: Draw detected marker corners.
            show_labels: Draw marker ID labels.
            
        Returns:
            Frame with overlays.
        """
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw detected markers
        if show_markers and self._last_detected_markers:
            for marker_id, corners in self._last_detected_markers.items():
                # Draw marker outline
                corners_int = corners.astype(np.int32)
                
                # Color code: green for corner markers, blue for distance markers
                if marker_id in self.CORNER_MARKER_IDS:
                    color = (0, 255, 0)  # Green
                elif marker_id in self.DISTANCE_MARKERS:
                    color = (255, 165, 0)  # Orange
                else:
                    color = (255, 0, 0)  # Blue for others
                
                cv2.polylines(frame, [corners_int], True, color, 2)
                
                # Draw center point
                center = corners.mean(axis=0).astype(int)
                cv2.circle(frame, tuple(center), 5, color, -1)
                
                # Draw label
                if show_labels:
                    label = f"ID:{marker_id}"
                    cv2.putText(frame, label, (center[0] + 10, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw world coordinate grid
        if show_grid and self._H_inv is not None:
            self._draw_world_grid(frame)
        
        # Draw calibration status
        if self.data.has_homography:
            status = f"Homography: {self.data.mat_width_m}x{self.data.mat_height_m}m"
            color = (0, 255, 0)
        else:
            status = "Homography: Not calibrated"
            color = (0, 165, 255)
        
        cv2.putText(frame, status, (10, frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def _draw_world_grid(self, frame: np.ndarray, grid_spacing_m: float = 0.10):
        """Draw world coordinate grid on frame."""
        if self._H_inv is None:
            return
        
        # Draw grid lines at 10cm intervals
        for x_m in np.arange(0, self.data.mat_width_m + grid_spacing_m, grid_spacing_m):
            # Vertical line from (x, 0) to (x, height)
            p1 = self.world_to_pixel(x_m, 0)
            p2 = self.world_to_pixel(x_m, self.data.mat_height_m)
            
            if p1 and p2:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                        (100, 100, 100), 1)
        
        for y_m in np.arange(0, self.data.mat_height_m + grid_spacing_m, grid_spacing_m):
            # Horizontal line from (0, y) to (width, y)
            p1 = self.world_to_pixel(0, y_m)
            p2 = self.world_to_pixel(self.data.mat_width_m, y_m)
            
            if p1 and p2:
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])),
                        (100, 100, 100), 1)
        
        # Draw origin marker (larger)
        origin = self.world_to_pixel(0, 0)
        if origin:
            cv2.circle(frame, (int(origin[0]), int(origin[1])), 8, (0, 0, 255), -1)
            cv2.putText(frame, "ORIGIN", (int(origin[0]) + 12, int(origin[1]) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        
        # Draw 0.5m marks
        for dist in [0.5, 1.0]:
            if dist <= self.data.mat_height_m:
                p = self.world_to_pixel(self.data.mat_width_m / 2, dist)
                if p:
                    cv2.putText(frame, f"{dist}m", (int(p[0]) + 5, int(p[1])),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def get_calibration_status(self) -> Dict[str, Any]:
        """
        Get current calibration status for WebSocket broadcast.
        
        Returns:
            Dict with calibration status suitable for JSON serialization.
        """
        detected_count = len([mid for mid in self._last_detected_markers if mid in self.CORNER_MARKER_IDS])
        
        return {
            "type": "calibration_status",
            "markers_detected": detected_count,
            "markers_required": 4,
            "marker_ids_detected": list(self._last_detected_markers.keys()),
            "homography_valid": self.data.has_homography,
            "mat_dimensions": {
                "width_m": self.data.mat_width_m,
                "height_m": self.data.mat_height_m,
            },
            "intrinsics_valid": self.data.has_intrinsics,
            "undistortion_enabled": self.data.undistortion_enabled,
        }
    
    @property
    def is_ready(self) -> bool:
        """Check if ArUco calibration is ready for use."""
        return self._H is not None
    
    @property
    def markers_detected_count(self) -> int:
        """Get count of detected corner markers."""
        return len([mid for mid in self._last_detected_markers if mid in self.CORNER_MARKER_IDS])


# =============================================================================
# Camera Intrinsics Calibration (Lens Undistortion)
# =============================================================================

class IntrinsicsCalibration:
    """
    Camera intrinsics calibration using ChArUco board.
    
    Computes camera matrix and distortion coefficients for lens undistortion.
    Uses ChArUco board (6x9 squares, 25mm squares, 18mm markers).
    
    Usage:
        1. Create IntrinsicsCalibration instance
        2. Collect 10-20 frames of ChArUco board at different angles
        3. Call compute_calibration()
        4. Use undistort_frame() during runtime
    """
    
    # ChArUco board parameters (must match printed board)
    CHARUCO_SQUARES_X = 6
    CHARUCO_SQUARES_Y = 9
    CHARUCO_SQUARE_LENGTH_M = 0.025  # 25mm
    CHARUCO_MARKER_LENGTH_M = 0.018  # 18mm
    
    def __init__(self, calibration_data: CalibrationData):
        """
        Initialize intrinsics calibration.
        
        Args:
            calibration_data: CalibrationData instance to store results.
        """
        self.data = calibration_data
        
        # ArUco dictionary (same as markers)
        self._aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
        
        # ChArUco board
        try:
            # OpenCV 4.7+ API
            self._charuco_board = aruco.CharucoBoard(
                (self.CHARUCO_SQUARES_X, self.CHARUCO_SQUARES_Y),
                self.CHARUCO_SQUARE_LENGTH_M,
                self.CHARUCO_MARKER_LENGTH_M,
                self._aruco_dict
            )
            self._detector_params = aruco.DetectorParameters()
            self._charuco_detector = aruco.CharucoDetector(self._charuco_board)
            self._use_new_api = True
        except (AttributeError, TypeError):
            # Older OpenCV API
            self._charuco_board = aruco.CharucoBoard_create(
                self.CHARUCO_SQUARES_X,
                self.CHARUCO_SQUARES_Y,
                self.CHARUCO_SQUARE_LENGTH_M,
                self.CHARUCO_MARKER_LENGTH_M,
                self._aruco_dict
            )
            self._detector_params = aruco.DetectorParameters_create()
            self._charuco_detector = None
            self._use_new_api = False
        
        # Collected calibration data
        self._all_charuco_corners: List[np.ndarray] = []
        self._all_charuco_ids: List[np.ndarray] = []
        self._image_size: Optional[Tuple[int, int]] = None
        
        # Undistortion maps (precomputed)
        self._undistort_map1: Optional[np.ndarray] = None
        self._undistort_map2: Optional[np.ndarray] = None
        
        # Load existing calibration if available
        self._load_intrinsics()
    
    def _load_intrinsics(self):
        """Load intrinsics from calibration data and compute undistortion maps."""
        K = self.data.get_camera_matrix()
        dist = self.data.get_dist_coeffs()
        
        if K is not None and dist is not None and self.data.undistort_map_size:
            self._compute_undistort_maps(
                K, dist,
                self.data.undistort_map_size[0],
                self.data.undistort_map_size[1]
            )
            print(f"[IntrinsicsCalibration] Loaded intrinsics for {self.data.undistort_map_size[0]}x{self.data.undistort_map_size[1]}")
    
    def detect_charuco(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """
        Detect ChArUco board corners in frame.
        
        Args:
            frame: BGR or grayscale image.
            
        Returns:
            (charuco_corners, charuco_ids, num_corners) or (None, None, 0) if not detected.
        """
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        self._image_size = (gray.shape[1], gray.shape[0])
        
        if self._use_new_api:
            # OpenCV 4.7+ API
            charuco_corners, charuco_ids, marker_corners, marker_ids = \
                self._charuco_detector.detectBoard(gray)
        else:
            # Older API
            marker_corners, marker_ids, rejected = aruco.detectMarkers(
                gray, self._aruco_dict, parameters=self._detector_params
            )
            
            if marker_ids is None or len(marker_ids) < 4:
                return None, None, 0
            
            ret, charuco_corners, charuco_ids = aruco.interpolateCornersCharuco(
                marker_corners, marker_ids, gray, self._charuco_board
            )
            
            if not ret or charuco_corners is None:
                return None, None, 0
        
        if charuco_corners is None or charuco_ids is None:
            return None, None, 0
        
        num_corners = len(charuco_ids)
        return charuco_corners, charuco_ids, num_corners
    
    def add_calibration_frame(self, frame: np.ndarray) -> Tuple[bool, int, str]:
        """
        Add a frame for calibration.
        
        Args:
            frame: BGR or grayscale image with ChArUco board visible.
            
        Returns:
            (success, num_corners, message)
        """
        charuco_corners, charuco_ids, num_corners = self.detect_charuco(frame)
        
        if charuco_corners is None or num_corners < 6:
            return False, 0, "Not enough corners detected (need at least 6)"
        
        self._all_charuco_corners.append(charuco_corners)
        self._all_charuco_ids.append(charuco_ids)
        
        return True, num_corners, f"Added frame {len(self._all_charuco_corners)} with {num_corners} corners"
    
    def get_collection_status(self) -> Dict[str, Any]:
        """Get status of calibration frame collection."""
        return {
            "frames_collected": len(self._all_charuco_corners),
            "frames_recommended": 15,
            "frames_minimum": 5,
            "ready_to_calibrate": len(self._all_charuco_corners) >= 5,
        }
    
    def compute_calibration(self) -> Tuple[bool, str]:
        """
        Compute camera intrinsics from collected frames.
        
        Returns:
            (success, message)
        """
        if len(self._all_charuco_corners) < 5:
            return False, f"Need at least 5 frames, have {len(self._all_charuco_corners)}"
        
        if self._image_size is None:
            return False, "No image size recorded"
        
        print(f"[IntrinsicsCalibration] Computing calibration from {len(self._all_charuco_corners)} frames...")
        
        # Calibrate camera
        try:
            if self._use_new_api:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                    self._all_charuco_corners,
                    self._all_charuco_ids,
                    self._charuco_board,
                    self._image_size,
                    None, None
                )
            else:
                ret, camera_matrix, dist_coeffs, rvecs, tvecs = aruco.calibrateCameraCharuco(
                    self._all_charuco_corners,
                    self._all_charuco_ids,
                    self._charuco_board,
                    self._image_size,
                    None, None
                )
        except cv2.error as e:
            return False, f"OpenCV calibration error: {e}"
        
        if not ret:
            return False, "Calibration failed"
        
        # Store results
        self.data.set_camera_matrix(camera_matrix)
        self.data.set_dist_coeffs(dist_coeffs.flatten())
        self.data.undistort_map_size = self._image_size
        
        # Compute undistortion maps
        self._compute_undistort_maps(
            camera_matrix, dist_coeffs,
            self._image_size[0], self._image_size[1]
        )
        
        # Calculate reprojection error
        total_error = 0
        for i in range(len(self._all_charuco_corners)):
            # Project points back
            obj_points = self._charuco_board.getChessboardCorners()[self._all_charuco_ids[i].flatten()]
            img_points, _ = cv2.projectPoints(
                obj_points, rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(self._all_charuco_corners[i], img_points, cv2.NORM_L2) / len(img_points)
            total_error += error
        
        mean_error = total_error / len(self._all_charuco_corners)
        
        print(f"[IntrinsicsCalibration] Calibration complete!")
        print(f"[IntrinsicsCalibration] Mean reprojection error: {mean_error:.4f} pixels")
        print(f"[IntrinsicsCalibration] Camera matrix:\n{camera_matrix}")
        
        # Clear collected frames
        self._all_charuco_corners = []
        self._all_charuco_ids = []
        
        return True, f"Calibration complete. Mean error: {mean_error:.4f} pixels"
    
    def _compute_undistort_maps(self, K: np.ndarray, dist: np.ndarray, width: int, height: int):
        """Precompute undistortion maps for fast remapping."""
        # Get optimal new camera matrix
        new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), 1, (width, height))
        
        # Compute undistortion maps
        self._undistort_map1, self._undistort_map2 = cv2.initUndistortRectifyMap(
            K, dist, None, new_K, (width, height), cv2.CV_32FC1
        )
        
        print(f"[IntrinsicsCalibration] Undistortion maps computed for {width}x{height}")
    
    def undistort_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Undistort a frame using precomputed maps.
        
        Very fast (~1ms) due to precomputed maps.
        
        Args:
            frame: Input frame (any format).
            
        Returns:
            Undistorted frame.
        """
        if not self.data.undistortion_enabled:
            return frame
        
        if self._undistort_map1 is None or self._undistort_map2 is None:
            return frame
        
        return cv2.remap(frame, self._undistort_map1, self._undistort_map2, cv2.INTER_LINEAR)
    
    def enable_undistortion(self, enabled: bool = True):
        """Enable or disable undistortion."""
        if enabled and not self.data.has_intrinsics:
            print("[IntrinsicsCalibration] Cannot enable undistortion without calibration")
            return
        
        self.data.undistortion_enabled = enabled
        print(f"[IntrinsicsCalibration] Undistortion {'enabled' if enabled else 'disabled'}")
    
    def draw_detected_corners(self, frame: np.ndarray) -> np.ndarray:
        """Draw detected ChArUco corners on frame."""
        if len(frame.shape) == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        charuco_corners, charuco_ids, num_corners = self.detect_charuco(frame)
        
        if charuco_corners is not None and charuco_ids is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
            cv2.putText(frame, f"Corners: {num_corners}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No ChArUco detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return frame
    
    def clear_collected_frames(self):
        """Clear collected calibration frames."""
        self._all_charuco_corners = []
        self._all_charuco_ids = []
        print("[IntrinsicsCalibration] Cleared collected frames")
    
    @property
    def is_ready(self) -> bool:
        """Check if intrinsics calibration is ready for undistortion."""
        return (
            self.data.has_intrinsics and
            self._undistort_map1 is not None and
            self._undistort_map2 is not None
        )
    
    @property
    def frames_collected(self) -> int:
        """Get number of collected calibration frames."""
        return len(self._all_charuco_corners)
