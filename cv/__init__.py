"""
Computer Vision module for golf ball tracking.

Supports multiple camera types:
- ZED 2i (high-speed, 100 FPS)
- Intel RealSense D455
- Standard USB webcams
"""

# Camera imports (with fallbacks for missing dependencies)
try:
    from .realsense_camera import RealsenseCamera
except ImportError:
    RealsenseCamera = None

from .webcam_camera import WebcamCamera
from .zed_camera import ZedCamera

# Core modules
from .calibration import Calibration
from .ball_tracker import BallTracker, TrackingResult
from .color_ball_tracker import ColorBallTracker
from .auto_ball_tracker import AutoBallTracker, TrackerState
from .shot_detector import ShotDetector
from .camera_manager import MultiCameraManager, CameraRole

__all__ = [
    # Cameras
    'RealsenseCamera',
    'WebcamCamera',
    'ZedCamera',
    'MultiCameraManager',
    'CameraRole',
    # Core
    'Calibration', 
    'BallTracker',
    'ColorBallTracker',
    'AutoBallTracker',
    'TrackerState',
    'TrackingResult',
    'ShotDetector',
]
