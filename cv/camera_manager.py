"""
Multi-Camera Manager for Golf Putting Monitor.

Manages multiple cameras simultaneously:
- ZED 2i: Primary high-speed tracking (87+ FPS)
- RealSense D455: Depth calibration
- iPhone/Webcam: Replay recording

Each camera serves a specific purpose in the system.
"""

import time
import threading
import queue
from typing import Optional, Dict, List, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

import numpy as np
import cv2


class CameraRole(Enum):
    """Role of each camera in the system."""
    PRIMARY_TRACKER = "primary_tracker"  # High-speed ball tracking (ZED)
    DEPTH_CALIBRATION = "depth_calibration"  # Depth sensing (RealSense)
    REPLAY_RECORDER = "replay_recorder"  # Video recording (iPhone)
    VALIDATION = "validation"  # Position cross-validation


@dataclass
class CameraInfo:
    """Information about a detected camera."""
    index: int
    name: str
    width: int
    height: int
    fps: float
    is_stereo: bool = False
    single_eye_width: int = 0
    backend: str = ""
    estimated_type: str = "unknown"  # zed, realsense, iphone, webcam


@dataclass
class MultiFrameData:
    """Frame data from multiple cameras."""
    timestamp: float
    
    # Primary tracker (ZED)
    tracker_frame: Optional[np.ndarray] = None
    tracker_gray: Optional[np.ndarray] = None
    tracker_fps: float = 0.0
    
    # Depth camera (RealSense)
    depth_frame: Optional[np.ndarray] = None
    depth_color: Optional[np.ndarray] = None
    depth_scale: float = 0.001
    
    # Replay camera (iPhone)
    replay_frame: Optional[np.ndarray] = None
    
    # Metadata
    frame_number: int = 0


@dataclass
class CalibrationData:
    """Calibration data from depth camera."""
    pixels_per_meter: float = 350.0
    ground_depth: float = 0.5
    depth_tolerance: float = 0.05
    ball_diameter_pixels: float = 30.0
    calibration_timestamp: str = ""
    calibration_source: str = "manual"


class CameraThread:
    """Thread wrapper for camera capture."""
    
    def __init__(
        self, 
        camera_index: int,
        role: CameraRole,
        target_fps: int = 30,
        resolution: Tuple[int, int] = (1280, 720),
        is_stereo: bool = False,
        use_right_eye: bool = True  # Right eye by default for ZED
    ):
        self.camera_index = camera_index
        self.role = role
        self.target_fps = target_fps
        self.resolution = resolution
        self.is_stereo = is_stereo
        self.use_right_eye = use_right_eye
        
        self.cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=3)
        
        self._frame_count = 0
        self._fps = 0.0
        self._last_fps_time = time.time()
        self._fps_frame_count = 0
    
    def start(self) -> bool:
        """Start camera capture thread."""
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                print(f"[CamThread] Failed to open camera {self.camera_index}")
                return False
            
            # Set resolution and FPS
            if self.is_stereo:
                # ZED stereo mode
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0] * 2)
            else:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            
            # Warmup
            for _ in range(10):
                self.cap.read()
            
            self._running = True
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()
            
            print(f"[CamThread] Started camera {self.camera_index} as {self.role.value}")
            return True
            
        except Exception as e:
            print(f"[CamThread] Error starting camera {self.camera_index}: {e}")
            return False
    
    def stop(self):
        """Stop camera capture thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
        print(f"[CamThread] Stopped camera {self.camera_index}")
    
    def _capture_loop(self):
        """Main capture loop running in thread."""
        while self._running:
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                continue
            
            timestamp = time.time()
            self._frame_count += 1
            self._fps_frame_count += 1
            
            # Update FPS calculation
            elapsed = timestamp - self._last_fps_time
            if elapsed >= 1.0:
                self._fps = self._fps_frame_count / elapsed
                self._fps_frame_count = 0
                self._last_fps_time = timestamp
            
            # For stereo cameras, extract single eye
            if self.is_stereo:
                half_width = frame.shape[1] // 2
                if self.use_right_eye:
                    frame = frame[:, half_width:].copy()  # Right eye
                else:
                    frame = frame[:, :half_width].copy()  # Left eye
            
            # Put in queue (drop old frames if full)
            try:
                if self._frame_queue.full():
                    try:
                        self._frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._frame_queue.put_nowait((frame, timestamp, self._frame_count))
            except queue.Full:
                pass
    
    def get_frame(self) -> Optional[Tuple[np.ndarray, float, int]]:
        """Get latest frame from queue."""
        try:
            return self._frame_queue.get_nowait()
        except queue.Empty:
            return None
    
    @property
    def fps(self) -> float:
        return self._fps
    
    @property
    def is_running(self) -> bool:
        return self._running


class ReplayRecorder:
    """Records video from replay camera for shot playback."""
    
    def __init__(self, output_dir: str = "replays"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._current_file: Optional[Path] = None
        self._frame_buffer: List[np.ndarray] = []
        self._pre_buffer_seconds = 2.0  # Keep 2 seconds before shot
        self._post_buffer_seconds = 3.0  # Record 3 seconds after shot
        self._fps = 30.0
        self._max_pre_frames = int(self._pre_buffer_seconds * self._fps)
        
        self._shot_triggered = False
        self._shot_time = 0.0
    
    def add_frame(self, frame: np.ndarray, timestamp: float):
        """Add frame to buffer."""
        # Always keep pre-buffer
        self._frame_buffer.append((frame.copy(), timestamp))
        
        # Trim pre-buffer
        while len(self._frame_buffer) > self._max_pre_frames:
            if not self._shot_triggered:
                self._frame_buffer.pop(0)
            else:
                break
        
        # If shot triggered, check if we should stop
        if self._shot_triggered:
            elapsed = timestamp - self._shot_time
            if elapsed >= self._post_buffer_seconds:
                self._save_recording()
    
    def trigger_shot(self, shot_data: dict):
        """Trigger shot recording."""
        self._shot_triggered = True
        self._shot_time = time.time()
        self._shot_data = shot_data
        print(f"[Replay] Recording triggered - capturing {self._post_buffer_seconds}s of footage")
    
    def _save_recording(self):
        """Save buffered frames to video file."""
        if not self._frame_buffer:
            self._shot_triggered = False
            return
        
        # Generate filename
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        speed = self._shot_data.get('speed_mps', 0)
        direction = self._shot_data.get('direction_deg', 0)
        filename = f"putt_{timestamp_str}_{speed:.1f}mps_{direction:+.0f}deg.mp4"
        filepath = self.output_dir / filename
        
        # Get frame size from first frame
        first_frame = self._frame_buffer[0][0]
        h, w = first_frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(filepath), fourcc, self._fps, (w, h))
        
        # Write all frames
        for frame, _ in self._frame_buffer:
            writer.write(frame)
        
        writer.release()
        
        print(f"[Replay] Saved: {filepath} ({len(self._frame_buffer)} frames)")
        
        # Reset
        self._frame_buffer = []
        self._shot_triggered = False
    
    def set_fps(self, fps: float):
        """Set recording FPS."""
        self._fps = fps
        self._max_pre_frames = int(self._pre_buffer_seconds * fps)


class DepthCalibrator:
    """Uses RealSense depth data for automatic calibration."""
    
    def __init__(self):
        self.calibration_frames: List[Tuple[np.ndarray, np.ndarray]] = []
        self._calibrating = False
        self._target_frames = 30
        
        # Calibration results
        self.ground_depth = 0.5
        self.depth_tolerance = 0.05
        self.pixels_per_meter = 350.0
    
    def start_calibration(self):
        """Start collecting calibration frames."""
        self._calibrating = True
        self.calibration_frames = []
        print("[DepthCal] Starting depth calibration - collecting frames...")
    
    def add_frame(self, color_frame: np.ndarray, depth_frame: np.ndarray) -> bool:
        """
        Add a calibration frame.
        
        Returns True when calibration is complete.
        """
        if not self._calibrating:
            return False
        
        self.calibration_frames.append((color_frame.copy(), depth_frame.copy()))
        
        if len(self.calibration_frames) >= self._target_frames:
            self._compute_calibration()
            return True
        
        print(f"[DepthCal] Frame {len(self.calibration_frames)}/{self._target_frames}")
        return False
    
    def _compute_calibration(self):
        """Compute calibration from collected frames."""
        print("[DepthCal] Computing calibration...")
        
        # Stack all depth frames
        depth_stack = np.stack([d for _, d in self.calibration_frames], axis=0)
        
        # Find ground plane (most common depth value in center region)
        h, w = depth_stack.shape[1:3]
        center_region = depth_stack[:, h//3:2*h//3, w//3:2*w//3]
        
        # Filter out zeros and invalid values
        valid_depths = center_region[center_region > 0]
        
        if len(valid_depths) > 0:
            # Use median for robustness
            self.ground_depth = float(np.median(valid_depths)) / 1000.0  # Convert mm to m
            
            # Compute depth tolerance from variance
            depth_std = float(np.std(valid_depths)) / 1000.0
            self.depth_tolerance = max(0.02, depth_std * 2)
            
            print(f"[DepthCal] Ground depth: {self.ground_depth:.3f}m")
            print(f"[DepthCal] Depth tolerance: {self.depth_tolerance:.3f}m")
        
        self._calibrating = False
        self.calibration_frames = []
    
    def get_calibration_data(self) -> CalibrationData:
        """Get current calibration data."""
        return CalibrationData(
            pixels_per_meter=self.pixels_per_meter,
            ground_depth=self.ground_depth,
            depth_tolerance=self.depth_tolerance,
            calibration_timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            calibration_source="realsense_depth"
        )


class PositionValidator:
    """Validates ball position across multiple cameras."""
    
    def __init__(self, max_deviation_pixels: float = 20.0):
        self.max_deviation = max_deviation_pixels
        self._positions: Dict[str, Tuple[float, float, float]] = {}  # camera -> (x, y, timestamp)
    
    def add_position(self, camera_id: str, x: float, y: float, timestamp: float):
        """Add a detected position from a camera."""
        self._positions[camera_id] = (x, y, timestamp)
    
    def validate(self, max_time_diff: float = 0.05) -> Tuple[bool, float, Tuple[float, float]]:
        """
        Validate positions across cameras.
        
        Returns:
            (is_valid, confidence, averaged_position)
        """
        if len(self._positions) < 2:
            # Single camera - trust it
            if self._positions:
                pos = list(self._positions.values())[0]
                return True, 0.8, (pos[0], pos[1])
            return False, 0.0, (0, 0)
        
        # Check time synchronization
        timestamps = [p[2] for p in self._positions.values()]
        time_spread = max(timestamps) - min(timestamps)
        
        if time_spread > max_time_diff:
            # Frames too far apart - use most recent
            latest = max(self._positions.items(), key=lambda x: x[1][2])
            return True, 0.6, (latest[1][0], latest[1][1])
        
        # Compute average position
        positions = [(p[0], p[1]) for p in self._positions.values()]
        avg_x = sum(p[0] for p in positions) / len(positions)
        avg_y = sum(p[1] for p in positions) / len(positions)
        
        # Check deviation
        deviations = [
            np.sqrt((p[0] - avg_x)**2 + (p[1] - avg_y)**2)
            for p in positions
        ]
        max_dev = max(deviations)
        
        if max_dev > self.max_deviation:
            # Positions don't agree - lower confidence
            confidence = max(0.3, 1.0 - (max_dev / self.max_deviation) * 0.5)
            return True, confidence, (avg_x, avg_y)
        
        # Positions agree - high confidence
        confidence = min(1.0, 0.9 + (1.0 - max_dev / self.max_deviation) * 0.1)
        return True, confidence, (avg_x, avg_y)
    
    def clear(self):
        """Clear stored positions."""
        self._positions.clear()


class MultiCameraManager:
    """
    Manages multiple cameras for the putting monitor.
    
    Camera roles:
    - PRIMARY_TRACKER: ZED 2i for high-speed ball tracking
    - DEPTH_CALIBRATION: RealSense for depth-based calibration
    - REPLAY_RECORDER: iPhone for video replay recording
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # Camera threads
        self._cameras: Dict[CameraRole, CameraThread] = {}
        
        # Components
        self.replay_recorder = ReplayRecorder()
        self.depth_calibrator = DepthCalibrator()
        self.position_validator = PositionValidator()
        
        # State
        self._running = False
        self._frame_number = 0
        
        # Detected cameras
        self._available_cameras: List[CameraInfo] = []
        
        # Callbacks
        self._on_calibration_complete: Optional[Callable] = None
    
    def detect_cameras(self) -> List[CameraInfo]:
        """Detect and classify all available cameras."""
        print("\n[MultiCam] Detecting cameras...")
        
        cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if not cap.isOpened():
                cap.release()
                continue
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            
            cap.release()
            
            # Classify camera
            aspect = width / height if height > 0 else 0
            is_stereo = aspect > 3.0
            
            # Estimate camera type
            if is_stereo:
                cam_type = "zed"
            elif width == 1280 and height == 960:
                cam_type = "realsense"
            elif fps >= 30 and width >= 1920:
                cam_type = "iphone"
            else:
                cam_type = "webcam"
            
            info = CameraInfo(
                index=i,
                name=f"Camera {i}",
                width=width,
                height=height,
                fps=fps,
                is_stereo=is_stereo,
                single_eye_width=width // 2 if is_stereo else width,
                backend=backend,
                estimated_type=cam_type
            )
            cameras.append(info)
            
            print(f"  [{i}] {cam_type.upper()}: {width}x{height} @ {fps:.0f}fps")
        
        self._available_cameras = cameras
        return cameras
    
    def auto_assign_roles(self) -> Dict[CameraRole, int]:
        """
        Automatically assign camera roles based on detection.
        
        Returns dict of role -> camera_index.
        """
        assignments = {}
        
        # Find ZED for primary tracking
        for cam in self._available_cameras:
            if cam.estimated_type == "zed":
                assignments[CameraRole.PRIMARY_TRACKER] = cam.index
                break
        
        # Find RealSense for depth
        for cam in self._available_cameras:
            if cam.estimated_type == "realsense":
                assignments[CameraRole.DEPTH_CALIBRATION] = cam.index
                break
        
        # Find iPhone/high-res for replay
        for cam in self._available_cameras:
            if cam.estimated_type == "iphone":
                assignments[CameraRole.REPLAY_RECORDER] = cam.index
                break
        
        # Find another camera for validation (if available)
        used_indices = set(assignments.values())
        for cam in self._available_cameras:
            if cam.index not in used_indices and cam.fps >= 25:
                assignments[CameraRole.VALIDATION] = cam.index
                break
        
        print("\n[MultiCam] Auto-assigned roles:")
        for role, idx in assignments.items():
            cam = next(c for c in self._available_cameras if c.index == idx)
            print(f"  {role.value}: Camera [{idx}] ({cam.estimated_type})")
        
        return assignments
    
    def start(
        self,
        assignments: Optional[Dict[CameraRole, int]] = None,
        tracker_fps: int = 100,
        tracker_resolution: Tuple[int, int] = (672, 376)
    ) -> bool:
        """
        Start all assigned cameras.
        
        Args:
            assignments: Dict of CameraRole -> camera index
            tracker_fps: Target FPS for primary tracker
            tracker_resolution: Resolution for primary tracker (single eye if stereo)
        """
        if assignments is None:
            self.detect_cameras()
            assignments = self.auto_assign_roles()
        
        if not assignments:
            print("[MultiCam] No cameras assigned!")
            return False
        
        # Start each camera
        for role, cam_idx in assignments.items():
            cam_info = next((c for c in self._available_cameras if c.index == cam_idx), None)
            if not cam_info:
                continue
            
            # Configure based on role
            if role == CameraRole.PRIMARY_TRACKER:
                fps = tracker_fps
                resolution = tracker_resolution
                is_stereo = cam_info.is_stereo
            elif role == CameraRole.REPLAY_RECORDER:
                fps = 30
                resolution = (1920, 1080)
                is_stereo = False
            else:
                fps = 30
                resolution = (1280, 720)
                is_stereo = cam_info.is_stereo
            
            # Use right eye for stereo cameras (configured in config.json)
            use_right_eye = not self.config.get('camera', {}).get('use_left_eye', True)
            
            thread = CameraThread(
                camera_index=cam_idx,
                role=role,
                target_fps=fps,
                resolution=resolution,
                is_stereo=is_stereo,
                use_right_eye=use_right_eye if is_stereo else False
            )
            
            if thread.start():
                self._cameras[role] = thread
            else:
                print(f"[MultiCam] Warning: Failed to start {role.value}")
        
        if CameraRole.PRIMARY_TRACKER not in self._cameras:
            print("[MultiCam] ERROR: No primary tracker started!")
            self.stop()
            return False
        
        self._running = True
        print(f"\n[MultiCam] Started {len(self._cameras)} cameras")
        return True
    
    def stop(self):
        """Stop all cameras."""
        self._running = False
        
        for role, thread in self._cameras.items():
            thread.stop()
        
        self._cameras.clear()
        print("[MultiCam] All cameras stopped")
    
    def get_frames(self) -> Optional[MultiFrameData]:
        """Get latest frames from all cameras."""
        if not self._running:
            return None
        
        self._frame_number += 1
        timestamp = time.time()
        
        data = MultiFrameData(
            timestamp=timestamp,
            frame_number=self._frame_number
        )
        
        # Get primary tracker frame
        if CameraRole.PRIMARY_TRACKER in self._cameras:
            frame_data = self._cameras[CameraRole.PRIMARY_TRACKER].get_frame()
            if frame_data:
                frame, ts, num = frame_data
                data.tracker_frame = frame
                data.tracker_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                data.tracker_fps = self._cameras[CameraRole.PRIMARY_TRACKER].fps
        
        # Get depth frame (if available)
        if CameraRole.DEPTH_CALIBRATION in self._cameras:
            frame_data = self._cameras[CameraRole.DEPTH_CALIBRATION].get_frame()
            if frame_data:
                frame, ts, num = frame_data
                data.depth_color = frame
                # Note: actual depth frame would need RealSense SDK
        
        # Get replay frame
        if CameraRole.REPLAY_RECORDER in self._cameras:
            frame_data = self._cameras[CameraRole.REPLAY_RECORDER].get_frame()
            if frame_data:
                frame, ts, num = frame_data
                data.replay_frame = frame
                # Add to replay buffer
                self.replay_recorder.add_frame(frame, ts)
        
        return data
    
    def trigger_shot_recording(self, shot_data: dict):
        """Trigger replay recording for a shot."""
        self.replay_recorder.trigger_shot(shot_data)
    
    def start_depth_calibration(self):
        """Start depth-based calibration using RealSense."""
        if CameraRole.DEPTH_CALIBRATION not in self._cameras:
            print("[MultiCam] No depth camera available for calibration")
            return
        
        self.depth_calibrator.start_calibration()
    
    def set_calibration_callback(self, callback: Callable):
        """Set callback for when calibration completes."""
        self._on_calibration_complete = callback
    
    @property
    def primary_fps(self) -> float:
        """Get primary tracker FPS."""
        if CameraRole.PRIMARY_TRACKER in self._cameras:
            return self._cameras[CameraRole.PRIMARY_TRACKER].fps
        return 0.0
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def active_cameras(self) -> List[str]:
        """Get list of active camera roles."""
        return [role.value for role in self._cameras.keys()]
