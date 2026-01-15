"""
Standard USB webcam interface for golf ball tracking.

Provides RGB streaming using OpenCV with exposure and settings control.
"""

import time
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2


class CameraError(Exception):
    """Exception raised for camera-related errors."""
    pass


@dataclass
class WebcamConfig:
    """Configuration for the webcam."""
    width: int = 1280
    height: int = 720
    fps: int = 30
    device_index: int = 0  # Camera index (0 = default, or specific index)
    auto_exposure: bool = True
    brightness: int = 128  # 0-255
    contrast: int = 128    # 0-255
    saturation: int = 128  # 0-255
    exposure: int = -6     # Exposure value (camera specific)


@dataclass
class FrameData:
    """Container for captured frame data."""
    color_frame: np.ndarray  # BGR image
    gray_frame: np.ndarray   # Grayscale image
    timestamp: float
    frame_number: int


class WebcamCamera:
    """
    Standard USB webcam interface.
    
    Captures color frames for ball tracking using OpenCV.
    """
    
    def __init__(self, config: Optional[WebcamConfig] = None):
        """
        Initialize the webcam.
        
        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        self.config = config or WebcamConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        
        self._is_running = False
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_time = 0.0
        self._fps_history: list = []
        
        # Simulated depth scale for compatibility
        self.depth_scale = 0.001
    
    @staticmethod
    def list_cameras() -> List[dict]:
        """
        List available cameras on the system.
        
        Returns:
            List of camera info dictionaries.
        """
        cameras = []
        
        # Check camera indices 0-9
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get camera info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()
                
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                })
                cap.release()
            else:
                cap.release()
                break  # Usually cameras are sequential
        
        return cameras
    
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if started successfully, False otherwise.
            
        Raises:
            CameraError: If camera cannot be opened.
        """
        try:
            # List available cameras first
            cameras = self.list_cameras()
            if not cameras:
                raise CameraError(
                    "No cameras found. Please connect a USB camera."
                )
            
            print(f"[Camera] Found {len(cameras)} camera(s):")
            for cam in cameras:
                print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']}fps ({cam['backend']})")
            
            # Select camera
            device_index = self.config.device_index
            if device_index >= len(cameras):
                print(f"[Camera] Requested index {device_index} not available, using index 0")
                device_index = 0
            
            # Open camera
            print(f"[Camera] Opening camera {device_index}...")
            self.cap = cv2.VideoCapture(device_index)
            
            if not self.cap.isOpened():
                raise CameraError(
                    f"Failed to open camera at index {device_index}. "
                    "Make sure no other application is using it."
                )
            
            # Set resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Apply camera settings
            self._apply_settings()
            
            # Verify actual settings
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"[Camera] Actual resolution: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            
            # Update config with actual values
            self.config.width = actual_width
            self.config.height = actual_height
            
            # Warm up - read a few frames
            print("[Camera] Warming up...")
            for _ in range(10):
                ret, _ = self.cap.read()
                if not ret:
                    raise CameraError("Failed to read frames during warmup")
            
            self._is_running = True
            self._frame_count = 0
            self._dropped_frames = 0
            self._last_frame_time = time.time()
            
            print(f"[Camera] Started successfully!")
            return True
            
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Unexpected error starting camera: {e}")
    
    def stop(self):
        """Stop the camera capture."""
        if self.cap and self._is_running:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[Camera] Warning during stop: {e}")
            finally:
                self._is_running = False
                self.cap = None
                print("[Camera] Stopped")
    
    def get_frames(self, timeout_ms: int = 1000) -> Optional[FrameData]:
        """
        Get the next frame from the camera.
        
        Args:
            timeout_ms: Timeout in milliseconds (not used for webcam, but kept for API compatibility).
            
        Returns:
            FrameData containing color frame, grayscale frame, and timestamp.
            None if no frames available.
        """
        if not self._is_running or not self.cap:
            return None
        
        try:
            ret, color_frame = self.cap.read()
            
            if not ret or color_frame is None:
                self._dropped_frames += 1
                return None
            
            # Convert to grayscale
            gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
            
            # Get timestamp
            timestamp = time.time()
            
            # Update stats
            self._frame_count += 1
            frame_delta = timestamp - self._last_frame_time
            self._last_frame_time = timestamp
            
            # Track FPS
            if frame_delta > 0:
                self._fps_history.append(1.0 / frame_delta)
                if len(self._fps_history) > 60:
                    self._fps_history.pop(0)
            
            return FrameData(
                color_frame=color_frame,
                gray_frame=gray_frame,
                timestamp=timestamp,
                frame_number=self._frame_count
            )
            
        except Exception as e:
            print(f"[Camera] Frame capture error: {e}")
            self._dropped_frames += 1
            return None
    
    def _apply_settings(self):
        """Apply camera settings."""
        if not self.cap:
            return
        
        try:
            # Auto exposure
            if self.config.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
            
            # Other settings
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            
        except Exception as e:
            print(f"[Camera] Warning: Could not apply some settings: {e}")
    
    def set_exposure(self, auto: bool, value: Optional[int] = None):
        """
        Set exposure mode and value.
        
        Args:
            auto: True for auto exposure, False for manual.
            value: Manual exposure value.
        """
        self.config.auto_exposure = auto
        if value is not None:
            self.config.exposure = value
        
        if self._is_running:
            self._apply_settings()
            mode_str = "auto" if auto else f"manual ({self.config.exposure})"
            print(f"[Camera] Exposure: {mode_str}")
    
    def adjust_exposure(self, delta: int):
        """
        Adjust exposure by delta amount.
        
        Args:
            delta: Amount to adjust exposure (positive or negative).
        """
        if not self.config.auto_exposure:
            self.config.exposure += delta
            if self._is_running:
                self._apply_settings()
            print(f"[Camera] Exposure: {self.config.exposure}")
    
    @property
    def is_running(self) -> bool:
        """Check if camera is running."""
        return self._is_running
    
    @property
    def fps(self) -> float:
        """Get current FPS estimate."""
        if self._fps_history:
            return sum(self._fps_history) / len(self._fps_history)
        return 0.0
    
    @property
    def frame_count(self) -> int:
        """Get total frame count."""
        return self._frame_count
    
    @property
    def dropped_frames(self) -> int:
        """Get dropped frame count."""
        return self._dropped_frames
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def test_camera():
    """Test camera connectivity and display live feed."""
    print("\n" + "="*60)
    print("  WEBCAM CAMERA TEST")
    print("="*60 + "\n")
    
    # List cameras
    cameras = WebcamCamera.list_cameras()
    if not cameras:
        print("ERROR: No cameras found!")
        return False
    
    print(f"Found {len(cameras)} camera(s)")
    
    # Test with default camera
    config = WebcamConfig(
        width=1280,
        height=720,
        fps=30,
        device_index=0
    )
    
    try:
        camera = WebcamCamera(config)
        camera.start()
        
        print("\nShowing live feed (press 'q' to quit)...")
        
        while True:
            frame_data = camera.get_frames()
            
            if frame_data is not None:
                # Display frame
                display = frame_data.color_frame.copy()
                
                # Add FPS overlay
                fps_text = f"FPS: {camera.fps:.1f}"
                cv2.putText(display, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Add frame count
                frame_text = f"Frame: {frame_data.frame_number}"
                cv2.putText(display, frame_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("Camera Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        camera.stop()
        cv2.destroyAllWindows()
        
        print("\nCamera test completed successfully!")
        return True
        
    except CameraError as e:
        print(f"\nERROR: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False


if __name__ == "__main__":
    test_camera()
