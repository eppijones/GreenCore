"""
ZED 2i camera interface for high-speed golf ball tracking.

Supports two modes:
1. Webcam mode (Mac/No NVIDIA): Uses raw stereo feed at up to 100 FPS
2. SDK mode (NVIDIA GPU): Full ZED SDK with depth (future)

In webcam mode, we capture the stereo feed and use only the left camera
for high-speed ball tracking without depth computation.
"""

import time
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2


class CameraError(Exception):
    """Exception raised for camera-related errors."""
    pass


class ZedMode(Enum):
    """ZED camera operation mode."""
    WEBCAM = "webcam"  # Raw stereo feed as webcam (no NVIDIA required)
    SDK = "sdk"        # Full ZED SDK with depth (requires NVIDIA)


@dataclass
class ZedConfig:
    """Configuration for the ZED 2i camera."""
    # Resolution and FPS
    # ZED 2i webcam mode supports:
    # - 2208x1242 @ 15fps (2K)
    # - 1920x1080 @ 30fps (1080p)
    # - 1280x720 @ 60fps (720p)
    # - 672x376 @ 100fps (WVGA)
    width: int = 672           # Single camera width (half of stereo)
    height: int = 376          # Frame height
    fps: int = 100             # Target FPS
    
    # Mode
    mode: ZedMode = ZedMode.WEBCAM
    
    # Camera selection
    device_index: int = -1     # -1 = auto-detect ZED, or specific index
    
    # Which eye to use (for webcam mode)
    use_left_eye: bool = True  # True = left camera, False = right
    
    # Image settings
    auto_exposure: bool = True
    brightness: int = 4        # ZED brightness (0-8)
    contrast: int = 4          # ZED contrast (0-8)
    saturation: int = 4        # ZED saturation (0-8)
    exposure: int = 50         # Manual exposure (0-100)
    gain: int = 50             # Manual gain (0-100)


@dataclass
class FrameData:
    """Container for captured frame data."""
    color_frame: np.ndarray    # BGR image (single eye)
    gray_frame: np.ndarray     # Grayscale image
    timestamp: float
    frame_number: int
    stereo_frame: Optional[np.ndarray] = None  # Full stereo (both eyes)


class ZedCamera:
    """
    ZED 2i camera interface with high-speed webcam mode.
    
    In webcam mode (no NVIDIA GPU):
    - Captures raw stereo feed at up to 100 FPS
    - Crops to single eye (left by default)
    - No depth, but excellent for high-speed ball tracking
    
    In SDK mode (with NVIDIA GPU):
    - Full ZED SDK features including depth
    - Neural depth estimation
    - (Future implementation)
    """
    
    # Known ZED camera identifiers
    ZED_VENDOR_ID = "2b03"  # Stereolabs vendor ID
    ZED_NAMES = ["ZED", "ZED 2", "ZED 2i", "ZED Mini", "ZED-M", "ZED-2i"]
    
    def __init__(self, config: Optional[ZedConfig] = None):
        """
        Initialize the ZED camera.
        
        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        self.config = config or ZedConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        
        self._is_running = False
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_time = 0.0
        self._fps_history: list = []
        
        # Stereo frame dimensions
        self._stereo_width = 0
        self._stereo_height = 0
        
        # Simulated depth scale for compatibility
        self.depth_scale = 0.001
    
    @staticmethod
    def find_zed_camera() -> Optional[int]:
        """
        Find ZED camera index by checking available cameras.
        
        Returns:
            Camera index if found, None otherwise.
        """
        print("[ZED] Searching for ZED camera...")
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Check if this looks like a ZED (stereo = double width)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # ZED cameras have doubled width (stereo side-by-side)
                # Common ZED stereo resolutions:
                # - 4416x1242 (2K stereo)
                # - 3840x1080 (1080p stereo)
                # - 2560x720 (720p stereo)
                # - 1344x376 (WVGA stereo)
                
                aspect_ratio = width / height if height > 0 else 0
                
                # ZED stereo has aspect ratio > 3 (side-by-side)
                if aspect_ratio > 3.0:
                    print(f"[ZED] Found potential ZED at index {i}: {width}x{height} (aspect: {aspect_ratio:.2f})")
                    cap.release()
                    return i
                
                cap.release()
            else:
                cap.release()
        
        return None
    
    @staticmethod
    def list_cameras() -> List[dict]:
        """
        List available cameras with ZED detection.
        
        Returns:
            List of camera info dictionaries.
        """
        cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                backend = cap.getBackendName()
                
                # Detect if ZED
                aspect_ratio = width / height if height > 0 else 0
                is_zed = aspect_ratio > 3.0
                
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'is_zed': is_zed,
                    'single_eye_width': width // 2 if is_zed else width,
                })
                cap.release()
            else:
                cap.release()
                break
        
        return cameras
    
    def start(self) -> bool:
        """
        Start the camera capture.
        
        Returns:
            True if started successfully.
            
        Raises:
            CameraError: If camera cannot be opened.
        """
        if self.config.mode == ZedMode.SDK:
            return self._start_sdk_mode()
        else:
            return self._start_webcam_mode()
    
    def _start_webcam_mode(self) -> bool:
        """Start camera in webcam mode (no NVIDIA required)."""
        try:
            # Find ZED camera
            if self.config.device_index < 0:
                device_index = self.find_zed_camera()
                if device_index is None:
                    raise CameraError(
                        "ZED camera not found. Make sure it's connected via USB 3.0."
                    )
            else:
                device_index = self.config.device_index
            
            print(f"[ZED] Opening camera at index {device_index} (webcam mode)...")
            
            # Open camera
            self.cap = cv2.VideoCapture(device_index)
            
            if not self.cap.isOpened():
                raise CameraError(
                    f"Failed to open ZED camera at index {device_index}. "
                    "Make sure no other application is using it."
                )
            
            # Set resolution and FPS
            # Request stereo resolution (double the single-eye width)
            stereo_width = self.config.width * 2
            stereo_height = self.config.height
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, stereo_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, stereo_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Disable auto-focus (ZED has fixed focus)
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Get actual values
            self._stereo_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._stereo_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            # Update config with actual single-eye dimensions
            self.config.width = self._stereo_width // 2
            self.config.height = self._stereo_height
            
            print(f"[ZED] Stereo resolution: {self._stereo_width}x{self._stereo_height}")
            print(f"[ZED] Single eye resolution: {self.config.width}x{self.config.height}")
            print(f"[ZED] Target FPS: {actual_fps}")
            
            # Apply settings
            self._apply_settings()
            
            # Warm up
            print("[ZED] Warming up...")
            for _ in range(30):  # More warmup frames for ZED
                ret, _ = self.cap.read()
                if not ret:
                    raise CameraError("Failed to read frames during warmup")
            
            self._is_running = True
            self._frame_count = 0
            self._dropped_frames = 0
            self._last_frame_time = time.time()
            
            eye_str = "LEFT" if self.config.use_left_eye else "RIGHT"
            print(f"[ZED] Started successfully! Using {eye_str} eye at {self.config.fps} FPS target")
            
            return True
            
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Unexpected error starting ZED camera: {e}")
    
    def _start_sdk_mode(self) -> bool:
        """Start camera in SDK mode (requires NVIDIA GPU)."""
        # Check if ZED SDK is available
        try:
            import pyzed.sl as sl
        except ImportError:
            raise CameraError(
                "ZED SDK not installed. Install with:\n"
                "  pip install pyzed\n"
                "Note: Requires NVIDIA GPU with CUDA support.\n"
                "For Mac without NVIDIA, use webcam mode instead."
            )
        
        # TODO: Implement full SDK mode when NVIDIA GPU is available
        raise CameraError(
            "SDK mode not yet implemented. "
            "Use webcam mode for now (config.mode = ZedMode.WEBCAM)"
        )
    
    def stop(self):
        """Stop the camera capture."""
        if self.cap and self._is_running:
            try:
                self.cap.release()
            except Exception as e:
                print(f"[ZED] Warning during stop: {e}")
            finally:
                self._is_running = False
                self.cap = None
                print("[ZED] Stopped")
    
    def get_frames(self, timeout_ms: int = 1000) -> Optional[FrameData]:
        """
        Get the next frame from the camera.
        
        Args:
            timeout_ms: Timeout in milliseconds (not used, kept for API compatibility).
            
        Returns:
            FrameData containing color frame, grayscale frame, and timestamp.
            None if no frames available.
        """
        if not self._is_running or not self.cap:
            return None
        
        try:
            ret, stereo_frame = self.cap.read()
            
            if not ret or stereo_frame is None:
                self._dropped_frames += 1
                return None
            
            # Extract single eye from stereo frame
            half_width = stereo_frame.shape[1] // 2
            
            if self.config.use_left_eye:
                color_frame = stereo_frame[:, :half_width].copy()
            else:
                color_frame = stereo_frame[:, half_width:].copy()
            
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
                instant_fps = 1.0 / frame_delta
                self._fps_history.append(instant_fps)
                if len(self._fps_history) > 100:  # Longer history for stability
                    self._fps_history.pop(0)
            
            return FrameData(
                color_frame=color_frame,
                gray_frame=gray_frame,
                timestamp=timestamp,
                frame_number=self._frame_count,
                stereo_frame=stereo_frame if self.config.mode == ZedMode.WEBCAM else None
            )
            
        except Exception as e:
            print(f"[ZED] Frame capture error: {e}")
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
                # ZED exposure is typically 0-100
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
                self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)
            
            # Brightness, contrast, saturation
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.config.saturation)
            
        except Exception as e:
            print(f"[ZED] Warning: Could not apply some settings: {e}")
    
    def set_exposure(self, auto: bool, value: Optional[int] = None):
        """
        Set exposure mode and value.
        
        Args:
            auto: True for auto exposure, False for manual.
            value: Manual exposure value (0-100).
        """
        self.config.auto_exposure = auto
        if value is not None:
            self.config.exposure = value
        
        if self._is_running:
            self._apply_settings()
            mode_str = "auto" if auto else f"manual ({self.config.exposure})"
            print(f"[ZED] Exposure: {mode_str}")
    
    def adjust_exposure(self, delta: int):
        """
        Adjust exposure by delta amount.
        
        Args:
            delta: Amount to adjust exposure (positive or negative).
        """
        if not self.config.auto_exposure:
            self.config.exposure = max(0, min(100, self.config.exposure + delta * 5))
            if self._is_running:
                self._apply_settings()
            print(f"[ZED] Exposure: {self.config.exposure}")
    
    def set_resolution_preset(self, preset: str):
        """
        Set resolution preset.
        
        Args:
            preset: One of '2k', '1080p', '720p', 'wvga'
        """
        presets = {
            '2k': (2208, 1242, 15),
            '1080p': (1920, 1080, 30),
            '720p': (1280, 720, 60),
            'wvga': (672, 376, 100),
        }
        
        if preset.lower() not in presets:
            print(f"[ZED] Unknown preset: {preset}. Use: {list(presets.keys())}")
            return
        
        width, height, fps = presets[preset.lower()]
        print(f"[ZED] Setting preset: {preset} ({width}x{height} @ {fps}fps)")
        
        # Would need to restart camera to apply
        self.config.width = width
        self.config.height = height
        self.config.fps = fps
    
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
    
    @property
    def frame_time_ms(self) -> float:
        """Get average frame time in milliseconds."""
        if self.fps > 0:
            return 1000.0 / self.fps
        return 0.0
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def test_zed_camera():
    """Test ZED camera connectivity and display live feed."""
    print("\n" + "="*60)
    print("  ZED 2i CAMERA TEST (Webcam Mode)")
    print("="*60 + "\n")
    
    # List cameras
    cameras = ZedCamera.list_cameras()
    if not cameras:
        print("ERROR: No cameras found!")
        return False
    
    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        zed_str = " [ZED]" if cam['is_zed'] else ""
        print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']}fps{zed_str}")
    
    # Find ZED
    zed_index = ZedCamera.find_zed_camera()
    if zed_index is None:
        print("\nERROR: No ZED camera found!")
        print("Make sure ZED 2i is connected via USB 3.0")
        return False
    
    print(f"\nUsing ZED at index {zed_index}")
    
    # Configure for high speed (WVGA @ 100fps)
    config = ZedConfig(
        width=672,
        height=376,
        fps=100,
        device_index=zed_index,
        mode=ZedMode.WEBCAM
    )
    
    try:
        camera = ZedCamera(config)
        camera.start()
        
        print("\nShowing live feed (press 'q' to quit, 'e' to toggle exposure)...")
        print("Press 1-4 for resolution presets: 1=WVGA(100fps), 2=720p(60fps), 3=1080p(30fps), 4=2K(15fps)")
        
        cv2.namedWindow("ZED 2i Test", cv2.WINDOW_NORMAL)
        
        while True:
            frame_data = camera.get_frames()
            
            if frame_data is not None:
                # Display frame
                display = frame_data.color_frame.copy()
                
                # Add info overlay
                h, w = display.shape[:2]
                
                # FPS
                fps_text = f"FPS: {camera.fps:.1f} (target: {config.fps})"
                cv2.putText(display, fps_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Frame time
                time_text = f"Frame time: {camera.frame_time_ms:.1f}ms"
                cv2.putText(display, time_text, (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Resolution
                res_text = f"Resolution: {w}x{h}"
                cv2.putText(display, res_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Frame count
                frame_text = f"Frame: {frame_data.frame_number}"
                cv2.putText(display, frame_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("ZED 2i Test", display)
                
                # Also show stereo if available
                if frame_data.stereo_frame is not None:
                    stereo_display = cv2.resize(frame_data.stereo_frame, (0, 0), fx=0.5, fy=0.5)
                    cv2.imshow("ZED 2i Stereo", stereo_display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                camera.set_exposure(not config.auto_exposure)
        
        camera.stop()
        cv2.destroyAllWindows()
        
        print(f"\nTest completed!")
        print(f"  Frames captured: {camera.frame_count}")
        print(f"  Frames dropped: {camera.dropped_frames}")
        print(f"  Average FPS: {camera.fps:.1f}")
        return True
        
    except CameraError as e:
        print(f"\nERROR: {e}")
        return False
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_zed_camera()
