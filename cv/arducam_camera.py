"""
Arducam OV9281 USB camera interface for high-speed golf ball tracking.

The OV9281 is a 1MP global shutter monochrome sensor - ideal for tracking:
- Global shutter eliminates motion blur on fast-moving objects
- Monochrome output = native grayscale (no Bayer demosaic overhead)
- Up to 100+ FPS at 1280x800 resolution
- Low latency USB UVC interface (plug-and-play)

Supported resolutions:
- 1280x800 @ ~100 FPS (MJPG)
- 1280x720 @ ~100 FPS
- 640x400 @ ~120+ FPS
- 320x200 @ ~200+ FPS
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


class ArducamResolution(Enum):
    """Preset resolutions for the Arducam OV9281."""
    FULL = (1280, 800, 100)       # Full resolution, ~100 FPS
    HD_720 = (1280, 720, 100)     # 720p, ~100 FPS
    HALF = (640, 400, 120)        # Half resolution, ~120 FPS
    QUARTER = (320, 200, 200)     # Quarter resolution, ~200 FPS


@dataclass
class ArducamConfig:
    """Configuration for the Arducam OV9281 camera."""
    # Resolution and FPS - MJPG 120fps@1280x800 per Arducam B0332 datasheet
    width: int = 1280
    height: int = 800
    fps: int = 120
    
    # Camera selection
    device_index: int = 2  # Arducam is typically at index 2 on macOS with FaceTime camera
    
    # Image settings
    auto_exposure: bool = False  # Manual exposure often better for tracking
    exposure: int = 50           # Exposure value (camera-specific, typically 1-1000 or percentage)
    gain: int = 50               # Gain value
    brightness: int = 128        # 0-255
    contrast: int = 128          # 0-255
    
    # Global shutter specific
    use_mjpg: bool = True        # MJPG often gives best FPS on USB
    
    # For compatibility with existing code
    auto_detect_enabled: bool = True


@dataclass
class FrameData:
    """Container for captured frame data."""
    color_frame: np.ndarray      # BGR image (converted from grayscale for compatibility)
    gray_frame: np.ndarray       # Grayscale image (native OV9281 output)
    timestamp: float
    frame_number: int


class ArducamCamera:
    """
    Arducam OV9281 global shutter camera interface.
    
    Optimized for high-speed ball tracking with:
    - Global shutter (no motion blur)
    - Monochrome sensor (native grayscale)
    - High frame rates (100+ FPS)
    - Low latency capture
    """
    
    # Camera identifiers
    ARDUCAM_NAMES = ["Arducam", "OV9281", "Global Shutter", "UC-593"]
    
    def __init__(self, config: Optional[ArducamConfig] = None):
        """
        Initialize the Arducam camera.
        
        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        self.config = config or ArducamConfig()
        self.cap: Optional[cv2.VideoCapture] = None
        
        self._is_running = False
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_time = 0.0
        self._fps_history: list = []
        self._is_monochrome = True  # OV9281 is natively grayscale
        
        # Timing stats for latency analysis
        self._capture_times: list = []
        self._frame_intervals: list = []
        
        # Simulated depth scale for compatibility
        self.depth_scale = 0.001
    
    @staticmethod
    def find_arducam() -> Optional[int]:
        """
        Find Arducam OV9281 camera index by looking for MONOCHROME camera.
        
        Returns:
            Camera index if found, None otherwise.
        """
        import os
        import platform
        
        # Suppress OpenCV warnings
        old_log = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        print("[Arducam] Searching for OV9281 (monochrome) camera...")
        
        try:
            # Only check indices 0-3 to avoid "out of bound" errors
            for i in range(4):
                try:
                    if platform.system() == 'Darwin':
                        cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                    else:
                        cap = cv2.VideoCapture(i)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    # Read a frame to check if monochrome
                    ret, frame = cap.read()
                    
                    if ret and frame is not None and len(frame.shape) == 3:
                        # Check if channels are identical (monochrome in BGR wrapper)
                        b, g, r = cv2.split(frame)
                        color_variance = np.std(b.astype(float) - g.astype(float))
                        
                        if color_variance < 5:  # Very low color variance = monochrome = Arducam!
                            print(f"[Arducam] Found MONOCHROME camera at index {i}: {width}x{height}")
                            cap.release()
                            return i
                        else:
                            print(f"[Arducam] Index {i}: COLOR camera (skipping)")
                    
                    cap.release()
                except Exception:
                    continue
            
            return None
        finally:
            # Restore log level
            if old_log:
                os.environ['OPENCV_LOG_LEVEL'] = old_log
            elif 'OPENCV_LOG_LEVEL' in os.environ:
                del os.environ['OPENCV_LOG_LEVEL']
    
    @staticmethod
    def list_cameras() -> List[dict]:
        """
        List available cameras with monochrome detection.
        
        Returns:
            List of camera info dictionaries.
        """
        cameras = []
        
        for i in range(10):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                
                # Check if monochrome
                ret, frame = cap.read()
                is_mono = False
                if ret and frame is not None and len(frame.shape) == 3:
                    b, g, r = cv2.split(frame)
                    color_variance = np.std(b.astype(float) - g.astype(float))
                    is_mono = color_variance < 5
                
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'is_monochrome': is_mono,
                    'likely_arducam': is_mono and (width >= 640 and height >= 400),
                })
                cap.release()
            else:
                cap.release()
                break
        
        return cameras
    
    @staticmethod
    def list_supported_modes(device_index: int = 0) -> List[dict]:
        """
        List supported video modes for the camera.
        
        Returns:
            List of supported modes with resolution and FPS.
        """
        modes = []
        
        # Common resolutions to test
        test_resolutions = [
            (1280, 800),   # Full
            (1280, 720),   # 720p
            (640, 480),    # VGA
            (640, 400),    # Half
            (320, 240),    # QVGA
            (320, 200),    # Quarter
        ]
        
        test_formats = ['MJPG', 'YUYV', 'GREY']
        
        cap = cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return modes
        
        for fmt in test_formats:
            fourcc = cv2.VideoWriter_fourcc(*fmt)
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            
            for w, h in test_resolutions:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                actual_fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_fmt = "".join([chr((actual_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                
                mode = {
                    'width': actual_w,
                    'height': actual_h,
                    'fps': actual_fps,
                    'format': actual_fmt.strip(),
                    'requested': f"{w}x{h} {fmt}"
                }
                
                # Avoid duplicates
                mode_key = f"{actual_w}x{actual_h}_{actual_fmt}"
                if not any(f"{m['width']}x{m['height']}_{m['format']}" == mode_key for m in modes):
                    modes.append(mode)
        
        cap.release()
        return modes
    
    def start(self) -> bool:
        """
        Start the camera capture - simple and fast, no scanning.
        
        Returns:
            True if started successfully.
            
        Raises:
            CameraError: If camera cannot be opened.
        """
        import platform
        import os
        
        # Suppress OpenCV warnings
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        try:
            # Find camera (quick monochrome detection)
            if self.config.device_index < 0:
                device_index = self.find_arducam()
                if device_index is None:
                    device_index = 0
            else:
                device_index = self.config.device_index
            
            print(f"[Arducam] Opening camera {device_index} at {self.config.width}x{self.config.height} @ {self.config.fps}fps...")
            
            # Open camera directly - no scanning!
            if platform.system() == 'Darwin':
                self.cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
            else:
                self.cap = cv2.VideoCapture(device_index, cv2.CAP_V4L2)
            
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(device_index)
            
            if not self.cap.isOpened():
                raise CameraError(f"Failed to open camera at index {device_index}")
            
            # Set properties directly - no scanning or verification
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Low latency
            
            # Quick warmup - just read a few frames
            for _ in range(10):
                self.cap.read()
            
            # Get actual values
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            self.config.width = actual_width
            self.config.height = actual_height
            
            print(f"[Arducam] Ready: {actual_width}x{actual_height} @ {actual_fps:.0f}fps")
            
            self._is_running = True
            self._frame_count = 0
            self._dropped_frames = 0
            self._last_frame_time = time.time()
            self._fps_history.clear()
            self._capture_times.clear()
            self._frame_intervals.clear()
            self._is_monochrome = True  # OV9281 is always monochrome
            
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
                print(f"[Arducam] Warning during stop: {e}")
            finally:
                self._is_running = False
                self.cap = None
                print("[Arducam] Stopped")
    
    def get_frames(self, timeout_ms: int = 100) -> Optional[FrameData]:
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
            capture_start = time.time()
            ret, frame = self.cap.read()
            capture_end = time.time()
            
            if not ret or frame is None:
                self._dropped_frames += 1
                return None
            
            # Get timestamp immediately after capture
            timestamp = capture_end
            
            # Track capture latency
            capture_time = (capture_end - capture_start) * 1000  # ms
            self._capture_times.append(capture_time)
            if len(self._capture_times) > 100:
                self._capture_times.pop(0)
            
            # Convert to grayscale if needed
            if len(frame.shape) == 2:
                # Already grayscale
                gray_frame = frame
                color_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif self._is_monochrome:
                # Monochrome in BGR wrapper - just take one channel
                gray_frame = frame[:, :, 0]
                color_frame = frame
            else:
                # Color camera - convert normally
                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                color_frame = frame
            
            # Update stats
            self._frame_count += 1
            frame_delta = timestamp - self._last_frame_time
            self._last_frame_time = timestamp
            
            # Track frame intervals
            if frame_delta > 0:
                self._frame_intervals.append(frame_delta * 1000)  # ms
                if len(self._frame_intervals) > 100:
                    self._frame_intervals.pop(0)
            
            # Track FPS
            if frame_delta > 0:
                instant_fps = 1.0 / frame_delta
                self._fps_history.append(instant_fps)
                if len(self._fps_history) > 100:
                    self._fps_history.pop(0)
            
            return FrameData(
                color_frame=color_frame,
                gray_frame=gray_frame,
                timestamp=timestamp,
                frame_number=self._frame_count
            )
            
        except Exception as e:
            print(f"[Arducam] Frame capture error: {e}")
            self._dropped_frames += 1
            return None
    
    def _apply_settings(self):
        """Apply camera settings."""
        if not self.cap:
            return
        
        try:
            # Exposure control
            if self.config.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # 3 = auto
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # 1 = manual
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.config.exposure)
            
            # Gain
            self.cap.set(cv2.CAP_PROP_GAIN, self.config.gain)
            
            # Brightness and contrast
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.config.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.config.contrast)
            
        except Exception as e:
            print(f"[Arducam] Warning: Could not apply some settings: {e}")
    
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
            print(f"[Arducam] Exposure: {mode_str}")
    
    def adjust_exposure(self, delta: int):
        """
        Adjust exposure by delta amount.
        
        Args:
            delta: Amount to adjust exposure (positive or negative).
        """
        if not self.config.auto_exposure:
            self.config.exposure = max(1, min(1000, self.config.exposure + delta * 10))
            if self._is_running:
                self._apply_settings()
            print(f"[Arducam] Exposure: {self.config.exposure}")
    
    def set_resolution(self, preset: ArducamResolution):
        """
        Set resolution preset (requires restart).
        
        Args:
            preset: Resolution preset.
        """
        width, height, fps = preset.value
        print(f"[Arducam] Setting preset: {preset.name} ({width}x{height} @ {fps}fps)")
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
    
    @property
    def capture_latency_ms(self) -> float:
        """Get average capture latency in milliseconds."""
        if self._capture_times:
            return sum(self._capture_times) / len(self._capture_times)
        return 0.0
    
    @property
    def frame_interval_jitter_ms(self) -> float:
        """Get frame interval jitter (standard deviation) in milliseconds."""
        if len(self._frame_intervals) > 1:
            return float(np.std(self._frame_intervals))
        return 0.0
    
    def get_stats(self) -> dict:
        """Get detailed camera statistics."""
        return {
            'fps': self.fps,
            'frame_count': self._frame_count,
            'dropped_frames': self._dropped_frames,
            'frame_time_ms': self.frame_time_ms,
            'capture_latency_ms': self.capture_latency_ms,
            'frame_jitter_ms': self.frame_interval_jitter_ms,
            'resolution': f"{self.config.width}x{self.config.height}",
            'is_monochrome': self._is_monochrome,
        }
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False


def test_arducam():
    """Test Arducam camera connectivity and display live feed."""
    print("\n" + "="*60)
    print("  ARDUCAM OV9281 CAMERA TEST")
    print("  Global Shutter High-Speed Camera")
    print("="*60 + "\n")
    
    # List cameras
    cameras = ArducamCamera.list_cameras()
    if not cameras:
        print("ERROR: No cameras found!")
        return False
    
    print(f"Found {len(cameras)} camera(s):")
    for cam in cameras:
        mono_str = " [MONO]" if cam['is_monochrome'] else ""
        ardu_str = " [Arducam?]" if cam['likely_arducam'] else ""
        print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']}fps{mono_str}{ardu_str}")
    
    # Find Arducam
    arducam_index = ArducamCamera.find_arducam()
    if arducam_index is not None:
        print(f"\nUsing Arducam at index {arducam_index}")
    else:
        print("\nNo Arducam auto-detected, using camera 0")
        arducam_index = 0
    
    # Configure for high speed
    config = ArducamConfig(
        width=1280,
        height=800,
        fps=100,
        device_index=arducam_index,
        auto_exposure=False,
        exposure=50
    )
    
    try:
        camera = ArducamCamera(config)
        camera.start()
        
        print("\nShowing live feed (press 'q' to quit, 'e' to toggle exposure)...")
        
        cv2.namedWindow("Arducam OV9281 Test", cv2.WINDOW_NORMAL)
        
        while True:
            frame_data = camera.get_frames()
            
            if frame_data is not None:
                # Display grayscale for monochrome camera
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
                
                # Capture latency
                latency_text = f"Capture latency: {camera.capture_latency_ms:.2f}ms"
                cv2.putText(display, latency_text, (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Jitter
                jitter_text = f"Frame jitter: {camera.frame_interval_jitter_ms:.2f}ms"
                cv2.putText(display, jitter_text, (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Resolution
                res_text = f"Resolution: {w}x{h}"
                cv2.putText(display, res_text, (10, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Frame count
                frame_text = f"Frame: {frame_data.frame_number}"
                cv2.putText(display, frame_text, (10, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Global shutter badge
                cv2.putText(display, "GLOBAL SHUTTER", (w - 200, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
                
                cv2.imshow("Arducam OV9281 Test", display)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('e'):
                camera.set_exposure(not config.auto_exposure)
            elif key == ord('+') or key == ord('='):
                camera.adjust_exposure(1)
            elif key == ord('-'):
                camera.adjust_exposure(-1)
        
        camera.stop()
        cv2.destroyAllWindows()
        
        # Print final stats
        stats = camera.get_stats()
        print(f"\nTest completed!")
        print(f"  Frames captured: {stats['frame_count']}")
        print(f"  Frames dropped: {stats['dropped_frames']}")
        print(f"  Average FPS: {stats['fps']:.1f}")
        print(f"  Average frame time: {stats['frame_time_ms']:.1f}ms")
        print(f"  Average capture latency: {stats['capture_latency_ms']:.2f}ms")
        print(f"  Frame interval jitter: {stats['frame_jitter_ms']:.2f}ms")
        
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
    test_arducam()
