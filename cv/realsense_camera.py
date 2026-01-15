"""
Intel RealSense D455 camera interface for golf ball tracking.

Provides IR and depth streaming at 848x480 @ 60fps with exposure control.
"""

import time
from typing import Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    rs = None


class CameraError(Exception):
    """Exception raised for camera-related errors."""
    pass


class ExposureMode(Enum):
    """Exposure control mode."""
    AUTO = "auto"
    MANUAL = "manual"


@dataclass
class CameraConfig:
    """Configuration for the RealSense camera."""
    width: int = 848
    height: int = 480
    fps: int = 60
    exposure_mode: ExposureMode = ExposureMode.AUTO
    manual_exposure: int = 8500  # microseconds (1-165000)
    manual_gain: int = 16  # 16-248


@dataclass
class FrameData:
    """Container for captured frame data."""
    ir_frame: np.ndarray
    depth_frame: np.ndarray
    timestamp: float
    frame_number: int


class RealsenseCamera:
    """
    Intel RealSense D455 camera interface.
    
    Captures IR (infrared) and depth streams for ball tracking.
    The IR stream is used as the primary detection source due to
    its high contrast with the golf ball.
    """
    
    def __init__(self, config: Optional[CameraConfig] = None):
        """
        Initialize the RealSense camera.
        
        Args:
            config: Camera configuration. Uses defaults if not provided.
        """
        if not REALSENSE_AVAILABLE:
            raise CameraError(
                "pyrealsense2 is not installed. "
                "Install with: pip install pyrealsense2"
            )
        
        self.config = config or CameraConfig()
        self.pipeline: Optional[rs.pipeline] = None
        self.profile: Optional[rs.pipeline_profile] = None
        self.align: Optional[rs.align] = None
        self.depth_sensor: Optional[rs.sensor] = None
        self.depth_scale: float = 0.001  # Default, will be updated
        
        self._is_running = False
        self._frame_count = 0
        self._dropped_frames = 0
        self._last_frame_time = 0.0
        self._fps_history: list = []
    
    def start(self) -> bool:
        """
        Start the camera pipeline.
        
        Returns:
            True if started successfully, False otherwise.
            
        Raises:
            CameraError: If no device is found or configuration fails.
        """
        try:
            # Check for connected devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                raise CameraError(
                    "No RealSense device found. "
                    "Please connect an Intel RealSense D455 camera."
                )
            
            # Log device info
            device = devices[0]
            device_name = device.get_info(rs.camera_info.name)
            serial = device.get_info(rs.camera_info.serial_number)
            firmware = device.get_info(rs.camera_info.firmware_version)
            print(f"[Camera] Found device: {device_name}")
            print(f"[Camera] Serial: {serial}")
            print(f"[Camera] Firmware: {firmware}")
            
            # Configure pipeline
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # Enable IR stream (left infrared camera)
            config.enable_stream(
                rs.stream.infrared, 1,  # Left IR camera
                self.config.width, self.config.height,
                rs.format.y8, self.config.fps
            )
            
            # Enable depth stream
            config.enable_stream(
                rs.stream.depth,
                self.config.width, self.config.height,
                rs.format.z16, self.config.fps
            )
            
            # Start pipeline
            try:
                self.profile = self.pipeline.start(config)
            except RuntimeError as e:
                if "USB" in str(e).upper() or "bandwidth" in str(e).lower():
                    raise CameraError(
                        f"USB bandwidth issue: {e}\n"
                        "Try using a USB 3.0 port directly (not through a hub)."
                    )
                raise CameraError(f"Failed to start pipeline: {e}")
            
            # Get depth sensor for exposure control
            device = self.profile.get_device()
            self.depth_sensor = device.first_depth_sensor()
            
            # Get depth scale for converting depth values to meters
            self.depth_scale = self.depth_sensor.get_depth_scale()
            print(f"[Camera] Depth scale: {self.depth_scale}")
            
            # Create align object to align depth to IR
            self.align = rs.align(rs.stream.infrared)
            
            # Apply initial exposure settings
            self._apply_exposure_settings()
            
            # Warm up - discard first few frames
            print("[Camera] Warming up...")
            for _ in range(30):
                self.pipeline.wait_for_frames(timeout_ms=1000)
            
            self._is_running = True
            self._frame_count = 0
            self._dropped_frames = 0
            self._last_frame_time = time.time()
            
            print(f"[Camera] Started at {self.config.width}x{self.config.height} @ {self.config.fps}fps")
            return True
            
        except CameraError:
            raise
        except Exception as e:
            raise CameraError(f"Unexpected error starting camera: {e}")
    
    def stop(self):
        """Stop the camera pipeline."""
        if self.pipeline and self._is_running:
            try:
                self.pipeline.stop()
            except Exception as e:
                print(f"[Camera] Warning during stop: {e}")
            finally:
                self._is_running = False
                self.pipeline = None
                self.profile = None
                print("[Camera] Stopped")
    
    def get_frames(self, timeout_ms: int = 1000) -> Optional[FrameData]:
        """
        Get the next frame from the camera.
        
        Args:
            timeout_ms: Timeout in milliseconds to wait for frames.
            
        Returns:
            FrameData containing IR frame, depth frame, and timestamp.
            None if no frames available or timeout.
        """
        if not self._is_running or not self.pipeline:
            return None
        
        try:
            # Wait for frames
            frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
            
            if not frames:
                self._dropped_frames += 1
                return None
            
            # Align depth to IR
            aligned_frames = self.align.process(frames)
            
            # Get IR frame
            ir_frame = aligned_frames.get_infrared_frame(1)
            if not ir_frame:
                self._dropped_frames += 1
                return None
            
            # Get depth frame
            depth_frame = aligned_frames.get_depth_frame()
            if not depth_frame:
                self._dropped_frames += 1
                return None
            
            # Convert to numpy arrays
            ir_image = np.asanyarray(ir_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Get timestamp
            timestamp = frames.get_timestamp() / 1000.0  # Convert to seconds
            
            # Update stats
            self._frame_count += 1
            current_time = time.time()
            frame_delta = current_time - self._last_frame_time
            self._last_frame_time = current_time
            
            # Track FPS
            if frame_delta > 0:
                self._fps_history.append(1.0 / frame_delta)
                if len(self._fps_history) > 60:
                    self._fps_history.pop(0)
            
            return FrameData(
                ir_frame=ir_image,
                depth_frame=depth_image,
                timestamp=timestamp,
                frame_number=self._frame_count
            )
            
        except RuntimeError as e:
            if "timeout" in str(e).lower():
                self._dropped_frames += 1
                return None
            raise CameraError(f"Frame capture error: {e}")
        except Exception as e:
            raise CameraError(f"Unexpected frame error: {e}")
    
    def set_exposure(self, auto: bool, value: Optional[int] = None):
        """
        Set exposure mode and value.
        
        Args:
            auto: True for auto exposure, False for manual.
            value: Manual exposure value in microseconds (1-165000).
        """
        if auto:
            self.config.exposure_mode = ExposureMode.AUTO
        else:
            self.config.exposure_mode = ExposureMode.MANUAL
            if value is not None:
                self.config.manual_exposure = max(1, min(165000, value))
        
        if self._is_running:
            self._apply_exposure_settings()
    
    def set_gain(self, value: int):
        """
        Set gain value.
        
        Args:
            value: Gain value (16-248).
        """
        self.config.manual_gain = max(16, min(248, value))
        
        if self._is_running:
            self._apply_exposure_settings()
    
    def adjust_exposure(self, delta: int):
        """
        Adjust exposure by delta amount.
        
        Args:
            delta: Amount to adjust exposure (positive or negative).
        """
        if self.config.exposure_mode == ExposureMode.MANUAL:
            new_value = self.config.manual_exposure + delta
            self.set_exposure(auto=False, value=new_value)
            print(f"[Camera] Exposure: {self.config.manual_exposure}")
    
    def _apply_exposure_settings(self):
        """Apply current exposure settings to the sensor."""
        if not self.depth_sensor:
            return
        
        try:
            # Get the stereo module (controls IR cameras)
            device = self.profile.get_device()
            
            # Find the stereo module sensor
            for sensor in device.query_sensors():
                if sensor.get_info(rs.camera_info.name) == "Stereo Module":
                    if self.config.exposure_mode == ExposureMode.AUTO:
                        sensor.set_option(rs.option.enable_auto_exposure, 1)
                        print("[Camera] Auto exposure enabled")
                    else:
                        sensor.set_option(rs.option.enable_auto_exposure, 0)
                        sensor.set_option(
                            rs.option.exposure, 
                            self.config.manual_exposure
                        )
                        sensor.set_option(
                            rs.option.gain,
                            self.config.manual_gain
                        )
                        print(f"[Camera] Manual exposure: {self.config.manual_exposure}, gain: {self.config.manual_gain}")
                    break
        except Exception as e:
            print(f"[Camera] Warning: Could not set exposure: {e}")
    
    def get_depth_in_meters(self, depth_frame: np.ndarray, x: int, y: int) -> float:
        """
        Get depth value at a point in meters.
        
        Args:
            depth_frame: Depth frame array.
            x: X coordinate.
            y: Y coordinate.
            
        Returns:
            Depth in meters, or 0 if invalid.
        """
        if 0 <= y < depth_frame.shape[0] and 0 <= x < depth_frame.shape[1]:
            return depth_frame[y, x] * self.depth_scale
        return 0.0
    
    def get_average_depth(
        self, 
        depth_frame: np.ndarray, 
        x: int, 
        y: int, 
        radius: int = 3
    ) -> float:
        """
        Get average depth in a small region around a point.
        
        Args:
            depth_frame: Depth frame array.
            x: Center X coordinate.
            y: Center Y coordinate.
            radius: Radius of region to average.
            
        Returns:
            Average depth in meters, or 0 if invalid.
        """
        h, w = depth_frame.shape
        x1 = max(0, x - radius)
        x2 = min(w, x + radius + 1)
        y1 = max(0, y - radius)
        y2 = min(h, y + radius + 1)
        
        region = depth_frame[y1:y2, x1:x2]
        valid = region[region > 0]
        
        if len(valid) > 0:
            return float(np.median(valid)) * self.depth_scale
        return 0.0
    
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
    
    def get_intrinsics(self) -> Optional[dict]:
        """
        Get camera intrinsics for the IR stream.
        
        Returns:
            Dictionary with fx, fy, ppx, ppy, or None if not available.
        """
        if not self.profile:
            return None
        
        try:
            ir_stream = self.profile.get_stream(rs.stream.infrared, 1)
            intrinsics = ir_stream.as_video_stream_profile().get_intrinsics()
            return {
                'fx': intrinsics.fx,
                'fy': intrinsics.fy,
                'ppx': intrinsics.ppx,
                'ppy': intrinsics.ppy,
                'width': intrinsics.width,
                'height': intrinsics.height,
            }
        except Exception:
            return None
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()
        return False
