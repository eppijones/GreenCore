#!/usr/bin/env python3
"""
Arducam B0332 OV9281 High-Speed Capture Script for macOS

Robust camera initialization with proper format negotiation for:
- macOS AVFoundation backend quirks
- MJPEG vs YUYV format selection
- USB 2.0 bandwidth constraints
- Measured FPS validation

Usage:
    python arducam_capture.py --list
    python arducam_capture.py --measure
    python arducam_capture.py --width 1280 --height 800 --fps 120 --format mjpg
    python arducam_capture.py --index 0 --format auto --measure

Camera: Arducam B0332 OV9281 UVC 120fps Global Shutter
"""

import sys
import time
import argparse
import platform
import subprocess
import re
from typing import Optional, List, Dict, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

import numpy as np
import cv2


# =============================================================================
# Constants
# =============================================================================

class PixelFormat(Enum):
    """Supported pixel formats."""
    MJPG = "MJPG"
    YUYV = "YUYV"
    GREY = "GREY"
    NV12 = "NV12"
    AUTO = "AUTO"


@dataclass
class CameraMode:
    """Represents a camera mode with resolution, FPS, and format."""
    width: int
    height: int
    fps: float
    format: str
    usb2_compatible: bool
    
    def bandwidth_mbps(self) -> float:
        """Calculate theoretical bandwidth in Mbps."""
        bytes_per_pixel = 2  # YUYV = 2 bytes/pixel
        if self.format.upper() in ('MJPG', 'MJPEG'):
            # MJPEG typically 1/5 to 1/10 compression
            bytes_per_pixel = 0.3  # Conservative estimate
        elif self.format.upper() in ('GREY', 'Y800'):
            bytes_per_pixel = 1
        
        bytes_per_sec = self.width * self.height * bytes_per_pixel * self.fps
        return (bytes_per_sec * 8) / 1_000_000
    
    def __str__(self) -> str:
        usb2 = "✓" if self.usb2_compatible else "✗"
        return f"{self.width}x{self.height} @ {self.fps:6.1f}fps ({self.format}) [{self.bandwidth_mbps():6.1f} Mbps] USB2:{usb2}"


@dataclass
class CameraDevice:
    """Represents a detected camera device."""
    index: int
    name: str
    vendor_id: str
    product_id: str
    is_arducam: bool
    unique_id: str  # For deterministic selection


@dataclass
class CaptureResult:
    """Result of a capture session."""
    success: bool
    backend: str
    requested_width: int
    requested_height: int
    requested_fps: float
    requested_format: str
    actual_width: int
    actual_height: int
    actual_fps_reported: float
    actual_format: str
    measured_fps: float
    frame_count: int
    error: Optional[str] = None


# =============================================================================
# macOS Camera Enumeration
# =============================================================================

def enumerate_macos_cameras() -> List[CameraDevice]:
    """
    Enumerate cameras on macOS using system_profiler and ioreg.
    Returns deterministic device list with vendor/product IDs.
    """
    devices = []
    
    try:
        # Get USB devices from system_profiler
        result = subprocess.run(
            ['system_profiler', 'SPUSBDataType', '-json'],
            capture_output=True, text=True, timeout=10
        )
        
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            
            def find_cameras(items, parent_name=""):
                """Recursively find camera devices in USB tree."""
                cameras = []
                if not isinstance(items, list):
                    return cameras
                    
                for item in items:
                    if not isinstance(item, dict):
                        continue
                        
                    for key, value in item.items():
                        if key.startswith('_'):
                            continue
                            
                        if isinstance(value, dict):
                            # Check if this is a camera device
                            name = key
                            vendor_id = value.get('vendor_id', '')
                            product_id = value.get('product_id', '')
                            
                            # Arducam identifiers
                            is_arducam = any(x in name.lower() for x in 
                                ['arducam', 'ov9281', 'b0332', 'uc-593', 'global shutter'])
                            
                            # Also check for generic UVC cameras
                            is_uvc = 'uvc' in name.lower() or 'camera' in name.lower()
                            
                            if is_arducam or is_uvc:
                                unique_id = f"{vendor_id}:{product_id}"
                                cameras.append(CameraDevice(
                                    index=-1,  # Will be set later
                                    name=name,
                                    vendor_id=vendor_id,
                                    product_id=product_id,
                                    is_arducam=is_arducam,
                                    unique_id=unique_id
                                ))
                            
                            # Recurse into children
                            if '_items' in value:
                                cameras.extend(find_cameras(value['_items'], name))
                        
                        elif isinstance(value, list):
                            cameras.extend(find_cameras(value, key))
                
                return cameras
            
            usb_data = data.get('SPUSBDataType', [])
            devices = find_cameras(usb_data)
            
    except Exception as e:
        print(f"[Enum] Warning: system_profiler failed: {e}")
    
    return devices


def enumerate_opencv_cameras(max_index: int = 3) -> List[Dict]:
    """
    Enumerate cameras using OpenCV.
    Only checks indices 0-2 to avoid "out of device bound" warnings.
    """
    cameras = []
    
    # Suppress OpenCV warnings during enumeration
    import os
    old_opencv_log = os.environ.get('OPENCV_LOG_LEVEL', '')
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    try:
        for i in range(max_index):
            try:
                # Use AVFoundation backend explicitly on macOS
                if platform.system() == 'Darwin':
                    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                else:
                    cap = cv2.VideoCapture(i)
                
                if not cap.isOpened():
                    cap.release()
                    continue
                
                # Get basic info
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                backend = cap.getBackendName()
                fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = decode_fourcc(fourcc_code)
                
                # Try to read a frame
                ret, frame = cap.read()
                is_readable = ret and frame is not None
                
                # Detect monochrome (OV9281 characteristic)
                is_mono = False
                if is_readable and frame is not None and len(frame.shape) == 3:
                    b, g, r = cv2.split(frame)
                    color_var = np.std(b.astype(float) - g.astype(float))
                    is_mono = color_var < 5
                
                cameras.append({
                    'index': i,
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'backend': backend,
                    'fourcc': fourcc_str,
                    'is_readable': is_readable,
                    'is_monochrome': is_mono,
                })
                
                cap.release()
                
            except Exception:
                continue
    finally:
        if old_opencv_log:
            os.environ['OPENCV_LOG_LEVEL'] = old_opencv_log
        elif 'OPENCV_LOG_LEVEL' in os.environ:
            del os.environ['OPENCV_LOG_LEVEL']
    
    return cameras


def decode_fourcc(fourcc: int) -> str:
    """Decode FOURCC integer to string."""
    if fourcc == 0:
        return "NONE"
    chars = []
    for i in range(4):
        c = (fourcc >> 8 * i) & 0xFF
        if 32 <= c <= 126:
            chars.append(chr(c))
        else:
            chars.append('?')
    return ''.join(chars)


def encode_fourcc(s: str) -> int:
    """Encode FOURCC string to integer."""
    return cv2.VideoWriter_fourcc(*s[:4].ljust(4))


# =============================================================================
# Format Negotiation
# =============================================================================

class FormatNegotiator:
    """
    Handles pixel format negotiation with macOS AVFoundation quirks.
    
    macOS AVFoundation issues:
    1. FOURCC setting may be ignored
    2. Format changes require cap.release() and reopen
    3. Some formats only work at specific resolutions
    4. Need to set format BEFORE resolution
    """
    
    # Priority order for formats (MJPG preferred for USB 2.0 bandwidth)
    FORMAT_PRIORITY = ['MJPG', 'YUYV', 'NV12', 'GREY']
    
    @staticmethod
    def negotiate_format(
        device_index: int,
        requested_format: PixelFormat,
        requested_width: int,
        requested_height: int,
        requested_fps: float,
        verbose: bool = True
    ) -> Tuple[Optional[cv2.VideoCapture], Dict]:
        """
        Negotiate the best format for the camera.
        
        Returns:
            Tuple of (VideoCapture object or None, negotiation info dict)
        """
        info = {
            'backend': '',
            'requested': {
                'format': requested_format.value,
                'width': requested_width,
                'height': requested_height,
                'fps': requested_fps,
            },
            'actual': {},
            'attempts': [],
            'success': False,
        }
        
        # Determine format order to try
        if requested_format == PixelFormat.AUTO:
            formats_to_try = FormatNegotiator.FORMAT_PRIORITY
        else:
            formats_to_try = [requested_format.value]
        
        for fmt in formats_to_try:
            if verbose:
                print(f"[Negotiate] Trying format: {fmt}")
            
            attempt = {'format': fmt, 'success': False, 'error': None}
            
            try:
                cap = FormatNegotiator._try_format(
                    device_index, fmt, requested_width, requested_height, requested_fps, verbose
                )
                
                if cap is not None and cap.isOpened():
                    # Verify we can read frames
                    frames_ok = FormatNegotiator._verify_frames(cap, num_frames=10, verbose=verbose)
                    
                    if frames_ok:
                        actual_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))
                        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        actual_fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        info['backend'] = cap.getBackendName()
                        info['actual'] = {
                            'format': actual_fourcc,
                            'width': actual_width,
                            'height': actual_height,
                            'fps': actual_fps,
                        }
                        info['success'] = True
                        attempt['success'] = True
                        info['attempts'].append(attempt)
                        
                        if verbose:
                            print(f"[Negotiate] SUCCESS: {actual_width}x{actual_height} @ {actual_fps}fps ({actual_fourcc})")
                        
                        return cap, info
                    else:
                        attempt['error'] = "Frame verification failed"
                        cap.release()
                else:
                    attempt['error'] = "Failed to open capture"
                    
            except Exception as e:
                attempt['error'] = str(e)
            
            info['attempts'].append(attempt)
            if verbose:
                print(f"[Negotiate] {fmt} failed: {attempt['error']}")
        
        return None, info
    
    @staticmethod
    def _try_format(
        device_index: int,
        format_str: str,
        width: int,
        height: int,
        fps: float,
        verbose: bool
    ) -> Optional[cv2.VideoCapture]:
        """
        Try to open camera with specific format.
        
        CRITICAL: On macOS AVFoundation, the order matters:
        1. Open with backend
        2. Set FOURCC first
        3. Then set resolution
        4. Then set FPS
        """
        import os
        
        # Suppress OpenCV warnings
        old_log = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        cap = None
        try:
            backend = cv2.CAP_AVFOUNDATION if platform.system() == 'Darwin' else cv2.CAP_ANY
            
            cap = cv2.VideoCapture(device_index, backend)
            if not cap.isOpened():
                # Fallback to default backend
                cap = cv2.VideoCapture(device_index)
                if not cap.isOpened():
                    return None
            
            # Step 1: Set FOURCC FIRST (critical for macOS)
            fourcc = encode_fourcc(format_str)
            result = cap.set(cv2.CAP_PROP_FOURCC, fourcc)
            if verbose:
                print(f"[Format] Set FOURCC {format_str}: {'OK' if result else 'ignored'}")
            
            # Step 2: Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify resolution was accepted
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if verbose:
                print(f"[Format] Resolution: requested {width}x{height}, got {actual_w}x{actual_h}")
            
            # Step 3: Set FPS
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            if verbose:
                print(f"[Format] FPS: requested {fps}, reported {actual_fps}")
            
            # Step 4: Minimize buffer for low latency
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Step 5: Disable autofocus (not applicable for OV9281 but good practice)
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            return cap
            
        finally:
            # Restore log level
            if old_log:
                os.environ['OPENCV_LOG_LEVEL'] = old_log
            elif 'OPENCV_LOG_LEVEL' in os.environ:
                del os.environ['OPENCV_LOG_LEVEL']
    
    @staticmethod
    def _verify_frames(cap: cv2.VideoCapture, num_frames: int = 10, verbose: bool = True) -> bool:
        """
        Verify camera can deliver frames.
        
        Handles warmup gracefully - doesn't fail on first few empty frames.
        """
        successful_frames = 0
        failed_frames = 0
        
        # Allow up to 2x num_frames attempts to get num_frames successful reads
        max_attempts = num_frames * 2 + 10
        
        for i in range(max_attempts):
            ret, frame = cap.read()
            
            if ret and frame is not None and frame.size > 0:
                successful_frames += 1
                if successful_frames >= num_frames:
                    break
            else:
                failed_frames += 1
                # Sleep briefly to let camera initialize
                time.sleep(0.01)
        
        if verbose:
            print(f"[Verify] Frames: {successful_frames} OK, {failed_frames} failed")
        
        return successful_frames >= num_frames


# =============================================================================
# FPS Measurement
# =============================================================================

def measure_fps(cap: cv2.VideoCapture, duration_seconds: float = 5.0, verbose: bool = True) -> Tuple[float, int, int]:
    """
    Measure actual FPS over a duration.
    
    Returns:
        Tuple of (measured_fps, total_frames, dropped_frames)
    """
    if verbose:
        print(f"[Measure] Running FPS measurement for {duration_seconds}s...")
    
    frame_times = []
    dropped = 0
    start_time = time.perf_counter()
    last_frame_time = start_time
    
    while True:
        current_time = time.perf_counter()
        elapsed = current_time - start_time
        
        if elapsed >= duration_seconds:
            break
        
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_times.append(current_time - last_frame_time)
            last_frame_time = current_time
        else:
            dropped += 1
    
    total_frames = len(frame_times)
    
    if total_frames > 0:
        # Calculate FPS from frame count / duration
        total_time = frame_times[-1] if frame_times else duration_seconds
        actual_duration = sum(frame_times)
        measured_fps = total_frames / actual_duration if actual_duration > 0 else 0
        
        # Also calculate from individual frame intervals
        if len(frame_times) > 1:
            avg_interval = np.mean(frame_times[1:])  # Skip first interval
            interval_fps = 1.0 / avg_interval if avg_interval > 0 else 0
            std_interval = np.std(frame_times[1:]) * 1000  # ms
            
            if verbose:
                print(f"[Measure] Frames: {total_frames}, Dropped: {dropped}")
                print(f"[Measure] FPS (count/time): {measured_fps:.1f}")
                print(f"[Measure] FPS (from intervals): {interval_fps:.1f}")
                print(f"[Measure] Frame interval std: {std_interval:.2f}ms")
        
        return measured_fps, total_frames, dropped
    
    return 0.0, 0, dropped


# =============================================================================
# Alternative Capture Methods
# =============================================================================

def try_ffmpeg_capture(device_index: int, width: int, height: int, fps: float) -> Optional[cv2.VideoCapture]:
    """
    Try FFmpeg/GStreamer pipeline as alternative to AVFoundation.
    
    This can sometimes achieve better format negotiation than AVFoundation.
    """
    # Check if FFmpeg backend is available
    if not cv2.videoio_registry.hasBackend(cv2.CAP_FFMPEG):
        return None
    
    # On macOS, try AVFoundation device through FFmpeg
    # Device name format: "0" or device index as string
    try:
        # FFmpeg pipeline for macOS
        pipeline = f"avfoundation -i {device_index} -video_size {width}x{height} -framerate {fps}"
        cap = cv2.VideoCapture(pipeline, cv2.CAP_FFMPEG)
        
        if cap.isOpened():
            return cap
    except:
        pass
    
    return None


def try_gstreamer_capture(device_index: int, width: int, height: int, fps: float) -> Optional[cv2.VideoCapture]:
    """
    Try GStreamer pipeline as alternative capture method.
    
    Note: GStreamer support depends on OpenCV build configuration.
    """
    # Check if GStreamer backend is available
    if not cv2.videoio_registry.hasBackend(cv2.CAP_GSTREAMER):
        return None
    
    try:
        # GStreamer pipeline for macOS AVFoundation source
        # This requires gstreamer and gst-plugins to be installed
        pipeline = (
            f"avfvideosrc device-index={device_index} ! "
            f"video/x-raw,width={width},height={height},framerate={fps}/1 ! "
            f"videoconvert ! appsink"
        )
        
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        
        if cap.isOpened():
            return cap
    except:
        pass
    
    return None


# =============================================================================
# High-Level Capture Class
# =============================================================================

class ArducamCapture:
    """
    Robust Arducam capture with automatic format negotiation.
    
    Handles macOS AVFoundation quirks and USB 2.0 bandwidth constraints.
    """
    
    # Known Arducam B0332 modes (from datasheet)
    KNOWN_MODES = [
        CameraMode(1280, 800, 120, 'MJPG', True),
        CameraMode(1280, 800, 60, 'YUYV', False),  # Exceeds USB2 bandwidth
        CameraMode(1280, 720, 120, 'MJPG', True),
        CameraMode(1280, 720, 60, 'YUYV', False),
        CameraMode(640, 480, 120, 'MJPG', True),
        CameraMode(640, 480, 120, 'YUYV', True),
        CameraMode(320, 240, 120, 'MJPG', True),
        CameraMode(320, 240, 120, 'YUYV', True),
    ]
    
    def __init__(
        self,
        index: Optional[int] = 2,  # Arducam typically at index 2 on macOS with built-in FaceTime
        width: int = 1280,
        height: int = 800,
        fps: float = 120,
        pixel_format: PixelFormat = PixelFormat.AUTO,
        verbose: bool = True
    ):
        self.requested_index = index
        self.requested_width = width
        self.requested_height = height
        self.requested_fps = fps
        self.requested_format = pixel_format
        self.verbose = verbose
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.device_index: int = -1
        self.negotiation_info: Dict = {}
        
        self._frame_count = 0
        self._dropped_frames = 0
    
    def find_arducam_index(self) -> int:
        """
        Find the Arducam camera index.
        
        Default: Use index 0 (most common when only Arducam is connected).
        """
        # Simple default: index 0 is almost always correct when only Arducam is connected
        if self.verbose:
            print(f"[Arducam] Using default camera index 0")
        return 0
    
    def open(self) -> bool:
        """
        Open the camera with format negotiation.
        
        Returns:
            True if successful.
        """
        # Use requested index (defaults to 0)
        self.device_index = self.requested_index if self.requested_index is not None else 0
        
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"  ARDUCAM CAPTURE INITIALIZATION")
            print(f"{'='*60}")
            print(f"  Device index: {self.device_index}")
            print(f"  Requested: {self.requested_width}x{self.requested_height} @ {self.requested_fps}fps")
            print(f"  Format: {self.requested_format.value}")
            print(f"{'='*60}\n")
        
        # Negotiate format
        self.cap, self.negotiation_info = FormatNegotiator.negotiate_format(
            device_index=self.device_index,
            requested_format=self.requested_format,
            requested_width=self.requested_width,
            requested_height=self.requested_height,
            requested_fps=self.requested_fps,
            verbose=self.verbose
        )
        
        if self.cap is None or not self.negotiation_info['success']:
            if self.verbose:
                print("[Arducam] ERROR: Format negotiation failed")
                for attempt in self.negotiation_info.get('attempts', []):
                    print(f"  - {attempt['format']}: {attempt.get('error', 'unknown')}")
            return False
        
        # Print summary
        if self.verbose:
            actual = self.negotiation_info['actual']
            print(f"\n[Arducam] NEGOTIATED SETTINGS:")
            print(f"  Backend: {self.negotiation_info['backend']}")
            print(f"  Resolution: {actual['width']}x{actual['height']}")
            print(f"  FPS (reported): {actual['fps']}")
            print(f"  FOURCC: {actual['format']}")
        
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None:
            return False, None
        
        ret, frame = self.cap.read()
        
        if ret and frame is not None:
            self._frame_count += 1
            return True, frame
        else:
            self._dropped_frames += 1
            return False, None
    
    def measure_fps(self, duration: float = 5.0) -> float:
        """Measure actual FPS."""
        if self.cap is None:
            return 0.0
        
        fps, frames, dropped = measure_fps(self.cap, duration, self.verbose)
        return fps
    
    def close(self):
        """Close the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
    
    def get_info(self) -> Dict:
        """Get capture information."""
        if self.cap is None:
            return {'open': False}
        
        return {
            'open': True,
            'device_index': self.device_index,
            'backend': self.negotiation_info.get('backend', ''),
            'requested': self.negotiation_info.get('requested', {}),
            'actual': self.negotiation_info.get('actual', {}),
            'frame_count': self._frame_count,
            'dropped_frames': self._dropped_frames,
        }
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# CLI Functions
# =============================================================================

def quick_fps_test(device_index: int, width: int = 640, height: int = 480) -> float:
    """
    Quick test to see what FPS a camera can actually achieve.
    Returns measured FPS over 1 second.
    """
    import os
    old_log = os.environ.get('OPENCV_LOG_LEVEL', '')
    os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
    
    try:
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION) if platform.system() == 'Darwin' else cv2.VideoCapture(device_index)
        if not cap.isOpened():
            return 0.0
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 120)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Warmup
        for _ in range(10):
            cap.read()
        
        # Measure
        frames = 0
        start = time.perf_counter()
        while time.perf_counter() - start < 1.0:
            ret, _ = cap.read()
            if ret:
                frames += 1
        
        cap.release()
        return frames
    except:
        return 0.0
    finally:
        if old_log:
            os.environ['OPENCV_LOG_LEVEL'] = old_log
        elif 'OPENCV_LOG_LEVEL' in os.environ:
            del os.environ['OPENCV_LOG_LEVEL']


def cmd_list_devices(verbose: bool = True):
    """List all detected cameras."""
    print("\n" + "="*60)
    print("  CAMERA ENUMERATION")
    print("="*60 + "\n")
    
    # OpenCV enumeration (simple)
    print("📷 Detected Cameras:")
    cv_cameras = enumerate_opencv_cameras()
    
    if not cv_cameras:
        print("  No cameras found!")
        return
    
    for cam in cv_cameras:
        mono = " [MONO]" if cam['is_monochrome'] else ""
        readable = "" if cam['is_readable'] else " [NOT READABLE]"
        print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']:.0f}fps ({cam['fourcc']}){mono}{readable}")
    
    # Quick FPS test on camera 0
    print(f"\n📋 Testing camera 0 modes:")
    test_camera_modes(0, verbose)


def test_camera_modes(device_index: int, verbose: bool = True):
    """Test what modes the camera actually supports."""
    test_configs = [
        (1280, 800, 120, 'MJPG'),
        (1280, 800, 120, 'YUYV'),
        (1280, 720, 120, 'MJPG'),
        (1280, 720, 120, 'YUYV'),
        (640, 480, 120, 'MJPG'),
        (640, 480, 120, 'YUYV'),
        (320, 240, 120, 'MJPG'),
        (320, 240, 120, 'YUYV'),
    ]
    
    results = []
    
    for width, height, fps, fmt in test_configs:
        cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
        if not cap.isOpened():
            cap = cv2.VideoCapture(device_index)
        
        if not cap.isOpened():
            print(f"  ✗ Cannot open camera")
            break
        
        # Set format first
        fourcc = encode_fourcc(fmt)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        
        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))
        
        # Try to read frames
        success_count = 0
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None:
                success_count += 1
        
        cap.release()
        
        readable = success_count >= 10
        match = actual_w == width and actual_h == height
        
        status = "✓" if readable and match else "○" if readable else "✗"
        
        print(f"  {status} {width}x{height}@{fps} {fmt:4s} → {actual_w}x{actual_h}@{actual_fps:.0f} {actual_fourcc} (frames: {success_count}/20)")
        
        results.append({
            'requested': f"{width}x{height}@{fps} {fmt}",
            'actual': f"{actual_w}x{actual_h}@{actual_fps} {actual_fourcc}",
            'readable': readable,
            'match': match,
        })
    
    return results


def cmd_measure(
    index: Optional[int],
    width: int,
    height: int,
    fps: float,
    pixel_format: PixelFormat,
    duration: float = 5.0
):
    """Measure actual FPS performance."""
    print("\n" + "="*60)
    print("  FPS MEASUREMENT")
    print("="*60 + "\n")
    
    capture = ArducamCapture(
        index=index,
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
        verbose=True
    )
    
    if not capture.open():
        print("\n❌ Failed to open camera")
        return
    
    info = capture.get_info()
    actual = info['actual']
    
    print(f"\n📊 Measuring FPS for {duration} seconds...")
    print(f"   Resolution: {actual['width']}x{actual['height']}")
    print(f"   Reported FPS: {actual['fps']}")
    print(f"   Format: {actual['format']}")
    print()
    
    measured_fps = capture.measure_fps(duration)
    
    print(f"\n{'='*60}")
    print(f"  RESULTS")
    print(f"{'='*60}")
    print(f"  Backend: {info['backend']}")
    print(f"  Requested: {width}x{height} @ {fps}fps ({pixel_format.value})")
    print(f"  Actual: {actual['width']}x{actual['height']} @ {actual['fps']}fps ({actual['format']})")
    print(f"  Measured FPS: {measured_fps:.1f}")
    print(f"  Achievement: {(measured_fps / fps) * 100:.1f}%")
    
    # USB bandwidth analysis
    mode = CameraMode(actual['width'], actual['height'], measured_fps, actual['format'], True)
    bandwidth = mode.bandwidth_mbps()
    usb2_limit = 480  # Mbps theoretical, ~400 practical
    
    print(f"\n  Estimated bandwidth: {bandwidth:.1f} Mbps")
    if bandwidth > 400:
        print(f"  ⚠️  Exceeds USB 2.0 practical limit (~400 Mbps)")
        print(f"      Consider using MJPG format or lower resolution")
    else:
        print(f"  ✓ Within USB 2.0 bandwidth")
    
    print(f"{'='*60}\n")
    
    capture.close()


def cmd_preview(
    index: Optional[int],
    width: int,
    height: int,
    fps: float,
    pixel_format: PixelFormat
):
    """Show live preview with stats overlay."""
    capture = ArducamCapture(
        index=index,
        width=width,
        height=height,
        fps=fps,
        pixel_format=pixel_format,
        verbose=True
    )
    
    if not capture.open():
        print("\n❌ Failed to open camera")
        return
    
    info = capture.get_info()
    actual = info['actual']
    
    print("\n📺 Showing preview (press 'q' to quit, 'm' to measure FPS)...")
    
    cv2.namedWindow("Arducam Preview", cv2.WINDOW_NORMAL)
    
    fps_history = []
    last_time = time.perf_counter()
    
    while True:
        ret, frame = capture.read()
        
        if ret and frame is not None:
            current_time = time.perf_counter()
            delta = current_time - last_time
            last_time = current_time
            
            if delta > 0:
                instant_fps = 1.0 / delta
                fps_history.append(instant_fps)
                if len(fps_history) > 60:
                    fps_history.pop(0)
            
            avg_fps = np.mean(fps_history) if fps_history else 0
            
            # Draw overlay
            display = frame.copy() if len(frame.shape) == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            h, w = display.shape[:2]
            
            # Stats
            cv2.putText(display, f"FPS: {avg_fps:.1f} (target: {fps})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Resolution: {w}x{h}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Format: {actual['format']}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Backend: {info['backend']}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Frame: {capture._frame_count}", (10, 150),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display, "Press 'q' to quit, 'm' to measure", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow("Arducam Preview", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            print("\n[Measure] Running 5-second FPS measurement...")
            measured = capture.measure_fps(5.0)
            print(f"[Measure] Result: {measured:.1f} FPS\n")
    
    cv2.destroyAllWindows()
    capture.close()
    
    print(f"\nSession stats:")
    print(f"  Total frames: {capture._frame_count}")
    print(f"  Dropped frames: {capture._dropped_frames}")


# =============================================================================
# MJPEG Diagnostic Test
# =============================================================================

def diagnose_mjpeg(device_index: int, width: int = 1280, height: int = 800, duration: float = 2.0):
    """
    Explicit MJPEG negotiation test.
    
    Tests whether MJPEG is actually being used (not just requested).
    """
    print("\n" + "="*60)
    print("  MJPEG DIAGNOSTIC TEST")
    print("="*60 + "\n")
    
    print(f"Testing: {width}x{height} with MJPEG at index {device_index}")
    print(f"Duration: {duration} seconds\n")
    
    # Open with AVFoundation
    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        cap = cv2.VideoCapture(device_index)
    
    if not cap.isOpened():
        print("❌ Failed to open camera")
        return False
    
    # Set MJPG format FIRST
    fourcc_mjpg = encode_fourcc('MJPG')
    result_fourcc = cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
    print(f"Set FOURCC to MJPG: {'accepted' if result_fourcc else 'ignored'}")
    
    # Then resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Then FPS
    cap.set(cv2.CAP_PROP_FPS, 120)
    
    # Read back actual values
    actual_fourcc = decode_fourcc(int(cap.get(cv2.CAP_PROP_FOURCC)))
    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps_reported = cap.get(cv2.CAP_PROP_FPS)
    backend = cap.getBackendName()
    
    print(f"\nNegotiated settings:")
    print(f"  Backend: {backend}")
    print(f"  FOURCC: {actual_fourcc}")
    print(f"  Resolution: {actual_w}x{actual_h}")
    print(f"  FPS (reported): {actual_fps_reported}")
    
    # Check if MJPEG was actually set
    is_mjpeg = actual_fourcc.upper() in ('MJPG', 'MJPE', 'JPEG')
    print(f"\n  MJPEG active: {'✓ YES' if is_mjpeg else '✗ NO'}")
    
    if not is_mjpeg:
        print(f"  ⚠️  AVFoundation ignored MJPEG request, using {actual_fourcc}")
        print(f"      This is a known macOS/OpenCV limitation.")
    
    # Measure actual FPS
    print(f"\nMeasuring actual FPS for {duration}s...")
    
    frame_count = 0
    start_time = time.perf_counter()
    frame_times = []
    last_time = start_time
    warmup_failures = 0
    
    while True:
        current_time = time.perf_counter()
        elapsed = current_time - start_time
        
        if elapsed >= duration:
            break
        
        ret, frame = cap.read()
        
        if ret and frame is not None:
            frame_count += 1
            frame_times.append(current_time - last_time)
            last_time = current_time
        else:
            if frame_count == 0:
                warmup_failures += 1
                if warmup_failures > 30:
                    print("❌ Too many warmup failures, aborting")
                    cap.release()
                    return False
    
    cap.release()
    
    if frame_count == 0:
        print("❌ No frames captured")
        return False
    
    measured_fps = frame_count / duration
    avg_interval = np.mean(frame_times[1:]) * 1000 if len(frame_times) > 1 else 0
    
    print(f"\nResults:")
    print(f"  Frames captured: {frame_count}")
    print(f"  Measured FPS: {measured_fps:.1f}")
    print(f"  Avg frame interval: {avg_interval:.2f}ms")
    
    # Bandwidth analysis
    if is_mjpeg:
        # MJPEG: estimate ~0.3 bytes/pixel with compression
        bandwidth = actual_w * actual_h * 0.3 * measured_fps * 8 / 1_000_000
    else:
        # Raw: 2 bytes/pixel for YUYV
        bandwidth = actual_w * actual_h * 2 * measured_fps * 8 / 1_000_000
    
    print(f"  Estimated bandwidth: {bandwidth:.1f} Mbps")
    
    # Assessment
    print(f"\n{'='*60}")
    if measured_fps >= 100:
        print("✓ SUCCESS: Achieving high frame rate")
        if is_mjpeg:
            print("  MJPEG compression is working")
        else:
            print("  High FPS without MJPEG (sufficient bandwidth)")
    elif measured_fps >= 60:
        print("○ PARTIAL: Moderate frame rate achieved")
        if not is_mjpeg:
            print("  Try lower resolution or ensure MJPEG is enabled")
    else:
        print("✗ FAILED: Low frame rate")
        print("  Likely USB 2.0 bandwidth limitation")
        print("  Recommendations:")
        print("    1. Use lower resolution (640x480)")
        print("    2. Verify camera supports MJPEG")
        print("    3. Try USB 3.0 port if available")
    print(f"{'='*60}\n")
    
    return measured_fps >= 100


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Arducam B0332 OV9281 High-Speed Capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List all cameras
    python arducam_capture.py --list
    
    # Measure FPS at 1280x800
    python arducam_capture.py --measure --width 1280 --height 800 --fps 120
    
    # Use specific camera with MJPEG
    python arducam_capture.py --index 0 --format mjpg --width 1280 --height 800
    
    # Show live preview
    python arducam_capture.py --preview
    
    # Run MJPEG diagnostic
    python arducam_capture.py --diagnose-mjpeg
        """
    )
    
    parser.add_argument('--index', type=int, default=None,
                        help='Camera device index (auto-detect if not specified)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Frame width (default: 1280)')
    parser.add_argument('--height', type=int, default=800,
                        help='Frame height (default: 800)')
    parser.add_argument('--fps', type=float, default=120,
                        help='Target FPS (default: 120)')
    parser.add_argument('--format', type=str, choices=['mjpg', 'yuyv', 'auto'], default='auto',
                        help='Pixel format (default: auto)')
    
    parser.add_argument('--list', action='store_true',
                        help='List discovered devices and formats')
    parser.add_argument('--measure', action='store_true',
                        help='Measure actual FPS over 5 seconds')
    parser.add_argument('--preview', action='store_true',
                        help='Show live preview window')
    parser.add_argument('--diagnose-mjpeg', action='store_true',
                        help='Run MJPEG negotiation diagnostic')
    
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Measurement duration in seconds (default: 5)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Parse format
    format_map = {
        'mjpg': PixelFormat.MJPG,
        'yuyv': PixelFormat.YUYV,
        'auto': PixelFormat.AUTO,
    }
    pixel_format = format_map[args.format]
    
    # Run requested command
    if args.list:
        cmd_list_devices(not args.quiet)
    elif args.diagnose_mjpeg:
        diagnose_mjpeg(args.index or 0, args.width, args.height, args.duration)
    elif args.measure:
        cmd_measure(args.index, args.width, args.height, args.fps, pixel_format, args.duration)
    elif args.preview:
        cmd_preview(args.index, args.width, args.height, args.fps, pixel_format)
    else:
        # Default: show help
        parser.print_help()


if __name__ == "__main__":
    main()
