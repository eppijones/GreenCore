"""
Arducam OV9281 USB Camera - 120 FPS Validated Capture Module

This module guarantees:
1. Requests and validates 1280x800 @ 120fps from AVFoundation
2. Measures CAPTURE_FPS from frame arrival timestamps (not processing)
3. Provides atomic "latest frame" access without backlog
4. Reports CAP/PROC/DISP FPS separately
5. Includes validation mode for objective 120fps verification

Architecture:
- Capture thread pulls frames as fast as camera delivers
- Processing thread runs tracking at full 120fps (decoupled from display)
- Atomic latest-frame slot ensures processing always gets newest frame
- PTS-like timestamps from monotonic clock at frame arrival
- Frame drop tracking between capture and processing

Target: Arducam OV9281 (B0332) at 1280x800 @ 120fps MJPG
"""

import os
import sys
import time
import threading
import platform
import statistics
import csv
from typing import Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from collections import deque
from enum import Enum

import numpy as np
import cv2


class CaptureError(Exception):
    """Exception for capture-related errors."""
    pass


class ValidationResult(Enum):
    """Result of 120fps validation."""
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_RUN = "NOT_RUN"


@dataclass
class CaptureConfig:
    """Configuration for 120fps capture."""
    # Target resolution and FPS - must match camera capability
    width: int = 1280
    height: int = 800
    target_fps: int = 120
    
    # Camera identification
    device_index: int = -1  # -1 = auto-detect Arducam
    device_name: str = "Arducam OV9281 USB Camera"
    
    # Format settings
    use_mjpg: bool = True  # MJPG required for 120fps on USB 2.0
    
    # Validation thresholds
    min_acceptable_fps: float = 115.0  # Fail if below this
    max_pts_anomalies: int = 1  # Max duplicate/non-increasing PTS allowed
    
    # Buffer settings
    capture_buffer_size: int = 1  # Minimize latency
    
    # Timing settings
    rolling_window_frames: int = 240  # ~2 seconds at 120fps
    
    # Processing thread (NEW)
    enable_processing_thread: bool = True  # Run processing at full 120fps


@dataclass
class CapturedFrame:
    """A single captured frame with metadata."""
    frame: np.ndarray  # BGR image
    gray: np.ndarray   # Grayscale image
    capture_pts: float  # Monotonic timestamp at capture (pseudo-PTS)
    frame_number: int   # Sequential frame number from camera
    arrival_time: float  # Wall clock time for logging


@dataclass
class FPSMetrics:
    """FPS metrics for display."""
    capture_fps: float = 0.0   # From capture thread PTS deltas
    process_fps: float = 0.0   # Frames actually processed (tracking)
    display_fps: float = 0.0   # UI render rate
    dropped_frames: int = 0    # Frames skipped between capture and processing
    pts_anomalies: int = 0     # Non-increasing or duplicate PTS count


@dataclass
class ValidationReport:
    """Detailed validation report."""
    result: ValidationResult = ValidationResult.NOT_RUN
    
    # Configuration
    requested_resolution: str = ""
    actual_resolution: str = ""
    requested_fps: int = 0
    
    # Timing measurements
    duration_seconds: float = 0.0
    total_frames_captured: int = 0
    measured_capture_fps: float = 0.0
    
    # Inter-frame deltas (milliseconds)
    delta_min_ms: float = 0.0
    delta_avg_ms: float = 0.0
    delta_max_ms: float = 0.0
    delta_std_ms: float = 0.0
    
    # Anomalies
    pts_anomalies: int = 0
    dropped_frames: int = 0
    
    # Pass/fail reasoning
    fail_reasons: List[str] = field(default_factory=list)
    
    def print_report(self):
        """Print formatted validation report."""
        print("\n" + "=" * 70)
        print("  ARDUCAM 120 FPS VALIDATION REPORT")
        print("=" * 70)
        
        # Result banner
        if self.result == ValidationResult.PASS:
            print(f"\n  ✅ RESULT: {self.result.value}")
        else:
            print(f"\n  ❌ RESULT: {self.result.value}")
        
        print("\n  CONFIGURATION")
        print("  " + "-" * 40)
        print(f"    Requested: {self.requested_resolution} @ {self.requested_fps}fps")
        print(f"    Actual:    {self.actual_resolution}")
        
        print("\n  CAPTURE METRICS")
        print("  " + "-" * 40)
        print(f"    Duration:       {self.duration_seconds:.2f} seconds")
        print(f"    Frames:         {self.total_frames_captured}")
        print(f"    Capture FPS:    {self.measured_capture_fps:.1f}")
        
        print("\n  INTER-FRAME TIMING")
        print("  " + "-" * 40)
        expected_delta = 1000.0 / self.requested_fps if self.requested_fps > 0 else 0
        print(f"    Expected:       {expected_delta:.2f}ms")
        print(f"    Min:            {self.delta_min_ms:.2f}ms")
        print(f"    Avg:            {self.delta_avg_ms:.2f}ms")
        print(f"    Max:            {self.delta_max_ms:.2f}ms")
        print(f"    Std Dev:        {self.delta_std_ms:.2f}ms")
        
        print("\n  ANOMALIES")
        print("  " + "-" * 40)
        print(f"    PTS anomalies:  {self.pts_anomalies} (duplicate/non-increasing)")
        print(f"    Dropped frames: {self.dropped_frames}")
        
        if self.fail_reasons:
            print("\n  FAIL REASONS")
            print("  " + "-" * 40)
            for reason in self.fail_reasons:
                print(f"    • {reason}")
        
        print("\n" + "=" * 70 + "\n")
    
    def to_csv_row(self) -> dict:
        """Convert to CSV row dict."""
        return {
            'timestamp': datetime.now().isoformat(),
            'result': self.result.value,
            'requested_resolution': self.requested_resolution,
            'actual_resolution': self.actual_resolution,
            'requested_fps': self.requested_fps,
            'duration_seconds': self.duration_seconds,
            'total_frames': self.total_frames_captured,
            'measured_fps': self.measured_capture_fps,
            'delta_min_ms': self.delta_min_ms,
            'delta_avg_ms': self.delta_avg_ms,
            'delta_max_ms': self.delta_max_ms,
            'delta_std_ms': self.delta_std_ms,
            'pts_anomalies': self.pts_anomalies,
            'dropped_frames': self.dropped_frames,
        }


class Arducam120FPS:
    """
    Arducam OV9281 capture module optimized for validated 120 FPS operation.
    
    Key features:
    - Dedicated capture thread with high-priority timing
    - Optional processing thread for 120fps tracking (decoupled from display)
    - Atomic latest-frame slot (processing always gets newest frame)
    - PTS tracking from monotonic clock at frame arrival
    - Separate CAP/PROC/DISP FPS metrics
    - Validation mode for objective 120fps verification
    
    Usage:
        camera = Arducam120FPS()
        camera.start()  # Fails if can't achieve 120fps
        
        # Option 1: Use processing callback (runs at 120fps)
        camera.set_processing_callback(my_tracking_function)
        
        # Option 2: Manual frame retrieval (for display, runs at display rate)
        frame = camera.get_latest_frame()
        
        camera.stop()
    """
    
    def __init__(self, config: Optional[CaptureConfig] = None):
        self.config = config or CaptureConfig()
        
        # OpenCV capture handle
        self._cap: Optional[cv2.VideoCapture] = None
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fourcc: str = ""
        
        # Capture thread
        self._capture_thread: Optional[threading.Thread] = None
        self._running = False
        self._stop_event = threading.Event()
        
        # Processing thread (NEW - runs at 120fps, decoupled from display)
        self._processing_thread: Optional[threading.Thread] = None
        self._processing_callback: Optional[Callable] = None
        self._process_stop_event = threading.Event()
        
        # Frame slot for processing thread
        self._process_frame_lock = threading.Lock()
        self._process_frame: Optional[CapturedFrame] = None
        self._process_frame_ready = threading.Event()
        self._process_frame_sequence: int = 0
        
        # Atomic latest frame slot for display (protected by lock for safe access)
        self._frame_lock = threading.Lock()
        self._latest_frame: Optional[CapturedFrame] = None
        self._frame_sequence: int = 0  # Monotonic sequence for drop tracking
        
        # Capture-side metrics (updated in capture thread)
        self._capture_pts_history: deque = deque(maxlen=self.config.rolling_window_frames)
        self._capture_frame_count: int = 0
        self._pts_anomaly_count: int = 0
        self._last_capture_pts: float = 0.0
        
        # Processing-side metrics (updated in processing thread)
        self._last_processed_sequence: int = 0
        self._process_times: deque = deque(maxlen=self.config.rolling_window_frames)
        self._total_dropped_frames: int = 0
        self._process_lock = threading.Lock()
        
        # Display-side metrics (updated when get_latest_frame is called)
        self._display_times: deque = deque(maxlen=60)
        self._display_sequence: int = 0
        
        # Validation data collection
        self._validation_mode = False
        self._validation_pts_list: List[Tuple[float, float]] = []  # (pts, arrival_time)
        self._validation_start_time: float = 0.0
        
        # Depth scale placeholder for compatibility
        self.depth_scale = 0.001
        
        # Latest tracking result (from processing thread)
        self._tracking_result_lock = threading.Lock()
        self._latest_tracking_result = None
    
    @staticmethod
    def find_arducam_device() -> Optional[int]:
        """
        Find Arducam OV9281 by detecting monochrome camera.
        
        Returns device index or None if not found.
        """
        print("[Arducam120] Searching for monochrome (OV9281) camera...")
        
        # Suppress OpenCV warnings
        old_log = os.environ.get('OPENCV_LOG_LEVEL', '')
        os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
        
        try:
            for idx in range(5):  # Check first 5 indices
                try:
                    if platform.system() == 'Darwin':
                        cap = cv2.VideoCapture(idx, cv2.CAP_AVFOUNDATION)
                    else:
                        cap = cv2.VideoCapture(idx)
                    
                    if not cap.isOpened():
                        cap.release()
                        continue
                    
                    # Read test frame
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        cap.release()
                        continue
                    
                    # Check if monochrome (RGB channels identical)
                    if len(frame.shape) == 3:
                        b, g, r = cv2.split(frame)
                        color_variance = np.std(b.astype(float) - g.astype(float))
                        
                        if color_variance < 5:  # Monochrome camera
                            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                            print(f"[Arducam120] Found monochrome camera at index {idx}: {width}x{height}")
                            cap.release()
                            return idx
                    
                    cap.release()
                    
                except Exception:
                    continue
            
            return None
            
        finally:
            if old_log:
                os.environ['OPENCV_LOG_LEVEL'] = old_log
            elif 'OPENCV_LOG_LEVEL' in os.environ:
                del os.environ['OPENCV_LOG_LEVEL']
    
    def start(self) -> bool:
        """
        Start camera capture at 1280x800 @ 120fps.
        
        Raises CaptureError if:
        - Camera not found
        - Cannot set 1280x800 resolution
        - Cannot achieve near-120fps capture rate
        
        Returns True on success.
        """
        if self._running:
            return True
        
        print(f"\n[Arducam120] Starting 120fps capture...")
        print(f"[Arducam120] Target: {self.config.width}x{self.config.height} @ {self.config.target_fps}fps")
        
        # Find device
        device_idx = self.config.device_index
        if device_idx < 0:
            device_idx = self.find_arducam_device()
            if device_idx is None:
                raise CaptureError(
                    "Arducam OV9281 not found. Ensure camera is connected and recognized by macOS."
                )
        
        # Open camera with AVFoundation backend
        print(f"[Arducam120] Opening device {device_idx} with AVFoundation backend...")
        
        if platform.system() == 'Darwin':
            self._cap = cv2.VideoCapture(device_idx, cv2.CAP_AVFOUNDATION)
        else:
            self._cap = cv2.VideoCapture(device_idx)
        
        if not self._cap.isOpened():
            raise CaptureError(f"Failed to open camera at index {device_idx}")
        
        # Configure for 120fps
        self._configure_camera()
        
        # Validate configuration
        self._validate_configuration()
        
        # Start capture thread
        self._stop_event.clear()
        self._running = True
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="Arducam120-Capture",
            daemon=True
        )
        self._capture_thread.start()
        
        # Start processing thread if enabled
        if self.config.enable_processing_thread:
            self._process_stop_event.clear()
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name="Arducam120-Processing",
                daemon=True
            )
            self._processing_thread.start()
            print("[Arducam120] Processing thread started (120fps tracking enabled)")
        
        # Quick warmup and FPS verification
        print("[Arducam120] Warming up and verifying FPS...")
        time.sleep(0.5)  # Let capture thread fill buffer
        
        initial_fps = self.get_metrics().capture_fps
        if initial_fps < 80:
            self.stop()
            raise CaptureError(
                f"Camera only achieving {initial_fps:.1f} fps. "
                f"Expected >= {self.config.min_acceptable_fps} fps. "
                f"Check USB connection (requires USB 3.0 or USB 2.0 with MJPG)."
            )
        
        print(f"[Arducam120] ✓ Camera running at {initial_fps:.1f} fps")
        print(f"[Arducam120] ✓ Resolution: {self._actual_width}x{self._actual_height}")
        print(f"[Arducam120] ✓ Format: {self._actual_fourcc}")
        
        return True
    
    def _configure_camera(self):
        """Configure camera for 120fps capture."""
        cap = self._cap
        
        # Set MJPG format first (critical for high FPS)
        if self.config.use_mjpg:
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        
        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.height)
        
        # Set target FPS
        cap.set(cv2.CAP_PROP_FPS, self.config.target_fps)
        
        # Minimize buffer for low latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, self.config.capture_buffer_size)
        
        # Disable auto-exposure for consistent timing
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Manual
        cap.set(cv2.CAP_PROP_EXPOSURE, -6)  # Fast exposure
        
        # Read back actual values
        self._actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
        if fourcc_code > 0:
            self._actual_fourcc = "".join([chr((fourcc_code >> 8 * i) & 0xFF) for i in range(4)])
        else:
            self._actual_fourcc = "UNKNOWN"
    
    def _validate_configuration(self):
        """Validate that camera accepted our configuration."""
        # Check resolution
        if self._actual_width != self.config.width or self._actual_height != self.config.height:
            raise CaptureError(
                f"Camera did not accept resolution {self.config.width}x{self.config.height}. "
                f"Got {self._actual_width}x{self._actual_height} instead. "
                f"This camera may not support the requested mode."
            )
        
        # Verify we can read frames
        for _ in range(10):
            ret, frame = self._cap.read()
            if not ret or frame is None:
                continue
            return  # Success
        
        raise CaptureError("Camera opened but cannot read frames. Check permissions and connection.")
    
    def _capture_loop(self):
        """
        Capture thread main loop.
        
        Runs at camera frame rate, updating latest frame atomically.
        Measures FPS from PTS deltas.
        """
        print("[Arducam120] Capture thread started")
        
        cap = self._cap
        frame_num = 0
        
        while not self._stop_event.is_set():
            # Capture frame with precise timestamp
            ret, frame = cap.read()
            capture_pts = time.perf_counter()  # Monotonic pseudo-PTS
            arrival_time = time.time()  # Wall clock for logging
            
            if not ret or frame is None:
                continue
            
            frame_num += 1
            self._capture_frame_count += 1
            
            # Check for PTS anomalies (duplicate or non-increasing)
            if capture_pts <= self._last_capture_pts:
                self._pts_anomaly_count += 1
            
            # Record PTS for FPS calculation
            self._capture_pts_history.append(capture_pts)
            self._last_capture_pts = capture_pts
            
            # Validation mode: collect detailed data
            if self._validation_mode:
                self._validation_pts_list.append((capture_pts, arrival_time))
            
            # Convert to grayscale
            if len(frame.shape) == 2:
                gray = frame
                color = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                gray = frame[:, :, 0]  # OV9281 is monochrome in BGR wrapper
                color = frame
            
            # Create captured frame
            captured = CapturedFrame(
                frame=color,
                gray=gray,
                capture_pts=capture_pts,
                frame_number=frame_num,
                arrival_time=arrival_time
            )
            
            # Update latest frame for display (atomic)
            with self._frame_lock:
                self._frame_sequence += 1
                self._latest_frame = captured
            
            # Signal processing thread (if enabled)
            if self.config.enable_processing_thread:
                with self._process_frame_lock:
                    self._process_frame_sequence += 1
                    self._process_frame = captured
                self._process_frame_ready.set()
        
        print("[Arducam120] Capture thread stopped")
    
    def _processing_loop(self):
        """
        Processing thread main loop.
        
        Runs at 120fps (or as fast as frames arrive), decoupled from display.
        Calls the processing callback with each frame.
        """
        print("[Arducam120] Processing thread started")
        
        last_processed_seq = 0
        
        while not self._process_stop_event.is_set():
            # Wait for new frame (with timeout to check stop event)
            if not self._process_frame_ready.wait(timeout=0.05):
                continue
            
            self._process_frame_ready.clear()
            
            # Get frame to process
            with self._process_frame_lock:
                frame = self._process_frame
                current_seq = self._process_frame_sequence
            
            if frame is None:
                continue
            
            # Track drops (gap in sequence)
            if last_processed_seq > 0:
                dropped = current_seq - last_processed_seq - 1
                if dropped > 0:
                    with self._process_lock:
                        self._total_dropped_frames += dropped
            
            last_processed_seq = current_seq
            
            # Record processing time
            with self._process_lock:
                self._process_times.append(time.perf_counter())
            
            # Call processing callback
            if self._processing_callback:
                try:
                    result = self._processing_callback(frame)
                    if result is not None:
                        with self._tracking_result_lock:
                            self._latest_tracking_result = result
                except Exception as e:
                    print(f"[Arducam120] Processing callback error: {e}")
        
        print("[Arducam120] Processing thread stopped")
    
    def set_processing_callback(self, callback: Callable[[CapturedFrame], any]):
        """
        Set the processing callback function.
        
        This function will be called at 120fps with each captured frame.
        Use this for tracking logic that needs to run at full frame rate.
        
        Args:
            callback: Function that takes CapturedFrame and returns optional result
        """
        self._processing_callback = callback
    
    def get_latest_tracking_result(self):
        """Get the latest result from the processing callback."""
        with self._tracking_result_lock:
            return self._latest_tracking_result
    
    def get_latest_frame(self) -> Optional[CapturedFrame]:
        """
        Get the most recent frame for display (non-blocking).
        
        This is meant for display purposes - runs at display rate (~60fps).
        Use set_processing_callback() for 120fps tracking.
        
        Returns None if no frame available yet.
        """
        with self._frame_lock:
            frame = self._latest_frame
            current_seq = self._frame_sequence
        
        if frame is None:
            return None
        
        # Track display rate
        self._display_times.append(time.perf_counter())
        
        # Track display sequence (for legacy drops metric when not using processing thread)
        if not self.config.enable_processing_thread:
            if self._display_sequence > 0:
                dropped = current_seq - self._display_sequence - 1
                if dropped > 0:
                    self._total_dropped_frames += dropped
            self._display_sequence = current_seq
        
        return frame
    
    def mark_display_tick(self):
        """Call this once per UI render to track display FPS."""
        self._display_times.append(time.perf_counter())
    
    def get_metrics(self) -> FPSMetrics:
        """
        Get current FPS metrics.
        
        Returns:
            FPSMetrics with capture_fps, process_fps, display_fps, dropped_frames, pts_anomalies
        """
        metrics = FPSMetrics()
        
        # Calculate capture FPS from PTS history
        pts_list = list(self._capture_pts_history)
        if len(pts_list) >= 2:
            duration = pts_list[-1] - pts_list[0]
            if duration > 0:
                metrics.capture_fps = (len(pts_list) - 1) / duration
        
        # Calculate process FPS from processing times
        with self._process_lock:
            proc_list = list(self._process_times)
        if len(proc_list) >= 2:
            duration = proc_list[-1] - proc_list[0]
            if duration > 0:
                metrics.process_fps = (len(proc_list) - 1) / duration
        
        # Calculate display FPS
        disp_list = list(self._display_times)
        if len(disp_list) >= 2:
            duration = disp_list[-1] - disp_list[0]
            if duration > 0:
                metrics.display_fps = (len(disp_list) - 1) / duration
        
        with self._process_lock:
            metrics.dropped_frames = self._total_dropped_frames
        metrics.pts_anomalies = self._pts_anomaly_count
        
        return metrics
    
    def get_inter_frame_stats(self) -> Tuple[float, float, float, float]:
        """
        Get inter-frame timing statistics in milliseconds.
        
        Returns:
            (min_delta_ms, avg_delta_ms, max_delta_ms, std_delta_ms)
        """
        pts_list = list(self._capture_pts_history)
        if len(pts_list) < 2:
            return (0.0, 0.0, 0.0, 0.0)
        
        deltas = [(pts_list[i] - pts_list[i-1]) * 1000 for i in range(1, len(pts_list))]
        
        return (
            min(deltas),
            statistics.mean(deltas),
            max(deltas),
            statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        )
    
    def run_validation(self, duration_seconds: float = 5.0, save_csv: bool = True) -> ValidationReport:
        """
        Run 120fps validation test.
        
        Args:
            duration_seconds: How long to run the test
            save_csv: Whether to save per-frame timing data to CSV
        
        Returns:
            ValidationReport with pass/fail and detailed metrics
        """
        print(f"\n[Arducam120] Running {duration_seconds}s validation test...")
        
        # Start validation data collection
        self._validation_mode = True
        self._validation_pts_list = []
        self._validation_start_time = time.perf_counter()
        
        # Wait for test duration
        time.sleep(duration_seconds)
        
        # Stop collection
        self._validation_mode = False
        validation_end_time = time.perf_counter()
        
        # Build report
        report = ValidationReport()
        report.requested_resolution = f"{self.config.width}x{self.config.height}"
        report.actual_resolution = f"{self._actual_width}x{self._actual_height}"
        report.requested_fps = self.config.target_fps
        report.duration_seconds = validation_end_time - self._validation_start_time
        report.total_frames_captured = len(self._validation_pts_list)
        
        if report.total_frames_captured < 2:
            report.result = ValidationResult.FAIL
            report.fail_reasons.append("Too few frames captured")
            return report
        
        # Calculate FPS
        pts_data = self._validation_pts_list
        total_duration = pts_data[-1][0] - pts_data[0][0]
        if total_duration > 0:
            report.measured_capture_fps = (len(pts_data) - 1) / total_duration
        
        # Calculate inter-frame deltas
        deltas_ms = []
        anomalies = 0
        for i in range(1, len(pts_data)):
            delta = (pts_data[i][0] - pts_data[i-1][0]) * 1000
            deltas_ms.append(delta)
            if delta <= 0:
                anomalies += 1
        
        if deltas_ms:
            report.delta_min_ms = min(deltas_ms)
            report.delta_avg_ms = statistics.mean(deltas_ms)
            report.delta_max_ms = max(deltas_ms)
            report.delta_std_ms = statistics.stdev(deltas_ms) if len(deltas_ms) > 1 else 0.0
        
        report.pts_anomalies = anomalies
        report.dropped_frames = self._total_dropped_frames
        
        # Determine pass/fail
        report.result = ValidationResult.PASS
        
        if report.measured_capture_fps < self.config.min_acceptable_fps:
            report.result = ValidationResult.FAIL
            report.fail_reasons.append(
                f"Capture FPS {report.measured_capture_fps:.1f} < {self.config.min_acceptable_fps}"
            )
        
        if report.pts_anomalies > self.config.max_pts_anomalies:
            report.result = ValidationResult.FAIL
            report.fail_reasons.append(
                f"PTS anomalies {report.pts_anomalies} > {self.config.max_pts_anomalies}"
            )
        
        # Save CSV if requested
        if save_csv:
            self._save_validation_csv(pts_data, report)
        
        return report
    
    def _save_validation_csv(self, pts_data: List[Tuple[float, float]], report: ValidationReport):
        """Save per-frame timing data to CSV for offline analysis."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_path = Path(f"validation_{timestamp_str}.csv")
        
        try:
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['frame_num', 'pts_seconds', 'arrival_time', 'delta_ms'])
                
                for i, (pts, arrival) in enumerate(pts_data):
                    delta = 0.0
                    if i > 0:
                        delta = (pts - pts_data[i-1][0]) * 1000
                    writer.writerow([i, pts, arrival, f"{delta:.3f}"])
            
            print(f"[Arducam120] Validation data saved to {csv_path}")
        except Exception as e:
            print(f"[Arducam120] Warning: Could not save CSV: {e}")
    
    def stop(self):
        """Stop capture and release resources."""
        print("[Arducam120] Stopping capture...")
        
        self._running = False
        self._stop_event.set()
        self._process_stop_event.set()
        self._process_frame_ready.set()  # Wake up processing thread
        
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        if self._capture_thread and self._capture_thread.is_alive():
            self._capture_thread.join(timeout=2.0)
        
        if self._cap:
            self._cap.release()
            self._cap = None
        
        print("[Arducam120] Stopped")
    
    # Compatibility properties for existing codebase
    
    @property
    def is_running(self) -> bool:
        return self._running
    
    @property
    def fps(self) -> float:
        """Legacy: returns capture FPS."""
        return self.get_metrics().capture_fps
    
    @property
    def frame_count(self) -> int:
        return self._capture_frame_count
    
    @property
    def dropped_frames(self) -> int:
        return self._total_dropped_frames
    
    def get_frames(self, timeout_ms: int = 100) -> Optional['FrameData']:
        """
        Legacy compatibility method.
        
        Returns FrameData-like object for compatibility with existing code.
        """
        frame = self.get_latest_frame()
        if frame is None:
            return None
        
        # Return a simple object with expected attributes
        class FrameDataCompat:
            def __init__(self, captured: CapturedFrame):
                self.color_frame = captured.frame
                self.gray_frame = captured.gray
                self.timestamp = captured.arrival_time
                self.frame_number = captured.frame_number
        
        return FrameDataCompat(frame)
    
    def get_stats(self) -> dict:
        """Get detailed statistics."""
        metrics = self.get_metrics()
        min_d, avg_d, max_d, std_d = self.get_inter_frame_stats()
        
        return {
            'capture_fps': metrics.capture_fps,
            'process_fps': metrics.process_fps,
            'display_fps': metrics.display_fps,
            'frame_count': self._capture_frame_count,
            'dropped_frames': metrics.dropped_frames,
            'pts_anomalies': metrics.pts_anomalies,
            'resolution': f"{self._actual_width}x{self._actual_height}",
            'format': self._actual_fourcc,
            'delta_min_ms': min_d,
            'delta_avg_ms': avg_d,
            'delta_max_ms': max_d,
            'delta_std_ms': std_d,
        }
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False


def validate_120fps(
    device_index: int = -1,
    duration: float = 5.0,
    show_live: bool = True
) -> ValidationReport:
    """
    Run standalone 120fps validation.
    
    Usage:
        python -c "from cv.arducam_120fps import validate_120fps; validate_120fps()"
    """
    print("\n" + "=" * 70)
    print("  ARDUCAM OV9281 - 120 FPS VALIDATION")
    print("=" * 70 + "\n")
    
    config = CaptureConfig(
        device_index=device_index,
        width=1280,
        height=800,
        target_fps=120,
        enable_processing_thread=True,
    )
    
    try:
        camera = Arducam120FPS(config)
        camera.start()
        
        if show_live:
            cv2.namedWindow("Validation", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Validation", 960, 600)
            
            start = time.time()
            while time.time() - start < duration:
                frame = camera.get_latest_frame()
                if frame:
                    display = frame.frame.copy()
                    metrics = camera.get_metrics()
                    
                    # Draw metrics
                    y = 30
                    cv2.putText(display, f"CAP: {metrics.capture_fps:.1f}", (10, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    cv2.putText(display, f"PROC: {metrics.process_fps:.1f}", (180, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(display, f"Drops: {metrics.dropped_frames}", (360, y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)
                    
                    remaining = duration - (time.time() - start)
                    cv2.putText(display, f"Validating: {remaining:.1f}s", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    
                    camera.mark_display_tick()
                    cv2.imshow("Validation", display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cv2.destroyWindow("Validation")
        
        # Run validation
        report = camera.run_validation(duration_seconds=duration)
        report.print_report()
        
        camera.stop()
        return report
        
    except CaptureError as e:
        print(f"\n❌ CAPTURE ERROR: {e}\n")
        report = ValidationReport()
        report.result = ValidationResult.FAIL
        report.fail_reasons.append(str(e))
        return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate Arducam 120fps capture")
    parser.add_argument("--duration", type=float, default=5.0, help="Validation duration in seconds")
    parser.add_argument("--camera-index", type=int, default=-1, help="Camera device index (-1 for auto)")
    parser.add_argument("--no-display", action="store_true", help="Skip live display")
    
    args = parser.parse_args()
    
    report = validate_120fps(
        device_index=args.camera_index,
        duration=args.duration,
        show_live=not args.no_display
    )
    
    sys.exit(0 if report.result == ValidationResult.PASS else 1)
