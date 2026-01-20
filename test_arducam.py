#!/usr/bin/env python3
"""
Arducam OV9281 Camera Test Suite

Tests for:
1. Frame rate consistency and actual FPS
2. Frame interval jitter/stability  
3. Ball tracking accuracy and detection rate
4. System latency measurement

Run: python test_arducam.py
"""

import sys
import time
import argparse
import statistics
from typing import Optional, List, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

from cv.arducam_camera import ArducamCamera, ArducamConfig, CameraError
from cv.calibration import Calibration
from cv.auto_ball_tracker import AutoBallTracker, AutoTrackerConfig, TrackerState


@dataclass
class TestResults:
    """Results from camera/tracking tests."""
    # FPS metrics
    target_fps: float
    actual_fps: float
    fps_std: float
    
    # Frame timing
    avg_frame_time_ms: float
    frame_time_std_ms: float
    frame_jitter_ms: float
    
    # Capture metrics
    total_frames: int
    dropped_frames: int
    drop_rate_pct: float
    
    # Tracking metrics (if applicable)
    detection_rate_pct: float = 0.0
    position_jitter_px: float = 0.0
    tracking_fps: float = 0.0
    
    def print_report(self):
        """Print formatted test report."""
        print("\n" + "="*60)
        print("  ARDUCAM OV9281 TEST RESULTS")
        print("="*60)
        
        print("\n📊 FRAME RATE PERFORMANCE")
        print("-"*40)
        print(f"  Target FPS:       {self.target_fps:.0f}")
        print(f"  Actual FPS:       {self.actual_fps:.1f}")
        fps_pct = (self.actual_fps / self.target_fps) * 100 if self.target_fps > 0 else 0
        print(f"  FPS Achievement:  {fps_pct:.1f}%")
        print(f"  FPS Std Dev:      {self.fps_std:.2f}")
        
        print("\n⏱️ FRAME TIMING")
        print("-"*40)
        print(f"  Avg Frame Time:   {self.avg_frame_time_ms:.2f}ms")
        print(f"  Frame Time Std:   {self.frame_time_std_ms:.2f}ms")
        print(f"  Frame Jitter:     {self.frame_jitter_ms:.2f}ms")
        
        print("\n📦 CAPTURE RELIABILITY")
        print("-"*40)
        print(f"  Total Frames:     {self.total_frames}")
        print(f"  Dropped Frames:   {self.dropped_frames}")
        print(f"  Drop Rate:        {self.drop_rate_pct:.2f}%")
        
        if self.detection_rate_pct > 0:
            print("\n🎯 BALL TRACKING")
            print("-"*40)
            print(f"  Detection Rate:   {self.detection_rate_pct:.1f}%")
            print(f"  Position Jitter:  {self.position_jitter_px:.2f}px")
            print(f"  Tracking FPS:     {self.tracking_fps:.1f}")
        
        # Overall assessment
        print("\n✅ ASSESSMENT")
        print("-"*40)
        if fps_pct >= 90 and self.drop_rate_pct < 1:
            print("  EXCELLENT - Camera performing at full speed!")
        elif fps_pct >= 70 and self.drop_rate_pct < 5:
            print("  GOOD - Camera performing well")
        elif fps_pct >= 50:
            print("  FAIR - Consider reducing resolution")
        else:
            print("  POOR - Check USB connection (use USB 3.0)")
        
        print("="*60 + "\n")


def test_fps(
    camera: ArducamCamera,
    duration_seconds: float = 10.0,
    show_preview: bool = True
) -> TestResults:
    """
    Test camera FPS performance.
    
    Args:
        camera: Started ArducamCamera instance
        duration_seconds: How long to run the test
        show_preview: Whether to show live preview
        
    Returns:
        TestResults with FPS and timing metrics
    """
    print(f"\n[Test] Running FPS test for {duration_seconds}s...")
    
    if show_preview:
        cv2.namedWindow("FPS Test", cv2.WINDOW_NORMAL)
    
    frame_times: List[float] = []
    fps_samples: List[float] = []
    last_time = time.time()
    start_time = last_time
    frame_count = 0
    dropped = 0
    
    while (time.time() - start_time) < duration_seconds:
        frame_data = camera.get_frames()
        
        if frame_data is None:
            dropped += 1
            continue
        
        current_time = time.time()
        frame_delta = current_time - last_time
        last_time = current_time
        
        if frame_delta > 0:
            frame_times.append(frame_delta * 1000)  # Convert to ms
            fps_samples.append(1.0 / frame_delta)
        
        frame_count += 1
        
        if show_preview:
            display = frame_data.color_frame.copy()
            h, w = display.shape[:2]
            
            # FPS overlay
            current_fps = fps_samples[-1] if fps_samples else 0
            avg_fps = statistics.mean(fps_samples) if fps_samples else 0
            
            cv2.putText(display, f"Current FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Average FPS: {avg_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display, f"Frame: {frame_count}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            cv2.putText(display, f"Test: {remaining:.1f}s remaining", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow("FPS Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if show_preview:
        cv2.destroyWindow("FPS Test")
    
    # Calculate statistics
    actual_fps = statistics.mean(fps_samples) if fps_samples else 0
    fps_std = statistics.stdev(fps_samples) if len(fps_samples) > 1 else 0
    avg_frame_time = statistics.mean(frame_times) if frame_times else 0
    frame_time_std = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
    
    # Jitter = difference from expected frame time
    expected_frame_time = 1000.0 / camera.config.fps if camera.config.fps > 0 else 0
    jitter_samples = [abs(t - expected_frame_time) for t in frame_times]
    frame_jitter = statistics.mean(jitter_samples) if jitter_samples else 0
    
    total_attempts = frame_count + dropped
    drop_rate = (dropped / total_attempts * 100) if total_attempts > 0 else 0
    
    return TestResults(
        target_fps=camera.config.fps,
        actual_fps=actual_fps,
        fps_std=fps_std,
        avg_frame_time_ms=avg_frame_time,
        frame_time_std_ms=frame_time_std,
        frame_jitter_ms=frame_jitter,
        total_frames=frame_count,
        dropped_frames=dropped,
        drop_rate_pct=drop_rate
    )


def test_tracking(
    camera: ArducamCamera,
    duration_seconds: float = 30.0,
    show_preview: bool = True
) -> TestResults:
    """
    Test ball tracking accuracy and stability.
    
    Args:
        camera: Started ArducamCamera instance
        duration_seconds: How long to run the test
        show_preview: Whether to show live preview
        
    Returns:
        TestResults with tracking metrics
    """
    print(f"\n[Test] Running tracking test for {duration_seconds}s...")
    print("[Test] Place a golf ball in view and keep it stationary")
    print("[Test] Then slowly move it to test tracking\n")
    
    # Initialize tracker
    calibration = Calibration("config.json")
    tracker_config = AutoTrackerConfig(
        auto_detect_enabled=True,
        min_brightness=150,  # Adjust for OV9281 monochrome
    )
    tracker = AutoBallTracker(calibration, tracker_config)
    
    if show_preview:
        cv2.namedWindow("Tracking Test", cv2.WINDOW_NORMAL)
    
    # Stats tracking
    frame_times: List[float] = []
    fps_samples: List[float] = []
    positions: List[Tuple[float, float]] = []
    detection_count = 0
    frame_count = 0
    dropped = 0
    
    last_time = time.time()
    start_time = last_time
    
    while (time.time() - start_time) < duration_seconds:
        frame_data = camera.get_frames()
        
        if frame_data is None:
            dropped += 1
            continue
        
        current_time = time.time()
        frame_delta = current_time - last_time
        last_time = current_time
        
        if frame_delta > 0:
            frame_times.append(frame_delta * 1000)
            fps_samples.append(1.0 / frame_delta)
        
        frame_count += 1
        
        # Run tracking
        result = tracker.update(
            frame_data.color_frame,
            frame_data.gray_frame,
            frame_data.timestamp,
            frame_data.frame_number
        )
        
        if result and result.detected:
            detection_count += 1
            positions.append((result.pixel_x, result.pixel_y))
        
        if show_preview:
            display = frame_data.color_frame.copy()
            h, w = display.shape[:2]
            
            # Draw tracker state
            state_colors = {
                TrackerState.SEARCHING: (0, 255, 255),
                TrackerState.TRACKING: (0, 255, 0),
                TrackerState.SHOT_DETECTED: (0, 0, 255),
                TrackerState.WAITING: (255, 165, 0),
                TrackerState.LOST: (128, 128, 128),
            }
            color = state_colors.get(tracker.state, (255, 255, 255))
            cv2.putText(display, f"State: {tracker.state.value}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw ball position
            if result and result.detected:
                cx, cy = int(result.pixel_x), int(result.pixel_y)
                cv2.circle(display, (cx, cy), 20, (0, 255, 0), 2)
                cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)
                cv2.putText(display, f"({cx}, {cy})", (cx + 25, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Stats overlay
            current_fps = fps_samples[-1] if fps_samples else 0
            det_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
            
            cv2.putText(display, f"FPS: {current_fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Detection: {det_rate:.1f}%", (10, 85),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(display, f"Frame: {frame_count}", (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            elapsed = time.time() - start_time
            remaining = duration_seconds - elapsed
            cv2.putText(display, f"Test: {remaining:.1f}s remaining", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw recent trajectory
            if len(positions) > 1:
                for i in range(1, min(50, len(positions))):
                    p1 = (int(positions[-i][0]), int(positions[-i][1]))
                    p2 = (int(positions[-i-1][0]), int(positions[-i-1][1]))
                    alpha = 1.0 - (i / 50)
                    color = (0, int(255 * alpha), 0)
                    cv2.line(display, p1, p2, color, 1)
            
            cv2.imshow("Tracking Test", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    if show_preview:
        cv2.destroyWindow("Tracking Test")
    
    # Calculate FPS statistics
    actual_fps = statistics.mean(fps_samples) if fps_samples else 0
    fps_std = statistics.stdev(fps_samples) if len(fps_samples) > 1 else 0
    avg_frame_time = statistics.mean(frame_times) if frame_times else 0
    frame_time_std = statistics.stdev(frame_times) if len(frame_times) > 1 else 0
    
    expected_frame_time = 1000.0 / camera.config.fps if camera.config.fps > 0 else 0
    jitter_samples = [abs(t - expected_frame_time) for t in frame_times]
    frame_jitter = statistics.mean(jitter_samples) if jitter_samples else 0
    
    total_attempts = frame_count + dropped
    drop_rate = (dropped / total_attempts * 100) if total_attempts > 0 else 0
    
    # Calculate position jitter (when ball is stationary)
    position_jitter = 0.0
    if len(positions) > 10:
        # Look for stationary periods (low movement)
        x_coords = [p[0] for p in positions]
        y_coords = [p[1] for p in positions]
        
        # Use recent positions for jitter calculation
        recent = min(100, len(positions))
        x_recent = x_coords[-recent:]
        y_recent = y_coords[-recent:]
        
        x_std = statistics.stdev(x_recent) if len(x_recent) > 1 else 0
        y_std = statistics.stdev(y_recent) if len(y_recent) > 1 else 0
        position_jitter = (x_std + y_std) / 2
    
    detection_rate = (detection_count / frame_count * 100) if frame_count > 0 else 0
    
    return TestResults(
        target_fps=camera.config.fps,
        actual_fps=actual_fps,
        fps_std=fps_std,
        avg_frame_time_ms=avg_frame_time,
        frame_time_std_ms=frame_time_std,
        frame_jitter_ms=frame_jitter,
        total_frames=frame_count,
        dropped_frames=dropped,
        drop_rate_pct=drop_rate,
        detection_rate_pct=detection_rate,
        position_jitter_px=position_jitter,
        tracking_fps=actual_fps
    )


def diagnose_camera(device_index: int = 0):
    """
    Diagnose camera capabilities and USB bandwidth.
    
    Fixed issues:
    - Safe camera index enumeration (no "out of device bound" errors)
    - Proper MJPEG availability detection before bandwidth calculation
    - Explicit MJPEG negotiation test with frame reading
    """
    print("\n" + "="*60)
    print("  CAMERA DIAGNOSTICS")
    print("="*60 + "\n")
    
    # List all cameras with safe enumeration
    cameras = safe_list_cameras()
    print(f"📷 Found {len(cameras)} camera(s):\n")
    for cam in cameras:
        mono_str = " [MONO]" if cam.get('is_monochrome') else ""
        readable_str = "" if cam.get('is_readable', True) else " [NOT READABLE]"
        print(f"  [{cam['index']}] {cam['width']}x{cam['height']} @ {cam['fps']:.0f}fps ({cam.get('fourcc', '?')}){mono_str}{readable_str}")
    
    if not cameras:
        print("  No cameras found!")
        return
    
    # Validate device_index
    valid_indices = [c['index'] for c in cameras if c.get('is_readable', True)]
    if device_index not in valid_indices:
        if valid_indices:
            device_index = valid_indices[0]
            print(f"\n⚠️  Requested index not available, using camera {device_index}")
        else:
            print("\n❌ No readable cameras available!")
            return
    
    # List supported modes with MJPEG detection
    print(f"\n📋 Testing modes for camera {device_index}:\n")
    modes = test_camera_modes_safe(device_index)
    
    if not modes:
        print("  No modes found (camera may not be accessible)")
        return
    
    # Check if MJPEG is actually available
    mjpeg_available = any(m.get('mjpeg_confirmed') for m in modes)
    print(f"  MJPEG support: {'✓ Confirmed' if mjpeg_available else '✗ Not available / ignored by macOS'}")
    print()
    
    # Group by format
    by_format = {}
    for m in modes:
        fmt = m['format']
        if fmt not in by_format:
            by_format[fmt] = []
        by_format[fmt].append(m)
    
    for fmt, fmt_modes in by_format.items():
        print(f"  Format: {fmt}")
        for m in fmt_modes:
            # Calculate bandwidth based on ACTUAL format, not requested
            actual_fmt = m.get('actual_format', fmt)
            bw = m['width'] * m['height'] * m['fps']
            
            # Only apply MJPEG compression estimate if MJPEG is confirmed
            if 'MJPG' in actual_fmt.upper() or 'MJPE' in actual_fmt.upper():
                bw *= 0.15  # Conservative compression estimate
                usb2_ok = "✓" if (bw * 8 / 1_000_000) < 400 else "✗"
            else:
                # Raw format - check actual bandwidth
                bw *= 2  # 2 bytes per pixel for YUYV
                usb2_ok = "✓" if (bw * 8 / 1_000_000) < 400 else "✗"
            
            bw_mbps = (bw * 8) / 1_000_000
            frames_ok = "✓" if m.get('frames_ok') else "✗"
            
            print(f"    {m['width']:4d}x{m['height']:<4d} @ {m['fps']:6.1f}fps  ({bw_mbps:6.1f} Mbps) USB2:{usb2_ok} Frames:{frames_ok}")
        print()
    
    # USB bandwidth info
    print("💡 USB Bandwidth Reference:")
    print("  USB 2.0: ~480 Mbps theoretical, ~35-40 MB/s practical")
    print("  USB 3.0: ~5 Gbps theoretical, ~400 MB/s practical")
    print()
    if not mjpeg_available:
        print("  ⚠️  MJPEG not available on this camera/backend.")
        print("      macOS AVFoundation often ignores FOURCC settings.")
        print("      For 120 FPS at 1280x800 without MJPEG, you need USB 3.0.")
    else:
        print("  For 120 FPS on USB 2.0, MJPG compression is recommended.")
    print()
    
    # Run explicit MJPEG negotiation test
    print("🔬 MJPEG Negotiation Test (2 seconds per mode)...\n")
    run_mjpeg_negotiation_test(device_index)
    
    # Test actual FPS at different settings
    print("\n🔬 Testing actual achievable FPS...\n")
    
    test_configs = [
        (1280, 800, 120, True, "1280x800 MJPG"),
        (1280, 800, 60, False, "1280x800 YUYV"),
        (640, 480, 120, True, "640x480 MJPG"),
        (640, 400, 120, True, "640x400 MJPG"),
        (320, 240, 120, True, "320x240 MJPG"),
    ]
    
    for width, height, fps, mjpg, label in test_configs:
        config = ArducamConfig(
            width=width,
            height=height,
            fps=fps,
            device_index=device_index,
            use_mjpg=mjpg
        )
        
        try:
            cam = ArducamCamera(config)
            cam.start()
            
            # Quick FPS test (2 seconds)
            start = time.time()
            frames = 0
            while time.time() - start < 2.0:
                if cam.get_frames() is not None:
                    frames += 1
            
            actual_fps = frames / 2.0
            actual_res = f"{cam.config.width}x{cam.config.height}"
            
            status = "✓" if actual_fps >= fps * 0.9 else "○" if actual_fps >= fps * 0.7 else "✗"
            print(f"  {status} {label:20s} → {actual_res:10s} @ {actual_fps:5.1f} fps (target: {fps})")
            
            cam.stop()
        except Exception as e:
            error_msg = str(e)
            # Truncate long error messages
            if len(error_msg) > 50:
                error_msg = error_msg[:50] + "..."
            print(f"  ✗ {label:20s} → Error: {error_msg}")
    
    print("\n" + "="*60 + "\n")


def safe_list_cameras(max_index: int = 10) -> List[dict]:
    """
    Safely enumerate cameras without 'out of device bound' errors.
    
    Uses AVFoundation backend explicitly on macOS and handles
    camera open failures gracefully.
    """
    import platform
    cameras = []
    consecutive_failures = 0
    
    for i in range(max_index):
        try:
            # Use AVFoundation backend explicitly on macOS
            if platform.system() == 'Darwin':
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(i, cv2.CAP_V4L2)
            
            if not cap.isOpened():
                # Try default backend as fallback
                cap = cv2.VideoCapture(i)
            
            if not cap.isOpened():
                consecutive_failures += 1
                cap.release()
                # Stop after 3 consecutive failures (likely no more cameras)
                if consecutive_failures >= 3:
                    break
                continue
            
            consecutive_failures = 0
            
            # Get basic info
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
            fourcc_str = "".join([chr((fourcc_code >> 8 * j) & 0xFF) for j in range(4)]) if fourcc_code else "NONE"
            
            # Try to read a frame to check if camera is truly accessible
            ret, frame = cap.read()
            is_readable = ret and frame is not None
            
            # Detect monochrome (OV9281 characteristic)
            is_mono = False
            if is_readable and frame is not None:
                if len(frame.shape) == 2:
                    is_mono = True
                elif len(frame.shape) == 3:
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
                'likely_arducam': is_mono and width >= 640 and height >= 400,
            })
            
            cap.release()
            
        except Exception as e:
            # Catch "out of device bound" and other errors gracefully
            error_str = str(e).lower()
            if 'bound' in error_str or 'index' in error_str or 'device' in error_str:
                break
            consecutive_failures += 1
            if consecutive_failures >= 3:
                break
    
    return cameras


def test_camera_modes_safe(device_index: int) -> List[dict]:
    """
    Test camera modes with actual frame reading.
    
    Returns mode info including whether MJPEG was actually negotiated.
    """
    import platform
    modes = []
    
    test_resolutions = [
        (1280, 800),
        (1280, 720),
        (640, 480),
        (640, 400),
        (320, 240),
    ]
    
    test_formats = ['MJPG', 'YUYV']
    
    for fmt in test_formats:
        for w, h in test_resolutions:
            try:
                # Use AVFoundation on macOS
                if platform.system() == 'Darwin':
                    cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
                else:
                    cap = cv2.VideoCapture(device_index)
                
                if not cap.isOpened():
                    continue
                
                # Set format FIRST (critical for macOS)
                fourcc = cv2.VideoWriter_fourcc(*fmt)
                cap.set(cv2.CAP_PROP_FOURCC, fourcc)
                
                # Then resolution
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
                
                # Then FPS
                cap.set(cv2.CAP_PROP_FPS, 120)
                
                # Read back actual values
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                actual_fps = cap.get(cv2.CAP_PROP_FPS)
                actual_fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
                actual_fourcc = "".join([chr((actual_fourcc_code >> 8 * j) & 0xFF) for j in range(4)]) if actual_fourcc_code else "NONE"
                
                # Try to read frames
                frames_ok = 0
                for _ in range(10):
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        frames_ok += 1
                
                cap.release()
                
                # Check if MJPEG was actually applied
                mjpeg_confirmed = fmt == 'MJPG' and actual_fourcc.upper() in ('MJPG', 'MJPE', 'JPEG')
                
                mode = {
                    'width': actual_w,
                    'height': actual_h,
                    'fps': actual_fps,
                    'format': fmt,
                    'actual_format': actual_fourcc,
                    'requested': f"{w}x{h} {fmt}",
                    'frames_ok': frames_ok >= 5,
                    'mjpeg_confirmed': mjpeg_confirmed,
                }
                
                # Avoid duplicates
                mode_key = f"{actual_w}x{actual_h}_{fmt}"
                if not any(f"{m['width']}x{m['height']}_{m['format']}" == mode_key for m in modes):
                    modes.append(mode)
                    
            except Exception:
                continue
    
    return modes


def run_mjpeg_negotiation_test(device_index: int, duration: float = 2.0):
    """
    Explicit MJPEG negotiation test that reads frames for duration seconds.
    
    Reports whether MJPEG is actually being used (not just requested).
    """
    import platform
    
    test_configs = [
        (1280, 800, "1280x800"),
        (640, 480, "640x480"),
    ]
    
    for width, height, label in test_configs:
        try:
            # Use AVFoundation on macOS
            if platform.system() == 'Darwin':
                cap = cv2.VideoCapture(device_index, cv2.CAP_AVFOUNDATION)
            else:
                cap = cv2.VideoCapture(device_index)
            
            if not cap.isOpened():
                print(f"  {label} MJPG: ✗ Failed to open")
                continue
            
            # Set MJPG format FIRST
            fourcc_mjpg = cv2.VideoWriter_fourcc(*'MJPG')
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_mjpg)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, 120)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Read back actual FOURCC
            actual_fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
            actual_fourcc = "".join([chr((actual_fourcc_code >> 8 * j) & 0xFF) for j in range(4)]) if actual_fourcc_code else "NONE"
            
            is_mjpeg = actual_fourcc.upper() in ('MJPG', 'MJPE', 'JPEG')
            
            # Read frames for duration
            start = time.time()
            frames = 0
            warmup_failures = 0
            
            while time.time() - start < duration:
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames += 1
                else:
                    if frames == 0:
                        warmup_failures += 1
                        if warmup_failures > 30:
                            break
            
            cap.release()
            
            if frames == 0:
                print(f"  {label} MJPG: ✗ Failed to read frames during warmup")
            else:
                measured_fps = frames / duration
                mjpeg_status = "✓" if is_mjpeg else "✗ (ignored)"
                
                if measured_fps >= 90:
                    result = "✓"
                elif measured_fps >= 60:
                    result = "○"
                else:
                    result = "✗"
                
                print(f"  {label} MJPG: {result} {measured_fps:.1f} fps, FOURCC={actual_fourcc} {mjpeg_status}")
                
        except Exception as e:
            print(f"  {label} MJPG: ✗ Error: {str(e)[:40]}")


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description="Arducam OV9281 Camera Test Suite"
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run camera diagnostics (list modes, test bandwidth)"
    )
    parser.add_argument(
        "--fps-only",
        action="store_true",
        help="Run FPS test only (no tracking)"
    )
    parser.add_argument(
        "--tracking-only",
        action="store_true",
        help="Run tracking test only (no FPS benchmark)"
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds (default: 10)"
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable live preview"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Camera width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Camera height (default: 800)"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        help="Target FPS (default: 100)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Camera device index (-1 for auto-detect)"
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  ARDUCAM OV9281 TEST SUITE")
    print("  Global Shutter High-Speed Camera")
    print("="*60 + "\n")
    
    # Configure camera
    config = ArducamConfig(
        width=args.width,
        height=args.height,
        fps=args.fps,
        device_index=args.camera_index,
        auto_exposure=False,
        exposure=50
    )
    
    # Run diagnostics if requested
    if args.diagnose:
        diagnose_camera(args.camera_index if args.camera_index >= 0 else 0)
        return 0
    
    print(f"Configuration:")
    print(f"  Resolution: {config.width}x{config.height}")
    print(f"  Target FPS: {config.fps}")
    print(f"  Duration: {args.duration}s per test")
    
    try:
        camera = ArducamCamera(config)
        camera.start()
        
        show_preview = not args.no_preview
        
        # Run tests
        if args.tracking_only:
            results = test_tracking(camera, args.duration, show_preview)
            results.print_report()
        elif args.fps_only:
            results = test_fps(camera, args.duration, show_preview)
            results.print_report()
        else:
            # Run both tests
            print("\n" + "="*40)
            print("  PHASE 1: FPS TEST")
            print("="*40)
            fps_results = test_fps(camera, args.duration, show_preview)
            fps_results.print_report()
            
            print("\n" + "="*40)
            print("  PHASE 2: TRACKING TEST")
            print("="*40)
            track_results = test_tracking(camera, args.duration * 2, show_preview)
            track_results.print_report()
        
        camera.stop()
        cv2.destroyAllWindows()
        
        print("\n[Test] All tests completed successfully!")
        return 0
        
    except CameraError as e:
        print(f"\n[ERROR] Camera error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[Test] Interrupted by user")
        return 0
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
