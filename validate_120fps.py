#!/usr/bin/env python3
"""
Arducam OV9281 - 120 FPS Validation Tool

This script verifies that the Arducam camera is truly delivering 120fps by:
1. Finding and configuring the camera for 1280x800 @ 120fps
2. Running a 5-second capture test with precise timing
3. Measuring FPS from frame arrival timestamps (not processing)
4. Detecting duplicate or non-increasing PTS values
5. Reporting PASS/FAIL with detailed diagnostics

Usage:
    python validate_120fps.py              # Run validation with live display
    python validate_120fps.py --headless   # Run without display (for CI/automation)
    python validate_120fps.py --duration 10  # Run for 10 seconds
    python validate_120fps.py --camera-index 2  # Use specific camera

Exit codes:
    0 = PASS (camera achieving >= 115 fps with minimal anomalies)
    1 = FAIL (camera not meeting requirements)
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import time
import statistics
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from cv.arducam_120fps import (
    Arducam120FPS, 
    CaptureConfig, 
    CaptureError, 
    ValidationResult,
    ValidationReport
)


def print_banner():
    """Print startup banner."""
    print("\n" + "=" * 70)
    print("   █████╗ ██████╗ ██████╗ ██╗   ██╗ ██████╗ █████╗ ███╗   ███╗")
    print("  ██╔══██╗██╔══██╗██╔══██╗██║   ██║██╔════╝██╔══██╗████╗ ████║")
    print("  ███████║██████╔╝██║  ██║██║   ██║██║     ███████║██╔████╔██║")
    print("  ██╔══██║██╔══██╗██║  ██║██║   ██║██║     ██╔══██║██║╚██╔╝██║")
    print("  ██║  ██║██║  ██║██████╔╝╚██████╔╝╚██████╗██║  ██║██║ ╚═╝ ██║")
    print("  ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝  ╚═════╝  ╚═════╝╚═╝  ╚═╝╚═╝     ╚═╝")
    print("            OV9281 - 120 FPS VALIDATION TOOL")
    print("=" * 70 + "\n")


def validate_with_display(
    camera: Arducam120FPS, 
    duration: float
) -> ValidationReport:
    """Run validation with live display showing metrics."""
    print(f"[Validate] Starting {duration}s test with live display...")
    print("[Validate] Display shows real-time metrics; validation uses capture thread data")
    print("[Validate] Press 'q' to abort early\n")
    
    cv2.namedWindow("120fps Validation", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("120fps Validation", 960, 600)
    
    # Pre-validation warmup
    print("[Validate] Warming up (0.5s)...")
    time.sleep(0.5)
    
    # Start camera's internal validation (runs in capture thread at 120fps)
    camera._validation_mode = True
    camera._validation_pts_list = []
    
    start_time = time.perf_counter()
    
    # Display loop (runs at ~60fps, but doesn't affect validation measurement)
    while time.perf_counter() - start_time < duration:
        frame = camera.get_latest_frame()
        
        if frame is None:
            time.sleep(0.001)
            continue
        
        # Update display
        display = frame.frame.copy()
        metrics = camera.get_metrics()
        h, w = display.shape[:2]
        
        # Draw semi-transparent overlay at top
        overlay = display.copy()
        cv2.rectangle(overlay, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # FPS metrics from threads (measured internally at full rate)
        elapsed = time.perf_counter() - start_time
        remaining = duration - elapsed
        
        # CAPTURE FPS (from capture thread - should be ~120)
        cap_color = (0, 255, 0) if metrics.capture_fps >= 115 else (0, 165, 255)
        cv2.putText(display, f"CAPTURE: {metrics.capture_fps:.1f} fps", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, cap_color, 2)
        
        # PROCESS FPS (from processing thread - should be ~120)
        proc_color = (0, 255, 0) if metrics.process_fps >= 115 else (0, 165, 255)
        cv2.putText(display, f"PROCESS: {metrics.process_fps:.1f} fps", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, proc_color, 2)
        
        # Drops
        drop_color = (0, 255, 0) if metrics.dropped_frames < 10 else (0, 0, 255)
        cv2.putText(display, f"Drops: {metrics.dropped_frames}", (350, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, drop_color, 1)
        
        # PTS anomalies
        anomaly_color = (0, 255, 0) if metrics.pts_anomalies <= 1 else (0, 0, 255)
        cv2.putText(display, f"PTS Anomalies: {metrics.pts_anomalies}", (350, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, anomaly_color, 1)
        
        # Progress bar
        progress = elapsed / duration
        bar_width = w - 40
        bar_x = 20
        bar_y = 85
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_width, bar_y + 10), (50, 50, 50), -1)
        cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 10), (0, 200, 0), -1)
        
        cv2.putText(display, f"Validating: {remaining:.1f}s remaining", (bar_x, bar_y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Resolution and format info
        stats = camera.get_stats()
        info_text = f"{stats['resolution']} | {stats['format']}"
        cv2.putText(display, info_text, (w - 300, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        camera.mark_display_tick()
        cv2.imshow("120fps Validation", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("\n[Validate] Aborted by user")
            break
    
    # Stop validation collection
    camera._validation_mode = False
    
    cv2.destroyWindow("120fps Validation")
    
    # Build report from capture thread's data (not display loop data)
    pts_data = camera._validation_pts_list
    
    report = ValidationReport()
    report.requested_resolution = f"{camera.config.width}x{camera.config.height}"
    report.actual_resolution = f"{camera._actual_width}x{camera._actual_height}"
    report.requested_fps = camera.config.target_fps
    
    if len(pts_data) < 2:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append("Too few frames captured")
        return report
    
    # Calculate from capture thread data
    actual_duration = pts_data[-1][0] - pts_data[0][0]
    report.duration_seconds = actual_duration
    report.total_frames_captured = len(pts_data)
    
    if actual_duration > 0:
        report.measured_capture_fps = (len(pts_data) - 1) / actual_duration
    
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
    report.dropped_frames = camera._total_dropped_frames
    
    # Determine pass/fail
    report.result = ValidationResult.PASS
    
    if report.measured_capture_fps < camera.config.min_acceptable_fps:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append(
            f"Capture FPS {report.measured_capture_fps:.1f} < {camera.config.min_acceptable_fps}"
        )
    
    if report.pts_anomalies > camera.config.max_pts_anomalies:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append(
            f"PTS anomalies {report.pts_anomalies} > {camera.config.max_pts_anomalies}"
        )
    
    # Print processing thread metrics too
    metrics = camera.get_metrics()
    print(f"\n[Validate] Thread metrics:")
    print(f"  Capture thread:    {metrics.capture_fps:.1f} fps")
    print(f"  Processing thread: {metrics.process_fps:.1f} fps")
    
    return report


def validate_headless(
    camera: Arducam120FPS, 
    duration: float
) -> ValidationReport:
    """Run validation without display (for CI/automation)."""
    print(f"[Validate] Starting {duration}s headless test...")
    print("[Validate] Using camera's internal capture-thread measurement...")
    
    # Use the camera's internal run_validation() which measures correctly
    # in the capture thread (not affected by display or processing loops)
    report = camera.run_validation(duration_seconds=duration, save_csv=True)
    
    # Also print processing thread metrics
    metrics = camera.get_metrics()
    print(f"\n[Validate] Final metrics from threads:")
    print(f"  Capture thread FPS:    {metrics.capture_fps:.1f}")
    print(f"  Processing thread FPS: {metrics.process_fps:.1f}")
    print(f"  Dropped frames:        {metrics.dropped_frames}")
    
    return report


def build_report_from_data(
    camera: Arducam120FPS,
    validation_data: list,
    anomaly_count: int,
    duration: float
) -> ValidationReport:
    """Build validation report from collected data."""
    report = ValidationReport()
    report.requested_resolution = f"{camera.config.width}x{camera.config.height}"
    report.actual_resolution = f"{camera._actual_width}x{camera._actual_height}"
    report.requested_fps = camera.config.target_fps
    
    if len(validation_data) < 2:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append("Too few frames captured")
        return report
    
    # Calculate metrics
    pts_data = validation_data
    actual_duration = pts_data[-1][0] - pts_data[0][0]
    report.duration_seconds = actual_duration
    report.total_frames_captured = len(pts_data)
    
    if actual_duration > 0:
        report.measured_capture_fps = (len(pts_data) - 1) / actual_duration
    
    # Calculate inter-frame deltas
    deltas_ms = []
    for i in range(1, len(pts_data)):
        delta = (pts_data[i][0] - pts_data[i-1][0]) * 1000
        deltas_ms.append(delta)
    
    if deltas_ms:
        report.delta_min_ms = min(deltas_ms)
        report.delta_avg_ms = statistics.mean(deltas_ms)
        report.delta_max_ms = max(deltas_ms)
        report.delta_std_ms = statistics.stdev(deltas_ms) if len(deltas_ms) > 1 else 0.0
    
    report.pts_anomalies = anomaly_count
    report.dropped_frames = camera._total_dropped_frames
    
    # Determine pass/fail
    report.result = ValidationResult.PASS
    
    if report.measured_capture_fps < camera.config.min_acceptable_fps:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append(
            f"Capture FPS {report.measured_capture_fps:.1f} < {camera.config.min_acceptable_fps}"
        )
    
    if report.pts_anomalies > camera.config.max_pts_anomalies:
        report.result = ValidationResult.FAIL
        report.fail_reasons.append(
            f"PTS anomalies {report.pts_anomalies} > {camera.config.max_pts_anomalies}"
        )
    
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate Arducam OV9281 120fps capture",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Validation duration in seconds (default: 5)"
    )
    parser.add_argument(
        "--camera-index", "-i",
        type=int,
        default=-1,
        help="Camera device index (-1 for auto-detect)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without display (for CI/automation)"
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save per-frame timing data to CSV"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Target width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=800,
        help="Target height (default: 800)"
    )
    parser.add_argument(
        "--target-fps",
        type=int,
        default=120,
        help="Target FPS (default: 120)"
    )
    parser.add_argument(
        "--min-fps",
        type=float,
        default=115.0,
        help="Minimum acceptable FPS for PASS (default: 115)"
    )
    
    args = parser.parse_args()
    
    print_banner()
    
    # Configuration
    config = CaptureConfig(
        device_index=args.camera_index,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        min_acceptable_fps=args.min_fps,
    )
    
    print(f"Configuration:")
    print(f"  Target: {config.width}x{config.height} @ {config.target_fps}fps")
    print(f"  Pass threshold: >= {config.min_acceptable_fps} fps")
    print(f"  Duration: {args.duration} seconds")
    print(f"  Mode: {'Headless' if args.headless else 'Display'}")
    print("")
    
    camera = None
    report = None
    
    try:
        # Initialize camera
        print("[Validate] Initializing camera...")
        camera = Arducam120FPS(config)
        camera.start()
        
        print(f"[Validate] Camera initialized: {camera._actual_width}x{camera._actual_height}")
        print(f"[Validate] Format: {camera._actual_fourcc}")
        print("")
        
        # Run validation
        if args.headless:
            report = validate_headless(camera, args.duration)
        else:
            report = validate_with_display(camera, args.duration)
        
        # Save CSV if requested
        if args.save_csv and report:
            # The camera module saves CSV during run_validation()
            # For our custom validation, we'd need to save separately
            pass
        
        # Print report
        report.print_report()
        
        # Final summary
        if report.result == ValidationResult.PASS:
            print("✅ VALIDATION PASSED - Camera is delivering 120fps!")
            return 0
        else:
            print("❌ VALIDATION FAILED - See report above for details")
            return 1
        
    except CaptureError as e:
        print(f"\n❌ CAPTURE ERROR: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Arducam OV9281 is connected")
        print("  2. Check USB connection (USB 3.0 recommended)")
        print("  3. Try unplugging and reconnecting camera")
        print("  4. Check if camera is being used by another app")
        return 1
        
    except KeyboardInterrupt:
        print("\n[Validate] Interrupted by user")
        return 1
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        if camera:
            camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
