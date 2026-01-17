#!/usr/bin/env python3
"""
Dual Camera Comparison Tool

Shows ZED 2i and RealSense side-by-side to find optimal camera height.

Usage:
    python test_dual_cameras.py
    python test_dual_cameras.py --zed 2 --realsense 0

Controls:
    l/r - Toggle ZED left/right eye
    1-4 - ZED resolution presets
    +/- - Adjust display scale
    s   - Save screenshot
    q   - Quit
"""

import cv2
import time
import argparse
import numpy as np
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Dual Camera Comparison")
    parser.add_argument("--zed", type=int, default=2, help="ZED camera index")
    parser.add_argument("--realsense", type=int, default=0, help="RealSense camera index")
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  DUAL CAMERA COMPARISON TOOL")
    print("  Find optimal camera height for your setup")
    print("="*60 + "\n")
    
    # Open ZED camera
    print(f"Opening ZED 2i at index {args.zed}...")
    zed_cap = cv2.VideoCapture(args.zed)
    
    if not zed_cap.isOpened():
        print(f"ERROR: Could not open ZED at index {args.zed}")
        return
    
    # Set ZED to WVGA 100fps
    zed_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)  # Stereo width
    zed_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
    zed_cap.set(cv2.CAP_PROP_FPS, 100)
    
    # Open RealSense camera
    print(f"Opening RealSense at index {args.realsense}...")
    rs_cap = cv2.VideoCapture(args.realsense)
    
    if not rs_cap.isOpened():
        print(f"ERROR: Could not open RealSense at index {args.realsense}")
        zed_cap.release()
        return
    
    # Set RealSense resolution
    rs_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    rs_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    rs_cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual resolutions
    zed_w = int(zed_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    zed_h = int(zed_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rs_w = int(rs_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rs_h = int(rs_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"\nZED: {zed_w}x{zed_h} (stereo)")
    print(f"RealSense: {rs_w}x{rs_h}")
    
    # Warmup
    print("\nWarming up cameras...")
    for _ in range(30):
        zed_cap.read()
        rs_cap.read()
    
    print("\nControls:")
    print("  l/r  - Toggle ZED left/right eye")
    print("  1-4  - ZED resolution (1=WVGA, 2=720p, 3=1080p, 4=2K)")
    print("  +/-  - Adjust display scale")
    print("  s    - Save screenshot")
    print("  g    - Toggle grid overlay")
    print("  q    - Quit")
    print("-"*60 + "\n")
    
    # Create window
    cv2.namedWindow("Dual Camera Comparison", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dual Camera Comparison", 1600, 600)
    
    # State
    use_right_eye = True
    display_scale = 1.0
    show_grid = True
    
    # FPS tracking
    zed_fps = 0
    rs_fps = 0
    zed_frame_count = 0
    rs_frame_count = 0
    fps_start = time.time()
    
    # Screenshot directory
    screenshot_dir = Path("screenshots")
    screenshot_dir.mkdir(exist_ok=True)
    
    while True:
        # Read frames
        zed_ret, zed_frame = zed_cap.read()
        rs_ret, rs_frame = rs_cap.read()
        
        if not zed_ret or zed_frame is None:
            zed_frame = np.zeros((376, 672, 3), dtype=np.uint8)
            cv2.putText(zed_frame, "ZED: No Signal", (200, 188),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            zed_frame_count += 1
            # Extract selected eye
            half_w = zed_frame.shape[1] // 2
            if use_right_eye:
                zed_frame = zed_frame[:, half_w:].copy()
            else:
                zed_frame = zed_frame[:, :half_w].copy()
        
        if not rs_ret or rs_frame is None:
            rs_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            cv2.putText(rs_frame, "RealSense: No Signal", (400, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            rs_frame_count += 1
        
        # Update FPS every second
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            zed_fps = zed_frame_count / elapsed
            rs_fps = rs_frame_count / elapsed
            zed_frame_count = 0
            rs_frame_count = 0
            fps_start = time.time()
        
        # Resize frames to same height for comparison
        target_height = 500
        
        # Resize ZED
        zed_scale = target_height / zed_frame.shape[0]
        zed_display = cv2.resize(zed_frame, (0, 0), fx=zed_scale, fy=zed_scale)
        
        # Resize RealSense
        rs_scale = target_height / rs_frame.shape[0]
        rs_display = cv2.resize(rs_frame, (0, 0), fx=rs_scale, fy=rs_scale)
        
        # Draw grid overlay (10cm markers assuming ~350 px/m)
        if show_grid:
            # ZED grid
            for i in range(0, zed_display.shape[1], 35):  # ~10cm intervals
                cv2.line(zed_display, (i, 0), (i, zed_display.shape[0]), (50, 50, 50), 1)
            for i in range(0, zed_display.shape[0], 35):
                cv2.line(zed_display, (0, i), (zed_display.shape[1], i), (50, 50, 50), 1)
            
            # RealSense grid
            for i in range(0, rs_display.shape[1], 35):
                cv2.line(rs_display, (i, 0), (i, rs_display.shape[0]), (50, 50, 50), 1)
            for i in range(0, rs_display.shape[0], 35):
                cv2.line(rs_display, (0, i), (rs_display.shape[1], i), (50, 50, 50), 1)
        
        # Add labels and info
        # ZED info panel
        cv2.rectangle(zed_display, (0, 0), (300, 100), (0, 0, 0), -1)
        eye_str = "RIGHT" if use_right_eye else "LEFT"
        cv2.putText(zed_display, f"ZED 2i ({eye_str} eye)", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(zed_display, f"FPS: {zed_fps:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(zed_display, f"Res: {zed_frame.shape[1]}x{zed_frame.shape[0]}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # RealSense info panel
        cv2.rectangle(rs_display, (0, 0), (300, 100), (0, 0, 0), -1)
        cv2.putText(rs_display, "Intel RealSense D455", (10, 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(rs_display, f"FPS: {rs_fps:.1f}", (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        cv2.putText(rs_display, f"Res: {rs_frame.shape[1]}x{rs_frame.shape[0]}", (10, 75),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Add center crosshairs
        zed_cx, zed_cy = zed_display.shape[1] // 2, zed_display.shape[0] // 2
        cv2.line(zed_display, (zed_cx - 30, zed_cy), (zed_cx + 30, zed_cy), (0, 255, 0), 1)
        cv2.line(zed_display, (zed_cx, zed_cy - 30), (zed_cx, zed_cy + 30), (0, 255, 0), 1)
        
        rs_cx, rs_cy = rs_display.shape[1] // 2, rs_display.shape[0] // 2
        cv2.line(rs_display, (rs_cx - 30, rs_cy), (rs_cx + 30, rs_cy), (0, 255, 0), 1)
        cv2.line(rs_display, (rs_cx, rs_cy - 30), (rs_cx, rs_cy + 30), (0, 255, 0), 1)
        
        # Combine side by side
        # Add separator
        separator = np.zeros((target_height, 10, 3), dtype=np.uint8)
        separator[:, :, :] = (100, 100, 100)
        
        combined = np.hstack([zed_display, separator, rs_display])
        
        # Add instruction bar at bottom
        bar_height = 40
        instruction_bar = np.zeros((bar_height, combined.shape[1], 3), dtype=np.uint8)
        cv2.putText(instruction_bar, "l/r: Eye | 1-4: ZED Res | g: Grid | s: Screenshot | q: Quit", 
                   (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        grid_status = "Grid: ON" if show_grid else "Grid: OFF"
        cv2.putText(instruction_bar, grid_status, (combined.shape[1] - 120, 28),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if show_grid else (100, 100, 100), 1)
        
        combined = np.vstack([combined, instruction_bar])
        
        # Apply display scale
        if display_scale != 1.0:
            combined = cv2.resize(combined, (0, 0), fx=display_scale, fy=display_scale)
        
        cv2.imshow("Dual Camera Comparison", combined)
        
        # Handle keyboard
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('l'):
            use_right_eye = False
            print("ZED: LEFT eye")
        
        elif key == ord('r'):
            use_right_eye = True
            print("ZED: RIGHT eye")
        
        elif key == ord('g'):
            show_grid = not show_grid
            print(f"Grid: {'ON' if show_grid else 'OFF'}")
        
        elif key == ord('s'):
            # Save screenshot
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = screenshot_dir / f"dual_cam_{timestamp}.png"
            cv2.imwrite(str(filename), combined)
            print(f"Screenshot saved: {filename}")
        
        elif key == ord('+') or key == ord('='):
            display_scale = min(2.0, display_scale + 0.1)
            print(f"Scale: {display_scale:.1f}")
        
        elif key == ord('-'):
            display_scale = max(0.5, display_scale - 0.1)
            print(f"Scale: {display_scale:.1f}")
        
        elif key == ord('1'):
            zed_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
            zed_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
            zed_cap.set(cv2.CAP_PROP_FPS, 100)
            print("ZED: WVGA 100fps")
        
        elif key == ord('2'):
            zed_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            zed_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            zed_cap.set(cv2.CAP_PROP_FPS, 60)
            print("ZED: 720p 60fps")
        
        elif key == ord('3'):
            zed_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            zed_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            zed_cap.set(cv2.CAP_PROP_FPS, 30)
            print("ZED: 1080p 30fps")
        
        elif key == ord('4'):
            zed_cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
            zed_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)
            zed_cap.set(cv2.CAP_PROP_FPS, 15)
            print("ZED: 2K 15fps")
    
    # Cleanup
    zed_cap.release()
    rs_cap.release()
    cv2.destroyAllWindows()
    print("\nDone!")


if __name__ == "__main__":
    main()
