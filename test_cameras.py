#!/usr/bin/env python3
"""
Camera Test Utility

Tests all available cameras and displays their capabilities.
Helps identify which camera to use for optimal performance.
"""

import cv2
import time
import sys


def list_all_cameras():
    """List all available cameras with detailed info."""
    print("\n" + "="*60)
    print("  CAMERA DETECTION UTILITY")
    print("="*60 + "\n")
    
    cameras = []
    
    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()
            
            # Detect camera type
            aspect_ratio = width / height if height > 0 else 0
            
            if aspect_ratio > 3.0:
                cam_type = "ZED (stereo)"
                single_width = width // 2
            else:
                cam_type = "Standard"
                single_width = width
            
            cameras.append({
                'index': i,
                'width': width,
                'height': height,
                'single_width': single_width,
                'fps': fps,
                'backend': backend,
                'type': cam_type,
                'aspect': aspect_ratio,
            })
            
            cap.release()
        else:
            cap.release()
    
    if not cameras:
        print("❌ No cameras found!")
        return []
    
    print(f"Found {len(cameras)} camera(s):\n")
    
    for cam in cameras:
        print(f"  📷 Camera [{cam['index']}]")
        print(f"     Type: {cam['type']}")
        print(f"     Resolution: {cam['width']}x{cam['height']}")
        if cam['type'] == "ZED (stereo)":
            print(f"     Single eye: {cam['single_width']}x{cam['height']}")
        print(f"     Reported FPS: {cam['fps']}")
        print(f"     Backend: {cam['backend']}")
        print()
    
    return cameras


def test_camera_fps(camera_index: int, duration: int = 5, target_fps: int = 100):
    """
    Test actual FPS of a camera.
    
    Args:
        camera_index: Camera index to test
        duration: Test duration in seconds
        target_fps: Target FPS to request
    """
    print(f"\n📊 Testing Camera [{camera_index}] at {target_fps} FPS target...")
    print("-" * 40)
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_index}")
        return
    
    # Get current resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Check if ZED (stereo)
    is_zed = (width / height) > 3.0 if height > 0 else False
    
    if is_zed:
        # Try WVGA mode (1344x376 stereo = 672x376 per eye)
        print("Detected ZED camera, setting WVGA mode (100 FPS capable)...")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
        cap.set(cv2.CAP_PROP_FPS, 100)
    else:
        # Standard camera
        cap.set(cv2.CAP_PROP_FPS, target_fps)
    
    # Verify settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Requested FPS: {target_fps}, Reported: {actual_fps}")
    
    # Warm up
    print("Warming up...")
    for _ in range(30):
        cap.read()
    
    # Test actual FPS
    print(f"Measuring actual FPS over {duration} seconds...")
    
    frame_count = 0
    start_time = time.time()
    frame_times = []
    
    while (time.time() - start_time) < duration:
        ret, frame = cap.read()
        if ret:
            frame_count += 1
            frame_times.append(time.time())
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    actual_measured_fps = frame_count / elapsed
    
    # Calculate frame time statistics
    if len(frame_times) > 1:
        deltas = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
        avg_delta = sum(deltas) / len(deltas)
        min_delta = min(deltas)
        max_delta = max(deltas)
        
        avg_frame_time_ms = avg_delta * 1000
        min_frame_time_ms = min_delta * 1000
        max_frame_time_ms = max_delta * 1000
    else:
        avg_frame_time_ms = min_frame_time_ms = max_frame_time_ms = 0
    
    cap.release()
    
    print("\n📈 RESULTS:")
    print(f"   Frames captured: {frame_count}")
    print(f"   Duration: {elapsed:.2f}s")
    print(f"   Actual FPS: {actual_measured_fps:.1f}")
    print(f"   Frame time avg: {avg_frame_time_ms:.1f}ms")
    print(f"   Frame time min: {min_frame_time_ms:.1f}ms")
    print(f"   Frame time max: {max_frame_time_ms:.1f}ms")
    
    # Performance assessment
    print("\n🎯 ASSESSMENT:")
    if actual_measured_fps >= 90:
        print(f"   ✅ Excellent! {actual_measured_fps:.0f} FPS achieved")
        print(f"   Est. shot detection latency: ~{50 + (1000/actual_measured_fps)*5:.0f}ms")
    elif actual_measured_fps >= 60:
        print(f"   ✅ Good! {actual_measured_fps:.0f} FPS achieved")
        print(f"   Est. shot detection latency: ~{50 + (1000/actual_measured_fps)*5:.0f}ms")
    elif actual_measured_fps >= 30:
        print(f"   ⚠️  Moderate: {actual_measured_fps:.0f} FPS")
        print(f"   Est. shot detection latency: ~{50 + (1000/actual_measured_fps)*5:.0f}ms")
    else:
        print(f"   ❌ Low FPS: {actual_measured_fps:.0f}")
        print("   Consider different camera or settings")


def visual_test(camera_index: int):
    """
    Visual test with live feed.
    
    Args:
        camera_index: Camera index to test
    """
    print(f"\n🎬 Starting visual test for Camera [{camera_index}]...")
    print("Press 'q' to quit, '1-4' for resolution presets\n")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera {camera_index}")
        return
    
    # Check if ZED
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    is_zed = (width / height) > 3.0 if height > 0 else False
    
    if is_zed:
        print("ZED camera detected!")
        print("  l = Left eye")
        print("  r = Right eye (default)")
        print("  1 = WVGA 100fps (672x376)")
        print("  2 = 720p 60fps (1280x720)")
        print("  3 = 1080p 30fps (1920x1080)")
        print("  4 = 2K 15fps (2208x1242)")
        
        # Start with WVGA for highest FPS
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
        cap.set(cv2.CAP_PROP_FPS, 100)
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    
    frame_count = 0
    fps_start = time.time()
    fps_display = 0
    use_right_eye = True  # Default to right eye
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            continue
        
        frame_count += 1
        
        # Calculate FPS
        elapsed = time.time() - fps_start
        if elapsed >= 1.0:
            fps_display = frame_count / elapsed
            frame_count = 0
            fps_start = time.time()
        
        # For ZED, show selected eye
        if is_zed:
            half_width = frame.shape[1] // 2
            if use_right_eye:
                eye_frame = frame[:, half_width:]  # Right eye
                eye_label = "RIGHT"
            else:
                eye_frame = frame[:, :half_width]  # Left eye
                eye_label = "LEFT"
            display = eye_frame.copy()
        else:
            display = frame.copy()
            eye_label = ""
        
        # Add overlay
        h, w = display.shape[:2]
        cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, f"Resolution: {w}x{h}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, f"Frame time: {1000/fps_display:.1f}ms" if fps_display > 0 else "Frame time: --", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show which eye for ZED
        if is_zed:
            cv2.putText(display, f"Eye: {eye_label} (press 'l'/'r' to switch)", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("Camera Test", display)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('l') and is_zed:
            use_right_eye = False
            print("Switched to LEFT eye")
        elif key == ord('r') and is_zed:
            use_right_eye = True
            print("Switched to RIGHT eye")
        elif key == ord('1') and is_zed:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1344)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 376)
            cap.set(cv2.CAP_PROP_FPS, 100)
            print("Set to WVGA 100fps")
        elif key == ord('2') and is_zed:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 60)
            print("Set to 720p 60fps")
        elif key == ord('3') and is_zed:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            cap.set(cv2.CAP_PROP_FPS, 30)
            print("Set to 1080p 30fps")
        elif key == ord('4') and is_zed:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4416)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1242)
            cap.set(cv2.CAP_PROP_FPS, 15)
            print("Set to 2K 15fps")
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Camera Test Utility")
    parser.add_argument("--list", "-l", action="store_true", help="List all cameras")
    parser.add_argument("--test", "-t", type=int, help="Test FPS for camera index")
    parser.add_argument("--visual", "-v", type=int, help="Visual test for camera index")
    parser.add_argument("--duration", "-d", type=int, default=5, help="Test duration in seconds")
    
    args = parser.parse_args()
    
    if args.list or (not args.test and args.visual is None):
        cameras = list_all_cameras()
        
        if cameras:
            # Find best camera for golf tracking
            print("\n💡 RECOMMENDATION:")
            zed_cameras = [c for c in cameras if "ZED" in c['type']]
            if zed_cameras:
                print(f"   Use ZED camera at index {zed_cameras[0]['index']} for lowest latency")
                print(f"   Run: python main.py --zed")
            else:
                print(f"   No ZED camera found. Using standard webcam.")
                print(f"   Run: python main.py --webcam")
    
    if args.test is not None:
        test_camera_fps(args.test, duration=args.duration)
    
    if args.visual is not None:
        visual_test(args.visual)


if __name__ == "__main__":
    main()
