#!/usr/bin/env python3
"""
Putting Launch Monitor MVP - Main Entry Point

Real-time golf ball tracking and 3D visualization system.
Supports ZED 2i (high-speed), Intel RealSense D455, or standard USB webcams.

Camera Priority:
    1. ZED 2i @ 100 FPS (lowest latency)
    2. Intel RealSense D455
    3. Standard USB webcam

Usage:
    python main.py              # Auto-detect best camera
    python main.py --zed        # Force ZED 2i mode
    python main.py --webcam     # Force webcam mode
    python main.py --realsense  # Force RealSense mode
    python main.py --demo       # Demo mode (no camera)

Keyboard Controls:
    r - Start ROI calibration (draw rectangle)
    c - Start target line calibration (click two points)
    s - Start scale calibration (click two points 1m apart)
    g - Calibrate ground plane depth
    e - Toggle auto/manual exposure
    +/- - Adjust exposure (manual mode)
    ESC - Cancel calibration
    q - Quit

Requirements:
    - ZED 2i OR USB Webcam OR Intel RealSense D455
    - Python 3.11+ (recommended)
    - See requirements.txt for dependencies
"""

import sys
import time
import signal
import argparse
import json
import webbrowser
from typing import Optional, Union
from pathlib import Path

import cv2
import numpy as np

# Import modules
from cv.calibration import Calibration, CalibrationMode
from cv.shot_detector import ShotDetector, ShotDetectorConfig, ShotEvent
from server.web_server import WebServer
from server.websocket_server import WebSocketServer

# Camera imports
from cv.webcam_camera import WebcamCamera, WebcamConfig
from cv.webcam_camera import CameraError as WebcamCameraError

# ZED camera import
from cv.zed_camera import ZedCamera, ZedConfig, ZedMode
from cv.zed_camera import CameraError as ZedCameraError

# Try to import RealSense (optional)
try:
    from cv.realsense_camera import RealsenseCamera, CameraError as RealSenseCameraError, CameraConfig, ExposureMode
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    RealsenseCamera = None
    RealSenseCameraError = Exception
    CameraConfig = None
    ExposureMode = None

# Ball tracker imports - use auto tracker for better UX
from cv.auto_ball_tracker import AutoBallTracker, AutoTrackerConfig, TrackerState


class CameraMode:
    """Camera mode enum."""
    ZED = "zed"
    WEBCAM = "webcam"
    REALSENSE = "realsense"
    DEMO = "demo"


class PuttingMonitor:
    """
    Main application class for the Putting Launch Monitor.
    
    Orchestrates camera capture, CV processing, and server communication.
    Optimized for low-latency operation with high-FPS cameras.
    """
    
    def __init__(
        self,
        config_path: str = "config.json",
        http_port: int = 8080,
        ws_port: int = 8765,
        open_browser: bool = True,
        camera_mode: str = CameraMode.ZED,
        camera_index: int = -1  # -1 = auto-detect
    ):
        """
        Initialize the putting monitor.
        
        Args:
            config_path: Path to calibration config file.
            http_port: Port for HTTP server.
            ws_port: Port for WebSocket server.
            open_browser: Whether to open browser automatically.
            camera_mode: Camera mode (zed, webcam, realsense, demo).
            camera_index: Camera device index (-1 for auto-detect).
        """
        self.config_path = config_path
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_browser = open_browser
        self.camera_mode = camera_mode
        self.camera_index = camera_index
        
        # Load config file
        self.app_config = self._load_config()
        
        # Components
        self.camera: Optional[Union[ZedCamera, WebcamCamera, 'RealsenseCamera']] = None
        self.calibration: Optional[Calibration] = None
        self.tracker: Optional[ColorBallTracker] = None
        self.shot_detector: Optional[ShotDetector] = None
        self.web_server: Optional[WebServer] = None
        self.ws_server: Optional[WebSocketServer] = None
        
        # State
        self._running = False
        self._frame_count = 0
        self._last_fps_time = time.time()
        self._fps = 0.0
        self._ground_calibration_frames = []
        
        # Window name
        self.window_name = "Putting Launch Monitor"
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"[Main] Loaded config from {config_path}")
                return config
            except Exception as e:
                print(f"[Main] Warning: Could not load config: {e}")
        
        # Return default config
        return {
            "camera": {
                "primary": "zed2i",
                "mode": "webcam",
                "fps": 100,
                "fallback_to_realsense": True
            },
            "shot_detection": {
                "measurement_window_ms": 100,
                "min_measurement_frames": 5
            }
        }
    
    def start(self):
        """Start all components."""
        print("\n" + "="*60)
        print("  PUTTING LAUNCH MONITOR")
        print("  High-Speed Ball Tracking System")
        print("="*60 + "\n")
        
        try:
            # Initialize calibration
            print("[Main] Loading calibration...")
            self.calibration = Calibration(self.config_path)
            
            # Initialize servers
            print("[Main] Starting servers...")
            self.web_server = WebServer(
                host="localhost",
                port=self.http_port,
                web_directory="web"
            )
            self.web_server.start()
            
            self.ws_server = WebSocketServer(
                host="localhost",
                port=self.ws_port
            )
            self.ws_server.start()
            
            # Wait for servers to start
            time.sleep(0.5)
            
            # Initialize camera based on mode
            if self.camera_mode != CameraMode.DEMO:
                self._init_camera()
            else:
                print("[Main] Running in DEMO MODE - press 't' to trigger test shots")
            
            # Initialize tracker and detector
            print("[Main] Initializing auto tracker and detector...")
            tracker_config = AutoTrackerConfig(
                auto_detect_enabled=True,
                wait_after_shot_ms=3000,  # Wait 3s for virtual ball
            )
            self.tracker = AutoBallTracker(self.calibration, tracker_config)
            
            # Configure shot detector from config file
            detector_config = ShotDetectorConfig()
            self.shot_detector = ShotDetector(
                self.calibration,
                on_shot_callback=self._on_shot_detected,
                config=detector_config
            )
            
            # Apply config file settings to shot detector
            if 'shot_detection' in self.app_config:
                self.shot_detector.update_config_from_dict(self.app_config['shot_detection'])
            
            # Open browser
            if self.open_browser:
                url = f"http://localhost:{self.http_port}"
                print(f"\n[Main] Opening browser: {url}")
                webbrowser.open(url)
            
            # Print instructions
            self._print_instructions()
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 720)
            cv2.setMouseCallback(self.window_name, self._mouse_callback)
            
            # Start main loop
            self._running = True
            self._main_loop()
            
        except KeyboardInterrupt:
            print("\n[Main] Interrupted by user")
        except Exception as e:
            print(f"\n[ERROR] Fatal error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def _init_camera(self):
        """Initialize camera based on mode with fallback support."""
        
        # Try ZED first if requested
        if self.camera_mode == CameraMode.ZED:
            if self._try_init_zed():
                return
            
            # Fallback to RealSense
            if self.app_config.get('camera', {}).get('fallback_to_realsense', True):
                print("[Main] ZED not available, trying RealSense...")
                if self._try_init_realsense():
                    return
            
            # Fallback to webcam
            print("[Main] Trying webcam fallback...")
            if self._try_init_webcam():
                return
            
            # All failed, go to demo mode
            print("[Main] No cameras available, switching to DEMO MODE")
            self.camera_mode = CameraMode.DEMO
            return
        
        # Try RealSense if requested
        if self.camera_mode == CameraMode.REALSENSE:
            if self._try_init_realsense():
                return
            
            print("[Main] RealSense not available, trying webcam...")
            if self._try_init_webcam():
                return
            
            print("[Main] No cameras available, switching to DEMO MODE")
            self.camera_mode = CameraMode.DEMO
            return
        
        # Try webcam if requested
        if self.camera_mode == CameraMode.WEBCAM:
            if self._try_init_webcam():
                return
            
            print("[Main] Webcam not available, switching to DEMO MODE")
            self.camera_mode = CameraMode.DEMO
            return
    
    def _try_init_zed(self) -> bool:
        """Try to initialize ZED camera."""
        try:
            print("[Main] Initializing ZED 2i camera (high-speed mode)...")
            
            cam_config = self.app_config.get('camera', {})
            
            config = ZedConfig(
                width=cam_config.get('width', 672),
                height=cam_config.get('height', 376),
                fps=cam_config.get('fps', 100),
                device_index=self.camera_index,
                mode=ZedMode.WEBCAM,  # Mac compatible mode
                use_left_eye=cam_config.get('use_left_eye', True),
                auto_exposure=cam_config.get('auto_exposure', True)
            )
            
            self.camera = ZedCamera(config)
            self.camera.start()
            
            self.camera_mode = CameraMode.ZED
            print(f"[Main] ZED 2i initialized at {config.fps} FPS target!")
            return True
            
        except ZedCameraError as e:
            print(f"[Main] ZED initialization failed: {e}")
            self.camera = None
            return False
    
    def _try_init_realsense(self) -> bool:
        """Try to initialize RealSense camera."""
        if not REALSENSE_AVAILABLE:
            print("[Main] RealSense SDK not available")
            return False
        
        try:
            print("[Main] Initializing RealSense camera...")
            self.camera = RealsenseCamera()
            self.camera.start()
            
            self.camera_mode = CameraMode.REALSENSE
            print("[Main] RealSense initialized!")
            return True
            
        except RealSenseCameraError as e:
            print(f"[Main] RealSense initialization failed: {e}")
            self.camera = None
            return False
    
    def _try_init_webcam(self) -> bool:
        """Try to initialize webcam."""
        try:
            print("[Main] Initializing webcam...")
            
            config = WebcamConfig(
                width=1280,
                height=720,
                fps=30,
                device_index=max(0, self.camera_index)  # Use 0 if auto-detect
            )
            
            self.camera = WebcamCamera(config)
            self.camera.start()
            
            self.camera_mode = CameraMode.WEBCAM
            print("[Main] Webcam initialized!")
            return True
            
        except WebcamCameraError as e:
            print(f"[Main] Webcam initialization failed: {e}")
            self.camera = None
            return False
    
    def stop(self):
        """Stop all components."""
        print("\n[Main] Shutting down...")
        self._running = False
        
        if self.camera:
            self.camera.stop()
        
        if self.ws_server:
            self.ws_server.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        cv2.destroyAllWindows()
        print("[Main] Goodbye!")
    
    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            start_time = time.time()
            
            # Get frame based on camera mode
            if self.camera_mode == CameraMode.ZED and self.camera:
                frame_data = self.camera.get_frames(timeout_ms=100)
                
                if frame_data is None:
                    continue
                
                color_frame = frame_data.color_frame
                gray_frame = frame_data.gray_frame
                timestamp = frame_data.timestamp
                frame_number = frame_data.frame_number
                
            elif self.camera_mode == CameraMode.WEBCAM and self.camera:
                frame_data = self.camera.get_frames(timeout_ms=100)
                
                if frame_data is None:
                    continue
                
                color_frame = frame_data.color_frame
                gray_frame = frame_data.gray_frame
                timestamp = frame_data.timestamp
                frame_number = frame_data.frame_number
                
            elif self.camera_mode == CameraMode.REALSENSE and self.camera:
                frame_data = self.camera.get_frames(timeout_ms=100)
                
                if frame_data is None:
                    continue
                
                # Convert IR frame to color for display
                ir_frame = frame_data.ir_frame
                gray_frame = ir_frame
                color_frame = cv2.cvtColor(ir_frame, cv2.COLOR_GRAY2BGR)
                timestamp = frame_data.timestamp
                frame_number = frame_data.frame_number
                
            else:
                # Demo mode - generate synthetic frame
                color_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                color_frame[:] = (30, 40, 30)  # Dark green background
                gray_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
                timestamp = time.time()
                frame_number = self._frame_count
            
            self._frame_count += 1
            
            # Process tracking (if not in demo mode)
            if self.camera_mode != CameraMode.DEMO and self.tracker:
                tracking_result = self.tracker.update(
                    color_frame,
                    gray_frame,
                    timestamp,
                    frame_number
                )
                
                # Update shot detector
                self.shot_detector.update(tracking_result)
            
            # Create display frame
            display_frame = self._create_display_frame(color_frame, gray_frame)
            
            # Show frame
            cv2.imshow(self.window_name, display_frame)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
            
            # Update FPS
            self._update_fps()
    
    def _create_display_frame(
        self, 
        color_frame: np.ndarray, 
        gray_frame: np.ndarray
    ) -> np.ndarray:
        """Create the display frame with overlays."""
        # Resize if ZED frame is small
        if color_frame.shape[0] < 400:
            # Scale up for better visibility
            scale = 2
            color_frame = cv2.resize(color_frame, (0, 0), fx=scale, fy=scale, 
                                     interpolation=cv2.INTER_LINEAR)
        
        # Use color frame for display
        display = color_frame.copy()
        
        # Draw calibration overlays
        display = self.calibration.draw_overlays(display)
        
        # Draw tracker debug info
        if self.tracker and self.tracker.debug_frame is not None and self.camera_mode != CameraMode.DEMO:
            # Overlay tracker debug in corner
            debug = self.tracker.debug_frame
            h, w = debug.shape[:2]
            scale = 0.25
            small = cv2.resize(debug, (int(w * scale), int(h * scale)))
            sh, sw = small.shape[:2]
            if sh > 0 and sw > 0 and display.shape[0] > sh + 10 and display.shape[1] > sw + 10:
                display[10:10+sh, display.shape[1]-sw-10:display.shape[1]-10] = small
        
        # Draw ball detection
        if self.tracker and self.tracker.has_detection and self.camera_mode != CameraMode.DEMO:
            traj = self.tracker.get_trajectory()
            if traj:
                last = traj[-1]
                if last.detected:
                    # Scale coordinates if frame was resized
                    scale = display.shape[0] / gray_frame.shape[0] if gray_frame.shape[0] > 0 else 1
                    cx, cy = int(last.pixel_x * scale), int(last.pixel_y * scale)
                    cv2.circle(display, (cx, cy), 15, (0, 255, 0), 2)
                    
                    # Show speed
                    speed_text = f"{last.speed:.2f} m/s"
                    cv2.putText(display, speed_text, (cx + 20, cy),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw status bar
        self._draw_status_bar(display)
        
        return display
    
    def _draw_status_bar(self, frame: np.ndarray):
        """Draw status bar at bottom of frame."""
        h, w = frame.shape[:2]
        bar_height = 50
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = h - 15
        
        # FPS and camera info
        camera_fps = 0
        if self.camera and hasattr(self.camera, 'fps'):
            camera_fps = self.camera.fps
        
        fps_text = f"FPS: {camera_fps:.1f}"
        cv2.putText(frame, fps_text, (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Camera mode with color coding
        mode_colors = {
            CameraMode.ZED: (0, 255, 255),      # Yellow - high speed
            CameraMode.REALSENSE: (255, 200, 0), # Cyan
            CameraMode.WEBCAM: (100, 255, 100),  # Green
            CameraMode.DEMO: (100, 100, 255),    # Red
        }
        mode_color = mode_colors.get(self.camera_mode, (255, 255, 255))
        
        mode_labels = {
            CameraMode.ZED: "ZED 100fps",
            CameraMode.REALSENSE: "RealSense",
            CameraMode.WEBCAM: "Webcam",
            CameraMode.DEMO: "DEMO",
        }
        mode_text = f"Mode: {mode_labels.get(self.camera_mode, self.camera_mode)}"
        cv2.putText(frame, mode_text, (120, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)
        
        # Calibration status
        cal_text = self.calibration.status_text
        cv2.putText(frame, cal_text, (320, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Tracking status (auto-tracker state)
        if self.tracker:
            state = self.tracker.state
            if state == TrackerState.TRACKING or state == TrackerState.SHOT_DETECTED:
                track_text = "TRACKING"
                track_color = (0, 255, 0)
            elif state == TrackerState.SEARCHING:
                track_text = "Auto-searching..."
                track_color = (0, 255, 255)
            elif state == TrackerState.WAITING:
                track_text = "Waiting for sim"
                track_color = (0, 165, 255)
            else:
                track_text = state.value.upper()
                track_color = (100, 100, 255)
            cv2.putText(frame, track_text, (550, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 1)
        
        # Shot detector status with latency info
        if self.shot_detector:
            shot_text = f"Shot: {self.shot_detector.status_text}"
            cv2.putText(frame, shot_text, (750, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        # WebSocket clients
        if self.ws_server:
            ws_text = f"Clients: {self.ws_server.client_count}"
            cv2.putText(frame, ws_text, (w - 120, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    def _handle_key(self, key: int) -> bool:
        """
        Handle keyboard input.
        
        Returns False if should quit.
        """
        if key == 255:  # No key pressed
            return True
        
        if key == ord('q') or key == 27:  # q or ESC to quit
            if self.calibration.mode != CalibrationMode.NONE:
                self.calibration.cancel_calibration()
            else:
                return False
        
        elif key == ord('r'):
            self.calibration.start_roi_calibration()
            # Reset tracker when starting calibration
            if self.tracker:
                self.tracker.reset()
        
        elif key == ord('c'):
            self.calibration.start_target_line_calibration()
        
        elif key == ord('s'):
            self.calibration.start_scale_calibration()
            # Reset tracker when scale changes
            if self.tracker:
                self.tracker.reset()
        
        elif key == ord('g'):
            print("[Main] Ground calibration: Set ground_depth manually in config.json")
            print("       (Webcams don't have depth sensing)")
        
        elif key == ord('e'):
            if self.camera and hasattr(self.camera, 'set_exposure'):
                # Toggle auto exposure
                if hasattr(self.camera, 'config') and hasattr(self.camera.config, 'auto_exposure'):
                    self.camera.set_exposure(auto=not self.camera.config.auto_exposure)
        
        elif key == ord('+') or key == ord('='):
            if self.camera and hasattr(self.camera, 'adjust_exposure'):
                self.camera.adjust_exposure(1)
        
        elif key == ord('-'):
            if self.camera and hasattr(self.camera, 'adjust_exposure'):
                self.camera.adjust_exposure(-1)
        
        elif key == ord('x'):
            if self.tracker:
                self.tracker.reset()
            if self.shot_detector:
                self.shot_detector.reset()
        
        elif key == ord('t'):
            # Test shot (works in demo mode too)
            self._trigger_test_shot()
        
        return True
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events for calibration and ball selection."""
        # First check if we're in calibration mode
        if self.calibration.mode != CalibrationMode.NONE:
            self.calibration.handle_mouse_event(event, x, y, flags)
            return
        
        # If not calibrating, clicking manually sets ball position (fallback)
        if event == cv2.EVENT_LBUTTONDOWN and self.tracker:
            # Adjust for frame scaling - ZED WVGA display is 2x scaled
            self.tracker.manual_set_ball(x // 2, y // 2)
            if self.shot_detector:
                self.shot_detector.reset()
    
    def _on_shot_detected(self, shot: ShotEvent):
        """Callback when shot is detected."""
        # Notify tracker that shot was detected
        if self.tracker:
            self.tracker.notify_shot_detected()
        
        # Broadcast to WebSocket clients
        if self.ws_server:
            self.ws_server.broadcast(shot.to_dict())
    
    def _trigger_test_shot(self):
        """Trigger a test shot for demonstration."""
        import random
        
        test_shot = ShotEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            detection_time_ms=80.0,  # Simulated low latency
            speed_mps=round(1.0 + random.random() * 2.0, 3),
            direction_deg=round((random.random() - 0.5) * 10, 2),
            confidence=round(0.8 + random.random() * 0.2, 2),
            start_position=(0.0, 0.0),
            frame_count=8,
            trajectory_length_m=0.1,
            camera_fps=100.0,
            latency_estimate_ms=90.0,
        )
        
        print("\n" + "="*50)
        print("TEST SHOT")
        print("="*50)
        print(f"  Speed: {test_shot.speed_mps:.2f} m/s")
        print(f"  Direction: {test_shot.direction_deg:+.1f}°")
        print(f"  Est. latency: {test_shot.latency_estimate_ms:.0f}ms")
        print("="*50 + "\n")
        
        if self.ws_server:
            self.ws_server.broadcast(test_shot.to_dict())
    
    def _update_fps(self):
        """Update FPS calculation."""
        current_time = time.time()
        elapsed = current_time - self._last_fps_time
        
        if elapsed >= 1.0:
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._last_fps_time = current_time
    
    def _print_instructions(self):
        """Print keyboard instructions."""
        print("\n" + "-"*60)
        print("HOW TO USE")
        print("-"*60)
        print("  1. Ball is AUTO-DETECTED (no click needed!)")
        print("  2. Green circle follows the ball automatically")
        print("  3. Putt the ball - shot is detected!")
        print("  4. System waits for virtual ball, then resumes")
        print("")
        print("KEYBOARD CONTROLS")
        print("-"*60)
        print("  CLICK - Manual ball select (if auto-detect fails)")
        print("  x     - Reset tracker")
        print("  t     - Trigger test shot")
        print("  r     - Calibrate ROI (optional)")
        print("  c     - Calibrate target line (optional)")  
        print("  s     - Calibrate scale (optional)")
        print("  e     - Toggle auto/manual exposure")
        print("  +/-   - Adjust exposure")
        print("  q     - Quit")
        print("-"*60)
        
        # Show latency expectations
        if self.camera_mode == CameraMode.ZED:
            print("\n📊 PERFORMANCE (ZED 2i @ 100 FPS)")
            print("   Expected latency: ~80-100ms (feels real-time)")
        elif self.camera_mode == CameraMode.WEBCAM:
            print("\n📊 PERFORMANCE (Webcam @ 30 FPS)")
            print("   Expected latency: ~150-200ms")
        elif self.camera_mode == CameraMode.REALSENSE:
            print("\n📊 PERFORMANCE (RealSense @ 30 FPS)")
            print("   Expected latency: ~150-200ms")
        print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Putting Launch Monitor - High-Speed Ball Tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to config file (default: config.json)"
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP server port (default: 8080)"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port (default: 8765)"
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically"
    )
    parser.add_argument(
        "--zed",
        action="store_true",
        help="Force ZED 2i mode (100 FPS, lowest latency)"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Force webcam mode (USB camera)"
    )
    parser.add_argument(
        "--realsense",
        action="store_true",
        help="Force RealSense mode"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode (no camera required)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Camera device index (-1 for auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Determine camera mode (priority: demo > zed > realsense > webcam > auto)
    if args.demo:
        camera_mode = CameraMode.DEMO
    elif args.zed:
        camera_mode = CameraMode.ZED
    elif args.realsense:
        camera_mode = CameraMode.REALSENSE
    elif args.webcam:
        camera_mode = CameraMode.WEBCAM
    else:
        # Default: try ZED first (highest FPS)
        camera_mode = CameraMode.ZED
    
    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        print("\n[Main] Received interrupt signal")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start monitor
    monitor = PuttingMonitor(
        config_path=args.config,
        http_port=args.http_port,
        ws_port=args.ws_port,
        open_browser=not args.no_browser,
        camera_mode=camera_mode,
        camera_index=args.camera_index
    )
    
    monitor.start()


if __name__ == "__main__":
    main()
