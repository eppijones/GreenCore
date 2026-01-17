#!/usr/bin/env python3
"""
Putting Launch Monitor - Multi-Camera Edition

Real-time golf ball tracking using multiple cameras:
- ZED 2i: Primary high-speed tracking (87+ FPS)
- RealSense D455: Depth calibration
- iPhone: Replay recording

Usage:
    python main_multicam.py              # Auto-detect and assign all cameras
    python main_multicam.py --single     # Use only primary tracker (ZED)
    python main_multicam.py --demo       # Demo mode (no cameras)

Keyboard Controls:
    r - Start ROI calibration
    d - Start depth calibration (uses RealSense)
    c - Start target line calibration
    s - Start scale calibration
    e - Toggle auto/manual exposure
    +/- - Adjust exposure
    v - Toggle validation overlay
    p - Toggle replay recording
    t - Trigger test shot
    x - Reset tracker
    q - Quit
"""

import sys
import time
import signal
import argparse
import json
import webbrowser
from typing import Optional
from pathlib import Path

import cv2
import numpy as np

# Import modules
from cv.calibration import Calibration, CalibrationMode
from cv.shot_detector import ShotDetector, ShotDetectorConfig, ShotEvent
from cv.color_ball_tracker import ColorBallTracker, ColorTrackerConfig
from cv.camera_manager import (
    MultiCameraManager, 
    CameraRole, 
    MultiFrameData,
    CameraInfo
)
from server.web_server import WebServer
from server.websocket_server import WebSocketServer


class MultiCamPuttingMonitor:
    """
    Multi-camera putting monitor.
    
    Uses multiple cameras for:
    - High-speed tracking (ZED 2i @ 87+ FPS)
    - Depth calibration (RealSense D455)
    - Replay recording (iPhone)
    - Position validation (cross-camera verification)
    """
    
    def __init__(
        self,
        config_path: str = "config.json",
        http_port: int = 8080,
        ws_port: int = 8765,
        open_browser: bool = True,
        single_camera: bool = False,
        demo_mode: bool = False
    ):
        self.config_path = config_path
        self.http_port = http_port
        self.ws_port = ws_port
        self.open_browser = open_browser
        self.single_camera = single_camera
        self.demo_mode = demo_mode
        
        # Load config
        self.app_config = self._load_config()
        
        # Components
        self.camera_manager: Optional[MultiCameraManager] = None
        self.calibration: Optional[Calibration] = None
        self.tracker: Optional[ColorBallTracker] = None
        self.shot_detector: Optional[ShotDetector] = None
        self.web_server: Optional[WebServer] = None
        self.ws_server: Optional[WebSocketServer] = None
        
        # State
        self._running = False
        self._frame_count = 0
        self._show_validation = True
        self._replay_enabled = True
        
        # Window
        self.window_name = "Putting Monitor - Multi-Camera"
    
    def _load_config(self) -> dict:
        """Load configuration from JSON file."""
        config_path = Path(self.config_path)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"[Main] Warning: Could not load config: {e}")
        return {}
    
    def _save_config(self):
        """Save configuration to JSON file."""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.app_config, f, indent=2)
            print(f"[Main] Config saved to {self.config_path}")
        except Exception as e:
            print(f"[Main] Warning: Could not save config: {e}")
    
    def start(self):
        """Start the multi-camera monitor."""
        print("\n" + "="*60)
        print("  PUTTING LAUNCH MONITOR - MULTI-CAMERA EDITION")
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
            time.sleep(0.5)
            
            # Initialize cameras
            if not self.demo_mode:
                self._init_cameras()
            else:
                print("[Main] Running in DEMO MODE")
            
            # Initialize tracker and shot detector
            self._init_tracking()
            
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
            
            # Main loop
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
    
    def _init_cameras(self):
        """Initialize multi-camera system."""
        print("[Main] Initializing multi-camera system...")
        
        self.camera_manager = MultiCameraManager(self.app_config)
        cameras = self.camera_manager.detect_cameras()
        
        if not cameras:
            print("[Main] No cameras found! Switching to demo mode.")
            self.demo_mode = True
            return
        
        if self.single_camera:
            # Only use primary tracker
            assignments = self.camera_manager.auto_assign_roles()
            # Filter to only primary tracker
            assignments = {
                k: v for k, v in assignments.items() 
                if k == CameraRole.PRIMARY_TRACKER
            }
        else:
            # Use all available cameras
            assignments = self.camera_manager.auto_assign_roles()
        
        if not self.camera_manager.start(assignments):
            print("[Main] Failed to start cameras! Switching to demo mode.")
            self.demo_mode = True
            return
        
        # Set up calibration callback
        self.camera_manager.set_calibration_callback(self._on_calibration_complete)
        
        print(f"\n[Main] Active cameras: {', '.join(self.camera_manager.active_cameras)}")
    
    def _init_tracking(self):
        """Initialize tracker and shot detector."""
        print("[Main] Initializing tracker and detector...")
        
        tracker_config = ColorTrackerConfig()
        self.tracker = ColorBallTracker(self.calibration, tracker_config)
        
        detector_config = ShotDetectorConfig()
        self.shot_detector = ShotDetector(
            self.calibration,
            on_shot_callback=self._on_shot_detected,
            config=detector_config
        )
        
        # Apply config settings
        if 'shot_detection' in self.app_config:
            self.shot_detector.update_config_from_dict(self.app_config['shot_detection'])
    
    def stop(self):
        """Stop all components."""
        print("\n[Main] Shutting down...")
        self._running = False
        
        if self.camera_manager:
            self.camera_manager.stop()
        
        if self.ws_server:
            self.ws_server.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        cv2.destroyAllWindows()
        print("[Main] Goodbye!")
    
    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            if self.demo_mode:
                # Demo mode - synthetic frames
                frame_data = self._generate_demo_frame()
            else:
                # Get frames from all cameras
                frame_data = self.camera_manager.get_frames()
                if frame_data is None or frame_data.tracker_frame is None:
                    time.sleep(0.001)
                    continue
            
            self._frame_count += 1
            
            # Process tracking on primary tracker frame
            if not self.demo_mode and self.tracker:
                tracking_result = self.tracker.update(
                    frame_data.tracker_frame,
                    frame_data.tracker_gray,
                    frame_data.timestamp,
                    frame_data.frame_number
                )
                
                # Add to position validator if we have other cameras
                if tracking_result and tracking_result.detected:
                    self.camera_manager.position_validator.add_position(
                        "primary",
                        tracking_result.pixel_x,
                        tracking_result.pixel_y,
                        frame_data.timestamp
                    )
                
                # Update shot detector
                self.shot_detector.update(tracking_result)
            
            # Create display frame
            display = self._create_display(frame_data)
            
            # Show frame
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
    
    def _generate_demo_frame(self) -> MultiFrameData:
        """Generate synthetic frame for demo mode."""
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[:] = (30, 40, 30)
        
        # Add demo text
        cv2.putText(frame, "DEMO MODE", (500, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (100, 100, 100), 2)
        cv2.putText(frame, "Press 't' for test shot", (480, 400),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
        
        return MultiFrameData(
            timestamp=time.time(),
            tracker_frame=frame,
            tracker_gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
            frame_number=self._frame_count
        )
    
    def _create_display(self, frame_data: MultiFrameData) -> np.ndarray:
        """Create display frame with all overlays."""
        # Start with tracker frame
        if frame_data.tracker_frame is None:
            return np.zeros((720, 1280, 3), dtype=np.uint8)
        
        frame = frame_data.tracker_frame.copy()
        
        # Scale up if small (ZED WVGA is 672x376)
        if frame.shape[0] < 500:
            frame = cv2.resize(frame, (0, 0), fx=2, fy=2, 
                              interpolation=cv2.INTER_LINEAR)
        
        h, w = frame.shape[:2]
        
        # Draw calibration overlays
        frame = self.calibration.draw_overlays(frame)
        
        # Draw tracker debug
        if self.tracker and self.tracker.debug_frame is not None:
            debug = self.tracker.debug_frame
            dh, dw = debug.shape[:2]
            scale = 0.3
            small = cv2.resize(debug, (int(dw * scale), int(dh * scale)))
            sh, sw = small.shape[:2]
            if h > sh + 10 and w > sw + 10:
                frame[10:10+sh, w-sw-10:w-10] = small
        
        # Draw ball detection
        if self.tracker and self.tracker.has_detection:
            traj = self.tracker.get_trajectory()
            if traj and traj[-1].detected:
                last = traj[-1]
                scale = frame.shape[0] / frame_data.tracker_frame.shape[0]
                cx, cy = int(last.pixel_x * scale), int(last.pixel_y * scale)
                cv2.circle(frame, (cx, cy), 15, (0, 255, 0), 2)
                
                # Speed
                cv2.putText(frame, f"{last.speed:.2f} m/s", (cx + 20, cy),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw validation status (if enabled and multi-camera)
        if self._show_validation and self.camera_manager:
            valid, conf, pos = self.camera_manager.position_validator.validate()
            
            if len(self.camera_manager.active_cameras) > 1:
                val_text = f"Validation: {conf:.0%}"
                color = (0, 255, 0) if conf > 0.8 else (0, 255, 255) if conf > 0.5 else (0, 0, 255)
                cv2.putText(frame, val_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Draw multi-camera info panel
        self._draw_camera_panel(frame, frame_data)
        
        # Draw status bar
        self._draw_status_bar(frame, frame_data)
        
        # Draw replay indicator
        if self._replay_enabled and CameraRole.REPLAY_RECORDER in (
            self.camera_manager._cameras if self.camera_manager else {}
        ):
            cv2.circle(frame, (w - 30, 30), 10, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 70, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        return frame
    
    def _draw_camera_panel(self, frame: np.ndarray, frame_data: MultiFrameData):
        """Draw camera status panel."""
        if self.demo_mode or not self.camera_manager:
            return
        
        h, w = frame.shape[:2]
        panel_x = 10
        panel_y = 60
        line_height = 25
        
        active = self.camera_manager.active_cameras
        
        cv2.putText(frame, "CAMERAS:", (panel_x, panel_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        for i, role in enumerate(active):
            y = panel_y + (i + 1) * line_height
            
            # Get FPS for this camera
            fps = 0
            if role == "primary_tracker":
                fps = frame_data.tracker_fps
                color = (0, 255, 255)  # Yellow for primary
                label = f"ZED: {fps:.0f}fps"
            elif role == "depth_calibration":
                color = (255, 200, 0)  # Cyan for depth
                label = "RealSense: OK"
            elif role == "replay_recorder":
                color = (100, 100, 255)  # Red for replay
                label = "iPhone: Recording"
            else:
                color = (150, 150, 150)
                label = f"{role}"
            
            cv2.putText(frame, f"• {label}", (panel_x, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    
    def _draw_status_bar(self, frame: np.ndarray, frame_data: MultiFrameData):
        """Draw status bar at bottom."""
        h, w = frame.shape[:2]
        bar_height = 50
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - bar_height), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y = h - 15
        
        # Primary FPS
        fps = frame_data.tracker_fps if not self.demo_mode else 0
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Mode
        if self.demo_mode:
            mode_text = "DEMO"
            mode_color = (100, 100, 255)
        elif self.single_camera:
            mode_text = "SINGLE-CAM"
            mode_color = (0, 255, 255)
        else:
            mode_text = "MULTI-CAM"
            mode_color = (0, 255, 0)
        
        cv2.putText(frame, f"Mode: {mode_text}", (120, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)
        
        # Tracking status
        if self.tracker:
            if self.tracker.is_tracking:
                track_text = "TRACKING"
                track_color = (0, 255, 0)
            else:
                track_text = "Click ball"
                track_color = (0, 255, 255)
            cv2.putText(frame, track_text, (320, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 1)
        
        # Shot detector status
        if self.shot_detector:
            cv2.putText(frame, f"Shot: {self.shot_detector.status_text}", (480, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 1)
        
        # Latency estimate
        if fps > 0:
            latency = 50 + (1000 / fps) * 5
            cv2.putText(frame, f"Latency: ~{latency:.0f}ms", (700, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Clients
        if self.ws_server:
            cv2.putText(frame, f"Clients: {self.ws_server.client_count}", (w - 120, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
    
    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == 255:
            return True
        
        if key == ord('q') or key == 27:
            if self.calibration.mode != CalibrationMode.NONE:
                self.calibration.cancel_calibration()
            else:
                return False
        
        elif key == ord('r'):
            self.calibration.start_roi_calibration()
            if self.tracker:
                self.tracker.reset()
        
        elif key == ord('c'):
            self.calibration.start_target_line_calibration()
        
        elif key == ord('s'):
            self.calibration.start_scale_calibration()
            if self.tracker:
                self.tracker.reset()
        
        elif key == ord('d'):
            # Depth calibration
            if self.camera_manager:
                self.camera_manager.start_depth_calibration()
        
        elif key == ord('v'):
            # Toggle validation overlay
            self._show_validation = not self._show_validation
            print(f"[Main] Validation overlay: {'ON' if self._show_validation else 'OFF'}")
        
        elif key == ord('p'):
            # Toggle replay recording
            self._replay_enabled = not self._replay_enabled
            print(f"[Main] Replay recording: {'ON' if self._replay_enabled else 'OFF'}")
        
        elif key == ord('e'):
            # Toggle exposure (would need camera access)
            print("[Main] Exposure toggle not yet implemented for multi-cam")
        
        elif key == ord('x'):
            if self.tracker:
                self.tracker.reset()
            if self.shot_detector:
                self.shot_detector.reset()
            if self.camera_manager:
                self.camera_manager.position_validator.clear()
        
        elif key == ord('t'):
            self._trigger_test_shot()
        
        return True
    
    def _mouse_callback(self, event: int, x: int, y: int, flags: int, param):
        """Handle mouse events."""
        if self.calibration.mode != CalibrationMode.NONE:
            self.calibration.handle_mouse_event(event, x, y, flags)
            return
        
        if event == cv2.EVENT_LBUTTONDOWN and self.tracker:
            # Adjust for display scaling
            self.tracker.set_ball_position(x // 2, y // 2)  # Assuming 2x scale
            if self.shot_detector:
                self.shot_detector.reset()
            if self.camera_manager:
                self.camera_manager.position_validator.clear()
    
    def _on_shot_detected(self, shot: ShotEvent):
        """Callback when shot is detected."""
        # Trigger replay recording
        if self._replay_enabled and self.camera_manager:
            self.camera_manager.trigger_shot_recording(shot.to_dict())
        
        # Broadcast to WebSocket clients
        if self.ws_server:
            self.ws_server.broadcast(shot.to_dict())
    
    def _on_calibration_complete(self, calibration_data):
        """Callback when depth calibration completes."""
        print("[Main] Depth calibration complete!")
        
        # Update calibration
        self.calibration.data.ground_depth = calibration_data.ground_depth
        self.calibration.data.depth_tolerance = calibration_data.depth_tolerance
        
        # Save to config
        if 'calibration' not in self.app_config:
            self.app_config['calibration'] = {}
        
        self.app_config['calibration']['ground_depth'] = calibration_data.ground_depth
        self.app_config['calibration']['depth_tolerance'] = calibration_data.depth_tolerance
        self.app_config['calibration']['calibration_source'] = 'realsense_depth'
        
        self._save_config()
    
    def _trigger_test_shot(self):
        """Trigger a test shot."""
        import random
        
        test_shot = ShotEvent(
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            detection_time_ms=80.0,
            speed_mps=round(1.0 + random.random() * 2.0, 3),
            direction_deg=round((random.random() - 0.5) * 10, 2),
            confidence=round(0.85 + random.random() * 0.15, 2),
            start_position=(0.0, 0.0),
            frame_count=8,
            trajectory_length_m=0.1,
            camera_fps=87.0,
            latency_estimate_ms=95.0,
        )
        
        print("\n" + "="*50)
        print("TEST SHOT")
        print("="*50)
        print(f"  Speed: {test_shot.speed_mps:.2f} m/s")
        print(f"  Direction: {test_shot.direction_deg:+.1f}°")
        print(f"  Confidence: {test_shot.confidence:.0%}")
        print("="*50 + "\n")
        
        # Trigger replay if enabled
        if self._replay_enabled and self.camera_manager:
            self.camera_manager.trigger_shot_recording(test_shot.to_dict())
        
        if self.ws_server:
            self.ws_server.broadcast(test_shot.to_dict())
    
    def _print_instructions(self):
        """Print keyboard instructions."""
        print("\n" + "-"*60)
        print("MULTI-CAMERA PUTTING MONITOR")
        print("-"*60)
        
        if self.camera_manager and not self.demo_mode:
            print(f"\nActive cameras: {', '.join(self.camera_manager.active_cameras)}")
        
        print("\nKEYBOARD CONTROLS:")
        print("-"*60)
        print("  CLICK - Select ball position")
        print("  x     - Reset tracker")
        print("  t     - Test shot")
        print("  d     - Depth calibration (RealSense)")
        print("  r     - ROI calibration")
        print("  c     - Target line calibration")
        print("  s     - Scale calibration")
        print("  v     - Toggle validation overlay")
        print("  p     - Toggle replay recording")
        print("  q     - Quit")
        print("-"*60)
        
        if self.camera_manager and not self.demo_mode:
            fps = self.camera_manager.primary_fps
            latency = 50 + (1000 / max(fps, 1)) * 5
            print(f"\n📊 Expected latency: ~{latency:.0f}ms")
        print("")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Putting Launch Monitor - Multi-Camera Edition"
    )
    
    parser.add_argument("--config", "-c", default="config.json",
                       help="Path to config file")
    parser.add_argument("--http-port", type=int, default=8080,
                       help="HTTP server port")
    parser.add_argument("--ws-port", type=int, default=8765,
                       help="WebSocket server port")
    parser.add_argument("--no-browser", action="store_true",
                       help="Don't open browser")
    parser.add_argument("--single", action="store_true",
                       help="Single camera mode (primary tracker only)")
    parser.add_argument("--demo", action="store_true",
                       help="Demo mode (no cameras)")
    
    args = parser.parse_args()
    
    # Signal handler
    def signal_handler(sig, frame):
        print("\n[Main] Interrupted")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Start monitor
    monitor = MultiCamPuttingMonitor(
        config_path=args.config,
        http_port=args.http_port,
        ws_port=args.ws_port,
        open_browser=not args.no_browser,
        single_camera=args.single,
        demo_mode=args.demo
    )
    
    monitor.start()


if __name__ == "__main__":
    main()
