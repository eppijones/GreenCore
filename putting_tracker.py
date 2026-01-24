#!/usr/bin/env python3
"""
Putting Tracker - Golf Simulator Ball Tracking

Unified tool for:
- Camera alignment and calibration
- Real-time ball tracking with shot detection
- Speed, direction, and angle measurement
- WebSocket output to virtual ball simulation

Two-zone system:
- HITTING ZONE: Area where ball starts (small, left side)
- MEASUREMENT ZONE: Rolling area for speed/direction tracking (large, right side)

Controls:
    S - Snap to ball (auto-calibrate scale)
    T - Toggle TRACKING mode (enables shot detection)
    G - Toggle grid
    L - Lock/unlock calibration
    ARROWS - Move ROI (when unlocked)
    +/- - Adjust scale
    R - Reset
    Q - Quit

Usage:
    python putting_tracker.py
    # or
    python main.py
"""

import cv2
import numpy as np
import json
import time
import math
import argparse
import webbrowser
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, List
from collections import deque

# Camera imports
from cv.arducam_120fps import Arducam120FPS, CaptureConfig, CaptureError

# Shared ball detector
from cv.ball_detector import BallDetector, DetectorConfig, BallDetection

# Shot detection and server
from cv.shot_detector import ShotDetector, ShotDetectorConfig, ShotEvent
from cv.calibration import Calibration
from server.websocket_server import WebSocketServer
from server.web_server import WebServer


@dataclass
class AlignmentConfig:
    """Alignment configuration."""
    # ROI for the full tracking area (putting + runway)
    # Uses nearly full frame: 5px margin on left/right, minimal top/bottom margins
    # Frame is 1280x800
    roi_x: int = 5           # 5px from left edge
    roi_y: int = 100         # Below header text
    roi_width: int = 1270    # 1280 - 5 - 5 = 1270 (5px margin each side)
    roi_height: int = 620    # Leaves room for status bar at bottom
    
    # Divider position (fraction of ROI width for putting/hitting zone)
    # Hitting zone is where ball starts - adjusted to match actual ball position
    putting_zone_fraction: float = 0.28  # ~350px for hitting area, rest is runway
    
    # Scale (pixels per cm) - calibrated for 80cm camera height
    # Formula: scale = focal_length * real_size / distance
    # With 50px ball at 80cm: 50px / 4.27cm = 11.7 px/cm
    scale_px_per_cm: float = 11.7
    
    # Camera height (cm) - distance from camera to putting surface
    camera_height_cm: float = 80.0
    
    # Zone depth (cm) - the tracking zone depth
    zone_depth_cm: float = 30.0
    
    # Expected ball diameter in pixels (based on scale and real ball ~4.27cm)
    @property
    def expected_ball_px(self) -> int:
        return int(4.27 * self.scale_px_per_cm)
    
    # Zone depth in pixels
    @property
    def zone_depth_px(self) -> int:
        return int(self.zone_depth_cm * self.scale_px_per_cm)
    
    # Lock state
    locked: bool = False
    
    # Tracking mode
    tracking_enabled: bool = True  # Enable shot detection by default


class PuttingTracker:
    """
    Unified camera alignment, calibration, and tracking tool.
    
    Provides:
    - Visual feedback for camera positioning
    - Scale calibration from ball size
    - Real-time shot detection with speed/direction
    - WebSocket output to virtual ball simulation
    """
    
    def __init__(
        self,
        config_path: str = "config.json",
        camera_index: int = -1,
        enable_server: bool = True,
        http_port: int = 8080,
        ws_port: int = 8765
    ):
        self.config_path = config_path
        self.camera_index = camera_index
        self.enable_server = enable_server
        self.http_port = http_port
        self.ws_port = ws_port
        
        # Load or create alignment config
        self.align_config = self._load_config()
        
        # Camera
        self.camera: Optional[Arducam120FPS] = None
        
        # Shared ball detector
        self.detector: Optional[BallDetector] = None
        self.last_detection: Optional[BallDetection] = None
        
        # Tracking state
        self._trajectory: deque = deque(maxlen=60)  # Last 60 frames of ball positions
        self._velocity: Tuple[float, float] = (0.0, 0.0)  # Current velocity in m/s
        self._last_tracking_time: float = 0.0
        self._tracking_frame_count: int = 0
        
        # Shot detection
        self.calibration: Optional[Calibration] = None
        self.shot_detector: Optional[ShotDetector] = None
        self.last_shot: Optional[ShotEvent] = None
        
        # Servers
        self.web_server: Optional[WebServer] = None
        self.ws_server: Optional[WebSocketServer] = None
        
        # State
        self._running = False
        self.show_grid = True
        
        # Window
        self.window_name = "Golf Putting Tracker"
    
    def _load_config(self) -> AlignmentConfig:
        """Load alignment config from JSON."""
        config_path = Path(self.config_path)
        config = AlignmentConfig()
        
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
                
                # Load alignment settings if present
                if 'alignment' in data:
                    align = data['alignment']
                    config.roi_x = align.get('roi_x', config.roi_x)
                    config.roi_y = align.get('roi_y', config.roi_y)
                    config.roi_width = align.get('roi_width', config.roi_width)
                    config.roi_height = align.get('roi_height', config.roi_height)
                    config.putting_zone_fraction = align.get('putting_zone_fraction', config.putting_zone_fraction)
                    config.scale_px_per_cm = align.get('scale_px_per_cm', config.scale_px_per_cm)
                    config.camera_height_cm = align.get('camera_height_cm', config.camera_height_cm)
                    config.locked = align.get('locked', config.locked)
                    config.tracking_enabled = align.get('tracking_enabled', config.tracking_enabled)
                
                print(f"[Align] Loaded config from {config_path}")
            except Exception as e:
                print(f"[Align] Could not load config: {e}")
        
        return config
    
    def _save_config(self):
        """Save alignment config to JSON."""
        config_path = Path(self.config_path)
        
        # Load existing config or create new
        data = {}
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    data = json.load(f)
            except:
                pass
        
        # Update alignment section
        data['alignment'] = {
            'roi_x': self.align_config.roi_x,
            'roi_y': self.align_config.roi_y,
            'roi_width': self.align_config.roi_width,
            'roi_height': self.align_config.roi_height,
            'putting_zone_fraction': self.align_config.putting_zone_fraction,
            'scale_px_per_cm': self.align_config.scale_px_per_cm,
            'camera_height_cm': self.align_config.camera_height_cm,
            'locked': self.align_config.locked,
            'tracking_enabled': self.align_config.tracking_enabled,
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Align] Saved config to {config_path}")
    
    def start(self):
        """Start alignment and tracking mode."""
        print("\n" + "="*60)
        print("  GOLF PUTTING TRACKER")
        print("  Alignment + Shot Detection")
        print("="*60 + "\n")
        
        try:
            # Initialize calibration (for shot direction calculation)
            print("[Tracker] Loading calibration...")
            self.calibration = Calibration(self.config_path)
            
            # Update calibration with alignment scale
            self.calibration.data.pixels_per_meter = self.align_config.scale_px_per_cm * 100
            
            # Initialize servers if enabled
            if self.enable_server:
                print("[Tracker] Starting servers...")
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
                
                # Open browser
                url = f"http://localhost:{self.http_port}"
                print(f"[Tracker] Opening browser: {url}")
                webbrowser.open(url)
            
            # Initialize camera
            print("[Tracker] Initializing Arducam...")
            config = CaptureConfig(
                width=1280,
                height=800,
                target_fps=120,
                device_index=self.camera_index,
            )
            self.camera = Arducam120FPS(config)
            self.camera.start()
            print("[Tracker] Camera initialized!")
            
            # Initialize shared ball detector
            detector_cfg = DetectorConfig(
                scale_px_per_cm=self.align_config.scale_px_per_cm,
                roi=(
                    self.align_config.roi_x,
                    self.align_config.roi_y,
                    self.align_config.roi_width,
                    self.align_config.roi_height
                )
            )
            self.detector = BallDetector(detector_cfg)
            print("[Tracker] Ball detector initialized!")
            
            # Initialize shot detector
            if self.align_config.tracking_enabled:
                shot_config = ShotDetectorConfig(
                    start_velocity_threshold=0.15,
                    measurement_window_ms=100,
                    min_measurement_frames=5,
                    early_fire_enabled=True,
                )
                self.shot_detector = ShotDetector(
                    self.calibration,
                    on_shot_callback=self._on_shot_detected,
                    config=shot_config
                )
                print("[Tracker] Shot detector initialized!")
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 800)
            
            # Print instructions
            self._print_instructions()
            
            # Main loop
            self._running = True
            self._main_loop()
            
        except CaptureError as e:
            print(f"[Tracker] Camera error: {e}")
        except Exception as e:
            print(f"[Tracker] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop alignment and tracking mode."""
        print("\n[Tracker] Shutting down...")
        self._running = False
        
        if self.camera:
            self.camera.stop()
        
        if self.ws_server:
            self.ws_server.stop()
        
        if self.web_server:
            self.web_server.stop()
        
        cv2.destroyAllWindows()
        print("[Tracker] Goodbye!")
    
    def _on_shot_detected(self, shot: ShotEvent):
        """Callback when shot is detected."""
        self.last_shot = shot
        
        # Broadcast to WebSocket clients
        if self.ws_server:
            self.ws_server.broadcast(shot.to_dict())
        
        # Reset trajectory for next shot
        self._trajectory.clear()
        self._velocity = (0.0, 0.0)
    
    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            # Get frame
            frame_data = self.camera.get_frames(timeout_ms=100)
            if frame_data is None:
                continue
            
            frame = frame_data.color_frame
            gray = frame_data.gray_frame
            timestamp = frame_data.timestamp
            
            # Detect ball using shared detector
            self.last_detection = self.detector.detect(frame, gray, timestamp)
            
            # Update tracking if enabled
            if self.align_config.tracking_enabled and self.last_detection:
                tracking_result = self._update_tracking(self.last_detection, timestamp)
                
                # Feed to shot detector
                if self.shot_detector and tracking_result:
                    self.shot_detector.update(tracking_result)
            
            # Draw overlays
            display = self._draw_overlays(frame)
            
            # Show frame
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
    
    def _update_tracking(self, detection: BallDetection, timestamp: float):
        """
        Update tracking state from ball detection.
        
        Converts BallDetection to TrackingResult format for shot detector.
        """
        from cv.ball_tracker import TrackingResult
        
        if not detection.detected:
            return None
        
        cfg = self.align_config
        
        # Convert pixel position to world coordinates (meters)
        # Origin is at center of ROI
        roi_center_x = cfg.roi_x + cfg.roi_width / 2
        roi_center_y = cfg.roi_y + cfg.roi_height / 2
        
        world_x = (detection.x - roi_center_x) / (cfg.scale_px_per_cm * 100)
        world_y = (detection.y - roi_center_y) / (cfg.scale_px_per_cm * 100)
        
        # Calculate velocity from trajectory
        velocity_x, velocity_y = 0.0, 0.0
        
        if self._trajectory and self._last_tracking_time > 0:
            dt = timestamp - self._last_tracking_time
            if dt > 0.001:  # Avoid division by zero
                last = self._trajectory[-1]
                dx = world_x - last.world_x
                dy = world_y - last.world_y
                
                # Calculate displacement in pixels for noise filtering
                dx_px = detection.x - last.pixel_x
                dy_px = detection.y - last.pixel_y
                displacement_px = math.sqrt(dx_px**2 + dy_px**2)
                
                # Filter out sub-pixel jitter (detection noise)
                # At 120fps, 1-2 pixel jitter creates false velocity readings
                # Only update velocity if ball moved more than 2 pixels
                if displacement_px > 2.0:
                    # Smooth velocity with exponential filter
                    # Higher alpha = more responsive, lower = more stable
                    alpha = 0.3  # Slightly lower for better stability
                    velocity_x = alpha * (dx / dt) + (1 - alpha) * self._velocity[0]
                    velocity_y = alpha * (dy / dt) + (1 - alpha) * self._velocity[1]
                else:
                    # Ball essentially stationary - decay velocity toward zero
                    decay = 0.7  # Quick decay when stationary
                    velocity_x = self._velocity[0] * decay
                    velocity_y = self._velocity[1] * decay
                
                self._velocity = (velocity_x, velocity_y)
        
        self._last_tracking_time = timestamp
        self._tracking_frame_count += 1
        
        # Create tracking result
        result = TrackingResult(
            detected=True,
            pixel_x=detection.x,
            pixel_y=detection.y,
            world_x=world_x,
            world_y=world_y,
            depth=0.8,  # Camera height
            circularity=detection.confidence,
            area=math.pi * detection.radius**2,
            timestamp=timestamp,
            frame_number=self._tracking_frame_count,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
        )
        
        # Add to trajectory
        self._trajectory.append(result)
        
        return result
    
    def _draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """Draw alignment overlays on frame."""
        display = frame.copy()
        h, w = display.shape[:2]
        cfg = self.align_config
        
        # Colors
        GREEN = (0, 255, 0)
        YELLOW = (0, 255, 255)
        RED = (0, 0, 255)
        WHITE = (255, 255, 255)
        
        # Semi-transparent overlay for zones
        overlay = display.copy()
        
        # Draw ROI rectangle
        roi_color = GREEN if cfg.locked else YELLOW
        cv2.rectangle(overlay, 
                     (cfg.roi_x, cfg.roi_y),
                     (cfg.roi_x + cfg.roi_width, cfg.roi_y + cfg.roi_height),
                     roi_color, 2)
        
        # Calculate divider position
        divider_x = cfg.roi_x + int(cfg.roi_width * cfg.putting_zone_fraction)
        
        # Draw divider line
        cv2.line(overlay,
                (divider_x, cfg.roi_y),
                (divider_x, cfg.roi_y + cfg.roi_height),
                YELLOW, 2)
        
        # Draw horizontal center line (target line)
        center_y = cfg.roi_y + cfg.roi_height // 2
        cv2.line(overlay,
                (cfg.roi_x, center_y),
                (cfg.roi_x + cfg.roi_width, center_y),
                YELLOW, 1)
        
        # Add runway arrow and label
        arrow_start = (cfg.roi_x + cfg.roi_width - 100, center_y)
        arrow_end = (cfg.roi_x + cfg.roi_width - 20, center_y)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, YELLOW, 2, tipLength=0.3)
        cv2.putText(overlay, "RUNWAY", 
                   (cfg.roi_x + cfg.roi_width - 80, center_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        
        # Add hitting zone label
        hitting_zone_center = cfg.roi_x + int(cfg.roi_width * cfg.putting_zone_fraction / 2)
        cv2.putText(overlay, "HIT",
                   (hitting_zone_center - 15, cfg.roi_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        
        # Add measurement zone label
        measure_zone_center = divider_x + int((cfg.roi_x + cfg.roi_width - divider_x) / 2)
        cv2.putText(overlay, "SPEED/DIRECTION MEASUREMENT",
                   (measure_zone_center - 130, cfg.roi_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, YELLOW, 1)
        
        # Grid overlay
        if self.show_grid:
            # Vertical grid lines every 50px
            for x in range(cfg.roi_x, cfg.roi_x + cfg.roi_width, 50):
                cv2.line(overlay, (x, cfg.roi_y), (x, cfg.roi_y + cfg.roi_height),
                        (100, 100, 100), 1)
            # Horizontal grid lines
            for y in range(cfg.roi_y, cfg.roi_y + cfg.roi_height, 50):
                cv2.line(overlay, (cfg.roi_x, y), (cfg.roi_x + cfg.roi_width, y),
                        (100, 100, 100), 1)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.7, display, 0.3, 0, display)
        
        # Draw ball detection
        det = self.last_detection
        if det and det.detected:
            bx, by = det.x, det.y
            br = det.radius
            
            # Target crosshairs
            cv2.circle(display, (bx, by), br + 10, GREEN, 2)
            cv2.line(display, (bx - br - 20, by), (bx + br + 20, by), GREEN, 1)
            cv2.line(display, (bx, by - br - 20), (bx, by + br + 20), GREEN, 1)
            
            # Show velocity vector if moving
            speed = math.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
            if speed > 0.05:  # 5cm/s threshold
                # Draw velocity arrow
                vx_px = self._velocity[0] * cfg.scale_px_per_cm * 100 * 0.3  # Scale for visibility
                vy_px = self._velocity[1] * cfg.scale_px_per_cm * 100 * 0.3
                arrow_end = (int(bx + vx_px), int(by + vy_px))
                cv2.arrowedLine(display, (bx, by), arrow_end, (0, 165, 255), 2, tipLength=0.3)
                
                # Show speed
                cv2.putText(display, f"{speed:.2f} m/s",
                           (bx + br + 25, by - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            else:
                # Ball stationary - show alignment info
                dist_from_center = by - center_y
                cv2.putText(display, f"{abs(dist_from_center)}px",
                           (bx + br + 25, by + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
                
                cv2.putText(display, "BALL OK", (bx - 35, by + br + 35),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            
            # Confidence indicator
            conf_text = f"Conf: {det.confidence:.0%}"
            cv2.putText(display, conf_text, (bx - 30, by + br + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        
        # Draw trajectory trail
        if len(self._trajectory) > 1:
            points = [(int(t.pixel_x), int(t.pixel_y)) for t in self._trajectory if t.detected]
            for i in range(1, len(points)):
                alpha = i / len(points)  # Fade trail
                color = (int(255 * alpha), int(100 * alpha), int(255 * (1-alpha)))
                cv2.line(display, points[i-1], points[i], color, 2)
        
        # Draw last shot info
        if self.last_shot:
            self._draw_shot_info(display)
        
        # Draw header
        if cfg.tracking_enabled:
            header_text = "PUTTING TRACKER"
            header_color = GREEN
        else:
            header_text = "ALIGNMENT MODE"
            header_color = YELLOW
        cv2.putText(display, header_text, (w//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, header_color, 2)
        
        # Sub-header based on state
        if cfg.tracking_enabled:
            sub_text = "Putt the ball to measure speed & direction"
            cv2.putText(display, sub_text, (w//2 - 200, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        else:
            sub_text = "Press 'S' to calibrate scale from ball, 'T' to start tracking"
            cv2.putText(display, sub_text, (w//2 - 250, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, WHITE, 1)
        
        # Draw status bar
        self._draw_status_bar(display)
        
        return display
    
    def _draw_shot_info(self, frame: np.ndarray):
        """Draw last shot info panel."""
        h, w = frame.shape[:2]
        shot = self.last_shot
        
        # Draw shot info box in top-right corner
        box_w, box_h = 250, 145
        box_x = w - box_w - 10
        box_y = 10
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
        
        # Border
        cv2.rectangle(frame, (box_x, box_y), (box_x + box_w, box_y + box_h), (0, 255, 0), 2)
        
        # Title
        cv2.putText(frame, "LAST SHOT", (box_x + 70, box_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Speed
        cv2.putText(frame, f"Speed: {shot.speed_mps:.2f} m/s",
                   (box_x + 15, box_y + 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Direction
        dir_color = (0, 255, 255) if abs(shot.direction_deg) < 2 else (0, 165, 255)
        cv2.putText(frame, f"Direction: {shot.direction_deg:+.1f}°",
                   (box_x + 15, box_y + 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, dir_color, 1)
        
        # Measured distance (tracked trajectory)
        dist_cm = shot.trajectory_length_m * 100
        cv2.putText(frame, f"Tracked: {dist_cm:.0f}cm",
                   (box_x + 15, box_y + 105),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # Latency
        cv2.putText(frame, f"Latency: {shot.latency_estimate_ms:.0f}ms",
                   (box_x + 15, box_y + 130),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def _draw_status_bar(self, frame: np.ndarray):
        """Draw status bar at bottom."""
        h, w = frame.shape[:2]
        cfg = self.align_config
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 70), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        y1 = h - 45
        y2 = h - 20
        
        # Coverage info
        coverage_w = cfg.roi_width / cfg.scale_px_per_cm
        coverage_h = cfg.roi_height / cfg.scale_px_per_cm
        cv2.putText(frame, f"Coverage: {int(coverage_w)}x{int(coverage_h)}cm",
                   (10, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Expected ball size
        cv2.putText(frame, f"Ball: {cfg.expected_ball_px}px",
                   (220, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Camera FPS
        if self.shot_detector:
            fps = self.shot_detector.estimated_fps
            cv2.putText(frame, f"FPS: {fps:.0f}",
                       (320, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Tracking/Shot detector status
        if cfg.tracking_enabled and self.shot_detector:
            status = self.shot_detector.status_text
            status_color = (0, 255, 0) if "Ready" in status else (0, 255, 255)
            cv2.putText(frame, f"Shot: {status}", (400, y1),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 1)
        
        # Lock status
        lock_text = "[LOCKED]" if cfg.locked else "[UNLOCKED]"
        lock_color = (0, 255, 0) if cfg.locked else (0, 255, 255)
        cv2.putText(frame, lock_text, (650, y1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 1)
        
        # WebSocket clients
        if self.ws_server:
            cv2.putText(frame, f"Clients: {self.ws_server.client_count}",
                       (w - 100, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        # Second row
        mode_text = "TRACKING" if cfg.tracking_enabled else "ALIGN"
        mode_color = (0, 255, 0) if cfg.tracking_enabled else (0, 255, 255)
        cv2.putText(frame, mode_text, (10, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 1)
        
        cv2.putText(frame, f"Scale: {cfg.scale_px_per_cm:.1f} px/cm",
                   (110, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Ball status
        det = self.last_detection
        if det and det.detected:
            speed = math.sqrt(self._velocity[0]**2 + self._velocity[1]**2)
            if speed > 0.05:
                ball_text = f"Ball: {speed:.2f} m/s"
                ball_color = (0, 165, 255)
            else:
                ball_text = f"Ball: {det.diameter}px READY"
                ball_color = (0, 255, 0)
        else:
            ball_text = "Ball: NOT FOUND"
            ball_color = (0, 100, 255)
        cv2.putText(frame, ball_text, (290, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 1)
        
        # Total shots
        if self.shot_detector:
            cv2.putText(frame, f"Shots: {self.shot_detector.total_shots}",
                       (500, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Controls hint
        cv2.putText(frame, "S=snap  T=toggle tracking  G=grid  L=lock  R=reset  Q=quit",
                   (10, h - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    def _handle_key(self, key: int) -> bool:
        """Handle keyboard input. Returns False to quit."""
        if key == 255:
            return True
        
        cfg = self.align_config
        
        if key == ord('q') or key == 27:
            return False
        
        elif key == ord('l'):
            # Toggle lock
            cfg.locked = not cfg.locked
            self._save_config()
            print(f"[Align] {'LOCKED' if cfg.locked else 'UNLOCKED'}")
        
        elif key == ord('t'):
            # Toggle tracking mode
            cfg.tracking_enabled = not cfg.tracking_enabled
            mode = "ENABLED" if cfg.tracking_enabled else "DISABLED"
            print(f"[Tracker] Shot detection: {mode}")
            
            # Initialize/destroy shot detector
            if cfg.tracking_enabled and not self.shot_detector:
                shot_config = ShotDetectorConfig(
                    start_velocity_threshold=0.15,
                    measurement_window_ms=100,
                    min_measurement_frames=5,
                )
                self.shot_detector = ShotDetector(
                    self.calibration,
                    on_shot_callback=self._on_shot_detected,
                    config=shot_config
                )
            self._save_config()
        
        elif key == ord('g'):
            # Toggle grid
            self.show_grid = not self.show_grid
        
        elif key == ord('r'):
            # Reset
            self.align_config = AlignmentConfig()
            print("[Align] Reset to defaults")
        
        elif key == ord('s'):
            # Snap to ball - auto-calibrate scale from detected ball
            det = self.last_detection
            if det and det.detected:
                # Estimate scale from ball size (real ball = 4.27cm)
                real_diameter_cm = 4.27
                detected_diameter_px = det.diameter
                cfg.scale_px_per_cm = detected_diameter_px / real_diameter_cm
                
                # Update detector config too
                if self.detector:
                    self.detector.config.scale_px_per_cm = cfg.scale_px_per_cm
                
                print(f"[Align] Scale calibrated: {cfg.scale_px_per_cm:.1f} px/cm")
                print(f"[Align] Ball detected at {det.diameter}px diameter")
                self._save_config()
        
        elif key == ord('c'):
            # Auto-calibrate (prompt user to place ball)
            print("[Align] Place golf ball in frame and press 'S' to calibrate scale")
        
        # Arrow keys for ROI movement
        elif key == 81 or key == 2:  # Left
            if not cfg.locked:
                cfg.roi_x = max(0, cfg.roi_x - 10)
        elif key == 83 or key == 3:  # Right
            if not cfg.locked:
                cfg.roi_x = min(1280 - cfg.roi_width, cfg.roi_x + 10)
        elif key == 82 or key == 0:  # Up
            if not cfg.locked:
                cfg.roi_y = max(0, cfg.roi_y - 10)
        elif key == 84 or key == 1:  # Down
            if not cfg.locked:
                cfg.roi_y = min(800 - cfg.roi_height, cfg.roi_y + 10)
        
        # Scale adjustment
        elif key == ord('+') or key == ord('='):
            if not cfg.locked:
                cfg.scale_px_per_cm += 0.1
                print(f"[Align] Scale: {cfg.scale_px_per_cm:.1f} px/cm")
        elif key == ord('-'):
            if not cfg.locked:
                cfg.scale_px_per_cm = max(1.0, cfg.scale_px_per_cm - 0.1)
                print(f"[Align] Scale: {cfg.scale_px_per_cm:.1f} px/cm")
        
        return True
    
    def _print_instructions(self):
        """Print usage instructions."""
        print("\n" + "-"*60)
        print("GOLF PUTTING TRACKER")
        print("-"*60)
        print("  1. Position ball in the HITTING zone (left)")
        print("  2. Putt the ball → speed/direction measured in RUNWAY")
        print("  3. Shot data sent to browser for virtual ball simulation")
        print("")
        print("CONTROLS")
        print("-"*60)
        print("  S     - Snap to ball (calibrate scale from ball size)")
        print("  T     - Toggle tracking/shot detection")
        print("  G     - Toggle grid overlay")
        print("  L     - Lock/unlock ROI adjustment")
        print("  ARROWS - Move ROI (when unlocked)")
        print("  +/-   - Adjust scale")
        print("  R     - Reset to defaults")
        print("  Q     - Quit")
        print("-"*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Golf Putting Tracker - Ball tracking with shot detection"
    )
    parser.add_argument(
        "--config", "-c",
        default="config.json",
        help="Path to config file"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Camera device index (-1 for auto-detect)"
    )
    parser.add_argument(
        "--no-server",
        action="store_true",
        help="Disable web/websocket servers"
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=8080,
        help="HTTP server port"
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        default=8765,
        help="WebSocket server port"
    )
    
    args = parser.parse_args()
    
    tracker = PuttingTracker(
        config_path=args.config,
        camera_index=args.camera_index,
        enable_server=not args.no_server,
        http_port=args.http_port,
        ws_port=args.ws_port,
    )
    
    tracker.start()


# Backwards compatibility alias
CameraAlignmentMode = PuttingTracker


if __name__ == "__main__":
    main()
