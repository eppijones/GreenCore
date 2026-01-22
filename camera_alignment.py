#!/usr/bin/env python3
"""
Camera Alignment Mode - Golf Simulator Calibration Tool

Two-zone system for putting alignment:
- PUTTING ZONE: Area where ball starts
- RUNWAY ZONE: Rolling area towards target

Features:
- Scale calibration (px/cm)
- Camera height estimation
- Ball detection and alignment
- Grid overlay for alignment

Controls:
    S - Snap to ball (auto-calibrate ball position)
    C - Auto-calibrate scale
    ARROWS - Move ROI
    +/- - Adjust scale
    G - Toggle grid
    R - Reset
    L - Lock/unlock calibration
    M - Switch to TRACKING mode
    Q - Quit
"""

import cv2
import numpy as np
import json
import time
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

# Camera imports
from cv.arducam_120fps import Arducam120FPS, CaptureConfig, CaptureError

# Shared ball detector
from cv.ball_detector import BallDetector, DetectorConfig, BallDetection


@dataclass
class AlignmentConfig:
    """Alignment configuration."""
    # ROI for the full tracking area (putting + runway)
    roi_x: int = 150
    roi_y: int = 200
    roi_width: int = 1100
    roi_height: int = 420
    
    # Divider position (fraction of ROI width for putting zone)
    putting_zone_fraction: float = 0.5
    
    # Scale (pixels per cm) - calibrated for 78cm camera height
    scale_px_per_cm: float = 11.7
    
    # Camera height (cm) - distance from camera to putting surface
    camera_height_cm: float = 78.0
    
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


class CameraAlignmentMode:
    """
    Camera alignment and calibration tool.
    
    Provides visual feedback for camera positioning and scale calibration.
    """
    
    def __init__(
        self,
        config_path: str = "config.json",
        camera_index: int = -1
    ):
        self.config_path = config_path
        self.camera_index = camera_index
        
        # Load or create alignment config
        self.align_config = self._load_config()
        
        # Camera
        self.camera: Optional[Arducam120FPS] = None
        
        # Shared ball detector
        self.detector: Optional[BallDetector] = None
        self.last_detection: Optional[BallDetection] = None
        
        # State
        self._running = False
        self.show_grid = True
        
        # Window
        self.window_name = "Camera Alignment Mode"
    
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
        }
        
        with open(config_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"[Align] Saved config to {config_path}")
    
    def start(self):
        """Start alignment mode."""
        print("\n" + "="*60)
        print("  CAMERA ALIGNMENT MODE")
        print("="*60 + "\n")
        
        try:
            # Initialize camera
            print("[Align] Initializing Arducam...")
            config = CaptureConfig(
                width=1280,
                height=800,
                target_fps=120,
                device_index=self.camera_index,
            )
            self.camera = Arducam120FPS(config)
            self.camera.start()
            print("[Align] Camera initialized!")
            
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
            print("[Align] Ball detector initialized!")
            
            # Create window
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1280, 800)
            
            # Print instructions
            self._print_instructions()
            
            # Main loop
            self._running = True
            self._main_loop()
            
        except CaptureError as e:
            print(f"[Align] Camera error: {e}")
        except Exception as e:
            print(f"[Align] Error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.stop()
    
    def stop(self):
        """Stop alignment mode."""
        print("\n[Align] Shutting down...")
        self._running = False
        
        if self.camera:
            self.camera.stop()
        
        cv2.destroyAllWindows()
        print("[Align] Goodbye!")
    
    def _main_loop(self):
        """Main processing loop."""
        while self._running:
            # Get frame
            frame_data = self.camera.get_frames(timeout_ms=100)
            if frame_data is None:
                continue
            
            frame = frame_data.color_frame
            gray = frame_data.gray_frame
            
            # Detect ball using shared detector
            self.last_detection = self.detector.detect(frame, gray, frame_data.timestamp)
            
            # Draw overlays
            display = self._draw_overlays(frame)
            
            # Show frame
            cv2.imshow(self.window_name, display)
            
            # Handle keyboard
            key = cv2.waitKey(1) & 0xFF
            if not self._handle_key(key):
                break
    
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
        
        # Add runway arrow
        arrow_start = (cfg.roi_x + cfg.roi_width - 100, center_y)
        arrow_end = (cfg.roi_x + cfg.roi_width - 20, center_y)
        cv2.arrowedLine(overlay, arrow_start, arrow_end, YELLOW, 2, tipLength=0.3)
        cv2.putText(overlay, "RUNWAY", 
                   (cfg.roi_x + cfg.roi_width - 80, center_y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, YELLOW, 1)
        
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
            
            # Distance from center line
            dist_from_center = by - center_y
            cv2.putText(display, f"{abs(dist_from_center)}px",
                       (bx + br + 25, by + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            
            # Labels
            cv2.putText(display, "TARGET", (bx - 30, by - br - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            cv2.putText(display, "BALL OK", (bx - 35, by + br + 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 1)
            
            # Confidence indicator
            conf_text = f"Conf: {det.confidence:.0%}"
            cv2.putText(display, conf_text, (bx - 30, by + br + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
        
        # Draw header
        header_text = "CAMERA LOCKED ???" if cfg.locked else "CAMERA ALIGNMENT MODE"
        header_color = GREEN if cfg.locked else YELLOW
        cv2.putText(display, header_text, (w//2 - 200, 80),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, header_color, 3)
        
        if cfg.locked:
            cv2.putText(display, "Press 'M' to switch to TRACKING mode",
                       (w//2 - 220, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 1)
        
        # Draw status bar
        self._draw_status_bar(display)
        
        return display
    
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
        cv2.putText(frame, f"Expected ball: {cfg.expected_ball_px}px",
                   (220, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Camera height
        cv2.putText(frame, f"Height: {cfg.camera_height_cm:.0f}cm",
                   (420, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Lock status
        lock_text = "[LOCKED]" if cfg.locked else "[UNLOCKED]"
        lock_color = (0, 255, 0) if cfg.locked else (0, 255, 255)
        cv2.putText(frame, lock_text, (560, y1),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 1)
        
        # Second row
        cv2.putText(frame, "ALIGN", (10, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        cv2.putText(frame, f"Scale: {cfg.scale_px_per_cm:.1f} px/cm",
                   (80, y2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Ball status
        det = self.last_detection
        if det and det.detected:
            ball_text = f"Ball: {det.diameter}px ALIGNED"
            ball_color = (0, 255, 0)
        else:
            ball_text = "Ball: NOT FOUND"
            ball_color = (0, 100, 255)
        cv2.putText(frame, ball_text, (260, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, ball_color, 1)
        
        # Lock status
        lock_text2 = "LOCKED" if cfg.locked else "UNLOCKED"
        cv2.putText(frame, lock_text2, (500, y2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, lock_color, 1)
        
        # Controls hint
        cv2.putText(frame, "S=snap to ball  C=auto-calibrate  ARROWS=move  +/-=scale  G=grid  R=reset  Q=quit",
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
        
        elif key == ord('m'):
            # Switch to tracking mode
            print("[Align] Switching to TRACKING mode...")
            self._save_config()
            return False
        
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
        print("CAMERA ALIGNMENT MODE")
        print("-"*60)
        print("  Position camera to cover putting area")
        print("  Adjust ROI to frame the mat/surface")
        print("")
        print("CONTROLS")
        print("-"*60)
        print("  S     - Snap to ball (calibrate scale)")
        print("  C     - Auto-calibrate prompt")
        print("  ARROWS - Move ROI")
        print("  +/-   - Adjust scale")
        print("  G     - Toggle grid")
        print("  L     - Lock/unlock calibration")
        print("  R     - Reset to defaults")
        print("  M     - Switch to TRACKING mode")
        print("  Q     - Quit")
        print("-"*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Camera Alignment Mode - Golf Simulator Calibration"
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
    
    args = parser.parse_args()
    
    alignment = CameraAlignmentMode(
        config_path=args.config,
        camera_index=args.camera_index
    )
    
    alignment.start()


if __name__ == "__main__":
    main()
