"""
Shot detection module for the putting launch monitor.

Detects when a putt starts and calculates launch metrics.
Optimized for high-FPS cameras (60-100 FPS) with low latency.
"""

import json
import math
import time
from dataclasses import dataclass, asdict
from typing import Optional, Callable, List
from collections import deque
from datetime import datetime

import numpy as np

from .ball_tracker import TrackingResult
from .calibration import Calibration


@dataclass
class ShotEvent:
    """Detected shot event with metrics."""
    # Timing
    timestamp: str
    detection_time_ms: float
    
    # Primary metrics
    speed_mps: float  # Ball speed in meters per second
    direction_deg: float  # Direction relative to target line (+ = right)
    
    # Quality
    confidence: float  # 0-1 confidence score
    
    # Additional data
    start_position: tuple  # (x, y) in meters
    frame_count: int  # Number of frames used
    trajectory_length_m: float  # Length of measured trajectory
    
    # Performance metrics
    camera_fps: float = 0.0  # Actual camera FPS during shot
    latency_estimate_ms: float = 0.0  # Estimated total latency
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ShotDetectorConfig:
    """
    Configuration for shot detector.
    
    Optimized defaults for high-FPS cameras (100 FPS):
    - measurement_window_ms: 100ms (was 400ms) - fires much faster
    - min_measurement_frames: 5 frames - enough for accuracy
    """
    # Velocity thresholds (m/s)
    start_velocity_threshold: float = 0.15  # Velocity to trigger shot start
    stop_velocity_threshold: float = 0.05  # Velocity to consider ball stopped
    
    # Measurement window - REDUCED for low latency
    measurement_window_ms: float = 100  # Time window to measure initial velocity
    min_measurement_frames: int = 5  # Minimum frames for valid measurement
    
    # Adaptive timing based on FPS
    adaptive_timing: bool = True  # Adjust windows based on actual FPS
    min_window_ms: float = 50  # Absolute minimum window
    max_window_ms: float = 200  # Maximum window (for slow cameras)
    
    # Detection
    min_trajectory_points: int = 3  # Minimum points for valid shot (reduced)
    max_detection_time_ms: float = 1500  # Maximum time to wait for shot completion
    
    # Early firing mode - fires as soon as confident
    early_fire_enabled: bool = True
    early_fire_min_frames: int = 3  # Fire after this many good frames if confident
    early_fire_confidence: float = 0.7  # Minimum confidence for early fire
    
    # Confidence thresholds
    min_confidence: float = 0.3  # Minimum confidence to report shot
    
    # State reset
    cooldown_ms: float = 800  # Time to wait before detecting new shot (reduced)


class ShotState:
    """State machine states for shot detection."""
    IDLE = "idle"  # Waiting for ball movement
    DETECTING = "detecting"  # Shot in progress, measuring
    COOLDOWN = "cooldown"  # Just completed a shot, waiting


class ShotDetector:
    """
    Detects putting shots and calculates launch metrics.
    
    Optimized for low-latency operation with high-FPS cameras.
    
    State machine:
    IDLE -> DETECTING (when velocity exceeds threshold)
    DETECTING -> IDLE (shot measured and reported)
    DETECTING -> COOLDOWN (if measurement times out)
    COOLDOWN -> IDLE (after cooldown period)
    
    Metrics calculated:
    - Ball speed (average over initial window)
    - Direction (relative to calibrated target line)
    - Confidence (based on tracking quality)
    
    Low-latency features:
    - Early firing when confidence is high
    - Adaptive measurement window based on FPS
    - Minimal frame requirements for fast detection
    """
    
    def __init__(
        self,
        calibration: Calibration,
        on_shot_callback: Optional[Callable[[ShotEvent], None]] = None,
        config: Optional[ShotDetectorConfig] = None
    ):
        """
        Initialize shot detector.
        
        Args:
            calibration: Calibration data for direction calculation.
            on_shot_callback: Callback function when shot is detected.
            config: Detector configuration.
        """
        self.calibration = calibration
        self.on_shot_callback = on_shot_callback
        self.config = config or ShotDetectorConfig()
        
        # State machine
        self._state = ShotState.IDLE
        self._state_start_time = time.time()
        
        # Trajectory buffer for current shot
        self._shot_trajectory: List[TrackingResult] = []
        
        # Last shot for display
        self.last_shot: Optional[ShotEvent] = None
        
        # Stats
        self._total_shots = 0
        
        # FPS tracking for adaptive timing
        self._recent_frame_times: deque = deque(maxlen=30)
        self._estimated_fps = 30.0
    
    def update(self, tracking_result: Optional[TrackingResult]):
        """
        Update shot detection with new tracking result.
        
        Args:
            tracking_result: Latest tracking result from ball tracker.
        """
        current_time = time.time()
        
        # Track frame timing for FPS estimation
        if tracking_result and tracking_result.timestamp > 0:
            self._recent_frame_times.append(tracking_result.timestamp)
            self._update_fps_estimate()
        
        if self._state == ShotState.IDLE:
            self._handle_idle_state(tracking_result, current_time)
        
        elif self._state == ShotState.DETECTING:
            self._handle_detecting_state(tracking_result, current_time)
        
        elif self._state == ShotState.COOLDOWN:
            self._handle_cooldown_state(current_time)
    
    def _update_fps_estimate(self):
        """Update estimated FPS from recent frame times."""
        if len(self._recent_frame_times) >= 2:
            times = list(self._recent_frame_times)
            deltas = [times[i] - times[i-1] for i in range(1, len(times))]
            avg_delta = sum(deltas) / len(deltas)
            if avg_delta > 0:
                self._estimated_fps = 1.0 / avg_delta
    
    def _get_adaptive_window_ms(self) -> float:
        """Get measurement window adjusted for current FPS."""
        if not self.config.adaptive_timing:
            return self.config.measurement_window_ms
        
        # At 100 FPS: use shorter window (faster detection)
        # At 30 FPS: use longer window (need more time for frames)
        
        # Target: ~5-10 frames worth of data
        target_frames = max(self.config.min_measurement_frames, 5)
        ideal_window = (target_frames / self._estimated_fps) * 1000
        
        # Clamp to configured range
        window = max(self.config.min_window_ms, 
                    min(self.config.max_window_ms, ideal_window))
        
        return window
    
    def _handle_idle_state(
        self, 
        tracking_result: Optional[TrackingResult],
        current_time: float
    ):
        """Handle IDLE state - waiting for shot to start."""
        if not tracking_result or not tracking_result.detected:
            return
        
        # Check if ball is moving fast enough
        if tracking_result.speed >= self.config.start_velocity_threshold:
            fps_str = f"{self._estimated_fps:.0f}" if self._estimated_fps > 0 else "?"
            print(f"[Shot] Motion detected: {tracking_result.speed:.2f} m/s @ {fps_str} FPS")
            self._state = ShotState.DETECTING
            self._state_start_time = current_time
            self._shot_trajectory = [tracking_result]
    
    def _handle_detecting_state(
        self,
        tracking_result: Optional[TrackingResult],
        current_time: float
    ):
        """Handle DETECTING state - measuring shot."""
        elapsed_ms = (current_time - self._state_start_time) * 1000
        
        # Add to trajectory if valid
        if tracking_result and tracking_result.detected:
            self._shot_trajectory.append(tracking_result)
        
        # Get adaptive measurement window
        window_ms = self._get_adaptive_window_ms()
        
        # Check for early fire (low latency mode)
        if self.config.early_fire_enabled:
            if self._should_early_fire(elapsed_ms):
                print(f"[Shot] Early fire! ({len(self._shot_trajectory)} frames, {elapsed_ms:.0f}ms)")
                self._complete_shot(current_time)
                return
        
        # Check for shot completion conditions
        should_complete = False
        
        # 1. Ball has stopped
        if tracking_result and tracking_result.detected:
            if tracking_result.speed < self.config.stop_velocity_threshold:
                if elapsed_ms > window_ms:
                    should_complete = True
                    print("[Shot] Ball stopped")
        
        # 2. Measurement window elapsed with enough data
        if elapsed_ms >= window_ms:
            if len(self._shot_trajectory) >= self.config.min_measurement_frames:
                should_complete = True
                print(f"[Shot] Window complete ({len(self._shot_trajectory)} frames in {elapsed_ms:.0f}ms)")
        
        # 3. Timeout
        if elapsed_ms >= self.config.max_detection_time_ms:
            should_complete = True
            print("[Shot] Detection timeout")
        
        # 4. Lost tracking
        if not tracking_result or not tracking_result.detected:
            # Allow a few missed frames
            recent = self._shot_trajectory[-5:] if len(self._shot_trajectory) >= 5 else self._shot_trajectory
            missed_frames = sum(1 for t in recent if not t.detected)
            if missed_frames >= 3:
                should_complete = True
                print("[Shot] Lost tracking")
        
        if should_complete:
            self._complete_shot(current_time)
    
    def _should_early_fire(self, elapsed_ms: float) -> bool:
        """Check if we should fire the shot event early."""
        # Need minimum frames
        if len(self._shot_trajectory) < self.config.early_fire_min_frames:
            return False
        
        # Need minimum elapsed time (avoid noise)
        if elapsed_ms < self.config.min_window_ms:
            return False
        
        # Filter to valid points
        valid_points = [t for t in self._shot_trajectory if t.detected]
        if len(valid_points) < self.config.early_fire_min_frames:
            return False
        
        # Calculate quick confidence estimate
        # Check if we have consistent velocity
        speeds = [t.speed for t in valid_points[-5:]]  # Last 5 valid frames
        if not speeds:
            return False
        
        avg_speed = sum(speeds) / len(speeds)
        if avg_speed < self.config.start_velocity_threshold:
            return False
        
        # Check velocity consistency (low variance = confident)
        if len(speeds) >= 3:
            variance = sum((s - avg_speed) ** 2 for s in speeds) / len(speeds)
            std_dev = math.sqrt(variance)
            
            # Coefficient of variation (lower = more consistent)
            cv = std_dev / avg_speed if avg_speed > 0 else 1.0
            
            # If velocity is very consistent, fire early
            if cv < 0.3:  # Less than 30% variation
                return True
        
        return False
    
    def _handle_cooldown_state(self, current_time: float):
        """Handle COOLDOWN state - waiting before next shot."""
        elapsed_ms = (current_time - self._state_start_time) * 1000
        
        if elapsed_ms >= self.config.cooldown_ms:
            self._state = ShotState.IDLE
            self._shot_trajectory = []
            print("[Shot] Ready for next shot")
    
    def _complete_shot(self, current_time: float):
        """Complete shot detection and calculate metrics."""
        detection_time = (current_time - self._state_start_time) * 1000
        
        # Filter to valid trajectory points
        valid_points = [t for t in self._shot_trajectory if t.detected]
        
        if len(valid_points) < self.config.min_trajectory_points:
            print(f"[Shot] Insufficient data ({len(valid_points)} points)")
            self._state = ShotState.COOLDOWN
            self._state_start_time = current_time
            return
        
        # Calculate metrics
        speed, direction, confidence = self._calculate_metrics(valid_points)
        
        if confidence < self.config.min_confidence:
            print(f"[Shot] Low confidence: {confidence:.2f}")
            self._state = ShotState.COOLDOWN
            self._state_start_time = current_time
            return
        
        # Calculate trajectory length
        trajectory_length = self._calculate_trajectory_length(valid_points)
        
        # Estimate total latency
        # Detection time + estimated processing + websocket
        latency_estimate = detection_time + 10  # +10ms for processing/network
        
        # Create shot event
        shot = ShotEvent(
            timestamp=datetime.now().isoformat(),
            detection_time_ms=round(detection_time, 1),
            speed_mps=round(speed, 3),
            direction_deg=round(direction, 2),
            confidence=round(confidence, 2),
            start_position=(
                round(valid_points[0].world_x, 3),
                round(valid_points[0].world_y, 3)
            ),
            frame_count=len(valid_points),
            trajectory_length_m=round(trajectory_length, 3),
            camera_fps=round(self._estimated_fps, 1),
            latency_estimate_ms=round(latency_estimate, 1),
        )
        
        self.last_shot = shot
        self._total_shots += 1
        
        # Print to stdout with latency info
        print("\n" + "="*50)
        print("SHOT DETECTED")
        print("="*50)
        print(f"  Speed: {shot.speed_mps:.2f} m/s")
        print(f"  Direction: {shot.direction_deg:+.1f}°")
        print(f"  Confidence: {shot.confidence:.0%}")
        print(f"  Detection time: {shot.detection_time_ms:.0f}ms")
        print(f"  Frames used: {shot.frame_count}")
        print(f"  Camera FPS: {shot.camera_fps:.0f}")
        print(f"  Est. latency: {shot.latency_estimate_ms:.0f}ms")
        print("="*50 + "\n")
        
        # Callback
        if self.on_shot_callback:
            self.on_shot_callback(shot)
        
        # Enter cooldown
        self._state = ShotState.COOLDOWN
        self._state_start_time = current_time
    
    def _calculate_metrics(
        self, 
        trajectory: List[TrackingResult]
    ) -> tuple:
        """
        Calculate shot metrics from trajectory.
        
        Returns:
            (speed_mps, direction_deg, confidence)
        """
        # Use initial portion for speed/direction
        window_ms = self._get_adaptive_window_ms()
        window_s = window_ms / 1000
        start_time = trajectory[0].timestamp
        
        initial_points = [
            t for t in trajectory 
            if t.timestamp - start_time <= window_s
        ]
        
        if len(initial_points) < 2:
            initial_points = trajectory[:min(10, len(trajectory))]
        
        # Calculate speed using linear regression
        speeds = []
        for i in range(1, len(initial_points)):
            dt = initial_points[i].timestamp - initial_points[i-1].timestamp
            if dt > 0:
                dx = initial_points[i].world_x - initial_points[i-1].world_x
                dy = initial_points[i].world_y - initial_points[i-1].world_y
                speed = math.sqrt(dx*dx + dy*dy) / dt
                speeds.append(speed)
        
        if not speeds:
            return 0.0, 0.0, 0.0
        
        # Average speed (filter outliers)
        speeds = np.array(speeds)
        median_speed = np.median(speeds)
        filtered_speeds = speeds[speeds < median_speed * 3]  # Remove extreme outliers
        
        if len(filtered_speeds) > 0:
            avg_speed = float(np.mean(filtered_speeds))
        else:
            avg_speed = float(median_speed)
        
        # Calculate direction from overall trajectory vector
        first = initial_points[0]
        last = initial_points[-1]
        dx = last.world_x - first.world_x
        dy = last.world_y - first.world_y
        
        velocity_angle = math.atan2(dy, dx)
        direction = self.calibration.get_direction_relative_to_target(velocity_angle)
        
        # Calculate confidence score
        confidence = self._calculate_confidence(initial_points, trajectory)
        
        return avg_speed, direction, confidence
    
    def _calculate_confidence(
        self,
        initial_points: List[TrackingResult],
        full_trajectory: List[TrackingResult]
    ) -> float:
        """
        Calculate confidence score for the shot.
        
        Based on:
        - Valid frame ratio (weight: 0.4)
        - Trajectory linearity (weight: 0.4)
        - Velocity consistency (weight: 0.2)
        """
        scores = []
        weights = []
        
        # 1. Valid frame ratio
        valid_ratio = len([t for t in full_trajectory if t.detected]) / max(len(full_trajectory), 1)
        scores.append(valid_ratio)
        weights.append(0.4)
        
        # 2. Trajectory linearity (R-squared of linear fit)
        if len(initial_points) >= 3:
            x = np.array([t.world_x for t in initial_points])
            y = np.array([t.world_y for t in initial_points])
            
            # Fit line
            if np.std(x) > 0.001:  # Avoid divide by zero
                try:
                    # Simple linear regression
                    n = len(x)
                    sum_x = np.sum(x)
                    sum_y = np.sum(y)
                    sum_xy = np.sum(x * y)
                    sum_x2 = np.sum(x * x)
                    
                    denom = n * sum_x2 - sum_x * sum_x
                    if abs(denom) > 1e-10:
                        slope = (n * sum_xy - sum_x * sum_y) / denom
                        intercept = (sum_y - slope * sum_x) / n
                        
                        # Calculate R-squared
                        y_pred = slope * x + intercept
                        ss_res = np.sum((y - y_pred) ** 2)
                        ss_tot = np.sum((y - np.mean(y)) ** 2)
                        
                        if ss_tot > 0:
                            r_squared = 1 - (ss_res / ss_tot)
                            r_squared = max(0, min(1, r_squared))
                            scores.append(r_squared)
                            weights.append(0.4)
                except Exception:
                    pass
        
        # 3. Velocity consistency (replaced depth consistency)
        speeds = [t.speed for t in initial_points if t.speed > 0]
        if len(speeds) >= 2:
            speed_std = np.std(speeds)
            speed_mean = np.mean(speeds)
            if speed_mean > 0:
                # Lower variation = higher confidence
                velocity_consistency = 1.0 - min(1.0, speed_std / speed_mean)
                scores.append(velocity_consistency)
                weights.append(0.2)
        
        # Weighted average
        if scores and weights:
            confidence = sum(s * w for s, w in zip(scores, weights)) / sum(weights)
            return float(confidence)
        
        return 0.5  # Default moderate confidence
    
    def _calculate_trajectory_length(self, trajectory: List[TrackingResult]) -> float:
        """Calculate total length of trajectory in meters."""
        total = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i].world_x - trajectory[i-1].world_x
            dy = trajectory[i].world_y - trajectory[i-1].world_y
            total += math.sqrt(dx*dx + dy*dy)
        return total
    
    def reset(self):
        """Reset detector state."""
        self._state = ShotState.IDLE
        self._shot_trajectory = []
        print("[Shot] Reset")
    
    def update_config_from_dict(self, config_dict: dict):
        """
        Update configuration from dictionary (e.g., from config.json).
        
        Args:
            config_dict: Dictionary with config values.
        """
        if 'measurement_window_ms' in config_dict:
            self.config.measurement_window_ms = config_dict['measurement_window_ms']
        if 'min_measurement_frames' in config_dict:
            self.config.min_measurement_frames = config_dict['min_measurement_frames']
        if 'start_velocity_threshold' in config_dict:
            self.config.start_velocity_threshold = config_dict['start_velocity_threshold']
        if 'stop_velocity_threshold' in config_dict:
            self.config.stop_velocity_threshold = config_dict['stop_velocity_threshold']
        if 'min_confidence' in config_dict:
            self.config.min_confidence = config_dict['min_confidence']
        if 'cooldown_ms' in config_dict:
            self.config.cooldown_ms = config_dict['cooldown_ms']
        
        print(f"[Shot] Config updated: window={self.config.measurement_window_ms}ms, "
              f"min_frames={self.config.min_measurement_frames}")
    
    @property
    def state(self) -> str:
        """Get current state."""
        return self._state
    
    @property
    def is_detecting(self) -> bool:
        """Check if currently detecting a shot."""
        return self._state == ShotState.DETECTING
    
    @property
    def total_shots(self) -> int:
        """Get total detected shots."""
        return self._total_shots
    
    @property
    def estimated_fps(self) -> float:
        """Get estimated camera FPS."""
        return self._estimated_fps
    
    @property
    def status_text(self) -> str:
        """Get status text for display."""
        if self._state == ShotState.IDLE:
            return f"Ready ({self._estimated_fps:.0f}fps)"
        elif self._state == ShotState.DETECTING:
            return f"Detecting... ({len(self._shot_trajectory)} frames)"
        else:
            return "Cooldown"
