"""
Shot detection module for the putting launch monitor.

Detects when a putt starts and calculates launch metrics.
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
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self), indent=2)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ShotDetectorConfig:
    """Configuration for shot detector."""
    # Velocity thresholds (m/s)
    start_velocity_threshold: float = 0.15  # Velocity to trigger shot start
    stop_velocity_threshold: float = 0.05  # Velocity to consider ball stopped
    
    # Measurement window
    measurement_window_ms: float = 400  # Time window to measure initial velocity
    min_measurement_frames: int = 5  # Minimum frames for valid measurement
    
    # Detection
    min_trajectory_points: int = 5  # Minimum points for valid shot
    max_detection_time_ms: float = 2000  # Maximum time to wait for shot completion
    
    # Confidence thresholds
    min_confidence: float = 0.3  # Minimum confidence to report shot
    
    # State reset
    cooldown_ms: float = 1000  # Time to wait before detecting new shot


class ShotState:
    """State machine states for shot detection."""
    IDLE = "idle"  # Waiting for ball movement
    DETECTING = "detecting"  # Shot in progress, measuring
    COOLDOWN = "cooldown"  # Just completed a shot, waiting


class ShotDetector:
    """
    Detects putting shots and calculates launch metrics.
    
    State machine:
    IDLE -> DETECTING (when velocity exceeds threshold)
    DETECTING -> IDLE (shot measured and reported)
    DETECTING -> COOLDOWN (if measurement times out)
    COOLDOWN -> IDLE (after cooldown period)
    
    Metrics calculated:
    - Ball speed (average over initial window)
    - Direction (relative to calibrated target line)
    - Confidence (based on tracking quality)
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
    
    def update(self, tracking_result: Optional[TrackingResult]):
        """
        Update shot detection with new tracking result.
        
        Args:
            tracking_result: Latest tracking result from ball tracker.
        """
        current_time = time.time()
        
        if self._state == ShotState.IDLE:
            self._handle_idle_state(tracking_result, current_time)
        
        elif self._state == ShotState.DETECTING:
            self._handle_detecting_state(tracking_result, current_time)
        
        elif self._state == ShotState.COOLDOWN:
            self._handle_cooldown_state(current_time)
    
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
            print(f"[Shot] Detected motion: {tracking_result.speed:.2f} m/s")
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
        
        # Check for shot completion conditions
        should_complete = False
        
        # 1. Ball has stopped
        if tracking_result and tracking_result.detected:
            if tracking_result.speed < self.config.stop_velocity_threshold:
                if elapsed_ms > self.config.measurement_window_ms:
                    should_complete = True
                    print("[Shot] Ball stopped")
        
        # 2. Measurement window elapsed with enough data
        if elapsed_ms >= self.config.measurement_window_ms:
            if len(self._shot_trajectory) >= self.config.min_measurement_frames:
                should_complete = True
                print(f"[Shot] Measurement window complete ({len(self._shot_trajectory)} frames)")
        
        # 3. Timeout
        if elapsed_ms >= self.config.max_detection_time_ms:
            should_complete = True
            print("[Shot] Detection timeout")
        
        # 4. Lost tracking
        if not tracking_result or not tracking_result.detected:
            # Allow a few missed frames
            missed_frames = sum(1 for t in self._shot_trajectory[-5:] if not t.detected)
            if missed_frames >= 3:
                should_complete = True
                print("[Shot] Lost tracking")
        
        if should_complete:
            self._complete_shot(current_time)
    
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
        
        # Create shot event
        shot = ShotEvent(
            timestamp=datetime.now().isoformat(),
            detection_time_ms=detection_time,
            speed_mps=round(speed, 3),
            direction_deg=round(direction, 2),
            confidence=round(confidence, 2),
            start_position=(
                round(valid_points[0].world_x, 3),
                round(valid_points[0].world_y, 3)
            ),
            frame_count=len(valid_points),
            trajectory_length_m=round(trajectory_length, 3),
        )
        
        self.last_shot = shot
        self._total_shots += 1
        
        # Print to stdout
        print("\n" + "="*50)
        print("SHOT DETECTED")
        print("="*50)
        print(shot.to_json())
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
        # Use initial portion for speed/direction (first 200-400ms)
        measurement_window_s = self.config.measurement_window_ms / 1000
        start_time = trajectory[0].timestamp
        
        initial_points = [
            t for t in trajectory 
            if t.timestamp - start_time <= measurement_window_s
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
        - Depth consistency (weight: 0.2)
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
        
        # 3. Depth consistency
        depths = [t.depth for t in initial_points if t.depth > 0]
        if len(depths) >= 2:
            depth_std = np.std(depths)
            depth_mean = np.mean(depths)
            if depth_mean > 0:
                depth_consistency = 1.0 - min(1.0, depth_std / depth_mean)
                scores.append(depth_consistency)
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
    def status_text(self) -> str:
        """Get status text for display."""
        if self._state == ShotState.IDLE:
            return "Ready"
        elif self._state == ShotState.DETECTING:
            return f"Detecting... ({len(self._shot_trajectory)} frames)"
        else:
            return "Cooldown"
