# Golf Putt Tracking System — Technical Breakdown

A detailed walkthrough of how the GolfSim project tracks putts using the **Arducam OV9281 global shutter camera** at 120 FPS.

---

## System Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           ARDUCAM OV9281 CAMERA                              │
│                     (Global Shutter, 1280x800 @ 120fps)                      │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ USB (MJPG Stream)
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          CAPTURE THREAD                                       │
│  • Pulls frames from camera at full 120fps                                   │
│  • Assigns monotonic PTS (pseudo-presentation timestamp)                     │
│  • Updates atomic "latest frame" slot                                        │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  │ CapturedFrame (color, gray, pts, frame_num)
                    ┌─────────────┴─────────────┐
                    ▼                           ▼
┌───────────────────────────────┐   ┌───────────────────────────────────────────┐
│     PROCESSING THREAD         │   │         DISPLAY THREAD (Main Loop)        │
│  (Runs at 120fps)             │   │  (Runs at ~30-60fps)                      │
│                               │   │                                           │
│  • AutoBallTracker.update()   │   │  • Gets latest processed frame            │
│  • ShotDetector.update()      │   │  • Draws overlays & debug visualization   │
│  • Returns TrackingResult     │   │  • Shows FPS metrics (CAP/PROC/DISP)      │
└───────────────────────────────┘   │  • Handles keyboard & mouse input         │
                │                   └───────────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          SHOT DETECTED?                                       │
│  When velocity > 0.15 m/s and sustained for N frames                         │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          WEBSOCKET SERVER                                     │
│  Broadcasts ShotEvent JSON to connected browser clients                      │
└─────────────────────────────────┬────────────────────────────────────────────┘
                                  ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                          WEB VISUALIZATION (Three.js)                         │
│  Receives: { speed_mps, direction_deg, confidence, ... }                     │
│  Simulates ball rolling on virtual green                                     │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 1. Camera Capture Layer (`cv/arducam_120fps.py`)

The Arducam OV9281 is a **1MP global shutter monochrome sensor** — ideal for high-speed tracking:
- **Global shutter** eliminates motion blur on fast-moving golf balls
- **Monochrome output** = native grayscale (no Bayer demosaic overhead)
- **120 FPS** at 1280×800 via MJPG over USB

### Key Class: `Arducam120FPS`

```python
class Arducam120FPS:
    """
    Validated 120 FPS capture with:
    - Dedicated capture thread (pulls frames as fast as camera delivers)
    - Dedicated processing thread (runs tracking at full 120fps)
    - Atomic latest-frame slot (no backlog, always newest frame)
    - PTS-like timestamps from monotonic clock at frame arrival
    - Separate CAP/PROC/DISP FPS metrics
    """
```

### Capture Flow

1. **Camera Discovery** — Auto-detects Arducam by finding monochrome camera (RGB channels identical)
2. **Configuration** — Sets 1280×800 @ 120fps with MJPG codec, buffer size = 1 for low latency
3. **Capture Loop** (dedicated thread):
   ```python
   while not self._stop_event.is_set():
       ret, frame = cap.read()
       capture_pts = time.perf_counter()  # Monotonic timestamp
       
       # Convert to grayscale (OV9281 is monochrome in BGR wrapper)
       gray = frame[:, :, 0]
       
       # Update atomic slot
       with self._frame_lock:
           self._latest_frame = CapturedFrame(frame, gray, capture_pts, frame_num)
       
       # Signal processing thread
       self._process_frame_ready.set()
   ```

### Data Structure: `CapturedFrame`

```python
@dataclass
class CapturedFrame:
    frame: np.ndarray       # BGR image
    gray: np.ndarray        # Grayscale image
    capture_pts: float      # Monotonic timestamp at capture
    frame_number: int       # Sequential frame number
    arrival_time: float     # Wall clock time for logging
```

---

## 2. Ball Detection (`cv/ball_detector.py`)

The `BallDetector` uses **multi-method detection with voting** for reliability:

### Detection Pipeline

```
┌─────────────────────┐
│  Input Frame (gray) │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│  METHOD 1: Threshold + Contour Analysis         │
│  • Threshold at brightness 180 (white ball)     │
│  • Morphological open/close cleanup             │
│  • Find contours, filter by area & circularity  │
│  • Score by size match to expected ball size    │
└─────────────────────────┬───────────────────────┘
                          │ If no detection
                          ▼
┌─────────────────────────────────────────────────┐
│  METHOD 2: Hough Circles (Fallback)             │
│  • Gaussian blur + HoughCircles                 │
│  • Lower confidence than threshold method       │
└─────────────────────────┬───────────────────────┘
                          │ If still no detection
                          ▼
┌─────────────────────────────────────────────────┐
│  METHOD 3: Predictive Tracking                  │
│  • If recent detection exists (< 100ms ago)     │
│  • Predict position from velocity               │
│  • Search in small region around prediction     │
└─────────────────────────────────────────────────┘
```

### Key Parameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `scale_px_per_cm` | ~11.7 | Camera calibration at 78cm height |
| `ball_diameter_cm` | 4.27 | Standard golf ball size |
| `size_tolerance` | 0.4 | Allow ±40% from expected size |
| `min_circularity` | 0.65 | Filter non-circular contours |
| `brightness_threshold` | 180 | White ball detection threshold |

### Output: `BallDetection`

```python
@dataclass
class BallDetection:
    detected: bool
    x: int              # Center X in pixels
    y: int              # Center Y in pixels
    radius: int         # Radius in pixels
    confidence: float   # 0-1 detection confidence
    timestamp: float    # Detection timestamp
```

---

## 3. Ball Tracking (`cv/auto_ball_tracker.py`)

The `AutoBallTracker` manages the **state machine** for tracking across frames:

### State Machine

```
┌─────────────┐     ball found      ┌─────────────┐
│  SEARCHING  │────────────────────▶│  TRACKING   │
└─────────────┘                     └─────────────┘
       ▲                                  │
       │                                  │ ball moving fast
       │ timeout                          ▼
       │                            ┌───────────────┐
┌─────────────┐◀────ball exits─────│ SHOT_DETECTED │
│   WAITING   │                     └───────────────┘
└─────────────┘

Also: TRACKING ──(ball drifts out)──▶ LOST ──(500ms)──▶ SEARCHING
```

### Key Features

1. **Auto-Detection** — No click required; scans full frame periodically
2. **Adaptive Search Radius** — Expands when tracking fails, shrinks when stable
3. **Velocity Computation** — Smoothed over N frames using first/last position delta
4. **Edge Detection** — Detects when ball exits frame (near edge margin)
5. **Shot Integration** — Transitions to WAITING state after shot detected

### Tracking Loop

```python
def update(self, color_frame, gray_frame, timestamp, frame_number):
    if self._state == TrackerState.SEARCHING:
        # Periodically try auto-detect in full frame
        ball_pos = self._auto_detect_ball(color_frame, gray_frame)
        if ball_pos:
            self._state = TrackerState.TRACKING
    
    elif self._state == TrackerState.TRACKING:
        # Search near last known position
        result = self._track_near_position(color_frame, gray_frame, center)
        if result.detected:
            self._compute_velocity(result)
            self._trajectory.append(result)
            
            # Check if near edge (ball exiting)
            if self._is_near_edge(result.pixel_x, result.pixel_y):
                self._state = TrackerState.WAITING
```

### Output: `TrackingResult`

```python
@dataclass
class TrackingResult:
    detected: bool = False
    pixel_x: float = 0.0       # Position in pixel coordinates
    pixel_y: float = 0.0
    world_x: float = 0.0       # Position in meters (relative to ROI center)
    world_y: float = 0.0
    depth: float = 0.5         # Ground depth (from calibration)
    velocity_x: float = 0.0    # Velocity in m/s
    velocity_y: float = 0.0
    timestamp: float = 0.0     # Frame timestamp
    frame_number: int = 0

    @property
    def speed(self) -> float:
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)
```

---

## 4. Shot Detection (`cv/shot_detector.py`)

The `ShotDetector` determines when a putt **starts** and computes launch metrics.

### State Machine

```
┌──────┐                    ┌───────────┐
│ IDLE │──velocity > 0.15──▶│ DETECTING │
└──────┘   m/s              └───────────┘
    ▲                              │
    │                              │ window elapsed + enough frames
    │ cooldown                     │ OR ball stopped
    │                              ▼
    │                        ┌───────────┐
    └────────────────────────│ COOLDOWN  │
                             └───────────┘
```

### Detection Logic

```python
def _handle_detecting_state(self, tracking_result, current_time):
    elapsed_ms = (current_time - self._state_start_time) * 1000
    
    # Add to trajectory if valid
    if tracking_result and tracking_result.detected:
        self._shot_trajectory.append(tracking_result)
    
    # Early fire for low latency (if velocity is consistent)
    if self.config.early_fire_enabled:
        if self._should_early_fire(elapsed_ms):
            self._complete_shot(current_time)
            return
    
    # Normal completion conditions
    if elapsed_ms >= window_ms:
        if len(self._shot_trajectory) >= min_frames:
            self._complete_shot(current_time)
```

### Key Parameters (Optimized for 120 FPS)

| Parameter | Value | Purpose |
|-----------|-------|---------|
| `start_velocity_threshold` | 0.15 m/s | Trigger shot detection |
| `stop_velocity_threshold` | 0.05 m/s | Ball considered stopped |
| `measurement_window_ms` | 100 | Time to measure initial velocity |
| `min_measurement_frames` | 5 | Minimum frames for valid measurement |
| `early_fire_min_frames` | 3 | Fire early if confident |
| `cooldown_ms` | 800 | Wait before next shot |

### Metrics Calculation

```python
def _calculate_metrics(self, trajectory):
    # Speed: average from filtered frame-to-frame speeds
    speeds = []
    for i in range(1, len(initial_points)):
        dt = points[i].timestamp - points[i-1].timestamp
        dx = points[i].world_x - points[i-1].world_x
        dy = points[i].world_y - points[i-1].world_y
        speed = math.sqrt(dx*dx + dy*dy) / dt
        speeds.append(speed)
    
    avg_speed = np.mean(filtered_speeds)
    
    # Direction: angle of trajectory vector relative to target line
    velocity_angle = math.atan2(dy_total, dx_total)
    direction = calibration.get_direction_relative_to_target(velocity_angle)
    
    # Confidence: weighted score from:
    # - Valid frame ratio (40%)
    # - Trajectory linearity R² (40%)
    # - Velocity consistency (20%)
```

### Output: `ShotEvent`

```python
@dataclass
class ShotEvent:
    timestamp: str              # ISO timestamp
    detection_time_ms: float    # Time from shot start to detection
    speed_mps: float            # Ball speed in m/s
    direction_deg: float        # Direction relative to target (+ = right)
    confidence: float           # 0-1 confidence score
    start_position: tuple       # (x, y) in meters
    frame_count: int            # Frames used for measurement
    trajectory_length_m: float  # Total trajectory length
    camera_fps: float           # Actual camera FPS during shot
    latency_estimate_ms: float  # Estimated total latency
```

---

## 5. Calibration System (`cv/calibration.py`)

The `Calibration` class handles coordinate conversion and target line definition.

### Calibration Data

```python
@dataclass
class CalibrationData:
    roi: Optional[Tuple[int, int, int, int]]  # (x, y, w, h) tracking region
    target_line_p1: Optional[Tuple[int, int]] # Start of target line
    target_line_p2: Optional[Tuple[int, int]] # End of target line
    pixels_per_meter: float = 500.0           # Scale calibration
    ground_depth: float = 0.5                 # Depth in meters
    forward_angle_rad: float = 0.0            # Computed from target line
```

### Coordinate Conversions

```python
# Pixel → World (meters)
world_x = calibration.pixels_to_meters(pixel_x - roi_center_x)
world_y = calibration.pixels_to_meters(pixel_y - roi_center_y)

# Velocity angle → Direction relative to target
rel_angle = velocity_angle_rad - forward_angle_rad
# Normalized to ±180°, positive = right of target
```

### Default Behavior

- **ROI**: Defaults to full frame if not set
- **Target Line**: Defaults to "forward is RIGHT" (0° angle)
- **Scale**: Loaded from `config.json` (pixels_per_meter)

---

## 6. Main Loop Integration (`main.py`)

The `PuttingMonitor` class orchestrates all components:

### Initialization

```python
class PuttingMonitor:
    def start(self):
        # 1. Load calibration from config.json
        self.calibration = Calibration(self.config_path)
        
        # 2. Start HTTP + WebSocket servers
        self.web_server = WebServer(port=8080)
        self.ws_server = WebSocketServer(port=8765)
        
        # 3. Initialize Arducam 120fps camera
        self.camera = Arducam120FPS(config)
        self.camera.start()
        
        # 4. Initialize tracker and shot detector
        self.tracker = AutoBallTracker(self.calibration)
        self.shot_detector = ShotDetector(
            self.calibration,
            on_shot_callback=self._on_shot_detected
        )
        
        # 5. Set processing callback for 120fps tracking
        self.camera.set_processing_callback(self._process_frame_120fps)
```

### 120 FPS Processing Callback

```python
def _process_frame_120fps(self, captured_frame: CapturedFrame):
    """Runs at 120fps, decoupled from display."""
    # Run tracking
    tracking_result = self.tracker.update(
        captured_frame.frame,
        captured_frame.gray,
        captured_frame.arrival_time,
        captured_frame.frame_number
    )
    
    # Update shot detector
    self.shot_detector.update(tracking_result)
    
    return tracking_result
```

### Shot Callback

```python
def _on_shot_detected(self, shot: ShotEvent):
    # Notify tracker to wait for virtual ball
    self.tracker.notify_shot_detected()
    
    # Broadcast to WebSocket clients
    self.ws_server.broadcast(shot.to_dict())
```

---

## 7. WebSocket Communication (`server/websocket_server.py`)

The `WebSocketServer` broadcasts shot events to connected browsers.

### Message Format (JSON)

```json
{
    "timestamp": "2026-01-22T10:30:45",
    "detection_time_ms": 85.0,
    "speed_mps": 1.45,
    "direction_deg": 2.3,
    "confidence": 0.92,
    "start_position": [0.0, 0.0],
    "frame_count": 8,
    "trajectory_length_m": 0.12,
    "camera_fps": 118.5,
    "latency_estimate_ms": 95.0
}
```

### Thread-Safe Broadcasting

```python
def broadcast(self, message: dict):
    """Thread-safe - can be called from any thread."""
    self._message_queue.put(message)

# Async loop processes queue
async def _process_queue(self):
    while True:
        message = self._message_queue.get_nowait()
        await self._broadcast_to_all(message)
```

---

## 8. Web Visualization (`web/app.js`)

A **Three.js** application receives shot data and simulates ball physics.

### Shot Reception

```javascript
state.ws.onmessage = (e) => {
    const data = JSON.parse(e.data);
    if (data.speed_mps) {
        updateHUD(data);
        startShot(data.speed_mps, data.direction_deg);
    }
};
```

### Physics Simulation

```javascript
function startShot(speedMps, directionDeg) {
    const angleRad = (directionDeg * Math.PI) / 180;
    
    state.ball.velocity.x = speedMps * Math.sin(angleRad);
    state.ball.velocity.y = -speedMps * Math.cos(angleRad); // -Z is forward
    state.ball.isRolling = true;
}

function updatePhysics(dt) {
    // Friction deceleration
    const frictionAccel = CONFIG.gravity * CONFIG.friction;
    const speedDrop = frictionAccel * dt;
    
    // Update velocity & position
    state.ball.velocity.multiplyScalar(newSpeed / speed);
    state.ball.position.x += state.ball.velocity.x * dt;
    state.ball.position.z += state.ball.velocity.y * dt;
}
```

---

## 9. Performance Metrics

The system tracks three separate FPS metrics:

| Metric | Description | Target |
|--------|-------------|--------|
| **CAP** | Camera capture rate (frames delivered) | ~120 fps |
| **PROC** | Tracking processing rate | ~120 fps |
| **DISP** | UI display refresh rate | ~30-60 fps |

### Validation Mode

Run `python main.py --validate` to verify 120fps capture:

```
======================================================================
  ARDUCAM 120 FPS VALIDATION REPORT
======================================================================

  ✅ RESULT: PASS

  CONFIGURATION
  ----------------------------------------
    Requested: 1280x800 @ 120fps
    Actual:    1280x800

  CAPTURE METRICS
  ----------------------------------------
    Duration:       5.00 seconds
    Frames:         600
    Capture FPS:    119.8

  INTER-FRAME TIMING
  ----------------------------------------
    Expected:       8.33ms
    Min:            7.89ms
    Avg:            8.35ms
    Max:            9.12ms
    Std Dev:        0.42ms

  ANOMALIES
  ----------------------------------------
    PTS anomalies:  0 (duplicate/non-increasing)
    Dropped frames: 0
```

---

## 10. File Structure Summary

```
GolfSim/
├── main.py                    # Entry point, orchestrates everything
├── config.json                # Calibration data (ROI, scale, etc.)
├── cv/
│   ├── arducam_120fps.py      # 120fps validated capture module
│   ├── arducam_camera.py      # Legacy arducam interface
│   ├── ball_detector.py       # Multi-method ball detection
│   ├── auto_ball_tracker.py   # State machine tracker with auto-detect
│   ├── ball_tracker.py        # Base tracker (for depth cameras)
│   ├── shot_detector.py       # Shot detection + metrics calculation
│   └── calibration.py         # ROI, scale, target line calibration
├── server/
│   ├── web_server.py          # HTTP server for static files
│   └── websocket_server.py    # Real-time shot event broadcasting
└── web/
    ├── index.html             # Visualization page
    ├── app.js                 # Three.js ball simulation
    └── style.css              # Styling
```

---

## Key Latency Breakdown

| Stage | Time |
|-------|------|
| Camera frame arrival | ~8.3ms (120fps) |
| Ball detection | ~1-2ms |
| Shot measurement window | 50-100ms (configurable) |
| WebSocket broadcast | ~1ms |
| **Total (detection → browser)** | **~60-100ms** |

This low latency makes the system feel "real-time" — the virtual ball starts rolling almost immediately after the physical putt.
