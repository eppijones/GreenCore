# Putting Launch Monitor MVP

A real-time putting launch monitor using Intel RealSense D455 depth camera with 3D web-based visualization.

## Quick Start

### 1. Set up Python 3.11 environment

```bash
cd ~/DEV/GolfSim

# Create venv with Python 3.11 (NOT 3.14)
/opt/homebrew/opt/python@3.11/bin/python3.11 -m venv .venv
source .venv/bin/activate

# Verify Python version
python --version  # Should show Python 3.11.x

# Upgrade pip
python -m pip install --upgrade pip setuptools wheel
```

### 2. Install dependencies

```bash
pip install numpy opencv-python websockets
```

### 3. Install pyrealsense2 (for camera support)

```bash
# Clone librealsense matching your Homebrew version
cd ~/Downloads
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
git checkout v2.57.5  # Match your brew version: brew info librealsense | head

# Install Python wrapper
cd wrappers/python
pip install .

# Test
python -c "import pyrealsense2 as rs; print('OK', rs.__version__)"
```

### 4. Run the monitor

```bash
cd ~/DEV/GolfSim
source .venv/bin/activate

# With camera
python main.py

# Demo mode (no camera needed)
python main.py --demo
```

## Features

- **Real-time Ball Tracking**: Classical CV using IR imagery for high-contrast detection
- **Shot Detection**: Automatic detection with speed and direction measurement
- **3D Visualization**: Browser-based Three.js putting simulator
- **Sub-second Latency**: Optimized for immediate visual feedback
- **Calibration Tools**: Interactive ROI, target line, and scale calibration

## Keyboard Controls

| Key | Action |
|-----|--------|
| `r` | Start ROI calibration (draw rectangle) |
| `c` | Start target line calibration (click two points) |
| `s` | Start scale calibration (click two points 1m apart) |
| `g` | Calibrate ground plane depth |
| `e` | Toggle auto/manual exposure |
| `+/-` | Adjust exposure (manual mode) |
| `t` | Trigger test shot |
| `x` | Reset tracker |
| `ESC` | Cancel calibration |
| `q` | Quit |

## Calibration Workflow

1. **Mount camera** looking down at your putting surface
2. **Start monitor**: `python main.py`
3. **Define ROI** (press `r`): Draw rectangle around tracking area
4. **Set target line** (press `c`): Click start point, then toward hole
5. **Scale calibration** (press `s`): Click two points 1 meter apart
6. **Ground calibration** (press `g`): Auto-detects ground depth

Calibration saves to `config.json` automatically.

## 3D Simulator Controls

- **Mouse drag**: Rotate view
- **Scroll**: Zoom
- **R**: Reset ball
- **T**: Test shot

## Hardware Requirements

- Intel RealSense D455
- USB 3.0 port (direct connection, not through hub)
- macOS (Apple Silicon or Intel)

## Troubleshooting

### Camera Not Found
1. Ensure D455 is connected via USB 3.0
2. Check with: `system_profiler SPUSBDataType | grep -i realsense`
3. Try `rs-enumerate-devices` to verify

### pyrealsense2 Build Fails
- Use Python 3.11, not 3.14
- Ensure librealsense is installed via Homebrew first

### Low FPS
1. Define a smaller ROI
2. Ensure USB 3.0 connection
3. Close other applications

## Project Structure

```
GolfSim/
├── cv/
│   ├── realsense_camera.py    # Camera interface
│   ├── ball_tracker.py        # CV detection pipeline
│   ├── shot_detector.py       # Shot event detection
│   └── calibration.py         # Calibration tools
├── server/
│   ├── web_server.py          # HTTP server
│   └── websocket_server.py    # Real-time events
├── web/
│   ├── index.html             # 3D simulator
│   ├── app.js                 # Three.js scene
│   └── style.css              # UI styling
├── main.py                    # Entry point
└── config.json                # Calibration (generated)
```
