#!/usr/bin/env python3
"""
Golf Putting Tracker - Main Entry Point

Real-time ball tracking with shot detection and virtual ball simulation.

Usage:
    python main.py              # Start tracker with default settings
    python main.py --no-browser # Don't auto-open browser
    python main.py --no-server  # Disable web/websocket servers (CV only)
    python main.py --config FILE # Use custom config file

Controls:
    S - Calibrate scale from ball size (snap to ball)
    T - Toggle tracking/shot detection
    G - Toggle grid overlay
    L - Lock/unlock ROI adjustment
    ARROWS - Move ROI (when unlocked)
    +/- - Adjust scale
    R - Reset
    Q - Quit

The system:
    1. Detects golf ball in HITTING zone (left side)
    2. Tracks ball through MEASUREMENT zone (right side)
    3. Calculates speed and direction
    4. Sends shot data to browser via WebSocket
    5. Browser shows virtual ball rolling toward hole
"""

import sys
import signal
import argparse

from putting_tracker import PuttingTracker


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Golf Putting Tracker - Ball tracking with shot detection",
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
        "--no-server",
        action="store_true",
        help="Disable web/websocket servers (CV only mode)"
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=-1,
        help="Camera device index (-1 for auto-detect)"
    )
    
    args = parser.parse_args()
    
    # Handle SIGINT gracefully
    def signal_handler(sig, frame):
        print("\n[Main] Interrupted")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Create and start the unified tracker
    tracker = PuttingTracker(
        config_path=args.config,
        camera_index=args.camera_index,
        enable_server=not args.no_server,
        http_port=args.http_port,
        ws_port=args.ws_port,
    )
    
    # Disable auto browser if requested
    if args.no_browser:
        tracker.enable_server = not args.no_server  # Keep server, just don't open browser
        # We need to patch the start method to not open browser
        original_start = tracker.start
        def start_no_browser():
            import webbrowser
            original_open = webbrowser.open
            webbrowser.open = lambda url: None  # Disable browser open
            try:
                original_start()
            finally:
                webbrowser.open = original_open
        tracker.start = start_no_browser
    
    tracker.start()


if __name__ == "__main__":
    main()
