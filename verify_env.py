#!/usr/bin/env python3
"""Environment verification for Arducam 120fps MJPEG capture on macOS."""

import sys

def check(name, condition, detail=""):
    status = "✅" if condition else "❌"
    print(f"{status} {name}" + (f": {detail}" if detail else ""))
    return condition

def main():
    print("=" * 60)
    print("macOS Video I/O Environment Verification")
    print("=" * 60)
    
    all_ok = True
    
    # Python
    print("\n[Python]")
    all_ok &= check("Python 3.11+", sys.version_info >= (3, 11), f"{sys.version_info.major}.{sys.version_info.minor}")
    
    # OpenCV
    print("\n[OpenCV]")
    try:
        import cv2
        all_ok &= check("OpenCV imported", True, cv2.__version__)
        backends = [cv2.videoio_registry.getBackendName(b) for b in cv2.videoio_registry.getBackends()]
        all_ok &= check("AVFoundation backend", "AVFOUNDATION" in backends)
        all_ok &= check("FFMPEG backend", "FFMPEG" in backends)
        all_ok &= check("libjpeg-turbo", "libjpeg-turbo" in cv2.getBuildInformation())
    except ImportError:
        all_ok &= check("OpenCV imported", False)
    
    # NumPy
    print("\n[NumPy]")
    try:
        import numpy as np
        all_ok &= check("NumPy imported", True, np.__version__)
    except ImportError:
        all_ok &= check("NumPy imported", False)
    
    # PyAV
    print("\n[PyAV]")
    try:
        import av
        all_ok &= check("PyAV imported", True, av.__version__)
    except ImportError:
        all_ok &= check("PyAV imported", False, "pip install av")
    
    # imageio
    print("\n[imageio]")
    try:
        import imageio
        all_ok &= check("imageio imported", True, imageio.__version__)
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        all_ok &= check("imageio-ffmpeg", bool(ffmpeg_exe), ffmpeg_exe)
    except ImportError as e:
        all_ok &= check("imageio/imageio-ffmpeg", False, str(e))
    
    # Camera test
    print("\n[Camera Access]")
    try:
        cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)
        opened = cap.isOpened()
        all_ok &= check("AVFoundation camera open", opened)
        if opened:
            cap.release()
    except Exception as e:
        all_ok &= check("AVFoundation camera open", False, str(e))
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All checks passed - environment is ready")
    else:
        print("❌ Some checks failed - review above")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())''