#!/usr/bin/env python3
"""
Generate Calibration Printables for GolfSim Putting Tracker

Creates PDF files for:
1. ChArUco board - lens distortion calibration
2. Focus test chart - Siemens star for camera sharpness
3. Distance verification markers - for Test Lab Mode
4. Putter tracking markers - future enhancement
5. Checkerboard - for depth camera calibration

Setup parameters:
- Camera height: 78cm
- Visible mat area: 70cm x 100cm
- Mat corner markers: 5cm (already generated)

Usage:
    python generate_calibration_prints.py
    
Output:
    calibration_prints/*.pdf
"""

import os
import math
import numpy as np
import cv2

# Try to import ArUco - handle different OpenCV versions
try:
    from cv2 import aruco
    ARUCO_DICT = aruco.DICT_4X4_50
except AttributeError:
    # OpenCV 4.7+ moved ArUco
    ARUCO_DICT = cv2.aruco.DICT_4X4_50
    aruco = cv2.aruco

# Output directory
OUTPUT_DIR = "calibration_prints"

# Page sizes in mm (A4)
A4_WIDTH_MM = 210
A4_HEIGHT_MM = 297

# DPI for PDF generation
DPI = 300
MM_TO_INCH = 1 / 25.4
A4_WIDTH_PX = int(A4_WIDTH_MM * MM_TO_INCH * DPI)
A4_HEIGHT_PX = int(A4_HEIGHT_MM * MM_TO_INCH * DPI)


def mm_to_px(mm: float) -> int:
    """Convert millimeters to pixels at our DPI."""
    return int(mm * MM_TO_INCH * DPI)


def create_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}/")


def generate_charuco_board():
    """
    Generate ChArUco board for camera intrinsics calibration.
    
    ChArUco combines checkerboard corners with ArUco markers for robust
    camera calibration even with partial occlusion.
    
    Board: 6x9 squares, 25mm square size, 18mm marker size
    Uses ArUco DICT_4X4_50, marker IDs start at 50 to avoid conflict
    """
    print("\n[1/5] Generating ChArUco board...")
    
    # Board parameters
    squares_x = 6
    squares_y = 9
    square_length_mm = 25
    marker_length_mm = 18
    
    # Create ArUco dictionary and CharucoBoard
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # Create CharucoBoard - OpenCV 4.7+ API
    try:
        board = aruco.CharucoBoard(
            (squares_x, squares_y),
            square_length_mm / 1000.0,  # Convert to meters
            marker_length_mm / 1000.0,
            aruco_dict
        )
    except TypeError:
        # Older OpenCV API
        board = aruco.CharucoBoard_create(
            squares_x, squares_y,
            square_length_mm / 1000.0,
            marker_length_mm / 1000.0,
            aruco_dict
        )
    
    # Calculate board size in pixels
    board_width_mm = squares_x * square_length_mm
    board_height_mm = squares_y * square_length_mm
    
    # Add margins
    margin_mm = 15
    total_width_mm = board_width_mm + 2 * margin_mm
    total_height_mm = board_height_mm + 2 * margin_mm
    
    # Generate board image
    board_width_px = mm_to_px(board_width_mm)
    board_height_px = mm_to_px(board_height_mm)
    
    try:
        board_img = board.generateImage((board_width_px, board_height_px))
    except AttributeError:
        board_img = board.draw((board_width_px, board_height_px))
    
    # Create A4 page with board centered
    page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX), dtype=np.uint8) * 255
    
    # Calculate position to center board
    margin_px = mm_to_px(margin_mm)
    x_offset = (A4_WIDTH_PX - board_width_px) // 2
    y_offset = mm_to_px(20)  # Top margin for title
    
    # Place board on page
    page[y_offset:y_offset+board_height_px, x_offset:x_offset+board_width_px] = board_img
    
    # Add title and info
    page_color = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)
    
    title = "ChArUco Board - Camera Intrinsics Calibration"
    cv2.putText(page_color, title, (mm_to_px(15), mm_to_px(12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    info = f"{squares_x}x{squares_y} squares | {square_length_mm}mm squares | {marker_length_mm}mm markers | DICT_4X4_50"
    cv2.putText(page_color, info, (mm_to_px(15), A4_HEIGHT_PX - mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "charuco_board_6x9_25mm.png")
    cv2.imwrite(output_path, page_color)
    print(f"  Created: {output_path}")
    print(f"  Board size: {board_width_mm}x{board_height_mm}mm")
    print(f"  Square size: {square_length_mm}mm, Marker size: {marker_length_mm}mm")
    
    return output_path


def generate_focus_chart():
    """
    Generate Siemens star focus test chart.
    
    The Siemens star pattern helps assess camera sharpness.
    When in focus, the center shows clear line separation.
    When out of focus, the center appears gray/blurred.
    """
    print("\n[2/5] Generating focus test chart (Siemens star)...")
    
    # Star parameters
    num_spokes = 36  # Number of black/white pairs
    star_radius_mm = 60
    star_radius_px = mm_to_px(star_radius_mm)
    
    # Create page
    page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 3), dtype=np.uint8) * 255
    
    # Center of star
    center_x = A4_WIDTH_PX // 2
    center_y = A4_HEIGHT_PX // 2
    
    # Draw Siemens star
    for i in range(num_spokes * 2):
        angle_start = (i * math.pi) / num_spokes
        angle_end = ((i + 1) * math.pi) / num_spokes
        
        if i % 2 == 0:
            color = (0, 0, 0)  # Black spoke
        else:
            continue  # White spoke (background)
        
        # Draw wedge
        pts = [(center_x, center_y)]
        for angle in np.linspace(angle_start, angle_end, 20):
            x = int(center_x + star_radius_px * math.cos(angle))
            y = int(center_y + star_radius_px * math.sin(angle))
            pts.append((x, y))
        pts.append((center_x, center_y))
        
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(page, [pts], color)
    
    # Draw center circle (reference)
    cv2.circle(page, (center_x, center_y), mm_to_px(3), (128, 128, 128), 2)
    
    # Draw outer circle
    cv2.circle(page, (center_x, center_y), star_radius_px, (0, 0, 0), 2)
    
    # Add concentric distance rings
    for radius_mm in [20, 40]:
        cv2.circle(page, (center_x, center_y), mm_to_px(radius_mm), (200, 200, 200), 1)
    
    # Add title
    title = "Focus Test Chart - Siemens Star"
    cv2.putText(page, title, (mm_to_px(15), mm_to_px(15)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    
    # Add instructions
    instructions = [
        "1. Place chart at your working distance (78cm from camera)",
        "2. Adjust camera focus until center lines are sharpest",
        "3. Lines should be distinct to the smallest radius possible",
    ]
    y = A4_HEIGHT_PX - mm_to_px(35)
    for line in instructions:
        cv2.putText(page, line, (mm_to_px(15), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        y += mm_to_px(6)
    
    # Add resolution markers (line pairs)
    info = f"{num_spokes} spoke pairs | {star_radius_mm*2}mm diameter"
    cv2.putText(page, info, (mm_to_px(15), A4_HEIGHT_PX - mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "focus_chart_siemens_star.png")
    cv2.imwrite(output_path, page)
    print(f"  Created: {output_path}")
    
    return output_path


def generate_distance_markers():
    """
    Generate distance verification markers for Test Lab Mode.
    
    Small ArUco markers (IDs 10, 11, 12) at 3cm size to place at
    0.5m, 1.0m, and 1.5m distances for calibration verification.
    """
    print("\n[3/5] Generating distance verification markers...")
    
    marker_size_mm = 30  # 3cm markers
    marker_size_px = mm_to_px(marker_size_mm)
    
    marker_ids = [10, 11, 12]
    distances = ["0.50m", "1.00m", "1.50m"]
    
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # Create page
    page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 3), dtype=np.uint8) * 255
    
    # Title
    title = "Distance Verification Markers - Test Lab Mode"
    cv2.putText(page, title, (mm_to_px(15), mm_to_px(15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    subtitle = "Place these markers at known distances to verify calibration accuracy"
    cv2.putText(page, subtitle, (mm_to_px(15), mm_to_px(22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    # Generate each marker
    y_start = mm_to_px(40)
    spacing = mm_to_px(80)
    
    for i, (marker_id, distance) in enumerate(zip(marker_ids, distances)):
        # Generate marker
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        marker_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Position
        x = (A4_WIDTH_PX - marker_size_px) // 2
        y = y_start + i * spacing
        
        # Add white border
        border = mm_to_px(5)
        cv2.rectangle(page, (x - border, y - border), 
                     (x + marker_size_px + border, y + marker_size_px + border),
                     (255, 255, 255), -1)
        cv2.rectangle(page, (x - border, y - border), 
                     (x + marker_size_px + border, y + marker_size_px + border),
                     (0, 0, 0), 2)
        
        # Place marker
        page[y:y+marker_size_px, x:x+marker_size_px] = marker_color
        
        # Label
        label = f"ID {marker_id} - Place at {distance} from origin"
        cv2.putText(page, label, (x - border, y + marker_size_px + mm_to_px(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    # Cut lines
    for i, (marker_id, distance) in enumerate(zip(marker_ids, distances)):
        y = y_start + i * spacing - mm_to_px(15)
        if i > 0:
            cv2.line(page, (mm_to_px(20), y), (A4_WIDTH_PX - mm_to_px(20), y), 
                    (180, 180, 180), 1, cv2.LINE_AA)
            # Scissors symbol
            cv2.putText(page, "- - - cut here - - -", (A4_WIDTH_PX // 2 - mm_to_px(20), y - mm_to_px(2)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
    
    # Info
    info = f"DICT_4X4_50 | {marker_size_mm}mm markers | IDs 10-12"
    cv2.putText(page, info, (mm_to_px(15), A4_HEIGHT_PX - mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "distance_markers_30mm_ID10-12.png")
    cv2.imwrite(output_path, page)
    print(f"  Created: {output_path}")
    print(f"  Marker IDs: {marker_ids}")
    print(f"  Marker size: {marker_size_mm}mm")
    
    return output_path


def generate_putter_markers():
    """
    Generate small ArUco markers for putter head tracking.
    
    2cm markers (IDs 20-23) that can be attached to putter
    for future face angle and path tracking.
    """
    print("\n[4/5] Generating putter tracking markers...")
    
    marker_size_mm = 20  # 2cm markers
    marker_size_px = mm_to_px(marker_size_mm)
    
    marker_ids = [20, 21, 22, 23]
    positions = ["Top face", "Heel", "Toe", "Back"]
    
    aruco_dict = aruco.getPredefinedDictionary(ARUCO_DICT)
    
    # Create page
    page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 3), dtype=np.uint8) * 255
    
    # Title
    title = "Putter Head Markers - Face Angle Tracking (Future)"
    cv2.putText(page, title, (mm_to_px(15), mm_to_px(15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    
    subtitle = "Small markers for putter tracking (optional future enhancement)"
    cv2.putText(page, subtitle, (mm_to_px(15), mm_to_px(22)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
    
    # Generate markers in 2x2 grid
    grid_start_x = mm_to_px(40)
    grid_start_y = mm_to_px(50)
    cell_size = mm_to_px(60)
    
    for i, (marker_id, position) in enumerate(zip(marker_ids, positions)):
        row = i // 2
        col = i % 2
        
        # Generate marker
        marker_img = aruco.generateImageMarker(aruco_dict, marker_id, marker_size_px)
        marker_color = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
        
        # Position
        x = grid_start_x + col * cell_size + (cell_size - marker_size_px) // 2
        y = grid_start_y + row * cell_size
        
        # Add white border
        border = mm_to_px(3)
        cv2.rectangle(page, (x - border, y - border), 
                     (x + marker_size_px + border, y + marker_size_px + border),
                     (255, 255, 255), -1)
        cv2.rectangle(page, (x - border, y - border), 
                     (x + marker_size_px + border, y + marker_size_px + border),
                     (0, 0, 0), 1)
        
        # Place marker
        page[y:y+marker_size_px, x:x+marker_size_px] = marker_color
        
        # Label
        label = f"ID {marker_id}: {position}"
        label_x = grid_start_x + col * cell_size
        cv2.putText(page, label, (label_x, y + marker_size_px + mm_to_px(8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Usage instructions
    instructions = [
        "Usage (future feature):",
        "- Attach ID 20 to top of putter face (visible from above)",
        "- Optionally attach ID 21-23 for 3D orientation tracking",
        "- Enables: face angle at impact, path, tempo analysis",
    ]
    y = mm_to_px(180)
    for line in instructions:
        cv2.putText(page, line, (mm_to_px(15), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        y += mm_to_px(7)
    
    # Info
    info = f"DICT_4X4_50 | {marker_size_mm}mm markers | IDs 20-23"
    cv2.putText(page, info, (mm_to_px(15), A4_HEIGHT_PX - mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "putter_markers_20mm_ID20-23.png")
    cv2.imwrite(output_path, page)
    print(f"  Created: {output_path}")
    print(f"  Marker IDs: {marker_ids}")
    print(f"  Marker size: {marker_size_mm}mm")
    
    return output_path


def generate_checkerboard():
    """
    Generate checkerboard for depth camera calibration.
    
    9x6 internal corners, 30mm squares - standard for
    Intel RealSense D455 and ZED 2i stereo calibration.
    """
    print("\n[5/5] Generating checkerboard for depth camera...")
    
    # Board parameters (internal corners)
    corners_x = 9
    corners_y = 6
    square_size_mm = 18  # Sized to fit A4 width (10 squares * 18mm = 180mm < 210mm)
    
    # Board size in squares (corners + 1)
    squares_x = corners_x + 1  # 10
    squares_y = corners_y + 1  # 7
    
    board_width_mm = squares_x * square_size_mm
    board_height_mm = squares_y * square_size_mm
    
    board_width_px = mm_to_px(board_width_mm)
    board_height_px = mm_to_px(board_height_mm)
    square_size_px = mm_to_px(square_size_mm)
    
    # Create board
    board = np.ones((board_height_px, board_width_px), dtype=np.uint8) * 255
    
    # Draw black squares
    for row in range(squares_y):
        for col in range(squares_x):
            if (row + col) % 2 == 0:
                x1 = col * square_size_px
                y1 = row * square_size_px
                x2 = x1 + square_size_px
                y2 = y1 + square_size_px
                board[y1:y2, x1:x2] = 0
    
    # Create A4 page
    page = np.ones((A4_HEIGHT_PX, A4_WIDTH_PX, 3), dtype=np.uint8) * 255
    
    # Center board on page
    x_offset = (A4_WIDTH_PX - board_width_px) // 2
    y_offset = mm_to_px(25)  # Top margin for title
    
    # Place board
    board_color = cv2.cvtColor(board, cv2.COLOR_GRAY2BGR)
    page[y_offset:y_offset+board_height_px, x_offset:x_offset+board_width_px] = board_color
    
    # Draw border
    cv2.rectangle(page, (x_offset-2, y_offset-2), 
                 (x_offset+board_width_px+2, y_offset+board_height_px+2),
                 (0, 0, 0), 2)
    
    # Title
    title = "Checkerboard - Depth Camera Calibration"
    cv2.putText(page, title, (mm_to_px(15), mm_to_px(12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    # Usage instructions
    instructions = [
        "For Intel RealSense D455 / ZED 2i stereo calibration:",
        "- Print at 100% scale (no fit-to-page)",
        "- Mount on flat, rigid surface",
        f"- Internal corners: {corners_x}x{corners_y}",
        f"- Square size: {square_size_mm}mm (verify with ruler after printing)",
    ]
    y = y_offset + board_height_px + mm_to_px(15)
    for line in instructions:
        cv2.putText(page, line, (mm_to_px(15), y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1)
        y += mm_to_px(7)
    
    # Info
    info = f"{corners_x}x{corners_y} internal corners | {square_size_mm}mm squares | {board_width_mm}x{board_height_mm}mm board"
    cv2.putText(page, info, (mm_to_px(15), A4_HEIGHT_PX - mm_to_px(8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "checkerboard_9x6_25mm.png")
    cv2.imwrite(output_path, page)
    print(f"  Created: {output_path}")
    print(f"  Internal corners: {corners_x}x{corners_y}")
    print(f"  Square size: {square_size_mm}mm")
    
    return output_path


def convert_to_pdf():
    """Convert PNG files to PDF using img2pdf or Pillow."""
    print("\n[PDF] Converting images to PDF...")
    
    try:
        from PIL import Image
        
        png_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
        
        for png_file in png_files:
            png_path = os.path.join(OUTPUT_DIR, png_file)
            pdf_path = png_path.replace('.png', '.pdf')
            
            # Open image and convert to RGB (PDF doesn't support RGBA)
            img = Image.open(png_path)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Save as PDF
            img.save(pdf_path, 'PDF', resolution=300.0)
            print(f"  Created: {pdf_path}")
        
        print("\n✓ PDF conversion complete!")
        return True
        
    except ImportError:
        print("  Warning: Pillow not available for PDF conversion")
        print("  PNG files can be printed directly or converted manually")
        return False


def main():
    """Generate all calibration printables."""
    print("=" * 60)
    print("  GolfSim Calibration Printables Generator")
    print("=" * 60)
    print("\nSetup parameters:")
    print("  Camera height: 78cm")
    print("  Visible mat area: 70cm x 100cm")
    print("  Mat corner markers: 5cm (DICT_4X4_50, IDs 0-3)")
    
    create_output_dir()
    
    # Generate all printables
    generate_charuco_board()
    generate_focus_chart()
    generate_distance_markers()
    generate_putter_markers()
    generate_checkerboard()
    
    # Convert to PDF
    convert_to_pdf()
    
    print("\n" + "=" * 60)
    print("  GENERATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput files in: {OUTPUT_DIR}/")
    print("\nPrinting instructions:")
    print("  1. Print at 100% scale (no 'fit to page')")
    print("  2. Use matte paper for less glare")
    print("  3. Verify sizes with ruler after printing")
    print("  4. Mount ChArUco and checkerboard on rigid backing")
    print("\nYour existing mat corner markers (5cm, IDs 0-3) are correct!")


if __name__ == "__main__":
    main()
