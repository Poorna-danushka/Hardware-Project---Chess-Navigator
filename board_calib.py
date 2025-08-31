import cv2
import numpy as np
import os
from picamera2 import Picamera2
import time

# === CONFIG ===
CROP_SIZE = 480  # final square board size
CALIB_FILE = "/home/pi/Desktop/ch/claibrate/board_calibration.npz"

# === Setup Camera ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# === Click Handling ===
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        if len(clicked_points) <= 4:
            print(f"🖱️ Corner {len(clicked_points)}: {x}, {y}")
        elif len(clicked_points) == 5:
            print(f"🖱️ A1 clicked at: {x}, {y}")
        elif len(clicked_points) == 6:
            print(f"🖱️ H1 clicked at: {x}, {y}")

# === Display for Manual Clicks ===
frame = picam2.capture_array()
clone = frame.copy()
cv2.namedWindow("Calibrate Chessboard")
cv2.setMouseCallback("Calibrate Chessboard", mouse_callback)

print("🔍 Click 4 corners in this order:")
print("   1. Top-Left   2. Top-Right   3. Bottom-Right   4. Bottom-Left")
print("Then click:")
print("   5. A1 square (bottom-left from white's view)")
print("   6. H1 square (bottom-right from white's view)")

while True:
    temp = clone.copy()
    for pt in clicked_points:
        cv2.circle(temp, pt, 5, (0, 255, 0), -1)
    cv2.imshow("Calibrate Chessboard", temp)

    if len(clicked_points) == 6:
        break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        exit()

cv2.destroyWindow("Calibrate Chessboard")

# === Compute Warp Matrix ===
corners = np.array(clicked_points[:4], dtype=np.float32)
destination = np.array([[0, 0], [CROP_SIZE, 0], [CROP_SIZE, CROP_SIZE], [0, CROP_SIZE]], dtype=np.float32)
warp_matrix = cv2.getPerspectiveTransform(corners, destination)

# === A1 & H1 Logic ===
a1_pt = cv2.perspectiveTransform(np.array([[clicked_points[4]]], dtype=np.float32), warp_matrix)[0][0]
h1_pt = cv2.perspectiveTransform(np.array([[clicked_points[5]]], dtype=np.float32), warp_matrix)[0][0]

# Determine file (column) and rank (row) directions
is_rank1_left_to_right = a1_pt[0] < h1_pt[0]
is_rank1_bottom_to_top = a1_pt[1] > h1_pt[1]

files = ['a','b','c','d','e','f','g','h']
ranks = ['1','2','3','4','5','6','7','8']
if not is_rank1_left_to_right:
    files = files[::-1]
if not is_rank1_bottom_to_top:
    ranks = ranks[::-1]

square_names = [[files[col] + ranks[7 - row] for col in range(8)] for row in range(8)]

# === Save Calibration Data ===
np.savez(CALIB_FILE,
         warp_matrix=warp_matrix,
         corners=corners,
         a1=clicked_points[4],
         h1=clicked_points[5],
         square_names=square_names)

print(f"✅ Calibration saved to {CALIB_FILE}")
picam2.stop()

