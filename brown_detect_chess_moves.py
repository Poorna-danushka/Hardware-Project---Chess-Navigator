import cv2
import numpy as np
import time
from picamera2 import Picamera2

# === CONFIG ===
CROP_SIZE = 480
GRID_ROWS = GRID_COLS = 8
SQUARE_SIZE = CROP_SIZE // GRID_ROWS
CALIB_BOX = (20, 20, 80, 80)
COLOR_MODEL_PATH = "/home/pi/Desktop/ch/claibrate/brown_hsv_model.npz"
BOARD_CALIB_PATH = "/home/pi/Desktop/ch/claibrate/board_calibration.npz"
MIN_AREA = 200
MAX_AREA = 10000  # anything bigger = hand
DETECTION_THRESHOLD = 0.05  # 5% of square must match color
HAND_CLEAR_DELAY = 1.5  # Seconds after hand clears before trusting new detection
MOVE_CONFIRM_TIME = 2.0  # Seconds to confirm a move

# === Load Calibrations ===
color_model = np.load(COLOR_MODEL_PATH)
lower_hsv = color_model["lower"]
upper_hsv = color_model["upper"]

board_calib = np.load(BOARD_CALIB_PATH, allow_pickle=True)
warp_matrix = board_calib["warp_matrix"]
square_names = board_calib["square_names"]

# === Camera Setup ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# === White Balance Utilities ===
def calibrate_white_balance(frame, region):
    x, y, w, h = region
    patch = frame[y:y+h, x:x+w]
    avg = np.mean(patch, axis=(0, 1))
    gain = 255 / avg
    return np.clip(gain, 0, 3.0)

def apply_rgb_gain(frame, gain):
    frame = frame.astype(np.float32)
    for i in range(3):
        frame[:, :, i] *= gain[i]
    return np.clip(frame, 0, 255).astype(np.uint8)

# === Main Loop State ===
rgb_gain = np.array([1.0, 1.0, 1.0])
last_grid = None
change_start = None
hand_present = False
hand_cleared_time = None
move_pending = False

print("♟️ Brown Piece Detection Running. Press 'c' to calibrate, 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = apply_rgb_gain(frame, rgb_gain)
    warped = cv2.warpPerspective(frame, warp_matrix, (CROP_SIZE, CROP_SIZE))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
    display = warped.copy()
    now = time.time()

    # === Detect hand ===
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in contours), default=0)

    if max_area > MAX_AREA:
        if not hand_present:
            print("✋ Hand or obstruction detected — pausing detection...")
        hand_present = True
        hand_cleared_time = None
        move_pending = False
        change_start = None
    else:
        if hand_present and hand_cleared_time is None:
            hand_cleared_time = now
        elif hand_present and (now - hand_cleared_time > HAND_CLEAR_DELAY):
            hand_present = False
            print("✅ Hand cleared — resuming detection")

    # === Only build grid if hand is not present ===
    if not hand_present and hand_cleared_time is not None and (now - hand_cleared_time > HAND_CLEAR_DELAY):
        new_grid = [[0] * GRID_COLS for _ in range(GRID_ROWS)]

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x, y = col * SQUARE_SIZE, row * SQUARE_SIZE
                square_mask = mask[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
                percent = cv2.countNonZero(square_mask) / (SQUARE_SIZE * SQUARE_SIZE)

                if percent > DETECTION_THRESHOLD:
                    new_grid[row][col] = 1
                    label = square_names[row][col]
                    cv2.rectangle(display, (x, y), (x+SQUARE_SIZE, y+SQUARE_SIZE), (0, 255, 0), 2)
                    cv2.putText(display, "Brown", (x+5, y+20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.putText(display, label, (x+5, y+SQUARE_SIZE-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1)

        # === Detect move ===
        if last_grid is not None and new_grid != last_grid:
            if not move_pending:
                move_pending = True
                change_start = now
            elif (now - change_start) >= MOVE_CONFIRM_TIME:
                print("♟️ Move Detected:")
                for i in range(GRID_ROWS):
                    for j in range(GRID_COLS):
                        if last_grid[i][j] == 1 and new_grid[i][j] == 0:
                            print(f"❌ Removed: {square_names[i][j]}")
                        elif last_grid[i][j] == 0 and new_grid[i][j] == 1:
                            print(f"✅ Placed:  {square_names[i][j]}")
                last_grid = [row[:] for row in new_grid]
                move_pending = False
        else:
            move_pending = False
            change_start = None

        if last_grid is None:
            last_grid = [row[:] for row in new_grid]

    # === Visuals ===
    cv2.rectangle(display, (CALIB_BOX[0], CALIB_BOX[1]),
                  (CALIB_BOX[0]+CALIB_BOX[2], CALIB_BOX[1]+CALIB_BOX[3]), (255, 0, 0), 1)
    cv2.putText(display, "Calib", (CALIB_BOX[0], CALIB_BOX[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    for i in range(1, GRID_COLS):
        cv2.line(display, (i*SQUARE_SIZE, 0), (i*SQUARE_SIZE, CROP_SIZE), (200, 200, 200), 1)
    for j in range(1, GRID_ROWS):
        cv2.line(display, (0, j*SQUARE_SIZE), (CROP_SIZE, j*SQUARE_SIZE), (200, 200, 200), 1)

    cv2.imshow("Brown Move Detector", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        rgb_gain = calibrate_white_balance(warped, CALIB_BOX)
        print(f"🎛️ Calibrated RGB gain: {rgb_gain.round(2)}")

cv2.destroyAllWindows()
picam2.stop()
