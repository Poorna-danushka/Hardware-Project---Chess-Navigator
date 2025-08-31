import cv2
import numpy as np
import time
from picamera2 import Picamera2

# === Paths ===
CALIB_PATH = "/home/pi/Desktop/ch/claibrate/board_calibration.npz"
BROWN_MODEL_PATH = "//home/pi/Desktop/ch/claibrate/brown_hsv_model.npz"

# === Load Calibration ===
data = np.load(CALIB_PATH, allow_pickle=True)
warp_matrix = data["warp_matrix"]
square_names = data["square_names"]

brown_model = np.load(BROWN_MODEL_PATH)
lower_brown = brown_model["lower"]
upper_brown = brown_model["upper"]

# === Config ===
CROP_SIZE = 480
GRID_ROWS = GRID_COLS = 8
SQUARE_SIZE = CROP_SIZE // GRID_ROWS
CALIB_BOX = (20, 20, 80, 80)
DETECTION_THRESHOLD = 0.05
MIN_AREA = 200
MAX_AREA = 10000
HAND_COOLDOWN = 2.0  # Seconds after hand disappears before tracking resumes

# === Helper Functions ===
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

# === Camera Setup ===
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (1280, 720), "format": "RGB888"})
picam2.configure(config)
picam2.start()
time.sleep(2)

# === State ===
rgb_gain = np.array([1.0, 1.0, 1.0])
last_grid = None
change_start = None
hand_present = False
hand_cleared_time = None

print("♟️ Brown + Black Piece Detector Running. Press 'c' to calibrate, 'q' to quit.")

while True:
    frame = picam2.capture_array()
    frame = apply_rgb_gain(frame, rgb_gain)
    warped = cv2.warpPerspective(frame, warp_matrix, (CROP_SIZE, CROP_SIZE))
    hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)
    rgb = cv2.cvtColor(warped, cv2.COLOR_BGR2RGB)

    # === Masks ===
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    _, black_mask = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)

    # === Hand Detection ===
    contours, _ = cv2.findContours(cv2.bitwise_or(brown_mask, black_mask),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(cnt) for cnt in contours), default=0)
    now = time.time()
    hand_safe_to_track = False

    if max_area > MAX_AREA:
        hand_present = True
        hand_cleared_time = None
        print("✋ Hand detected — waiting...")
    else:
        if hand_present and hand_cleared_time is None:
            hand_cleared_time = now
        elif hand_present and (now - hand_cleared_time > HAND_COOLDOWN):
            hand_present = False
            print("✅ Hand removed — resuming detection.")
        if not hand_present:
            hand_safe_to_track = True

    # === Analyze Grid ===
    display = warped.copy()
    new_grid = [["" for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x = col * SQUARE_SIZE
            y = row * SQUARE_SIZE

            roi_brown = brown_mask[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]
            roi_black = black_mask[y:y+SQUARE_SIZE, x:x+SQUARE_SIZE]

            brown_score = cv2.countNonZero(roi_brown) / (SQUARE_SIZE * SQUARE_SIZE)
            black_score = cv2.countNonZero(roi_black) / (SQUARE_SIZE * SQUARE_SIZE)

            label = ""
            if brown_score > DETECTION_THRESHOLD:
                label = "Brown"
            elif black_score > DETECTION_THRESHOLD:
                label = "Black"

            if label:
                square = square_names[row][col]
                new_grid[row][col] = label
                cv2.rectangle(display, (x, y), (x+SQUARE_SIZE, y+SQUARE_SIZE), (0, 255, 0), 2)
                cv2.putText(display, f"{label}", (x+5, y+20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.putText(display, square, (x+5, y+SQUARE_SIZE-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    # === Movement Detection ===
    if hand_safe_to_track:
        if last_grid is not None and new_grid != last_grid:
            if change_start is None:
                change_start = now
            elif now - change_start >= 2:
                print("♟️ Detected Move:")
                for i in range(GRID_ROWS):
                    for j in range(GRID_COLS):
                        prev = last_grid[i][j]
                        curr = new_grid[i][j]
                        if prev and not curr:
                            print(f"❌ Removed {prev} from {square_names[i][j]}")
                        elif not prev and curr:
                            print(f"✅ Placed  {curr} at {square_names[i][j]}")
                last_grid = [row[:] for row in new_grid]
                change_start = None
        else:
            change_start = None
    else:
        change_start = None  # Reset if hand is present or in cooldown

    if last_grid is None:
        last_grid = [row[:] for row in new_grid]

    # === Draw Calibration Box and Grid ===
    cv2.rectangle(display, (CALIB_BOX[0], CALIB_BOX[1]),
                  (CALIB_BOX[0]+CALIB_BOX[2], CALIB_BOX[1]+CALIB_BOX[3]),
                  (255, 0, 0), 1)
    for i in range(1, GRID_COLS):
        cv2.line(display, (i*SQUARE_SIZE, 0), (i*SQUARE_SIZE, CROP_SIZE), (200, 200, 200), 1)
    for j in range(1, GRID_ROWS):
        cv2.line(display, (0, j*SQUARE_SIZE), (CROP_SIZE, j*SQUARE_SIZE), (200, 200, 200), 1)

    cv2.imshow("Combined Brown & Black Detector", display)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        rgb_gain = calibrate_white_balance(warped, CALIB_BOX)
        print(f"🎛️ Calibrated RGB gain: {rgb_gain.round(2)}")

cv2.destroyAllWindows()
picam2.stop()

