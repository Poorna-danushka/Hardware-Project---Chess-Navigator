import cv2
import numpy as np
import os

# === CONFIGURATION ===
IMAGE_FOLDER = "/home/pi/Desktop/ch/claibrate/brown_samples"  # Folder with brown piece samples
SAVE_PATH = "/home/pi/board/brown_rgb_model.npz"
CROP_SIZE = 30  # Size of center crop
all_samples = []

# === Get image files ===
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.lower().endswith(('.jpg', '.png'))
])

if not image_files:
    raise RuntimeError(f"❌ No images found in {IMAGE_FOLDER}")

print(f"📂 Found {len(image_files)} brown piece images.")

# === Extract RGB patches from each image ===
for filename in image_files:
    path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(path)

    if img is None:
        print(f"⚠️ Skipped unreadable image: {filename}")
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    cx, cy = w // 2, h // 2

    x1 = max(0, cx - CROP_SIZE // 2)
    y1 = max(0, cy - CROP_SIZE // 2)
    x2 = min(w, cx + CROP_SIZE // 2)
    y2 = min(h, cy + CROP_SIZE // 2)

    patch = rgb[y1:y2, x1:x2]
    if patch.shape[0] == CROP_SIZE and patch.shape[1] == CROP_SIZE:
        all_samples.append(patch.reshape(-1, 3))

if not all_samples:
    raise RuntimeError("❌ No valid color patches found in images.")

# === Compute RGB color model ===
samples = np.vstack(all_samples)
mean_rgb = np.mean(samples, axis=0)
std_rgb = np.std(samples, axis=0)

lower_rgb = np.clip(mean_rgb - 1.5 * std_rgb, 0, 255).astype(np.uint8)
upper_rgb = np.clip(mean_rgb + 1.5 * std_rgb, 0, 255).astype(np.uint8)

# === Save model ===
os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
np.savez(SAVE_PATH, lower=lower_rgb, upper=upper_rgb)

# === Report ===
print("\n✅ Brown Piece RGB Calibration Completed!")
print("Lower RGB:", lower_rgb)
print("Upper RGB:", upper_rgb)
print(f"💾 Model saved to: {SAVE_PATH}")
