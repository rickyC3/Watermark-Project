import cv2
from pathlib import Path
import numpy as np


# =========================
# Path settings
# =========================

imgs_fold = r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\YMCK_softproof"

output_fold = r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\Noise_Break"

input_dir = Path(imgs_fold)
output_dir = Path(output_fold)

output_dir.mkdir(parents=True, exist_ok=True)

imgs_list = list(input_dir.glob("*.png"))


# =========================
# Gaussian noise settings
# =========================

# Y channel
noise_mean_Y = -0.982731
noise_std_Y = 23.335096


# =========================
# Add Gaussian noise
# =========================

for idx, f in enumerate(imgs_list):

    # -------------------------
    # Read image
    # -------------------------
    img = cv2.imread(str(f), cv2.IMREAD_COLOR)

    if img is None:
        print(f"Failed to read: {f}")
        continue

    # -------------------------
    # BGR -> YUV
    # -------------------------
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # uint8 -> float32
    img_yuv = img_yuv.astype(np.float32)


    # -------------------------
    # Generate Y-channel noise
    # -------------------------
    noise_Y = np.random.normal(
        loc=noise_mean_Y,
        scale=noise_std_Y,
        size=img_yuv[:, :, 0].shape
    ).astype(np.float32)


    # -------------------------
    # Add noise only to Y
    # -------------------------
    img_yuv[:, :, 0] += noise_Y


    # -------------------------
    # Clip to valid range
    # -------------------------
    img_yuv = np.clip(img_yuv, 0, 255)


    # -------------------------
    # float32 -> uint8
    # -------------------------
    img_yuv = img_yuv.astype(np.uint8)


    # -------------------------
    # YUV -> BGR
    # -------------------------
    noisy_img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)


    # -------------------------
    # Output path
    # -------------------------
    output_path = output_dir / (f.stem + "_noisebreak.png")


    # -------------------------
    # Save
    # -------------------------
    cv2.imwrite(str(output_path), noisy_img)

    print(f"[{idx + 1}/{len(imgs_list)}] Saved: {output_path}")


print("Done!")