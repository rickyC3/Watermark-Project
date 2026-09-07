import os
from PIL import Image

# =========================
# 設定
# =========================
INPUT_DIR = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\coco_test_144_original"      # 原始圖片資料夾
OUTPUT_DIR = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\coco_test_400"   # 裁切後圖片資料夾

CROP_SIZE = 400


def center_crop(image, crop_size):
    """以圖片中心為基準裁切 crop_size x crop_size"""
    width, height = image.size

    if width < crop_size or height < crop_size:
        return None

    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    return image.crop((left, top, right, bottom))


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    files = [
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(valid_extensions)
    ]

    print(f"找到 {len(files)} 張圖片")

    for filename in files:
        input_path = os.path.join(INPUT_DIR, filename)
        save_file_name = filename[:-3] + 'png'
        try:
            with Image.open(input_path) as image:
                original_size = image.size

                cropped = center_crop(image, CROP_SIZE)

                if cropped is None:
                    print(
                        f"[SKIP] {filename}: "
                        f"原始大小 {original_size} 小於 {CROP_SIZE}x{CROP_SIZE}"
                    )
                    continue

                output_path = os.path.join(OUTPUT_DIR, save_file_name)

                cropped.save(output_path)

                print(
                    f"[OK] {filename}: "
                    f"{original_size} -> {cropped.size}"
                )

        except Exception as e:
            print(f"[ERROR] {filename}: {e}")

    print("\n處理完成！")
    print(f"輸出資料夾：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()