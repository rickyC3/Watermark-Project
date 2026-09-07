import os
import sys
import cv2
import numpy as np

# 將上一層目錄加入 Python 的搜尋路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def fix_bgr(img, offset, scale):
    img = img.astype(np.float32)
    img = (img - offset) / scale
    return np.clip(img, 0, 255).astype(np.uint8)

"""
def fix_yuv(img, offset, scale):
    img = img.astype(np.float32)

    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    yuv = (yuv - offset) / scale
    yuv = np.clip(yuv, 0, 255)

    return cv2.cvtColor(yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
"""

def fix_yuv(img, offset, scale):
    # 修正：先在 uint8 狀態下轉成 YUV，再轉成 float32 做數學運算
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV).astype(np.float32)
    
    yuv = (yuv - offset) / scale
    yuv = np.clip(yuv, 0, 255).astype(np.uint8)

    return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)


# 讀取圖片
input_path = r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\Christina_camera_fix.jpg"
output_path = r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\Christina_camera_fix_out"

bgr = cv2.imread(input_path)

if bgr is None:
    raise FileNotFoundError(f"Cannot read image: {input_path}")

print(bgr.shape)


print("Fix BGR")

# BGR 修正
offset_bgr = np.array([64.85, 48.16, 35.12], np.float32)
scale_bgr  = np.array([0.74, 0.73, 0.72], np.float32)
bgr_result = fix_bgr(bgr, offset_bgr, scale_bgr)

yuv_bgr = bgr.copy()

# YUV 修正
offset_yuv = np.array([45.71, 48.75, 25.82], np.float32)
scale_yuv  = np.array([0.73, 0.69, 0.72], np.float32)
yuv_result = fix_yuv(yuv_bgr, offset_yuv, scale_yuv)


# 儲存
cv2.imwrite(output_path + "_bgr.png", bgr_result)
cv2.imwrite(output_path + "_yuv.png", yuv_result)


print(f"Save image to: {output_path}")