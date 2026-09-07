import os
import sys
import cv2

# 將上一層目錄加入 Python 的搜尋路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imwatermark import WatermarkEncoder

print("開始進行浮水印加密...")

# 讀取同一個資料夾底下的原始圖片
input_path = r'D:\Ricky\program\invisible-watermark\invisible-watermark\images\image9.png'
bgr = cv2.imread(input_path)

# 設定你的秘密文字
secret_msg = 'nthueecs2026'
wm_bytes = secret_msg.encode('utf-8')

# 召喚加密機
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm_bytes)

# # 啟動 DWT-DCT 演算法進行加工
# bgr_encoded = encoder.encode(bgr, 'dwtDct', scales=[0, 20, 20])

# 啟動 DWT-DCT 演算法進行加工
# 這裡加入我們剛剛新增的 dwt_band 和 dct_coord 參數
bgr_encoded = encoder.encode(
    bgr, 
    'dwtDct', 
    scales = [0, 36, 36],       # 建議把強度調到 36 比較耐打
    dwt_band = 'ca1',           # ★ 可以在這裡改成 'h1', 'v1'
    dct_coord = (2, 2),          # ★ 可以在這裡改成 (2, 2), (3, 3)
    set_channel = "yuv",
    dwt_domain = "LL",
)
"""
        -------------------
        |        |        |
        | cA(LL) | cH(LH) |
        |        |        |
        -------------------
        |        |        |
        | cV(HL) | cD(HH) |
        |        |        |
        -------------------
"""
# 將加工後的圖片存放在同一個資料夾底下
output_path = r'D:\Ricky\program\invisible-watermark\invisible-watermark\images\image9_out.png'
cv2.imwrite(output_path, bgr_encoded)

print(f"加密成功！密碼 '{secret_msg}' (長度: {encoder.get_length()} bits) 已藏入 {output_path} 中。")
