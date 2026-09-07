import os
import sys
import cv2

# 將上一層目錄加入 Python 的搜尋路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imwatermark import WatermarkEncoder

print("開始進行浮水印加密...")

# 讀取同一個資料夾底下的原始圖片
bgr = cv2.imread('test_gray.png')

# 設定你的秘密文字
secret_msg = 'nthueecs2026'
wm_bytes = secret_msg.encode('utf-8')

# 召喚加密機
encoder = WatermarkEncoder()
encoder.set_watermark('bytes', wm_bytes)

# 啟動 DWT-DCT 演算法進行加工
bgr_encoded = encoder.encode(bgr, 'dwtDct', scales=[5, 0, 0])

# 將加工後的圖片存放在同一個資料夾底下
cv2.imwrite('output_gray_5.png', bgr_encoded)

print(f"加密成功！密碼 '{secret_msg}' (長度: {encoder.get_length()} bits) 已藏入 output_gray_2026.png 中。")