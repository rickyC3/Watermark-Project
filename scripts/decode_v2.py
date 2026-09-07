import os
import sys
import cv2
import numpy as np

# 將上一層目錄加入 Python 的搜尋路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imwatermark import WatermarkDecoder

def bytes_to_bits(byte_data):
    """將 bytes 轉換為 0 和 1 的整數陣列"""
    bits = []
    for b in byte_data:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    return np.array(bits)

print("開始進行浮水印解密與位元比對...")

# 1. 讀取要檢測的圖片 (這裡以 test.png 原圖為例)
input_path = r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\image9_out.png"
bgr = cv2.imread(input_path)

# 2. 設定標準答案 (Ground Truth)
secret_msg = 'nthueecs2026'
original_bytes = secret_msg.encode('utf-8')
# 將標準密碼轉換成 0 和 1 的陣列
original_bits = bytes_to_bits(original_bytes)
watermark_length = len(original_bits) 

# 3. 召喚解密機
decoder = WatermarkDecoder('bytes', watermark_length)

# # 4. 啟動解密 (注意：這裡保留了先前針對灰階/Y頻道的設定 scales=[10, 0, 0])
# wm_bytes = decoder.decode(bgr, 'dwtDct', scales=[0, 36, 36])

# 4. 啟動解密
# 必須傳入跟加密時一模一樣的 dwt_band 和 dct_coord
wm_bytes = decoder.decode(
    bgr, 
    'dwtDct', 
    scales=[0, 36, 36], 
    dwt_band='v1',           # ★ 要跟加密時一致
    dct_coord=(2, 2),          # ★ 要跟加密時一致
    set_channel = "yuv",
    dwt_domain = "LL",
)

# 5. 將挖出來的未知資料，同樣轉換成 0 和 1 的陣列
extracted_bits = bytes_to_bits(wm_bytes)


# 6. 計算準確率：把兩個陣列疊在一起，計算完全相同的比例
accuracy = np.mean(original_bits == extracted_bits) * 100

print("-" * 40)
print(f"檢測檔案: {input_path}")
print(f"檢測目標: {secret_msg}")
print(f"位元準確率 (BAR): {accuracy:.2f}%")


# 使用 errors='replace'，遇到解不出來的 byte 會自動跳過並填入官方未知符號
decoded_msg = wm_bytes.decode('utf-8', errors='replace')

# 使用 \ufffd (官方未知符號的底層編碼) 來進行替換，確保不會切碎原本的字串
decoded_msg = decoded_msg.replace('\ufffd', '?')

print(f"強制文字翻譯: {decoded_msg}")

print("-" * 40)