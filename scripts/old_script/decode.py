import os
import sys
import cv2

# 將上一層目錄加入 Python 的搜尋路徑中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from imwatermark import WatermarkDecoder

print("開始進行浮水印解密...")

# 讀取同一個資料夾底下有藏秘密的圖片
bgr = cv2.imread(r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\image10_out.png")

# 設定預期要挖出來的長度 (Secret123 是 9 個英文字母，9 * 8 = 72 bits)
watermark_length = 72

# 召喚解密機
decoder = WatermarkDecoder('bytes', watermark_length)

# 啟動 DWT-DCT 演算法把秘密挖出來
wm_bytes = decoder.decode(bgr, 'dwtDct', scales=[0, 36, 36])

# 把挖出來的 bytes 翻譯回人類看得懂的文字
decoded_msg = wm_bytes.decode('utf-8')

print("-" * 30)
print(f"成功解出浮水印： {decoded_msg}")
print("-" * 30)