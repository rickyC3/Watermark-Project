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

def main():
    print("啟動浮水印進階檢測系統 (滑動視窗 & 位元比對)...")

    # 1. 設定測試圖片與目標密碼
    image_path = 'output_gray.png'  # 這裡可以換成遭受攻擊(壓縮、裁切)後的圖片
    secret_msg = 'Secret123'
    
    if not os.path.exists(image_path):
        print(f"找不到圖片: {image_path}")
        return

    bgr = cv2.imread(image_path)
    (row, col, channels) = bgr.shape

    # 2. 建立標準答案的位元陣列 (Ground Truth)
    original_bytes = secret_msg.encode('utf-8')
    original_bits = bytes_to_bits(original_bytes)
    watermark_length = len(original_bits)
    
    # 初始化解密機
    decoder = WatermarkDecoder('bytes', watermark_length)

    # 3. 設定滑動視窗的範圍 (例如 -2 到 +2 像素)
    # 若懷疑位移很大，可將數值調高，但運算時間會隨之增加
    shift_range = range(-2, 3) 
    
    best_accuracy = 0.0
    best_shift = (0, 0)
    best_extracted_msg = ""

    print(f"目標位元長度: {watermark_length} bits")
    print("-" * 40)

    # 4. 開始多方位位移檢測
    for x_shift in shift_range:
        for y_shift in shift_range:
            # 建立平移矩陣
            M = np.float32([[1, 0, x_shift], [0, 1, y_shift]])
            # 使用 cv2.warpAffine 進行平移，邊界採用 BORDER_REPLICATE (複製邊緣像素) 避免產生黑邊干擾頻率
            shifted_bgr = cv2.warpAffine(bgr, M, (col, row), borderMode=cv2.BORDER_REPLICATE)

            try:
                # 從平移後的圖片抽出 bytes
                extracted_bytes = decoder.decode(shifted_bgr, 'dwtDct', scales=[10, 0, 0])
                # 將抽出的 bytes 轉換為 bits
                extracted_bits = bytes_to_bits(extracted_bytes)

                # 計算位元準確率 (Bit Accuracy Rate, BAR)
                accuracy = np.mean(original_bits == extracted_bits) * 100

                # 嘗試將解出的 bytes 轉回文字 (若位元錯亂可能會無法解碼 utf-8，所以用 try-except 包裝)
                try:
                    decoded_text = extracted_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    decoded_text = "[亂碼]"

                # 記錄最佳結果
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_shift = (x_shift, y_shift)
                    best_extracted_msg = decoded_text

            except Exception as e:
                # 忽略解密過程中的極端意外錯誤，繼續測試下一個位移
                continue

    # 5. 輸出最終檢測報告
    print(f"最佳匹配位移 (X, Y): {best_shift}")
    print(f"最佳位元準確率 (BAR): {best_accuracy:.2f}%")
    print(f"最高準確率下解出的字串: {best_extracted_msg}")
    print("-" * 40)

    # 統計學判斷標準 (一般隨機猜測準確率為 50%，通常設定 70%~75% 以上即視為含有浮水印)
    threshold = 75.0
    if best_accuracy >= threshold:
        print("檢測結論: 極高機率含有目標浮水印 (通過統計閾值)。")
    else:
        print("檢測結論: 未檢測到目標浮水印，或圖片受損過於嚴重。")

if __name__ == '__main__':
    main()