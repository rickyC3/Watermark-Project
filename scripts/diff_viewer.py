import cv2
import numpy as np
import pywt
import os

def analyze_diff(orig_path, wm_path, block_size=4, freq_pos=(2, 2)):
    print(f"開始透視分析: {orig_path} vs {wm_path}")
    print("-" * 60)

    if not os.path.exists(orig_path) or not os.path.exists(wm_path):
        print("錯誤：找不到圖片，請確認檔案路徑。")
        return

    img_orig = cv2.imread(orig_path)
    img_wm = cv2.imread(wm_path)

    if img_orig.shape != img_wm.shape:
        print("錯誤：兩張圖片尺寸不同，無法進行精準比對。")
        return

    # 1. 產生空間域（像素層面）的誤差放大圖
    # 將兩張圖相減取絕對值，並將微小的差異放大 15 倍以利肉眼觀察
    diff_pixel = cv2.absdiff(img_orig, img_wm)
    diff_magnified = cv2.convertScaleAbs(diff_pixel, alpha=15.0)
    cv2.imwrite('diff_magnified.png', diff_magnified)
    print("已生成空間域像素差異圖: diff_magnified.png (誤差已放大 15 倍)")

    # 2. 準備進入頻域分析
    y_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    y_wm = cv2.cvtColor(img_wm, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)

    # 執行一階 DWT 取得低頻 cA 矩陣
    cA_orig, _ = pywt.dwt2(y_orig, 'haar')
    cA_wm, _ = pywt.dwt2(y_wm, 'haar')

    h, w = cA_orig.shape
    r, c = freq_pos

    print("\nDCT 中頻係數 [2, 2] 變化分析 (僅列出前 10 組有變動的區塊):")
    print(f"{'區塊座標(Y, X)':<15} | {'原圖係數':<12} | {'加密後係數':<12} | {'變化量 (Delta)':<12}")
    print("-" * 60)

    diff_count = 0
    total_blocks = 0
    sum_delta = 0.0

    # 3. 遍歷所有 4x4 區塊，計算 DCT 並比較特定係數
    for i in range(h // block_size):
        for j in range(w // block_size):
            total_blocks += 1
            
            block_orig = cA_orig[i*block_size : i*block_size+block_size, 
                                 j*block_size : j*block_size+block_size]
            block_wm = cA_wm[i*block_size : i*block_size+block_size, 
                               j*block_size : j*block_size+block_size]

            dct_orig = cv2.dct(block_orig)
            dct_wm = cv2.dct(block_wm)

            val_orig = dct_orig[r, c]
            val_wm = dct_wm[r, c]

            # 計算係數差異
            delta = val_wm - val_orig

            # 設定閾值，若差異大於 0.1 即視為該區塊被修改過
            if abs(delta) > 0.1:
                sum_delta += abs(delta)
                if diff_count < 10:
                    print(f"Block ({i:02d}, {j:02d})   | {val_orig:>10.2f} | {val_wm:>10.2f} | {delta:>10.2f}")
                diff_count += 1

    print("-" * 60)
    print(f"總區塊數量: {total_blocks}")
    print(f"實際被修改的區塊數量: {diff_count}")
    
    if diff_count > 0:
        avg_change = sum_delta / diff_count
        print(f"平均修改幅度: {avg_change:.2f}")
        print("\n提示：如果「平均修改幅度」遠低於你設定的 scale，代表受到 IDWT 稀釋或影像存檔截斷的嚴重影響。")

if __name__ == '__main__':
    # 執行分析，請將檔名替換為你實際使用的圖片名稱
    analyze_diff('test_gray.png', 'output_gray_5_print2.png')