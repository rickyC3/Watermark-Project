import cv2
import numpy as np
import pywt
import os

# 設定 Numpy 顯示格式，讓矩陣對齊、小數點取兩位，方便閱讀
np.set_printoptions(formatter={'float': '{: 8.2f}'.format}, linewidth=150)

def print_top_blocks(orig_path, wm_path):
    if not os.path.exists(orig_path) or not os.path.exists(wm_path):
        print("找不到圖片，請確認檔案路徑。")
        return

    # 1. 讀取圖片並轉為 Y 頻道 (float32)
    img_orig = cv2.imread(orig_path)
    img_wm = cv2.imread(wm_path)
    y_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    y_wm = cv2.cvtColor(img_wm, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)

    # 2. 進行一階 DWT 轉換
    cA_orig, _ = pywt.dwt2(y_orig, 'haar')
    cA_wm, _ = pywt.dwt2(y_wm, 'haar')

    block_size = 4
    
    print(f"分析圖片: {orig_path} vs {wm_path}")
    print("=" * 80)

    # 取左上角前 4 個區塊 (座標 j=0 到 j=3)
    for j in range(4):
        # 提取 4x4 區塊
        block_orig = cA_orig[0:block_size, j*block_size:(j+1)*block_size]
        block_wm = cA_wm[0:block_size, j*block_size:(j+1)*block_size]

        # 進行 DCT 轉換
        dct_orig = cv2.dct(block_orig)
        dct_wm = cv2.dct(block_wm)

        # 判斷這屬於 Block A 還是 Block B
        pair_name = "Block A" if j % 2 == 0 else "Block B"
        pair_index = j // 2

        print(f"【 第 {pair_index} 對區塊 - {pair_name} | 座標 (0, {j}) 】")
        print("-" * 40)
        print(">>> 原圖 DCT 矩陣:")
        print(dct_orig)
        print("\n>>> 加密後 DCT 矩陣:")
        print(dct_wm)
        print("-" * 40)
        
        # 特別標示 [2, 2] 係數的變化
        val_orig = dct_orig[2, 2]
        val_wm = dct_wm[2, 2]
        print(f"鎖定中頻點 [2, 2] 數值變化: {val_orig:8.2f}  ->  {val_wm:8.2f}  (Delta: {val_wm - val_orig:8.2f})")
        print("=" * 80)

if __name__ == '__main__':
    # 請根據你的實際檔名修改
    print_top_blocks('test_gray.png', 'output_gray_2026.png')