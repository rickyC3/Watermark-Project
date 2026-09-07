import numpy as np
import cv2

# 1. 建立一個 4x4 的全灰像素區塊，數值全部都是 120（型態是 float32，模擬頻域運算時的浮點數環境）
block_float = np.full((4, 4), 120.0, dtype=np.float32)

print("=== 1. 原始浮點數區塊 ===")
print(block_float)

# 2. 對它做正向 DCT 轉換
dct_block = cv2.dct(block_float)
print("\n=== 2. 原始 DCT 矩陣 ===")
# 此時除了 (0,0) 的直流成分外，其他高頻中頻（包含 (2,2)）都應該是完美的 0.0
print(np.round(dct_block, 2))

# 3. 模擬你的演算法：我們「只」把 (2, 2) 這一格加上 4.5（故意帶有小數點！）
dct_block[2, 2] = 4.5
print("\n=== 3. 修改後的 DCT 矩陣（(2,2) 被改成了 4.5） ===")
print(np.round(dct_block, 2))

# 4. 做反向 IDCT 轉換回空間域
block_idct = cv2.idct(dct_block)
print("\n=== 4. IDCT 轉回來的浮點數像素（帶有精準的小數點 121.12 與 118.88！） ===")
# 根據公式，4.5 * 0.25 = 1.125。所以有的點是 120 + 1.125 = 121.125，有的是 120 - 1.125 = 118.875
print(block_idct)

# 5. 關鍵一步：模擬 OpenCV 存檔成圖片的 uint8 (0~255) 格式
# 電腦在存成 PNG 時，會強行將數值轉換成無號整數，小數點直接被截斷或四捨五入
block_uint8 = np.clip(block_idct, 0, 255).astype(np.uint8)
print("\n=== 5. 存檔成圖片後的真實 uint8 整數像素（小數點全部不見了！） ===")
print(block_uint8)

# 6. 模擬解密：把這張「整數圖片」讀進來，重新做 DCT
dct_recovered = cv2.dct(block_uint8.astype(np.float32))
print("\n=== 6. 重新做 DCT 後，(2,2) 讀回來的數值 ===")
print(f"原本寫入: 4.5")
print(f"實際讀回: {dct_recovered[2, 2]:.2f}  <-- 發現了嗎？它變成了完美的 4.00！")
print(f"變化量 (Delta): {dct_recovered[2, 2] - 0.0:.2f}")