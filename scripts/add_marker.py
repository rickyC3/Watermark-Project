import cv2
import numpy as np
import os  # ★ 新增：匯入 os 模組，用來處理檔案路徑與擷取檔名
from pathlib import Path
import glob


def add_markers_to_image(image_path, output_path):
    # 1. 讀取你已經加好浮水印的圖片[cite: 6]
    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    # 2. 設定白邊的寬度(120 像素)與定位點的大小(邊長為 80 像素)[cite: 6]
    border_size = 120
    marker_size = 80

    # 3. 幫圖片加上白邊 (純白背景)[cite: 6]
    canvas = cv2.copyMakeBorder(img, border_size, border_size, border_size, border_size, 
                                cv2.BORDER_CONSTANT, value=[255, 255, 255])

    # 4. 準備 ArUco 字典 (這裡選用 DICT_4X4_50，最簡單快速)[cite: 6]
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # 5. 在四個角落畫上編號 0, 1, 2, 3 的定位點[cite: 6]
    corners = [
        (20, 20),                                      # 左上 (ID 0)[cite: 6]
        (w + border_size + 20, 20),                    # 右上 (ID 1)[cite: 6]
        (w + border_size + 20, h + border_size + 20),  # 右下 (ID 2)[cite: 6]
        (20, h + border_size + 20)                     # 左下 (ID 3)[cite: 6]
    ]

    for i in range(4):
        # 生成標記圖並轉為 3 通道[cite: 6]
        marker_gray = cv2.aruco.generateImageMarker(aruco_dict, i, marker_size)
        marker_img = cv2.cvtColor(marker_gray, cv2.COLOR_GRAY2BGR)
        
        # 貼上畫布[cite: 6]
        x, y = corners[i]
        canvas[y:y+marker_size, x:x+marker_size] = marker_img

    # ==========================================
    # ★ 6. 新增步驟：在上方空白處置中印上檔名 ★
    # ==========================================
    
    # 6-1. 從 output_path 擷取檔名 (去掉資料夾路徑與 .png 副檔名)
    # 例如 'pic_v2/m_output_ca1_22_10.png' 會變成 'm_output_ca1_22_10'
    filename = os.path.splitext(os.path.basename(output_path))[0]

    print_txt = filename[:-15]
    
    # 6-2. 設定 OpenCV 文字參數
    font = cv2.FONT_HERSHEY_COMPLEX  # 選擇帶有襯線的字體，比較接近你的參考圖
    font_scale = 1.0                 # 字體大小
    thickness = 2                    # 字體粗細
    text_color = (0, 0, 0)           # 黑色字體 (BGR 格式)
    
    # 6-3. 計算文字的寬高，用來精準置中
    (text_w, text_h), baseline = cv2.getTextSize(print_txt, font, font_scale, thickness)
    
    # 6-4. 計算文字的 X 與 Y 座標
    # X 座標：(整張畫布寬度 - 文字寬度) / 2，達成水平置中
    text_x = (canvas.shape[1] - text_w) // 2
    
    # Y 座標：定位點的垂直中心是 20 + (80/2) = 60，再加上文字高度的一半來達成垂直置中
    text_y = 60 + (text_h // 2)
    
    # 6-5. 將文字印在白邊畫布上，並加上 cv2.LINE_AA 讓邊緣平滑抗鋸齒
    cv2.putText(canvas, print_txt, (text_x, text_y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    # ==========================================

    # 7. 存檔，這張就是你要拿去印出來的最終版本[cite: 6]
    cv2.imwrite(output_path, canvas)
    print(f"已成功加上定位點與檔名並儲存為 {output_path}")

if __name__ == '__main__':
    input_folder = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\print_cam_1\single imgs_aligned"
    output_folder = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\COCO_print_cam_marker"  # 可以改成任何你想要的資料夾

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(glob.glob(f"{input_folder}/*"))

    for f in files:
        try:
            # 取得原檔名（含副檔名）
            basename = os.path.basename(f)                    # 例如：000000001.jpg
            name, ext = os.path.splitext(basename)            # name = 000000001, ext = .jpg

            # 新檔名 = 原檔名 + add_marker
            new_filename = f"{name}_add_marker{ext}"          # 例如：000000001_add_marker.jpg

            # 完整輸出路徑
            output_path = os.path.join(output_folder, new_filename)

            # 呼叫函式（第二個參數改成完整路徑）
            add_markers_to_image(f, output_path)

        except Exception as e:
            print(f"Skip: {f}  →  {e}")

    