import cv2
import numpy as np
import os  # ★ 新增：匯入 os 模組，用來處理檔案路徑與擷取檔名
import glob



def auto_align_photo(photo_path, original_w, original_h, output_path):
    # 1. 讀取你用手機拍的照片
    photo = cv2.imread(photo_path)
    
    # 2. 準備尋找 ArUco 標記
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # 3. 偵測照片中的標記
    corners, ids, rejected = detector.detectMarkers(photo)

    if ids is None or len(ids) != 4:
        print("錯誤：無法在照片中找到完整的 4 個定位點，請確保沒有反光或裁切到邊緣。")
        return


    # 4. 算出 4 個定位點的「中心點座標」
    center_points = {}
    
    # 將 ids 強制攤平成 1D 陣列，避免版本差異引發錯誤
    flat_ids = ids.flatten() 
    
    for i in range(len(flat_ids)):
        marker_id = int(flat_ids[i])
        # corners[i][0] 包含了一個標記的四個角座標，取平均得到中心點
        center_x = np.mean(corners[i][0][:, 0])
        center_y = np.mean(corners[i][0][:, 1])
        center_points[marker_id] = [center_x, center_y]

    # 5. 確保順序是 左上(0), 右上(1), 右下(2), 左下(3)
    src_points = np.float32([
        center_points[0], 
        center_points[1], 
        center_points[2], 
        center_points[3]
    ])

    # 6. 設定目標座標 (我們當初把標記放在白邊上的位置)
    border_size = 120
    marker_offset = 20 + 40 - 0.5 # 20是邊距，40是標記大小的一半(中心點)
    
    dst_points = np.float32([
        [marker_offset, marker_offset],
        [original_w + border_size*2 - marker_offset, marker_offset],
        [original_w + border_size*2 - marker_offset, original_h + border_size*2 - marker_offset],
        [marker_offset, original_h + border_size*2 - marker_offset]
    ])

    # 7. 計算透視變換矩陣 (Perspective Transform Matrix)
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # 8. 執行拉直與變形還原
    canvas_w = original_w + border_size * 2
    canvas_h = original_h + border_size * 2
    aligned_img = cv2.warpPerspective(photo, matrix, (canvas_w, canvas_h))

    # 9. 把外圍白邊切掉，只留下原本乾淨的圖片區域
    final_img = aligned_img[border_size : border_size+original_h, 
                            border_size : border_size+original_w]

    # 10. 存檔！這張圖就可以直接丟進你的 decode 程式了
    cv2.imwrite(output_path, final_img)
    print(f"校正成功！已儲存為 {output_path}")

if __name__ == '__main__':
    # 第一個參數：你用手機拍的照片檔名
    # 第二與第三個參數：你「原始」圖片的寬度與高度 (例如原本 test_gray.png 是 400x400)
    #auto_align_photo('pic_v2_2/output_d1_33_404040_m2ps2.png', 512, 512, 'pic_v2_2/output_d1_33_404040_m2ps2a.png')
    #auto_align_photo('pic_v2_2/output_d1_22_404040_mpspc.jpg', 512, 512, 'pic_v2_2/output_d1_22_404040_mpspca.png')
    #auto_align_photo('pic_v2_3/output_d1_22_404040_mps.png', 1920, 1080, 'pic_v2_3/output_d1_22_404040_mpsa.png')
    # auto_align_photo(r'D:\Ricky\program\invisible-watermark\invisible-watermark\align_image\ca5ad3b7-3c98-492c-83c8-933f165a72f4.jpg', 400, 400, 
    #                  r'D:\Ricky\program\invisible-watermark\invisible-watermark\align_image\lenna_camera_fix.png')

    input_folder = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\print_cam_print_cam\Print cam print cam_PNG"
    output_folder = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\print_cam_print_cam\Print cam print cam imgs_aligned"  # 可以改成任何你想要的資料夾

    # 確保輸出資料夾存在
    os.makedirs(output_folder, exist_ok=True)

    files = sorted(glob.glob(f"{input_folder}/*"))

    for f in files:
        try:
            # 取得原檔名（含副檔名）
            basename = os.path.basename(f)                    # 例如：000000001.jpg
            name, ext = os.path.splitext(basename)            # name = 000000001, ext = .jpg

            # 新檔名 = 原檔名 + add_marker
            new_filename = f"{name}_aligned{ext}"          # 例如：000000001_add_marker.jpg

            # 完整輸出路徑
            output_path = os.path.join(output_folder, new_filename)

            # 呼叫函式（第二個參數改成完整路徑）
            auto_align_photo(f, 400, 400, output_path)

        except Exception as e:
            print(f"Skip: {f}  →  {e}")
    