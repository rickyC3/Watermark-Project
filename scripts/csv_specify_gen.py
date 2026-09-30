from pathlib import Path
import pandas as pd
from PIL import Image
import csv

folder1_path = r"D:\Ricky\NTHU\Project_DM\print-cam\image\noise_data\noise_image"
folder2_path = r"D:\Ricky\NTHU\Project_DM\print-cam\image\test_img_noise_add_noise"

csv_table_path = r"D:\Ricky\NTHU\Project_DM\print-cam\image\noise_data\Noise_pc_pred&noise.csv"

folder1 = Path(folder1_path)
folder2 = Path(folder2_path)
save_path = Path(csv_table_path)



file_list1 = sorted(list(folder1.glob("*_noise.png")))
file_list2 = sorted(list(folder2.glob("*.png")))
gen_files = []

if (len(file_list1) != len(file_list2)):
    raise ValueError(f"2 folder has diff. file counts: {len(file_list1)}, {len(file_list2)}")

save_path.parent.mkdir(parents=True, exist_ok=True)

with save_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)

    # Header
    writer.writerow(["img1", "img2"])

    # 依照相同順序配對
    for img1, img2 in zip(file_list1, file_list2):
        writer.writerow([str(img1), str(img2)])

print(f"完成！共建立 {len(file_list1)} 個 pairs")
print(f"CSV: {save_path}")
print()