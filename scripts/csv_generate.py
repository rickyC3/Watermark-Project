from pathlib import Path
import csv

input_dir = Path(r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\encoded_images_50")
output_dir = Path(r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\YMCK_softproof")
csv_path = output_dir / "cmyk_pairs.csv"

filenames = [
    "001818_ca1_22_404040.png",
    "002923_ca1_22_404040.png",
    "003553_ca1_22_404040.png",
    "005477_ca1_22_404040.png",
    "006763_ca1_22_404040.png",
    "008277_ca1_22_404040.png",
    "015956_ca1_22_404040.png",
    "017178_ca1_22_404040.png",
    "023023_ca1_22_404040.png",
    "025603_ca1_22_404040.png",
    "039785_ca1_22_404040.png",
    "044590_ca1_22_404040.png",
    "051712_ca1_22_404040.png",
    "055002_ca1_22_404040.png",
    "062692_ca1_22_404040.png",
    "070229_ca1_22_404040.png",
    "078565_ca1_22_404040.png",
    "079014_ca1_22_404040.png",
    "079034_ca1_22_404040.png",
    "081766_ca1_22_404040.png",
    "085157_ca1_22_404040.png",
    "089648_ca1_22_404040.png",
    "090155_ca1_22_404040.png",
    "097022_ca1_22_404040.png",
    "097278_ca1_22_404040.png",
    "103723_ca1_22_404040.png",
    "106563_ca1_22_404040.png",
    "109916_ca1_22_404040.png",
    "110211_ca1_22_404040.png",
    "110884_ca1_22_404040.png",
    "114871_ca1_22_404040.png",
    "122927_ca1_22_404040.png",
    "126110_ca1_22_404040.png",
    "127135_ca1_22_404040.png",
    "131444_ca1_22_404040.png",
    "137246_ca1_22_404040.png",
    "137950_ca1_22_404040.png",
    "142472_ca1_22_404040.png",
    "150224_ca1_22_404040.png",
    "151516_ca1_22_404040.png",
    "154431_ca1_22_404040.png",
    "155051_ca1_22_404040.png",
    "156924_ca1_22_404040.png",
    "157365_ca1_22_404040.png",
    "170099_ca1_22_404040.png",
    "170893_ca1_22_404040.png",
    "176232_ca1_22_404040.png",
    "182155_ca1_22_404040.png",
    "182611_ca1_22_404040.png",
    "185250_ca1_22_404040.png",
]

output_dir.mkdir(parents=True, exist_ok=True)

rows = []
for filename in filenames:
    input_path = input_dir / filename
    output_path = output_dir / f"{Path(filename).stem}cmyk{Path(filename).suffix}"
    rows.append([str(input_path), str(output_path)])

with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.writer(f)
    writer.writerow(["img1", "img2"])
    writer.writerows(rows)

print(f"完成：{len(rows)} 筆")
print(f"CSV：{csv_path}")
print(f"輸出資料夾：{output_dir}")
