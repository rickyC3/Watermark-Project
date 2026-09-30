
from pathlib import Path
import csv


def generate_csv(folder1, folder2, output_csv):
    folder1 = Path(folder1)
    folder2 = Path(folder2)
    output_csv = Path(output_csv)

    # 取得兩個資料夾中的檔案，並依照檔名字串排序
    files1 = sorted(
        [p for p in folder1.iterdir() if p.is_file()],
        key=lambda p: p.name
    )

    files2 = sorted(
        [p for p in folder2.iterdir() if p.is_file()],
        key=lambda p: p.name
    )

    # 確認兩個資料夾檔案數量相同
    if len(files1) != len(files2):
        raise ValueError(
            f"檔案數量不同：folder1 = {len(files1)}, "
            f"folder2 = {len(files2)}"
        )

    # 建立輸出資料夾
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # 寫入 CSV
    with output_csv.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["img1", "img2"])

        # 依照相同順序配對
        for img1, img2 in zip(files1, files2):
            writer.writerow([str(img1), str(img2)])

    print(f"完成！共建立 {len(files1)} 個 pairs")
    print(f"CSV：{output_csv}")


# ============================================================
# 使用設定
# ============================================================
# original D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\encoded_images_50
# ymck D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\YMCK_softproof
# pc:  D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\5cmx50_pc_aligned_images
# pspc: D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\5cmx50_pspc_aligned_images
folder1 = r"D:\Ricky\NTHU\Project_DM\print-cam\image\sharpen2\aligned_images_sharpen2_pc"

folder2 = r"D:\Ricky\NTHU\Project_DM\print-cam\image\sharpen2\aligned_images_sharpen2_pspc"

output_csv = r"D:\Ricky\NTHU\Project_DM\print-cam\image\noise_data_0917\sharpen2_pc_pspc.csv"


generate_csv(folder1, folder2, output_csv)

