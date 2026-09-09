from pathlib import Path
import pandas as pd
from PIL import Image

csv_table_path = r"D:\Ricky\NTHU\Project_DM\print-cam\image\Noise_image\PC200_orginal_pc_noise.csv"
store_dir = r"D:\Ricky\NTHU\Project_DM\print-cam\image\Noise_image\noise_image"

save_path = Path(store_dir)
save_path.mkdir(parents=True, exist_ok=True)

file_list = list(save_path.glob("*.png"))
gen_files = []

df = pd.read_csv(csv_table_path)

for idx, row in df.iterrows(): # 迭代 row
    img1_path = row["img1"]
    img2_path = row["img2"]
    save_name = Path(img1_path).stem

    img1 = Image.open(img1_path).convert("RGB")
    img2 = Image.open(img2_path).convert("RGB")

    # setting save file name
    
    save_original_path = save_path / (save_name + "_original.png")
    save_noise_path = save_path / (save_name + "_noise.png")

    while True:
        if ((save_original_path in file_list or save_noise_path in file_list) and 
                (save_original_path in gen_files or save_noise_path in gen_files)):
            save_noise_path = "1_" + save_noise_path
            save_original_path = "1_" + save_original_path
        else:
            gen_files.append(save_original_path)
            gen_files.append(save_noise_path)
            break

    img1.save(save_original_path, "PNG")
    img2.save(save_noise_path, "PNG")

    print(f"\rfinish {idx}/{len(df)}", end="", flush=True)
print()