from PIL import Image, ImageOps
import glob
import math

# =========================
# 設定
# =========================
input_folder = r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\COCO_print_cam_marker"
output_pdf = "output_stegastamp_print_cam.pdf"

# A4 @ 300 DPI
DPI = 300 # PPI pixel per ft(英吋)
# 330
A4_W = int(8.27 * DPI) 
A4_H = int(11.69 * DPI)

# 一頁幾張
COLS = 4
ROWS = 6

# 圖片之間的間距
# 0.066 英吋 -> 2 cm
GAP = int(0.4 * DPI)

# A4 邊界
MARGIN = int(0.2 * DPI)


# =========================
# 讀取圖片
# =========================
files = sorted(glob.glob(f"{input_folder}/*"))

images = []

for f in files:
    try:
        img = Image.open(f).convert("RGB")
        images.append(img)
    except:
        print(f"Skip: {f}")

# =========================
# 計算每張圖片可以放多大
# =========================
cell_w = (
    A4_W
    - 2 * MARGIN
    - (COLS - 1) * GAP
) // COLS

cell_h = (
    A4_H
    - 2 * MARGIN
    - (ROWS - 1) * GAP
) // ROWS

# =========================
# 建立 PDF 頁面
# =========================
pages = []

images_per_page = COLS * ROWS

for page_start in range(0, len(images), images_per_page):

    page = Image.new(
        "RGB",
        (A4_W, A4_H),
        "white"
    )

    batch = images[
        page_start:
        page_start + images_per_page
    ]

    for i, img in enumerate(batch):

        # 等比例縮放
        # 計算等比例縮放比例（可放大也可縮小）
        width_ratio = cell_w / img.width
        height_ratio = cell_h / img.height
        scale = min(width_ratio, height_ratio)   # 取較小的，確保能完整放進格子

        new_w = int(img.width * scale)
        new_h = int(img.height * scale)

        img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        #img.thumbnail((cell_w, cell_h), Image.Resampling.LANCZOS)

        # 該格的位置
        row = i // COLS
        col = i % COLS

        # x0 = MARGIN + col * (cell_w + GAP)
        # y0 = MARGIN + row * (cell_h + GAP)

        # # 置中
        # x = x0 + (cell_w - img.width) // 2
        # y = y0 + (cell_h - img.height) // 2

        x0 = MARGIN + col * (new_w + GAP)
        y0 = MARGIN + row * (new_h + GAP)

        # 置中
        x = x0
        y = y0

        page.paste(img, (x, y))

    pages.append(page)

# =========================
# 輸出 PDF
# =========================
if pages:
    pages[0].save(
        output_pdf,
        save_all=True,
        append_images=pages[1:],
        resolution=DPI
    )

print(f"cell_w length: {cell_w / DPI:.4f}, cell_h length: {cell_h / DPI:.4f}")
print(f"完成：{output_pdf}")