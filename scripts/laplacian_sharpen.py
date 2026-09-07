import cv2
import numpy as np
from pathlib import Path

def laplacian_sharpen(img, alpha=1.0):
    """
    Laplacian sharpening

    Parameters
    ----------
    img : np.ndarray
        uint8 image, RGB or BGR
    alpha : float
        Sharpening strength

    Returns
    -------
    sharpened : np.ndarray
        Sharpened image
    """

    # Convert to float to avoid overflow / underflow
    img_float = img.astype(np.float32)

    # Laplacian
    laplacian = cv2.Laplacian(
        img_float,
        cv2.CV_32F,
        ksize=3
    )

    # Sharpen
    sharpened = img_float - alpha * laplacian

    # Keep valid image range
    sharpened = np.clip(sharpened, 0, 255)

    return sharpened.astype(np.uint8)

img_fold = r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\encoded_images_50"
output_fold = r"D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\encoded_images_50\encoded_images_50_sharpen"

img_path = Path(img_fold)
output_dir = Path(output_fold)

output_dir.mkdir(parents=True, exist_ok=True)
files = list(img_path.glob("*.png"))

for idx, f in enumerate(files):

    img = cv2.imread(f)

    sharp = laplacian_sharpen(
        img,
        alpha=0.5
    )
    output_path = output_dir / (f.stem + "_sharpen.png")
    cv2.imwrite(output_path, sharp)
    print(f"process images: {idx}/{len(files)}")
