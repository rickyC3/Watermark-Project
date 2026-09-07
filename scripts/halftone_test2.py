import numpy as np
import cv2


# ============================================================
# RGB -> CMYK
# ============================================================

def rgb_to_cmyk(img):
    """
    img:
        BGR uint8 image from cv2.imread()

    return:
        C, M, Y, K
        float32, range [0, 1]
    """

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb = img_rgb.astype(np.float32) / 255.0

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    K = 1.0 - np.max(rgb, axis=2)

    denominator = 1.0 - K

    C = np.zeros_like(K)
    M = np.zeros_like(K)
    Y = np.zeros_like(K)

    mask = denominator > 1e-8

    C[mask] = (1.0 - R[mask] - K[mask]) / denominator[mask]
    M[mask] = (1.0 - G[mask] - K[mask]) / denominator[mask]
    Y[mask] = (1.0 - B[mask] - K[mask]) / denominator[mask]

    return C, M, Y, K


# ============================================================
# AM Halftone
# ============================================================

def am_halftone_gray(gray, cell_size=8, angle_deg=0):

    H, W = gray.shape

    y, x = np.mgrid[0:H, 0:W]

    # Move origin to image center
    x_center = x - W / 2.0
    y_center = y - H / 2.0

    # Rotation
    theta = np.deg2rad(angle_deg)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = cos_t * x_center + sin_t * y_center
    yr = -sin_t * x_center + cos_t * y_center

    # Determine halftone cell
    cell_x = np.floor(xr / cell_size).astype(np.int32)
    cell_y = np.floor(yr / cell_size).astype(np.int32)

    min_x = cell_x.min()
    min_y = cell_y.min()

    cell_x = cell_x - min_x
    cell_y = cell_y - min_y

    num_x = cell_x.max() + 1

    cell_id = cell_y * num_x + cell_x

    # --------------------------------------------------------
    # Average intensity inside each cell
    # --------------------------------------------------------

    flat_gray = gray.reshape(-1)
    flat_id = cell_id.reshape(-1)

    cell_sum = np.bincount(
        flat_id,
        weights=flat_gray
    )

    cell_count = np.bincount(flat_id)

    cell_mean = cell_sum / np.maximum(cell_count, 1)

    # --------------------------------------------------------
    # Dot radius
    # --------------------------------------------------------

    radius_per_cell = (
        1.0 - cell_mean
    ) * (cell_size / 2.0)

    radius = radius_per_cell[cell_id]

    # --------------------------------------------------------
    # Cell center
    # --------------------------------------------------------

    center_x = (
        (cell_x + 0.5) * cell_size
        + min_x * cell_size
    )

    center_y = (
        (cell_y + 0.5) * cell_size
        + min_y * cell_size
    )

    dx = xr - center_x
    dy = yr - center_y

    r = np.sqrt(
        dx ** 2 + dy ** 2
    )

    # Binary halftone
    output = (
        r <= radius
    ).astype(np.uint8)

    return output


# ============================================================
# CMYK Halftone
# ============================================================

def cmyk_halftone(img, cell_size=8):

    C, M, Y, K = rgb_to_cmyk(img)

    C_halftone = am_halftone_gray(
        C,
        cell_size=cell_size,
        angle_deg=15
    )

    M_halftone = am_halftone_gray(
        M,
        cell_size=cell_size,
        angle_deg=75
    )

    Y_halftone = am_halftone_gray(
        Y,
        cell_size=cell_size,
        angle_deg=0
    )

    K_halftone = am_halftone_gray(
        K,
        cell_size=cell_size,
        angle_deg=45
    )

    return (
        C_halftone,
        M_halftone,
        Y_halftone,
        K_halftone
    )


# ============================================================
# CMYK -> RGB
# ============================================================

def cmyk_to_rgb(C, M, Y, K):

    C = C.astype(np.float32)
    M = M.astype(np.float32)
    Y = Y.astype(np.float32)
    K = K.astype(np.float32)

    R = (1.0 - C) * (1.0 - K)
    G = (1.0 - M) * (1.0 - K)
    B = (1.0 - Y) * (1.0 - K)

    rgb = np.stack(
        [R, G, B],
        axis=2
    )

    rgb = np.clip(
        rgb * 255.0,
        0,
        255
    ).astype(np.uint8)

    return rgb


# ============================================================
# Main
# ============================================================

def main():

    input_path = (
        r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\image10.jpg"
    )

    output_path = "halftone_cmyk_rosette.png"

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    UPSCALE = 4

    # Halftone cell size is now defined
    # in the HIGH-RESOLUTION domain.
    CELL_SIZE = 8

    # Printer / optical blur
    BLUR_SIGMA = 1.0

    # --------------------------------------------------------
    # Read image
    # --------------------------------------------------------

    img = cv2.imread(input_path)

    if img is None:
        raise FileNotFoundError(
            f"Cannot read image:\n{input_path}"
        )

    original_h, original_w = img.shape[:2]

    print("Original size:")
    print(original_w, "x", original_h)

    # ========================================================
    # 1. Upsampling
    # ========================================================

    high_w = original_w * UPSCALE
    high_h = original_h * UPSCALE

    img_high = cv2.resize(
        img,
        (high_w, high_h),
        interpolation=cv2.INTER_CUBIC
    )

    print("Upsampled size:")
    print(high_w, "x", high_h)

    cv2.imwrite(
        "01_upsampled.png",
        img_high
    )

    # ========================================================
    # 2. RGB -> CMYK + Halftone
    # ========================================================

    C, M, Y, K = cmyk_halftone(
        img_high,
        cell_size=CELL_SIZE
    )

    # Save individual channels
    cv2.imwrite(
        "02_C_halftone.png",
        C * 255
    )

    cv2.imwrite(
        "03_M_halftone.png",
        M * 255
    )

    cv2.imwrite(
        "04_Y_halftone.png",
        Y * 255
    )

    cv2.imwrite(
        "05_K_halftone.png",
        K * 255
    )

    # ========================================================
    # 3. CMYK -> RGB
    # ========================================================

    rgb_halftone = cmyk_to_rgb(
        C,
        M,
        Y,
        K
    )

    bgr_halftone = cv2.cvtColor(
        rgb_halftone,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        "06_halftone_highres.png",
        bgr_halftone
    )

    # ========================================================
    # 4. Printer / optical blur
    # ========================================================

    blurred = cv2.GaussianBlur(
        bgr_halftone,
        (0, 0),
        sigmaX=BLUR_SIGMA,
        sigmaY=BLUR_SIGMA
    )

    cv2.imwrite(
        "07_halftone_blurred.png",
        blurred
    )

    # ========================================================
    # 5. Downsampling
    # ========================================================

    final = cv2.resize(
        blurred,
        (original_w, original_h),
        interpolation=cv2.INTER_AREA
    )

    cv2.imwrite(
        output_path,
        final
    )

    print("Final size:")
    print(final.shape[1], "x", final.shape[0])

    print("\nSaved:")
    print("01_upsampled.png")
    print("02_C_halftone.png")
    print("03_M_halftone.png")
    print("04_Y_halftone.png")
    print("05_K_halftone.png")
    print("06_halftone_highres.png")
    print("07_halftone_blurred.png")
    print(output_path)


if __name__ == "__main__":
    main()