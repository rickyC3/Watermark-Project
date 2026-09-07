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

    # OpenCV is BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    rgb = img_rgb.astype(np.float32) / 255.0

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    # Black component
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
# Rotated AM Halftone
# ============================================================

def am_halftone_gray(gray, cell_size=8, angle_deg=0):
    """
    AM halftone with rotated screen.

    gray:
        grayscale image, float32, range [0, 1]

    cell_size:
        halftone cell size in pixels

    angle_deg:
        screen angle
    """

    H, W = gray.shape

    # --------------------------------------------------------
    # Pixel coordinates
    # --------------------------------------------------------

    y, x = np.mgrid[0:H, 0:W]

    # Move origin to image center
    x_center = x - W / 2.0
    y_center = y - H / 2.0

    # --------------------------------------------------------
    # Rotate coordinate system
    #
    # The halftone screen itself is rotated by angle_deg.
    # --------------------------------------------------------

    theta = np.deg2rad(angle_deg)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = cos_t * x_center + sin_t * y_center
    yr = -sin_t * x_center + cos_t * y_center

    # --------------------------------------------------------
    # Determine which halftone cell each pixel belongs to
    # --------------------------------------------------------

    cell_x = np.floor(xr / cell_size).astype(np.int32)
    cell_y = np.floor(yr / cell_size).astype(np.int32)

    # Shift indices to positive
    min_x = cell_x.min()
    min_y = cell_y.min()

    cell_x = cell_x - min_x
    cell_y = cell_y - min_y

    num_x = cell_x.max() + 1
    num_y = cell_y.max() + 1

    cell_id = cell_y * num_x + cell_x

    # --------------------------------------------------------
    # Calculate average intensity of each halftone cell
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
    #
    # Darker region -> larger dot
    # Brighter region -> smaller dot
    # --------------------------------------------------------

    radius_per_cell = (
        1.0 - cell_mean
    ) * (cell_size / 2.0)

    radius = radius_per_cell[cell_id]

    # --------------------------------------------------------
    # Center of each halftone cell
    # --------------------------------------------------------

    center_x = (
        (cell_x + 0.5) * cell_size
        + min_x * cell_size
    )

    center_y = (
        (cell_y + 0.5) * cell_size
        + min_y * cell_size
    )

    # Distance from cell center
    dx = xr - center_x
    dy = yr - center_y

    r = np.sqrt(dx ** 2 + dy ** 2)

    # --------------------------------------------------------
    # Generate binary halftone
    # --------------------------------------------------------

    output = (r <= radius).astype(np.uint8)

    return output


# ============================================================
# CMYK Halftone
# ============================================================

def cmyk_halftone(img, cell_size=8):

    # --------------------------------------------------------
    # RGB -> CMYK
    # --------------------------------------------------------

    C, M, Y, K = rgb_to_cmyk(img)

    # --------------------------------------------------------
    # Different screen angles
    # --------------------------------------------------------

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
# CMYK Halftone -> RGB
# ============================================================

def cmyk_to_rgb(C, M, Y, K):
    """
    Convert binary CMYK halftone to RGB.

    C, M, Y, K:
        uint8, 0 or 1
    """

    C = C.astype(np.float32)
    M = M.astype(np.float32)
    Y = Y.astype(np.float32)
    K = K.astype(np.float32)

    # Simple subtractive printing model
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

    # --------------------------------------------------------
    # Input
    # --------------------------------------------------------

    input_path = (
        r"D:\Ricky\program\invisible-watermark"
        r"\invisible-watermark\images\image2.png"
    )

    img = cv2.imread(input_path)

    if img is None:
        raise FileNotFoundError(
            f"Cannot read image:\n{input_path}"
        )

    print("Input shape:", img.shape)

    # --------------------------------------------------------
    # Halftone
    # --------------------------------------------------------

    cell_size = 2

    C, M, Y, K = cmyk_halftone(
        img,
        cell_size=cell_size
    )

    # --------------------------------------------------------
    # Save individual CMYK screens
    # --------------------------------------------------------

    cv2.imwrite(
        "halftone_C.png",
        C * 255
    )

    cv2.imwrite(
        "halftone_M.png",
        M * 255
    )

    cv2.imwrite(
        "halftone_Y.png",
        Y * 255
    )

    cv2.imwrite(
        "halftone_K.png",
        K * 255
    )

    # --------------------------------------------------------
    # Combine CMYK
    # --------------------------------------------------------

    rgb_halftone = cmyk_to_rgb(
        C,
        M,
        Y,
        K
    )

    # RGB -> BGR for OpenCV
    bgr_halftone = cv2.cvtColor(
        rgb_halftone,
        cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(
        "halftone_cmyk_rosette.png",
        bgr_halftone
    )

    print("Saved:")
    print("  halftone_C.png")
    print("  halftone_M.png")
    print("  halftone_Y.png")
    print("  halftone_K.png")
    print("  halftone_cmyk_rosette.png")


if __name__ == "__main__":
    main()