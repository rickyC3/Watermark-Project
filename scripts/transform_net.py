import numpy as np
import cv2

def am_halftone_gray(gray, cell_size=8):
    """
    AM halftone simulation.

    gray:
        grayscale uint8 image [H,W]

    cell_size:
        halftone cell size
    """
    # 点扩散法(Dot Diffusion)

    gray = gray.astype(np.float32) / 255.0

    H, W = gray.shape

    # pixel coordinates
    y, x = np.mgrid[0:H, 0:W]
    """
    np.mgrid[0:5, 0:5]
    array([[[0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2],
            [3, 3, 3, 3, 3],
            [4, 4, 4, 4, 4]],
        [[0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4]]])
    """

    # center of each halftone cell
    cx = (x // cell_size) * cell_size + cell_size / 2
    cy = (y // cell_size) * cell_size + cell_size / 2
    # 將影像拆成8*8的格子組成
    # 每個idx座標對應到該點屬於哪個格子的座標
    # (0, 0) -> (4, 4): 該點屬於 (4, 4)的格子

    # print(cx)
    # print(cy)

    # local coordinate
    dx = x - cx
    dy = y - cy

    # distance from dot center
    r = np.sqrt(dx ** 2 + dy ** 2)

    # brightness -> dot radius
    radius = (1.0 - gray) * (cell_size / 2)
    """
    灰階值越暗（gray 接近 0）→ 半徑越大（接近 cell_size / 2）→ 點越大。
    灰階值越亮（gray 接近 1）→ 半徑越小 → 點越小或幾乎沒有。
    這就是 AM 半色調 的核心：用點的大小（振幅）來表示不同的灰階。
    """

    output = (r <= radius).astype(np.uint8) * 255

    return output

def rgb_to_cmyk(img):
    rgb = img.astype(np.float32) / 255.0

    K = 1.0 - np.max(rgb, axis=2)

    C = (1.0 - rgb[:, :, 0] - K) / (1.0 - K + 1e-8)
    M = (1.0 - rgb[:, :, 1] - K) / (1.0 - K + 1e-8)
    Y = (1.0 - rgb[:, :, 2] - K) / (1.0 - K + 1e-8)

    C[K >= 0.999] = 0
    M[K >= 0.999] = 0
    Y[K >= 0.999] = 0

    return C, M, Y, K

def rotate_coordinates(x, y, angle_deg):
    theta = np.deg2rad(angle_deg)

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    xr = cos_t * x + sin_t * y
    yr = -sin_t * x + cos_t * y

    return xr, yr

def main():
    img = cv2.imread("D:\Ricky\program\invisible-watermark\invisible-watermark\images\image2.png")
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2GRAY
    )

    halftone = am_halftone_gray(
        gray,
        cell_size=2
    )

    cv2.imwrite(
        "halftone2.png",
        halftone
    )

if __name__ == '__main__':
    main()
