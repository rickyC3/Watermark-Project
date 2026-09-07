import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_channels(img1, img2, channel_names, title):

    n = len(channel_names)
    if (n != img1.shape[2]):
        print("ERROR! channel size and img dim. are not matched")
        return
    
    plt.figure(figsize=(5*n,4))

    for i, name in enumerate(channel_names):

        diff = cv2.absdiff(img1[:,:,i], img2[:,:,i])

        img1_mean = np.mean(img1[:, :, i])
        img2_mean = np.mean(img2[:, :, i])
        img1_std = np.std(img1[:, :, i])
        img2_std = np.std(img2[:, :, i])

        mae = np.mean(diff)
        mse = np.mean((img1[:,:,i].astype(np.float32) -
                       img2[:,:,i].astype(np.float32))**2)
        max_err = np.max(diff)

        
        print(f"{title} - {name}")
        print(f"img1 static\n mean: {img1_mean}, std: {img1_std}")
        print(f"img2 static\n mean: {img2_mean}, std: {img2_std}")
        print()
        print(f"    MAE : {mae:.4f}")
        print(f"    MSE : {mse:.4f}")
        print(f"    Max : {max_err}")
        print()

        plt.subplot(1,n,i+1)
        plt.imshow(diff, cmap="hot")
        plt.title(f"{name}\nMAE={mae:.2f}")
        plt.axis("off")
        plt.colorbar(fraction=0.046)

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def compare_channels_diff(img1, img2, channel_names, title):

    # =========================
    # Check input
    # =========================
    if img1.shape != img2.shape:
        print("ERROR! img1 and img2 shapes are not matched")
        print(f"img1 shape: {img1.shape}")
        print(f"img2 shape: {img2.shape}")
        return

    if len(img1.shape) != 3:
        print("ERROR! Input images must be 3-dimensional")
        return

    n = len(channel_names)

    if n != img1.shape[2]:
        print("ERROR! channel size and img dim. are not matched")
        return


    # =========================
    # Plot
    # =========================
    plt.figure(figsize=(6 * n, 5))


    for i, name in enumerate(channel_names):

        # ---------------------------------
        # Calculate signed difference
        # Important:
        # uint8 - uint8 will overflow
        # so convert to int16 first
        # Range theoretically: -255 ~ 255
        # ---------------------------------
        channel1 = img1[:, :, i].astype(np.int16)
        channel2 = img2[:, :, i].astype(np.int16)

        diff = channel1 - channel2


        # =========================
        # Image statistics
        # =========================
        img1_mean = np.mean(channel1)
        img2_mean = np.mean(channel2)

        img1_std = np.std(channel1)
        img2_std = np.std(channel2)


        # =========================
        # Difference statistics
        # =========================
        mae = np.mean(np.abs(diff))
        mse = np.mean(diff.astype(np.float32) ** 2)

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        min_diff = np.min(diff)
        max_diff = np.max(diff)


        # =========================
        # Print
        # =========================
        print("=" * 50)
        print(f"{title} - {name}")

        print("\nimg1 statistic")
        print(f"  mean: {img1_mean:.4f}")
        print(f"  std : {img1_std:.4f}")

        print("\nimg2 statistic")
        print(f"  mean: {img2_mean:.4f}")
        print(f"  std : {img2_std:.4f}")

        print("\nDifference statistic (img1 - img2)")
        print(f"  Mean diff : {mean_diff:.4f}")
        print(f"  Std diff  : {std_diff:.4f}")
        print(f"  MAE       : {mae:.4f}")
        print(f"  MSE       : {mse:.4f}")
        print(f"  Min       : {min_diff}")
        print(f"  Max       : {max_diff}")
        print()


        # =========================
        # Count each difference
        # =========================
        # bincount needs non-negative values
        # shift [-255, 255] -> [0, 510]
        counts = np.bincount(
            diff.flatten() + 255,
            minlength=511
        )

        diff_values = np.arange(-255, 256)


        # =========================
        # Plot histogram
        # =========================
        plt.subplot(1, n, i + 1)

        plt.bar(
            diff_values,
            counts,
            width=1.0
        )

        plt.axvline(
            0,
            linestyle="--",
            linewidth=1
        )

        plt.title(
            f"{name}\n"
            f"Mean={mean_diff:.2f}, MAE={mae:.2f}"
        )

        plt.xlabel("Pixel Difference (img1 - img2)")
        plt.ylabel("Number of Pixels")

        plt.xlim(-255, 255)

        plt.grid(axis="y", alpha=0.3)


    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def compare_rgb(path1, path2):

    img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    # compare_channels(
    #     img1,
    #     img2,
    #     ["R","G","B"],
    #     "RGB Comparison"
    # )

    compare_channels_diff(
        img1,
        img2,
        ["R","G","B"],
        "RGB Comparison"
    )

def compare_yuv(path1, path2):

    img1 = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2YUV)
    img2 = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2YUV)

    # compare_channels(
    #     img1,
    #     img2,
    #     ["Y","U","V"],
    #     "YUV Comparison"
    # )

    compare_channels_diff(
        img1,
        img2,
        ["Y","U","V"],
        "YUV Comparison"
    )

def rgb_to_cmyk(img):

    rgb = img.astype(np.float32)/255

    R = rgb[:,:,0]
    G = rgb[:,:,1]
    B = rgb[:,:,2]

    K = 1 - np.maximum.reduce([R,G,B])

    C = np.zeros_like(K)
    M = np.zeros_like(K)
    Y = np.zeros_like(K)

    mask = K < 0.999999

    C[mask] = (1-R[mask]-K[mask])/(1-K[mask])
    M[mask] = (1-G[mask]-K[mask])/(1-K[mask])
    Y[mask] = (1-B[mask]-K[mask])/(1-K[mask])

    cmyk = np.stack([C,M,Y,K],axis=2)

    return cmyk

def cmyk_to_rgb(cmyk_img):
    """
    Convert CMYK image (uint8, 0~255) to RGB image (uint8, 0~255).

    Parameters
    ----------
    cmyk_img : numpy.ndarray
        Shape = (H, W, 4)

    Returns
    -------
    rgb_img : numpy.ndarray
        Shape = (H, W, 3)
    """

    cmyk = cmyk_img.astype(np.float32)

    C = cmyk[:, :, 0]
    M = cmyk[:, :, 1]
    Y = cmyk[:, :, 2]
    K = cmyk[:, :, 3]

    R = (1 - C) * (1 - K)
    G = (1 - M) * (1 - K)
    B = (1 - Y) * (1 - K)

    rgb = np.stack([R, G, B], axis=2)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)

def compare_cmyk(path1, path2, save_img = True):

    img1_rgb = cv2.cvtColor(cv2.imread(path1), cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(cv2.imread(path2), cv2.COLOR_BGR2RGB)

    img1_cmyk = rgb_to_cmyk(img1_rgb)
    img2_cmyk = rgb_to_cmyk(img2_rgb)

    img1_rgb_2 = cmyk_to_rgb(img1_cmyk)
    img2_rgb_2 = cmyk_to_rgb(img2_cmyk)

    if (save_img):
        save_cmyk_as_rgb(img1_cmyk, "./img1_cmyk_out.png")
        #save_cmyk_as_rgb(img2_cmyk, "./img2_cmyk_out.png")
        img1_bgr = cv2.cvtColor(img1_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite("./img2_cmyk_out.png", img1_bgr)

    # compare_channels(
    #     img1_cmyk,
    #     img2_cmyk,
    #     ["C","M","Y","K"],
    #     "CMYK Comparison"
    # )

    compare_channels(img1_rgb, img1_rgb_2, ["R", "G", "B"], "RGB -> CMYK -> RGB Comparison")

def save_cmyk_as_rgb(cmyk_img, output_path):

    rgb = cmyk_to_rgb(cmyk_img)

    # OpenCV 存檔需轉成 BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    cv2.imwrite(output_path, bgr)

    print(f"Saved to {output_path}")

def compare_gray(img1_path, img2_path):
    # 讀取圖片 (OpenCV 為 BGR)
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("圖片讀取失敗")
        return

    if img1.shape != img2.shape:
        print("圖片尺寸不同")
        return

    # BGR -> RGB (Matplotlib 顯示需要)
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # 灰階
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # 差異
    diff = cv2.absdiff(gray1, gray2)

    # 差異遮罩
    _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
    # diff > 10 設為255, else set 0 (BINARY)

    different_pixels = np.count_nonzero(mask)
    total_pixels = mask.size

    print(f"Difference Ratio = {100*different_pixels/total_pixels:.4f}%")

    # ========= Visualization =========
    plt.figure(figsize=(14,5))

    plt.subplot(1,4,1)
    plt.imshow(img1_rgb)
    plt.title("Image 1")
    plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(img2_rgb)
    plt.title("Image 2")
    plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(diff, cmap='hot')
    plt.title("Difference Heatmap")
    plt.colorbar(fraction=0.046)

    plt.subplot(1,4,4)
    plt.imshow(mask, cmap='gray')
    plt.title("Difference Mask")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    compare_rgb(r"D:\Ricky\program\invisible-watermark\invisible-watermark\align_image\Christina_camera_fix.jpg", 
                r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\Christina_hidden.png")
    #compare_cmyk(r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\im11607_hidden.png", r"D:\Ricky\program\invisible-watermark\invisible-watermark\align_image\bear_camera_fix.jpg")