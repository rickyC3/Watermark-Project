import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import pywt
from pathlib import Path



def compare_channels_diff_batch(img1, img2, channel_names, title):

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
    # Plot / Statistics
    # =========================

    local_static = []
    local_counts = []

    for i, name in enumerate(channel_names):

        channel1 = img1[:, :, i].astype(np.int16)
        channel2 = img2[:, :, i].astype(np.int16)

        diff = channel1 - channel2

        mae = np.mean(np.abs(diff))

        mean_diff = np.mean(diff)
        std_diff = np.std(diff)

        min_diff = np.min(diff)
        max_diff = np.max(diff)

        static_inf = (
            mean_diff,
            std_diff,
            mae,
            min_diff,
            max_diff
        )

        # =========================
        # Count each difference
        # =========================

        # bincount needs non-negative values
        # shift [-255, 255] -> [0, 510]
    
        counts = np.bincount(
            diff.flatten() + 255,
            minlength=511
        )

        local_counts.append(counts)
        local_static.append(static_inf)

    return local_static, local_counts

def preprocess_image(
    img,
    color_domain="RGB",
    dwt_domain=None,
    do_dct = False
):
    """
    Image preprocessing:
    Parameters
    ----------
    img : np.ndarray
        Input RGB image, shape = (H, W, 3)

    color_domain : str
        "RGB" or "YUV"

    dwt_domain : str or None
        None
        "LL"
        "LH"
        "HL"
        "HH"

    Returns
    -------
    output : np.ndarray
        Preprocessed image / coefficients
    """

    # ==================================================
    # 1. Check input
    # ==================================================

    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError(
            f"Input image must be H x W x 3, "
            f"but got {img.shape}"
        )

    # ==================================================
    # 2. RGB -> selected color domain
    # ==================================================

    if color_domain == "RGB":
        data = img.copy()
        channel_names = ["R", "G", "B"]

    elif color_domain == "YUV":
        # TODO: YUV use PIL or OpenCV
        data = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        # pil_img = Image.fromarray(img.astype('uint8'), 'RGB')
        # data = np.array(pil_img.convert('YCbCr'))
        channel_names = ["Y", "U", "V"]

    else:
        raise ValueError(
            "color_domain must be 'RGB' or 'YUV'"
        )

    # ==================================================
    # 3. No DWT
    # ==================================================

    if dwt_domain is None:
        print("No freq. domain process")
        return data, channel_names

    # ==================================================
    # 4. DWT preprocessing
    # ==================================================

    if dwt_domain not in ["LL", "LH", "HL", "HH"]:
        raise ValueError(
            "dwt_domain must be "
            "'LL', 'LH', 'HL', 'HH', or None"
        )

    # --------------------------------------------------
    # Make image size divisible by 2
    # Haar DWT requires even dimensions.
    # --------------------------------------------------

    row, col, channel = data.shape
    # TODO: for 400 * 400, it should be ok
    row_even = (row // 2) * 2
    col_even = (col // 2) * 2

    data = data[ :row_even, :col_even, : ]

    # --------------------------------------------------
    # Store selected DWT coefficients
    #
    # Shape will be approximately:
    # (H/2, W/2, 3)
    # --------------------------------------------------

    dwt_result = []

    for c in range(channel):

        channel_data = data[:, :, c].astype(np.float32)

        # ----------------------------------------------
        # 2D Haar DWT
        #
        # ca = LL
        # h  = LH
        # v  = HL
        # d  = HH
        # ----------------------------------------------

        ca, (h, v, d) = pywt.dwt2(channel_data, "haar")

        # ----------------------------------------------
        # Select DWT domain
        # ----------------------------------------------

        if dwt_domain == "LL":
            selected = ca

        elif dwt_domain == "LH":
            selected = h

        elif dwt_domain == "HL":
            selected = v

        elif dwt_domain == "HH":
            selected = d

        if do_dct is not False:
            dct_row, dct_cal = selected.shape
            block = 4
            ROW = dct_row // block
            COL = dct_cal // block
        
            TOTAL_BLOCK = ROW * COL
            for idx in range(0, TOTAL_BLOCK, 1):
                r1 = idx // COL
                c1 = idx % COL
                block_A = selected[r1*block : r1*block + block,
                                  c1*block : c1*block + block]

                dct_A = cv2.dct(block_A)

                selected[r1*block : r1*block + block,
                    c1*block : c1*block + block] = dct_A
            
                
        dwt_result.append(selected)

    # --------------------------------------------------
    # Stack channels
    #
    # (H/2, W/2, 3)
    # --------------------------------------------------

    output = np.stack(dwt_result, axis=2)
    return output, channel_names

def run_freq_diff(
    table,
    color_domain,
    dwt_domain,
    dct_en=False,
    block_size=4,
    save_dir="./freq_result",
    show_plot=True,
    do_normalize=False
):
    

    # ==========================================================
    # Imports
    # ==========================================================

    import os
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import cv2
    from PIL import Image

    # ==========================================================
    # Create output directory
    # ==========================================================

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(table)

    if "img1" not in df.columns or "img2" not in df.columns:
        print("ERROR! CSV must contain 'img1' and 'img2' columns")
        return None

    global_static = []

    spatial_abs_sum = None
    spatial_sq_sum = None
    spatial_signed_sum = None

    block_abs_sum = None
    block_sq_sum = None

    # DCT coefficient frequency statistics
    # Shape:
    #     channel x block_size x block_size
    dct_freq_abs_sum = None
    dct_freq_sq_sum = None

    num_pairs = 0

    channel_names = None

    # ==========================================================
    # Process every image pair
    # ==========================================================

    for idx, row in df.iterrows():

        img1_path = row["img1"]
        img2_path = row["img2"]

        print(f"\r[{idx + 1}/{len(df)}] Processing...", end="", flush=True)

        try:

            # ==================================================
            # Read images
            # ==================================================

            img1 = np.array(Image.open(img1_path).convert("RGB"))
            img2 = np.array(Image.open(img2_path).convert("RGB"))

            # ==================================================
            # Preprocess
            # ==================================================

            img1_proc, channel_names = preprocess_image(
                img1,
                color_domain=color_domain,
                dwt_domain=dwt_domain,
                do_dct=dct_en
            )

            img2_proc, _ = preprocess_image(
                img2,
                color_domain=color_domain,
                dwt_domain=dwt_domain,
                do_dct=dct_en
            )

            # ==================================================
            # Make sure shapes match
            # ==================================================

            if img1_proc.shape != img2_proc.shape:
                print("WARNING: transformed image shapes are different, skip.")
                print(f"img1: {img1_proc.shape}")
                print(f"img2: {img2_proc.shape}")
                continue

            # ==================================================
            # Difference
            # ==================================================

            # Convert to float
            #
            # Important:
            # DWT/DCT coefficients can be negative
            #
            img1_f = img1_proc.astype(np.float64)
            img2_f = img2_proc.astype(np.float64)

            diff = img1_f - img2_f

            abs_diff = np.abs(diff)

            # ==================================================
            # Initialize accumulators
            # ==================================================

            H, W, C = diff.shape

            if spatial_abs_sum is None:
                spatial_abs_sum = np.zeros((H, W, C),dtype=np.float64)
                spatial_sq_sum = np.zeros((H, W, C),dtype=np.float64)
                spatial_signed_sum = np.zeros((H, W, C),dtype=np.float64)

            # ==================================================
            # Accumulate spatial coefficient statistics
            # ==================================================

            spatial_abs_sum += abs_diff
            spatial_sq_sum += diff ** 2
            spatial_signed_sum += diff

            # ==================================================
            # Block statistics
            # ==================================================

            if dct_en:
                block_rows = H // block_size
                block_cols = W // block_size
                if block_abs_sum is None:
                    block_abs_sum = np.zeros((block_rows, block_cols, C),dtype=np.float64)
                    block_sq_sum = np.zeros((block_rows, block_cols, C), dtype=np.float64)
                    dct_freq_abs_sum = np.zeros((block_size, block_size, C), dtype=np.float64)
                    dct_freq_sq_sum = np.zeros((block_size, block_size, C), dtype=np.float64)

                # ==================================================
                # Loop over blocks
                # ==================================================

                for br in range(block_rows):
                    for bc in range(block_cols):

                        r1 = br * block_size
                        r2 = r1 + block_size

                        c1 = bc * block_size
                        c2 = c1 + block_size

                        block_diff = diff[
                            r1:r2,
                            c1:c2,
                            :
                        ]

                        block_abs = np.abs(block_diff)

                        block_abs_sum[br, bc, :] += np.mean(block_abs, axis=(0, 1)) # axis 2保留維度
                        block_sq_sum[br, bc, :] += np.mean(
                            block_diff ** 2,
                            axis=(0, 1)
                        )


                        for ch in range(C):

                            dct_freq_abs_sum[:, :, ch] += block_abs[:, :, ch]
                            dct_freq_sq_sum[:, :, ch] += block_diff[:, :, ch] ** 2

            # ==================================================
            # Global statistics for this image pair
            # ==================================================

            for ch, name in enumerate(channel_names):

                ch_diff = diff[:, :, ch]
                mean_diff = np.mean(ch_diff)
                std_diff = np.std(ch_diff)

                mae = np.mean(np.abs(ch_diff))
                rmse = np.sqrt(np.mean(ch_diff ** 2))

                min_diff = np.min(ch_diff)
                max_diff = np.max(ch_diff)

                max_abs_diff = np.max(np.abs(ch_diff))

                global_static.append({
                    "pair": idx,
                    "channel": name,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "mae": mae,
                    "rmse": rmse,
                    "min_diff": min_diff,
                    "max_diff": max_diff,
                    "max_abs_diff": max_abs_diff
                })

            num_pairs += 1

        # ======================================================
        # Exception
        # ======================================================

        except Exception as e:

            print(f"ERROR processing pair {idx + 1}")
            print(f"img1: {img1_path}")
            print(f"img2: {img2_path}")
            print(f"Error: {e}")

    # ==========================================================
    # No valid pairs
    # ==========================================================

    if num_pairs == 0:
        print("ERROR: No valid image pairs processed.")
        return None

    print()
    print("=" * 70)
    print("Processing finished")
    print(f"Valid pairs : {num_pairs}")
    print(f"Channels    : {channel_names}")
    print(f"Color       : {color_domain}")
    print(f"DWT         : {dwt_domain}")
    print(f"DCT         : {dct_en}")
    print("=" * 70)

    # ==========================================================
    # Average spatial statistics
    # ==========================================================

    spatial_mean_abs = (spatial_abs_sum / num_pairs)
    spatial_rmse = np.sqrt(spatial_sq_sum / num_pairs)

    spatial_mean_signed = (spatial_signed_sum / num_pairs)

    if (do_normalize):
        spatial_mean_abs_visual = np.zeros((H, W, C), dtype=np.float64)
        for ch in range(C):
            mean_abs_max = np.max(spatial_mean_abs[:, :, ch])
            mean_abs_min = np.min(spatial_mean_abs[:, :, ch])
            spatial_mean_abs_visual[:, :, ch] = (spatial_mean_abs[:, :, ch] - mean_abs_min) / (mean_abs_max - mean_abs_min) 

    # ==========================================================
    # Global transformed-domain statistics
    # ==========================================================

    global_results = []

    for ch, name in enumerate(channel_names):

        abs_map = spatial_mean_abs[:, :, ch]
        rmse_map = spatial_rmse[:, :, ch]
        signed_map = spatial_mean_signed[:, :, ch]

        global_results.append({
            "channel": name,
            "mean_abs_diff": np.mean(abs_map),
            "mean_signed_diff": np.mean(signed_map),
            "spatial_std": np.std(signed_map),
            "rmse": np.sqrt(np.mean(rmse_map ** 2)),
            "max_abs_diff": np.max(abs_map),
            "median_abs_diff": np.median(abs_map),
            "p95_abs_diff": np.percentile(abs_map, 95),
            "p99_abs_diff": np.percentile(abs_map, 99)
        })

    global_df = pd.DataFrame(global_results)

    # ==========================================================
    # Print global statistics
    # ==========================================================

    print()
    print("GLOBAL STATISTICS")
    print("-" * 70)
    print(global_df.to_string(index=False))

    # ==========================================================
    # Save global statistics
    # ==========================================================

    global_csv = os.path.join(save_dir, "global_statistics.csv")
    global_df.to_csv(global_csv, index=False)

    # ==========================================================
    # Plot spatial coefficient heatmaps
    # ==========================================================

    for ch, name in enumerate(channel_names):

        heatmap = spatial_mean_abs[:, :, ch]
        if (do_normalize):
            heatmap = spatial_mean_abs_visual[:, :, ch]

        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap="hot", interpolation="nearest")
        plt.colorbar(label="Mean Absolute Coefficient Difference")
        plt.title(f"Coefficient Change Heatmap - {name}")

        plt.xlabel("Coefficient X")
        plt.ylabel("Coefficient Y")

        plt.tight_layout()

        save_path = os.path.join(save_dir, f"spatial_heatmap_{name}.png")

        plt.savefig(save_path, dpi=200, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    # ==========================================================
    # DCT block-level statistics
    # ==========================================================

    block_mean_abs = None
    block_rmse = None

    if dct_en:

        block_mean_abs = (block_abs_sum / num_pairs)
        block_rmse = np.sqrt(block_sq_sum / num_pairs)

        if (do_normalize):
            block_mean_abs_visual = np.zeros((block_rows, block_cols, C), dtype=np.float64)
            for ch in range(C):
                block_mean_abs_max = np.max(block_mean_abs[:, :, ch])
                block_mean_abs_min = np.min(block_mean_abs[:, :, ch])
                block_mean_abs_visual[:, :, ch] = (block_mean_abs[:, :, ch] - block_mean_abs_min) / (block_mean_abs_max - block_mean_abs_min) 

        # ------------------------------------------------------
        # Block heatmap
        # ------------------------------------------------------

        for ch, name in enumerate(channel_names):

            heatmap = block_mean_abs[:, :, ch]

            if (do_normalize):
                heatmap = block_mean_abs_visual[:, :, ch]

            plt.figure(figsize=(10, 8))
            plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            plt.colorbar(label="Mean Absolute DCT Difference")
            plt.title(f"DCT Block Change Heatmap - {name}")

            plt.xlabel("DCT Block X")
            plt.ylabel("DCT Block Y")

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"dct_block_heatmap_{name}.png")

            plt.savefig(save_path, dpi=200, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

        # ======================================================
        # DCT frequency-position statistics
        # ======================================================

        dct_mean_abs = (dct_freq_abs_sum / num_pairs / 2500)
        dct_rmse = np.sqrt(dct_freq_sq_sum / num_pairs)


        if (do_normalize):
            dct_mean_abs_visual = np.zeros((block_size, block_size, C), dtype=np.float64)
            for ch in range(C):
                dct_mean_abs_max = np.max(dct_mean_abs[:, :, ch])
                dct_mean_abs_min = np.min(dct_mean_abs[:, :, ch])
                dct_mean_abs_visual[:, :, ch] = (dct_mean_abs[:, :, ch] - dct_mean_abs_min) / (dct_mean_abs_max - dct_mean_abs_min) 
            
        # ------------------------------------------------------
        # Print DCT frequency statistics
        # ------------------------------------------------------

        print()
        print("=" * 70)
        print("DCT FREQUENCY STATISTICS")
        print("=" * 70)

        for ch, name in enumerate(channel_names):

            print()
            print(f"Channel: {name}")
            print("Mean absolute difference for each DCT coefficient:")

            print(pd.DataFrame(dct_mean_abs[:, :, ch]).round(4).to_string(index=True, header=True))

        # ======================================================
        # DCT 4x4 frequency heatmap
        # ======================================================

        for ch, name in enumerate(channel_names):

            heatmap = dct_mean_abs[:, :, ch]

            if (do_normalize):
                heatmap = dct_mean_abs_visual[:, :, ch]

            plt.figure(figsize=(7, 6))
            plt.imshow(heatmap, cmap="hot", interpolation="nearest")
            plt.colorbar(label="Mean Absolute DCT Difference")
            plt.title(f"DCT Frequency Change - {name}")

            plt.xlabel("DCT Frequency u")
            plt.ylabel("DCT Frequency v")

            # --------------------------------------------------
            # Put actual values on heatmap
            # --------------------------------------------------

            for u in range(block_size):
                for v in range(block_size):

                    plt.text(
                        v,
                        u,
                        f"{heatmap[u, v]:.2f}",
                        ha="center",
                        va="center"
                    )

            plt.xticks(range(block_size))
            plt.yticks(range(block_size))

            plt.tight_layout()

            save_path = os.path.join(save_dir, f"dct_frequency_heatmap_{name}.png")

            plt.savefig(save_path, dpi=200, bbox_inches="tight")

            if show_plot:
                plt.show()
            else:
                plt.close()

        # ======================================================
        # Save DCT frequency statistics as CSV
        # ======================================================

        dct_rows = []

        for ch, name in enumerate(channel_names):
            for u in range(block_size):
                for v in range(block_size):

                    dct_rows.append({
                        "channel": name,
                        "u": u,
                        "v": v,
                        "mean_abs_diff":
                            dct_mean_abs[u, v, ch],
                        "rmse":
                            dct_rmse[u, v, ch]
                    })

        dct_df = pd.DataFrame(dct_rows)
        dct_csv = os.path.join(save_dir, "dct_frequency_statistics.csv")
        dct_df.to_csv(dct_csv, index=False)

    # ==========================================================
    # Save pair-level statistics
    # ==========================================================

    pair_df = pd.DataFrame(global_static)
    pair_csv = os.path.join(save_dir, "pair_statistics.csv")
    pair_df.to_csv(pair_csv, index=False)

    # ==========================================================
    # Return everything
    # ==========================================================

    results = {
        "global_statistics":
            global_df,
        "pair_statistics":
            pair_df,
        "spatial_mean_abs":
            spatial_mean_abs,
        "spatial_rmse":
            spatial_rmse,
        "spatial_mean_signed":
            spatial_mean_signed,
        "channel_names":
            channel_names,
        "num_pairs":
            num_pairs
    }

    if dct_en:
        results.update({
            "block_mean_abs":
                block_mean_abs,
            "block_rmse":
                block_rmse,
            "dct_mean_abs":
                dct_mean_abs,
            "dct_rmse":
                dct_rmse
        })

    print()
    print("=" * 70)
    print("Output files saved to:")
    print(save_dir)
    print("=" * 70)

    return results

def run_comparsion_diff(
    table,
    color_domain="RGB",
    dwt_domain=None,
    dct_en = False
):

    df = pd.read_csv(table)
    if "img1" not in df.columns or "img2" not in df.columns:
        print("ERROR! CSV must contain 'img1' and 'img2' columns")
        return
    
    global_counts = None
    all_static = []
    channel_names = None

    # ==========================================
    # Process every image pair
    # ==========================================

    for idx, row in df.iterrows(): # 迭代 row

        img1_path = row["img1"]
        img2_path = row["img2"]

        print(f"[{idx + 1}/{len(df)}] Processing...")

        try:
            img1 = np.array(Image.open(img1_path).convert("RGB"))
            img2 = np.array(Image.open(img2_path).convert("RGB"))

            img1, channel_names = preprocess_image(
                img1,
                color_domain=color_domain,
                dwt_domain=dwt_domain,
                do_dct = dct_en
            )

            img2, _ = preprocess_image(
                img2,
                color_domain=color_domain,
                dwt_domain=dwt_domain,
                do_dct = dct_en
            )

            # ======================================
            # Compare images
            # ======================================
            result = compare_channels_diff_batch(
                img1,
                img2,
                channel_names,
                title=f"Pair {idx + 1}"
            )
            

            if result is None:
                print(f"Skip pair {idx + 1}")
                continue

            local_static, local_counts = result

            # ======================================
            # Initialize global histogram
            # ======================================

            if global_counts is None:

                global_counts = np.zeros(
                    (
                        len(channel_names),
                        511
                    ),
                    dtype=np.int64
                )

            # ======================================
            # Accumulate histogram
            # ======================================

            for c in range(len(channel_names)):

                global_counts[c] += local_counts[c]

            # ======================================
            # Save statistics
            # ======================================

            for c, channel_name in enumerate(channel_names):

                (
                    mean_diff,
                    std_diff,
                    mae,
                    min_diff,
                    max_diff
                ) = local_static[c]

                all_static.append({
                    "pair": idx + 1,
                    "img1": img1_path,
                    "img2": img2_path,
                    "channel": channel_name,
                    "mean_diff": mean_diff,
                    "std_diff": std_diff,
                    "MAE": mae,
                    "min_diff": min_diff,
                    "max_diff": max_diff
                })

        except Exception as e:

            print(f"ERROR processing pair {idx + 1}")

            print(f"img1: {img1_path}")
            print(f"img2: {img2_path}")
            print(f"Error: {e}")

    # ==========================================
    # Check result
    # ==========================================

    if global_counts is None:

        print("No valid image pairs found.")

        return

    # ==========================================
    # DataFrame
    # ==========================================

    static_df = pd.DataFrame(all_static)

    # ==========================================
    # Plot histogram
    # ==========================================

    x = np.arange(-255, 256)

    n_channels = len(channel_names)

    plt.figure(figsize=(15, 5 * n_channels))

    for c, channel_name in enumerate(channel_names):

        plt.subplot(n_channels, 1, c + 1)
        plt.bar(x, global_counts[c], width=1.0)
        plt.axvline(0, linestyle="--")
        plt.title(f"{channel_name}")
        plt.xlabel("Difference")
        plt.ylabel("Pixel Count")

        plt.xlim(-255, 255)
        plt.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    plt.show()

    # ==========================================
    # Overall statistics
    # ==========================================

    overall_static = (
        static_df
        .groupby("channel")
        .agg({
            "mean_diff": "mean",
            "std_diff": "mean",
            "MAE": "mean",
            "min_diff": "min",
            "max_diff": "max"
        })
    )

    print(
        "\n========== "
        "Overall Statistics "
        "=========="
    )

    print(overall_static)

    return (overall_static, global_counts, static_df)

if __name__ == "__main__":
    r"""
    D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\original_pc_table.csv
    D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\original_pspc_table.csv
    D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\cmyk_pc_table.csv
    D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\cmyk_pspc_table.csv

    D:\Ricky\NTHU\Project_DM\print-cam\image\5cmx50_aligned_images\original_ymck_table.csv
    """

    table = r"D:\Ricky\NTHU\Project_DM\print-cam\image\PCx200\pc_v2_pspc.csv"
    table_path = Path(table)
    color_encode = "YUV"
    dwt_domain = "HH"
    other_inf = None
    is_norm = False
    dwt_domain_list = ["LL", "LH", "HL", "HH"]
    


    # run_comparsion_diff(
    #     r"D:\Ricky\NTHU\Project_DM\print-cam\image\PCx200\original_ps.csv",
    #     color_domain="YUV",
    #     dwt_domain=None,
    #     dct_en = False
    # )

    for d in dwt_domain_list:

        dwt_domain = d
        save_path = str(table_path.parent / ("csv_" + table_path.stem) / ("_" + color_encode + "_" + dwt_domain))
        if (is_norm):
            save_path += ("_norm_")
        if (other_inf is not None):
            save_path += ("_" + other_inf)
            
        results = run_freq_diff(
            table = table,
            color_domain=color_encode,
            dwt_domain=dwt_domain,
            dct_en=True,
            block_size=4,
            save_dir=save_path,
            show_plot=False,
            do_normalize=is_norm
        )
    #compare_cmyk(r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\im11607_hidden.png", r"D:\Ricky\program\invisible-watermark\invisible-watermark\align_image\bear_camera_fix.jpg")