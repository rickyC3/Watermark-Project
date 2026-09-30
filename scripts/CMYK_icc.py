from PIL import Image, ImageCms
from pathlib import Path

# --------------------------------------------------
# 1. Load image
# --------------------------------------------------
#img = Image.open(r"D:\Ricky\program\invisible-watermark\invisible-watermark\test_image\Christina_hidden.png").convert("RGB")
img_fold = r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\test"
output_fold = r"D:\Ricky\program\invisible-watermark\invisible-watermark\images\test\sofrproof"

input_path = Path(img_fold)
output_path = Path(output_fold)

output_path.mkdir(parents=True, exist_ok=True)
proc_files = list(input_path.glob("*.png"))

# --------------------------------------------------
# 2. ICC profiles
# --------------------------------------------------
srgb_profile = ImageCms.createProfile("sRGB")

cmyk_profile = ImageCms.getOpenProfile(
    r"D:\Ricky\program\invisible-watermark\invisible-watermark\CMYK_ICC\JapanColor2011Coated.icc"
)

# Monitor profile
monitor_profile = ImageCms.createProfile("sRGB")

# --------------------------------------------------
# 3. Build CMYK soft-proof transform
# --------------------------------------------------
proof_transform = ImageCms.buildProofTransform(
    inputProfile=srgb_profile,
    outputProfile=monitor_profile,
    proofProfile=cmyk_profile,
    inMode="RGB",
    outMode="RGB",
    renderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    proofRenderingIntent=ImageCms.Intent.RELATIVE_COLORIMETRIC,
    flags=ImageCms.Flags.SOFTPROOFING
)

# --------------------------------------------------
# 4. Apply
# --------------------------------------------------
cnt = 0
for f in proc_files:
    
    output_img_path = output_path / (f.stem + "cmyk.png")
    try:
        img = Image.open(f).convert("RGB")
        proced_img = ImageCms.applyTransform(
            img,
            proof_transform
        )

        proced_img.save(output_img_path)
        cnt+=1
        print(f"process success: {output_img_path.name}, process: {(cnt) / len(proc_files)*100:.2f}%")
    except Exception as e:
        print(f"[ERROR] {f.name}: {e}")