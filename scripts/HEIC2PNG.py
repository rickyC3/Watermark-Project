from pathlib import Path
from PIL import Image
import pillow_heif


pillow_heif.register_heif_opener()


def convert_folder(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    heic_files = list(input_dir.glob("*.heic")) + list(input_dir.glob("*.HEIC"))

    for heic_path in heic_files:
        output_path = output_dir / (heic_path.stem + ".png")

        try:
            img = Image.open(heic_path)
            img = img.convert("RGB")
            img.save(output_path, "PNG")

            print(f"[OK] {heic_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"[ERROR] {heic_path.name}: {e}")


if __name__ == "__main__":
    convert_folder(
        r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\print_cam_print_cam\Print cam print cam",
        r"D:\Ricky\NTHU\Project_DM\print-cam\image\coco_test_144_original\print_cam_print_cam\Print cam print cam_PNG"
    )