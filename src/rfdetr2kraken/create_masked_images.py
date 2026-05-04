import cv2
from pathlib import Path

image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

def run_masked_images(images_dir, masks_dir, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for image_path in sorted(images_dir.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in image_extensions:
            continue

        stem = image_path.stem
        mask_path = masks_dir / f"{stem}_mask.png"
        if not mask_path.exists():
            print(f"Skip (missing mask): {mask_path}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if image is None:
            print(f"Skip (could not read image): {image_path}")
            continue
        if mask is None:
            print(f"Skip (could not read mask): {mask_path}")
            continue
        if image.shape != mask.shape:
            print(f"Skip (shape mismatch): {image_path} / {mask_path}")
            continue

        # keep only allowed zones (mask==0), hide others
        masked = image.copy()
        masked[mask != 0] = 255

        out_path = output_dir / image_path.name
        cv2.imwrite(str(out_path), masked)
        print(f"Wrote {out_path}")
        count += 1

    print(f"Done. Wrote {count} masked binary image(s) to: {output_dir}")
    return output_dir