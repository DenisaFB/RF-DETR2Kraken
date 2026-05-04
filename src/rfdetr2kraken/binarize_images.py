import cv2
from pathlib import Path


image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def binarize_image(input_image, output_image):
    """
    Binarize image in grayscale using adaptive thresholding.
    """
    output_image.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(input_image), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read image: {input_image}")

    bin_image = cv2.adaptiveThreshold(image, 255, 
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15)

    cv2.imwrite(str(output_image), bin_image)


def iter_images(input_dir):
    for img_path in sorted(input_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            yield img_path


def run_binarization(input_dir, output_dir):
    """
    Binarize all images from input_dir and write them to output_dir.
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(input_dir))
    if not images:
        print(f"No images found in: {input_dir}")
        return output_dir

    count = 0

    for input_image in images:
        output_image = output_dir / input_image.name
        try:
            binarize_image(input_image, output_image)
            print(f"Wrote {output_image}")
            count += 1
        except Exception as e:
            print(f"Skip (error): {input_image} -> {e}")

    print(f"Done. Binarized {count} image(s) to: {output_dir}")
    return output_dir