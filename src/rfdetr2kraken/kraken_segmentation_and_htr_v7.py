import shutil
import subprocess
import tempfile
from pathlib import Path
from lxml import etree


kraken_cmd = shutil.which("kraken")


def detect_kraken_device():
    """
    Prefer CUDA (for Linux), then Apple MPS, otherwise CPU.
    """
    try:
        import torch
    except ImportError:
        return "cpu"

    if torch.cuda.is_available():
        return "cuda:0"

    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def detect_kraken_precision(device):
    if device.startswith("cuda"):
        return "bf16-mixed"
    return "32-true"


def patch_input_xml(regions_xml, images_dir):
    images_dir = images_dir.resolve()

    tree = etree.parse(str(regions_xml))
    page = tree.find(".//{*}Page")

    if page is not None:
        img = page.get("imageFilename") or ""
        img_name = Path(img).name
        abs_img = (images_dir / img_name).resolve()

        if abs_img.exists():
            page.set("imageFilename", str(abs_img))

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".xml")
    tmp_path = Path(tmp.name)
    tmp.close()

    tree.write(str(tmp_path), encoding="utf-8", 
               xml_declaration=True, pretty_print=True)
    return tmp_path


def strip_imgfilename_to_basename(page_xml_path):
    tree = etree.parse(str(page_xml_path))
    page = tree.find(".//{*}Page")

    if page is not None:
        img = page.get("imageFilename")
        if img:
            page.set("imageFilename", Path(img).name)

    tree.write(str(page_xml_path), encoding="utf-8", 
               xml_declaration=True, pretty_print=False)


def segment_page(regions_xml, output_xml, images_dir, 
                 text_direction = "horizontal-lr", device=None, precision=None, htr_model=None):
    
    if kraken_cmd is None:
        raise RuntimeError("Kraken executable not found in PATH")

    if device is None:
        device = detect_kraken_device()

    if precision is None:
        precision = detect_kraken_precision(device)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    patched_xml = patch_input_xml(regions_xml, images_dir)
    
    # Kraken, version 7.0 (not Mac Intel compatible)
    # TODO : Add padding for the text lines if possible - check if it possible for both Kraken versions
    cmd = [
        kraken_cmd,
        "-i",
        str(patched_xml),
        str(output_xml),
        "-f",
        "page",
        "-x",
        "--device",
        device,
        "--precision",
        precision,
        "segment",
        "-bl",
        "-d",
        text_direction,
        "--remove-hlines",
        "-i",
        "models/segmentation/fine-tuned-model_best.mlmodel",
    ]

    # Kraken, version 5.2.9 (Mac Intel compatible) - command used inside the script
    # kraken_segmentation_and_htr.py adapted for Mac Intel users
    """cmd = [
        kraken_cmd,
        "-x",
        "-i",
        str(patched_xml),
        str(output_xml),
        "-f",
        "xml",
        "segment",
        "-bl",
    ]"""

    
    if htr_model is not None:
        cmd += ["ocr", "-m", str(htr_model)]

    print("Running:", " ".join(cmd))

    try:
        subprocess.check_call(cmd)
        strip_imgfilename_to_basename(output_xml)
    finally:
        patched_xml.unlink(missing_ok=True)


def segment_pages_batch(regions_dir, images_dir, output_dir, 
                        text_direction = "horizontal-lr", device=None, precision=None, htr_model=None):
    if kraken_cmd is None:
        raise RuntimeError("Kraken executable not found in PATH")

    if device is None:
        device = detect_kraken_device()

    if precision is None:
        precision = detect_kraken_precision(device)

    print(f"Using Kraken device: {device}")
    print(f"Using Kraken precision: {precision}")

    if not regions_dir.exists():
        raise FileNotFoundError(f"Missing regions dir: {regions_dir}")

    if not images_dir.exists():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    region_files = sorted(regions_dir.glob("*_regions.xml"))
    if not region_files:
        print(f"No region XML files found in: {regions_dir}")
        return output_dir

    count = 0

    for regions_xml in region_files:
        stem = regions_xml.name.removesuffix("_regions.xml")
        output_xml = output_dir / f"{stem}_baselines.xml"

        try:
            segment_page(regions_xml=regions_xml, output_xml=output_xml, images_dir=images_dir, 
                         text_direction=text_direction, device=device, precision=precision, htr_model=htr_model)
            print(f"Wrote {output_xml}")
            count += 1
        except Exception as e:
            print(f"Skip (error): {regions_xml} -> {e}")

    print(f"Done. Wrote {count} baseline XML file(s) to: {output_dir}")
    return output_dir