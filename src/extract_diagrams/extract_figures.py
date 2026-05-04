"""
Extract crops of <TextRegion> elements from XML Page files. 

Each <TextRegion> is filtered by the "type:" value inside its "custom" attribute.

Ex: 
    <TextRegion id="r4" custom="structure {type:GraphicZone-figure;}">
      <Coords points="403,791 139,776 104,932 114,991 178,1072 254,1106 393,1121 491,1095 520,907 498,839"/>
    </TextRegion> 

Zone = GraphicZone-figure
Polygon points : 403,791 139,776 104,932 114,991 178,1072 254,1106 393,1121 491,1095 520,907 498,839       

Two extraction modes are possible:
    - "bbox"    : rectangular crop around of the polygon (by default).
    - "polygon" : precise polygon cut-out
Results are saved as PNG.
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw


DEFAULT_REGION_TYPE = "GraphicZone-figure"
VALID_MODES = ("bbox", "polygon")


def parse_points(points_str):
    points = []
    for token in points_str.strip().split():
        x, y = token.split(",")
        points.append((int(round(float(x))), int(round(float(y)))))
    return points


def bbox_from_points(points, width, height):
    """
    Compute a bbox (left, top, right, bottom) from polygon points.
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left = max(0, min(xs))
    top = max(0, min(ys))
    right = min(width, max(xs))
    bottom = min(height, max(ys))
    return left, top, right, bottom


def get_namespace(root):
    if root.tag.startswith("{") and "}" in root.tag:
        return root.tag.split("}")[0].strip("{")
    return ""


def get_image_path(root, xml_path, img_dir):
    ns = get_namespace(root)
    if ns: 
        page = root.find(".//{" + ns + "}Page")
    else: 
        page = root.find(".//Page")

    if page is None:
        raise ValueError(f"No <Page> element found in {xml_path}")

    image_name = page.attrib.get("imageFilename")
    if not image_name:
        raise ValueError(f"No imageFilename found in <Page> of {xml_path}")

    if img_dir:
        img_path = Path(img_dir) / image_name
    else:
        img_path = xml_path.parent / image_name

    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    return img_path


def get_regions(root, ns, region_type):
    """"
    Extract all <TextRegion> elements matching a give type.
    """
    regions = []

    if ns:
        nsmap = {"pc": ns}
        text_regions = root.findall(".//pc:TextRegion", nsmap)
    else:
        nsmap = {}
        text_regions = root.findall(".//TextRegion")

    for region in text_regions:
        custom = region.attrib.get("custom", "")
        if f"type:{region_type};" in custom:
            regions.append(region)

    return regions, nsmap


def crop_bbox(im, points, width, height):
    left, top, right, bottom = bbox_from_points(points, width, height)
    if right <= left or bottom <= top:
        return None, None
    return im.crop((left, top, right, bottom)), None


def crop_polygon(im, points, width, height):
    left, top, right, bottom = bbox_from_points(points, width, height)
    if right <= left or bottom <= top:
        return None, None

    rgba = im.convert("RGBA")

    # Create mask: white inside polygon, black outside
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(points, outline=255, fill=255)

    # Apply mask
    cut = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    cut.paste(rgba, (0, 0), mask)

    # Crop to bbox
    cut = cut.crop((left, top, right, bottom))
    return cut, ".png"


def extract_regions(xml_path, img_dir, out_dir, region_type, mode = "bbox"):
    """
    Extract every region of "region_type" from an XML Page file (bbox or/and polygon shape)
    """

    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode '{mode}', expected one of {VALID_MODES}")

    xml_path = Path(xml_path)
    out_dir = Path(out_dir)

    if not xml_path.exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")

    root = ET.parse(xml_path).getroot()
    ns = get_namespace(root)
    img_path = get_image_path(root, xml_path, img_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with Image.open(img_path) as im:
        width, height = im.size
        regions, nsmap = get_regions(root, ns, region_type)

        if not regions:
            print(f"{xml_path.name} [{region_type}/{mode}]: no regions found")
            return 0

        count = 0
        stem = Path(img_path.name).stem
        original_ext = Path(img_path.name).suffix.lower()

        for i, region in enumerate(regions, 1):
            if ns:
                coords = region.find("./pc:Coords", nsmap)
            else:
                coords = region.find("./Coords")

            if coords is None:
                continue

            points_str = coords.attrib.get("points", "").strip()
            if not points_str:
                continue

            points = parse_points(points_str)

            if mode == "bbox":
                crop, forced_ext = crop_bbox(im, points, width, height)
            else:  # "polygon"
                crop, forced_ext = crop_polygon(im, points, width, height)

            if crop is None:
                continue
            
            out_ext = forced_ext if forced_ext else original_ext
            out_path = out_dir / f"{stem}_crop{i}{out_ext}"

            if out_ext in (".jpg", ".jpeg"):
                # JPEG doesn't support alpha; fall back to RGB if needed.
                if crop.mode in ("RGBA", "LA", "P"):
                    crop = crop.convert("RGB")
                crop.save(out_path, quality=95)
            else:
                crop.save(out_path)

            count += 1

        print(f"{xml_path.name} [{region_type}/{mode}]: saved {count} crop(s)")
        return count


def extract_regions_batch(xml_dir, img_dir, out_dir, region_types, mode = "bbox"):
    """
    Batch process all XML Page files in a directory and extract regions matching one ore more "type:" values. 

    Each region_type is written to its own subfolder: out_dir / <region_type>.
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode '{mode}', expected one of {VALID_MODES}")

    if region_types is None:
        region_types = [DEFAULT_REGION_TYPE]

    xml_dir = Path(xml_dir)
    img_dir = Path(img_dir) if img_dir is not None else None
    out_dir = Path(out_dir)

    if not xml_dir.is_dir():
        raise NotADirectoryError(f"XML folder not found: {xml_dir}")

    xml_files = sorted(xml_dir.glob("*.xml"))
    if not xml_files:
        print(f"[extract] no XML files found in {xml_dir}")
        return 0

    grand_total = 0
    for region_type in region_types:
        type_out = out_dir / region_type
        print(f"\nExtraction type='{region_type}' mode='{mode}' -> {type_out}")
        type_total = 0
        for xml_path in xml_files:
            try:
                type_total += extract_regions(
                    xml_path, img_dir, type_out, region_type, mode=mode
                )
            except FileNotFoundError as e:
                print(f"  ! {xml_path.name}: {e}")
            except Exception as e:
                print(f"  ! {xml_path.name}: {type(e).__name__}: {e}")
        print(f"Extraction type='{region_type}' mode='{mode}': "
              f"{type_total} crop(s) across {len(xml_files)} file(s)")
        grand_total += type_total

    print(f"\nTotal crops saved across {len(region_types)} type(s): {grand_total}")
    return grand_total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract region crops from PAGE XML + source images."
    )
    parser.add_argument("--xml", type=Path, help="Single XML file")
    parser.add_argument("--xml-dir", type=Path, help="Folder with XML files")
    parser.add_argument("--img-dir", type=Path, help="Folder containing the source images")
    parser.add_argument("--out", required=True, type=Path, help="Output folder")
    parser.add_argument("--type", nargs="+", default=[DEFAULT_REGION_TYPE],
        help=f"Region type(s) from the `custom` attribute (default: {DEFAULT_REGION_TYPE}). "
             "Pass one or more values separated by spaces.")
    parser.add_argument("--mode", choices=VALID_MODES, default="bbox",
        help="Extraction mode: 'bbox' for a rectangular crop (default), "
             "or 'polygon' for a polygon cut-out with transparent background (PNG).")
    
    args = parser.parse_args()

    if args.xml:
        total = 0
        for region_type in args.type:
            # When multiple types on a single file, also split into subfolders.
            type_out = args.out / region_type if len(args.type) > 1 else args.out
            total += extract_regions(args.xml, args.img_dir, type_out, region_type, mode=args.mode)
        print(f"\nTotal crops saved: {total}")

    elif args.xml_dir:
        extract_regions_batch(xml_dir=args.xml_dir, img_dir=args.img_dir, 
                              out_dir=args.out, region_types=args.type, mode=args.mode)