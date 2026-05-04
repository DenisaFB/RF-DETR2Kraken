import json
from pathlib import Path
from lxml import etree


ns = {None: "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"}


def json_to_page_xml(data, image_filename):
    """
    Convert one RF-DETR prediction JSON object into a PAGE XML root element.
    """
    width = str(int(data["image"]["width"]))
    height = str(int(data["image"]["height"]))

    root = etree.Element("PcGts", nsmap=ns)
    page = etree.SubElement(root, "Page", imageFilename=image_filename, 
                            imageWidth=width, imageHeight=height,)

    predictions = data.get("predictions") or []
    for i, pred in enumerate(predictions):
        points = pred.get("points") or []
        if not points:
            continue

        coords = " ".join(f"{int(p['x'])},{int(p['y'])}" for p in points)
        zone_class = (pred.get("class") or "").strip()

        region = etree.SubElement(page, "TextRegion", id=f"r{i}", 
                                  custom=f"structure {{type:{zone_class};}}" if zone_class else "",)
        etree.SubElement(region, "Coords", points=coords)

    return root


def infer_image_filename_from_json(data, json_path):
    """
    Recover the image filename stored in the prediction JSON.
    Falls back to <json_stem>.jpg if the original extension is unavailable.
    """
    image_info = data.get("image") or {}

    original_path = image_info.get("Path", "") or image_info.get("path", "")
    ext = Path(original_path).suffix if original_path else ""

    if not ext:
        ext = ".jpg"

    return f"{json_path.stem}{ext}"


def run_json_to_regions_xml(json_dir: Path, xml_dir: Path) -> Path:
    """
    Convert all prediction JSON files in json_dir into PAGE XML region files
    written to xml_dir.
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing JSON directory: {json_dir}")

    if not json_dir.is_dir():
        raise NotADirectoryError(f"JSON path is not a directory: {json_dir}")

    xml_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return xml_dir

    count = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            image_filename = infer_image_filename_from_json(data, json_path)

            root = json_to_page_xml(data, image_filename)
            output_xml = xml_dir / f"{json_path.stem}_regions.xml"

            etree.ElementTree(root).write(str(output_xml), encoding="utf-8", 
                                          xml_declaration=True, pretty_print=True,)

            print(f"Wrote {output_xml}")
            count += 1

        except Exception as e:
            print(f"Skip (error): {json_path} -> {e}")

    print(f"Done. Wrote {count} region XML file(s) to: {xml_dir}")
    return xml_dir