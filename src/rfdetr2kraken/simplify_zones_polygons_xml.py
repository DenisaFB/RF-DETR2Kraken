""""
The polygons produced by the RF-DETR model have many points.
When correcting the polygons, it can become very difficult to move all the small points.
This script simplifies the polygons in order to have fewer points.
"""

from pathlib import Path
from lxml import etree
from shapely.geometry import MultiPolygon, Polygon


def parse_points(points_str):
    return [tuple(map(float, p.split(","))) for p in points_str.split() if p]


def format_points(coords):
    pts = list(coords)

    if len(pts) > 1 and pts[0] == pts[-1]:
        pts = pts[:-1]

    return " ".join(f"{int(round(x))},{int(round(y))}" for x, y in pts)


def simplify_textregion_coords(points_str, tolerance):
    """
    Simplify <TextRegion> polygon
    """
    pts = parse_points(points_str)
    if len(pts) < 4:
        return points_str

    poly = Polygon(pts)
    if not poly.is_valid:
        poly = poly.buffer(0)

    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda g: g.area)

    simplified = poly.simplify(tolerance, preserve_topology=True)

    if isinstance(simplified, MultiPolygon):
        simplified = max(simplified.geoms, key=lambda g: g.area)

    if simplified.is_empty or simplified.geom_type != "Polygon":
        return points_str

    if len(simplified.exterior.coords) < 4:
        return points_str

    return format_points(simplified.exterior.coords)


def simplify_polygons_page(input_path, output_path, tolerance):
    tree = etree.parse(str(input_path))
    root = tree.getroot()

    page_ns = root.nsmap.get(None)
    if not page_ns:
        raise ValueError(f"No PAGE namespace found in {input_path}")

    ns = {"pc": page_ns}

    for region in root.xpath(".//pc:TextRegion", namespaces=ns):
        coords = region.find("pc:Coords", namespaces=ns)
        if coords is None:
            continue

        points = coords.get("points")
        if not points:
            continue

        coords.set("points", simplify_textregion_coords(points, tolerance))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), encoding="UTF-8", 
               xml_declaration=True, pretty_print=True)


def simplify_polygons_batch(input_dir, output_dir, tolerance = 15):
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing XML input directory: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"XML input path is not a directory: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        print(f"No XML files found in: {input_dir}")
        return output_dir

    count = 0

    for xml_path in xml_files:
        output_xml = output_dir / xml_path.name
        try:
            simplify_polygons_page(xml_path, output_xml, tolerance)
            print(f"Simplified zone polygons for: {xml_path.name}")
            count += 1
        except Exception as e:
            print(f"Skip (error): {xml_path} -> {e}")

    print(f"Done. Wrote {count} simplified XML file(s) to: {output_dir}")
    return output_dir