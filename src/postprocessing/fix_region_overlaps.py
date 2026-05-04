"""
Subtract MarginTextZone zones from MainZone zones.
"""

from pathlib import Path
import argparse
from lxml import etree
from shapely.geometry import Polygon
from shapely.ops import unary_union


page_ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ns = {"p": page_ns}

def get_polygon(region):
    """Read <Coords points="..."> as a Shapely Polygon."""
    coords = region.find(f"p:Coords", namespaces=ns)
    if coords is None:
        return None
    raw = (coords.get("points") or "").strip()
    if not raw:
        return None
    try:
        pts = [tuple(map(float, t.split(","))) for t in raw.split()]
        poly = Polygon(pts).buffer(0)   # buffer(0) repairs invalid rings
        return poly if not poly.is_empty and len(pts) >= 3 else None
    except Exception:
        return None


def set_polygon(region, poly):
    """Write a Shapely Polygon back into <Coords points="...">."""
    coords = region.find(f"p:Coords", namespaces=ns)
    if coords is None:
        coords = etree.SubElement(region, f"{{{page_ns}}}Coords")
    pts = list(poly.exterior.coords)[:-1]   # drop closing duplicate
    coords.set("points", " ".join(f"{int(round(x))},{int(round(y))}" for x, y in pts))


def get_region_type(region):
    """Extract region type"""

    # Get type from the "type" attribute
    t = (region.get("type") or "").strip()
    if t:
        return t
    
    # Fallback to "custom" attribute
    custom = region.get("custom") or ""
    for zone_type in {"MainZone", "MarginTextZone"}:
        if f"type:{zone_type}" in custom or f"type={zone_type}" in custom:
            return zone_type
    return ""


def fix_overlaps(input_xml, output_xml):
    """
    1. Extract all <TextRegion> elements
    2. Substract MarginTextZone polygons from MainZone polygons
    3. Update <TextRegion> points
    """
    tree = etree.parse(str(input_xml), etree.XMLParser(remove_blank_text=False))
    regions = tree.xpath("//p:TextRegion", namespaces=ns)

    main_regions, margin_polys = [], []
    for r in regions:
        poly = get_polygon(r)
        if poly is None:
            continue
        t = get_region_type(r)
        if t == "MainZone":
            main_regions.append((r, poly))
        elif t == "MarginTextZone":
            margin_polys.append(poly)

    if margin_polys:
        # Create a temporary combined Polygon (margin polygons touch/overlap) 
        # or MultiPolygons (margin polygons do not touch/overlap)
        margin_union = unary_union(margin_polys)
        for region, main_poly in main_regions:
            diff = main_poly.difference(margin_union)
            # If the remaining MainZone is fragmented, keep only the largest piece.
            if diff.geom_type == "MultiPolygon":
                diff = max(diff.geoms, key=lambda p: p.area)
            if diff.is_empty:
                region.getparent().remove(region)
            else:
                set_polygon(region, diff)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_xml), encoding="UTF-8", xml_declaration=True, pretty_print=False)

def fix_overlaps_batch(input_dir, output_dir):
    for xml_file in sorted(input_dir.glob("*.xml")):
        try:
            fix_overlaps(xml_file, output_dir / xml_file.name)
            print(f"Processed: {xml_file.name}")
        except Exception as e:
            print(f"Skip: {xml_file.name} — {e}")

def main():
    """
    Example usage:
        fix_region_overlaps.py --input input_folder --output output_folder
    """
    parser = argparse.ArgumentParser(description="Subtract MarginTextZone polygons from MainZone polygons in PAGE XML.")
    parser.add_argument("--input",  required=True, type=Path, dest="input_dir")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for xml_file in sorted(args.input_dir.glob("*.xml")):
        try:
            fix_overlaps(xml_file, args.output_dir / xml_file.name)
            print(f"Processed: {xml_file.name}")
        except Exception as e:
            print(f"Skip: {xml_file.name} — {e}")


if __name__ == "__main__":
    main()