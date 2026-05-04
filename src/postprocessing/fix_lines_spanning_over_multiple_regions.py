"""
This script iterates over each line in the XML Page file. 
Clips it so it fits inside the polygon (<TextRegion> element).
If a line goes accross multiple polygons, it splits it into pieces.
Then cleans the result by eliminating broken or very small pieces.
"""

from pathlib import Path
import argparse
from lxml import etree
from shapely.geometry import Polygon, LineString
import copy
import uuid

page_ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
ns = {"p": page_ns}

# Minimum allowed baseline length after clipping, set to 30 px by default
min_baseline_length = 30.0


def get_polygon(el, child_name = "Coords"):
    child = el.find(f"p:{child_name}", namespaces=ns)
    if child is None:
        return None
    raw = (child.get("points") or "").strip()
    if not raw:
        return None
    try:
        pts = [tuple(map(float, t.split(","))) for t in raw.split()]
        if len(pts) < 3:
            return None
        poly = Polygon(pts).buffer(0)
        return poly if not poly.is_empty else None
    except Exception:
        return None


def get_baseline(tl):
    child = tl.find("p:Baseline", namespaces=ns)
    if child is None:
        return None
    raw = (child.get("points") or "").strip()
    if not raw:
        return None
    try:
        pts = [tuple(map(float, t.split(","))) for t in raw.split()]
        if len(pts) < 2:
            return None
        line = LineString(pts)
        return line if line.length > 0 else None
    except Exception:
        return None


def set_polygon(el, poly):
    child = el.find("p:Coords", namespaces=ns)
    if child is None:
        child = etree.SubElement(el, f"{{{page_ns}}}Coords")
    pts = list(poly.exterior.coords)[:-1]  # drop closing duplicate
    child.set("points", " ".join(f"{int(round(x))},{int(round(y))}" for x, y in pts))


def set_baseline(tl, line):
    child = tl.find("p:Baseline", namespaces=ns)
    if child is None:
        child = etree.SubElement(tl, f"{{{page_ns}}}Baseline")
    child.set("points", " ".join(f"{int(round(x))},{int(round(y))}" for x, y in line.coords))


def longest_line(geom):
    """
    Extract the longest valid LineString.
    Useful because intersection may return MultiLineString.
    """
    if geom is None or geom.is_empty:
        return None
    if geom.geom_type == "LineString":
        return geom if geom.length > 0 else None
    if geom.geom_type in {"MultiLineString", "GeometryCollection"}:
        lines = [g for g in geom.geoms if g.geom_type == "LineString" and g.length > 0]
        return max(lines, key=lambda g: g.length, default=None)
    return None

def fix_lines(input_xml, output_xml):
    """
    Split lines if they spread multiple TextRegion elements.
    Removes invalid or short resulting lines.
    """
    tree = etree.parse(str(input_xml), etree.XMLParser(remove_blank_text=False))

    for tl in list(tree.xpath("//p:TextLine", namespaces=ns)):
        parent = tl.getparent()

        if parent is None or etree.QName(parent).localname != "TextRegion":
            if parent is not None:
                parent.remove(tl)
            continue

        mask = get_polygon(tl)
        baseline = get_baseline(tl)

        if mask is None or baseline is None:
            parent.remove(tl)
            continue

        # Find all TextRegions on the page
        page = tree.xpath("//p:Page", namespaces=ns)[0]
        all_regions = page.xpath(".//p:TextRegion", namespaces=ns)

        # Try to clip this line against every region it overlaps
        clipped_lines = []  # list of (region_el, clipped_mask, clipped_baseline)

        for region in all_regions:
            region_poly = get_polygon(region)
            if region_poly is None:
                continue
            
            # Intersect TextLine polygon with TextRegion polygon
            clipped_mask = mask.intersection(region_poly)

            # If multiple pieces, keep the largest one
            if clipped_mask.geom_type == "MultiPolygon":
                clipped_mask = max(clipped_mask.geoms, key=lambda p: p.area)

            if clipped_mask.is_empty or clipped_mask.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            
            # Clip baseline to fit the resulting mask
            clipped_baseline = longest_line(baseline.intersection(clipped_mask))

            # Ignore short or invalid baselines
            if clipped_baseline is None or clipped_baseline.length < min_baseline_length:
                continue

            clipped_lines.append((region, clipped_mask, clipped_baseline))

        # Remove the original line from its parent (will be replaced by clipped ones)
        parent.remove(tl)

        if not clipped_lines:
            # Line doesn't fit in any region -> drop it entirely
            continue

        # Insert one new valid TextLine per TextRegion
        for region, clipped_mask, clipped_baseline in clipped_lines:
            new_tl = copy.deepcopy(tl)
            
            # new ID
            new_tl.set("id", str(uuid.uuid4()))

            # Update mask polygon and baseline
            set_polygon(new_tl, clipped_mask)
            set_baseline(new_tl, clipped_baseline)

            region.append(new_tl)

    output_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_xml), encoding="UTF-8", xml_declaration=True, pretty_print=False)
    print(f"Processed: {input_xml.name} -> {output_xml}")

def fix_lines_batch(input_dir, output_dir):
    for xml_file in sorted(input_dir.glob("*.xml")):
        try:
            fix_lines(xml_file, output_dir / xml_file.name)
        except Exception as e:
            print(f"Skip: {xml_file.name} — {e}")

def main():
    parser = argparse.ArgumentParser(
        description="Clip PAGE XML TextLines to their parent TextRegion polygons."
    )
    parser.add_argument("--input",  required=True, type=Path, dest="input_dir")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for xml_file in sorted(args.input_dir.glob("*.xml")):
        try:
            fix_lines(xml_file, args.output_dir / xml_file.name)
        except Exception as e:
            print(f"Skip: {xml_file.name} — {e}")


if __name__ == "__main__":
    main()