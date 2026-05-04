"""Kraken overwrites the initial detected zones and their corresponding classes inside the <TextRegion> elements.
This code will reassign lines back into our initial zones and with the correct classes.
First pass : check if the midpoint of the baseline or of the mask polygon falls inside one of the zone
If first pass fails : check for IoU between TextLine bbox and TextRegion bbox
Output : new XML Page file with the correct <TextRegion> elements and the correct reassigned <TextLine> elements."""


from pathlib import Path
import cv2
import numpy as np
from lxml import etree


page_ns = "http://schema.primaresearch.org/PAGE/gts/pagecontent/2019-07-15"
xsi_ns = "http://www.w3.org/2001/XMLSchema-instance"
ns = {"p": page_ns}

parser = etree.XMLParser(remove_blank_text=False)


def parse_points(points_str):
    return [(int(x), int(y))
        for x, y in (tok.split(",") for tok in points_str.strip().split())]


def bbox(pts):
    """
    Compute bbox from a list of points.

    Returns: (mix_x, min_y, max_x, max_y)
    """
    xs, ys = zip(*pts)
    return min(xs), min(ys), max(xs), max(ys)


def iou(a,b):
    """
    Compute Intersection over Union (IoU) between two bboxes
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union else 0.0


def get_baseline_midpoint(textline_el, textline_bb):
    """ 
    Get a representative point of a TextLine : the middle point.

    Fallback to center of the TextLine bbox.
    """
    baseline_el = textline_el.find("p:Baseline", namespaces=ns)
    if baseline_el is not None and baseline_el.get("points"):
        baseline_pts = parse_points(baseline_el.get("points"))
        return baseline_pts[len(baseline_pts) // 2]

    x1, y1, x2, y2 = textline_bb
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def assign_line_to_region(textline_el, regions):
    """
    Assign a TextLine to the best matching TextRegion.

    Strategy: 
    1. Try by checking if the midpoint of the baseline falls inside the TextRegion
    2. Fallback to IoU score
    """
    textline_coords = textline_el.find("p:Coords", namespaces=ns)
    if textline_coords is None or not textline_coords.get("points"):
        return None

    textline_poly = parse_points(textline_coords.get("points"))
    textline_bb = bbox(textline_poly)
    midpoint = get_baseline_midpoint(textline_el, textline_bb)

    # Step1 : Point inside the polygon
    for region_el, region_poly, _ in regions:
        if region_poly is None:
            continue
        if cv2.pointPolygonTest(region_poly, midpoint, False) >= 0:
            return region_el

    # Step2 : IoU fallback
    best_region = None
    best_score = -1.0

    for region_el, _, region_bb in regions:
        if region_bb is None:
            continue
        score = iou(textline_bb, region_bb)
        if score > best_score:
            best_score = score
            best_region = region_el

    return best_region


def collect_regions(regions_tree):
    """
    Extract all TextRegion elements from the XML and.
    
    Extract Polygon (precise geometry - for step1, check if point is inside the region) 
    and Bbox (approximate geometry - for step2/fallback IoU comparison) representations
    
    Expected output : [(region1_xml, poly1, bbox1),
    (region2_xml, poly2, bbox2),...]
    """

    regions = []

    for region_el in regions_tree.xpath("//p:TextRegion", namespaces=ns):
        coords_el = region_el.find("p:Coords", namespaces=ns)

        if coords_el is None or not coords_el.get("points"):
            regions.append((region_el, None, None))
            continue

        poly_pts = parse_points(coords_el.get("points"))
        poly = np.array(poly_pts, dtype=np.int32)
        regions.append((region_el, poly, bbox(poly_pts)))

    return regions


def remove_existing_lines(regions_tree):
    """
    Remove all existing TextLine elements from the TextRegion elements

    They will be reassigned later.
    """
    for textline_el in regions_tree.xpath("//p:TextLine", namespaces=ns):
        parent = textline_el.getparent()
        if parent is not None:
            parent.remove(textline_el)


def assign_baselines_to_regions(regions_tree, baselines_tree, regions):
    """
    For each <TextLine>, find the best matching <TextRegion> and assign/copy it to it.

    If no region matches, skip it.

    Returns: 
    - number of assigned/copied lines
    - total number of lines
    """

    baseline_lines = baselines_tree.xpath("//p:TextLine", namespaces=ns)
    assigned = 0

    for textline_el in baseline_lines:
        best_region = assign_line_to_region(textline_el, regions)
        if best_region is None:
            continue
        
        if best_region.text is None or best_region.text.strip() == "":
            best_region.text = "\n      "
        
        new_line = etree.fromstring(etree.tostring(textline_el))

        children = list(best_region)
        if children and children[-1].tail is not None:
            new_line.tail = children[-1].tail
        else:
            new_line.tail = "\n      "

        best_region.append(new_line)
        assigned += 1

    return assigned, len(baseline_lines)


def build_output_tree(regions_tree, metadata_el):
    """
    Build the final new XML Page file.
    """
    schema_location = f"{page_ns} {page_ns}/pagecontent.xsd"

    new_root = etree.Element(
        f"{{{page_ns}}}PcGts",
        nsmap={None: page_ns, "xsi": xsi_ns},
    )
    new_root.set(etree.QName(xsi_ns, "schemaLocation"), schema_location)

    if metadata_el is not None:
        new_root.append(metadata_el)

    regions_root = regions_tree.getroot()
    for child in list(regions_root):
        if etree.QName(child).localname == "Metadata":
            continue

        if etree.QName(child).localname == "Page":
            img = child.get("imageFilename")
            if img:
                child.set("imageFilename", Path(img).name)

        new_root.append(child)

    return etree.ElementTree(new_root)


def merge_zone_types_and_lines(regions_xml, baselines_xml, output_xml):
    """
    Load region and baselines.
    Assign lines to regions.
    Merge them into the final XML Page file.
    """
    regions_tree = etree.parse(str(regions_xml), parser)
    baselines_tree = etree.parse(str(baselines_xml), parser)

    regions = collect_regions(regions_tree)
    remove_existing_lines(regions_tree)

    assigned, total = assign_baselines_to_regions(regions_tree=regions_tree, 
                                                  baselines_tree=baselines_tree, regions=regions)

    metadata_el = baselines_tree.find(".//{*}Metadata")
    metadata_el = etree.fromstring(etree.tostring(metadata_el)) if metadata_el is not None else None

    output_tree = build_output_tree(regions_tree, metadata_el)
    output_xml.parent.mkdir(parents=True, exist_ok=True)
    output_tree.write(str(output_xml), encoding="UTF-8", 
                      xml_declaration=True, pretty_print=False)

    print(f"Assigned {assigned}/{total} lines")
    print(f"Wrote {output_xml}")


def merge_zone_types_and_lines_batch(regions_dir, baselines_dir, output_dir):
    if not regions_dir.exists():
        raise FileNotFoundError(f"Missing regions dir: {regions_dir}")

    if not baselines_dir.exists():
        raise FileNotFoundError(f"Missing baselines dir: {baselines_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    region_files = sorted(regions_dir.glob("*_regions.xml"))
    if not region_files:
        print(f"No region XML files found in: {regions_dir}")
        return output_dir

    count = 0

    for regions_xml in region_files:
        stem = regions_xml.name.removesuffix("_regions.xml")
        baselines_xml = baselines_dir / f"{stem}_baselines.xml"
        output_xml = output_dir / f"{stem}_merged.xml"

        if not baselines_xml.exists():
            print(f"Skip (missing baselines): {baselines_xml}")
            continue

        try:
            merge_zone_types_and_lines(regions_xml=regions_xml, 
                                       baselines_xml=baselines_xml, output_xml=output_xml)
            count += 1
        except Exception as e:
            print(f"Skip (error): {regions_xml} -> {e}")

    print(f"Done. Wrote {count} merged XML file(s) to: {output_dir}")
    return output_dir