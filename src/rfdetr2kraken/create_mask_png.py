import json
from pathlib import Path
import cv2
import numpy as np


def read_rfdetr_model_predictions(data):
    width = int(data["image"]["width"])
    height = int(data["image"]["height"])

    regions = []

    for pred in data.get("predictions") or []:
        cls = pred.get("class")
        points = pred.get("points") or []
        
        if not cls or not points:
            continue

        polygon = [(int(p["x"]), int(p["y"])) for p in points]

        regions.append({"class": cls, "polygon": polygon})

    return width, height, regions

def read_math_model_predictions(data):
    width = int(data["img_size"][1])
    height = int(data["img_size"][0])

    math_regions = []

    for pred in data.get("math") or []:
        points = pred.get("polygon") or []

        if not points:
            continue

        polygon = [(int(x), int(y)) for x, y in points]

        math_regions.append({"class" : "math", "polygon": polygon})

    return width, height, math_regions

def build_mask(zone_data, allowed, math_data=None):
    """
    Build a binary mask:
    - 0 (black)   = allowed / kept for segmentation
    - 255 (white) = ignored

    Math polygons from the second model predictions are painted white on top of the first model black, allowed polygons.
    """

    width, height, zones = read_rfdetr_model_predictions(zone_data)

    mask = np.full((height, width), 255, dtype=np.uint8)

    # Step 1 : Paint readable zones black

    for region in zones:
        if region["class"] in allowed:
            poly = np.array(region["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 0)

    #  Step 2 : Paint math zones white on top
    if math_data is not None:
        math_width, math_height, math_regions = read_math_model_predictions(math_data)

        if math_width != width or math_height != height:
            raise ValueError (f"Image size mismatch: "
                f"RF-DETR Zones JSON is {width}x{height}, "
                f"Math Predictions JSON is {math_width}x{math_height}")
        
        for region in math_regions:
            poly = np.array(region["polygon"], dtype=np.int32)
            cv2.fillPoly(mask, [poly], 255)

    return mask


def run_masks_from_zones(json_dir, mask_dir, allowed, math_json_dir=None):
    """
    Generate PNG masks from prediction JSON files.

    If math predictions are present (json files in provenance of the math segmentation model), 
    the math polygons will be masked out (white zones).
    """
    json_dir = Path(json_dir)
    mask_dir = Path(mask_dir)

    if math_json_dir is not None:
        math_json_dir = Path(math_json_dir)

    if not json_dir.exists():
        raise FileNotFoundError(f"Missing JSON directory: {json_dir}")

    if not json_dir.is_dir():
        raise NotADirectoryError(f"JSON path is not a directory: {json_dir}")

    if math_json_dir is not None:
        if not math_json_dir.exists():
            print(f"[Warning] Missing math JSON directory, continuing without math masks: {math_json_dir}")
            math_json_dir = None

        elif not math_json_dir.is_dir():
            print(f"[Warning] Math JSON path is not a directory, continuing without math masks: {math_json_dir}")
            math_json_dir = None

    mask_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return mask_dir

    count = 0

    for json_path in json_files:
        try:
            zone_data = json.loads(json_path.read_text(encoding="utf-8"))

            math_data = None

            if math_json_dir is not None:
                math_path = math_json_dir / json_path.name

                if math_path.exists():
                    math_data = json.loads(math_path.read_text(encoding="utf-8"))
                else:
                    print(f"No matching math JSON for: {json_path.name}")

            mask = build_mask(
                zone_data=zone_data,
                allowed=allowed,
                math_data=math_data,
            )

            output_mask = mask_dir / f"{json_path.stem}_mask.png"

            ok = cv2.imwrite(str(output_mask), mask)
            if not ok:
                raise IOError(f"cv2.imwrite failed for: {output_mask}")

            print(f"Wrote {output_mask}")
            count += 1

        except Exception as e:
            print(f"Skip (error): {json_path} -> {e}")

    print(f"Done. Wrote {count} mask(s) to: {mask_dir}")
    return mask_dir