import json
from pathlib import Path
import cv2
import numpy as np


def build_mask(data, allowed):
    """
    Build a binary mask:
    - 0 (black)   = allowed / kept for segmentation
    - 255 (white) = ignored
    """
    width = int(data["image"]["width"])
    height = int(data["image"]["height"])

    mask = np.full((height, width), 255, dtype=np.uint8)

    for pred in data.get("predictions") or []:
        if pred.get("class") not in allowed:
            continue

        points = pred.get("points") or []
        if not points:
            continue

        polygon = np.array(
            [[int(p["x"]), int(p["y"])] for p in points],
            dtype=np.int32,
        )
        cv2.fillPoly(mask, [polygon], 0)

    return mask


def run_masks_from_zones(json_dir, mask_dir, allowed):
    """
    Generate PNG masks from prediction JSON files
    """
    if not json_dir.exists():
        raise FileNotFoundError(f"Missing JSON directory: {json_dir}")

    if not json_dir.is_dir():
        raise NotADirectoryError(f"JSON path is not a directory: {json_dir}")

    mask_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in: {json_dir}")
        return mask_dir

    count = 0

    for json_path in json_files:
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            mask = build_mask(data, allowed)

            output_mask = mask_dir / f"{json_path.stem}_mask.png"
            cv2.imwrite(str(output_mask), mask)

            print(f"Wrote {output_mask}")
            count += 1

        except Exception as e:
            print(f"Skip (error): {json_path} -> {e}")

    print(f"Done. Wrote {count} mask(s) to: {mask_dir}")
    return mask_dir