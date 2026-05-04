import argparse
from pathlib import Path

from src.predictions.predict_api import run_detection
from src.rfdetr2kraken.binarize_images import run_binarization
from src.rfdetr2kraken.create_mask_png import run_masks_from_zones
from src.rfdetr2kraken.create_masked_images import run_masked_images
from src.rfdetr2kraken.create_pagexml_regions_from_json import run_json_to_regions_xml
from src.rfdetr2kraken.kraken_segmentation_and_htr_v7 import segment_pages_batch
from src.rfdetr2kraken.merge_kraken_lines_back_into_my_regions import merge_zone_types_and_lines_batch
from src.rfdetr2kraken.simplify_zones_polygons_xml import simplify_polygons_batch
from src.postprocessing.fix_region_overlaps import fix_overlaps_batch
from src.postprocessing.fix_lines_spanning_over_multiple_regions import fix_lines_batch
from src.extract_diagrams.extract_figures import DEFAULT_REGION_TYPE, extract_regions_batch

# Allowed Zones for baseline (line) detection. 
# Graphic, Mathematical zones are ignored.
ALLOWED_ZONES = {
    "MainZone",
    "MarginTextZone",
    "NumberingZone",
    "DigitizationArtefactZone",
}

# Used to crop a zone out of an image
# User can select between bbox and/or polygon shape image crop
VALID_EXTRACT_MODES = ("bbox", "polygon")


def build_pipeline_dirs(output_dir):
    return {
        "predictions":         output_dir / "01_predictions_json",
        "annotated":           output_dir / "02_annotated_images",
        "regions":             output_dir / "03_regions_xml",
        "masks":               output_dir / "04_masks",
        "binarized":           output_dir / "05_binarized",
        "masked":              output_dir / "05b_masked_for_baselines",
        "baselines":           output_dir / "06_baselines_and_htr",
        "merged":              output_dir / "07_merged_xml",
        "simplified_polygons": output_dir / "08_simplified_pagexml",
        "fixed_regions":       output_dir / "09_fixed_region_overlaps",
        "fixed_lines":         output_dir / "10_fixed_lines",
        "extracted":           output_dir / "11_extracted_regions",
    }


def has_outputs(out_dir):
    """
    Return True if the directory exists and contains at least one entry.
    Allows to check if a step has already been run.
    """
    return out_dir.exists() and any(out_dir.iterdir())


def should_skip(step_name, out_dir, force):
    """
    Skip step if output already exists (default)
    --force : runs the step, even if files already exist. 
    This will overwrite existing files. 
    """
    if force:
        return False
    if has_outputs(out_dir):
        print(f"[Skip] {step_name}: outputs already present in {out_dir}")
        return True
    return False


def resolve_htr_model(htr_arg):
    if htr_arg is None:
        return None

    model_path = Path(htr_arg)
    if not model_path.exists():
        model_path = Path("htr_models") / htr_arg
    if not model_path.exists():
        raise FileNotFoundError(f"HTR model not found: {model_path}")

    return model_path


def run_full_pipeline(input_dir, output_dir, config_path, 
                      htr_model, force, extract_types, extract_mode = "bbox"):
    dirs = build_pipeline_dirs(output_dir)
    print("\n=== [Step 1 and 2] Zone prediction and image annotation ===")
    # Step 01 + 02: zones prediction (produces predictions JSON files and annotated images)
    if not should_skip("01_predictions / 02_annotated", dirs["predictions"], force):
        run_detection(
            input_dir=input_dir,
            json_dir=dirs["predictions"],
            annotated_dir=dirs["annotated"],
            config_path=config_path)

    print("\n=== [Step 3] JSON regions -> Page XML regions ===")    
    # Step 03: JSON predictions -> PAGE XML regions
    if not should_skip("03_regions_xml", dirs["regions"], force):
        run_json_to_regions_xml(
            json_dir=dirs["predictions"],
            xml_dir=dirs["regions"])
        
    print("\n=== [Step 4] Binary masks with allowed zones ===")    
    # Step 04: binary masks from allowed zones
    if not should_skip("04_masks", dirs["masks"], force):
        run_masks_from_zones(
            json_dir=dirs["predictions"],
            mask_dir=dirs["masks"],
            allowed=ALLOWED_ZONES)

    print("\n=== [Step 5] Binarize images ===")   
    # Step 05: binarized source images
    if not should_skip("05_binarized", dirs["binarized"], force):
        run_binarization(
            input_dir=input_dir,
            output_dir=dirs["binarized"])
    
    print("\n=== [Step 5b] Mask zones in binarized images ===")
    # Step 05b: masked binarized images (used as input for Kraken baselines)
    # Only applies to Kraken 0.7 version
    
    if not should_skip("05b_masked_for_baselines", dirs["masked"], force):
        run_masked_images(
            images_dir=dirs["binarized"],
            masks_dir=dirs["masks"],
            output_dir=dirs["masked"])

    print("\n=== [Step 6] Kraken baselines segmentation ===") 
    # Step 06: Kraken segmentation + HTR
    if not should_skip("06_baselines_and_htr", dirs["baselines"], force):
        segment_pages_batch(
            regions_dir=dirs["regions"],
            images_dir=dirs["masked"],
            output_dir=dirs["baselines"],
            htr_model=htr_model)

    print("\n=== [Step 7] Merge detected baselines with original detected zones ===")  
    # Step 07: merge original zone types with Kraken lines
    if not should_skip("07_merged_xml", dirs["merged"], force):
        merge_zone_types_and_lines_batch(
            regions_dir=dirs["regions"],
            baselines_dir=dirs["baselines"],
            output_dir=dirs["merged"])

    print("\n=== [Step 8] Simplify polygon coordinates ===")
    # Step 08: simplify polygon coordinates
    if not should_skip("08_simplified_pagexml", dirs["simplified_polygons"], force):
        simplify_polygons_batch(
            input_dir=dirs["merged"],
            output_dir=dirs["simplified_polygons"],
            tolerance=15)
        
    print("\n=== [Step 9] Subtract MarginTextZone polygons from MainZone polygons ===")    
    # Step 09: subtract MarginTextZone polygons from MainZone polygons
    if not should_skip("09_fixed_region_overlaps", dirs["fixed_regions"], force):
        fix_overlaps_batch(
            input_dir=dirs["simplified_polygons"],
            output_dir=dirs["fixed_regions"])

    print("\n=== [Step 10] Split lines spanning multiple regions. Drop anything outside all regions ===")    
    # Step 10: split lines spanning multiple regions, drop anything outside all regions
    if not should_skip("10_fixed_lines", dirs["fixed_lines"], force):
        fix_lines_batch(
            input_dir=dirs["fixed_regions"],
            output_dir=dirs["fixed_lines"])

    # Step 11 (optional): extract crops of selected region types.
    # Runs only when --extract was passed. The "extract_mode" ("bbox" or
    # "polygon") becomes a subfolder under 11_extracted_regions, and each
    # region type then gets its own subfolder under that. Types whose folder
    # already has files are skipped (unless --force).
    # Layout: 11_extracted_regions/<extract_mode>/<region_type>/...
    if extract_types is not None:
        print("\n=== [Step 11] Extract crops of selected region types ===")    
        extracted_root = dirs["extracted"] / extract_mode
        types_to_run = [t for t in extract_types if force or not has_outputs(extracted_root / t)]
        skipped = [t for t in extract_types if t not in types_to_run]
        for t in skipped:
            print(f"[Skip] 11_extracted_regions/{extract_mode}/{t}: outputs already present")

        if types_to_run:
            extract_regions_batch(
                xml_dir=dirs["regions"],
                img_dir=input_dir,
                out_dir=extracted_root,
                region_types=types_to_run,
                mode=extract_mode,
            )

    return dirs["fixed_lines"]


def main():
    parser = argparse.ArgumentParser(description="Run the full manuscript PAGE XML pipeline.")
    parser.add_argument("--input",  required=True, type=Path, dest="input_dir", help="Path to the folder containing the input images to process.")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir", help="Path to the folder where output results will be saved.")
    parser.add_argument("--config", required=True, type=Path, dest="config_path", help="Path to config file for the Roboflow API.")
    parser.add_argument("--htr", type=str, default=None, help="Path to HTR/OCR model, or filename inside models/htr/.")
    parser.add_argument("-f", "--force", action="store_true", help="Force re-run every step even if output files already exist. This will override existing files.")
    parser.add_argument("--extract", nargs="*", default=None, metavar="TYPE",
        help=("Enable zone extraction. "
            f"If no types are provided, defaults to '{DEFAULT_REGION_TYPE}'. "
            "Pass space separated values to extract multiple types, e.g.\n"
            "  --extract GraphicZone-figure\n"
            "  --extract MainZone MarginTextZone GraphicZone-figure"))
    parser.add_argument("--extract-mode", choices=VALID_EXTRACT_MODES, default="bbox",
        help=("How to crop matching regions in step 11."
            "'bbox' (default) saves a rectangular crop around the region's polygon.\n"
            "'polygon' saves the polygon itself"))

    args = parser.parse_args()

    if args.extract is None:
        extract_types = None
    else:
        extract_types = args.extract if args.extract else [DEFAULT_REGION_TYPE]

    run_full_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
        htr_model=resolve_htr_model(args.htr),
        force=args.force,
        extract_types=extract_types,
        extract_mode=args.extract_mode,
    )

if __name__ == "__main__":
    main()