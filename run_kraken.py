import argparse
from pathlib import Path
from src.rfdetr2kraken.kraken_segmentation_and_htr import segment_pages_batch
from src.rfdetr2kraken.merge_kraken_lines_back_into_my_regions import merge_zone_types_and_lines_batch
from src.rfdetr2kraken.simplify_zones_polygons_xml import simplify_polygons_batch
from src.postprocessing.fix_region_overlaps import fix_overlaps_batch
from src.postprocessing.fix_lines_spanning_over_multiple_regions import fix_lines_batch
from src.extract_diagrams.extract_figures import DEFAULT_REGION_TYPE, extract_regions_batch


VALID_EXTRACT_MODES = ("bbox", "polygon")

def build_pipeline_dirs(output_dir):
    return {
        "predictions": output_dir / "01_predictions_json",
        "annotated": output_dir / "02_annotated_images",
        "regions": output_dir / "03_regions_xml",
        "masks": output_dir / "04_masks",
        "binarized": output_dir / "05_binarized",
        "masked": output_dir / "05b_masked_for_baselines",
        "baselines": output_dir / "06_baselines_and_htr",
        "merged": output_dir / "07_merged_xml",
        "simplified_polygons": output_dir / "08_simplified_pagexml",
        "fixed_regions": output_dir / "09_fixed_region_overlaps",
        "fixed_lines": output_dir / "10_fixed_lines",
        "extracted": output_dir / "11_extracted_regions",
    }


def resolve_htr_model(htr_arg):
    if htr_arg is None:
        return None

    model_path = Path(htr_arg)
    if not model_path.exists():
        model_path = Path("htr_models") / htr_arg

    if not model_path.exists():
        raise FileNotFoundError(f"HTR model not found: {model_path}")

    return model_path


def run_kraken_pipeline(input_dir, output_root, htr_model, extract_types, extract_mode = "bbox"):
    dirs = build_pipeline_dirs(output_root)
    
    segment_pages_batch(
        regions_dir=dirs["regions"],
        images_dir=input_dir,
        masks_dir=dirs["masks"],
        output_dir=dirs["baselines"],
        htr_model=htr_model,
    )

    merge_zone_types_and_lines_batch(
        regions_dir=dirs["regions"],
        baselines_dir=dirs["baselines"],
        output_dir=dirs["merged"],
    )

    simplify_polygons_batch(
        input_dir=dirs["merged"],
        output_dir=dirs["simplified_polygons"],
        tolerance=15,
    )

    fix_overlaps_batch(
            input_dir=dirs["simplified_polygons"],
            output_dir=dirs["fixed_regions"])
    
    fix_lines_batch(
        input_dir=dirs["fixed_regions"],
        output_dir=dirs["fixed_lines"])
    
    if extract_types is not None:
        extracted_root = dirs["extracted"] / extract_mode
        types_to_run = [t for t in extract_types]
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

    return dirs["simplified_polygons"]


def main():
    parser = argparse.ArgumentParser(
        description="Run the Kraken stage of the manuscript pipeline."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        dest="input_dir",
        help="Path to the folder containing the input images to process.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        dest="output_dir",
        help="Path to the folder where output results will be saved.",
    )
    parser.add_argument(
        "--htr",
        type=str,
        default=None,
        help="Path to the HTR/OCR model, or model filename inside htr_models/.",
    )

    parser.add_argument(
        "--extract", 
        nargs="*", 
        default=None, 
        metavar="TYPE",
        help=("Enable region extraction. "
            f"If no values are given, defaults to '{DEFAULT_REGION_TYPE}'. "
            "Pass space separated values to extract multiple types, e.g.\n"
            "  --extract GraphicZone-figure\n"
            "  --extract MainZone MarginTextZone GraphicZone-figure"))
    parser.add_argument(
        "--extract-mode",
        choices=VALID_EXTRACT_MODES,
        default="bbox",
        help=("How to crop matching regions in step 11. "
            "'bbox' (default) saves a rectangular crop around the region's polygon"
            "'polygon' saves the polygon itself."))

    args = parser.parse_args()

    if args.extract is None:
        extract_types = None
    else:
        extract_types = args.extract if args.extract else [DEFAULT_REGION_TYPE]

    final_dir = run_kraken_pipeline(
        input_dir=args.input_dir,
        output_root=args.output_dir,
        htr_model=resolve_htr_model(args.htr),
        extract_types=extract_types,
        extract_mode=args.extract_mode,
    )

    print(f"\nKraken stage finished. Final PAGE XML written to: {final_dir}")


if __name__ == "__main__":
    main()