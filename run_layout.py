import argparse
from pathlib import Path
from src.predictions.predict_api import run_detection
from src.rfdetr2kraken.binarize_images import run_binarization
from src.rfdetr2kraken.create_mask_png import run_masks_from_zones
from src.rfdetr2kraken.create_masked_images import run_masked_images
from src.rfdetr2kraken.create_pagexml_regions_from_json import run_json_to_regions_xml


ALLOWED_ZONES = {
    "MainZone",
    "MarginTextZone",
    "NumberingZone",
    "DigitizationArtefactZone",
}


def build_pipeline_dirs(output_dir):
    return {
        "predictions":         output_dir / "01_predictions_json",
        "predictions_rfdetr":  output_dir / "01_predictions_json" / "rf-detr",
        "predictions_maths":   output_dir / "01_predictions_json" / "maths",
        "annotated":           output_dir / "02_annotated_images",
        "regions":             output_dir / "03_regions_xml",
        "masks":               output_dir / "04_masks",
        "binarized":           output_dir / "05_binarized",
        "masked":              output_dir / "05b_masked_for_baselines",
    }


def run_full_pipeline(input_dir, output_dir, config_path):
    dirs = build_pipeline_dirs(output_dir)

    run_detection(
        input_dir=input_dir,
        json_dir=dirs["predictions_rfdetr"],
        annotated_dir=dirs["annotated"],
        config_path=config_path,
    )

    # TODO : add mathematical expression detection here

    run_json_to_regions_xml(
        json_dir=dirs["predictions_rfdetr"],
        xml_dir=dirs["regions"],
    )

    run_masks_from_zones(
        json_dir=dirs["predictions_rfdetr"],
        math_json_dir=dirs["predictions_maths"],
        mask_dir=dirs["masks"],
        allowed=ALLOWED_ZONES,
    )

    run_binarization(
        input_dir=input_dir,
        output_dir=dirs["binarized"],
    )

    run_masked_images(
        images_dir=dirs["binarized"],
        masks_dir=dirs["masks"],
        output_dir=dirs["masked"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run the full manuscript PAGE XML pipeline.")
    parser.add_argument("--input", required=True, type=Path, dest="input_dir")
    parser.add_argument("--output", required=True, type=Path, dest="output_dir")
    parser.add_argument("--config", required=True, type=Path, dest="config_path")
    parser.add_argument("--htr", type=str, default=None, help="Path to the HTR/OCR model, or model filename inside models/htr/.")

    args = parser.parse_args()

    final_dir = run_full_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config_path,
    )

    print(f"\nFinal output written to: {final_dir}")


if __name__ == "__main__":
    main()