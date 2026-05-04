import json
from pathlib import Path
import cv2
import supervision as sv
from inference_sdk import InferenceHTTPClient
from tqdm import tqdm


image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}

# Adjust this threshold in function of how confident you want the model to be in regards to the prediction
# A threshold that is to high (close to 1) can result in having less predictions
inference_confidence = 0.40


def load_config(config_path):
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config file: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    required_keys = {"api_url", "api_key", "model_id"}
    missing = required_keys - set(config.keys())
    if missing:
        raise ValueError(
            f"Missing required config keys in {config_path}: {', '.join(sorted(missing))}"
        )
    return config


def build_annotators():
    """
    Supervision annotators for segmentation masks, bboxes, and labels
    """
    mask_annotator = sv.MaskAnnotator()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.CENTER)
    return mask_annotator, box_annotator, label_annotator


def iter_images(input_dir):
    for img_path in sorted(input_dir.iterdir()):
        if img_path.is_file() and img_path.suffix.lower() in image_extensions:
            yield img_path


def save_json(result, json_dir, stem):
    """
    Save inference result as JSON file
    """
    json_path = json_dir / f"{stem}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    return json_path


def annotate_image(image, result, annotators, conf_threshold):
    """
    Draw masks, boxes, and labels on an image from inference results
    """
    mask_annotator, box_annotator, label_annotator = annotators

    detections = sv.Detections.from_inference(result)
    detections = detections[detections.confidence >= conf_threshold]

    class_names = detections.data.get("class_name", [])
    confidences = detections.confidence if detections.confidence is not None else []
    labels = [f"{class_name} {conf:.2f}" for class_name, conf in zip(class_names, confidences)]

    annotated = image.copy()
    annotated = mask_annotator.annotate(annotated, detections)
    annotated = box_annotator.annotate(annotated, detections)
    annotated = label_annotator.annotate(annotated, detections, labels)
    return annotated


def process_image(img_path, client, model_id, json_dir, annotated_dir, annotators, conf_threshold):
    """
    Run inference on one image, save JSON file and annotated image file
    """
    result = client.infer(str(img_path), model_id=model_id)
    save_json(result, json_dir, img_path.stem)

    image = cv2.imread(str(img_path))
    if image is None:
        return False

    annotated = annotate_image(image, result, annotators, conf_threshold)
    out_path = annotated_dir / img_path.name
    cv2.imwrite(str(out_path), annotated)
    return True


def run_detection(input_dir, json_dir, annotated_dir, config_path, conf_threshold = inference_confidence):
    """
    Run RF-DETR inference on all images in input_dir
    """
    if not input_dir.exists():
        raise FileNotFoundError(f"Missing input directory: {input_dir}")

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

    json_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(config_path)
    client = InferenceHTTPClient(api_url=config["api_url"], api_key=config["api_key"],)
    model_id = config["model_id"]

    annotators = build_annotators()

    images = list(iter_images(input_dir))
    if not images:
        print(f"No images found in: {input_dir}")
        return json_dir, annotated_dir

    count = 0
    failed = []

    with tqdm(
        images,
        desc="Detecting zones",
        unit="img",
        bar_format="{desc}: {n}/{total} |{bar}| {percentage:3.0f}% [{elapsed}<{remaining}]",
    ) as progress:
        for img_path in progress:
            progress.set_postfix(file=img_path.name, refresh=False)
            try:
                ok = process_image(img_path=img_path, client=client, model_id=model_id, json_dir=json_dir, 
                                   annotated_dir=annotated_dir, annotators=annotators, conf_threshold=conf_threshold,)
                if ok:
                    count += 1
                else:
                    failed.append((img_path, "could not read image with OpenCV"))
            except Exception as e:
                failed.append((img_path, str(e)))

    print(f"\nDone. Processed {count}/{len(images)} image(s).")

    if failed:
        print(f"{len(failed)} image(s) skipped:")
        for img_path, reason in failed:
            print(f"  - {img_path.name}: {reason}")

    return json_dir, annotated_dir