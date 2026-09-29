"""Classify one image or a flat image directory with a pretrained ImageNet model."""

import csv
from pathlib import Path

from geophototagger.custom.dataset import SUPPORTED_IMAGE_SUFFIXES
from geophototagger.imagenet21k import classify_images

INPUT_PATH = Path("images")
OUTPUT_PATH = Path("imagenet_predictions.csv")
MODEL_NAME = "efficientnetv2"
THRESHOLD = 0.01
BATCH_SIZE = 8


def find_images(input_path: Path) -> list[Path]:
    """Find supported images in a file or a non-recursive directory."""
    if input_path.is_file():
        image_paths = (
            [input_path]
            if input_path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            else []
        )
    elif input_path.is_dir():
        image_paths = sorted(
            path
            for path in input_path.iterdir()
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
    else:
        raise ValueError(f"Image path does not exist: {input_path}")
    if not image_paths:
        raise ValueError(f"No supported images found in {input_path}")
    return image_paths


def main() -> None:
    """Predict and save one row per matching class and image."""
    image_paths = find_images(INPUT_PATH)
    results = classify_images(
        image_paths, model_name=MODEL_NAME, threshold=THRESHOLD, batch_size=BATCH_SIZE
    )
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            ("image_path", "model", "class_index", "synset", "label", "probability")
        )
        for image_path, matches in results:
            for match in matches:
                writer.writerow(
                    (
                        str(image_path),
                        MODEL_NAME,
                        match.class_index,
                        match.synset,
                        match.label,
                        match.probability,
                    )
                )
            if len(image_paths) == 1:
                for match in matches:
                    print(f"{match.label}: {match.probability:.4%} ({match.synset})")
                if not matches:
                    print(f"No classes above {THRESHOLD:.2%} for {image_path}")
    print(f"Wrote predictions for {len(image_paths)} images to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
