"""Classify one image or a flat image directory with a pretrained ImageNet model."""

from pathlib import Path

from geophototagger.imagenet21k import classify_images

INPUT_PATH = Path("Q:/phototagger/trainingdata_raw/varia/test_extra")
OUTPUT_DIR = Path("Q:/phototagger/trainingdata_raw/varia/test_extra_imagenet")
MODEL_NAME = "efficientnetv2"
THRESHOLD = 0.01
BATCH_SIZE = 8


def main() -> None:
    """Predict and append one row per image, skipping already-known images."""
    classify_images(
        INPUT_PATH,
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
