"""Detect objects in one image or a flat image directory with OWLv2."""

from pathlib import Path

from geophototagger.owlv2 import DEFAULT_MODEL, detect_objects

INPUT_PATH = Path("Q:/phototagger/trainingdata_raw/varia/test_extra")
OUTPUT_DIR = Path("Q:/phototagger/trainingdata_raw/varia/test_extra_owlv2")
LABELS = [
    "car",
    "person",
    "dog",
    "cat",
    "bicycle",
    "tractor",
    "house",
    "tree",
    "sky",
    "grassland",
    "field",
    "bare soil",
    "phone",
    "laptop",
    "tablet",
    "screen",
    "keyboard",
    "mouse",
    "moire",
]
MODEL_NAME = DEFAULT_MODEL
THRESHOLD = 0.1
BATCH_SIZE = 4


def main() -> None:
    """Detect and append one row per image, skipping already-known images."""
    detect_objects(
        INPUT_PATH,
        labels=LABELS,
        model_name=MODEL_NAME,
        threshold=THRESHOLD,
        batch_size=BATCH_SIZE,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
