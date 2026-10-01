"""Classify one image or a flat image directory with a pretrained ImageNet model."""

from pathlib import Path

from geophototagger.imagenet21k import classify_images


def main() -> None:
    """Predict and append one row per image, skipping already-known images."""
    input_path = Path("Q:/phototagger/trainingdata_raw/varia/test_extra")
    output_dir = input_path.parent / f"{input_path.name}_imagenet"
    model_name = "efficientnetv2"
    threshold = 0.01
    batch_size = 8

    classify_images(
        input_path,
        model_name=model_name,
        threshold=threshold,
        batch_size=batch_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
