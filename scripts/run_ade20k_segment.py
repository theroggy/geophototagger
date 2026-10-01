"""Segment one image or a flat image directory into ADE20K scene classes."""

from pathlib import Path

from geophototagger.ade20k import DEFAULT_MODEL, segment_images


def main() -> None:
    """Segment and append one row per image, skipping already-known images."""
    # input_path = Path("Q:/phototagger/trainingdata_raw/varia/test_extra")
    input_path = Path("X:/Monitoring/phototagger/trainingsdata/train")
    output_dir = input_path.parent / f"{input_path.name}_ade20k"
    model_name = DEFAULT_MODEL
    min_fraction = 0.02
    batch_size = 4

    segment_images(
        input_path,
        model_name=model_name,
        min_fraction=min_fraction,
        batch_size=batch_size,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
