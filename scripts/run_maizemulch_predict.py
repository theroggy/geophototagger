"""Report maize/mulch misclassifications for a labeled image directory."""

import json
import logging
from pathlib import Path

from geophototagger.custom.classifier import predict_images
from geophototagger.custom.dataset import discover_records


def main() -> None:
    """Predict a labeled directory and write threshold-specific reports."""
    input_dir = Path("Q:/phototagger/trainingdata_raw/stalmest/agrilens_stalmest_2025")
    input_dir = Path("Q:/phototagger/trainingdata_raw/varia/test_extra")
    project = "maizemulch"
    version = "02.large"
    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    model_path = project_dir / "models" / f"phototagger-{project}_{version}.keras"
    output_dir = input_dir.parent / f"{input_dir.name}_{version}"
    thresholds = (0.5, 0.6, 0.7, 0.8)
    batch_size = 32
    workers = 4

    logging.basicConfig(level=logging.INFO)

    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    vocabulary = metadata["labels"]
    records = discover_records(input_dir, label_whitelist=vocabulary)
    if not records:
        raise ValueError(f"No labeled images found in {input_dir}")
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or not all(0 <= threshold <= 1 for threshold in thresholds)
    ):
        raise ValueError("Thresholds must be unique values between 0 and 1")

    logging.info(
        "Writing prediction reports and misclassified images to %s", output_dir
    )
    predict_images(
        model_path,
        records,
        batch_size=batch_size,
        workers=workers,
        output_dir=output_dir,
        thresholds=thresholds,
    )


if __name__ == "__main__":
    main()
