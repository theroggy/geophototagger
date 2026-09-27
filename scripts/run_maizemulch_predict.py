"""Report maize/mulch misclassifications for a labeled image directory."""

import json
import logging
from pathlib import Path

from geophototagger.custom.classifier import predict_images
from geophototagger.custom.dataset import discover_records


def main() -> None:
    """Predict a labeled directory and write threshold-specific reports."""
    input_dir = Path(r"Q:\phototagger\trainingdata_raw\stalmest\agrilens_stalmest_2025")
    project = "maizemulch"
    version = "01.large"
    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    model_path = project_dir / "models" / f"geophototagger-{project}_{version}.keras"
    output_dir = input_dir.parent / f"{input_dir.name}_{version}/misclassified"
    thresholds = (0.5,)
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
