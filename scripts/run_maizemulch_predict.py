"""Report maize/mulch misclassifications for a labeled image directory."""

import json
import logging
from pathlib import Path

from geophototagger.custom.classifier import (
    predict_images,
    write_prediction_evaluation,
)
from geophototagger.custom.dataset import discover_records


def main() -> None:
    """Predict a labeled directory and write threshold-specific reports."""
    base_dir = Path("Q:/phototagger/trainingdata_raw")
    input_dirs = [
        base_dir / "varia/test",
        base_dir / "varia/test_extra",
        base_dir / "korrelmais/agrilens_korrelmais_2023",
        base_dir / "korrelmais/agrilens_korrelmais_2024",
        base_dir / "korrelmais/agrilens_korrelmais_2025",
        base_dir / "stalmest/agrilens_stalmest_2025",
        base_dir / "grasklaver/images",
    ]

    project = "maizemulch"
    version = "04-b1"
    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    model_path = project_dir / "models" / f"phototagger-{project}_{version}.keras"

    thresholds = (0.5, 0.6)
    batch_size = 12
    workers = 4

    logging.basicConfig(level=logging.INFO)

    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or not all(0 <= threshold <= 1 for threshold in thresholds)
    ):
        raise ValueError("Thresholds must be unique values between 0 and 1")
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    vocabulary = metadata["labels"]

    for input_dir in input_dirs:  # Loop over input directories if needed
        output_dir = input_dir.parent / f"{input_dir.name}_{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        records = discover_records(input_dir, label_whitelist=vocabulary)
        if not records:
            raise ValueError(f"No labeled images found in {input_dir}")

        logging.info(
            "Writing prediction reports and misclassified images to %s", output_dir
        )
        predictions = predict_images(
            model_path,
            records,
            batch_size=batch_size,
            workers=workers,
            output_path=output_dir / "predictions.csv",
        )
        predictions_by_path = {
            str(record.image_path.resolve()): prediction
            for record, prediction in zip(records, predictions, strict=True)
        }
        write_prediction_evaluation(
            records,
            vocabulary,
            predictions_by_path,
            output_dir,
            thresholds,
        )


if __name__ == "__main__":
    main()
