"""Run a configured geophototagger training job directly."""

import logging
from pathlib import Path

from geophototagger.custom.classifier import train_model
from geophototagger.custom.dataset import (
    label_vocabulary,
    read_manifest,
    write_manifest,
)


def main() -> None:
    """Generate train/validation/test manifests and train the configured model."""
    logging.basicConfig(level=logging.INFO)

    logging.info("Starting the training run.")
    # Edit these values for the training run you want to execute.
    TRAIN_DATASET_ROOT = Path(r"X:\Monitoring\geophototagger\trainingsdata\train")
    VALIDATION_DATASET_ROOT = Path(
        r"X:\Monitoring\geophototagger\trainingsdata\validation"
    )
    TEST_DATASET_ROOT = Path(r"X:\Monitoring\geophototagger\trainingsdata\test")

    project = "maizemulch"
    version = "01.large"
    project_dir = Path(f"X:/Monitoring/geophototagger/{project}")
    project_dir.mkdir(parents=True, exist_ok=True)
    training_dir = project_dir / "training" / version
    training_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = training_dir / "train-images.csv"
    validation_manifest_path = training_dir / "validation-images.csv"

    test_dir = project_dir / "test" / version
    test_dir.mkdir(parents=True, exist_ok=True)
    test_manifest_path = test_dir / "test-images.csv"

    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"geophototagger-{project}_{version}.keras"
    misclassified_dir = training_dir / "misclassified"
    misclassified_dir.mkdir(parents=True, exist_ok=True)
    classes = {"korrelmais"}
    image_size = (1024, 1024)

    logging.info("Writing training manifests.")
    write_manifest(
        TRAIN_DATASET_ROOT,
        train_manifest_path,
        classes=classes,
    )
    write_manifest(
        VALIDATION_DATASET_ROOT,
        validation_manifest_path,
        classes=classes,
    )
    write_manifest(
        TEST_DATASET_ROOT,
        test_manifest_path,
        classes=classes,
    )
    records = read_manifest(train_manifest_path, dataset_root=TRAIN_DATASET_ROOT)
    validation_records = read_manifest(
        validation_manifest_path, dataset_root=VALIDATION_DATASET_ROOT
    )
    test_records = read_manifest(test_manifest_path, dataset_root=TEST_DATASET_ROOT)
    train_model(
        records,
        label_vocabulary(records),
        model_path,
        validation_records=validation_records,
        test_records=test_records,
        image_size=image_size,
        misclassified_dir=misclassified_dir,
        force=False,
    )


if __name__ == "__main__":
    main()
