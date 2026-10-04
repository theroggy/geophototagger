"""Run a configured geophototagger training job directly."""

import logging
from pathlib import Path

from geophototagger.custom.classifier import train_model
from geophototagger.custom.dataset import (
    determine_classes,
    read_manifest,
    write_manifest,
)


def main() -> None:
    """Generate train/validation/test manifests and train the configured model."""
    # Edit these values for the training run you want to execute.
    train_data_dir = Path(r"X:\Monitoring\phototagger\trainingsdata\train")
    validation_data_dir = Path(r"X:\Monitoring\phototagger\trainingsdata\validation")
    test_data_dir = Path(r"X:\Monitoring\phototagger\trainingsdata\test")

    project = "maizemulch"
    version = "05.ratio"
    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    project_dir.mkdir(parents=True, exist_ok=True)
    training_dir = project_dir / "training" / version
    training_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = training_dir / "train-images.csv"
    validation_manifest_path = training_dir / "validation-images.csv"

    # Init logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(training_dir / "train.log"),
        ],
    )
    logging.info("Starting the training run.")

    test_dir = project_dir / "test" / version
    test_dir.mkdir(parents=True, exist_ok=True)
    test_manifest_path = test_dir / "test-images.csv"

    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"phototagger-{project}_{version}.keras"
    evaluation_dir = training_dir / "evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    classes = {"korrelmais"}
    image_size = (512, 512)
    crop_to_aspect_ratio = True
    backbone = "EfficientNetV2S"
    weights = "imagenet"
    evaluation_thresholds = (0.5, 0.6, 0.7, 0.8)
    frozen_backbone_epochs = 5
    unfrozen_backbone_layers = 0.1
    learning_rate = 1e-3
    head_dropout = 0.5
    head_l2 = 0.005
    reduce_lr_patience = 3
    reduce_lr_factor = 0.2
    min_learning_rate = 1e-6
    force = False
    batch_size = 16

    logging.info("Writing training manifests.")
    write_manifest(train_data_dir, train_manifest_path, classes=classes, force=force)
    write_manifest(
        validation_data_dir, validation_manifest_path, classes=classes, force=force
    )
    write_manifest(test_data_dir, test_manifest_path, classes=classes, force=force)

    train_images = read_manifest(train_manifest_path, dataset_root=train_data_dir)
    validation_images = read_manifest(
        validation_manifest_path, dataset_root=validation_data_dir
    )
    test_images = read_manifest(test_manifest_path, dataset_root=test_data_dir)

    # Train the model with the specified parameters.
    train_model(
        train_images,
        classes=determine_classes(train_images),
        output_path=model_path,
        validation_images=validation_images,
        test_images=test_images,
        image_size=image_size,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        backbone=backbone,
        weights=weights,
        class_weighting=False,
        batch_size=batch_size,
        frozen_backbone_epochs=frozen_backbone_epochs,
        unfrozen_backbone_layers=unfrozen_backbone_layers,
        learning_rate=learning_rate,
        head_dropout=head_dropout,
        head_l2=head_l2,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        min_learning_rate=min_learning_rate,
        # monitor_metric="precision",
        evaluation_dir=evaluation_dir,
        evaluation_thresholds=evaluation_thresholds,
        force=force,
    )


if __name__ == "__main__":
    main()
