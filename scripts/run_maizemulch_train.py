"""Train a custom phototagger model."""

import logging
from pathlib import Path

from geophototagger.custom.classifier import (
    predict_images,
    train_model,
)
from geophototagger.custom.dataset import (
    determine_classes,
    discover_records,
    read_manifest,
    write_manifest,
)


def main() -> None:
    """Train a model and write evaluation results."""
    # Edit these values for the training run you want to execute.
    project = "maizemulch"
    version = "07-b0-0.7"
    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    project_dir.mkdir(parents=True, exist_ok=True)
    train_log_dir = project_dir / "train_log" / version
    train_log_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = train_log_dir / "train-images.csv"
    validation_manifest_path = train_log_dir / "validation-images.csv"

    trainingdata_dir = Path(f"X:/Monitoring/phototagger/{project}/trainingdata")
    train_data_dir = trainingdata_dir / "train"
    validation_data_dir = trainingdata_dir / "validation"

    trainingdata_eval_dir = project_dir / "trainingdata_eval"
    testdata_dir = Path("Q:/phototagger/trainingdata_raw")
    evaluation_dirs = [
        (train_data_dir, trainingdata_eval_dir / f"{version}/train"),
        (validation_data_dir, trainingdata_eval_dir / f"{version}/validation"),
        (testdata_dir / "varia/test", None),
        (testdata_dir / "varia/test_extra", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2023", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2024", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2025", None),
        (testdata_dir / "stalmest/agrilens_stalmest_2025", None),
    ]

    # Init logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(train_log_dir / "train.log"),
        ],
    )

    logging.info("Starting the training run.")
    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"phototagger-{project}_{version}.keras"
    classes = {"korrelmais"}
    image_size = (512, 512)
    crop_to_aspect_ratio = True
    crop_window_scale = 0.7
    backbone = "EfficientNetV2B0"
    weights = "imagenet"
    evaluation_thresholds = (0.5, 0.6)
    frozen_backbone_epochs = 5
    unfrozen_backbone_layers = 0.15
    learning_rate = 1e-3
    head_dropout = 0.5
    head_l2 = 0.005
    early_stopping_patience = 30
    reduce_lr_patience = 10
    reduce_lr_factor = 0.2
    min_learning_rate = 1e-6
    force = False
    batch_size = 16
    epochs = 100

    logging.info("Writing training manifests.")
    write_manifest(train_data_dir, train_manifest_path, classes=classes, force=force)
    write_manifest(
        validation_data_dir, validation_manifest_path, classes=classes, force=force
    )
    train_images = read_manifest(train_manifest_path, dataset_root=train_data_dir)
    validation_images = read_manifest(
        validation_manifest_path, dataset_root=validation_data_dir
    )

    # Train the model with the specified parameters.
    train_model(
        train_images,
        classes=determine_classes(train_images),
        output_path=model_path,
        validation_images=validation_images,
        image_size=image_size,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        crop_window_scale=crop_window_scale,
        backbone=backbone,
        weights=weights,
        epochs=epochs,
        class_weighting=False,
        batch_size=batch_size,
        frozen_backbone_epochs=frozen_backbone_epochs,
        unfrozen_backbone_layers=unfrozen_backbone_layers,
        learning_rate=learning_rate,
        head_dropout=head_dropout,
        head_l2=head_l2,
        early_stopping_patience=early_stopping_patience,
        reduce_lr_patience=reduce_lr_patience,
        reduce_lr_factor=reduce_lr_factor,
        min_learning_rate=min_learning_rate,
        # monitor_metric="val_precision",
        force=force,
    )

    # Evaluate the trained model on the specified evaluation directories.
    for input_dir, output_dir in evaluation_dirs:
        if output_dir is None:
            output_dir = input_dir.parent / f"{input_dir.name}_{version}"
        output_dir.mkdir(parents=True, exist_ok=True)
        records = discover_records(input_dir, label_whitelist=classes)
        if not records:
            raise ValueError(f"No labeled images found in {input_dir}")

        logging.info(
            "Writing prediction reports and misclassified images to %s", output_dir
        )
        predict_images(
            model_path,
            records,
            batch_size=batch_size,
            workers=4,
            output_dir=output_dir,
            thresholds=evaluation_thresholds,
        )


if __name__ == "__main__":
    main()
