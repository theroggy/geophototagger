"""Train a custom phototagger model."""

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

from phototagger.custom.dataset import (
    determine_classes,
    discover_records,
    discover_records_csv,
    read_manifest,
    write_manifest,
)
from phototagger.custom.prediction import (
    predict_images,
    prediction_evaluation_is_complete,
    write_prediction_evaluation,
)
from phototagger.custom.training import train_model

if TYPE_CHECKING:
    from phototagger.custom.image_processing import ImageInterpolation


def main() -> None:
    """Train a model and write evaluation results."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    # Edit these values for the training run you want to execute.
    project = "maizemulch"
    version = "03-b0-0.7-bicubic"
    classes = {"korrelmais"}
    image_size = (512, 512)
    crop_window_scale = 0.7
    backbone = "EfficientNetV2B0"
    evaluation_thresholds = (0.5, 0.6)
    # Use FBeta with beta=0.5 to give more weight to precision because we want to avoid
    # false positive "maizemulch" predictions.
    additional_metrics = [
        keras.metrics.FBetaScore(
            beta=0.5, average="macro", threshold=0.5, name="fbeta_prec"
        ),
    ]
    monitor_metric = "val_loss"
    force = False

    project_dir = Path(f"X:/Monitoring/phototagger/{project}")
    trainingdata_dir = Path(f"X:/Monitoring/phototagger/{project}/trainingdata")
    train_data_dir = trainingdata_dir / "train"
    val_data_dir = trainingdata_dir / "validation"
    doubt_data_dir = trainingdata_dir / "twijfel"

    traindata_eval_dir = project_dir / "trainingdata_eval"
    testdata_dir = Path("Q:/phototagger/trainingdata_raw")
    evaluation_dirs = [
        (train_data_dir, traindata_eval_dir / f"{version}/{train_data_dir.name}"),
        (val_data_dir, traindata_eval_dir / f"{version}/{val_data_dir.name}"),
        (doubt_data_dir, traindata_eval_dir / f"{version}/{doubt_data_dir.name}"),
        (testdata_dir / "varia/test", None),
        (testdata_dir / "varia/test_extra", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2023", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2024", None),
        (testdata_dir / "korrelmais/agrilens_korrelmais_2025", None),
        (testdata_dir / "stalmest/agrilens_stalmest_2023", None),
        (testdata_dir / "stalmest/agrilens_stalmest_2024", None),
        (testdata_dir / "stalmest/agrilens_stalmest_2025", None),
        (testdata_dir / "grasklaver/overview.csv", None),
    ]

    model_dir = project_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"phototagger-{project}_{version}.keras"

    project_dir.mkdir(parents=True, exist_ok=True)
    train_log_dir = project_dir / "train_log" / version
    train_log_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_path = train_log_dir / "train-images.csv"
    validation_manifest_path = train_log_dir / "validation-images.csv"

    # Init logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(train_log_dir / "train.log"),
        ],
    )

    logging.info(f"Starting the training run {version} of project {project}.")
    if model_path.exists():
        logging.info(f"Model already exists at {model_path}, skipping training.")
    else:
        logging.info("Writing training manifests.")
        write_manifest(
            train_data_dir, train_manifest_path, classes=classes, force=force
        )
        write_manifest(
            val_data_dir, validation_manifest_path, classes=classes, force=force
        )
        train_images = read_manifest(train_manifest_path, dataset_root=train_data_dir)
        validation_images = read_manifest(
            validation_manifest_path, dataset_root=val_data_dir
        )

        # Train the model with the specified parameters.
        train_model(
            train_images,
            classes=determine_classes(train_images),
            output_path=model_path,
            validation_images=validation_images,
            image_size=image_size,
            crop_window_scale=crop_window_scale,
            backbone=backbone,
            additional_metrics=additional_metrics,
            monitor_metric=monitor_metric,
        )

    # Evaluate the trained model on the specified evaluation inputs.
    for input_path, output_dir in evaluation_dirs:
        if output_dir is None:
            output_dir = input_path.parent / f"{input_path.stem}_{version}"
        if prediction_evaluation_is_complete(output_dir, evaluation_thresholds):
            logging.info("Evaluation already complete for %s; skipping", input_path)
            continue
        output_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f"Run predict on {input_path} to {output_dir}")
        predictions_path = output_dir / "predictions.csv"
        if input_path.suffix.lower() == ".csv":
            images = discover_records_csv(
                input_path,
                image_path_column="image",
                classes_column="type",
                image_dir=input_path.parent / "images",
                label_whitelist=classes,
            )
        else:
            images = discover_records(input_path, label_whitelist=classes)
        if not images:
            raise ValueError(f"No images found in {input_path}")

        if not predictions_path.exists():
            predict_images(
                model_path=model_path,
                images=images,
                workers=4,
                output_path=predictions_path,
            )

        write_prediction_evaluation(
            images=images,
            classes=sorted(classes),
            predictions=predictions_path,
            output_dir=output_dir,
            thresholds=evaluation_thresholds,
        )


if __name__ == "__main__":
    main()
