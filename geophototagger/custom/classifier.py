"""Keras 3 image tagging with the PyTorch backend."""

from __future__ import annotations

import csv
import json
import logging
import os
import shutil
from collections.abc import Collection
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from .dataset import ImageRecord


def _keras() -> Any:
    """Import Keras only after selecting the requested backend."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    return keras


def _load_image_batch(
    records: list[ImageRecord], image_size: tuple[int, int]
) -> np.ndarray:
    """Load a batch of images into a single ndarray."""
    keras = _keras()
    images = [
        keras.utils.img_to_array(
            keras.utils.load_img(record.image_path, target_size=image_size)
        )
        for record in records
    ]
    return np.array(images)


def _load_images(records: list[ImageRecord], image_size: tuple[int, int]) -> Any:
    keras = _keras()
    return keras.ops.convert_to_tensor(_load_image_batch(records, image_size))


def _make_image_sequence_class(keras: Any) -> type:
    """Build a ``PyDataset`` subclass that lazily loads images per batch."""

    class _ImageSequence(keras.utils.PyDataset):
        def __init__(
            self,
            records: list[ImageRecord],
            image_size: tuple[int, int],
            *,
            labels: np.ndarray | None = None,
            sample_weights: np.ndarray | None = None,
            batch_size: int = 32,
            shuffle: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.records = records
            self.image_size = image_size
            self.labels = labels
            self.sample_weights = sample_weights
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(records))
            self._rng = np.random.default_rng()

        def __len__(self) -> int:
            return max(1, -(-len(self.records) // self.batch_size))

        def __getitem__(self, index: int) -> Any:
            batch_indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batch_images = _load_image_batch(
                [self.records[i] for i in batch_indices], self.image_size
            )
            if self.labels is None:
                return batch_images
            batch_labels = self.labels[batch_indices]
            if self.sample_weights is None:
                return batch_images, batch_labels
            return batch_images, batch_labels, self.sample_weights[batch_indices]

        def on_epoch_end(self) -> None:
            if self.shuffle:
                self._rng.shuffle(self.indices)

    return _ImageSequence


def calculate_class_weights(
    labels: list[list[int]], vocabulary: list[str]
) -> dict[str, dict[str, float]]:
    """Calculate balanced positive and negative weights for each label.

    Args:
        labels: Multi-hot label vectors for the training records.
        vocabulary: Ordered label names corresponding to vector columns.

    Returns:
        Mapping from each label to its positive and negative training weights.

    Raises:
        ValueError: If inputs are empty or a label has no positive or negative
            examples.
    """
    if not labels or not vocabulary:
        raise ValueError("Class weights require labels and a vocabulary")
    record_count = len(labels)
    weights: dict[str, dict[str, float]] = {}
    for index, label in enumerate(vocabulary):
        positive_count = sum(row[index] for row in labels)
        negative_count = record_count - positive_count
        if not positive_count or not negative_count:
            raise ValueError(
                f"Label {label!r} must have both positive and negative examples"
            )
        weights[label] = {
            "positive": record_count / (2 * positive_count),
            "negative": record_count / (2 * negative_count),
        }
    return weights


def _sample_weights(
    labels: list[list[int]],
    vocabulary: list[str],
    class_weights: dict[str, dict[str, float]],
) -> list[float]:
    """Collapse per-label weights to one compatible weight per image."""
    return [
        sum(
            class_weights[label]["positive" if value else "negative"]
            for label, value in zip(vocabulary, row, strict=True)
        )
        / len(vocabulary)
        for row in labels
    ]


def _prediction_progress_callback(
    keras: Any, record_count: int, batch_size: int
) -> Any:
    """Create a callback that displays prediction progress with tqdm."""
    progress = tqdm(total=record_count, desc="Predicting", unit="image")
    logging.info("Predicting %d files", record_count)

    def update_progress(batch: int, logs: dict[str, Any] | None = None) -> None:
        del batch
        del logs
        progress.update(min(batch_size, record_count - progress.n))

    def close_progress(logs: dict[str, Any] | None = None) -> None:
        del logs
        progress.close()

    return keras.callbacks.LambdaCallback(
        on_predict_batch_end=update_progress,
        on_predict_end=close_progress,
    )


def _write_misclassified(
    model: Any,
    records: list[ImageRecord],
    vocabulary: list[str],
    *,
    image_size: tuple[int, int],
    thresholds: Collection[float],
    output_dir: Path,
    batch_size: int = 32,
    workers: int = 4,
) -> None:
    """Copy misclassified images for each threshold using one prediction pass."""
    if not records:
        return

    pending_reports = [
        (threshold, output_dir / f"threshold-{threshold:g}")
        for threshold in thresholds
        if not (
            output_dir / f"threshold-{threshold:g}" / "classification-results.csv"
        ).exists()
    ]
    if not pending_reports:
        logging.info("Classification results already exist under %s", output_dir)
        return

    logging.info("Writing misclassified images to %s", output_dir)
    keras = _keras()
    image_sequence_class = _make_image_sequence_class(keras)
    dataset_kwargs: dict[str, Any] = (
        {"workers": workers, "use_multiprocessing": False} if workers else {}
    )
    dataset = image_sequence_class(
        records, image_size, batch_size=batch_size, **dataset_kwargs
    )
    probabilities = model.predict(
        dataset,
        verbose=0,
        callbacks=[_prediction_progress_callback(keras, len(records), batch_size)],
    )
    for threshold, report_dir in pending_reports:
        _write_misclassification_report(
            records,
            vocabulary,
            probabilities,
            threshold=threshold,
            output_dir=report_dir,
        )


def _write_misclassification_report(
    records: list[ImageRecord],
    vocabulary: list[str],
    probabilities: Any,
    *,
    threshold: float,
    output_dir: Path,
) -> None:
    """Write one threshold-specific misclassification report."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "classification-results.csv"
    probability_columns = [f"{label}_probability" for label in vocabulary]
    correctly_classified_count = 0
    label_statistics = {
        label: {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }
        for label in vocabulary
    }
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "image_path",
                "expected_labels",
                "predicted_labels",
                "correctly_classified",
                *probability_columns,
            ],
        )
        writer.writeheader()
        for record, probability_row in zip(records, probabilities, strict=True):
            predicted_probabilities = dict(
                zip(
                    vocabulary,
                    (float(value) for value in probability_row),
                    strict=True,
                )
            )
            predicted_labels = {
                label
                for label, probability in predicted_probabilities.items()
                if probability >= threshold
            }
            correctly_classified = predicted_labels == set(record.labels)
            correctly_classified_count += correctly_classified
            for label in vocabulary:
                expected = label in record.labels
                predicted = label in predicted_labels
                if expected and predicted:
                    label_statistics[label]["true_positives"] += 1
                elif predicted:
                    label_statistics[label]["false_positives"] += 1
                elif expected:
                    label_statistics[label]["false_negatives"] += 1
                else:
                    label_statistics[label]["true_negatives"] += 1
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "expected_labels": ";".join(record.labels),
                    "predicted_labels": ";".join(sorted(predicted_labels)),
                    "correctly_classified": correctly_classified,
                    **{
                        f"{label}_probability": predicted_probabilities[label]
                        for label in vocabulary
                    },
                }
            )
            if not correctly_classified:
                shutil.copy2(record.image_path, output_dir / record.image_path.name)
                misclassified_report_path = (
                    output_dir / f"{record.image_path.stem}.json"
                )
                misclassified_report_path.write_text(
                    json.dumps(
                        {
                            "image": str(record.image_path),
                            "expected_labels": list(record.labels),
                            "predicted_labels": sorted(predicted_labels),
                            "probabilities": predicted_probabilities,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
    label_metrics = {}
    for label, counts in label_statistics.items():
        precision_denominator = counts["true_positives"] + counts["false_positives"]
        recall_denominator = counts["true_positives"] + counts["false_negatives"]
        precision = (
            counts["true_positives"] / precision_denominator
            if precision_denominator
            else 0.0
        )
        recall = (
            counts["true_positives"] / recall_denominator if recall_denominator else 0.0
        )
        label_metrics[label] = {
            **counts,
            "accuracy": (counts["true_positives"] + counts["true_negatives"])
            / len(records),
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    misclassified_count = len(records) - correctly_classified_count
    statistics_path = output_dir / "classification-statistics.json"
    statistics_path.write_text(
        json.dumps(
            {
                "classification_threshold": threshold,
                "total_files": len(records),
                "correctly_classified_files": correctly_classified_count,
                "correctly_classified_percentage": 100
                * correctly_classified_count
                / len(records),
                "misclassified_files": misclassified_count,
                "misclassified_percentage": 100 * misclassified_count / len(records),
                "per_label": label_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _compile_model(keras: Any, model: Any, learning_rate: float | None = None) -> None:
    """Compile a classifier model with the configured optimizer and metrics."""
    optimizer_kwargs = {"learning_rate": learning_rate} if learning_rate else {}
    model.compile(
        optimizer=keras.optimizers.Adam(**optimizer_kwargs),
        loss=keras.losses.BinaryFocalCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )


def _make_csv_logger(keras: Any, path: Path, *, append: bool) -> Any:
    """Build a CSVLogger that reopens its file with newline="" to avoid blank rows."""

    class _CSVLogger(keras.callbacks.CSVLogger):
        def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
            super().on_train_begin(logs)
            self.csv_file.close()
            self.csv_file = open(  # noqa: SIM115
                self.filename, "a" if self.append else "w", newline=""
            )

    return _CSVLogger(path, append=append)


def _build_model(
    class_count: int,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "EfficientNetV2B0",
    weights: str | None = "imagenet",
    augment: bool = True,
) -> tuple[Any, Any]:
    """Build a compiled classifier and its initially frozen feature extractor."""
    keras = _keras()
    try:
        backbone_factory = getattr(keras.applications, backbone)
    except AttributeError as error:
        raise ValueError(
            f"Unsupported Keras Applications backbone: {backbone}"
        ) from error
    feature_extractor = backbone_factory(
        include_top=False,
        weights=weights,
        input_shape=(*image_size, 3),
        pooling="avg",
    )
    feature_extractor.trainable = False
    inputs = keras.Input(shape=(*image_size, 3))
    augmented_inputs = inputs
    if augment:
        augmented_inputs = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.08),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.1),
            ],
            name="data_augmentation",
        )(augmented_inputs)
    features = feature_extractor(augmented_inputs, training=False)
    outputs = keras.layers.Dense(class_count, activation="sigmoid")(features)
    model = keras.Model(inputs, outputs)
    _compile_model(keras, model)
    return model, feature_extractor


def build_model(
    class_count: int,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "EfficientNetV2B0",
    weights: str | None = "imagenet",
    augment: bool = True,
) -> Any:
    """Build a frozen Keras Applications backbone with a sigmoid head.

    Args:
        class_count: Number of output labels.
        image_size: Height and width expected by the model.
        backbone: Name of the Keras Applications backbone to use.
        weights: Backbone weights, typically ``"imagenet"`` or ``None``.
        augment: Whether to include random training-time augmentation layers.

    Returns:
        A compiled Keras multilabel classification model.

    Raises:
        ValueError: If ``backbone`` is not available in Keras Applications.
    """
    model, _ = _build_model(class_count, image_size, backbone, weights, augment)
    return model


def _split_training_records(
    records: list[ImageRecord],
    validation_records: list[ImageRecord] | None,
    validation_split: float,
    seed: int,
) -> tuple[list[ImageRecord], list[ImageRecord]]:
    """Split training records while preserving explicit validation data."""
    if validation_records is not None:
        return records, validation_records
    if not validation_split:
        return records, []
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    split_at = int(len(records) * (1 - validation_split))
    return (
        [records[index] for index in order[:split_at]],
        [records[index] for index in order[split_at:]],
    )


def _log_class_counts(
    dataset_name: str, records: list[ImageRecord], vocabulary: list[str]
) -> None:
    """Log the number of records containing each class and no classes."""
    counts = {
        "no classes": sum(not record.labels for record in records),
        **{
            label: sum(label in record.labels for record in records)
            for label in vocabulary
        },
    }
    logging.info("%s dataset example counts: %s", dataset_name, counts)


def _fit_model(
    records: list[ImageRecord],
    vocabulary: list[str],
    output_path: Path,
    *,
    validation_records: list[ImageRecord],
    image_size: tuple[int, int],
    backbone: str,
    weights: str | None,
    epochs: int,
    seed: int,
    frozen_backbone_epochs: int,
    augment: bool,
    class_weighting: bool,
    early_stopping_patience: int,
    monitor_metric: str,
    batch_size: int,
    workers: int,
) -> Any:
    """Fit, checkpoint, and return a model using prepared record splits."""
    logging.info("Starting training")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".csv").unlink(missing_ok=True)
    _log_class_counts("Training", records, vocabulary)
    _log_class_counts("Validation", validation_records, vocabulary)

    keras = _keras()
    keras.utils.set_random_seed(seed)
    image_sequence_class = _make_image_sequence_class(keras)
    dataset_kwargs: dict[str, Any] = (
        {"workers": workers, "use_multiprocessing": False} if workers else {}
    )
    labels = np.array(
        [[int(label in record.labels) for label in vocabulary] for record in records]
    )
    class_weights = calculate_class_weights(labels.tolist(), vocabulary)
    if class_weighting:
        logging.info("Using class weights: %s", class_weights)
    else:
        logging.info("Class weighting is disabled")
    train_labels = np.array(
        [[int(label in record.labels) for label in vocabulary] for record in records]
    )
    validation_dataset = None
    if validation_records:
        validation_labels = np.array(
            [
                [int(label in record.labels) for label in vocabulary]
                for record in validation_records
            ]
        )
        validation_dataset = image_sequence_class(
            validation_records,
            image_size,
            labels=validation_labels,
            batch_size=batch_size,
            **dataset_kwargs,
        )
    sample_weights = (
        np.array(_sample_weights(train_labels.tolist(), vocabulary, class_weights))
        if class_weighting
        else None
    )
    train_dataset = image_sequence_class(
        records,
        image_size,
        labels=train_labels,
        sample_weights=sample_weights,
        batch_size=batch_size,
        shuffle=True,
        **dataset_kwargs,
    )
    model, feature_extractor = _build_model(
        len(vocabulary),
        image_size=image_size,
        backbone=backbone,
        weights=weights,
        augment=augment,
    )
    fit_kwargs: dict[str, Any] = {"epochs": epochs}
    if validation_dataset is not None:
        fit_kwargs["validation_data"] = validation_dataset
    monitor = (
        monitor_metric
        if validation_dataset is None or monitor_metric.startswith("val_")
        else f"val_{monitor_metric}"
    )
    callbacks: list[Any] = [
        _make_csv_logger(keras, output_path.with_suffix(".csv"), append=True),
        keras.callbacks.ModelCheckpoint(
            output_path, monitor=monitor, save_best_only=True
        ),
    ]
    if early_stopping_patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    fit_kwargs["callbacks"] = callbacks

    frozen_epochs = min(epochs, frozen_backbone_epochs)
    if frozen_epochs:
        fit_kwargs["epochs"] = frozen_epochs
        model.fit(train_dataset, **fit_kwargs)
    if frozen_epochs < epochs:
        logging.info("Unfreezing the feature extractor after %d epochs", frozen_epochs)
        feature_extractor.trainable = True
        _compile_model(keras, model, learning_rate=1e-4)
        fit_kwargs["initial_epoch"] = frozen_epochs
        fit_kwargs["epochs"] = epochs
        model.fit(train_dataset, **fit_kwargs)
    model = keras.models.load_model(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "labels": vocabulary,
                "image_size": list(image_size),
                "backbone": backbone,
                "threshold": 0.5,
                "augmentation": augment,
                "class_weighting": class_weighting,
                "class_weights": class_weights,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def train_model(
    records: list[ImageRecord],
    vocabulary: list[str],
    output_path: Path,
    *,
    validation_records: list[ImageRecord] | None = None,
    test_records: list[ImageRecord] | None = None,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "EfficientNetV2B0",
    weights: str | None = "imagenet",
    epochs: int = 30,
    validation_split: float = 0.0,
    seed: int = 42,
    thresholds: Collection[float] = (0.5,),
    frozen_backbone_epochs: int = 5,
    augment: bool = True,
    class_weighting: bool = True,
    early_stopping_patience: int = 10,
    monitor_metric: str = "loss",
    misclassified_dir: Path | None = None,
    batch_size: int = 32,
    workers: int = 4,
    force: bool = False,
) -> Any:
    """Train and save a multilabel classifier and its metadata.

    Args:
        records: Training image records.
        vocabulary: Ordered labels used by the model output.
        output_path: Destination for the saved Keras model.
        validation_records: Optional explicit validation records. When set,
            ``validation_split`` must be zero.
        test_records: Optional held-out records used only to collect
            misclassified images; never used for training or validation.
        image_size: Height and width used when loading images.
        backbone: Name of the Keras Applications backbone.
        weights: Backbone weights, typically ``"imagenet"`` or ``None``.
        epochs: Number of training epochs.
        validation_split: Fraction of training data used for validation when
            explicit validation records are not supplied.
        seed: Random seed used by Keras.
        thresholds: Prediction thresholds used for separate misclassification
            reports.
        frozen_backbone_epochs: Number of initial epochs to train only the
            classification head before fine-tuning the feature extractor.
        augment: Whether to enable random training-time augmentation.
        class_weighting: Whether to apply balanced per-image sample weights.
        early_stopping_patience: Epochs with no monitored improvement before
            training stops early. Set to zero to disable early stopping.
        monitor_metric: Metric used to select the best checkpoint and to
            evaluate early stopping, e.g. ``"loss"``, ``"binary_accuracy"``,
            ``"precision"`` or ``"recall"``. A ``val_`` prefix is added
            automatically when validation data is available.
        misclassified_dir: Optional directory under which misclassified
            images are copied into ``train_wrong``, ``validation_wrong`` and
            ``test_wrong`` subdirectories.
        batch_size: Number of images loaded into memory per training or
            prediction batch.
        workers: Number of background threads used to load image batches
            ahead of time while the model trains. Set to zero to load
            synchronously on the main thread.
        force: Whether to overwrite an existing model at ``output_path``.

    Returns:
        The best checkpointed Keras model, matching what was saved to
        ``output_path``.

    Raises:
        ValueError: If records are empty, validation settings conflict, or the
            validation split is outside the range [0, 1).
    """
    if not records:
        raise ValueError("Cannot train without image records")
    if frozen_backbone_epochs < 0:
        raise ValueError("frozen_backbone_epochs must not be negative")
    thresholds = tuple(thresholds)
    if not thresholds:
        raise ValueError("At least one prediction threshold is required")
    if len(set(thresholds)) != len(thresholds) or not all(
        0 <= threshold <= 1 for threshold in thresholds
    ):
        raise ValueError("Prediction thresholds must be unique values between 0 and 1")
    if (validation_records is not None and validation_split) or (
        validation_records and not validation_split == 0.0
    ):
        raise ValueError(
            "Use either validation_records or validation_split, not both nor neither"
        )
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    train_records, validation_records_for_reporting = _split_training_records(
        records, validation_records, validation_split, seed
    )
    if output_path.exists() and not force:
        logging.info("Model already exists at %s; skipping training", output_path)
        model = _keras().models.load_model(output_path)
    else:
        logging.info("Training model to %s", output_path)
        model = _fit_model(
            train_records,
            vocabulary,
            output_path,
            validation_records=validation_records_for_reporting,
            image_size=image_size,
            backbone=backbone,
            weights=weights,
            epochs=epochs,
            seed=seed,
            frozen_backbone_epochs=frozen_backbone_epochs,
            augment=augment,
            class_weighting=class_weighting,
            early_stopping_patience=early_stopping_patience,
            monitor_metric=monitor_metric,
            batch_size=batch_size,
            workers=workers,
        )

    if misclassified_dir is not None:
        _write_misclassified(
            model,
            records,
            vocabulary,
            image_size=image_size,
            thresholds=thresholds,
            output_dir=misclassified_dir / "train_wrong",
            batch_size=batch_size,
            workers=workers,
        )
        _write_misclassified(
            model,
            validation_records_for_reporting,
            vocabulary,
            image_size=image_size,
            thresholds=thresholds,
            output_dir=misclassified_dir / "validation_wrong",
            batch_size=batch_size,
            workers=workers,
        )
        _write_misclassified(
            model,
            test_records or [],
            vocabulary,
            image_size=image_size,
            thresholds=thresholds,
            output_dir=misclassified_dir / "test_wrong",
            batch_size=batch_size,
            workers=workers,
        )
    return model


def predict_images(
    model_path: Path,
    image_paths: list[Path],
    batch_size: int = 32,
    workers: int = 4,
) -> list[dict[str, float]]:
    """Predict label probabilities for a list of images.

    Args:
        model_path: Saved Keras model path with adjacent JSON metadata.
        image_paths: Images to classify.
        batch_size: Number of images loaded into memory per prediction batch.
        workers: Number of background threads used to load image batches
            ahead of time while predicting. Set to zero to load synchronously
            on the main thread.

    Returns:
        One label-to-probability mapping per image, in ``image_paths`` order.
    """
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    image_size = tuple(metadata["image_size"])
    keras = _keras()
    model = keras.models.load_model(model_path)
    records = [ImageRecord(image_path, ()) for image_path in image_paths]
    image_sequence_class = _make_image_sequence_class(keras)
    dataset_kwargs: dict[str, Any] = (
        {"workers": workers, "use_multiprocessing": False} if workers else {}
    )
    dataset = image_sequence_class(
        records, image_size, batch_size=batch_size, **dataset_kwargs
    )
    probabilities = model.predict(
        dataset,
        verbose=0,
        callbacks=[_prediction_progress_callback(keras, len(image_paths), batch_size)],
    )
    return [
        dict(
            zip(
                metadata["labels"],
                (float(value) for value in probability_row),
                strict=True,
            )
        )
        for probability_row in probabilities
    ]


def predict_image(model_path: Path, image_path: Path) -> dict[str, float]:
    """Predict label probabilities for one image.

    Args:
        model_path: Saved Keras model path with adjacent JSON metadata.
        image_path: Image to classify.

    Returns:
        Mapping from metadata label names to predicted probabilities.
    """
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    image_size = tuple(metadata["image_size"])
    keras = _keras()
    model = keras.models.load_model(model_path)
    image = _load_images([ImageRecord(image_path, ())], image_size)
    logging.info("Predicting %s", image_path)
    probabilities = model.predict(image, verbose=0)[0]
    logging.info("Prediction complete for %s", image_path)
    return dict(
        zip(metadata["labels"], (float(value) for value in probabilities), strict=True)
    )
