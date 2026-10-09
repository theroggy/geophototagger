"""Keras 3 image tagging with the PyTorch backend.

Some inspiration for the default training hyperparameters is inspired from:
- https://www.identifyshell.org/blog-fine-tuning-efficientnetv2-models.php
- https://keras.io/examples/vision/image_classification_efficientnet_fine_tuning

"""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image, ImageOps
from tqdm.auto import tqdm

from .dataset import ImageRecord

if TYPE_CHECKING:
    from collections.abc import Collection

logger = logging.getLogger(__name__)


def _keras() -> Any:
    """Import Keras only after selecting the requested backend."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    return keras


def _load_image_batch(
    records: list[ImageRecord],
    image_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
) -> np.ndarray:
    """Load a batch of images into a single ndarray."""
    _validate_crop_window_scale(crop_window_scale)
    keras = _keras()
    images = []
    for record in records:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                image = _load_img(
                    record.image_path,
                    image_size,
                    crop_to_aspect_ratio=crop_to_aspect_ratio,
                    crop_window_scale=crop_window_scale,
                )
            except OSError as error:
                raise OSError(
                    f"Failed to load image {record.image_path}: {error}"
                ) from error
        for caught_warning in caught_warnings:
            warnings.warn(
                f"{record.image_path}: {caught_warning.message}",
                caught_warning.category,
                stacklevel=2,
            )
        images.append(keras.utils.img_to_array(image))
    return np.array(images)


def _load_img(
    image_path: Path,
    target_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
) -> Image.Image:
    """Load an RGB image and resize it to the requested dimensions.

    Args:
        image_path: Path to the source image.
        target_size: Output size as ``(height, width)``.
        crop_to_aspect_ratio: Whether to center-crop the source to the target
            aspect ratio. If false, resize the entire image and distort its
            aspect ratio as needed.
        crop_window_scale: Fraction of the aspect-fitted crop's width and
            height to retain, in ``(0, 1]``. Values below one apply an extra
            centered crop when it still leaves at least ``target_size`` pixels
            in both dimensions. Otherwise, the standard aspect-ratio crop is
            used. This option has no effect when ``crop_to_aspect_ratio`` is
            false.

    Returns:
        The resized RGB image.

    Raises:
        ValueError: If ``crop_window_scale`` is outside ``(0, 1]``.
    """
    _validate_crop_window_scale(crop_window_scale)
    target_height, target_width = target_size
    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")
    if crop_to_aspect_ratio:
        image_width, image_height = image.size
        target_ratio = target_width / target_height
        if image_width / image_height > target_ratio:
            crop_width = int(image_height * target_ratio)
            crop_height = image_height
        else:
            crop_width = image_width
            crop_height = int(image_width / target_ratio)

        if crop_window_scale < 1:
            tighter_width = int(crop_width * crop_window_scale)
            tighter_height = int(crop_height * crop_window_scale)
            if tighter_width >= target_width and tighter_height >= target_height:
                crop_width = tighter_width
                crop_height = tighter_height

        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))

    return image.resize(
        (target_width, target_height), resample=Image.Resampling.NEAREST
    )


def _validate_crop_window_scale(crop_window_scale: float) -> None:
    if (
        isinstance(crop_window_scale, bool)
        or not isinstance(crop_window_scale, (int, float))
        or not math.isfinite(crop_window_scale)
        or not 0 < crop_window_scale <= 1
    ):
        raise ValueError("crop_window_scale must be greater than 0 and at most 1")


def _crop_window_scale_from_metadata(metadata: dict[str, Any]) -> float:
    if "crop_window_scale" in metadata:
        return metadata["crop_window_scale"]
    legacy_crop_percent = metadata.get("aspect_ratio_crop_percent", 0.0)
    return 1.0 - legacy_crop_percent / 100


def _load_images(
    records: list[ImageRecord],
    image_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
) -> Any:
    keras = _keras()
    return keras.ops.convert_to_tensor(
        _load_image_batch(
            records,
            image_size,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            crop_window_scale=crop_window_scale,
        )
    )


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
            crop_to_aspect_ratio: bool = True,
            crop_window_scale: float = 1.0,
            batch_size: int = 32,
            shuffle: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.records = records
            self.image_size = image_size
            self.labels = labels
            self.sample_weights = sample_weights
            self.crop_to_aspect_ratio = crop_to_aspect_ratio
            self.crop_window_scale = crop_window_scale
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
                [self.records[i] for i in batch_indices],
                self.image_size,
                crop_to_aspect_ratio=self.crop_to_aspect_ratio,
                crop_window_scale=self.crop_window_scale,
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
    labels: list[list[int]], classes: list[str]
) -> dict[str, dict[str, float]]:
    """Calculate balanced positive and negative weights for each label.

    Args:
        labels: Multi-hot label vectors for the training records.
        classes: Ordered label names corresponding to vector columns.

    Returns:
        Mapping from each label to its positive and negative training weights.

    Raises:
        ValueError: If inputs are empty or a label has no positive or negative
            examples.
    """
    if not labels or not classes:
        raise ValueError("Class weights require labels and classes")
    record_count = len(labels)
    weights: dict[str, dict[str, float]] = {}
    for index, label in enumerate(classes):
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
    classes: list[str],
    class_weights: dict[str, dict[str, float]],
) -> list[float]:
    """Collapse per-label weights to one compatible weight per image."""
    return [
        sum(
            class_weights[label]["positive" if value else "negative"]
            for label, value in zip(classes, row, strict=True)
        )
        / len(classes)
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


def write_prediction_evaluation_info(
    records: list[ImageRecord],
    classes: list[str],
    predictions_by_path: dict[str, dict[str, float]],
    *,
    threshold: float,
    output_dir: Path,
) -> None:
    """Write an evaluation information for one classification threshold."""
    output_dir.mkdir(parents=True, exist_ok=True)
    label_statistics = {
        label: {
            "true_positives": 0,
            "false_positives": 0,
            "true_negatives": 0,
            "false_negatives": 0,
        }
        for label in classes
    }
    correctly_classified_count = 0
    output_dir_ok = output_dir.parent / f"{output_dir.name}_ok"
    output_dir_ok.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "prediction-results.csv"
    with report_path.open("w", encoding="utf-8", newline="") as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=[
                "image_path",
                "expected_labels",
                "predicted_labels",
                "correctly_classified",
                *(f"{label}_probability" for label in classes),
            ],
        )
        writer.writeheader()
        for record in tqdm(records, desc=f"Create eval dir {threshold}", unit="image"):
            path_key = str(record.image_path.resolve())
            predicted_probabilities = predictions_by_path[path_key]
            predicted_labels = {
                label
                for label, probability in predicted_probabilities.items()
                if probability >= threshold
            }
            correctly_classified = predicted_labels == set(record.labels)
            correctly_classified_count += correctly_classified
            for label, counts in label_statistics.items():
                expected = label in record.labels
                predicted = label in predicted_labels
                if expected and predicted:
                    counts["true_positives"] += 1
                elif predicted:
                    counts["false_positives"] += 1
                elif expected:
                    counts["false_negatives"] += 1
                else:
                    counts["true_negatives"] += 1
            writer.writerow(
                {
                    "image_path": str(record.image_path),
                    "expected_labels": ";".join(record.labels),
                    "predicted_labels": ";".join(sorted(predicted_labels)),
                    "correctly_classified": correctly_classified,
                    **{
                        f"{label}_probability": predicted_probabilities[label]
                        for label in classes
                    },
                }
            )

            # Copy the image and write the JSON metadata to the appropriate output
            # directory based on classification correctness.
            if correctly_classified:
                dst_path = output_dir_ok / record.image_path.name
                json_dst_path = output_dir_ok / f"{record.image_path.stem}.json"
            else:
                dst_path = output_dir / record.image_path.name
                json_dst_path = output_dir / f"{record.image_path.stem}.json"

            if not dst_path.exists():
                os.link(record.image_path, dst_path)
            json_dst_path.write_text(
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
    record_count = len(records)
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
            / record_count,
            "precision": precision,
            "recall": recall,
            "f1_score": 2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0,
        }
    misclassified_count = record_count - correctly_classified_count
    (output_dir / "prediction-statistics.json").write_text(
        json.dumps(
            {
                "classification_threshold": threshold,
                "total_files": record_count,
                "correctly_classified_files": correctly_classified_count,
                "correctly_classified_percentage": 100
                * correctly_classified_count
                / record_count,
                "misclassified_files": misclassified_count,
                "misclassified_percentage": 100 * misclassified_count / record_count,
                "per_label": label_metrics,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _compile_model(keras: Any, model: Any, learning_rate: float = 1e-3) -> None:
    """Compile a classifier model with the configured optimizer and metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
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
            self.csv_file = Path(self.filename).open(
                "a" if self.append else "w", newline=""
            )

    return _CSVLogger(path, append=append)


def _make_reduce_lr_callback(
    keras: Any,
    *,
    monitor: str,
    factor: float,
    patience: int,
    min_learning_rate: float,
) -> Any:
    """Lower the optimizer learning rate when the monitored metric plateaus."""
    return keras.callbacks.ReduceLROnPlateau(
        monitor=monitor,
        factor=factor,
        patience=patience,
        min_lr=min_learning_rate,
        verbose=1,
    )


def _make_best_model_metrics_callback(keras: Any, checkpoint: Any) -> Any:
    """Capture epoch metrics whenever the checkpoint saves a new best model."""

    class _BestModelMetricsCallback(keras.callbacks.Callback):
        def __init__(self) -> None:
            super().__init__()
            self._previous_best = checkpoint.best
            self.metrics: dict[str, float] | None = None
            self.epoch: int | None = None

        def on_epoch_end(self, epoch: int, logs: dict[str, Any] | None = None) -> None:
            current_best = checkpoint.best
            if current_best == self._previous_best:
                return
            self._previous_best = current_best
            self.metrics = {
                name: float(value)
                for name, value in (logs or {}).items()
                if value is not None and math.isfinite(float(value))
            }
            self.epoch = epoch + 1

    return _BestModelMetricsCallback()


def _build_model(
    class_count: int,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "EfficientNetV2S",
    weights: str | None = "imagenet",
    augment: bool = True,
    *,
    learning_rate: float = 1e-3,
    head_dropout: float = 0.5,
    head_l2: float = 0.005,
) -> tuple[Any, Any]:
    """Build a compiled classifier and its initially frozen feature extractor."""
    _validate_learning_rate(learning_rate)
    _validate_head_regularization(head_dropout, head_l2)
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
                keras.layers.RandomBrightness(0.1),
            ],
            name="data_augmentation",
        )(augmented_inputs)
    features = feature_extractor(augmented_inputs, training=False)
    if head_dropout:
        features = keras.layers.Dropout(head_dropout, name="head_dropout")(features)
    outputs = keras.layers.Dense(
        class_count,
        activation="sigmoid",
        kernel_regularizer=(keras.regularizers.L2(head_l2) if head_l2 else None),
    )(features)
    model = keras.Model(inputs, outputs)
    _compile_model(keras, model, learning_rate=learning_rate)
    return model, feature_extractor


def build_model(
    class_count: int,
    image_size: tuple[int, int] = (224, 224),
    backbone: str = "EfficientNetV2S",
    weights: str | None = "imagenet",
    augment: bool = True,
    *,
    learning_rate: float = 1e-3,
    head_dropout: float = 0.5,
    head_l2: float = 0.005,
) -> Any:
    """Build a frozen Keras Applications backbone with a sigmoid head.

    Args:
        class_count: Number of output labels.
        image_size: Height and width expected by the model.
        backbone: Name of the Keras Applications backbone to use.
        weights: Backbone weights, typically ``"imagenet"`` or ``None``.
        augment: Whether to include random training-time augmentation layers.
        learning_rate: Initial Adam learning rate.
        head_dropout: Dropout rate applied before the classification output.
        head_l2: L2 regularization strength for the output kernel.

    Returns:
        A compiled Keras multilabel classification model.

    Raises:
        ValueError: If ``backbone`` is not available in Keras Applications.
    """
    model, _ = _build_model(
        class_count,
        image_size,
        backbone,
        weights,
        augment,
        learning_rate=learning_rate,
        head_dropout=head_dropout,
        head_l2=head_l2,
    )
    return model


def _validate_learning_rate(learning_rate: float) -> None:
    if not learning_rate > 0:
        raise ValueError("learning_rate must be greater than zero")


def _validate_head_regularization(head_dropout: float, head_l2: float) -> None:
    if not 0 <= head_dropout < 1:
        raise ValueError("head_dropout must be between 0 and 1")
    if not head_l2 >= 0:
        raise ValueError("head_l2 must not be negative")


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
    dataset_name: str, records: list[ImageRecord], classes: list[str]
) -> None:
    """Log the number of records containing each class and no classes."""
    counts = {
        "no classes": sum(not record.labels for record in records),
        **{
            label: sum(label in record.labels for record in records)
            for label in classes
        },
    }
    logging.info("%s dataset example counts: %s", dataset_name, counts)


def _unfreeze_top_backbone_layers(
    feature_extractor: Any,
    layer_count: int | float = 0.1,
    *,
    expand_to_blocks: bool = True,
) -> int:
    """Unfreeze final layers by count or fraction, optionally expanding blocks.

    Keeping each block's layers trainable or frozen together avoids fine-tuning
    only part of a block while the rest of its transformation remains fixed.
    Integer values specify an exact layer count; floats from 0 to 1 specify a
    fraction of all layers and are rounded up. Set ``expand_to_blocks=False``
    to use the resulting count without expanding to whole blocks.
    """
    _validate_unfrozen_backbone_layers(layer_count)
    layers = feature_extractor.layers
    if isinstance(layer_count, float):
        layer_count = math.ceil(len(layers) * layer_count)
    if not layer_count:
        feature_extractor.trainable = False
        return 0

    feature_extractor.trainable = True
    first_trainable_layer = max(0, len(layers) - layer_count)
    if expand_to_blocks:
        selected_blocks = {
            layer.name.partition("_")[0]
            for layer in layers[first_trainable_layer:]
            if getattr(layer, "name", "").startswith("block") and "_" in layer.name
        }
        for index, layer in enumerate(layers):
            if layer.name.partition("_")[0] in selected_blocks:
                first_trainable_layer = min(first_trainable_layer, index)
    for index, layer in enumerate(layers):
        layer.trainable = index >= first_trainable_layer
    actual_layer_count = sum(layer.trainable for layer in layers)

    if expand_to_blocks:
        logging.info(
            "Unfroze %d of %d backbone layers",
            actual_layer_count,
            len(layers),
        )

    return actual_layer_count


def _validate_unfrozen_backbone_layers(layer_count: int | float) -> None:
    if isinstance(layer_count, bool) or not isinstance(layer_count, (int, float)):
        raise ValueError(
            "unfrozen_backbone_layers must be a nonnegative integer count or "
            "a float fraction between 0 and 1"
        )
    if isinstance(layer_count, int):
        if layer_count < 0:
            raise ValueError("unfrozen_backbone_layers must not be negative")
    elif not 0 <= layer_count <= 1:
        raise ValueError("float unfrozen_backbone_layers must be between 0 and 1")


def _fit_model(
    records: list[ImageRecord],
    classes: list[str],
    output_path: Path,
    *,
    validation_records: list[ImageRecord],
    image_size: tuple[int, int],
    crop_to_aspect_ratio: bool,
    crop_window_scale: float,
    backbone: str,
    weights: str | None,
    epochs: int,
    seed: int,
    frozen_epochs: int,
    augment: bool,
    unfrozen_backbone_layers: int | float,
    head_dropout: float,
    head_l2: float,
    learning_rate: float,
    class_weighting: bool,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    reduce_lr_factor: float,
    min_learning_rate: float,
    monitor_metric: str,
    batch_size: int,
    workers: int,
) -> Any:
    """Fit, checkpoint, and return a model using prepared record splits.

    Args:
        records: Training image records.
        classes: Ordered labels used by the model output.
        output_path: Destination for the saved Keras model.
        validation_records: Records used only for validation.
        image_size: Height and width used when loading images.
        crop_to_aspect_ratio: Whether to center-crop images to the target aspect
            ratio before resizing, rather than stretch them.
        crop_window_scale: Fraction of each aspect-fitted crop dimension to keep.
            Values below one tighten the centered crop when resolution permits.
        backbone: Keras Applications backbone name.
        weights: Initial backbone weights, or ``None``.
        epochs: Total training epochs.
        seed: Random seed used by Keras.
        frozen_epochs: Initial epochs with a completely frozen backbone.
        augment: Whether to enable training-time image augmentation.
        unfrozen_backbone_layers: Number of final backbone layers to fine-tune, or
            a float fraction of the backbone layers.
        head_dropout: Dropout rate applied before the classification output.
        head_l2: L2 regularization strength for the output kernel.
        learning_rate: Initial Adam learning rate for both training phases.
        class_weighting: Whether to apply balanced per-image weights.
        early_stopping_patience: Epochs without improvement before stopping.
        reduce_lr_patience: Epochs without improvement before reducing learning rate.
        reduce_lr_factor: Multiplier applied to learning rate after a plateau.
        min_learning_rate: Lower bound for the learning rate.
        monitor_metric: Metric used to select checkpoints and stop training.
        batch_size: Images per training batch.
        workers: Background image-loading threads; zero loads synchronously.

    Returns:
        The best checkpointed Keras model.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.with_suffix(".csv").unlink(missing_ok=True)
    _log_class_counts("Training", records, classes)
    _log_class_counts("Validation", validation_records, classes)

    keras = _keras()
    keras.utils.set_random_seed(seed)
    image_sequence_class = _make_image_sequence_class(keras)
    dataset_kwargs: dict[str, Any] = (
        {"workers": workers, "use_multiprocessing": False} if workers else {}
    )
    labels = np.array(
        [[int(label in record.labels) for label in classes] for record in records]
    )
    class_weights = calculate_class_weights(labels.tolist(), classes)
    if class_weighting:
        logging.info("Using class weights: %s", class_weights)
    else:
        logging.info("Class weighting is disabled")
    train_labels = np.array(
        [[int(label in record.labels) for label in classes] for record in records]
    )
    validation_dataset = None
    if validation_records:
        validation_labels = np.array(
            [
                [int(label in record.labels) for label in classes]
                for record in validation_records
            ]
        )
        validation_dataset = image_sequence_class(
            validation_records,
            image_size,
            labels=validation_labels,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            crop_window_scale=crop_window_scale,
            batch_size=batch_size,
            **dataset_kwargs,
        )
    sample_weights = (
        np.array(_sample_weights(train_labels.tolist(), classes, class_weights))
        if class_weighting
        else None
    )
    train_dataset = image_sequence_class(
        records,
        image_size,
        labels=train_labels,
        sample_weights=sample_weights,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        crop_window_scale=crop_window_scale,
        batch_size=batch_size,
        shuffle=True,
        **dataset_kwargs,
    )
    model, feature_extractor = _build_model(
        len(classes),
        image_size=image_size,
        backbone=backbone,
        weights=weights,
        augment=augment,
        learning_rate=learning_rate,
        head_dropout=head_dropout,
        head_l2=head_l2,
    )
    fit_kwargs: dict[str, Any] = {"epochs": epochs}
    if validation_dataset is not None:
        fit_kwargs["validation_data"] = validation_dataset
    checkpoint = keras.callbacks.ModelCheckpoint(
        output_path, monitor=monitor_metric, save_best_only=True, verbose=1
    )
    best_model_metrics_callback = _make_best_model_metrics_callback(keras, checkpoint)
    callbacks: list[Any] = [
        _make_csv_logger(keras, output_path.with_suffix(".csv"), append=True),
        checkpoint,
        best_model_metrics_callback,
        _make_reduce_lr_callback(
            keras,
            monitor=monitor_metric,
            factor=reduce_lr_factor,
            patience=reduce_lr_patience,
            min_learning_rate=min_learning_rate,
        ),
    ]
    if early_stopping_patience:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor=monitor_metric,
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )
    fit_kwargs["callbacks"] = callbacks

    actual_unfrozen_backbone_layers = 0
    frozen_epochs = min(epochs, frozen_epochs)
    if frozen_epochs:
        fit_kwargs["epochs"] = frozen_epochs
        model.fit(train_dataset, **fit_kwargs)
    if frozen_epochs < epochs:
        logging.info("Unfreezing the feature extractor after %d epochs", frozen_epochs)
        actual_unfrozen_backbone_layers = _unfreeze_top_backbone_layers(
            feature_extractor, layer_count=unfrozen_backbone_layers
        )
        _compile_model(keras, model, learning_rate=learning_rate)
        fit_kwargs["initial_epoch"] = frozen_epochs
        fit_kwargs["epochs"] = epochs
        model.fit(train_dataset, **fit_kwargs)

    logger.info(
        f"Best model was {best_model_metrics_callback.epoch=} "
        f"with {best_model_metrics_callback.metrics=}"
    )
    model = keras.models.load_model(output_path)
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "labels": classes,
                "image_size": list(image_size),
                "crop_to_aspect_ratio": crop_to_aspect_ratio,
                "crop_window_scale": crop_window_scale,
                "backbone": backbone,
                "threshold": 0.5,
                "augmentation": augment,
                "class_weighting": class_weighting,
                "class_weights": class_weights,
                "unfrozen_backbone_layers": unfrozen_backbone_layers,
                "actual_unfrozen_backbone_layers": actual_unfrozen_backbone_layers,
                "head_dropout": head_dropout,
                "head_l2": head_l2,
                "learning_rate": learning_rate,
                "reduce_lr_patience": reduce_lr_patience,
                "reduce_lr_factor": reduce_lr_factor,
                "min_learning_rate": min_learning_rate,
                "best_model_epoch": best_model_metrics_callback.epoch,
                "best_model_metrics": best_model_metrics_callback.metrics,
            },
            indent=4,
        ),
        encoding="utf-8",
    )
    return model


def train_model(
    train_images: list[ImageRecord],
    classes: list[str],
    output_path: Path,
    *,
    validation_images: list[ImageRecord] | None = None,
    image_size: tuple[int, int] = (224, 224),
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
    backbone: str = "EfficientNetV2S",
    weights: str | None = "imagenet",
    epochs: int = 100,
    validation_split: float = 0.0,
    seed: int = 42,
    frozen_backbone_epochs: int = 5,
    augment: bool = True,
    unfrozen_backbone_layers: int | float = 0.15,
    learning_rate: float = 1e-3,
    head_dropout: float = 0.5,
    head_l2: float = 0.005,
    class_weighting: bool = False,
    early_stopping_patience: int = 20,
    reduce_lr_patience: int = 5,
    reduce_lr_factor: float = 0.2,
    min_learning_rate: float = 1e-6,
    monitor_metric: str = "val_loss",
    batch_size: int = 32,
    workers: int = 4,
    force: bool = False,
) -> Any:
    """Train and save a multilabel classifier and its metadata.

    Args:
        train_images: Training image records.
        classes: Ordered labels used by the model output.
        output_path: Destination for the saved Keras model.
        validation_images: Optional explicit validation records. When set,
            ``validation_split`` must be zero.
        image_size: Height and width used when loading images.
        crop_to_aspect_ratio: Center-crop images to the target aspect ratio,
            preserving geometry but possibly trimming edges. Set to ``False``
            to stretch images to the target dimensions.
        crop_window_scale: Fraction of each aspect-fitted crop dimension to
            keep. Values below one tighten the centered crop when resolution
            permits; ``1.0`` keeps the standard aspect-ratio crop.
        backbone: Name of the Keras Applications backbone.
        weights: Backbone weights, typically ``"imagenet"`` or ``None``.
        epochs: Number of training epochs.
        validation_split: Fraction of training data used for validation when
            explicit validation records are not supplied.
        seed: Random seed used by Keras.
        frozen_backbone_epochs: Number of initial epochs to train only the
            classification head before fine-tuning the feature extractor.
        augment: Whether to enable random training-time augmentation.
        unfrozen_backbone_layers: Number of final backbone layers to fine-tune, or
            a float fraction from 0 to 1. Fractional counts round up before
            optional whole-block expansion.
        learning_rate: Initial Adam learning rate for both training phases.
        head_dropout: Dropout rate applied before the classification output.
        head_l2: L2 regularization strength for the output kernel.
        class_weighting: Whether to apply balanced per-image sample weights.
        early_stopping_patience: Epochs with no monitored improvement before
            training stops early. Set to zero to disable early stopping.
        reduce_lr_patience: Epochs without monitored improvement before reducing
            the learning rate.
        reduce_lr_factor: Multiplier applied to the learning rate after a plateau.
        min_learning_rate: Lower bound for the learning rate.
        monitor_metric: Metric used to select the best checkpoint and to
            evaluate early stopping, e.g. ``"loss"``, ``"binary_accuracy"``,
            ``"precision"`` or ``"recall"``. A ``val_`` prefix is added
            automatically when validation data is available.
        batch_size: Number of images loaded into memory per training batch.
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
    if not train_images:
        raise ValueError("Cannot train without image records")
    if frozen_backbone_epochs < 0:
        raise ValueError("frozen_backbone_epochs must not be negative")
    _validate_crop_window_scale(crop_window_scale)
    _validate_unfrozen_backbone_layers(unfrozen_backbone_layers)
    _validate_learning_rate(learning_rate)
    if reduce_lr_patience < 0:
        raise ValueError("reduce_lr_patience must not be negative")
    if not 0 < reduce_lr_factor < 1:
        raise ValueError("reduce_lr_factor must be between 0 and 1")
    if min_learning_rate < 0:
        raise ValueError("min_learning_rate must not be negative")
    _validate_head_regularization(head_dropout, head_l2)
    if (validation_images is not None and validation_split) or (
        validation_images and not validation_split == 0.0
    ):
        raise ValueError(
            "Use either validation_records or validation_split, not both nor neither"
        )
    if not 0 <= validation_split < 1:
        raise ValueError("validation_split must be between 0 and 1")
    train_records, validation_records_for_reporting = _split_training_records(
        train_images, validation_images, validation_split, seed
    )
    if output_path.exists() and not force:
        logging.info("Model already exists at %s; skipping training", output_path)
        model = _keras().models.load_model(output_path)
    else:
        logging.info("Training model to %s", output_path)

        # Try avoiding memory errors
        if os.environ.get("PYTORCH_CUDA_ALLOC_CONF") is None:
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

        model = _fit_model(
            train_records,
            classes,
            output_path,
            validation_records=validation_records_for_reporting,
            image_size=image_size,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            crop_window_scale=crop_window_scale,
            backbone=backbone,
            weights=weights,
            epochs=epochs,
            seed=seed,
            frozen_epochs=frozen_backbone_epochs,
            augment=augment,
            unfrozen_backbone_layers=unfrozen_backbone_layers,
            learning_rate=learning_rate,
            head_dropout=head_dropout,
            head_l2=head_l2,
            class_weighting=class_weighting,
            early_stopping_patience=early_stopping_patience,
            reduce_lr_patience=reduce_lr_patience,
            reduce_lr_factor=reduce_lr_factor,
            min_learning_rate=min_learning_rate,
            monitor_metric=monitor_metric,
            batch_size=batch_size,
            workers=workers,
        )

    return model


def _read_prediction_cache(
    output_path: Path, labels: list[str]
) -> dict[str, dict[str, float]]:
    """Read and validate cached predictions from a label-specific CSV file."""
    fieldnames = ["image_path", *(f"{label}_probability" for label in labels)]
    cached_predictions: dict[str, dict[str, float]] = {}
    with output_path.open(encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        if reader.fieldnames != fieldnames:
            raise ValueError(
                f"Prediction CSV must have columns in this order: {fieldnames}"
            )
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(
                row.get(fieldname) is None for fieldname in fieldnames
            ):
                raise ValueError(f"Malformed prediction CSV row {row_number}")
            raw_image_path = row["image_path"]
            if not raw_image_path.strip():
                raise ValueError(
                    f"Prediction CSV row {row_number} has an empty image path"
                )
            image_key = str(Path(raw_image_path).resolve())
            if image_key in cached_predictions:
                raise ValueError(
                    f"Prediction CSV contains a duplicate image path: {raw_image_path}"
                )
            predictions: dict[str, float] = {}
            for label in labels:
                try:
                    probability = float(row[f"{label}_probability"])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid probability on prediction CSV row {row_number}"
                    ) from error
                if not np.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError(
                        f"Invalid probability on prediction CSV row {row_number}"
                    )
                predictions[label] = probability
            cached_predictions[image_key] = predictions
    return cached_predictions


def _iter_prediction_batches(dataset: Any, workers: int) -> Any:
    """Yield ordered image batches, prefetching with background threads."""
    batch_count = len(dataset)
    if not workers:
        for batch_index in range(batch_count):
            yield dataset[batch_index]
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        next_batch = min(workers, batch_count)
        futures = {
            batch_index: executor.submit(dataset.__getitem__, batch_index)
            for batch_index in range(next_batch)
        }
        for batch_index in range(batch_count):
            yield futures.pop(batch_index).result()
            if next_batch < batch_count:
                futures[next_batch] = executor.submit(dataset.__getitem__, next_batch)
                next_batch += 1


def predict_images(
    model_path: Path,
    image_paths: list[Path] | list[ImageRecord],
    batch_size: int = 12,
    workers: int = 4,
    output_path: Path | None = None,
) -> list[dict[str, float]]:
    """Predict batches, optionally writing probabilities to a CSV file.

    Args:
        model_path: Saved Keras model path with adjacent JSON metadata.
        image_paths: Image paths or labeled ``ImageRecord`` values to classify.
        batch_size: Number of images loaded and predicted per batch.
        workers: Number of background threads used to load batches ahead of
            prediction. Set to zero to load synchronously.
        output_path: Optional path for the prediction CSV. Existing rows are
            reused by resolved image path; new predictions are staged in a
            sibling ``_busy.csv`` file until prediction completes.

    Returns:
        One label-to-probability mapping per input image, in input order.

    Raises:
        ValueError: If the prediction CSV is malformed.
    """
    metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
    image_size = tuple(metadata["image_size"])
    crop_to_aspect_ratio = metadata.get("crop_to_aspect_ratio", False)
    crop_window_scale = _crop_window_scale_from_metadata(metadata)
    labels = metadata["labels"]

    records = [
        image_path
        if isinstance(image_path, ImageRecord)
        else ImageRecord(image_path, ())
        for image_path in image_paths
    ]
    input_keys = [str(record.image_path.resolve()) for record in records]

    busy_path = (
        output_path.with_stem(f"{output_path.stem}_busy")
        if output_path is not None
        else None
    )
    cached_predictions: dict[str, dict[str, float]] = {}

    if output_path is not None and output_path.exists():
        ready_predictions = _read_prediction_cache(output_path, labels)
        if all(key in ready_predictions for key in input_keys):
            return [dict(ready_predictions[key]) for key in input_keys]
        cached_predictions.update(ready_predictions)

    if busy_path is not None and busy_path.exists():
        cached_predictions.update(_read_prediction_cache(busy_path, labels))

    pending_records = []
    pending_keys = []
    seen_keys = set(cached_predictions)
    for record, image_key in zip(records, input_keys, strict=True):
        if image_key not in seen_keys:
            pending_records.append(record)
            pending_keys.append(image_key)
            seen_keys.add(image_key)

    fieldnames = ["image_path", *(f"{label}_probability" for label in labels)]
    if busy_path is not None:
        busy_path.parent.mkdir(parents=True, exist_ok=True)
        with busy_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(
                {
                    "image_path": image_key,
                    **{f"{label}_probability": predictions[label] for label in labels},
                }
                for image_key, predictions in cached_predictions.items()
            )
    if not pending_records:
        if busy_path is not None and output_path is not None:
            busy_path.replace(output_path)
        return [dict(cached_predictions[key]) for key in input_keys]

    keras = _keras()
    model = keras.models.load_model(model_path)
    image_sequence_class = _make_image_sequence_class(keras)
    dataset = image_sequence_class(
        pending_records,
        image_size,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        crop_window_scale=crop_window_scale,
        batch_size=batch_size,
    )
    output_context = (
        busy_path.open("a", encoding="utf-8", newline="")
        if busy_path is not None
        else nullcontext(None)
    )
    with output_context as output_file:
        writer = (
            csv.DictWriter(output_file, fieldnames=fieldnames)
            if output_file is not None
            else None
        )
        with tqdm(total=len(pending_records), desc="Predict", unit="image") as progress:
            for batch_index, batch_images in enumerate(
                _iter_prediction_batches(dataset, workers)
            ):
                batch_start = batch_index * batch_size
                batch_keys = pending_keys[batch_start : batch_start + batch_size]
                batch_count = len(batch_keys)
                batch_predictions = np.asarray(model.predict_on_batch(batch_images))
                expected_shape = (batch_count, len(labels))
                if batch_predictions.shape != expected_shape:
                    raise ValueError(
                        "Model returned prediction shape "
                        f"{batch_predictions.shape}, expected {expected_shape}"
                    )
                rows = []
                for image_key, probability_row in zip(
                    batch_keys, batch_predictions, strict=True
                ):
                    predictions = dict(
                        zip(
                            labels,
                            (float(value) for value in probability_row),
                            strict=True,
                        )
                    )
                    cached_predictions[image_key] = predictions
                    rows.append(
                        {
                            "image_path": image_key,
                            **{
                                f"{label}_probability": predictions[label]
                                for label in labels
                            },
                        }
                    )
                if writer is not None and output_file is not None:
                    writer.writerows(rows)
                    output_file.flush()
                progress.update(batch_count)

    if busy_path is not None and output_path is not None:
        busy_path.replace(output_path)

    predictions = [dict(cached_predictions[key]) for key in input_keys]
    return predictions


def prediction_evaluation_is_complete(
    output_dir: Path, thresholds: Collection[float]
) -> bool:
    """Return whether all threshold-specific evaluation reports are complete."""
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or not all(0 <= threshold <= 1 for threshold in thresholds)
    ):
        return False

    return all(
        (report_dir / "prediction-results.csv").is_file()
        and (report_dir / "prediction-statistics.json").is_file()
        for report_dir in (
            output_dir / f"threshold-{threshold:g}" for threshold in thresholds
        )
    )


def write_prediction_evaluation(
    records: list[ImageRecord],
    classes: list[str],
    predictions_by_path: dict[str, dict[str, float]] | Path,
    output_dir: Path,
    thresholds: Collection[float] = (0.5,),
) -> None:
    """Write information that makes it possible to evaluate the predictions.

    Args:
        records: Labeled images to include in the reports.
        classes: Model classes, in prediction-column order.
        predictions_by_path: Predictions keyed by resolved image path, or a CSV
            file containing predictions in the format written by
            ``predict_images``.
        output_dir: Base directory for threshold-specific reports.
        thresholds: Unique probability thresholds in [0, 1].

    Raises:
        ValueError: If thresholds are invalid, a record has no prediction, or a
            record contains a label outside ``classes``.
    """
    if not records:
        return
    if isinstance(predictions_by_path, Path):
        predictions_by_path = _read_prediction_cache(predictions_by_path, classes)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or not all(0 <= threshold <= 1 for threshold in thresholds)
    ):
        raise ValueError("Thresholds must be unique values between 0 and 1")
    for record in records:
        path_key = str(record.image_path.resolve())
        if path_key not in predictions_by_path:
            raise ValueError(f"Prediction missing for image: {record.image_path}")
        unknown_labels = set(record.labels) - set(classes)
        if unknown_labels:
            raise ValueError(
                f"Image labels missing from classes: {sorted(unknown_labels)}"
            )

    for threshold in thresholds:
        report_dir = output_dir / f"threshold-{threshold:g}"
        report_path = report_dir / "prediction-results.csv"
        statistics_path = report_dir / "prediction-statistics.json"
        if report_path.exists() and statistics_path.exists():
            continue
        write_prediction_evaluation_info(
            records,
            classes,
            predictions_by_path,
            threshold=threshold,
            output_dir=report_dir,
        )


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
    crop_to_aspect_ratio = metadata.get("crop_to_aspect_ratio", False)
    crop_window_scale = _crop_window_scale_from_metadata(metadata)
    keras = _keras()
    model = keras.models.load_model(model_path)
    image = _load_images(
        [ImageRecord(image_path, ())],
        image_size,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        crop_window_scale=crop_window_scale,
    )
    logging.info("Predicting %s", image_path)
    probabilities = model.predict(image, verbose=0)[0]
    logging.info("Prediction complete for %s", image_path)
    return dict(
        zip(metadata["labels"], (float(value) for value in probabilities), strict=True)
    )
