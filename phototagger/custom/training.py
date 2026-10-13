"""Model construction and training for custom image taggers."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

import numpy as np

from .image_processing import (
    _keras,
    _make_image_sequence_class,
    _validate_crop_window_scale,
)

if TYPE_CHECKING:
    from .dataset import ImageRecord

logger = logging.getLogger(__name__)


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
        backbone: Name of the Keras Applications backbone.
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
    epochs: int = 300,
    validation_split: float = 0.0,
    seed: int = 42,
    frozen_backbone_epochs: int = 5,
    augment: bool = True,
    unfrozen_backbone_layers: int | float = 0.15,
    learning_rate: float = 1e-3,
    head_dropout: float = 0.5,
    head_l2: float = 0.005,
    class_weighting: bool = False,
    early_stopping_patience: int = 100,
    reduce_lr_patience: int = 30,
    reduce_lr_factor: float = 0.5,
    min_learning_rate: float = 1e-6,
    monitor_metric: str = "val_loss",
    additional_metrics: list[Any] | None = None,
    batch_size: int = 16,
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
        frozen_backbone_epochs: Initial epochs to train only the classification
            head before fine-tuning the feature extractor.
        augment: Whether to enable random training-time augmentation.
        unfrozen_backbone_layers: Number of final backbone layers to fine-tune, or
            a float fraction from 0 to 1. Fractional counts round up before
            optional whole-block expansion.
        learning_rate: Initial Adam learning rate for both training phases.
        head_dropout: Dropout rate applied before the classification output.
        head_l2: L2 regularization strength for the output kernel.
        class_weighting: Whether to apply balanced per-image sample weights.
        early_stopping_patience: Epochs with no monitored improvement before
            stopping. Set to zero to disable early stopping.
        reduce_lr_patience: Epochs without improvement before reducing the
            learning rate.
        reduce_lr_factor: Multiplier applied to the learning rate after a plateau.
        min_learning_rate: Lower bound for the learning rate.
        monitor_metric: Metric used to select the best checkpoint and to
            evaluate early stopping. A ``val_`` prefix is added automatically
            when validation data is available.
        additional_metrics: Keras metrics to add alongside the default metrics.
        batch_size: Number of images loaded into memory per training batch.
        workers: Number of background threads used to load image batches ahead
            of time. Set to zero to load synchronously on the main thread.
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
            additional_metrics=additional_metrics,
        )

    return model


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


def _compile_model(
    keras: Any,
    model: Any,
    learning_rate: float = 1e-3,
    *,
    additional_metrics: list[Any] | None = None,
) -> None:
    """Compile a classifier model with the configured optimizer and metrics."""
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryFocalCrossentropy(),
        metrics=[
            keras.metrics.BinaryAccuracy(name="binary_accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            *(additional_metrics or ()),
        ],
    )


def _make_csv_logger(keras: Any, path: Path, *, append: bool) -> Any:
    """Build a CSVLogger that reopens its file with newline="" to avoid blank rows."""

    class _CSVLogger(keras.callbacks.CSVLogger):
        def on_train_begin(self, logs: dict[str, Any] | None = None) -> None:
            super().on_train_begin(logs)
            self.csv_file.close()
            self.csv_file: TextIO = Path(self.filename).open(
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
    backbone: str = "EfficientNetV2B0",
    weights: str | None = "imagenet",
    augment: bool = True,
    *,
    learning_rate: float = 1e-3,
    head_dropout: float = 0.5,
    head_l2: float = 0.005,
    additional_metrics: list[Any] | None = None,
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
    _compile_model(
        keras,
        model,
        learning_rate=learning_rate,
        additional_metrics=additional_metrics,
    )
    return model, feature_extractor


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
    additional_metrics: list[Any] | None = None,
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
        epochs: Total epochs.
        seed: Random seed used by Keras.
        frozen_epochs: Initial epochs with a completely frozen backbone.
        augment: Whether to enable training-time image augmentation.
        unfrozen_backbone_layers: Number of final backbone layers to fine-tune, or
            a float fraction of the backbone layers.
        head_dropout: Dropout rate applied before the classification output.
        head_l2: L2 regularization strength for the output kernel.
        learning_rate: Initial Adam learning rate for both training phases.
        class_weighting: Whether to apply balanced per-image sample weights.
        early_stopping_patience: Epochs without improvement before stopping.
        reduce_lr_patience: Epochs without improvement before reducing learning rate.
        reduce_lr_factor: Multiplier applied to learning rate after a plateau.
        min_learning_rate: Lower bound for the learning rate.
        monitor_metric: Metric used to select checkpoints and stop training.
        batch_size: Images per training batch.
        workers: Background image-loading threads; zero loads synchronously.
        additional_metrics: Keras metrics to add alongside the default metrics.

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
        additional_metrics=additional_metrics,
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
        _compile_model(
            keras,
            model,
            learning_rate=learning_rate,
            additional_metrics=additional_metrics,
        )
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
