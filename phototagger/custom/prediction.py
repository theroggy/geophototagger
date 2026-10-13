"""Prediction caching and evaluation reports for custom image taggers."""

from __future__ import annotations

import csv
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from tqdm.auto import tqdm

from .dataset import ImageRecord
from .image_processing import (
    _crop_window_scale_from_metadata,
    _keras,
    _load_images,
    _make_image_sequence_class,
)

if TYPE_CHECKING:
    from collections.abc import Collection


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
    interpolation = metadata.get("interpolation", "nearest")
    keras = _keras()
    model = keras.models.load_model(model_path)
    image = _load_images(
        [ImageRecord(image_path, ())],
        image_size,
        crop_to_aspect_ratio=crop_to_aspect_ratio,
        crop_window_scale=crop_window_scale,
        interpolation=interpolation,
    )
    logging.info("Predicting %s", image_path)
    probabilities = model.predict(image, verbose=0)[0]
    logging.info("Prediction complete for %s", image_path)
    return dict(
        zip(metadata["labels"], (float(value) for value in probabilities), strict=True)
    )


def predict_images(
    model_path: Path,
    images: list[Path] | list[ImageRecord],
    batch_size: int = 12,
    workers: int = 4,
    output_path: Path | None = None,
) -> list[dict[str, float]]:
    """Predict batches, optionally writing probabilities to a CSV file.

    Args:
        model_path: Saved Keras model path with adjacent JSON metadata.
        images: Image paths or labeled ``ImageRecord`` values to classify.
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
    interpolation = metadata.get("interpolation", "nearest")
    labels = metadata["labels"]

    records = [
        image_path
        if isinstance(image_path, ImageRecord)
        else ImageRecord(image_path, ())
        for image_path in images
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
            cache_writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            cache_writer.writeheader()
            cache_writer.writerows(
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
        interpolation=interpolation,
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

    ordered_predictions = [dict(cached_predictions[key]) for key in input_keys]
    return ordered_predictions


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
        all(
            path.is_file()
            for path in _prediction_evaluation_paths(output_dir, threshold)
        )
        for threshold in thresholds
    )


def write_prediction_evaluation(
    images: list[ImageRecord],
    classes: list[str],
    predictions: dict[str, dict[str, float]] | Path,
    output_dir: Path,
    thresholds: Collection[float] = (0.5,),
) -> None:
    """Write information that makes it possible to evaluate the predictions.

    Args:
        images: Labeled images to include in the reports.
        classes: Model classes, in prediction-column order.
        predictions: Predictions keyed by resolved image path, or a CSV file containing
            predictions in the format written by ``predict_images``.
        output_dir: Base directory for threshold-specific reports.
        thresholds: Unique probability thresholds in [0, 1].

    Raises:
        ValueError: If thresholds are invalid, a record has no prediction, or a
            record contains a label outside ``classes``.
    """
    if not images:
        return
    if isinstance(predictions, Path):
        predictions = _read_prediction_cache(predictions, classes)
    if (
        not thresholds
        or len(set(thresholds)) != len(thresholds)
        or not all(0 <= threshold <= 1 for threshold in thresholds)
    ):
        raise ValueError("Thresholds must be unique values between 0 and 1")
    for record in images:
        path_key = str(record.image_path.resolve())
        if path_key not in predictions:
            raise ValueError(f"Prediction missing for image: {record.image_path}")
        unknown_labels = set(record.labels) - set(classes)
        if unknown_labels:
            raise ValueError(
                f"Image labels missing from classes: {sorted(unknown_labels)}"
            )

    for threshold in thresholds:
        _write_prediction_image_evaluation(
            images,
            classes,
            predictions,
            threshold=threshold,
            output_dir=output_dir,
        )


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


def _write_prediction_image_evaluation(
    records: list[ImageRecord],
    classes: list[str],
    predictions_by_path: dict[str, dict[str, float]],
    *,
    threshold: float,
    output_dir: Path,
) -> None:
    """Write image evaluation information for one classification threshold.

    This function evaluates the predictions for each image against the expected labels,
    calculates per-label statistics, and writes the results to a CSV file.

    It also creates subdirectories with the images split into "ok" and "nok" directories
    based on whether they were correctly classified according to the specified
    threshold.

    Args:
        records: A list of ImageRecord objects representing the images to evaluate.
        classes: A list of class labels.
        predictions_by_path: A dictionary mapping image paths to predicted
            probabilities for each class.
        threshold: The classification threshold for determining predicted labels.
        output_dir: The directory where evaluation results will be written.
    """
    report_path, statistics_path = _prediction_evaluation_paths(output_dir, threshold)
    if report_path.exists() and statistics_path.exists():
        return

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
    output_ok_dir = output_dir / f"threshold-{threshold:g}_ok"
    output_ok_dir.mkdir(parents=True, exist_ok=True)
    output_nok_dir = output_dir / f"threshold-{threshold:g}_nok"
    output_nok_dir.mkdir(parents=True, exist_ok=True)
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
        for record in tqdm(records, desc=f"Create eval dirs {threshold}", unit="image"):
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

            if correctly_classified:
                dst_path = output_ok_dir / record.image_path.name
                json_dst_path = output_ok_dir / f"{record.image_path.stem}.json"
            else:
                dst_path = output_nok_dir / record.image_path.name
                json_dst_path = output_nok_dir / f"{record.image_path.stem}.json"

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

    statistics_path.write_text(
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


def _prediction_evaluation_paths(
    output_dir: Path, threshold: float
) -> tuple[Path, Path]:
    """Return the report and statistics paths for one evaluation threshold."""
    return (
        output_dir / f"prediction-report_{threshold:g}.csv",
        output_dir / f"prediction-statistics_{threshold:g}.json",
    )
