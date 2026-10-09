"""Dataset discovery and manifest utilities for image classification."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection

SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
LABEL_SEPARATOR = ";"
FORMATTED_LABEL_SEPARATOR = "__"


@dataclass(frozen=True)
class ImageRecord:
    """An image path and its one or more classification labels."""

    image_path: Path
    labels: tuple[str, ...]


def discover_records(
    dataset_root: Path,
    label_whitelist: Collection[str] | None = None,
    *,
    include_unmatched_as_negative: bool = True,
) -> list[ImageRecord]:
    """Discover flat image records and parse labels from their filenames.

    Filenames must contain a final ``__`` separator followed by one or more
    hyphen-separated labels, for example ``photo__fake-korrelmais.jpg``.

    Args:
        dataset_root: Directory containing the flat image dataset.
        label_whitelist: Optional labels to retain. Images with no retained
            labels are excluded, unless ``include_unmatched_as_negative`` is set.
        include_unmatched_as_negative: When ``label_whitelist`` is set, keep
            images with no whitelisted label as explicit negative examples
            (an empty label tuple) instead of dropping them.

    Returns:
        Image records sorted by their resolved image path.

    Raises:
        ValueError: If ``dataset_root`` is not a directory or an image filename
            does not contain a valid label suffix.
    """
    if not dataset_root.is_dir():
        raise ValueError(
            f"Dataset root does not exist or is not a directory: {dataset_root}"
        )

    allowed_labels = (
        {label.strip() for label in label_whitelist if label.strip()}
        if label_whitelist is not None
        else None
    )
    records = []
    for image_path in sorted(dataset_root.iterdir()):
        if (
            not image_path.is_file()
            or image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES
        ):
            continue
        name_parts = image_path.stem.rsplit(FORMATTED_LABEL_SEPARATOR, 1)
        if len(name_parts) != 2 or not name_parts[1]:
            raise ValueError(
                f"Formatted image name does not contain a label: {image_path}"
            )
        labels = {label for label in name_parts[1].split("-") if label}
        if allowed_labels is None:
            if labels:
                records.append(ImageRecord(image_path.resolve(), tuple(sorted(labels))))
            continue
        matched_labels = labels & allowed_labels
        if matched_labels:
            records.append(
                ImageRecord(image_path.resolve(), tuple(sorted(matched_labels)))
            )
        elif include_unmatched_as_negative:
            records.append(ImageRecord(image_path.resolve(), ()))
    return records


def discover_records_csv(
    csv_path: Path,
    image_path_column: str,
    classes_column: str,
    image_dir: Path,
    label_whitelist: Collection[str] | None = None,
    *,
    include_unmatched_as_negative: bool = True,
) -> list[ImageRecord]:
    """Discover image records from a CSV containing paths and classes.

    Relative image paths are resolved against ``image_dir``. Class values may
    contain multiple labels separated by ``LABEL_SEPARATOR``.

    Args:
        csv_path: CSV containing image paths and class labels.
        image_path_column: Name of the CSV column containing image paths.
        classes_column: Name of the CSV column containing class labels.
        image_dir: Directory used to resolve relative image paths.
        label_whitelist: Optional labels to retain. Images with no retained
            labels are excluded, unless ``include_unmatched_as_negative`` is set.
        include_unmatched_as_negative: When ``label_whitelist`` is set, keep
            images with no whitelisted label as explicit negative examples
            (an empty label tuple) instead of dropping them.

    Returns:
        Image records sorted by their resolved image path.

    Raises:
        ValueError: If ``image_dir`` is not a directory, a required column is
            missing, a CSV row has no image path, or a referenced image is
            missing.
    """
    if not image_dir.is_dir():
        raise ValueError(f"Image directory does not exist: {image_dir}")

    allowed_labels = (
        {label.strip() for label in label_whitelist if label.strip()}
        if label_whitelist is not None
        else None
    )
    records = []
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required_columns = {image_path_column, classes_column}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain columns {sorted(required_columns)}")

        for row_number, row in enumerate(reader, start=2):
            raw_image_path = (row.get(image_path_column) or "").strip()
            if not raw_image_path:
                raise ValueError(f"CSV row {row_number} must contain an image path")
            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                image_path = image_dir / image_path
            if image_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
                continue
            if not image_path.is_file():
                logging.warning(
                    "CSV row %s references missing image; skipping: %s",
                    row_number,
                    image_path,
                )
                continue

            raw_labels = (row.get(classes_column) or "").strip()
            labels = tuple(
                sorted(
                    {
                        label.strip()
                        for label in raw_labels.split(LABEL_SEPARATOR)
                        if label.strip()
                    }
                )
            )
            if allowed_labels is None:
                if labels:
                    records.append(ImageRecord(image_path.resolve(), labels))
                continue
            matched_labels = set(labels) & allowed_labels
            if matched_labels:
                records.append(
                    ImageRecord(image_path.resolve(), tuple(sorted(matched_labels)))
                )
            elif include_unmatched_as_negative:
                records.append(ImageRecord(image_path.resolve(), ()))

    return sorted(records, key=lambda record: record.image_path)


def write_manifest(
    dataset_root: Path,
    manifest_path: Path,
    records: list[ImageRecord] | None = None,
    classes: Collection[str] | None = None,
    *,
    include_unmatched_as_negative: bool = True,
    force: bool = False,
) -> int:
    """Write a deterministic ``image_path,labels`` CSV manifest.

    Args:
        dataset_root: Root directory used to make paths relative where possible.
        manifest_path: Destination path for the CSV file.
        records: Records to write. If omitted, records are discovered first.
        classes: Optional labels to retain while discovering or writing
            records.
        include_unmatched_as_negative: When ``classes`` is set, keep
            unmatched images as explicit negative examples instead of
            dropping them.
        force: Replace an existing manifest. Otherwise, an existing manifest is
            left unchanged.

    Returns:
        Number of records written to the manifest.
    """
    if manifest_path.exists() and not force:
        logging.info("Manifest already exists, leaving it unchanged: %s", manifest_path)
        return 0

    if records is None:
        records = discover_records(
            dataset_root,
            classes,
            include_unmatched_as_negative=include_unmatched_as_negative,
        )
    elif classes is not None:
        allowed_labels = {label.strip() for label in classes if label.strip()}
        filtered_records = []
        for record in records:
            matched_labels = set(record.labels) & allowed_labels
            if matched_labels:
                filtered_records.append(
                    ImageRecord(record.image_path, tuple(sorted(matched_labels)))
                )
            elif include_unmatched_as_negative:
                filtered_records.append(ImageRecord(record.image_path, ()))
        records = filtered_records
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=["image_path", "labels"])
        writer.writeheader()
        for record in records:
            try:
                image_path = record.image_path.relative_to(dataset_root.resolve())
            except ValueError:
                image_path = record.image_path
            writer.writerow(
                {
                    "image_path": image_path.as_posix(),
                    "labels": LABEL_SEPARATOR.join(record.labels),
                }
            )

    return len(records)


def read_manifest(
    manifest_path: Path, *, dataset_root: Path | None = None
) -> list[ImageRecord]:
    """Read and validate a CSV manifest.

    Args:
        manifest_path: CSV file containing ``image_path`` and ``labels`` columns.
        dataset_root: Directory used to resolve relative image paths.

    Returns:
        Validated image records with normalized, deduplicated labels.

    Raises:
        ValueError: If the columns, rows, labels, or referenced image files are
            invalid, or if the manifest contains no records.
    """
    records: list[ImageRecord] = []
    with manifest_path.open(encoding="utf-8", newline="") as manifest_file:
        reader = csv.DictReader(manifest_file)
        if reader.fieldnames != ["image_path", "labels"]:
            raise ValueError(
                "Manifest must contain exactly image_path and labels columns"
            )
        for row_number, row in enumerate(reader, start=2):
            raw_path = (row.get("image_path") or "").strip()
            labels = tuple(
                label.strip()
                for label in (row.get("labels") or "").split(LABEL_SEPARATOR)
                if label.strip()
            )
            if not raw_path:
                raise ValueError(f"Manifest row {row_number} must contain a path")
            image_path = Path(raw_path)
            if dataset_root is not None and not image_path.is_absolute():
                image_path = dataset_root / image_path
            if not image_path.is_file():
                raise ValueError(
                    f"Manifest row {row_number} references missing image: {image_path}"
                )
            records.append(
                ImageRecord(image_path.resolve(), tuple(sorted(set(labels))))
            )
    if not records:
        raise ValueError("Manifest contains no image records")
    return records


def determine_classes(images: list[ImageRecord]) -> list[str]:
    """Build a sorted class vocabulary from image records.

    Args:
        images: Image records whose labels should be collected.

    Returns:
        Sorted unique classes suitable for multi-hot encoding.
    """
    return sorted({label for record in images for label in record.labels})


def encode_labels(records: list[ImageRecord], vocabulary: list[str]) -> list[list[int]]:
    """Encode image labels as multi-hot vectors.

    Args:
        records: Image records to encode.
        vocabulary: Ordered labels defining vector positions.

    Returns:
        One integer vector per record, with one for each present label.

    Raises:
        ValueError: If a record contains a label absent from ``vocabulary``.
    """
    label_indexes = {label: index for index, label in enumerate(vocabulary)}
    encoded: list[list[int]] = []
    for record in records:
        unknown_labels = set(record.labels) - label_indexes.keys()
        if unknown_labels:
            raise ValueError(
                f"Labels missing from vocabulary: {sorted(unknown_labels)}"
            )
        encoded.append([int(label in record.labels) for label in vocabulary])
    return encoded
