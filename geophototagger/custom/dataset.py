"""Dataset discovery and manifest utilities for image classification."""

from __future__ import annotations

import csv
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

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
                f"Formatted image name does not contain a label: {image_path.name}"
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


def write_manifest(
    dataset_root: Path,
    manifest_path: Path,
    records: list[ImageRecord] | None = None,
    classes: Collection[str] | None = None,
    *,
    include_unmatched_as_negative: bool = True,
) -> int:
    """Write a deterministic ``image_path,labels`` CSV manifest.

    Args:
        dataset_root: Root directory used to make paths relative where possible.
        manifest_path: Destination path for the CSV file.
        records: Records to write. If omitted, records are discovered first.
        label_whitelist: Optional labels to retain while discovering or writing
            records.
        include_unmatched_as_negative: When ``label_whitelist`` is set, keep
            unmatched images as explicit negative examples instead of
            dropping them.

    Returns:
        Number of records written to the manifest.
    """
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


def label_vocabulary(records: list[ImageRecord]) -> list[str]:
    """Build a sorted label vocabulary from image records.

    Args:
        records: Image records whose labels should be collected.

    Returns:
        Sorted unique labels suitable for multi-hot encoding.
    """
    return sorted({label for record in records for label in record.labels})


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
