"""Add class-balanced JRC images to train, validation, and test directories."""

from __future__ import annotations

import csv
import shutil
from collections import defaultdict
from pathlib import Path

CSV_PATH = Path(r"X:\Monitoring\ControlefotosJRC\traindata_with_simplified_classes.csv")
DATASET_ROOT = Path(r"X:\Monitoring\geophototagger\trainingsdata")
SPLITS = (
    ("train", 0, 10),
    ("validation", 10, 15),
    ("test", 15, 215),
    ("test_extra", 215, None),
)


def read_class_rows(csv_path: Path) -> dict[str, list[dict[str, str]]]:
    """Read CSV rows grouped by simplified English class in source order.

    Args:
        csv_path: CSV containing ``PATH`` and ``CLASSES_SIMPLIFIED_EN`` columns.

    Returns:
        Rows grouped by class, with duplicate source paths removed per class.

    Raises:
        ValueError: If required columns are missing or a source path is absent.
    """
    with csv_path.open(encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"PATH", "CLASSES_SIMPLIFIED_EN"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"CSV must contain {sorted(required)}")
        grouped: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for row in reader:
            class_name = (row.get("CLASSES_SIMPLIFIED_EN") or "").strip()
            source_path = Path((row.get("PATH") or "").strip())
            if not class_name or not source_path:
                raise ValueError("CSV contains an empty class or source path")
            source_key = str(source_path).lower()
            if source_key not in seen[class_name]:
                grouped[class_name].append(row)
                seen[class_name].add(source_key)
    return dict(grouped)


def plan_copies(
    grouped_rows: dict[str, list[dict[str, str]]],
) -> list[tuple[str, Path, Path]]:
    """Plan copies using 10 train, 5 validation, 200 test, and extra images."""
    planned: list[tuple[str, Path, Path]] = []
    for class_name in sorted(grouped_rows):
        rows = grouped_rows[class_name]
        for split_name, start, end in SPLITS:
            for row in rows[start:end]:
                source_path = Path(row["PATH"])
                target_name = f"{source_path.stem}__{class_name}{source_path.suffix}"
                planned.append(
                    (split_name, source_path, DATASET_ROOT / split_name / target_name)
                )
    return planned


def copy_images(planned: list[tuple[str, Path, Path]]) -> None:
    """Synchronize planned images without overwriting existing destinations."""
    missing = [str(source) for _, source, _ in planned if not source.is_file()]
    if missing:
        raise ValueError(f"Missing source images, first examples: {missing[:5]}")
    target_keys = [str(target).lower() for _, _, target in planned]
    if len(target_keys) != len(set(target_keys)):
        raise ValueError("Planned copies contain duplicate destination names")
    for split, source, target in planned:
        if target.exists():
            continue
        alternate_split = (
            "test_extra"
            if split == "test"
            else "test"
            if split == "test_extra"
            else None
        )
        if alternate_split is not None:
            alternate = DATASET_ROOT / alternate_split / target.name
            if alternate.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(alternate, target)
                continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


def main() -> None:
    """Copy the configured per-class image selections into dataset splits."""
    planned = plan_copies(read_class_rows(CSV_PATH))
    copy_images(planned)
    remaining = [target for _, _, target in planned if not target.exists()]
    print(f"Planned {len(planned)} images; remaining {len(remaining)}")


if __name__ == "__main__":
    main()
