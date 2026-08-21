from pathlib import Path

import pytest

from geophototagger.custom.classifier import calculate_class_weights
from geophototagger.custom.dataset import (
    ImageRecord,
    discover_records,
    encode_labels,
    label_vocabulary,
    read_manifest,
    write_manifest,
)


def test_manifest_discovers_images_and_ignores_artifacts(tmp_path: Path) -> None:
    dataset_root = tmp_path / "training"
    dataset_root.mkdir()
    (dataset_root / "one__maize.JPG").write_bytes(b"image")
    (dataset_root / "two__manure.png").write_bytes(b"image")
    (dataset_root / "Thumbs.db").write_bytes(b"artifact")
    (dataset_root / "invalid.txt").write_text("ignored", encoding="utf-8")

    manifest_path = tmp_path / "manifests" / "images.csv"
    assert write_manifest(dataset_root, manifest_path) == 2

    records = read_manifest(manifest_path, dataset_root=dataset_root)
    assert [record.image_path.name for record in records] == [
        "one__maize.JPG",
        "two__manure.png",
    ]
    assert [record.labels for record in records] == [("maize",), ("manure",)]


def test_manifest_round_trip_supports_multiple_labels(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")
    manifest_path = tmp_path / "images.csv"
    manifest_path.write_text(
        "image_path,labels\nphoto.jpg,maize;manure\n", encoding="utf-8"
    )

    records = read_manifest(manifest_path, dataset_root=tmp_path)

    assert label_vocabulary(records) == ["maize", "manure"]
    assert encode_labels(records, ["maize", "manure"]) == [[1, 1]]


def test_manifest_rejects_missing_images(tmp_path: Path) -> None:
    manifest_path = tmp_path / "images.csv"
    manifest_path.write_text("image_path,labels\nmissing.jpg,maize\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing image"):
        read_manifest(manifest_path, dataset_root=tmp_path)


def test_discover_records_reads_formatted_flat_names(tmp_path: Path) -> None:
    (tmp_path / "original_stem__fake-korrelmais.JPG").write_bytes(b"image")

    records = discover_records(tmp_path)

    assert records[0].labels == ("fake", "korrelmais")


def test_whitelist_filters_labels_and_drops_unmatched_images(tmp_path: Path) -> None:
    (tmp_path / "stem__fake-korrelmais.png").write_bytes(b"image")
    (tmp_path / "stem2__stalmest.png").write_bytes(b"image")

    records = discover_records(
        tmp_path, {"korrelmais"}, include_unmatched_as_negative=False
    )

    assert len(records) == 1
    assert records[0].labels == ("korrelmais",)


def test_whitelist_can_keep_unmatched_images_as_negative_examples(
    tmp_path: Path,
) -> None:
    (tmp_path / "stem__fake-korrelmais.png").write_bytes(b"image")
    (tmp_path / "stem2__stalmest.png").write_bytes(b"image")

    records = discover_records(
        tmp_path, {"korrelmais"}, include_unmatched_as_negative=True
    )

    assert len(records) == 2
    assert {record.labels for record in records} == {("korrelmais",), ()}


def test_manifest_round_trip_supports_negative_examples(tmp_path: Path) -> None:
    image_path = tmp_path / "photo.jpg"
    image_path.write_bytes(b"image")
    manifest_path = tmp_path / "images.csv"

    write_manifest(
        tmp_path,
        manifest_path,
        records=[ImageRecord(image_path, ())],
    )
    records = read_manifest(manifest_path, dataset_root=tmp_path)

    assert records[0].labels == ()
    assert encode_labels(records, ["fake", "korrelmais"]) == [[0, 0]]


def test_class_weights_balance_positive_and_negative_examples() -> None:
    weights = calculate_class_weights(
        [[1, 0], [1, 0], [0, 1], [0, 0]], ["common", "rare"]
    )

    assert weights["common"] == {"positive": 1.0, "negative": 1.0}
    assert weights["rare"] == {"positive": 2.0, "negative": 0.6666666666666666}
