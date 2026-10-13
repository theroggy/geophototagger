from pathlib import Path

import pytest

from phototagger.custom.dataset import (
    ImageRecord,
    determine_classes,
    discover_records,
    discover_records_csv,
    encode_labels,
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

    assert determine_classes(records) == ["maize", "manure"]
    assert encode_labels(records, ["maize", "manure"]) == [[1, 1]]


def test_manifest_skips_existing_file_unless_forced(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_path = tmp_path / "photo__maize.jpg"
    image_path.write_bytes(b"image")
    manifest_path = tmp_path / "images.csv"
    manifest_path.write_text("original manifest", encoding="utf-8")

    assert write_manifest(tmp_path, manifest_path) == 0

    assert manifest_path.read_text(encoding="utf-8") == "original manifest"
    assert "Manifest already exists, leaving it unchanged" in caplog.text
    assert write_manifest(tmp_path, manifest_path, force=True) == 1


def test_manifest_rejects_missing_images(tmp_path: Path) -> None:
    manifest_path = tmp_path / "images.csv"
    manifest_path.write_text("image_path,labels\nmissing.jpg,maize\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing image"):
        read_manifest(manifest_path, dataset_root=tmp_path)


def test_discover_records_reads_formatted_flat_names(tmp_path: Path) -> None:
    (tmp_path / "original_stem__fake-korrelmais.JPG").write_bytes(b"image")

    records = discover_records(tmp_path)

    assert records[0].labels == ("fake", "korrelmais")


def test_discover_records_csv_reads_relative_paths_and_labels(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    (image_dir / "second.JPG").write_bytes(b"image")
    (image_dir / "first.jpg").write_bytes(b"image")
    csv_path = tmp_path / "overview.csv"
    csv_path.write_text(
        "image,type\nsecond.JPG,bufet; raai;bufet\nfirst.jpg,raai\n",
        encoding="utf-8",
    )

    records = discover_records_csv(csv_path, "image", "type", image_dir)

    assert [record.image_path.name for record in records] == ["first.jpg", "second.JPG"]
    assert [record.labels for record in records] == [("raai",), ("bufet", "raai")]

    filtered_records = discover_records_csv(
        csv_path, "image", "type", image_dir, label_whitelist={"bufet"}
    )
    assert [record.labels for record in filtered_records] == [(), ("bufet",)]

    matched_records = discover_records_csv(
        csv_path,
        "image",
        "type",
        image_dir,
        label_whitelist={"bufet"},
        include_unmatched_as_negative=False,
    )
    assert [record.labels for record in matched_records] == [("bufet",)]


def test_discover_records_csv_rejects_missing_columns_and_skips_missing_images(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    csv_path = tmp_path / "overview.csv"
    csv_path.write_text("filename,type\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV must contain columns"):
        discover_records_csv(csv_path, "image", "type", image_dir)

    csv_path.write_text("image,type\nmissing.jpg,bufet\n", encoding="utf-8")
    records = discover_records_csv(csv_path, "image", "type", image_dir)

    assert records == []
    assert "CSV row 2 references missing image; skipping:" in caplog.text
    assert str(image_dir / "missing.jpg") in caplog.text


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
