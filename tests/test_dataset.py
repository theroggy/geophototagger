import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from geophototagger.custom import classifier
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


def test_prediction_progress_logs_once_per_minute_and_on_completion(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    clock_values = iter([0.0, 30.0, 60.0, 90.0, 91.0])
    monkeypatch.setattr(classifier.time, "monotonic", lambda: next(clock_values))
    keras = SimpleNamespace(
        callbacks=SimpleNamespace(
            LambdaCallback=lambda **kwargs: SimpleNamespace(**kwargs)
        )
    )

    callback = classifier._prediction_progress_callback(keras, 100, 10)
    callback.on_predict_batch_end(0, logs={})
    callback.on_predict_batch_end(1)
    callback.on_predict_batch_end(2)
    callback.on_predict_batch_end(9)

    assert "Prediction progress: 10/100 files (10%)" not in caplog.text
    assert (
        "Prediction progress: 20/100 files (20%); estimated 0:04:00 remaining"
        in caplog.text
    )
    assert "Prediction progress: 30/100 files (30%)" not in caplog.text
    assert (
        "Prediction progress: 100/100 files (100%); estimated 0:00:00 remaining"
        in caplog.text
    )


def test_misclassification_report_includes_correct_and_incorrect_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")

    class FakeDataset:
        def __init__(self, records, *_args, **_kwargs) -> None:
            self.records = records

    class FakeModel:
        def predict(
            self, _dataset, verbose: int, callbacks: list[SimpleNamespace]
        ) -> list[list[float]]:
            assert verbose == 0
            callbacks[0].on_predict_batch_end(0)
            return [[0.9, 0.1], [0.8, 0.2]]

    monkeypatch.setattr(
        classifier,
        "_keras",
        lambda: SimpleNamespace(
            callbacks=SimpleNamespace(
                LambdaCallback=lambda **kwargs: SimpleNamespace(**kwargs)
            )
        ),
    )
    monkeypatch.setattr(
        classifier, "_make_image_sequence_class", lambda _keras: FakeDataset
    )

    output_dir = tmp_path / "report"
    classifier._write_misclassified(
        FakeModel(),
        [
            ImageRecord(first_image, ("maize",)),
            ImageRecord(second_image, ("manure",)),
        ],
        ["maize", "manure"],
        image_size=(224, 224),
        threshold=0.5,
        output_dir=output_dir,
        workers=0,
    )

    with (output_dir / "classification-results.csv").open(
        encoding="utf-8", newline=""
    ) as report_file:
        rows = list(csv.DictReader(report_file))

    assert rows == [
        {
            "image_path": str(first_image),
            "expected_labels": "maize",
            "predicted_labels": "maize",
            "correctly_classified": "True",
            "maize_probability": "0.9",
            "manure_probability": "0.1",
        },
        {
            "image_path": str(second_image),
            "expected_labels": "manure",
            "predicted_labels": "maize",
            "correctly_classified": "False",
            "maize_probability": "0.8",
            "manure_probability": "0.2",
        },
    ]
    assert not (output_dir / "first.jpg").exists()
    assert (output_dir / "second.jpg").exists()
    assert (output_dir / "second.json").exists()
    assert "Prediction progress: 2/2 files (100%)" in caplog.text
    statistics = json.loads(
        (output_dir / "classification-statistics.json").read_text(encoding="utf-8")
    )
    assert statistics["total_files"] == 2
    assert statistics["correctly_classified_files"] == 1
    assert statistics["correctly_classified_percentage"] == 50.0
    assert statistics["misclassified_files"] == 1
    assert statistics["per_label"]["maize"] == {
        "true_positives": 1,
        "false_positives": 1,
        "true_negatives": 0,
        "false_negatives": 0,
        "accuracy": 0.5,
        "precision": 0.5,
        "recall": 1.0,
        "f1_score": 2 / 3,
    }


def test_existing_model_skips_training_and_still_writes_reports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    training_record = ImageRecord(tmp_path / "training.jpg", ("maize",))
    validation_record = ImageRecord(tmp_path / "validation.jpg", ("maize",))
    test_record = ImageRecord(tmp_path / "test.jpg", ("maize",))
    model = object()
    reports = []

    monkeypatch.setattr(
        classifier,
        "_keras",
        lambda: SimpleNamespace(models=SimpleNamespace(load_model=lambda _path: model)),
    )
    monkeypatch.setattr(
        classifier,
        "_fit_model",
        lambda *_args, **_kwargs: pytest.fail("Existing models must not be trained"),
    )
    monkeypatch.setattr(
        classifier,
        "_write_misclassified",
        lambda reported_model, reported_records, *_args, **kwargs: reports.append(
            (reported_model, reported_records, kwargs["output_dir"].name)
        ),
    )

    returned_model = classifier.train_model(
        [training_record],
        ["maize"],
        model_path,
        validation_records=[validation_record],
        test_records=[test_record],
        misclassified_dir=tmp_path / "reports",
    )

    assert returned_model is model
    assert reports == [
        (model, [training_record], "train_wrong"),
        (model, [validation_record], "validation_wrong"),
        (model, [test_record], "test_wrong"),
    ]
