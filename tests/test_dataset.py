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
    determine_classes,
    read_manifest,
    write_manifest,
)


@pytest.fixture
def fake_progress(monkeypatch: pytest.MonkeyPatch):
    progress_instances = []

    class FakeProgress:
        def __init__(self, total: int, **_kwargs) -> None:
            self.total = total
            self.n = 0
            self.closed = False
            progress_instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.close()

        def update(self, amount: int) -> None:
            self.n += amount

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(classifier, "tqdm", FakeProgress)
    return progress_instances


def _make_lambda_callback(**kwargs):
    return SimpleNamespace(**kwargs)


def _configure_fake_predictor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    probabilities: dict[str, list[float]],
):
    model_path = tmp_path / "model.keras"
    model_path.with_suffix(".json").write_text(
        json.dumps({"labels": ["maize", "manure"], "image_size": [8, 8]}),
        encoding="utf-8",
    )
    load_calls = []

    class FakeDataset:
        def __init__(self, records, _image_size, batch_size: int, **_kwargs) -> None:
            self.records = records
            self.batch_size = batch_size

        def __len__(self) -> int:
            return (len(self.records) + self.batch_size - 1) // self.batch_size

        def __getitem__(self, index: int) -> list[str]:
            batch = self.records[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            return [record.image_path.name for record in batch]

    class FakeModel:
        def __init__(self) -> None:
            self.batch_inputs = []
            self.full_predict_calls = 0
            self.fail_on_batch_call = None
            self.on_batch = None

        def predict_on_batch(self, batch: list[str]) -> list[list[float]]:
            self.batch_inputs.append(batch)
            if self.on_batch is not None:
                self.on_batch(len(self.batch_inputs))
            if self.fail_on_batch_call == len(self.batch_inputs):
                raise RuntimeError("simulated batch failure")
            return [probabilities[image_name] for image_name in batch]

        def predict(self, dataset, *, verbose: int, callbacks):
            self.full_predict_calls += 1
            assert verbose == 0
            for batch_index in range(len(dataset)):
                callbacks[0].on_predict_batch_end(batch_index)
            callbacks[0].on_predict_end()
            return [probabilities[record.image_path.name] for record in dataset.records]

    model = FakeModel()

    def load_model(path: Path):
        load_calls.append(path)
        return model

    def make_lambda_callback(**kwargs):
        return SimpleNamespace(**kwargs)

    keras = SimpleNamespace(
        models=SimpleNamespace(load_model=load_model),
        callbacks=SimpleNamespace(LambdaCallback=make_lambda_callback),
    )
    monkeypatch.setattr(classifier, "_keras", lambda: keras)
    monkeypatch.setattr(
        classifier, "_make_image_sequence_class", lambda _keras: FakeDataset
    )
    return model_path, model, load_calls


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


def test_log_class_counts_includes_each_class_and_unlabeled_records(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    classifier._log_class_counts(
        "Training",
        [
            ImageRecord(tmp_path / "one.jpg", ("maize",)),
            ImageRecord(tmp_path / "two.jpg", ("maize", "manure")),
            ImageRecord(tmp_path / "three.jpg", ()),
        ],
        ["maize", "manure"],
    )

    assert (
        "Training dataset example counts: {'no classes': 1, 'maize': 2, 'manure': 1}"
        in caplog.text
    )


def test_predict_images_without_output_path_keeps_existing_return_behavior(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_progress,
) -> None:
    probabilities = {
        "first.jpg": [0.9, 0.1],
        "second.jpg": [0.2, 0.8],
    }
    model_path, model, load_calls = _configure_fake_predictor(
        monkeypatch, tmp_path, probabilities
    )
    image_paths = [tmp_path / "first.jpg", tmp_path / "second.jpg"]

    predictions = classifier.predict_images(
        model_path, image_paths, batch_size=1, workers=0
    )

    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.2, "manure": 0.8},
    ]
    assert model.full_predict_calls == 0
    assert model.batch_inputs == [["first.jpg"], ["second.jpg"]]
    assert load_calls == [model_path]
    assert fake_progress[0].n == 2
    assert fake_progress[0].closed


def test_predict_images_writes_csv_and_reuses_existing_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_progress,
) -> None:
    probabilities = {
        "first.jpg": [0.9, 0.1],
        "second.jpg": [0.2, 0.8],
    }
    model_path, model, load_calls = _configure_fake_predictor(
        monkeypatch, tmp_path, probabilities
    )
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    output_dir = tmp_path / "results"
    prediction_path = output_dir / "predictions.csv"

    predictions = classifier.predict_images(
        model_path,
        [first_image, second_image, first_image],
        batch_size=1,
        workers=2,
        output_dir=output_dir,
    )

    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.2, "manure": 0.8},
        {"maize": 0.9, "manure": 0.1},
    ]
    assert model.batch_inputs == [["first.jpg"], ["second.jpg"]]
    assert fake_progress[0].n == 2
    assert fake_progress[0].closed
    with prediction_path.open(encoding="utf-8", newline="") as output_file:
        reader = csv.DictReader(output_file)
        assert reader.fieldnames == [
            "image_path",
            "maize_probability",
            "manure_probability",
        ]
        rows = list(reader)
    assert [row["image_path"] for row in rows] == [
        str(first_image.resolve()),
        str(second_image.resolve()),
    ]
    assert [(row["maize_probability"], row["manure_probability"]) for row in rows] == [
        ("0.9", "0.1"),
        ("0.2", "0.8"),
    ]

    resumed_predictions = classifier.predict_images(
        model_path,
        [second_image, first_image],
        output_dir=output_dir,
        workers=0,
    )
    assert resumed_predictions == [predictions[1], predictions[0]]
    assert len(model.batch_inputs) == 2
    assert load_calls == [model_path]


def test_predict_images_resumes_after_a_later_batch_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_progress,
) -> None:
    probabilities = {
        "first.jpg": [0.9, 0.1],
        "second.jpg": [0.2, 0.8],
        "third.jpg": [0.3, 0.7],
    }
    model_path, model, _load_calls = _configure_fake_predictor(
        monkeypatch, tmp_path, probabilities
    )
    image_paths = [tmp_path / name for name in probabilities]
    output_dir = tmp_path / "predictions"
    prediction_path = output_dir / "predictions.csv"
    model.fail_on_batch_call = 2

    with pytest.raises(RuntimeError, match="simulated batch failure"):
        classifier.predict_images(
            model_path,
            image_paths,
            batch_size=1,
            workers=0,
            output_dir=output_dir,
        )

    with prediction_path.open(encoding="utf-8", newline="") as output_file:
        saved_rows = list(csv.DictReader(output_file))
    assert [row["image_path"] for row in saved_rows] == [str(image_paths[0].resolve())]

    model.fail_on_batch_call = None
    predictions = classifier.predict_images(
        model_path,
        image_paths,
        batch_size=1,
        workers=0,
        output_dir=output_dir,
    )

    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.2, "manure": 0.8},
        {"maize": 0.3, "manure": 0.7},
    ]
    assert model.batch_inputs == [
        ["first.jpg"],
        ["second.jpg"],
        ["second.jpg"],
        ["third.jpg"],
    ]
    assert all(progress.closed for progress in fake_progress)


@pytest.mark.parametrize(
    ("contents", "error_match"),
    [
        ("image_path,other_probability\nphoto.jpg,0.5\n", "columns"),
        (
            "image_path,maize_probability,manure_probability\nphoto.jpg,0.5\n",
            "Malformed",
        ),
        (
            (
                "image_path,maize_probability,manure_probability\n"
                "photo.jpg,0.5,0.5\nphoto.jpg,0.2,0.8\n"
            ),
            "duplicate image path",
        ),
    ],
)
def test_predict_images_rejects_invalid_existing_csv(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    contents: str,
    error_match: str,
) -> None:
    model_path, _model, load_calls = _configure_fake_predictor(
        monkeypatch,
        tmp_path,
        {"photo.jpg": [0.5, 0.5]},
    )
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    prediction_path = output_dir / "predictions.csv"
    prediction_path.write_text(contents, encoding="utf-8")
    original_contents = prediction_path.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match=error_match):
        classifier.predict_images(
            model_path,
            [tmp_path / "photo.jpg"],
            output_dir=output_dir,
            workers=0,
        )

    assert prediction_path.read_text(encoding="utf-8") == original_contents
    assert load_calls == []


def test_predict_images_empty_input_writes_header_without_loading_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path, _model, load_calls = _configure_fake_predictor(
        monkeypatch, tmp_path, {}
    )
    output_dir = tmp_path / "predictions"

    assert (
        classifier.predict_images(model_path, [], output_dir=output_dir, workers=0)
        == []
    )

    assert (output_dir / "predictions.csv").read_text(encoding="utf-8") == (
        "image_path,maize_probability,manure_probability\n"
    )
    assert load_calls == []


def test_prediction_progress_callback_updates_and_closes_tqdm(
    fake_progress,
) -> None:
    keras = SimpleNamespace(
        callbacks=SimpleNamespace(LambdaCallback=_make_lambda_callback)
    )

    callback = classifier._prediction_progress_callback(keras, 100, 10)
    for batch in range(10):
        callback.on_predict_batch_end(batch)
    callback.on_predict_end()

    assert fake_progress[0].total == 100
    assert fake_progress[0].n == 100
    assert fake_progress[0].closed


def test_predict_images_writes_prediction_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_progress,
) -> None:
    first_image = tmp_path / "first.jpg"
    second_image = tmp_path / "second.jpg"
    first_image.write_bytes(b"first")
    second_image.write_bytes(b"second")
    model_path, model, _load_calls = _configure_fake_predictor(
        monkeypatch,
        tmp_path,
        {"first.jpg": [0.9, 0.1], "second.jpg": [0.8, 0.2]},
    )
    records = [
        ImageRecord(first_image, ("maize",)),
        ImageRecord(second_image, ("manure",)),
    ]

    output_dir = tmp_path / "report"
    predictions = classifier.predict_images(
        model_path,
        records,
        batch_size=1,
        workers=0,
        output_dir=output_dir,
        thresholds=(0.5, 0.7),
    )

    prediction_path = output_dir / "predictions.csv"
    assert prediction_path.is_file()
    report_dir = output_dir / "threshold-0.5"
    with (report_dir / "prediction-results.csv").open(
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
    assert not (report_dir / "first.jpg").exists()
    assert (report_dir / "second.jpg").exists()
    assert (report_dir / "second.json").exists()
    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.8, "manure": 0.2},
    ]
    assert model.full_predict_calls == 0
    assert model.batch_inputs == [["first.jpg"], ["second.jpg"]]
    assert fake_progress[0].n == 2
    assert fake_progress[0].closed
    assert (output_dir / "threshold-0.7" / "prediction-statistics.json").exists()
    statistics = json.loads(
        (report_dir / "prediction-statistics.json").read_text(encoding="utf-8")
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
        "predict_images",
        lambda reported_model_path, reported_records, **kwargs: reports.append(
            (
                reported_model_path,
                reported_records,
                kwargs["output_dir"],
                kwargs["batch_size"],
                kwargs["workers"],
                kwargs["thresholds"],
            )
        ),
    )

    returned_model = classifier.train_model(
        [training_record],
        ["maize"],
        model_path,
        validation_images=[validation_record],
        test_images=[test_record],
        evaluation_dir=tmp_path / "reports",
        thresholds=(0.5, 0.7),
    )

    assert returned_model is model
    assert reports == [
        (
            model_path,
            [training_record],
            tmp_path / "reports" / "train_wrong",
            32,
            4,
            (0.5, 0.7),
        ),
        (
            model_path,
            [validation_record],
            tmp_path / "reports" / "validation_wrong",
            32,
            4,
            (0.5, 0.7),
        ),
        (
            model_path,
            [test_record],
            tmp_path / "reports" / "test_wrong",
            32,
            4,
            (0.5, 0.7),
        ),
    ]
