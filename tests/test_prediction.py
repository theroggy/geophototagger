import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from phototagger.custom import prediction as module
from phototagger.custom.dataset import ImageRecord


@pytest.fixture
def fake_progress(monkeypatch: pytest.MonkeyPatch):
    progress_instances = []

    class FakeProgress:
        def __init__(
            self, iterable=None, *, total: int | None = None, **_kwargs
        ) -> None:
            self.iterable = iterable
            self.total = total if total is not None else len(iterable)
            self.n = 0
            self.closed = False
            progress_instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            self.close()

        def update(self, amount: int) -> None:
            self.n += amount

        def __iter__(self):
            try:
                for item in self.iterable:
                    self.update(1)
                    yield item
            finally:
                self.close()

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(module, "tqdm", FakeProgress)
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
        json.dumps(
            {
                "labels": ["maize", "manure"],
                "image_size": [8, 8],
                "crop_to_aspect_ratio": True,
            }
        ),
        encoding="utf-8",
    )
    load_calls = []

    class FakeDataset:
        def __init__(
            self,
            records,
            _image_size,
            batch_size: int,
            crop_to_aspect_ratio: bool,
            **_kwargs,
        ) -> None:
            assert crop_to_aspect_ratio
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
            self.batch_inputs: list[list[str]] = []
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
    monkeypatch.setattr(module, "_keras", lambda: keras)
    monkeypatch.setattr(
        module, "_make_image_sequence_class", lambda _keras: FakeDataset
    )
    return model_path, model, load_calls


@pytest.mark.parametrize(
    "crop_settings",
    [
        (
            {
                "crop_to_aspect_ratio": True,
                "crop_window_scale": 0.75,
                "interpolation": "lanczos",
            },
            True,
            0.75,
            "lanczos",
        ),
        (
            {"crop_to_aspect_ratio": True, "aspect_ratio_crop_percent": 25},
            True,
            0.75,
            "nearest",
        ),
        ({}, False, 1.0, "nearest"),
    ],
)
def test_predict_image_uses_crop_setting_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crop_settings: tuple[dict[str, bool | float | int | str], bool, float, str],
) -> None:
    (
        stored_crop_settings,
        expected_crop_setting,
        expected_crop_scale,
        expected_interpolation,
    ) = crop_settings
    model_path = tmp_path / "model.keras"
    metadata = {"labels": ["maize"], "image_size": [8, 8]}
    metadata.update(stored_crop_settings)
    model_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    model = SimpleNamespace(predict=lambda _images, **_kwargs: [[0.8]])
    keras = SimpleNamespace(models=SimpleNamespace(load_model=lambda _path: model))
    prediction_options = {}
    monkeypatch.setattr(module, "_keras", lambda: keras)

    def load_images(
        _records,
        image_size,
        *,
        crop_to_aspect_ratio,
        crop_window_scale,
        interpolation,
    ):
        prediction_options["image_size"] = image_size
        prediction_options["crop_to_aspect_ratio"] = crop_to_aspect_ratio
        prediction_options["crop_window_scale"] = crop_window_scale
        prediction_options["interpolation"] = interpolation
        return "image batch"

    monkeypatch.setattr(module, "_load_images", load_images)

    predictions = module.predict_image(model_path, tmp_path / "photo.jpg")

    assert predictions == {"maize": 0.8}
    assert prediction_options == {
        "image_size": (8, 8),
        "crop_to_aspect_ratio": expected_crop_setting,
        "crop_window_scale": expected_crop_scale,
        "interpolation": expected_interpolation,
    }


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

    predictions = module.predict_images(
        model_path, image_paths, batch_size=1, workers=0
    )

    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.2, "manure": 0.8},
    ]
    assert model.full_predict_calls == 0
    assert model.batch_inputs == [["first.jpg"], ["second.jpg"]]
    assert load_calls == [model_path]
    assert all(progress.n == 2 for progress in fake_progress)
    assert all(progress.closed for progress in fake_progress)


def test_predict_images_reuses_complete_ready_prediction_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fake_progress,
) -> None:
    model_path, _model, load_calls = _configure_fake_predictor(
        monkeypatch, tmp_path, {"first.jpg": [0.9, 0.1]}
    )
    image_path = tmp_path / "first.jpg"
    output_dir = tmp_path / "results"
    output_dir.mkdir()
    (output_dir / "predictions.csv").write_text(
        "image_path,maize_probability,manure_probability\n"
        f"{image_path.resolve()},0.9,0.1\n",
        encoding="utf-8",
    )

    predictions = module.predict_images(
        model_path, [image_path], output_path=output_dir / "predictions.csv"
    )

    assert predictions == [{"maize": 0.9, "manure": 0.1}]
    assert load_calls == []
    assert fake_progress == []


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

    predictions = module.predict_images(
        model_path,
        [first_image, second_image, first_image],
        batch_size=1,
        workers=2,
        output_path=prediction_path,
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

    resumed_predictions = module.predict_images(
        model_path,
        [second_image, first_image],
        output_path=prediction_path,
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
    output_dir.mkdir()
    prediction_path.write_text(
        "image_path,maize_probability,manure_probability\n"
        f"{(tmp_path / 'cached.jpg').resolve()},0.4,0.6\n",
        encoding="utf-8",
    )
    original_prediction_csv = prediction_path.read_text(encoding="utf-8")
    model.fail_on_batch_call = 2

    with pytest.raises(RuntimeError, match="simulated batch failure"):
        module.predict_images(
            model_path,
            image_paths,
            batch_size=1,
            workers=0,
            output_path=prediction_path,
        )

    busy_path = output_dir / "predictions_busy.csv"
    assert prediction_path.read_text(encoding="utf-8") == original_prediction_csv
    assert busy_path.is_file()
    with busy_path.open(encoding="utf-8", newline="") as output_file:
        saved_rows = list(csv.DictReader(output_file))
    assert [row["image_path"] for row in saved_rows] == [
        str((tmp_path / "cached.jpg").resolve()),
        str(image_paths[0].resolve()),
    ]

    model.fail_on_batch_call = None
    predictions = module.predict_images(
        model_path,
        image_paths,
        batch_size=1,
        workers=0,
        output_path=prediction_path,
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
    assert prediction_path.is_file()
    assert not busy_path.exists()


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
        module.predict_images(
            model_path,
            [tmp_path / "photo.jpg"],
            output_path=prediction_path,
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
    prediction_path = output_dir / "predictions.csv"

    assert (
        module.predict_images(model_path, [], output_path=prediction_path, workers=0)
        == []
    )

    assert prediction_path.read_text(encoding="utf-8") == (
        "image_path,maize_probability,manure_probability\n"
    )
    assert load_calls == []


def test_prediction_progress_callback_updates_and_closes_tqdm(
    fake_progress,
) -> None:
    keras = SimpleNamespace(
        callbacks=SimpleNamespace(LambdaCallback=_make_lambda_callback)
    )

    callback = module._prediction_progress_callback(keras, 100, 10)
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
    prediction_path = output_dir / "predictions.csv"
    predictions = module.predict_images(
        model_path,
        records,
        batch_size=1,
        workers=0,
        output_path=prediction_path,
    )

    assert prediction_path.is_file()
    assert not (output_dir / "threshold-0.5").exists()
    module.write_prediction_evaluation(
        records,
        ["maize", "manure"],
        prediction_path,
        output_dir,
        thresholds=(0.5, 0.7),
    )
    with (output_dir / "prediction-report_0.5.csv").open(
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
    assert not (output_dir / "second.jpg").exists()
    assert not (output_dir / "second.json").exists()
    assert (output_dir / "threshold-0.5_ok" / "first.jpg").exists()
    assert (output_dir / "threshold-0.5_nok" / "second.jpg").exists()
    assert (output_dir / "threshold-0.5_nok" / "second.json").exists()
    assert predictions == [
        {"maize": 0.9, "manure": 0.1},
        {"maize": 0.8, "manure": 0.2},
    ]
    assert model.full_predict_calls == 0
    assert model.batch_inputs == [["first.jpg"], ["second.jpg"]]
    assert fake_progress[0].n == 2
    assert fake_progress[0].closed
    assert (output_dir / "prediction-report_0.7.csv").exists()
    assert (output_dir / "prediction-statistics_0.5.json").exists()
    assert (output_dir / "prediction-statistics_0.7.json").exists()
    statistics = json.loads(
        (output_dir / "prediction-statistics_0.5.json").read_text(encoding="utf-8")
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


def test_write_prediction_report_updates_progress(
    tmp_path: Path, fake_progress
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    record = ImageRecord(image_path, ("maize",))

    module._write_prediction_image_evaluation(
        [record],
        ["maize"],
        {str(image_path.resolve()): {"maize": 0.9}},
        threshold=0.5,
        output_dir=tmp_path / "report",
    )

    assert fake_progress[0].total == 1
    assert fake_progress[0].n == 1
    assert fake_progress[0].closed


def test_write_prediction_evaluation_reads_prediction_csv(tmp_path: Path) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    record = ImageRecord(image_path, ("maize",))
    predictions_path = tmp_path / "predictions.csv"
    predictions_path.write_text(
        f"image_path,maize_probability\n{image_path.resolve()},0.9\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "evaluation"
    assert not module.prediction_evaluation_is_complete(output_dir, (0.5,))

    module.write_prediction_evaluation(
        [record], ["maize"], predictions_path, output_dir, thresholds=(0.5,)
    )
    assert module.prediction_evaluation_is_complete(output_dir, (0.5,))
    assert not module.prediction_evaluation_is_complete(output_dir, (0.5, 0.7))

    with (output_dir / "prediction-report_0.5.csv").open(
        encoding="utf-8", newline=""
    ) as report_file:
        rows = list(csv.DictReader(report_file))
    assert rows[0]["predicted_labels"] == "maize"
    assert rows[0]["correctly_classified"] == "True"
