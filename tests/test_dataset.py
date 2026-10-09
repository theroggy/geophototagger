import csv
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from geophototagger.custom import classifier
from geophototagger.custom.classifier import calculate_class_weights
from geophototagger.custom.dataset import (
    ImageRecord,
    determine_classes,
    discover_records,
    discover_records_csv,
    encode_labels,
    read_manifest,
    write_manifest,
)


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


def test_load_image_batch_crops_to_aspect_ratio_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_options = []

    def load_img(
        _path,
        _target_size,
        *,
        crop_to_aspect_ratio,
        crop_window_scale,
    ):
        load_options.append(
            {
                "crop_to_aspect_ratio": crop_to_aspect_ratio,
                "crop_window_scale": crop_window_scale,
            }
        )
        return "image"

    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=lambda _image: [1]))
    monkeypatch.setattr(classifier, "_keras", lambda: keras)
    monkeypatch.setattr(classifier, "_load_img", load_img)
    record = ImageRecord(Path("image.jpg"), ())

    classifier._load_image_batch([record], (8, 8))
    classifier._load_image_batch([record], (8, 8), crop_to_aspect_ratio=False)

    assert load_options == [
        {"crop_to_aspect_ratio": True, "crop_window_scale": 1.0},
        {"crop_to_aspect_ratio": False, "crop_window_scale": 1.0},
    ]


def test_load_image_batch_reemits_warnings_with_filename(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    record = ImageRecord(Path("oversized.jpg"), ())

    def load_img(
        _path,
        _target_size,
        *,
        crop_to_aspect_ratio,
        crop_window_scale,
    ):
        del crop_to_aspect_ratio
        del crop_window_scale
        warnings.warn("oversized image", Image.DecompressionBombWarning, stacklevel=2)
        warnings.warn("unrelated image warning", UserWarning, stacklevel=2)
        return "image"

    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=lambda _image: [1]))
    monkeypatch.setattr(classifier, "_keras", lambda: keras)
    monkeypatch.setattr(classifier, "_load_img", load_img)

    with pytest.warns(Warning) as caught_warnings:
        classifier._load_image_batch([record], (8, 8))

    assert [warning.category for warning in caught_warnings] == [
        Image.DecompressionBombWarning,
        UserWarning,
    ]
    assert [str(warning.message) for warning in caught_warnings] == [
        "oversized.jpg: oversized image",
        "oversized.jpg: unrelated image warning",
    ]


def test_load_image_batch_includes_path_in_image_load_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = Path("corrupt.jpg")
    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=np.asarray))
    monkeypatch.setattr(classifier, "_keras", lambda: keras)

    def load_img(*_args, **_kwargs):
        raise OSError("image file is truncated (19 bytes not processed)")

    monkeypatch.setattr(classifier, "_load_img", load_img)

    with pytest.raises(
        OSError,
        match=r"Failed to load image corrupt\.jpg: image file is truncated",
    ):
        classifier._load_image_batch([ImageRecord(image_path, ())], (8, 8))


def test_load_image_batch_applies_additional_center_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "gradient.png"
    source = np.zeros((120, 200, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(200, dtype=np.uint8)
    Image.fromarray(source).save(image_path)
    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=np.asarray))
    monkeypatch.setattr(classifier, "_keras", lambda: keras)
    record = ImageRecord(image_path, ())

    standard_crop = classifier._load_image_batch([record], (40, 40))
    tighter_crop = classifier._load_image_batch(
        [record], (40, 40), crop_window_scale=0.75
    )

    assert standard_crop.shape == tighter_crop.shape == (1, 40, 40, 3)
    assert standard_crop[0, 0, 0, 0] == 41
    assert tighter_crop[0, 0, 0, 0] == 56


def test_load_image_batch_skips_additional_crop_below_output_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (60, 60)).save(image_path)
    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=np.asarray))
    monkeypatch.setattr(classifier, "_keras", lambda: keras)

    images = classifier._load_image_batch(
        [ImageRecord(image_path, ())],
        (40, 40),
        crop_window_scale=0.5,
    )

    assert images.shape == (1, 40, 40, 3)


@pytest.mark.parametrize("crop_scale", [0, -0.1, 1.1, float("inf"), float("nan")])
def test_load_image_batch_rejects_invalid_crop_scale(crop_scale: float) -> None:
    with pytest.raises(ValueError, match="crop_window_scale"):
        classifier._load_image_batch([], (40, 40), crop_window_scale=crop_scale)


@pytest.mark.parametrize(
    ("layer_count", "expected_trainable_count"),
    [(0, 0), (20, 20), (80, 60), (0.1, 6), (1.0, 60)],
)
def test_unfreeze_top_backbone_layers_uses_configured_count(
    layer_count: int | float, expected_trainable_count: int
) -> None:
    class FakeLayer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.trainable = False

    class FakeFeatureExtractor:
        def __init__(self, layer_count: int) -> None:
            self.layers = [FakeLayer(f"layer_{index}") for index in range(layer_count)]
            self._trainable = False

        @property
        def trainable(self) -> bool:
            return self._trainable

        @trainable.setter
        def trainable(self, value: bool) -> None:
            self._trainable = value
            for layer in self.layers:
                layer.trainable = value

    feature_extractor = FakeFeatureExtractor(60)

    actual_trainable_count = classifier._unfreeze_top_backbone_layers(
        feature_extractor, layer_count=layer_count
    )

    assert feature_extractor.trainable is bool(layer_count)
    assert actual_trainable_count == expected_trainable_count
    assert sum(layer.trainable for layer in feature_extractor.layers) == (
        expected_trainable_count
    )


def test_unfreeze_top_backbone_layers_defaults_to_ten_percent() -> None:
    feature_extractor = SimpleNamespace(
        layers=[
            SimpleNamespace(name=f"layer_{index}", trainable=True)
            for index in range(60)
        ],
        trainable=False,
    )

    classifier._unfreeze_top_backbone_layers(feature_extractor)

    assert sum(layer.trainable for layer in feature_extractor.layers) == 6


@pytest.mark.parametrize("layer_count", [-0.1, 1.1])
def test_unfreeze_top_backbone_layers_rejects_invalid_fraction(
    layer_count: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        classifier._unfreeze_top_backbone_layers(
            SimpleNamespace(layers=[]), layer_count
        )


@pytest.mark.parametrize(
    ("expand_to_blocks", "expected_trainable"),
    [
        (True, [False, False, False, True, True, True, True]),
        (False, [False, False, False, False, True, True, True]),
    ],
)
def test_unfreeze_top_backbone_layers_can_toggle_block_expansion(
    expand_to_blocks: bool,
    expected_trainable: list[bool],
    caplog: pytest.LogCaptureFixture,
) -> None:
    class FakeLayer:
        def __init__(self, name: str) -> None:
            self.name = name
            self.trainable = False

    class FakeFeatureExtractor:
        def __init__(self, names: list[str]) -> None:
            self.layers = [FakeLayer(name) for name in names]
            self._trainable = False

        @property
        def trainable(self) -> bool:
            return self._trainable

        @trainable.setter
        def trainable(self, value: bool) -> None:
            self._trainable = value
            for layer in self.layers:
                layer.trainable = value

    feature_extractor = FakeFeatureExtractor(
        [
            "stem_conv",
            "block6l_expand_conv",
            "block6l_add",
            "block6m_expand_conv",
            "block6m_project_conv",
            "block6m_add",
            "top_conv",
        ]
    )

    caplog.set_level("INFO")
    if expand_to_blocks:
        actual_trainable_count = classifier._unfreeze_top_backbone_layers(
            feature_extractor, layer_count=3
        )
    else:
        actual_trainable_count = classifier._unfreeze_top_backbone_layers(
            feature_extractor, layer_count=3, expand_to_blocks=False
        )

    assert [layer.trainable for layer in feature_extractor.layers] == expected_trainable
    assert actual_trainable_count == sum(expected_trainable)
    if expand_to_blocks:
        assert "Unfroze 4 of 7 backbone layers" in caplog.text
    else:
        assert "Unfroze" not in caplog.text


@pytest.mark.parametrize(
    ("head_dropout", "head_l2", "expected_regularizer"),
    [(0.5, 0.005, ("l2", 0.005)), (0.0, 0.0, None)],
)
def test_build_model_configures_head_regularization(
    monkeypatch: pytest.MonkeyPatch,
    head_dropout: float,
    head_l2: float,
    expected_regularizer: tuple[str, float] | None,
) -> None:
    layer_configuration = {}

    class FakeLayer:
        def __init__(self, kind: str, **kwargs) -> None:
            self.kind = kind
            self.kwargs = kwargs

        def __call__(self, inputs):
            return self.kind, inputs

    class FakeBackbone:
        def __init__(self) -> None:
            self.layers = []

        def __call__(self, _inputs, *, training: bool):
            assert training is False
            return "features"

    def make_dense(class_count: int, activation: str, kernel_regularizer):
        layer_configuration["dense"] = {
            "class_count": class_count,
            "activation": activation,
            "kernel_regularizer": kernel_regularizer,
        }
        return FakeLayer("dense")

    layers = SimpleNamespace(
        Dense=make_dense,
        Dropout=lambda rate, name: FakeLayer("dropout", rate=rate, name=name),
    )
    keras = SimpleNamespace(
        applications=SimpleNamespace(FakeBackbone=lambda **_kwargs: FakeBackbone()),
        Input=lambda **_kwargs: "input",
        layers=layers,
        Model=lambda inputs, outputs: SimpleNamespace(inputs=inputs, outputs=outputs),
        regularizers=SimpleNamespace(L2=lambda value: ("l2", value)),
    )
    monkeypatch.setattr(classifier, "_keras", lambda: keras)
    compile_options = {}

    def fake_compile(_keras, _model, *, learning_rate):
        compile_options["learning_rate"] = learning_rate

    monkeypatch.setattr(classifier, "_compile_model", fake_compile)

    model = classifier.build_model(
        2,
        backbone="FakeBackbone",
        weights=None,
        augment=False,
        head_dropout=head_dropout,
        head_l2=head_l2,
    )

    assert layer_configuration["dense"]["class_count"] == 2
    assert layer_configuration["dense"]["activation"] == "sigmoid"
    assert layer_configuration["dense"]["kernel_regularizer"] == expected_regularizer
    assert compile_options["learning_rate"] == 1e-3
    if head_dropout:
        assert model.outputs[0] == "dense"
        assert model.outputs[1][0] == "dropout"
    else:
        assert model.outputs == ("dense", "features")


@pytest.mark.parametrize("layer_count", [24, 0.1])
def test_train_model_forwards_finetuning_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    layer_count: int | float,
) -> None:
    record = ImageRecord(tmp_path / "training.jpg", ("maize",))
    model = object()
    fit_kwargs = {}

    def fake_fit_model(*_args, **kwargs):
        fit_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(classifier, "_fit_model", fake_fit_model)

    returned_model = classifier.train_model(
        [record],
        ["maize"],
        tmp_path / "model.keras",
        unfrozen_backbone_layers=layer_count,
        crop_to_aspect_ratio=False,
        crop_window_scale=0.75,
        learning_rate=5e-4,
        head_dropout=0.35,
        head_l2=0.002,
        reduce_lr_patience=4,
        reduce_lr_factor=0.3,
        min_learning_rate=1e-7,
    )

    assert returned_model is model
    assert fit_kwargs["unfrozen_backbone_layers"] == layer_count
    assert fit_kwargs["crop_to_aspect_ratio"] is False
    assert fit_kwargs["crop_window_scale"] == 0.75
    assert fit_kwargs["learning_rate"] == 5e-4
    assert fit_kwargs["head_dropout"] == 0.35
    assert fit_kwargs["head_l2"] == 0.002
    assert fit_kwargs["reduce_lr_patience"] == 4
    assert fit_kwargs["reduce_lr_factor"] == 0.3
    assert fit_kwargs["min_learning_rate"] == 1e-7


@pytest.mark.parametrize(
    "crop_settings",
    [
        ({"crop_to_aspect_ratio": True, "crop_window_scale": 0.75}, True, 0.75),
        (
            {"crop_to_aspect_ratio": True, "aspect_ratio_crop_percent": 25},
            True,
            0.75,
        ),
        ({}, False, 1.0),
    ],
)
def test_predict_image_uses_crop_setting_from_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crop_settings: tuple[dict[str, bool | float | int], bool, float],
) -> None:
    stored_crop_settings, expected_crop_setting, expected_crop_scale = crop_settings
    model_path = tmp_path / "model.keras"
    metadata = {"labels": ["maize"], "image_size": [8, 8]}
    metadata.update(stored_crop_settings)
    model_path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
    model = SimpleNamespace(predict=lambda _images, **_kwargs: [[0.8]])
    keras = SimpleNamespace(models=SimpleNamespace(load_model=lambda _path: model))
    prediction_options = {}
    monkeypatch.setattr(classifier, "_keras", lambda: keras)

    def load_images(
        _records,
        image_size,
        *,
        crop_to_aspect_ratio,
        crop_window_scale,
    ):
        prediction_options["image_size"] = image_size
        prediction_options["crop_to_aspect_ratio"] = crop_to_aspect_ratio
        prediction_options["crop_window_scale"] = crop_window_scale
        return "image batch"

    monkeypatch.setattr(classifier, "_load_images", load_images)

    predictions = classifier.predict_image(model_path, tmp_path / "photo.jpg")

    assert predictions == {"maize": 0.8}
    assert prediction_options == {
        "image_size": (8, 8),
        "crop_to_aspect_ratio": expected_crop_setting,
        "crop_window_scale": expected_crop_scale,
    }


def test_reduce_lr_callback_uses_configured_plateau_settings() -> None:
    callback_options = {}

    def make_reduce_lr_on_plateau(**kwargs):
        callback_options.update(kwargs)
        return object()

    keras = SimpleNamespace(
        callbacks=SimpleNamespace(ReduceLROnPlateau=make_reduce_lr_on_plateau)
    )

    classifier._make_reduce_lr_callback(
        keras,
        monitor="val_loss",
        factor=0.2,
        patience=3,
        min_learning_rate=1e-6,
    )

    assert callback_options == {
        "monitor": "val_loss",
        "factor": 0.2,
        "patience": 3,
        "min_lr": 1e-6,
        "verbose": 1,
    }


def test_best_model_metrics_follow_checkpoint_improvements() -> None:
    class FakeCallback:
        def __init__(self) -> None:
            pass

    checkpoint = SimpleNamespace(best=1.0)
    keras = SimpleNamespace(callbacks=SimpleNamespace(Callback=FakeCallback))
    callback = classifier._make_best_model_metrics_callback(keras, checkpoint)

    callback.on_epoch_end(0, {"loss": 1.2, "val_loss": 1.1})
    assert callback.metrics is None

    checkpoint.best = 0.9
    callback.on_epoch_end(1, {"loss": 1.0, "val_loss": 0.9, "val_binary_accuracy": 0.8})
    assert callback.epoch == 2
    assert callback.metrics == {
        "loss": 1.0,
        "val_loss": 0.9,
        "val_binary_accuracy": 0.8,
    }

    checkpoint.best = 0.8
    callback.on_epoch_end(2, {"loss": 0.7, "val_loss": 0.8})
    assert callback.epoch == 3
    assert callback.metrics == {"loss": 0.7, "val_loss": 0.8}


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

    predictions = classifier.predict_images(
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

    predictions = classifier.predict_images(
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

    resumed_predictions = classifier.predict_images(
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
        classifier.predict_images(
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
    predictions = classifier.predict_images(
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
        classifier.predict_images(
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
        classifier.predict_images(
            model_path, [], output_path=prediction_path, workers=0
        )
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
    prediction_path = output_dir / "predictions.csv"
    predictions = classifier.predict_images(
        model_path,
        records,
        batch_size=1,
        workers=0,
        output_path=prediction_path,
    )

    assert prediction_path.is_file()
    assert not (output_dir / "threshold-0.5").exists()
    classifier.write_prediction_evaluation(
        records,
        ["maize", "manure"],
        prediction_path,
        output_dir,
        thresholds=(0.5, 0.7),
    )
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
    assert (output_dir / "threshold-0.5_correctly_classified" / "first.jpg").exists()
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


def test_write_prediction_report_updates_progress(
    tmp_path: Path, fake_progress
) -> None:
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"image")
    record = ImageRecord(image_path, ("maize",))

    classifier.write_prediction_evaluation_info(
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
    assert not classifier.prediction_evaluation_is_complete(output_dir, (0.5,))

    classifier.write_prediction_evaluation(
        [record], ["maize"], predictions_path, output_dir, thresholds=(0.5,)
    )
    assert classifier.prediction_evaluation_is_complete(output_dir, (0.5,))
    assert not classifier.prediction_evaluation_is_complete(output_dir, (0.5, 0.7))

    with (output_dir / "threshold-0.5" / "prediction-results.csv").open(
        encoding="utf-8", newline=""
    ) as report_file:
        rows = list(csv.DictReader(report_file))
    assert rows[0]["predicted_labels"] == "maize"
    assert rows[0]["correctly_classified"] == "True"


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
        evaluation_thresholds=(0.5, 0.7),
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
