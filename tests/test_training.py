from pathlib import Path
from types import SimpleNamespace

import pytest

from phototagger.custom import training as module
from phototagger.custom.dataset import ImageRecord


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

    actual_trainable_count = module._unfreeze_top_backbone_layers(
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

    module._unfreeze_top_backbone_layers(feature_extractor)

    assert sum(layer.trainable for layer in feature_extractor.layers) == 6


@pytest.mark.parametrize("layer_count", [-0.1, 1.1])
def test_unfreeze_top_backbone_layers_rejects_invalid_fraction(
    layer_count: float,
) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        module._unfreeze_top_backbone_layers(SimpleNamespace(layers=[]), layer_count)


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
        actual_trainable_count = module._unfreeze_top_backbone_layers(
            feature_extractor, layer_count=3
        )
    else:
        actual_trainable_count = module._unfreeze_top_backbone_layers(
            feature_extractor, layer_count=3, expand_to_blocks=False
        )

    assert [layer.trainable for layer in feature_extractor.layers] == expected_trainable
    assert actual_trainable_count == sum(expected_trainable)
    if expand_to_blocks:
        assert "Unfroze 4 of 7 backbone layers" in caplog.text
    else:
        assert "Unfroze" not in caplog.text


def test_compile_model_appends_additional_metrics() -> None:
    compile_options = {}

    class FakeMetric:
        def __init__(self, name: str, **kwargs) -> None:
            self.name = name
            self.kwargs = kwargs

    class FakeModel:
        def compile(self, **kwargs) -> None:
            compile_options.update(kwargs)

    keras = SimpleNamespace(
        optimizers=SimpleNamespace(Adam=lambda **_kwargs: "adam"),
        losses=SimpleNamespace(BinaryFocalCrossentropy=lambda: "loss"),
        metrics=SimpleNamespace(
            BinaryAccuracy=FakeMetric,
            Precision=FakeMetric,
            Recall=FakeMetric,
        ),
    )

    custom_metric = object()
    module._compile_model(keras, FakeModel(), additional_metrics=[custom_metric])

    metrics = compile_options["metrics"]
    assert [metric.name for metric in metrics[:3]] == [
        "binary_accuracy",
        "precision",
        "recall",
    ]
    assert metrics[3] is custom_metric


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
            self.layers: list[object] = []

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
    monkeypatch.setattr(module, "_keras", lambda: keras)
    compile_options = {}

    def fake_compile(_keras, _model, *, learning_rate, additional_metrics=None):
        compile_options["learning_rate"] = learning_rate
        compile_options["additional_metrics"] = additional_metrics

    monkeypatch.setattr(module, "_compile_model", fake_compile)

    model = module.build_model(
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
    custom_metric = object()
    fit_kwargs = {}

    def fake_fit_model(*_args, **kwargs):
        fit_kwargs.update(kwargs)
        return model

    monkeypatch.setattr(module, "_fit_model", fake_fit_model)

    returned_model = module.train_model(
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
        additional_metrics=[custom_metric],
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
    assert fit_kwargs["additional_metrics"] == [custom_metric]


def test_reduce_lr_callback_uses_configured_plateau_settings() -> None:
    callback_options = {}

    def make_reduce_lr_on_plateau(**kwargs):
        callback_options.update(kwargs)
        return object()

    keras = SimpleNamespace(
        callbacks=SimpleNamespace(ReduceLROnPlateau=make_reduce_lr_on_plateau)
    )

    module._make_reduce_lr_callback(
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
    callback = module._make_best_model_metrics_callback(keras, checkpoint)

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


def test_class_weights_balance_positive_and_negative_examples() -> None:
    weights = module.calculate_class_weights(
        [[1, 0], [1, 0], [0, 1], [0, 0]], ["common", "rare"]
    )

    assert weights["common"] == {"positive": 1.0, "negative": 1.0}
    assert weights["rare"] == {"positive": 2.0, "negative": 0.6666666666666666}


def test_log_class_counts_includes_each_class_and_unlabeled_records(
    caplog: pytest.LogCaptureFixture, tmp_path: Path
) -> None:
    module._log_class_counts(
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


def test_existing_model_skips_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_path = tmp_path / "model.keras"
    model_path.write_bytes(b"model")
    training_record = ImageRecord(tmp_path / "training.jpg", ("maize",))
    validation_record = ImageRecord(tmp_path / "validation.jpg", ("maize",))
    model = object()

    monkeypatch.setattr(
        module,
        "_keras",
        lambda: SimpleNamespace(models=SimpleNamespace(load_model=lambda _path: model)),
    )
    monkeypatch.setattr(
        module,
        "_fit_model",
        lambda *_args, **_kwargs: pytest.fail("Existing models must not be trained"),
    )

    returned_model = module.train_model(
        [training_record],
        ["maize"],
        model_path,
        validation_images=[validation_record],
    )

    assert returned_model is model
