import csv
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from geophototagger import imagenet21k
from scripts import run_imagenet_classify as script


@pytest.fixture
def fake_inference(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class FakeLabels:
        def __init__(self, subset: str) -> None:
            calls.append(subset)

        def num_classes(self) -> int:
            return 3

        def index_to_label_name(self, index: int) -> str:
            return f"n{index}"

        def index_to_description(self, index: int) -> str:
            return f"class {index}"

    class FakeProbabilities:
        def __init__(self, logits: list[list[float]]) -> None:
            self.logits = logits
            self.shape = (len(logits), 3)

        def softmax(self, dim: int):
            assert dim == 1
            self.logits = [
                [
                    math.exp(score) / sum(math.exp(value) for value in row)
                    for score in row
                ]
                for row in self.logits
            ]
            return self

        def cpu(self):
            return self

        def tolist(self) -> list[list[float]]:
            return self.logits

    class FakeModel:
        num_classes = 3

        def eval(self):
            return self

        def __call__(self, images: list[int]) -> FakeProbabilities:
            return FakeProbabilities(
                [
                    [0.0, 2.0, 1.0] if value == 10 else [3.0, 0.0, 0.0]
                    for value in images
                ]
            )

    def create_model(checkpoint: str, *, pretrained: bool):
        calls.append(checkpoint)
        assert pretrained
        return FakeModel()

    timm = SimpleNamespace(
        create_model=create_model,
        data=SimpleNamespace(
            ImageNetInfo=FakeLabels,
            resolve_model_data_config=lambda _model: {"size": 224},
            create_transform=lambda **_kwargs: lambda image: image.getpixel((0, 0))[0],
        ),
    )
    torch = SimpleNamespace(stack=lambda images: images, inference_mode=nullcontext)
    monkeypatch.setitem(sys.modules, "timm", timm)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


def make_image(path: Path, value: int) -> Path:
    Image.new("RGB", (2, 2), (value, 0, 0)).save(path)
    return path


@pytest.mark.parametrize(
    ("model_name", "checkpoint", "subset"),
    [
        ("efficientnetv2", "tf_efficientnetv2_s.in21k", "imagenet-21k-goog"),
        ("convnext", "convnext_tiny.fb_in22k", "imagenet-22k"),
    ],
)
def test_models_use_matching_labels_and_threshold(
    tmp_path: Path, fake_inference, model_name: str, checkpoint: str, subset: str
) -> None:
    first = make_image(tmp_path / "first.png", 10)
    second = make_image(tmp_path / "second.png", 20)
    results = imagenet21k.classify_images(
        [first, second], model_name=model_name, threshold=0.2, batch_size=1
    )
    assert fake_inference == [checkpoint, subset]
    assert [path for path, _ in results] == [first, second]
    assert [match.class_index for match in results[0][1]] == [1, 2]
    assert results[0][1][0].synset == "n1"
    assert results[0][1][0].label == "class 1"
    assert results[0][1][0].probability == pytest.approx(0.66524, abs=0.00001)
    assert [match.class_index for match in results[1][1]] == [0]


@pytest.mark.usefixtures("fake_inference")
def test_threshold_can_return_no_matches(tmp_path: Path) -> None:
    image_path = make_image(tmp_path / "image.png", 10)
    assert imagenet21k.classify_images([image_path], threshold=1) == [(image_path, [])]


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": -0.1},
        {"threshold": float("nan")},
        {"batch_size": 0},
        {"model_name": "unknown"},
    ],
)
def test_invalid_options_fail_before_download(options: dict) -> None:
    with pytest.raises(ValueError):
        imagenet21k.classify_images([Path("image.png")], **options)


def test_empty_and_missing_input_fail_before_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No images"):
        imagenet21k.classify_images([])
    with pytest.raises(ValueError, match="does not exist"):
        imagenet21k.classify_images([tmp_path / "missing.jpg"])


@pytest.mark.usefixtures("fake_inference")
def test_script_discovers_files_and_writes_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = make_image(tmp_path / "image.PNG", 10)
    (tmp_path / "other.txt").write_text("ignore", encoding="utf-8")
    assert script.find_images(tmp_path) == [image_path]
    assert script.find_images(image_path) == [image_path]
    output_path = tmp_path / "output" / "predictions.csv"
    monkeypatch.setattr(script, "INPUT_PATH", tmp_path)
    monkeypatch.setattr(script, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(script, "THRESHOLD", 0.2)
    script.main()
    with output_path.open(encoding="utf-8", newline="") as output_file:
        rows = list(csv.DictReader(output_file))
    assert len(rows) == 2
    assert rows[0] == {
        "image_path": str(image_path),
        "model": "efficientnetv2",
        "class_index": "1",
        "synset": "n1",
        "label": "class 1",
        "probability": str(rows[0]["probability"]),
    }
    assert float(rows[0]["probability"]) == pytest.approx(0.66524, abs=0.00001)


@pytest.mark.usefixtures("fake_inference")
def test_script_keeps_header_with_no_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    make_image(tmp_path / "image.png", 10)
    output_path = tmp_path / "empty.csv"
    monkeypatch.setattr(script, "INPUT_PATH", tmp_path)
    monkeypatch.setattr(script, "OUTPUT_PATH", output_path)
    monkeypatch.setattr(script, "THRESHOLD", 1.0)
    script.main()
    assert output_path.read_text(encoding="utf-8").strip() == (
        "image_path,model,class_index,synset,label,probability"
    )
