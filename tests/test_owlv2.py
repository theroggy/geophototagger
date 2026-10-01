import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from geophototagger import owlv2


def make_image(path: Path, value: int) -> Path:
    Image.new("RGB", (4, 4), (value, 0, 0)).save(path)
    return path


@pytest.fixture
def fake_detection(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class FakeBox(list):
        def tolist(self):
            return list(self)

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("processor", model_name))
            return cls()

        def __call__(self, text, images, return_tensors):
            assert return_tensors == "pt"
            return FakeInputs(text=text, images=images)

        def post_process_grounded_object_detection(
            self, outputs, target_sizes, threshold, text_labels
        ):
            predictions = []
            for image, labels in zip(outputs.images, text_labels, strict=True):
                value = image.getpixel((0, 0))[0]
                if value == 10:
                    predictions.append(
                        {
                            "boxes": [FakeBox([0.0, 0.0, 1.0, 1.0])],
                            "scores": [0.9],
                            "text_labels": [labels[0]],
                        }
                    )
                else:
                    predictions.append({"boxes": [], "scores": [], "text_labels": []})
            return predictions

    class FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

        def __call__(self, text, images):
            return SimpleNamespace(images=images)

        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("model", model_name))
            return cls()

    class FakeTensor(list):
        def to(self, device):
            return self

    torch = SimpleNamespace(
        tensor=lambda values: list(values),
        stack=FakeTensor,
        inference_mode=nullcontext,
        device=lambda name: name,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    transformers = SimpleNamespace(
        Owlv2Processor=FakeProcessor, Owlv2ForObjectDetection=FakeModel
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.mark.usefixtures("fake_detection")
def test_detects_labels_above_threshold(tmp_path: Path) -> None:
    match = make_image(tmp_path / "match.png", 10)
    no_match = make_image(tmp_path / "no_match.png", 20)
    results = owlv2.detect_objects(
        [match, no_match], labels=["car", "person"], threshold=0.5, batch_size=1
    )
    assert [path for path, _ in results] == [match, no_match]
    assert [detection.label for detection in results[0][1]] == ["car"]
    assert results[0][1][0].score == pytest.approx(0.9)
    assert results[0][1][0].box == (0.0, 0.0, 1.0, 1.0)
    assert results[1][1] == []


@pytest.mark.usefixtures("fake_detection")
def test_directory_and_single_file_input(tmp_path: Path) -> None:
    first = make_image(tmp_path / "a.png", 10)
    second = make_image(tmp_path / "b.png", 20)
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")
    results = owlv2.detect_objects(tmp_path, labels=["car"])
    assert [path for path, _ in results] == [first, second]
    assert owlv2.detect_objects(first, labels=["car"]) == [(first, results[0][1])]


@pytest.mark.parametrize(
    "options",
    [
        {"threshold": -0.1},
        {"threshold": 1.5},
        {"batch_size": 0},
    ],
)
def test_invalid_options_fail_before_download(options: dict) -> None:
    with pytest.raises(ValueError):
        owlv2.detect_objects([Path("image.png")], labels=["car"], **options)


def test_empty_labels_and_missing_input_fail_before_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No labels"):
        owlv2.detect_objects([tmp_path / "image.png"], labels=[])
    with pytest.raises(ValueError, match="No images"):
        owlv2.detect_objects([], labels=["car"])
    with pytest.raises(ValueError, match="does not exist"):
        owlv2.detect_objects([tmp_path / "missing.jpg"], labels=["car"])


@pytest.mark.usefixtures("fake_detection")
def test_output_dir_appends_csv_and_skips_known_images(tmp_path: Path) -> None:
    match = make_image(tmp_path / "match.png", 10)
    no_match = make_image(tmp_path / "no_match.png", 20)
    output_dir = tmp_path / "output"

    owlv2.detect_objects([match], labels=["car"], output_dir=output_dir)
    csv_path = output_dir / "detections.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["image_path"] == str(match)
    detections = json.loads(rows[0]["detections"])
    assert detections == [
        {"label": "car", "score": pytest.approx(0.9), "box": [0.0, 0.0, 1.0, 1.0]}
    ]

    results = owlv2.detect_objects(
        [match, no_match], labels=["car"], output_dir=output_dir
    )
    assert [path for path, _ in results] == [no_match]
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert {row["image_path"] for row in rows} == {str(match), str(no_match)}
