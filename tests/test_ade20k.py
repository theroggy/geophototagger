import csv
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from geophototagger import ade20k


def make_image(path: Path, value: int) -> Path:
    Image.new("RGB", (4, 4), (value, 0, 0)).save(path)
    return path


@pytest.fixture
def fake_segmentation(monkeypatch: pytest.MonkeyPatch):
    calls = []

    class FakeArray(list):
        def tolist(self):
            return list(self)

        def sum(self):
            return sum(self)

    class FakeInputs(dict):
        def to(self, device):
            return self

    class FakeConfig:
        id2label = {0: "sky", 1: "grass", 2: "wall"}

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, model_name, backend=None):
            calls.append(("processor", model_name))
            return cls()

        def __call__(self, images, return_tensors):
            assert return_tensors == "pt"
            return FakeInputs(images=images)

        def post_process_semantic_segmentation(self, outputs, target_sizes=None):
            return [image.getpixel((0, 0))[0] for image in outputs.images]

    class FakeModel:
        config = FakeConfig()

        def eval(self):
            return self

        def to(self, device):
            return self

        def half(self):
            return self

        def __call__(self, images):
            return SimpleNamespace(images=images)

        @classmethod
        def from_pretrained(cls, model_name):
            calls.append(("model", model_name))
            return cls()

    def fake_unique(segmentation, return_counts=False):
        assert return_counts
        if segmentation == 10:
            return FakeArray([0, 1]), FakeArray([80, 20])
        return FakeArray([2]), FakeArray([100])

    torch = SimpleNamespace(
        unique=fake_unique,
        inference_mode=nullcontext,
        device=lambda name: SimpleNamespace(type=name),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    transformers = SimpleNamespace(
        AutoImageProcessor=FakeProcessor,
        AutoModelForSemanticSegmentation=FakeModel,
    )
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return calls


@pytest.mark.usefixtures("fake_segmentation")
def test_segments_classes_above_min_fraction(tmp_path: Path) -> None:
    scene_a = make_image(tmp_path / "a.png", 10)
    scene_b = make_image(tmp_path / "b.png", 20)
    results = ade20k.segment_images([scene_a, scene_b], batch_size=1)
    assert [path for path, _ in results] == [scene_a, scene_b]
    assert [(cls.label, cls.fraction) for cls in results[0][1]] == [
        ("sky", 0.8),
        ("grass", 0.2),
    ]
    assert [(cls.label, cls.fraction) for cls in results[1][1]] == [("wall", 1.0)]


@pytest.mark.usefixtures("fake_segmentation")
def test_min_fraction_filters_small_classes(tmp_path: Path) -> None:
    scene_a = make_image(tmp_path / "a.png", 10)
    results = ade20k.segment_images([scene_a], min_fraction=0.5)
    assert [cls.label for cls in results[0][1]] == ["sky"]


@pytest.mark.usefixtures("fake_segmentation")
def test_directory_and_single_file_input(tmp_path: Path) -> None:
    first = make_image(tmp_path / "a.png", 10)
    second = make_image(tmp_path / "b.png", 20)
    (tmp_path / "ignore.txt").write_text("ignore", encoding="utf-8")
    results = ade20k.segment_images(tmp_path)
    assert [path for path, _ in results] == [first, second]
    assert ade20k.segment_images(first) == [(first, results[0][1])]


@pytest.mark.parametrize(
    "options",
    [
        {"min_fraction": -0.1},
        {"min_fraction": 1.5},
        {"batch_size": 0},
    ],
)
def test_invalid_options_fail_before_download(options: dict) -> None:
    with pytest.raises(ValueError):
        ade20k.segment_images([Path("image.png")], **options)


def test_empty_and_missing_input_fail_before_download(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No images"):
        ade20k.segment_images([])
    with pytest.raises(ValueError, match="does not exist"):
        ade20k.segment_images([tmp_path / "missing.jpg"])


@pytest.mark.usefixtures("fake_segmentation")
def test_output_dir_appends_csv_and_skips_known_images(tmp_path: Path) -> None:
    scene_a = make_image(tmp_path / "a.png", 10)
    scene_b = make_image(tmp_path / "b.png", 20)
    output_dir = tmp_path / "output"

    ade20k.segment_images([scene_a], output_dir=output_dir)
    csv_path = output_dir / "segments.csv"
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 1
    assert rows[0]["image_path"] == str(scene_a)
    assert json.loads(rows[0]["segments"]) == [
        {"label": "sky", "fraction": 0.8},
        {"label": "grass", "fraction": 0.2},
    ]

    results = ade20k.segment_images([scene_a, scene_b], output_dir=output_dir)
    assert [path for path, _ in results] == [scene_b]
    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert len(rows) == 2
    assert {row["image_path"] for row in rows} == {str(scene_a), str(scene_b)}
