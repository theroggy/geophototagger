import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import numpy as np
import pytest
from PIL import Image

from phototagger.custom import image_processing as module
from phototagger.custom.dataset import ImageRecord


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
        interpolation,
    ):
        load_options.append(
            {
                "crop_to_aspect_ratio": crop_to_aspect_ratio,
                "crop_window_scale": crop_window_scale,
                "interpolation": interpolation,
            }
        )
        return "image"

    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=lambda _image: [1]))
    monkeypatch.setattr(module, "_keras", lambda: keras)
    monkeypatch.setattr(module, "_load_img", load_img)
    record = ImageRecord(Path("image.jpg"), ())

    module._load_image_batch([record], (8, 8))
    module._load_image_batch([record], (8, 8), crop_to_aspect_ratio=False)

    assert load_options == [
        {
            "crop_to_aspect_ratio": True,
            "crop_window_scale": 1.0,
            "interpolation": "bilinear",
        },
        {
            "crop_to_aspect_ratio": False,
            "crop_window_scale": 1.0,
            "interpolation": "bilinear",
        },
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
        interpolation,
    ):
        del crop_to_aspect_ratio
        del crop_window_scale
        del interpolation
        warnings.warn("oversized image", Image.DecompressionBombWarning, stacklevel=2)
        warnings.warn("unrelated image warning", UserWarning, stacklevel=2)
        return "image"

    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=lambda _image: [1]))
    monkeypatch.setattr(module, "_keras", lambda: keras)
    monkeypatch.setattr(module, "_load_img", load_img)

    with pytest.warns(Warning) as caught_warnings:
        module._load_image_batch([record], (8, 8))

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
    monkeypatch.setattr(module, "_keras", lambda: keras)

    def load_img(*_args, **_kwargs):
        raise OSError("image file is truncated (19 bytes not processed)")

    monkeypatch.setattr(module, "_load_img", load_img)

    with pytest.raises(
        OSError,
        match=r"Failed to load image corrupt\.jpg: image file is truncated",
    ):
        module._load_image_batch([ImageRecord(image_path, ())], (8, 8))


def test_load_image_batch_applies_additional_center_crop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "gradient.png"
    source = np.zeros((120, 200, 3), dtype=np.uint8)
    source[:, :, 0] = np.arange(200, dtype=np.uint8)
    Image.fromarray(source).save(image_path)
    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=np.asarray))
    monkeypatch.setattr(module, "_keras", lambda: keras)
    record = ImageRecord(image_path, ())

    standard_crop = module._load_image_batch([record], (40, 40))
    tighter_crop = module._load_image_batch([record], (40, 40), crop_window_scale=0.75)

    assert standard_crop.shape == tighter_crop.shape == (1, 40, 40, 3)
    assert standard_crop[0, 0, 0, 0] == 41
    assert tighter_crop[0, 0, 0, 0] == 56


def test_load_image_batch_skips_additional_crop_below_output_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image_path = tmp_path / "small.png"
    Image.new("RGB", (60, 60)).save(image_path)
    keras = SimpleNamespace(utils=SimpleNamespace(img_to_array=np.asarray))
    monkeypatch.setattr(module, "_keras", lambda: keras)

    images = module._load_image_batch(
        [ImageRecord(image_path, ())],
        (40, 40),
        crop_window_scale=0.5,
    )

    assert images.shape == (1, 40, 40, 3)


@pytest.mark.parametrize("crop_scale", [0, -0.1, 1.1, float("inf"), float("nan")])
def test_load_image_batch_rejects_invalid_crop_scale(crop_scale: float) -> None:
    with pytest.raises(ValueError, match="crop_window_scale"):
        module._load_image_batch([], (40, 40), crop_window_scale=crop_scale)


def test_image_sequence_forwards_interpolation_to_batch_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_options = []

    def load_image_batch(_records, _image_size, **kwargs):
        batch_options.append(kwargs)
        return np.zeros((1, 8, 8, 3))

    class FakePyDataset:
        def __init__(self, **_kwargs) -> None:
            pass

    monkeypatch.setattr(module, "_load_image_batch", load_image_batch)
    image_sequence_class = module._make_image_sequence_class(
        SimpleNamespace(utils=SimpleNamespace(PyDataset=FakePyDataset))
    )
    sequence = image_sequence_class(
        [ImageRecord(Path("image.jpg"), ())],
        (8, 8),
        interpolation="bicubic",
    )

    sequence[0]

    assert batch_options == [
        {
            "crop_to_aspect_ratio": True,
            "crop_window_scale": 1.0,
            "interpolation": "bicubic",
        }
    ]


@pytest.mark.parametrize(
    ("interpolation", "expected_resampling"),
    [
        ("nearest", Image.Resampling.NEAREST),
        ("box", Image.Resampling.BOX),
        ("bilinear", Image.Resampling.BILINEAR),
        ("hamming", Image.Resampling.HAMMING),
        ("bicubic", Image.Resampling.BICUBIC),
        ("lanczos", Image.Resampling.LANCZOS),
    ],
)
def test_load_img_passes_interpolation_to_resize(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interpolation: Literal[
        "nearest", "box", "bilinear", "hamming", "bicubic", "lanczos"
    ],
    expected_resampling: Image.Resampling,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    original_resize = Image.Image.resize
    resampling_options = []

    def resize(image, size, resample=None, box=None, reducing_gap=None):
        resampling_options.append(resample)
        return original_resize(
            image, size, resample=resample, box=box, reducing_gap=reducing_gap
        )

    monkeypatch.setattr(Image.Image, "resize", resize)

    module._load_img(image_path, (4, 4), interpolation=interpolation)

    assert resampling_options == [expected_resampling]


def test_load_img_defaults_to_bilinear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (8, 8)).save(image_path)
    original_resize = Image.Image.resize
    resampling_options = []

    def resize(image, size, resample=None, box=None, reducing_gap=None):
        resampling_options.append(resample)
        return original_resize(
            image, size, resample=resample, box=box, reducing_gap=reducing_gap
        )

    monkeypatch.setattr(Image.Image, "resize", resize)

    module._load_img(image_path, (4, 4))

    assert resampling_options == [Image.Resampling.BILINEAR]
