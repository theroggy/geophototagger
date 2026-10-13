"""Image loading and batching helpers for custom classifiers."""

from __future__ import annotations

import math
import os
import warnings
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
from PIL import Image, ImageOps

if TYPE_CHECKING:
    from pathlib import Path

    from .dataset import ImageRecord

ImageInterpolation = Literal[
    "nearest",
    "box",
    "bilinear",
    "hamming",
    "bicubic",
    "lanczos",
]


def _keras() -> Any:
    """Import Keras only after selecting the requested backend."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras  # noqa: PLC0415

    return keras


def _load_image_batch(
    records: list[ImageRecord],
    image_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
    interpolation: ImageInterpolation = "bilinear",
) -> np.ndarray:
    """Load a batch of images into a single ndarray."""
    _validate_crop_window_scale(crop_window_scale)
    keras = _keras()
    images = []
    for record in records:
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            try:
                image = _load_img(
                    record.image_path,
                    image_size,
                    crop_to_aspect_ratio=crop_to_aspect_ratio,
                    crop_window_scale=crop_window_scale,
                    interpolation=interpolation,
                )
            except OSError as error:
                raise OSError(
                    f"Failed to load image {record.image_path}: {error}"
                ) from error
        for caught_warning in caught_warnings:
            warnings.warn(
                f"{record.image_path}: {caught_warning.message}",
                caught_warning.category,
                stacklevel=2,
            )
        images.append(keras.utils.img_to_array(image))
    return np.array(images)


def _load_img(
    image_path: Path,
    target_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
    interpolation: ImageInterpolation = "bilinear",
) -> Image.Image:
    """Load an RGB image and resize it to the requested dimensions.

    Args:
        image_path: Path to the source image.
        target_size: Output size as ``(height, width)``.
        crop_to_aspect_ratio: Whether to center-crop the source to the target
            aspect ratio. If false, resize the entire image and distort its
            aspect ratio as needed.
        crop_window_scale: Fraction of the aspect-fitted crop's width and
            height to retain, in ``(0, 1]``. Values below one apply an extra
            centered crop when it still leaves at least ``target_size`` pixels
            in both dimensions. Otherwise, the standard aspect-ratio crop is
            used. This option has no effect when ``crop_to_aspect_ratio`` is
            false.
        interpolation: Pillow resampling filter name to use when resizing.

    Returns:
        The resized RGB image.

    Raises:
        ValueError: If ``crop_window_scale`` is outside ``(0, 1]``.
    """
    _validate_crop_window_scale(crop_window_scale)
    target_height, target_width = target_size
    with Image.open(image_path) as opened_image:
        image = ImageOps.exif_transpose(opened_image).convert("RGB")
    if crop_to_aspect_ratio:
        image_width, image_height = image.size
        target_ratio = target_width / target_height
        if image_width / image_height > target_ratio:
            crop_width = int(image_height * target_ratio)
            crop_height = image_height
        else:
            crop_width = image_width
            crop_height = int(image_width / target_ratio)

        if crop_window_scale < 1:
            tighter_width = int(crop_width * crop_window_scale)
            tighter_height = int(crop_height * crop_window_scale)
            if tighter_width >= target_width and tighter_height >= target_height:
                crop_width = tighter_width
                crop_height = tighter_height

        left = (image_width - crop_width) // 2
        top = (image_height - crop_height) // 2
        image = image.crop((left, top, left + crop_width, top + crop_height))

    return image.resize(
        (target_width, target_height),
        resample=Image.Resampling[interpolation.upper()],
    )


def _validate_crop_window_scale(crop_window_scale: float) -> None:
    if (
        isinstance(crop_window_scale, bool)
        or not isinstance(crop_window_scale, (int, float))
        or not math.isfinite(crop_window_scale)
        or not 0 < crop_window_scale <= 1
    ):
        raise ValueError("crop_window_scale must be greater than 0 and at most 1")


def _crop_window_scale_from_metadata(metadata: dict[str, Any]) -> float:
    if "crop_window_scale" in metadata:
        return metadata["crop_window_scale"]
    legacy_crop_percent = metadata.get("aspect_ratio_crop_percent", 0.0)
    return 1.0 - legacy_crop_percent / 100


def _load_images(
    records: list[ImageRecord],
    image_size: tuple[int, int],
    *,
    crop_to_aspect_ratio: bool = True,
    crop_window_scale: float = 1.0,
    interpolation: ImageInterpolation = "bilinear",
) -> Any:
    keras = _keras()
    return keras.ops.convert_to_tensor(
        _load_image_batch(
            records,
            image_size,
            crop_to_aspect_ratio=crop_to_aspect_ratio,
            crop_window_scale=crop_window_scale,
            interpolation=interpolation,
        )
    )


def _make_image_sequence_class(keras: Any) -> type:
    """Build a ``PyDataset`` subclass that lazily loads images per batch."""

    class _ImageSequence(keras.utils.PyDataset):
        def __init__(
            self,
            records: list[ImageRecord],
            image_size: tuple[int, int],
            *,
            labels: np.ndarray | None = None,
            sample_weights: np.ndarray | None = None,
            crop_to_aspect_ratio: bool = True,
            crop_window_scale: float = 1.0,
            interpolation: ImageInterpolation = "bilinear",
            batch_size: int = 32,
            shuffle: bool = False,
            **kwargs: Any,
        ) -> None:
            super().__init__(**kwargs)
            self.records = records
            self.image_size = image_size
            self.labels = labels
            self.sample_weights = sample_weights
            self.crop_to_aspect_ratio = crop_to_aspect_ratio
            self.crop_window_scale = crop_window_scale
            self.interpolation = interpolation
            self.batch_size = batch_size
            self.shuffle = shuffle
            self.indices = np.arange(len(records))
            self._rng = np.random.default_rng()

        def __len__(self) -> int:
            return max(1, -(-len(self.records) // self.batch_size))

        def __getitem__(self, index: int) -> Any:
            batch_indices = self.indices[
                index * self.batch_size : (index + 1) * self.batch_size
            ]
            batch_images = _load_image_batch(
                [self.records[i] for i in batch_indices],
                self.image_size,
                crop_to_aspect_ratio=self.crop_to_aspect_ratio,
                crop_window_scale=self.crop_window_scale,
                interpolation=self.interpolation,
            )
            if self.labels is None:
                return batch_images
            batch_labels = self.labels[batch_indices]
            if self.sample_weights is None:
                return batch_images, batch_labels
            return batch_images, batch_labels, self.sample_weights[batch_indices]

        def on_epoch_end(self) -> None:
            if self.shuffle:
                self._rng.shuffle(self.indices)

    return _ImageSequence
