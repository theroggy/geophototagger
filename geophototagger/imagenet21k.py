"""Direct inference with pretrained wide-label ImageNet classifiers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from pathlib import Path

MODELS = {
    "efficientnetv2": ("tf_efficientnetv2_s.in21k", "imagenet-21k-goog"),
    "convnext": ("convnext_tiny.fb_in22k", "imagenet-22k"),
}


@dataclass(frozen=True)
class Prediction:
    """A pretrained model's class and its softmax probability."""

    class_index: int
    synset: str
    label: str
    probability: float


def classify_images(
    image_paths: list[Path],
    model_name: str = "efficientnetv2",
    threshold: float = 0.01,
    batch_size: int = 8,
) -> list[tuple[Path, list[Prediction]]]:
    """Return all classes meeting the threshold for each image, in input order.

    Weights are downloaded and cached by timm on first use. ConvNeXt uses
    ImageNet-22k labels, which differ from EfficientNetV2's ImageNet-21k labels.
    """
    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}; choose from {sorted(MODELS)}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not image_paths:
        raise ValueError("No images to classify")
    for image_path in image_paths:
        if not image_path.is_file():
            raise ValueError(f"Image does not exist: {image_path}")

    timm = importlib.import_module("timm")
    torch = importlib.import_module("torch")

    checkpoint, subset = MODELS[model_name]
    model = timm.create_model(checkpoint, pretrained=True).eval()
    labels = timm.data.ImageNetInfo(subset)
    if model.num_classes != labels.num_classes():
        raise ValueError(
            f"Model has {model.num_classes} classes, "
            f"but {subset} has {labels.num_classes()} labels"
        )
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    results: list[tuple[Path, list[Prediction]]] = []
    for start in range(0, len(image_paths), batch_size):
        paths = image_paths[start : start + batch_size]
        images = []
        for image_path in paths:
            with Image.open(image_path) as image:
                images.append(transform(image.convert("RGB")))
        with torch.inference_mode():
            probabilities = model(torch.stack(images)).softmax(dim=1)
        expected_shape = (len(paths), labels.num_classes())
        if tuple(probabilities.shape) != expected_shape:
            raise ValueError(
                f"Model returned shape {tuple(probabilities.shape)}, "
                f"expected {expected_shape}"
            )
        for image_path, scores in zip(paths, probabilities.cpu().tolist(), strict=True):
            matches = [
                Prediction(
                    class_index=index,
                    synset=labels.index_to_label_name(index),
                    label=labels.index_to_description(index),
                    probability=score,
                )
                for index, score in enumerate(scores)
                if score >= threshold
            ]
            matches.sort(
                key=lambda prediction: (
                    -prediction.probability,
                    prediction.class_index,
                )
            )
            results.append((image_path, matches))
    return results
