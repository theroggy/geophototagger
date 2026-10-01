"""Direct inference with pretrained wide-label ImageNet classifiers."""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from geophototagger.custom.dataset import SUPPORTED_IMAGE_SUFFIXES

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
    images: list[Path] | Path,
    model_name: str = "efficientnetv2",
    threshold: float = 0.01,
    batch_size: int = 8,
    output_dir: Path | None = None,
) -> list[tuple[Path, list[Prediction]]]:
    """Return all classes meeting the threshold for each image, in input order.

    ``images`` can be a list of image paths, a single image file, or a
    directory, in which case all supported images directly inside it are
    classified. Weights are downloaded and cached by timm on first use.
    ConvNeXt uses ImageNet-22k labels, which differ from EfficientNetV2's
    ImageNet-21k labels.

    If output_dir is given, predictions are appended to
    ``output_dir/predictions.csv`` after each batch, one row per image with
    its matches as a JSON list in the ``predictions`` column. Any image
    already listed in that file is skipped instead of being reclassified.
    """
    if isinstance(images, Path):
        images = (
            sorted(
                path
                for path in images.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
            )
            if images.is_dir()
            else [images]
        )

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}; choose from {sorted(MODELS)}")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not images:
        raise ValueError("No images to classify")
    for image_path in images:
        if not image_path.is_file():
            raise ValueError(f"Image does not exist: {image_path}")

    csv_path = None
    already_processed: set[str] = set()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "predictions.csv"
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8", newline="") as csv_file:
                already_processed = {
                    row["image_path"] for row in csv.DictReader(csv_file)
                }

    images = [
        image_path for image_path in images if str(image_path) not in already_processed
    ]
    if not images:
        return []

    timm = importlib.import_module("timm")
    torch = importlib.import_module("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint, subset = MODELS[model_name]
    model = timm.create_model(checkpoint, pretrained=True).eval().to(device)
    labels = timm.data.ImageNetInfo(subset)
    if model.num_classes != labels.num_classes():
        raise ValueError(
            f"Model has {model.num_classes} classes, "
            f"but {subset} has {labels.num_classes()} labels"
        )
    data_config = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**data_config, is_training=False)

    results: list[tuple[Path, list[Prediction]]] = []
    write_header = csv_path is not None and not csv_path.is_file()
    with tqdm(total=len(images), desc="Classifying", unit="image") as progress:
        for start in range(0, len(images), batch_size):
            paths = images[start : start + batch_size]
            tensors = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    tensors.append(transform(image.convert("RGB")))
            with torch.inference_mode():
                batch = torch.stack(tensors).to(device)
                probabilities = model(batch).softmax(dim=1)
            expected_shape = (len(paths), labels.num_classes())
            if tuple(probabilities.shape) != expected_shape:
                raise ValueError(
                    f"Model returned shape {tuple(probabilities.shape)}, "
                    f"expected {expected_shape}"
                )
            batch_results: list[tuple[Path, list[Prediction]]] = []
            for image_path, scores in zip(
                paths, probabilities.cpu().tolist(), strict=True
            ):
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
                batch_results.append((image_path, matches))
            results.extend(batch_results)
            if csv_path is not None:
                with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if write_header:
                        writer.writerow(("image_path", "model", "predictions"))
                        write_header = False
                    for image_path, matches in batch_results:
                        predictions = [
                            {
                                "class_index": match.class_index,
                                "synset": match.synset,
                                "label": match.label,
                                "probability": match.probability,
                            }
                            for match in matches
                        ]
                        writer.writerow(
                            (str(image_path), model_name, json.dumps(predictions))
                        )
            progress.update(len(paths))

    return results
