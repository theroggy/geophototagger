"""Zero-shot, open-vocabulary object detection with OWLv2 (Apache-2.0)."""

from __future__ import annotations

import csv
import importlib
import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from geophototagger.custom.dataset import SUPPORTED_IMAGE_SUFFIXES

DEFAULT_MODEL = "google/owlv2-base-patch16-ensemble"


@dataclass(frozen=True)
class Detection:
    """A detected object's text label, confidence, and bounding box."""

    label: str
    score: float
    box: tuple[float, float, float, float]


def detect_objects(
    images: list[Path] | Path,
    labels: list[str],
    model_name: str = DEFAULT_MODEL,
    threshold: float = 0.1,
    batch_size: int = 8,
    output_dir: Path | None = None,
) -> list[tuple[Path, list[Detection]]]:
    """Detect any of `labels` in each image, independently scored, in input order.

    ``images`` can be a list of image paths, a single image file, or a
    directory, in which case all supported images directly inside it are
    detected. `labels` are free-text class names (e.g. "car", "person") scored
    independently per box, unlike a softmax classifier. Weights are downloaded
    and cached by `transformers` on first use.

    If output_dir is given, detections are appended to
    ``output_dir/detections.csv`` after each batch, one row per image with
    its matches as a JSON list in the ``detections`` column. Any image
    already listed in that file is skipped instead of being reprocessed.
    """
    if not labels:
        raise ValueError("No labels to detect")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

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
    if not images:
        raise ValueError("No images to detect")
    for image_path in images:
        if not image_path.is_file():
            raise ValueError(f"Image does not exist: {image_path}")

    csv_path = None
    already_processed: set[str] = set()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "detections.csv"
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

    transformers = importlib.import_module("transformers")
    torch = importlib.import_module("torch")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    processor = transformers.Owlv2Processor.from_pretrained(model_name)
    model = transformers.Owlv2ForObjectDetection.from_pretrained(model_name)
    model = model.eval().to(device)

    results: list[tuple[Path, list[Detection]]] = []
    write_header = csv_path is not None and not csv_path.is_file()
    with tqdm(total=len(images), desc="Detecting", unit="image") as progress:
        for start in range(0, len(images), batch_size):
            paths = images[start : start + batch_size]
            pil_images = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    pil_images.append(image.convert("RGB"))
            text_labels = [list(labels) for _ in paths]
            inputs = processor(
                text=text_labels, images=pil_images, return_tensors="pt"
            ).to(device)
            with torch.inference_mode():
                outputs = model(**inputs)
            target_sizes = torch.tensor([image.size[::-1] for image in pil_images])
            predictions = processor.post_process_grounded_object_detection(
                outputs=outputs,
                target_sizes=target_sizes,
                threshold=threshold,
                text_labels=text_labels,
            )

            batch_results: list[tuple[Path, list[Detection]]] = []
            for image_path, prediction in zip(paths, predictions, strict=True):
                detections = [
                    Detection(
                        label=label,
                        score=float(score),
                        box=tuple(round(value, 2) for value in box.tolist()),
                    )
                    for box, score, label in zip(
                        prediction["boxes"],
                        prediction["scores"],
                        prediction["text_labels"],
                        strict=True,
                    )
                ]
                detections.sort(key=lambda detection: -detection.score)
                batch_results.append((image_path, detections))
            results.extend(batch_results)

            if csv_path is not None:
                with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if write_header:
                        writer.writerow(("image_path", "model", "detections"))
                        write_header = False
                    for image_path, detections in batch_results:
                        payload = [
                            {
                                "label": detection.label,
                                "score": detection.score,
                                "box": list(detection.box),
                            }
                            for detection in detections
                        ]
                        writer.writerow(
                            (str(image_path), model_name, json.dumps(payload))
                        )
            progress.update(len(paths))
    return results
