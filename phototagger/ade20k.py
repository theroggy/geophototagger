"""Semantic segmentation with a permissively-licensed ADE20K model (UPerNet)."""

from __future__ import annotations

import csv
import importlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm.auto import tqdm

from geophototagger.custom.dataset import SUPPORTED_IMAGE_SUFFIXES

DEFAULT_MODEL = "openmmlab/upernet-convnext-tiny"
# The model only sees ~512px anyway, so decoding full-resolution JPEGs is
# wasted work; this hints libjpeg to decode near this size directly.
DRAFT_SIZE = (512, 512)


@dataclass(frozen=True)
class SceneClass:
    """An ADE20K class and the fraction of image pixels it covers."""

    label: str
    fraction: float


def segment_images(
    images: list[Path] | Path,
    model_name: str = DEFAULT_MODEL,
    min_fraction: float = 0.02,
    batch_size: int = 8,
    output_dir: Path | None = None,
) -> list[tuple[Path, list[SceneClass]]]:
    """Return ADE20K scene classes covering at least `min_fraction` of each image.

    ``images`` can be a list of image paths, a single image file, or a
    directory, in which case all supported images directly inside it are
    segmented. Unlike a whole-image classifier, this reports every class
    present in the scene (e.g. "sky", "grass", "road") based on per-pixel
    coverage, so multiple classes can legitimately score highly at once.
    Weights are downloaded and cached by `transformers` on first use.

    If output_dir is given, results are appended to
    ``output_dir/segments.csv`` after each batch, one row per image with its
    classes as a JSON list in the ``segments`` column. Any image already
    listed in that file is skipped instead of being reprocessed.
    """
    if not 0 <= min_fraction <= 1:
        raise ValueError("min_fraction must be between 0 and 1")
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
        raise ValueError("No images to segment")
    for image_path in images:
        if not image_path.is_file():
            raise ValueError(f"Image does not exist: {image_path}")

    csv_path = None
    already_processed: set[str] = set()
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        csv_path = output_dir / "segments.csv"
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
    use_half_precision = device.type == "cuda"

    processor = transformers.AutoImageProcessor.from_pretrained(
        model_name, backend="torchvision"
    )
    model = transformers.AutoModelForSemanticSegmentation.from_pretrained(model_name)
    model = model.eval().to(device)
    if use_half_precision:
        model = model.half()
    id2label = model.config.id2label

    results: list[tuple[Path, list[SceneClass]]] = []
    write_header = csv_path is not None and not csv_path.is_file()
    start_time = time.perf_counter()
    read_time = 0.0
    predict_time = 0.0
    postprocess_time = 0.0
    progress_desc = f"Segmenting ({device})"
    with tqdm(total=len(images), desc=progress_desc, unit="image") as progress:
        for start in range(0, len(images), batch_size):
            paths = images[start : start + batch_size]
            stage_start = time.perf_counter()
            pil_images = []
            for image_path in paths:
                with Image.open(image_path) as image:
                    image.draft("RGB", DRAFT_SIZE)
                    pil_images.append(image.convert("RGB"))
            stage_end = time.perf_counter()
            read_time += stage_end - stage_start
            stage_start = stage_end

            inputs = processor(images=pil_images, return_tensors="pt").to(device)
            if use_half_precision:
                inputs = inputs.to(torch.float16)
            with torch.inference_mode():
                outputs = model(**inputs)
            stage_end = time.perf_counter()
            predict_time += stage_end - stage_start
            stage_start = stage_end

            # Skip resizing back to the original photo resolution: the model's
            # native output resolution (matching its input size) is already an
            # upper bound on real detail, and upsampling large photos here is
            # by far the most expensive step for no extra information.
            segmentation_maps = processor.post_process_semantic_segmentation(outputs)

            batch_results: list[tuple[Path, list[SceneClass]]] = []
            for image_path, segmentation in zip(paths, segmentation_maps, strict=True):
                class_ids, counts = torch.unique(segmentation, return_counts=True)
                total_pixels = int(counts.sum())
                classes = [
                    SceneClass(
                        label=id2label[int(class_id)],
                        fraction=round(int(count) / total_pixels, 4),
                    )
                    for class_id, count in zip(
                        class_ids.tolist(), counts.tolist(), strict=True
                    )
                    if count / total_pixels >= min_fraction
                ]
                classes.sort(key=lambda scene_class: -scene_class.fraction)
                batch_results.append((image_path, classes))
            results.extend(batch_results)

            if csv_path is not None:
                with csv_path.open("a", encoding="utf-8", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    if write_header:
                        writer.writerow(("image_path", "model", "segments"))
                        write_header = False
                    for image_path, classes in batch_results:
                        payload = [
                            {
                                "label": scene_class.label,
                                "fraction": scene_class.fraction,
                            }
                            for scene_class in classes
                        ]
                        writer.writerow(
                            (str(image_path), model_name, json.dumps(payload))
                        )
            postprocess_time += time.perf_counter() - stage_start
            elapsed = time.perf_counter() - start_time
            progress.update(len(paths))
            progress.set_postfix(rate=f"{progress.n / elapsed:.2f} img/s")
    elapsed = time.perf_counter() - start_time
    print(
        f"Segmented {len(images)} images in {elapsed:.1f}s "
        f"({len(images) / elapsed:.2f} img/s)"
    )
    print(
        f"  reading: {read_time:.1f}s, predicting: {predict_time:.1f}s, "
        f"postprocessing: {postprocess_time:.1f}s"
    )
    return results
