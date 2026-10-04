# geophototagger

This project contains the original PlantNet proof of concept and a local image
tagger based on Keras 3 with the PyTorch backend.

## Environment

Create the development environment from `environment-dev.yml`, then select the
Keras backend before running Python:

```powershell
conda env create -f environment-dev.yml
conda activate geophototagger-dev
$env:KERAS_BACKEND = "torch"
```

## Create a training manifest

The training script below generates separate CSV manifests for the `train` and
`validation` directories before training. Formatted filenames use labels after
the final underscore, with multiple labels separated by hyphens.

The generated CSV has two columns:

```text
image_path,labels
korrelmais/example.jpg,korrelmais
```

Multiple tags are represented in the same `labels` field with semicolons:

```text
image_path,labels
shared/example.jpg,korrelmais;stalmest
```

Images must exist relative to the dataset root when a relative path is used.
The scanner accepts JPG, JPEG, PNG, BMP, and WebP files, and ignores `Thumbs.db`
and `logs` directories.

## Train

Train the configured multilabel EfficientNetV2 classifier and save the model
plus its label and training metadata:

Edit the paths and settings in
[run_maizemulch_train.py](scripts/run_maizemulch_train.py), then run:

```powershell
python scripts/run_maizemulch_train.py
```

The training API defaults to five frozen-backbone epochs, then fine-tunes the
last 10% of backbone layers (rounded up and expanded to whole blocks). Its
classification head applies Dropout at 0.5 and
L2 kernel regularization at 0.005 by default; the script exposes these values
as local settings. Set either regularization value to zero to disable it.
When the monitored loss or metric plateaus for three epochs, training reduces
the learning rate by a factor of 0.2, down to a minimum of `1e-6`. The initial
learning rate defaults to `1e-3` and can be changed in the training script.
Input images are center-cropped to the target aspect ratio by default, which
preserves geometry but can trim edges. Set `crop_to_aspect_ratio = False` in the
training script to stretch images instead.

Random horizontal flips, rotations, zoom, contrast, and brightness are enabled
by default during training. Class weighting is disabled by default.

Validation uses the separate validation directory directly; it is not sampled
from the training directory.

The default `imagenet` weights may require network access on the first run. For
an offline smoke test, set `weights = None` in the script.

## Predict

Run tagging for one image. The command prints labels whose probability is at
least the threshold stored beside the model:

```powershell
python -m geophototagger.predict `
	X:\Monitoring\phototagger\geophototagger.keras `
	X:\Monitoring\phototagger\new-image.jpg
```

The model output uses a sigmoid probability for every label, allowing one
image to receive zero, one, or several tags.

## Classify with pretrained ImageNet categories

Pretrained ImageNet21k classification is available through the
[`scripts/run_imagenet_classify.py`](scripts/run_imagenet_classify.py) script.

## Simplify ControlefotosJRC classes

Convert the JRC training CSV while preserving its original columns:

```powershell
python convert_traindata_classes.py
```

The script writes these files beside the source CSV:

- `class_mappings.csv` contains `hoofdteelt`, `source_class`,
	`simplified_nl`, and `simplified_en`.
- `traindata_with_simplified_classes.csv` contains the original columns plus
	`CLASSES_SIMPLIFIED_NL` and `CLASSES_SIMPLIFIED_EN`.

Mappings are keyed by the `(HOOFDTEELT, CLASSES)` pair. This preserves source
identifiers such as maize codes 201 and 202 even when they share the simplified
label `mais` / `maize`. Review or edit `class_mappings.csv` before rerunning a
conversion when a classification needs to change.

## Label whitelist

Set `classes` in [run_maizemulch_train.py](scripts/run_maizemulch_train.py) to
limit training to selected labels. Labels not in this set are removed from
multi-label images; images with no remaining labels are excluded.
