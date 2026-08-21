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

Train a frozen pretrained EfficientNetV2 baseline and save the model plus its
label metadata:

Edit the constants at the top of [run_training.py](run_training.py), then run:

```powershell
python run_training.py
```

Training enables random horizontal flips, small rotations, zoom, and contrast
changes by default. These transformations are active only during training and
are disabled automatically during prediction. Set `AUGMENT = False` in the
script for a controlled comparison run.

Balanced positive and negative class weighting is also enabled by default. This
gives rare labels more influence during binary cross-entropy training. Use
`CLASS_WEIGHTING = False` in the script for an unweighted comparison run.

Validation uses the separate validation directory directly; it is not sampled
from the training directory.

The default `imagenet` weights may require network access on the first run. For
an offline smoke test, set `WEIGHTS = None` in the script.

## Predict

Run tagging for one image. The command prints labels whose probability is at
least the threshold stored beside the model:

```powershell
python -m geophototagger.predict `
	X:\Monitoring\geophototagger\geophototagger.keras `
	X:\Monitoring\geophototagger\new-image.jpg
```

The model output uses a sigmoid probability for every label, allowing one
image to receive zero, one, or several tags.

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

Set `LABEL_WHITELIST` in [run_training.py](run_training.py) to limit training to
selected labels. Labels not in the whitelist are removed from multi-label
images; images with no remaining labels are excluded.
