# phototagger

Project to automatically tag photos using different AI models.

The following tags can be attributed using existing models/API's:
- the imagenet classes
- use the plantnet API to tag plants

The project also supports creating your own custom models if you create/have your own
training data.

## Environment

Create the development environment from `environment-dev.yml`, then select the
Keras backend before running Python:

```powershell
conda env create -f environment-dev.yml
conda activate phototagger-dev
$env:KERAS_BACKEND = "torch"
```

## Classify with pretrained ImageNet categories

Pretrained ImageNet21k classification is available through the
[`scripts/run_imagenet_classify.py`](scripts/run_imagenet_classify.py) script.

## Train custom model

### Maize mulch classification

The project includes a sample script to train a classifier that detects if an image
contains maize mulch or not.

The script can be found here: [run_maizemulch_train.py](scripts/run_maizemulch_train.py)

**Note**: the training data for the maize mulch classification is not available as open
data!
