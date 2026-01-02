# MNIST Digit Recognition
Simple MNIST digit recognition project using the CSV dataset files included in this repository.

## Project structure

- `main.py` - entrypoint script (training / inference pipeline using TensorFlow/Keras).
- `train_example.py` - minimal example training script (small CNN using `keras.datasets.mnist`).
- `data/train.csv` - training data (labels + pixels).
- `data/test.csv` - test data (pixels only).
- `data/submission.csv` - example submission / output format.
- `requirements.txt` - Python dependencies.

## Setup

1. Create a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

- To run the main script (train or predict), run:

```bash
python main.py
```

- To run the minimal example:

```bash
python train_example.py --epochs 1 --batch-size 128
```

## Preparing CSV data

If you have `data/train.csv` and/or `data/test.csv`, convert them to compressed numpy archives for faster loading:

```bash
python scripts/prepare_csv.py
```

This will create `data/train.npz` and/or `data/test.npz` next to the CSVs.

## Ignored files

A `.gitignore` file was added to exclude virtual environments, editor configs and generated models.

## Data format

- `train.csv` typically contains a `label` column followed by pixel columns (`pixel0`,`pixel1`,...).
- `test.csv` contains the same pixel columns without the `label` column.
- `submission.csv` should contain at least two columns: `ImageId` and `Label` for Kaggle-style submissions.

## Tips

- Start with a simple model (logistic regression or a small CNN) and a minimal preprocessing pipeline (reshape pixels to 28x28, normalize to [0,1]).
- Use `scikit-learn`, `pandas`, and `tensorflow`/`torch` as needed (see `requirements.txt`).

## Contributing

Contributions are welcome — open an issue or submit a PR with improvements.

## License

This repository does not specify a license. Add one if you plan to publish or allow reuse.
