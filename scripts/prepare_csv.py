"""Prepare CSV image datasets for training/evaluation.

Reads CSVs in `data/` where `train.csv` has a `label` column and pixel columns,
and `test.csv` has only pixel columns. Saves numpy .npz files to `data/`.
"""
import os
import numpy as np
import pandas as pd


def prepare_train(csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    if "label" not in df.columns:
        raise ValueError("train.csv must contain a 'label' column")
    y = df["label"].values.astype("int64")
    X = df.drop(columns=["label"]).values.astype("float32")
    X = X.reshape((-1, 28, 28, 1)) / 255.0
    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "train.npz")
    np.savez_compressed(out_path, X=X, y=y)
    print(f"Wrote {out_path} ({X.shape[0]} samples)")


def prepare_test(csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    X = df.values.astype("float32")
    X = X.reshape((-1, 28, 28, 1)) / 255.0
    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "test.npz")
    np.savez_compressed(out_path, X=X)
    print(f"Wrote {out_path} ({X.shape[0]} samples)")


def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    data_dir = os.path.abspath(data_dir)
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")
    if os.path.exists(train_csv):
        prepare_train(train_csv)
    else:
        print("No train.csv found — skipping train prep.")
    if os.path.exists(test_csv):
        prepare_test(test_csv)
    else:
        print("No test.csv found — skipping test prep.")


if __name__ == "__main__":
    main()
