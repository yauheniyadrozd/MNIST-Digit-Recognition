import os
import numpy as np
import pandas as pd


def load_images_from_csv(path):
    df = pd.read_csv(path)
    X = df.values.astype("float32")
    X = X.reshape((-1, 28, 28, 1))
    X /= 255.0
    return X


def load_npz(path):
    data = np.load(path)
    X = data["X"]
    y = data["y"] if "y" in data else None
    return X, y


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
    return out_path


def prepare_test(csv_path, out_path=None):
    df = pd.read_csv(csv_path)
    X = df.values.astype("float32")
    X = X.reshape((-1, 28, 28, 1)) / 255.0
    if out_path is None:
        out_path = os.path.join(os.path.dirname(csv_path), "test.npz")
    np.savez_compressed(out_path, X=X)
    return out_path
