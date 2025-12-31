import os
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers


def load_images_from_csv(path):
    df = pd.read_csv(path)
    X = df.values.astype("float32")
    X = X.reshape((-1, 28, 28, 1))
    X /= 255.0
    return X


def build_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(32, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def main():
    # Load labeled MNIST from Keras for training
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    model = build_model()

    # Train quickly; increase epochs for better accuracy
    model.fit(x_train, y_train, epochs=3, batch_size=128, validation_split=0.1)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"MNIST test accuracy: {acc:.4f}")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    test_csv = os.path.join(data_dir, "test.csv")
    if os.path.exists(test_csv):
        print("Loading provided test.csv and predicting labels...")
        X_test_provided = load_images_from_csv(test_csv)
        preds = model.predict(X_test_provided)
        labels = np.argmax(preds, axis=1)
        submission = pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels})
        out_path = os.path.join(data_dir, "submission.csv")
        submission.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")
    else:
        print("No data/test.csv found — skipping prediction step.")


if __name__ == "__main__":
    main()
