import os
import argparse
import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers


def build_simple_model(input_shape=(28, 28, 1), num_classes=10):
    model = keras.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Conv2D(16, 3, activation="relu"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 
    return model


def load_dataset():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype("float32") / 255.0
    import os
    import argparse
    import numpy as np
    import pandas as pd

    from src.model import build_model
    from src.data import load_images_from_csv
    from tensorflow import keras


    def load_dataset():
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
        return (x_train, y_train), (x_test, y_test)


    def predict_on_provided_test(model, out_dir="data"):
        test_csv = os.path.join(out_dir, "test.csv")
        if not os.path.exists(test_csv):
            print("No data/test.csv found — skipping prediction step.")
            return
        X = load_images_from_csv(test_csv)
        preds = model.predict(X)
        labels = np.argmax(preds, axis=1)
        submission = pd.DataFrame({"ImageId": np.arange(1, len(labels) + 1), "Label": labels})
        out_path = os.path.join(out_dir, "submission.csv")
        submission.to_csv(out_path, index=False)
        print(f"Saved predictions to {out_path}")


    def main():
        parser = argparse.ArgumentParser(description="Minimal MNIST training example")
        parser.add_argument("--epochs", type=int, default=1)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--save-model", default="models/example_model.h5")
        args = parser.parse_args()

        (x_train, y_train), (x_test, y_test) = load_dataset()
        model = build_model(input_shape=(28, 28, 1))
        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)
        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Test accuracy: {acc:.4f}")

        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        model.save(args.save_model)
        print(f"Saved model to {args.save_model}")

        predict_on_provided_test(model)


    if __name__ == "__main__":
        main()
