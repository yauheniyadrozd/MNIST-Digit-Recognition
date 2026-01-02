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
    import os
    import argparse
    import numpy as np
    import pandas as pd

    from src.model import build_model
    from src.data import load_images_from_csv
    from tensorflow import keras


    def main():
        parser = argparse.ArgumentParser(description="Train a simple MNIST model and optionally predict on provided CSV test set")
        parser.add_argument("--epochs", type=int, default=3)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--save-model", default="models/model.h5")
        args = parser.parse_args()

        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)

        model = build_model()

        model.fit(x_train, y_train, epochs=args.epochs, batch_size=args.batch_size, validation_split=0.1)

        loss, acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"MNIST test accuracy: {acc:.4f}")

        os.makedirs(os.path.dirname(args.save_model), exist_ok=True)
        model.save(args.save_model)
        print(f"Saved model to {args.save_model}")

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
