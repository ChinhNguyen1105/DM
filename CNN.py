import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models


class CNNFramework:

    def __init__(self, input_shape=(28, 28, 1), num_classes=10):

        self.input_shape = input_shape
        self.num_classes = num_classes

        self.model = None
        self.train_data = None
        self.test_data = None

        self.class_names = None
        self.history = None

    def load_numpy_data(self, X_train, y_train, X_test, y_test):

        X_train = np.asarray(X_train, dtype="float32")
        X_test = np.asarray(X_test, dtype="float32")

        if X_train.ndim == 3:
            X_train = np.expand_dims(X_train, -1)
            X_test = np.expand_dims(X_test, -1)

        if X_train.max() > 1:
            X_train /= 255.0
            X_test /= 255.0

        self.train_data = (X_train, np.asarray(y_train))
        self.test_data = (X_test, np.asarray(y_test))

        self.input_shape = X_train.shape[1:]

        print("Train:", X_train.shape)
        print("Test:", X_test.shape)

    def load_folder_data(
        self, train_dir, test_dir,
        image_size=(224, 224), batch_size=32,
        color_mode="rgb"
    ):

        self.train_data = tf.keras.utils.image_dataset_from_directory(
            train_dir,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode="int"
        )

        self.test_data = tf.keras.utils.image_dataset_from_directory(
            test_dir,
            image_size=image_size,
            batch_size=batch_size,
            color_mode=color_mode,
            label_mode="int",
            shuffle=False
        )

        self.class_names = self.train_data.class_names

        if self.class_names != self.test_data.class_names:
            raise ValueError("Train/Test classes do not match!")

        self.num_classes = len(self.class_names)

        channels = 1 if color_mode == "grayscale" else 3
        self.input_shape = (*image_size, channels)

        self.train_data = self.train_data.map(
            lambda x, y: (x / 255.0, y)
        )

        self.test_data = self.test_data.map(
            lambda x, y: (x / 255.0, y)
        )

        print("Classes:", self.class_names)

    def build_model(self):

        self.model = models.Sequential([

            layers.Input(shape=self.input_shape),

            layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
            layers.MaxPooling2D((2, 2)),

            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.3),

            layers.Dense(self.num_classes, activation="softmax")
        ])

        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )

        self.model.summary()

        return self.model

    def train(self, epochs=10, batch_size=32, validation_split=0.1):

        if self.model is None or self.train_data is None:
            raise ValueError("Build model and load data first!")

        if isinstance(self.train_data, tuple):
            X_train, y_train = self.train_data

            self.history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split
            )
        else:
            self.history = self.model.fit(
                self.train_data,
                validation_data=self.test_data,
                epochs=epochs
            )

        return self.history

    def evaluate(self):

        if self.model is None or self.test_data is None:
            raise ValueError("Build model and load data first!")

        if isinstance(self.test_data, tuple):
            loss, accuracy = self.model.evaluate(*self.test_data)
        else:
            loss, accuracy = self.model.evaluate(self.test_data)

        print(f"Loss: {loss:.4f}")
        print(f"Accuracy: {accuracy * 100:.2f}%")

        return loss, accuracy

    def predict(self, image):

        image = np.asarray(image, dtype="float32")

        if image.max() > 1:
            image /= 255.0

        if image.ndim == 2:
            image = np.expand_dims(image, -1)

        prediction = self.model.predict(
            np.expand_dims(image, 0), verbose=0
        )

        index = np.argmax(prediction[0])
        confidence = prediction[0][index]

        label = self.class_names[index] if self.class_names else index

        print("Prediction:", label)
        print(f"Confidence: {confidence * 100:.2f}%")

        return index, confidence

    def plot_history(self):

        if self.history is None:
            raise ValueError("Train model first!")

        plt.figure(figsize=(12, 4))

        for i, metric in enumerate(["accuracy", "loss"]):
            plt.subplot(1, 2, i + 1)
            plt.plot(self.history.history[metric], label="Train")
            plt.plot(self.history.history["val_" + metric], label="Validation")
            plt.title(metric.capitalize())
            plt.legend()

        plt.show()

    def save_model(self, path="cnn_model.keras"):

        self.model.save(path)
        print("Model saved:", path)

    def load_model(self, path):

        self.model = tf.keras.models.load_model(path)
        print("Model loaded:", path)


if __name__ == "__main__":

    cnn = CNNFramework()

    (X_train, y_train), (X_test, y_test) = (
        tf.keras.datasets.mnist.load_data()
    )

    cnn.load_numpy_data(X_train, y_train, X_test, y_test)

    cnn.build_model()
    cnn.train(epochs=5)

    cnn.evaluate()

    cnn.predict(X_test[0])

    cnn.plot_history()

    cnn.save_model()