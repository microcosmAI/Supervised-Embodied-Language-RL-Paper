import os
from pathlib import Path


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    Flatten,
    Dense,
    Reshape,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import matplotlib.pyplot as plt


class Autoencoder(Model):

    def __init__(self, latent_dim, input_shape):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = tf.keras.Sequential(
            [
                Input(shape=input_shape),
                Conv2D(
                    32,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                Conv2D(
                    64,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    activation="relu",
                ),
                Flatten(),
                Dense(latent_dim, activation="sigmoid"),
            ]
        )
        self.decoder = tf.keras.Sequential(
            [
                Dense(16 * 16 * 256, activation="relu"),
                Reshape((16, 16, 256)),
                Conv2DTranspose(
                    64, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                Conv2DTranspose(
                    32, kernel_size=3, strides=2, padding="same", activation="relu"
                ),
                Conv2DTranspose(3, kernel_size=3, activation="sigmoid", padding="same"),
            ]
        )

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def sample(self, n_samples):
        z = tf.random.normal(shape=(n_samples, self.latent_dim))
        return self.decode(z).numpy()


def load_data(data_dir, image_size=(64, 64)):
    images = []
    image_names = os.listdir(data_dir)
    for image_name in image_names:
        if image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image = cv2.imread(os.path.join(data_dir, image_name))
            image = cv2.resize(image, image_size[:2])
            image = image / 255.0
            images.append(image)
    return np.array(images)


def preprocess_image(image, image_size=(64, 64)):
    """Load and prepare a single image."""
    image = cv2.resize(image, image_size)
    image = image / 255.0
    return image


def get_a_single_image_embedding(autoencoder, image):
    """Load and encode a single image."""
    image = preprocess_image(image)
    encoded_image = autoencoder.encode(image[None, ...]).numpy()[0]
    return encoded_image


def dssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)) / 2


def display_samples(samples, n_cols=5):
    n_rows = (len(samples) + n_cols - 1) // n_cols
    plt.figure(figsize=(n_cols * 2, n_rows * 2))
    for i, image in enumerate(samples):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(image)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig("generated_samples.png")


def plot_original_and_reconstructed(
    autoencoder, image, save_path="reconstructed_image.png"
):
    test_image = preprocess_image(image)
    test_image_processed = test_image[None, ...]

    reconstructed_image = autoencoder.predict(test_image_processed)[0]

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(test_image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_image)
    plt.title("Reconstructed")
    plt.axis("off")

    plt.savefig(save_path)
    plt.show()


def load_autoencoder_model(model_path):
    return load_model(
        model_path,
        custom_objects={"Autoencoder": Autoencoder, "dssim_loss": dssim_loss},
    )


def train_model(
    data_dir,
    input_shape=(64, 64, 3),
    latent_dim=50,
    batch_size=8,
    epochs=20,
    model_path=None,
):
    # Load and prepare data
    X = load_data(data_dir, input_shape)
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # Create autoencoder model
    autoencoder = Autoencoder(latent_dim, input_shape)
    autoencoder.compile(optimizer=Adam(1e-3), loss=dssim_loss)

    # Set up callbacks
    callbacks = [
        ModelCheckpoint(
            filepath=str(model_path),
            save_best_only=True,
            verbose=1,
            monitor="val_loss",
        ),
        EarlyStopping(monitor="val_loss", patience=5, verbose=1),
    ]

    # Train the model
    autoencoder.fit(
        x=X_train,
        y=X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
    )
    return autoencoder


if __name__ == "__main__":

    input_shape = (64, 64, 3)
    latent_dim = 30
    batch_size = 64
    epochs = 100

    experiment_name = '3colors'

    repo_root = Path.cwd()
    dataset_dir = repo_root / "data" / experiment_name
    model_path = repo_root / "models" / f"{experiment_name}_{latent_dim}.tf"

    if model_path.exists():
        print("Loading autoencoder model...")
        autoencoder = load_autoencoder_model(model_path)
    else:
        print("Training autoencoder model...")
        autoencoder = train_model(
            data_dir=dataset_dir,
            input_shape=input_shape,
            latent_dim=latent_dim,
            batch_size=batch_size,
            epochs=epochs,
            model_path=model_path,
        )

    # Test the model with a sample image
    test_image_path = dataset_dir / "300.png"
    image = cv2.imread(str(test_image_path))
    plot_original_and_reconstructed(autoencoder, image)
    embedding = get_a_single_image_embedding(autoencoder, image)
    print('Embedding:', embedding)
