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
    BatchNormalization,
    Activation
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
                Conv2D(32, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Conv2D(64, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Conv2D(128, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Flatten(),
                Dense(latent_dim),
                BatchNormalization(),
                Activation("sigmoid"),
            ]
        )

        self.decoder = tf.keras.Sequential(
            [
                Dense(8 * 8 * 128),
                BatchNormalization(),
                Activation('relu'),
                Reshape((8, 8, 128)),
                Conv2DTranspose(128, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Conv2DTranspose(64, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Conv2DTranspose(32, (3, 3), 2, padding="same"),
                BatchNormalization(),
                Activation('relu'),
                Conv2DTranspose(3, (3, 3), 1, padding="same"),
                Activation("sigmoid")
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


def load_data(data_dir, image_size=(64, 64)):
    images = []
    image_names = os.listdir(data_dir)
    for image_name in image_names:
        if image_name.lower().endswith((".jpg", ".png", ".jpeg")):
            image = cv2.imread(os.path.join(data_dir, image_name))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = preprocess_image(image, image_size=image_size[:2])
            images.append(image)
    return np.array(images)


def preprocess_image(image, image_size=(64, 64)):
    image = cv2.resize(image, image_size)
    image = image.astype("uint8") / 255.0
    return image


def get_a_single_image_embedding(autoencoder, image):
    """Load and encode a single image."""
    image = preprocess_image(image)
    encoded_image = autoencoder.encode(image[None, ...]).numpy()[0]
    return encoded_image


def dssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, 1.0)) / 2


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
    input_shape,
    latent_dim,
    batch_size,
    epochs=200,
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
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.1,
            patience=2,
            verbose=1,
            mode='auto',
            min_delta=0.0001,
            cooldown=0,
            min_lr=0
        ),
    ]

    # Train the model
    autoencoder.fit(
        x=X_train,
        y=X_train,
        epochs=epochs,
        shuffle=True,
        batch_size=batch_size,
        validation_data=(X_val, X_val),
        callbacks=callbacks,
    )
    return autoencoder


if __name__ == "__main__":

    input_shape = (64, 64, 3)
    latent_dim = 50
    batch_size = 256

    experiment_name = '3colors_2shapes'

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
            model_path=model_path,
        )

    # Test the model with a sample image
    for image_path in [
        "receiver_8567.png",
        "receiver_8607.png",
        "receiver_193.png",
        "receiver_1635.png",
        "receiver_2019.png",
    ]:
        test_image_path = dataset_dir / image_path 
        image = cv2.imread(str(test_image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plot_original_and_reconstructed(autoencoder, image)
        embedding = get_a_single_image_embedding(autoencoder, image)
        print('Embedding:', embedding)
