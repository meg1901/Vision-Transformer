from tensorflow import keras
from tensorflow.keras import layers

from .config import ViTConfig

def load_cifar10():
    return keras.datasets.cifar10.load_data()

def build_augmentation(x_train, config: ViTConfig):
    augmentation = keras.Sequential(
        [
            layers.Normalization(),
            layers.Resizing(config.image_size, config.image_size),
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(factor=0.02),
            layers.RandomZoom(height_factor=0.2, width_factor=0.2),
        ],
        name="data_augmentation",
    )
    augmentation.layers[0].adapt(x_train)
    return augmentation
