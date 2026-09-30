import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .config import ViTConfig

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

class Patches(layers.Layer):
    def __init__(self, patch_size, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        return tf.reshape(patches, [batch_size, -1, patch_dims])

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"patch_size": self.patch_size})
        return cfg

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim, **kwargs):
        super().__init__(**kwargs)
        self.num_patches = num_patches
        self.projection_dim = projection_dim
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patches):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        return self.projection(patches) + self.position_embedding(positions)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(
            {"num_patches": self.num_patches, "projection_dim": self.projection_dim}
        )
        return cfg

def create_vit_classifier(augmentation, config: ViTConfig):
    inputs = layers.Input(shape=config.input_shape)
    augmented = augmentation(inputs)
    patches = Patches(config.patch_size)(augmented)
    encoded_patches = PatchEncoder(
        config.num_patches, config.projection_dim
    )(patches)

    for _ in range(config.transformer_layers):
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        attention_output = layers.MultiHeadAttention(
            num_heads=config.num_heads,
            key_dim=config.projection_dim,
            dropout=0.1,
        )(x1, x1)
        x2 = layers.Add()([attention_output, encoded_patches])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        x3 = mlp(
            x3,
            hidden_units=config.transformer_units,
            dropout_rate=0.1,
        )
        encoded_patches = layers.Add()([x3, x2])

    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)

    features = mlp(
        representation,
        hidden_units=config.mlp_head_units,
        dropout_rate=0.5,
    )
    logits = layers.Dense(config.num_classes)(features)
    return keras.Model(inputs=inputs, outputs=logits, name="vit_cifar10")
