from dataclasses import dataclass

@dataclass(frozen=True)
class ViTConfig:
    input_shape: tuple = (32, 32, 3)
    num_classes: int = 10
    image_size: int = 72
    patch_size: int = 6
    projection_dim: int = 64
    num_heads: int = 4
    transformer_layers: int = 8
    transformer_units: tuple = (128, 64)
    mlp_head_units: tuple = (2048, 1024)
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 40
    validation_split: float = 0.1
    seed: int = 42

    @property
    def num_patches(self) -> int:
        return (self.image_size // self.patch_size) ** 2
