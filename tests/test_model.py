import unittest

try:
    import tensorflow as tf
    from src.config import ViTConfig
    from src.data import build_augmentation
    from src.model import create_vit_classifier
except Exception:
    tf = None

@unittest.skipIf(tf is None, "TensorFlow is not installed")
class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        cfg = ViTConfig()
        x = tf.zeros((8, 32, 32, 3), dtype=tf.float32)
        aug = build_augmentation(x, cfg)
        model = create_vit_classifier(aug, cfg)
        y = model(x, training=False)
        self.assertEqual(tuple(y.shape), (8, 10))

if __name__ == "__main__":
    unittest.main()
