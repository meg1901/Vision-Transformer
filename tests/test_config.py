import unittest
from src.config import ViTConfig

class ConfigTests(unittest.TestCase):
    def test_patch_count(self):
        self.assertEqual(ViTConfig().num_patches, 144)

if __name__ == "__main__":
    unittest.main()
