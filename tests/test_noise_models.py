import unittest

import numpy as np

from src.filters.noise_models import add_gaussian_noise, add_salt_pepper_noise


class TestNoiseModels(unittest.TestCase):
    def test_gaussian_noise_shape(self) -> None:
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        noisy = add_gaussian_noise(img, sigma=10.0, seed=1)
        self.assertEqual(noisy.shape, img.shape)
        self.assertEqual(noisy.dtype, np.uint8)

    def test_salt_pepper_noise_values_and_pixel_level(self) -> None:
        img = np.full((16, 16, 3), 100, dtype=np.uint8)
        noisy = add_salt_pepper_noise(img, amount=0.2, seed=1)
        self.assertEqual(noisy.shape, img.shape)
        self.assertTrue(np.any(noisy == 0) or np.any(noisy == 255))
        changed = np.any(noisy != 100, axis=2)
        if np.any(changed):
            pixels = noisy[changed]
            self.assertTrue(np.all((pixels == 0) | (pixels == 255)))


if __name__ == "__main__":
    unittest.main()

