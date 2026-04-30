import unittest

import numpy as np

from src.metrics.quality_metrics import mse, psnr, ssim_channels_mean, ssim_rgb


class TestMetrics(unittest.TestCase):
    def test_identical_images(self) -> None:
        img = np.full((8, 8, 3), 120, dtype=np.uint8)
        self.assertEqual(mse(img, img), 0.0)
        self.assertEqual(psnr(img, img), float("inf"))
        self.assertAlmostEqual(ssim_rgb(img, img), 1.0, places=6)
        self.assertAlmostEqual(ssim_channels_mean(img, img), 1.0, places=6)

    def test_metrics_change_on_difference(self) -> None:
        a = np.zeros((8, 8, 3), dtype=np.uint8)
        b = np.full((8, 8, 3), 20, dtype=np.uint8)
        self.assertGreater(mse(a, b), 0.0)
        self.assertLess(psnr(a, b), 30.0)
        self.assertLess(ssim_rgb(a, b), 1.0)
        self.assertLess(ssim_channels_mean(a, b), 1.0)


if __name__ == "__main__":
    unittest.main()

