import unittest

import numpy as np

from src.transforms.dct_manual import dct_2d, idct_2d
from src.transforms.dft_manual import fft_2d, ifft_2d


class TestTransforms(unittest.TestCase):
    def test_fft_ifft_roundtrip(self) -> None:
        x = np.array([[1, 2, 3, 4], [4, 3, 2, 1], [1, 1, 2, 2], [0, 1, 0, 1]], dtype=float)
        spectrum = fft_2d(x)
        restored = np.real(ifft_2d(spectrum))
        self.assertTrue(np.allclose(x, restored, atol=1e-6))

    def test_dct_idct_roundtrip(self) -> None:
        x = np.array([[10, 20, 30, 40], [5, 10, 15, 20], [3, 6, 9, 12], [100, 80, 60, 40]], dtype=float)
        coeffs = dct_2d(x)
        restored = idct_2d(coeffs)
        self.assertTrue(np.allclose(x, restored, atol=1e-6))


if __name__ == "__main__":
    unittest.main()

