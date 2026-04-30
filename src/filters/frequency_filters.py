"""
Частотные фильтры в виде масок.
"""

import numpy as np


def gaussian_low_pass(shape: tuple[int, int], sigma_ratio: float) -> np.ndarray:
    """
    Универсальный Gaussian LPF.
    sigma_ratio ∈ (0, 1] — доля от размера изображения.
    """
    rows, cols = shape

    # центр спектра (для FFT-области)
    crow, ccol = rows // 2, cols // 2

    # переводим относительный sigma в пиксели
    sigma = sigma_ratio * min(rows, cols)

    Y, X = np.ogrid[:rows, :cols]
    distance2 = (Y - crow) ** 2 + (X - ccol) ** 2

    return np.exp(-distance2 / (2 * sigma * sigma))


def gaussian_high_pass(shape: tuple[int, int], sigma_ratio: float) -> np.ndarray:
    """
    Gaussian HPF.
    """
    return 1.0 - gaussian_low_pass(shape, sigma_ratio)

def gaussian_low_pass_dct(shape: tuple[int, int], sigma_ratio: float) -> np.ndarray:
    """
    Gaussian LPF для DCT (энергия в (0,0)).
    """
    rows, cols = shape
    sigma = sigma_ratio * min(rows, cols)

    Y, X = np.ogrid[:rows, :cols]

    # для DCT низкие частоты в углу (0,0)
    distance2 = Y**2 + X**2

    return np.exp(-distance2 / (2 * sigma * sigma))
