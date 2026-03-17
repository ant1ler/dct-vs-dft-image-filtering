"""
Частотные фильтры в виде масок.
"""

import numpy as np


def ideal_low_pass(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Идеальный НЧ-фильтр (круг).

    shape: (H, W)
    cutoff: радиус в пикселях (в координатах частотной матрицы)
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance = np.sqrt((Y - crow) ** 2 + (X - ccol) ** 2)
    return (distance <= cutoff).astype(float)


def ideal_high_pass(shape: tuple[int, int], cutoff: float) -> np.ndarray:
    """
    Идеальный ВЧ-фильтр = 1 - НЧ.
    """
    return 1.0 - ideal_low_pass(shape, cutoff)


def gaussian_low_pass(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """
    Гауссов НЧ-фильтр (sigma тоже в пикселях частотной матрицы).
    """
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    distance2 = (Y - crow) ** 2 + (X - ccol) ** 2
    H = np.exp(-distance2 / (2 * sigma * sigma))
    return H


def gaussian_high_pass(shape: tuple[int, int], sigma: float) -> np.ndarray:
    """
    Гауссов ВЧ-фильтр = 1 - НЧ.
    """
    return 1.0 - gaussian_low_pass(shape, sigma)

