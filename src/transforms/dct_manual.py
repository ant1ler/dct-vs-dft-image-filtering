"""
Ручная DCT (DCT-II) и обратное преобразование.
"""

import numpy as np


def _dct_matrix(n: int) -> np.ndarray:
    """
    Собирает матрицу DCT-II размера n x n (ортонормированную).
    """
    C = np.zeros((n, n), dtype=float)
    factor = np.pi / (2 * n)
    scale0 = np.sqrt(1.0 / n)
    scale = np.sqrt(2.0 / n)
    for k in range(n):
        alpha = scale0 if k == 0 else scale
        for i in range(n):
            C[k, i] = alpha * np.cos((2 * i + 1) * k * factor)
    return C


def dct_1d(x: np.ndarray) -> np.ndarray:
    """
    1D DCT-II.
    """
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    C = _dct_matrix(n)
    return C @ x


def idct_1d(X: np.ndarray) -> np.ndarray:
    """
    Обратное к DCT-II (по сути DCT-III для ортонормированного случая).
    """
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    C = _dct_matrix(n)
    # Для ортонормированной матрицы обратное — это транспонирование.
    return C.T @ X


def dct_2d(x: np.ndarray) -> np.ndarray:
    """
    2D DCT: по строкам, потом по столбцам.
    """
    x = np.asarray(x, dtype=float)
    temp = np.apply_along_axis(dct_1d, axis=1, arr=x)
    X = np.apply_along_axis(dct_1d, axis=0, arr=temp)
    return X


def idct_2d(X: np.ndarray) -> np.ndarray:
    """
    2D обратное DCT: по строкам, потом по столбцам.
    """
    X = np.asarray(X, dtype=float)
    temp = np.apply_along_axis(idct_1d, axis=1, arr=X)
    x = np.apply_along_axis(idct_1d, axis=0, arr=temp)
    return x

