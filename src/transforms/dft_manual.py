"""
Ручные реализации DFT/FFT.
"""

import numpy as np


def dft_1d(x: np.ndarray) -> np.ndarray:
    """
    Прямое DFT по формуле.

    x: 1D массив (real/complex)
    return: 1D спектр (complex)
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    exponent = -2j * np.pi * k * n / N
    W = np.exp(exponent)
    return W @ x


def idft_1d(X: np.ndarray) -> np.ndarray:
    """
    Обратное DFT по формуле.
    """
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    exponent = 2j * np.pi * k * n / N
    W_inv = np.exp(exponent) / N
    return W_inv @ X


def fft_1d(x: np.ndarray) -> np.ndarray:
    """
    FFT (Кули–Тьюки), нормальная скорость только для N = 2^m.

    Если N нечётное, пока просто падаем обратно на dft_1d (медленно, но работает).
    """
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N == 1:
        return x.copy()
    if N % 2 != 0:
        return dft_1d(x)

    X_even = fft_1d(x[0::2])
    X_odd = fft_1d(x[1::2])

    k = np.arange(N)
    twiddle = np.exp(-2j * np.pi * k / N)

    X = np.zeros(N, dtype=complex)
    half = N // 2
    X[:half] = X_even + twiddle[:half] * X_odd
    X[half:] = X_even + twiddle[half:] * X_odd
    return X


def ifft_1d(X: np.ndarray) -> np.ndarray:
    """
    Обратная FFT.

    Делаем через сопряжение: ifft(X) = conj(fft(conj(X))) / N.
    """
    X = np.asarray(X, dtype=complex)
    N = X.shape[0]
    x_conj = np.conjugate(X)
    time_domain = fft_1d(x_conj)
    return np.conjugate(time_domain) / N


def dft_2d(x: np.ndarray) -> np.ndarray:
    """
    2D DFT: сначала по строкам, потом по столбцам.
    """
    x = np.asarray(x, dtype=complex)
    temp = np.apply_along_axis(dft_1d, axis=1, arr=x)
    X = np.apply_along_axis(dft_1d, axis=0, arr=temp)
    return X


def idft_2d(X: np.ndarray) -> np.ndarray:
    """
    2D IDFT: сначала по строкам, потом по столбцам.
    """
    X = np.asarray(X, dtype=complex)
    temp = np.apply_along_axis(idft_1d, axis=1, arr=X)
    x = np.apply_along_axis(idft_1d, axis=0, arr=temp)
    return x


def fft_2d(x: np.ndarray) -> np.ndarray:
    """
    2D FFT: сначала по строкам, потом по столбцам.
    """
    x = np.asarray(x, dtype=complex)
    temp = np.apply_along_axis(fft_1d, axis=1, arr=x)
    X = np.apply_along_axis(fft_1d, axis=0, arr=temp)
    return X


def ifft_2d(X: np.ndarray) -> np.ndarray:
    """
    2D IFFT: сначала по строкам, потом по столбцам.
    """
    X = np.asarray(X, dtype=complex)
    temp = np.apply_along_axis(ifft_1d, axis=1, arr=X)
    x = np.apply_along_axis(ifft_1d, axis=0, arr=temp)
    return x

