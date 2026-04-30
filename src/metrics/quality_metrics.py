"""
Метрики качества для сравнения результатов.
"""

from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity


def _to_float_image(image: np.ndarray) -> np.ndarray:
    """Приводит вход к float64 и проверяет размерность."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim not in (2, 3):
        raise ValueError("Ожидается изображение вида HxW или HxWxC.")
    return arr


def mse(original: np.ndarray, restored: np.ndarray) -> float:
    """Mean Squared Error."""
    a = _to_float_image(original)
    b = _to_float_image(restored)
    if a.shape != b.shape:
        raise ValueError("Изображения должны быть одного размера.")
    return float(np.mean((a - b) ** 2))


def psnr(original: np.ndarray, restored: np.ndarray, data_range: float = 255.0) -> float:
    """Peak Signal-to-Noise Ratio в dB."""
    err = mse(original, restored)
    if err == 0.0:
        return float("inf")
    return float(10.0 * np.log10((data_range**2) / err))


def ssim_rgb(original: np.ndarray, restored: np.ndarray, data_range: float = 255.0) -> float:
    """SSIM с учетом каналов (для RGB — единое значение)."""
    a = _to_float_image(original)
    b = _to_float_image(restored)
    if a.shape != b.shape:
        raise ValueError("Изображения должны быть одного размера.")
    if a.ndim == 2:
        return float(structural_similarity(a, b, data_range=data_range))
    return float(structural_similarity(a, b, data_range=data_range, channel_axis=2))


def ssim_channels_mean(original: np.ndarray, restored: np.ndarray, data_range: float = 255.0) -> float:
    """SSIM по каналам с усреднением (R/G/B или любой C)."""
    a = _to_float_image(original)
    b = _to_float_image(restored)
    if a.shape != b.shape:
        raise ValueError("Изображения должны быть одного размера.")
    if a.ndim == 2:
        return float(structural_similarity(a, b, data_range=data_range))

    per_channel_scores = [
        float(structural_similarity(a[:, :, ch], b[:, :, ch], data_range=data_range))
        for ch in range(a.shape[2])
    ]
    return float(np.mean(per_channel_scores))


def ssim(original: np.ndarray, restored: np.ndarray, data_range: float = 255.0) -> float:
    """Обратная совместимость: возвращает channel-aware SSIM."""
    return ssim_rgb(original, restored, data_range=data_range)


