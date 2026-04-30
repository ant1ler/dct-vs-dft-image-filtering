"""
Модели шума для синтетических экспериментов.
"""

from __future__ import annotations

import numpy as np


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def add_gaussian_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Добавляет гауссов шум N(0, sigma^2)."""
    img = np.asarray(image, dtype=np.float64)
    noise = _rng(seed).normal(loc=0.0, scale=sigma, size=img.shape)
    return np.clip(np.rint(img + noise), 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image: np.ndarray, amount: float, seed: int | None = None) -> np.ndarray:
    """Добавляет salt & pepper шум. amount = доля испорченных пикселей."""
    if not (0.0 <= amount <= 1.0):
        raise ValueError("amount должен быть в диапазоне [0, 1].")

    rng = _rng(seed)
    out = np.asarray(image, dtype=np.uint8).copy()
    if out.ndim == 2:
        mask = rng.random(out.shape)
    elif out.ndim == 3:
        mask = rng.random(out.shape[:2])
    else:
        raise ValueError("Ожидается изображение HxW или HxWxC.")
    salt_threshold = amount / 2.0
    pepper_threshold = amount

    out[mask < salt_threshold] = 255
    out[(mask >= salt_threshold) & (mask < pepper_threshold)] = 0
    return out

