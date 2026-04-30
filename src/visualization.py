"""
Всякие функции для визуализации: картинки, спектры, графики.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def plot_metric_distributions(metrics_csv: Path, output_dir: Path) -> None:
    """Boxplot по метрикам для DFT и DCT."""
    rows = _read_csv(metrics_csv)
    if not rows:
        return

    _ensure_dir(output_dir)
    metrics = ["mse", "psnr", "ssim_rgb", "ssim_channels_mean"]
    methods = ["dft", "dct"]

    for metric in metrics:
        data: list[list[float]] = []
        labels: list[str] = []
        for method in methods:
            values = [float(r[metric]) for r in rows if r.get("method") == method]
            if values:
                data.append(values)
                labels.append(method.upper())
        if not data:
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.boxplot(data, tick_labels=labels)
        ax.set_title(f"{metric} distribution")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"boxplot_{metric}.png", dpi=150)
        plt.close(fig)


def plot_metric_violins(metrics_csv: Path, output_dir: Path) -> None:
    """Violin plot по метрикам для DFT и DCT."""
    rows = _read_csv(metrics_csv)
    if not rows:
        return
    _ensure_dir(output_dir)
    metrics = ["mse", "psnr", "ssim_rgb", "ssim_channels_mean"]
    methods = ["dft", "dct"]

    for metric in metrics:
        data = []
        labels = []
        for method in methods:
            values = [float(r[metric]) for r in rows if r.get("method") == method]
            if values:
                data.append(values)
                labels.append(method.upper())
        if not data:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.violinplot(data, showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_title(f"{metric} violin")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / f"violin_{metric}.png", dpi=150)
        plt.close(fig)


def plot_noise_curves(metrics_csv: Path, output_dir: Path) -> None:
    """Line plot метрика vs noise_level (только если есть noise_level)."""
    rows = _read_csv(metrics_csv)
    if not rows or "noise_level" not in rows[0]:
        return
    _ensure_dir(output_dir)

    metrics = ["psnr", "ssim_rgb"]
    methods = ["dft", "dct"]
    noises = sorted({r["noise"] for r in rows})

    for noise_name in noises:
        for metric in metrics:
            fig, ax = plt.subplots(figsize=(6, 4))
            plotted = False
            for method in methods:
                subset = [
                    (float(r["noise_level"]), float(r[metric]))
                    for r in rows
                    if r["noise"] == noise_name and r["method"] == method and r.get("noise_level", "") != ""
                ]
                if not subset:
                    continue
                subset.sort(key=lambda x: x[0])
                levels = sorted({x[0] for x in subset})
                means = []
                for level in levels:
                    vals = [v for lv, v in subset if lv == level]
                    means.append(float(np.mean(vals)))
                ax.plot(levels, means, marker="o", label=method.upper())
                plotted = True

            if not plotted:
                plt.close(fig)
                continue
            ax.set_title(f"{noise_name}: {metric} vs noise_level")
            ax.set_xlabel("noise_level")
            ax.set_ylabel(metric)
            ax.grid(alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"curve_{noise_name}_{metric}.png", dpi=150)
            plt.close(fig)


def save_error_map(clean: np.ndarray, restored: np.ndarray, output_path: Path) -> None:
    """Сохраняет карту ошибки |restored-clean| по RGB (усреднение каналов)."""
    err = np.abs(restored.astype(np.float64) - clean.astype(np.float64))
    if err.ndim == 3:
        err = np.mean(err, axis=2)

    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(err, cmap="inferno")
    ax.set_title("Absolute error map")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dft_spectrum(rgb_image: np.ndarray, output_path: Path) -> None:
    """Амплитудный DFT-спектр по яркости."""
    from src.transforms.dft_manual import fft_2d

    gray = np.mean(rgb_image.astype(np.float64), axis=2)
    spec = fft_2d(gray)
    shifted = np.roll(np.roll(spec, gray.shape[0] // 2, axis=0), gray.shape[1] // 2, axis=1)
    amp = np.log1p(np.abs(shifted))

    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(amp, cmap="magma")
    ax.set_title("DFT amplitude spectrum")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_dct_heatmap(rgb_image: np.ndarray, output_path: Path) -> None:
    """Heatmap DCT-коэффициентов по яркости."""
    from src.transforms.dct_manual import dct_2d

    gray = np.mean(rgb_image.astype(np.float64), axis=2)
    coeffs = dct_2d(gray)
    vis = np.log1p(np.abs(coeffs))

    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(vis, cmap="viridis")
    ax.set_title("DCT coefficients heatmap")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


