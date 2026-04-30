"""
Сценарий сравнения фильтрации через DCT и через DFT/FFT.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.filters.noise_models import add_gaussian_noise, add_salt_pepper_noise
from src.metrics.quality_metrics import mse, psnr, ssim_channels_mean, ssim_rgb
from src.transforms.dct_manual import dct_2d, idct_2d
from src.transforms.dft_manual import fft_2d, ifft_2d
from src.visualization import (
    plot_metric_distributions,
    plot_metric_violins,
    plot_noise_curves,
    save_dct_heatmap,
    save_dft_spectrum,
    save_error_map,
)


@dataclass
class ExperimentItem:
    name: str
    clean: np.ndarray
    noisy: np.ndarray
    noise_label: str
    noise_level: float | None = None


def _load_rgb_uint8(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _clip_to_uint8(arr: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(arr), 0, 255).astype(np.uint8)


def _fftshift_manual(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    return np.roll(np.roll(x, rows // 2, axis=0), cols // 2, axis=1)


def _ifftshift_manual(x: np.ndarray) -> np.ndarray:
    rows, cols = x.shape
    return np.roll(np.roll(x, -(rows // 2), axis=0), -(cols // 2), axis=1)


def _center_rect_mask(shape: tuple[int, int], cutoff_ratio: float) -> np.ndarray:
    rows, cols = shape
    keep_rows = max(1, int(rows * cutoff_ratio))
    keep_cols = max(1, int(cols * cutoff_ratio))
    mask = np.zeros((rows, cols), dtype=np.float64)
    r0 = (rows - keep_rows) // 2
    c0 = (cols - keep_cols) // 2
    mask[r0 : r0 + keep_rows, c0 : c0 + keep_cols] = 1.0
    return mask


def _filter_dft_gaussian_lowpass(noisy: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    if noisy.ndim != 3 or noisy.shape[2] != 3:
        raise ValueError("Ожидается RGB изображение HxWx3.")

    rows, cols, _ = noisy.shape

    keep_rows = max(1, int(rows * cutoff_ratio))
    keep_cols = max(1, int(cols * cutoff_ratio))

    mask = np.zeros((rows, cols), dtype=np.float64)

    r0 = (rows - keep_rows) // 2
    c0 = (cols - keep_cols) // 2

    mask[r0:r0 + keep_rows, c0:c0 + keep_cols] = 1.0

    out = np.zeros_like(noisy, dtype=np.float64)

    for ch in range(3):
        channel = noisy[:, :, ch].astype(np.float64)

        spectrum = fft_2d(channel)

        # центрируем спектр
        spectrum = np.fft.fftshift(spectrum)

        filtered = spectrum * mask

        spectrum = np.fft.ifftshift(filtered)

        out[:, :, ch] = np.real(ifft_2d(spectrum))

    return _clip_to_uint8(out)

def _filter_dct_lowpass(noisy: np.ndarray, cutoff_ratio: float) -> np.ndarray:
    if noisy.ndim != 3 or noisy.shape[2] != 3:
        raise ValueError("Ожидается RGB изображение HxWx3.")

    rows, cols, _ = noisy.shape

    keep_rows = max(1, int(rows * cutoff_ratio))
    keep_cols = max(1, int(cols * cutoff_ratio))

    # DCT: НИЗКИЕ ЧАСТОТЫ В ЛЕВОМ ВЕРХНЕМ УГЛУ
    mask = np.zeros((rows, cols), dtype=np.float64)
    mask[:keep_rows, :keep_cols] = 1.0

    out = np.zeros_like(noisy, dtype=np.float64)

    for ch in range(3):
        coeffs = dct_2d(noisy[:, :, ch].astype(np.float64))
        out[:, :, ch] = idct_2d(coeffs * mask)

    return _clip_to_uint8(out)


def _save_preview(path: Path, clean: np.ndarray, noisy: np.ndarray, dft_img: np.ndarray, dct_img: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    images = [clean, noisy, dft_img, dct_img]
    titles = ["Clean", "Noisy", "DFT filtered", "DCT filtered"]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figures_subdirs(figures_dir: Path, mode: str) -> dict[str, Path]:
    base = figures_dir / mode
    return {
        "base": base,
        "comparisons": base / "comparisons",
        "distributions": base / "distributions",
        "noise_curves": base / "noise_curves",
        "error_maps": base / "error_maps",
        "spectra": base / "spectra",
    }


def _collect_paired_items(raw_dir: Path, noisy_dir: Path) -> list[ExperimentItem]:
    items: list[ExperimentItem] = []
    for clean_path in sorted(raw_dir.glob("*")):
        if clean_path.is_dir():
            continue
        noisy_path = noisy_dir / clean_path.name
        if not noisy_path.exists():
            continue
        clean = _load_rgb_uint8(clean_path)
        noisy = _load_rgb_uint8(noisy_path)
        if clean.shape != noisy.shape:
            continue
        items.append(ExperimentItem(name=clean_path.stem, clean=clean, noisy=noisy, noise_label="dataset_paired"))
    return items


def _collect_synthetic_items(
    raw_dir: Path,
    gaussian_sigmas: Iterable[float],
    sp_amounts: Iterable[float],
    seed: int,
) -> list[ExperimentItem]:
    items: list[ExperimentItem] = []
    for index, clean_path in enumerate(sorted(raw_dir.glob("*"))):
        if clean_path.is_dir():
            continue
        clean = _load_rgb_uint8(clean_path)
        for sigma_idx, sigma in enumerate(gaussian_sigmas):
            rng_seed = seed + index * 100 + sigma_idx * 10
            noisy_gauss = add_gaussian_noise(clean, sigma=float(sigma), seed=rng_seed)
            items.append(
                ExperimentItem(
                    name=f"{clean_path.stem}_gauss_{float(sigma):g}",
                    clean=clean,
                    noisy=noisy_gauss,
                    noise_label="gaussian",
                    noise_level=float(sigma),
                )
            )
        for sp_idx, amount in enumerate(sp_amounts):
            rng_seed = seed + index * 100 + 50 + sp_idx * 10
            noisy_sp = add_salt_pepper_noise(clean, amount=float(amount), seed=rng_seed)
            items.append(
                ExperimentItem(
                    name=f"{clean_path.stem}_saltpepper_{float(amount):g}",
                    clean=clean,
                    noisy=noisy_sp,
                    noise_label="salt_pepper",
                    noise_level=float(amount),
                )
            )
    return items


def _run_items(
    items: list[ExperimentItem],
    mode: str,
    cutoff_ratio: float,
    results_dir: Path,
    figures_dir: Path,
) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    fig_dirs = _figures_subdirs(figures_dir, mode)
    for item in items:
        dft_img = _filter_dft_gaussian_lowpass(item.noisy, cutoff_ratio=cutoff_ratio)
        dct_img = _filter_dct_lowpass(item.noisy, cutoff_ratio=cutoff_ratio)

        item_dir = results_dir / mode / item.name
        item_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(item_dir / "clean.png"), cv2.cvtColor(item.clean, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(item_dir / "noisy.png"), cv2.cvtColor(item.noisy, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(item_dir / "dft_filtered.png"), cv2.cvtColor(dft_img, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(item_dir / "dct_filtered.png"), cv2.cvtColor(dct_img, cv2.COLOR_RGB2BGR))
        _save_preview(fig_dirs["comparisons"] / f"{item.name}_comparison.png", item.clean, item.noisy, dft_img, dct_img)

        err_dir = fig_dirs["error_maps"] / item.name
        save_error_map(item.clean, dft_img, err_dir / "dft_error.png")
        save_error_map(item.clean, dct_img, err_dir / "dct_error.png")

        spectra_dir = fig_dirs["spectra"] / item.name
        save_dft_spectrum(item.noisy, spectra_dir / "noisy_dft_spectrum.png")
        save_dft_spectrum(dft_img, spectra_dir / "dft_filtered_spectrum.png")
        save_dct_heatmap(item.noisy, spectra_dir / "noisy_dct_heatmap.png")
        save_dct_heatmap(dct_img, spectra_dir / "dct_filtered_heatmap.png")

        for method, image in [("dft", dft_img), ("dct", dct_img)]:
            rows.append(
                {
                    "mode": mode,
                    "image": item.name,
                    "noise": item.noise_label,
                    "noise_level": "" if item.noise_level is None else item.noise_level,
                    "method": method,
                    "mse": mse(item.clean, image),
                    "psnr": psnr(item.clean, image),
                    "ssim_rgb": ssim_rgb(item.clean, image),
                    "ssim_channels_mean": ssim_channels_mean(item.clean, image),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, str | float]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _aggregate_means(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    grouped: dict[tuple[str, str], list[dict[str, str | float]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["method"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, str | float]] = []
    for (mode, method), group_rows in grouped.items():
        summary.append(
            {
                "mode": mode,
                "method": method,
                "mean_mse": float(np.mean([float(x["mse"]) for x in group_rows])),
                "mean_psnr": float(np.mean([float(x["psnr"]) for x in group_rows])),
                "mean_ssim_rgb": float(np.mean([float(x["ssim_rgb"]) for x in group_rows])),
                "mean_ssim_channels_mean": float(np.mean([float(x["ssim_channels_mean"]) for x in group_rows])),
                "count": len(group_rows),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Сравнение фильтрации DCT vs DFT.")
    parser.add_argument("--mode", choices=["paired", "synthetic"], required=True)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--noisy-dir", type=Path, default=Path("data/noisy"))
    parser.add_argument("--results-dir", type=Path, default=Path("data/results"))
    parser.add_argument("--tables-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--figures-dir", type=Path, default=Path("reports/figures"))
    parser.add_argument("--cutoff-ratio", type=float, default=0.25)
    parser.add_argument("--gaussian-sigma", type=float, default=15.0)
    parser.add_argument("--salt-pepper-amount", type=float, default=0.03)
    parser.add_argument("--gaussian-sigmas", type=str, default="")
    parser.add_argument("--salt-pepper-amounts", type=str, default="")
    parser.add_argument("--max-spectra", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _parse_float_list(value: str, fallback: list[float]) -> list[float]:
    if not value.strip():
        return fallback
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def generate_report_artifacts(mode: str, tables_dir: Path, figures_dir: Path) -> None:
    metrics_csv = tables_dir / f"metrics_{mode}.csv"
    fig_dirs = _figures_subdirs(figures_dir, mode)
    plot_metric_distributions(metrics_csv, fig_dirs["distributions"])
    plot_metric_violins(metrics_csv, fig_dirs["distributions"])
    if mode == "synthetic":
        plot_noise_curves(metrics_csv, fig_dirs["noise_curves"])


def main() -> None:
    args = parse_args()

    if args.mode == "paired":
        items = _collect_paired_items(args.raw_dir, args.noisy_dir)
    else:
        gaussian_sigmas = _parse_float_list(args.gaussian_sigmas, [args.gaussian_sigma, args.gaussian_sigma * 1.5])
        sp_amounts = _parse_float_list(args.salt_pepper_amounts, [args.salt_pepper_amount, args.salt_pepper_amount * 1.5])
        items = _collect_synthetic_items(
            args.raw_dir,
            gaussian_sigmas=gaussian_sigmas,
            sp_amounts=sp_amounts,
            seed=args.seed,
        )

    if not items:
        raise RuntimeError("Не нашлось входных изображений для выбранного режима.")

    rows = _run_items(
        items=items,
        mode=args.mode,
        cutoff_ratio=args.cutoff_ratio,
        results_dir=args.results_dir,
        figures_dir=args.figures_dir,
    )
    summary = _aggregate_means(rows)

    _write_csv(
        args.tables_dir / f"metrics_{args.mode}.csv",
        rows,
        ["mode", "image", "noise", "noise_level", "method", "mse", "psnr", "ssim_rgb", "ssim_channels_mean"],
    )
    _write_csv(
        args.tables_dir / f"metrics_{args.mode}_summary.csv",
        summary,
        ["mode", "method", "mean_mse", "mean_psnr", "mean_ssim_rgb", "mean_ssim_channels_mean", "count"],
    )
    generate_report_artifacts(args.mode, args.tables_dir, args.figures_dir)

    print(f"Done. Processed items: {len(items)}")
    print(f"Detailed CSV: {args.tables_dir / f'metrics_{args.mode}.csv'}")
    print(f"Summary CSV: {args.tables_dir / f'metrics_{args.mode}_summary.csv'}")


if __name__ == "__main__":
    main()


