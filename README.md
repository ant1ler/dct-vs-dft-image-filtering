# DCT vs DFT Image Filtering

## Цели

- Сделать **ручные** реализации DFT/FFT и DCT (без `numpy.fft`, `scipy.fft`, `cv2.dct` и т.п.).
- Применять частотные фильтры к **RGB** изображениям через DFT и через DCT.
- Сравнить результаты по метрикам (MSE/PSNR/SSIM) и визуально.

## Структура проекта (кратко)

- `src/` — исходный код:
  - `transforms/` — ручные реализации DFT/FFT и DCT,
  - `filters/` — частотные фильтры и модели шума,
  - `metrics/` — метрики качества,
  - `experiments/` — сценарии запуска экспериментов,
  - `visualization.py` — функции визуализации.
- `data/` — изображения (сырые, зашумлённые, результаты).
- `reports/` — рисунки и таблицы для диплома.
- `tests/` — базовые тесты корректности реализаций.

Полная “карта” проекта: `PROJECT_STRUCTURE.md`.

## Быстрый старт

Пока проект в процессе, но базовая установка такая:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Про данные

В данном репозитории будет **храниться небольшой набор** исходных изображений и **результаты** для него.

- `data/raw/` — исходные **RGB** изображения (маленький набор для экспериментов).
- `data/noisy/` — зашумлённые/промежуточные файлы (можно пересоздавать, если нужно).
- `data/results/` — результаты фильтрации.
- `reports/` — финальные картинки/таблицы, которые удобно вставлять в диплом.


## Формат входных данных

- Режим `paired`:
  - `data/raw/xxx.png` — чистое RGB изображение;
  - `data/noisy/xxx.png` — соответствующее RGB зашумленное изображение (то же имя файла).
- Режим `synthetic`:
  - используются изображения из `data/raw/`,
  - шум генерируется в коде (Gaussian и Salt&Pepper).

## Запуск экспериментов

Из корня проекта:

```bash
python -m src.experiments.compare_dct_dft --mode paired
python -m src.experiments.compare_dct_dft --mode synthetic
```

Полезные параметры:

- `--cutoff-ratio` — насколько агрессивно режем высокие частоты;
- `--gaussian-sigma` — сила гауссова шума (для `synthetic`);
- `--salt-pepper-amount` — доля испорченных пикселей (для `synthetic`).
- `--gaussian-sigmas` — список уровней Gaussian через запятую (например `10,20,30`);
- `--salt-pepper-amounts` — список уровней Salt&Pepper (например `0.01,0.03,0.05`).

Выход:

- CSV с метриками: `reports/tables/metrics_<mode>.csv`;
- усреднение по методу: `reports/tables/metrics_<mode>_summary.csv`;
- картинки после фильтрации: `data/results/<mode>/...`;
- превью-сравнения: `reports/figures/<mode>/comparisons/...`.

Дополнительно после каждого запуска автоматически строятся:

- `reports/figures/<mode>/distributions/` — boxplot/violin по метрикам;
- `reports/figures/synthetic/noise_curves/` — графики метрика vs уровень шума;
- `reports/figures/<mode>/error_maps/<image>/` — карты абсолютной ошибки DFT/DCT;
- `reports/figures/<mode>/spectra/<image>/` — DFT-спектры и DCT heatmap до/после фильтрации.

Колонки в `metrics_<mode>.csv`:

- `mse`, `psnr`
- `ssim_rgb` — SSIM как единая RGB метрика (`channel-aware`)
- `ssim_channels_mean` — средний SSIM по каналам R/G/B
- `noise_level` — уровень шума (для synthetic; для paired пусто)

