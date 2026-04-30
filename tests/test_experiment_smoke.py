import subprocess
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np


class TestExperimentSmoke(unittest.TestCase):
    def test_cli_runs_paired_and_synthetic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw_dir = root / "raw"
            noisy_dir = root / "noisy"
            results_dir = root / "results"
            tables_dir = root / "tables"
            figures_dir = root / "figures"
            raw_dir.mkdir()
            noisy_dir.mkdir()

            base = np.array(
                [
                    [10, 20, 30, 40, 50, 60, 70, 80],
                    [80, 70, 60, 50, 40, 30, 20, 10],
                    [10, 30, 50, 70, 90, 110, 130, 150],
                    [150, 130, 110, 90, 70, 50, 30, 10],
                    [0, 20, 40, 60, 80, 100, 120, 140],
                    [140, 120, 100, 80, 60, 40, 20, 0],
                    [15, 35, 55, 75, 95, 115, 135, 155],
                    [155, 135, 115, 95, 75, 55, 35, 15],
                ],
                dtype=np.uint8,
            )
            clean_rgb = np.stack([base, np.flipud(base), np.fliplr(base)], axis=2)
            noisy_rgb = np.clip(clean_rgb.astype(np.int16) + 10, 0, 255).astype(np.uint8)
            cv2.imwrite(str(raw_dir / "sample.png"), cv2.cvtColor(clean_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(noisy_dir / "sample.png"), cv2.cvtColor(noisy_rgb, cv2.COLOR_RGB2BGR))

            cmd_base = [
                "python",
                "-m",
                "src.experiments.compare_dct_dft",
                "--raw-dir",
                str(raw_dir),
                "--noisy-dir",
                str(noisy_dir),
                "--results-dir",
                str(results_dir),
                "--tables-dir",
                str(tables_dir),
                "--figures-dir",
                str(figures_dir),
                "--cutoff-ratio",
                "0.25",
            ]

            paired = subprocess.run(cmd_base + ["--mode", "paired"], capture_output=True, text=True)
            self.assertEqual(paired.returncode, 0, msg=paired.stderr)
            self.assertTrue((tables_dir / "metrics_paired.csv").exists())

            synthetic = subprocess.run(
                cmd_base + ["--mode", "synthetic", "--gaussian-sigma", "10", "--salt-pepper-amount", "0.05"],
                capture_output=True,
                text=True,
            )
            self.assertEqual(synthetic.returncode, 0, msg=synthetic.stderr)
            self.assertTrue((tables_dir / "metrics_synthetic.csv").exists())


if __name__ == "__main__":
    unittest.main()

