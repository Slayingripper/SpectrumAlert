"""End-to-end smoke test for the synthetic IQ fixture generator.

The generator is the supported way to exercise the pipeline without an SDR, so
its own contract is worth pinning: clean captures must not alert, the four
signal scenarios must, and the files it writes must load back through the same
storage layer the dashboard uses.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from spectrum_alert.infrastructure.storage import DataStorage

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "generate_iq_samples.py"


def run_generator(tmp_path: Path) -> dict:
    data_dir = tmp_path / "data"
    raw_dir = tmp_path / "raw"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--data-dir", str(data_dir),
            "--raw-dir", str(raw_dir),
            "--baseline", "6",
            "--samples", "8192",
        ],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr + result.stdout
    return json.loads((data_dir / "iq_fixture_manifest.json").read_text())


def test_generator_detects_expected_signals_and_ignores_clean_captures(tmp_path):
    manifest = run_generator(tmp_path)

    assert manifest["failures"] == []
    assert len(manifest["captures"]) == 11  # 6 baseline + 5 scenarios
    assert manifest["expected_anomalies"] == 4
    assert manifest["detected_anomalies"] == 4

    clean = [c for c in manifest["captures"] if not c["expected_anomaly"]]
    assert clean, "generator produced no negative controls"
    assert all(not c["detected"] for c in clean)

    detected = [c for c in manifest["captures"] if c["expected_anomaly"]]
    assert all(c["detected"] for c in detected)
    for capture in detected:
        assert capture["anomalies"][0]["score"] >= 0.7
        assert capture["anomalies"][0]["severity"] in {"low", "medium", "high", "critical"}


def test_generated_files_load_through_the_real_storage(tmp_path):
    run_generator(tmp_path)
    data_dir = tmp_path / "data"

    storage = DataStorage(str(data_dir))
    spectrum_files = storage.get_spectrum_files()
    assert len(spectrum_files) == 11

    loaded = storage.load_spectrum_data(spectrum_files[0])
    assert loaded.samples
    assert loaded.frequency_hz == 145e6
    assert loaded.sample_rate_hz == 2.048e6

    full = storage.load_training_data(mode="full")
    lite = storage.load_training_data(mode="lite")
    assert len(full) == 11 and len(lite) == 11
    assert {"mean_amplitude", "spectral_entropy", "papr_db"} <= set(full.columns)
    assert {"mean_amplitude", "std_amplitude"} <= set(lite.columns)

    anomalies = storage.get_recent_anomalies(limit=20)
    assert len(anomalies) == 4
    assert all(a.frequency_hz > 0 for a in anomalies)

    raw_files = sorted((tmp_path / "raw").glob("*.npy"))
    assert len(raw_files) == 11
    iq = np.load(raw_files[0])
    assert iq.dtype == np.complex64
    assert iq.shape == (8192,)
