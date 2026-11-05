#!/usr/bin/env python3
"""Generate deterministic synthetic IQ captures for an end-to-end test.

The point is to exercise the real pipeline without an SDR:

    IQ -> SpectrumData -> stored captures -> features (full + lite)
       -> anomaly detector -> anomaly CSV -> training data

The output is written in the same formats the application already uses, so the
dashboard, training endpoint and CLI read it without any special casing.

Examples
--------
Generate a self-contained test tree with both raw IQ and application data:

    python scripts/generate_iq_samples.py \
        --data-dir /tmp/spectrumalert-iq/data \
        --raw-dir  /tmp/spectrumalert-iq/raw

Generate into the live data directory (spectrum/features are gitignored):

    python scripts/generate_iq_samples.py --baseline 40

The script is deterministic for a given ``--seed``. It exits non-zero if the
detector misses an expected signal or fires on a clean capture, so it doubles
as a smoke test.
"""

from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

# Allow running as `python scripts/...` from a checkout without an install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from spectrum_alert.application.use_cases.anomaly_detection import (  # noqa: E402
    AnomalyDetectionUseCase,
)
from spectrum_alert.core.domain.models import (  # noqa: E402
    DetectionMode,
    FeatureVector,
    SpectrumData,
)
from spectrum_alert.core.services.feature_extraction import FeatureExtractor  # noqa: E402
from spectrum_alert.infrastructure.storage import DataStorage  # noqa: E402

NAMESPACE = uuid.UUID("6f1d9a1e-6c47-5d6d-9c2d-2f8f6f7a1b10")


def _noise(rng: np.random.Generator, n: int, sigma: float = 0.35) -> np.ndarray:
    return (rng.normal(0.0, sigma, n) + 1j * rng.normal(0.0, sigma, n)).astype(np.complex64)


def _cw(t: np.ndarray, offset_hz: float, amplitude: float) -> np.ndarray:
    return amplitude * np.exp(2j * np.pi * offset_hz * t)


def build_iq(kind: str, rng: np.random.Generator, n: int, sample_rate: float) -> np.ndarray:
    """Return one complex baseband capture for a named scenario."""
    t = np.arange(n) / sample_rate
    base = _noise(rng, n)

    if kind == "noise_floor":
        return base
    if kind == "cw_narrow":
        return (base + _cw(t, 300e3, 1.4)).astype(np.complex64)
    if kind == "two_tone":
        return (base + _cw(t, 150e3, 0.7) + _cw(t, 420e3, 0.7)).astype(np.complex64)
    if kind == "fsk":
        shift = 80e3 * np.sign(np.sin(2 * np.pi * 2e3 * t))
        return (base + _cw(t, 250e3 + shift, 1.2)).astype(np.complex64)
    if kind == "ask":
        keyed = (np.sign(np.sin(2 * np.pi * 8e3 * t)) > 0).astype(float)
        return (base + _cw(t, 320e3, 1.4) * keyed).astype(np.complex64)
    if kind == "wideband_reference":
        # Broadband energy has no narrow FFT peak, so the narrowband detector
        # should reject it. It is here as a negative control, not an anomaly.
        wide = 2.5 * (rng.normal(0, 1, n) + 1j * rng.normal(0, 1, n))
        return (base + wide).astype(np.complex64)
    raise ValueError(f"unknown scenario: {kind}")


SCENARIOS: list[tuple[str, bool]] = [
    ("cw_narrow", True),
    ("two_tone", True),
    ("fsk", True),
    ("ask", True),
    ("wideband_reference", False),
]


def generate(
    data_dir: str | Path,
    raw_dir: str | Path | None = None,
    baseline: int = 30,
    seed: int = 20261106,
    sample_rate: float = 2.048e6,
    center_freq: float = 145e6,
    samples: int = 8192,
) -> dict:
    """Generate captures and return a manifest describing what was written."""
    if baseline < 1:
        raise ValueError("baseline must be at least 1")
    if samples < 256:
        raise ValueError("samples must be at least 256")

    data_path = Path(data_dir)
    raw_path = Path(raw_dir) if raw_dir else None
    if raw_path:
        raw_path.mkdir(parents=True, exist_ok=True)

    storage = DataStorage(str(data_path), max_stored_samples=samples)
    full_extractor = FeatureExtractor(lite_mode=False)
    lite_extractor = FeatureExtractor(lite_mode=True)
    detector = AnomalyDetectionUseCase(FeatureExtractor(lite_mode=False))

    start = datetime(2026, 11, 6, 9, 0, 0)
    entries: list[dict] = []
    failures: list[str] = []
    counter = 0

    def emit(name: str, kind: str, expected: bool, rng: np.random.Generator) -> None:
        nonlocal counter
        timestamp = start + timedelta(seconds=counter * 8)
        counter += 1
        iq = build_iq(kind, rng, samples, sample_rate)
        capture_id = str(uuid.uuid5(NAMESPACE, f"{name}:{timestamp.isoformat()}"))

        data = SpectrumData(
            id=capture_id,
            timestamp=timestamp,
            frequency_hz=center_freq,
            sample_rate_hz=sample_rate,
            gain_db=20.0,
            samples=[complex(v) for v in iq],
            duration_seconds=samples / sample_rate,
        )
        spectrum_file = storage.save_spectrum_data(data)

        for extractor, mode in ((full_extractor, DetectionMode.FULL),
                                (lite_extractor, DetectionMode.LITE)):
            feature_set = extractor.extract_features(iq, center_freq)
            storage.save_feature_vector(
                FeatureVector(
                    id=str(uuid.uuid5(NAMESPACE, f"{name}:{mode.value}")),
                    spectrum_data_id=capture_id,
                    timestamp=timestamp,
                    mode=mode,
                    features=dict(zip(feature_set.feature_names, feature_set.features)),
                    feature_names=list(feature_set.feature_names),
                )
            )

        detections = detector.detect_anomalies(data)
        for anomaly in detections:
            storage.save_anomaly(anomaly)

        detected = bool(detections)
        if expected and not detected:
            failures.append(f"{name}: expected a detection, got none")
        if not expected and detected:
            frequencies = ", ".join(f"{a.frequency_hz / 1e6:.4f} MHz" for a in detections)
            failures.append(f"{name}: false positive at {frequencies}")

        raw_file = None
        if raw_path:
            raw_file = raw_path / f"{counter - 1:02d}_{name}.npy"
            np.save(raw_file, iq.astype(np.complex64))

        entries.append({
            "name": name,
            "kind": kind,
            "expected_anomaly": expected,
            "detected": detected,
            "frequency_hz": center_freq,
            "sample_rate_hz": sample_rate,
            "samples": samples,
            "spectrum_file": str(spectrum_file),
            "raw_iq_file": str(raw_file) if raw_file else None,
            "anomalies": [
                {
                    "frequency_hz": a.frequency_hz,
                    "score": a.confidence_score,
                    "severity": a.severity,
                    "description": a.description,
                }
                for a in detections
            ],
        })

    rng = np.random.default_rng(seed)
    for index in range(baseline):
        emit(f"noise_{index:02d}", "noise_floor", False, rng)
    for kind, expected in SCENARIOS:
        emit(kind, kind, expected, rng)

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "seed": seed,
        "data_dir": str(data_path.resolve()),
        "raw_dir": str(raw_path.resolve()) if raw_path else None,
        "baseline_captures": baseline,
        "captures": entries,
        "expected_anomalies": sum(e["expected_anomaly"] for e in entries),
        "detected_anomalies": sum(e["detected"] for e in entries),
        "failures": failures,
    }
    (data_path / "iq_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", default="data",
                        help="application data directory (default: data)")
    parser.add_argument("--raw-dir", default=None,
                        help="optional directory for inspectable .npy IQ captures")
    parser.add_argument("--baseline", type=int, default=30,
                        help="number of clean noise captures (default: 30)")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--sample-rate", type=float, default=2.048e6)
    parser.add_argument("--center-freq", type=float, default=145e6)
    parser.add_argument("--samples", type=int, default=8192)
    args = parser.parse_args(argv)

    manifest = generate(
        data_dir=args.data_dir,
        raw_dir=args.raw_dir,
        baseline=args.baseline,
        seed=args.seed,
        sample_rate=args.sample_rate,
        center_freq=args.center_freq,
        samples=args.samples,
    )

    print(f"Captures: {len(manifest['captures'])}")
    print(f"Expected anomalies: {manifest['expected_anomalies']}")
    print(f"Detected anomalies: {manifest['detected_anomalies']}")
    print(f"Manifest: {Path(manifest['data_dir']) / 'iq_fixture_manifest.json'}")
    if manifest["raw_dir"]:
        print(f"Raw IQ: {manifest['raw_dir']}")

    if manifest["failures"]:
        for failure in manifest["failures"]:
            print(f"FAIL: {failure}", file=sys.stderr)
        return 1
    print("Detector behaved as expected on every capture.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
