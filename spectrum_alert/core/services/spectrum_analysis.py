"""Spectrum analysis primitives shared by detection and geolocation.

Ported from the ``RTL_Triangulator`` project's scan-analysis routines and
adapted for numpy. The two ideas that mattered most over there:

1. **Noise floor from the *lowest* portion of the spectrum**, not the median.
   Taking the median of all bins means a wideband signal (FM broadcast, for
   example) drags the "floor" up with it and the emitter then looks weak. Using
   the mean of the quietest 15 % of bins is robust to signals of any bandwidth.
2. **Prominence + reliability instead of a bare SNR threshold.** A single global
   ``argmax`` SNR says nothing about whether the peak is distinguishable from
   the noise it sits in, or whether the receiver was saturated.

Everything here works in **dBFS** by default and converts to **dBm** only when a
calibration offset is supplied, because an RTL-SDR without a calibrated front end
cannot produce absolute dBm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

__all__ = [
    "ScanResult",
    "SpectralAnalysis",
    "window_samples",
    "compute_psd",
    "estimate_noise_floor",
    "analyze_spectrum",
    "dbm_at_frequency",
    "downsample_points",
    "apply_gain_normalization",
]

#: Fraction of the quietest bins used to estimate the noise floor.
NOISE_FLOOR_FRACTION = 0.15

#: Minimum number of bins used for the noise-floor estimate.
NOISE_FLOOR_MIN_BINS = 4

#: Peak-to-average ratio (dB) below which the capture is considered saturated.
OVERLOAD_PAR_RATIO_DB = 3.0

#: Mean dBFS above which a saturated (flat-topped) capture is expected.
OVERLOAD_MEAN_DBFS = -60.0

#: Display window used by the waterfall and by the confidence mapping.
DISPLAY_FLOOR_DBFS = -120.0
DISPLAY_CEILING_DBFS = -40.0

_WINDOW_CACHE: Dict[int, np.ndarray] = {}


def window_samples(n: int, kind: str = "hann") -> np.ndarray:
    """Return a cached analysis window of length ``n``.

    Cache is keyed by length only, so an unknown ``kind`` gets its own entry
    under a distinct key to avoid returning the wrong window.
    """
    key = n * 8 + {"hann": 0, "hamming": 1, "blackman": 2, "none": 3}.get(kind, 7)
    cached = _WINDOW_CACHE.get(key)
    if cached is not None and cached.shape[0] == n:
        return cached

    if kind == "hann":
        window = np.hanning(n)
    elif kind == "hamming":
        window = np.hamming(n)
    elif kind == "blackman":
        window = np.blackman(n)
    elif kind == "none":
        window = np.ones(n)
    else:
        raise ValueError(f"unknown window kind: {kind!r}")

    window = window.astype(np.float32, copy=False)
    _WINDOW_CACHE[key] = window
    return window


@dataclass
class ScanResult:
    """A frequency-domain scan in a UI/backend friendly shape.

    ``points`` is a list of ``{"hz": float, "dbm": float}`` dicts sorted by
    frequency. ``dbm`` is really dBFS unless ``calibration_offset_db`` was
    applied, in which case it has been shifted toward calibrated dBm.
    """

    center_hz: float
    span_hz: float
    bin_hz: float
    points: List[Dict[str, float]]
    peak_dbfs: float
    noise_floor_dbfs: float
    integration_seconds: float = 1.0
    rms_dbfs: Optional[float] = None
    source: str = "unknown"
    warning: Optional[str] = None

    @property
    def peak_hz(self) -> Optional[float]:
        """Frequency of the strongest bin, or ``None`` for an empty scan."""
        if not self.points:
            return None
        best = max(self.points, key=lambda p: p["dbm"])
        return float(best["hz"])

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "centerHz": self.center_hz,
            "spanHz": self.span_hz,
            "binHz": self.bin_hz,
            "integrationSec": self.integration_seconds,
            "points": self.points,
            "peakDbm": self.peak_dbfs,
            "noiseFloorDbm": self.noise_floor_dbfs,
            "source": self.source,
        }
        if self.rms_dbfs is not None:
            payload["rmsDbm"] = self.rms_dbfs
        if self.warning:
            payload["warning"] = self.warning
        return payload


@dataclass
class SpectralAnalysis:
    """Per-frequency characterisation of a single tone inside a scan."""

    frequency_hz: float
    target_dbfs: float
    noise_floor_dbfs: float
    prominence_db: float
    reliability: float
    overloaded: bool = False
    peak_to_avg_db: float = 0.0
    bin_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frequencyHz": self.frequency_hz,
            "targetDbm": self.target_dbfs,
            "noiseFloorDbm": self.noise_floor_dbfs,
            "prominenceDb": self.prominence_db,
            "reliability": self.reliability,
            "overloaded": self.overloaded,
            "peakToAvgDb": self.peak_to_avg_db,
            "binCount": self.bin_count,
        }


def compute_psd(
    samples: np.ndarray,
    sample_rate_hz: float,
    center_hz: Optional[float] = None,
    nperseg: Optional[int] = None,
    noverlap: Optional[int] = None,
    window: str = "hann",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute a Welch-averaged PSD with frequencies shifted to be monotonic.

    Returns ``(freqs_hz, power_dbfs)`` where ``freqs_hz`` is centred on
    ``center_hz`` when supplied, otherwise on zero (baseband offsets).

    Averaging matters: a single periodogram of a short capture has ~10 dB of
    variance per bin, which makes a fixed SNR threshold behave erratically.
    Welch's overlapped segments cut that variance substantially for the same
    amount of data.
    """
    x = np.asarray(samples)
    if x.size == 0:
        raise ValueError("cannot compute PSD of empty samples")
    # Cast to a complex type, never to a real one: casting complex -> real
    # silently discards the Q component and folds the signal onto DC.
    if not np.iscomplexobj(x):
        x = x.astype(np.complex64, copy=False)
    else:
        x = x.astype(np.complex64, copy=False)

    n = int(x.shape[0])
    default_nperseg = min(n, 1024)
    seg = int(nperseg or default_nperseg)
    seg = max(16, min(seg, n))
    overlap = int(noverlap if noverlap is not None else seg // 2)
    overlap = max(0, min(overlap, seg - 1))

    win = window_samples(seg, window)
    step = seg - overlap
    starts = range(0, max(1, n - seg + 1), step)

    accumulated: Optional[np.ndarray] = None
    segments = 0
    for start in starts:
        chunk = x[start : start + seg]
        if chunk.shape[0] < seg:
            break
        spectrum = np.fft.fftshift(np.fft.fft(chunk * win))
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float64)
        accumulated = power if accumulated is None else accumulated + power
        segments += 1

    if accumulated is None or segments == 0:
        # Fewer samples than one segment: fall back to a single periodogram.
        spectrum = np.fft.fftshift(np.fft.fft(x * window_samples(n, window)))
        power = (spectrum.real ** 2 + spectrum.imag ** 2).astype(np.float64)
        segments = 1
        offsets = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / float(sample_rate_hz)))
        return offsets + (center_hz or 0.0), _power_to_dbfs(power)

    mean_power = accumulated / float(segments)
    # Normalise by window coherent gain so levels are comparable across nperseg.
    coherent_gain = float(np.mean(win)) or 1.0
    mean_power = mean_power / (seg * coherent_gain**2)
    offsets = np.fft.fftshift(np.fft.fftfreq(seg, d=1.0 / float(sample_rate_hz)))
    return offsets + (center_hz or 0.0), _power_to_dbfs(mean_power)


def _power_to_dbfs(power: np.ndarray) -> np.ndarray:
    """Linear power to dBFS (dB relative to full scale)."""
    return 10.0 * np.log10(np.maximum(power, 1e-20))


def estimate_noise_floor(db_values: np.ndarray) -> float:
    """Noise floor from the quietest ``NOISE_FLOOR_FRACTION`` of bins.

    Taking the mean of the *lowest* bins rather than the median of all bins is
    what makes wideband emitters (FM broadcast, DVB, Wi-Fi) still register a
    large prominence: the signal only occupies a slice of the span, so the
    quietest slice is genuine noise regardless of how wide the signal is.
    """
    values = np.asarray(db_values, dtype=np.float64)
    if values.size == 0:
        return float(DISPLAY_FLOOR_DBFS)
    sorted_values = np.sort(values)
    count = max(NOISE_FLOOR_MIN_BINS, int(values.size * NOISE_FLOOR_FRACTION))
    count = min(count, sorted_values.size)
    return float(np.mean(sorted_values[:count]))


def dbm_at_frequency(
    freqs_hz: Sequence[float],
    db_values: Sequence[float],
    target_hz: float,
    window_hz: float = 10_000.0,
    bin_hz: Optional[float] = None,
) -> float:
    """Strongest bin within ``+/- window_hz`` of ``target_hz``.

    The single strongest bin is used rather than a top-N average: averaging
    across sidelobes of a narrow carrier underestimates it.

    Falls back to the nearest available bin when the target lies outside the
    scanned span, so a slightly-off centre frequency still yields a usable
    reading.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    values = np.asarray(db_values, dtype=np.float64)
    if freqs.size == 0:
        return float("nan")
    if not math.isfinite(float(target_hz)):
        return float(np.max(values))

    finite = np.isfinite(freqs) & np.isfinite(values)
    if not finite.any():
        return float("nan")
    freqs, values = freqs[finite], values[finite]

    if bin_hz and math.isfinite(bin_hz) and bin_hz > 0:
        estimated_bin = abs(float(bin_hz))
    elif freqs.size > 1:
        estimated_bin = abs(float(freqs[-1] - freqs[0]) / (freqs.size - 1))
    else:
        estimated_bin = 2000.0

    half_window = max(float(window_hz) / 2.0, estimated_bin * 1.5, 1500.0)
    mask = np.abs(freqs - float(target_hz)) <= half_window
    if not mask.any():
        nearest = int(np.argmin(np.abs(freqs - float(target_hz))))
        return float(values[nearest])
    return float(np.max(values[mask]))


def downsample_points(
    points: Sequence[Dict[str, float]], max_points: int = 256, minimum: int = 64, maximum: int = 1024
) -> List[Dict[str, float]]:
    """Reduce a point list to a display-friendly length.

    Keeps the peak bin: a naive stride can drop the single most important
    sample in the capture.
    """
    target = max(int(minimum), min(int(maximum), int(max_points)))
    items = list(points)
    if len(items) <= target:
        return items

    step = max(1, len(items) // target)
    reduced = items[::step][:target]

    peak = max(items, key=lambda p: float(p["dbm"]))
    if all(abs(float(p["hz"]) - float(peak["hz"])) > 1e-6 for p in reduced):
        reduced[-1] = {"hz": float(peak["hz"]), "dbm": float(peak["dbm"])}
    return reduced


def apply_gain_normalization(dbfs: np.ndarray, reference_gain_db: Optional[float]) -> np.ndarray:
    """Shift dBFS levels to a common gain so thresholds stay meaningful.

    The existing detector's fixed SNR threshold is only valid at a fixed gain:
    the same physical signal scores higher at 30 dB than at 20 dB. Comparing
    against a known reference gain removes that dependency.
    """
    if reference_gain_db is None:
        return np.asarray(dbfs, dtype=np.float64)
    return np.asarray(dbfs, dtype=np.float64) - float(reference_gain_db)


def analyze_spectrum(
    freqs_hz: Sequence[float],
    db_values: Sequence[float],
    target_hz: float,
    window_hz: float = 12_000.0,
    reliability_floor_db: float = 1.5,
    reliability_span_db: float = 8.0,
) -> SpectralAnalysis:
    """Characterise the signal at ``target_hz`` inside a scan.

    ``reliability`` maps linearly from ``reliability_floor_db`` to
    ``reliability_floor_db + reliability_span_db`` of prominence onto ``0..1``,
    and saturates when the receiver looks saturated.
    """
    freqs = np.asarray(freqs_hz, dtype=np.float64)
    values = np.asarray(db_values, dtype=np.float64)
    if values.size == 0:
        return SpectralAnalysis(
            frequency_hz=float(target_hz),
            target_dbfs=float("nan"),
            noise_floor_dbfs=float("nan"),
            prominence_db=0.0,
            reliability=0.0,
        )

    noise_floor = estimate_noise_floor(values)
    target_db = dbm_at_frequency(freqs, values, target_hz, window_hz)

    average = float(np.mean(values))
    peak = float(np.max(values))
    peak_to_avg = peak - average

    # A flat-topped capture means the front end is saturated: the signal is
    # actually very strong, so prominence under-reports it. Correct upwards.
    overloaded = peak_to_avg < OVERLOAD_PAR_RATIO_DB and average > OVERLOAD_MEAN_DBFS
    prominence = target_db - noise_floor
    if overloaded:
        prominence = max(prominence, 15.0)

    reliability = max(0.0, min(1.0, (prominence - reliability_floor_db) / max(reliability_span_db, 1e-6)))

    return SpectralAnalysis(
        frequency_hz=float(target_hz),
        target_dbfs=float(target_db),
        noise_floor_dbfs=float(noise_floor),
        prominence_db=float(prominence),
        reliability=float(reliability),
        overloaded=bool(overloaded),
        peak_to_avg_db=float(peak_to_avg),
        bin_count=int(values.size),
    )


def scan_result_from_arrays(
    freqs_hz: np.ndarray,
    db_values: np.ndarray,
    source: str = "unknown",
    integration_seconds: float = 1.0,
    warning: Optional[str] = None,
    calibration_offset_db: float = 0.0,
) -> ScanResult:
    """Build a :class:`ScanResult` from frequency/power arrays."""
    order = np.argsort(freqs_hz)
    freqs = np.asarray(freqs_hz, dtype=np.float64)[order]
    values = np.asarray(db_values, dtype=np.float64)[order] + float(calibration_offset_db)

    span = float(freqs[-1] - freqs[0]) if freqs.size > 1 else 0.0
    bin_hz = float(span / max(1, freqs.size - 1)) if freqs.size > 1 else 0.0
    center = float((freqs[0] + freqs[-1]) / 2.0) if freqs.size else 0.0

    points = [{"hz": float(f), "dbm": float(v)} for f, v in zip(freqs, values)]
    return ScanResult(
        center_hz=center,
        span_hz=span,
        bin_hz=bin_hz,
        points=points,
        peak_dbfs=float(np.max(values)) if values.size else float("nan"),
        noise_floor_dbfs=estimate_noise_floor(values),
        integration_seconds=float(integration_seconds),
        source=source,
        warning=warning,
    )


def flat_spectrum_scan(
    center_hz: float,
    span_hz: float,
    level_db: float,
    bins: int = 256,
    source: str = "manual_override",
    warning: Optional[str] = None,
) -> ScanResult:
    """Synthesise a flat scan at ``level_db``.

    Used for receivers pinned to a manual RSSI override (which cannot be
    measured) and for constant-level fallbacks.
    """
    bins = max(1, int(bins))
    start = float(center_hz) - float(span_hz) / 2.0
    step = float(span_hz) / max(1, bins - 1)
    points = [{"hz": start + i * step, "dbm": float(level_db)} for i in range(bins)]
    return ScanResult(
        center_hz=float(center_hz),
        span_hz=float(span_hz),
        bin_hz=step,
        points=points,
        peak_dbfs=float(level_db),
        noise_floor_dbfs=float(level_db),
        integration_seconds=1.0,
        source=source,
        warning=warning,
    )