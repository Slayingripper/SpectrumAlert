"""Per-receiver calibration offsets and path-loss-exponent fitting.

Ported from the ``RTL_Triangulator`` project's ``calibration.json`` handling and
``/api/fit-ple`` endpoint.

Two problems are solved here:

**Absolute levels.** An RTL-SDR reports dBFS. Turning that into a comparable dBm
requires a per-receiver, per-frequency offset, because the offset depends on the
antenna, cable, gain setting and the dongle's own IF chain. Offsets are keyed by
``"k<kHz>"`` so a single flat JSON object can hold both receiver and frequency
dimensions.

**Propagation exponent.** The path-loss exponent ``n`` is not a constant; it
depends on band and environment. It is fitted here from known transmitters whose
locations are known, using a distance-weighted linear regression with iterative
sigma clipping so a single bad reading cannot drag the slope.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from spectrum_alert.core.exceptions import ConfigurationError
from spectrum_alert.core.services.propagation import (
    frequency_default_path_loss,
    haversine_meters,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CalibrationTable",
    "PathLossFitPoint",
    "PathLossFitResult",
    "frequency_key",
    "parse_frequency_key",
    "fit_path_loss_exponent",
]

#: Offsets beyond this magnitude are almost certainly misaligned and would
#: destabilise geolocation, so they are clamped.
MAX_ABS_OFFSET_DB = 20.0

#: Minimum MAD-derived rejection threshold, in dB. Below this the "outlier"
#: filter starts eating legitimate readings.
MIN_REJECTION_THRESHOLD_DB = 6.0

#: MAD scale factor making it a consistent estimator of sigma for normal data.
MAD_TO_SIGMA = 1.4826

#: Physically plausible range for a terrestrial VHF/UHF path-loss exponent.
MIN_PATH_LOSS_EXPONENT = 1.5
MAX_PATH_LOSS_EXPONENT = 4.5

#: A reference power at 1 m above this is impossible for a broadcast transmitter.
MAX_REFERENCE_RSSI_DBM = -20.0


def frequency_key(frequency_hz: float) -> str:
    """Frequency component of a calibration key, rounded to kHz.

    Rounding to kHz lets measurements taken a few Hz apart share an offset.
    """
    return f"k{int(round(float(frequency_hz) / 1000.0))}"


def parse_frequency_key(key: str) -> float:
    """Inverse of :func:`frequency_key`."""
    text = str(key).strip().lower()
    if text.startswith("k"):
        text = text[1:]
    try:
        return float(text) * 1000.0
    except ValueError as exc:  # pragma: no cover - defensive
        raise ConfigurationError(f"invalid calibration frequency key: {key!r}") from exc


@dataclass
class CalibrationTable:
    """Per-receiver, per-frequency dB offsets with atomic JSON persistence.

    Offsets live in a nested mapping ``{receiver_id: {frequency_key: db}}``.
    """

    path: Optional[str] = None
    offsets: Dict[str, Dict[str, float]] = field(default_factory=dict)
    updated_at: Optional[str] = None
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------------ load

    @classmethod
    def load(cls, path: str) -> "CalibrationTable":
        """Load a table from ``path``; returns an empty table if absent/corrupt."""
        table = cls(path=path)
        if not path or not os.path.exists(path):
            return table
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load calibration from %s: %s", path, exc)
            return table

        offsets = raw.get("offsets") if isinstance(raw, dict) else None
        if not isinstance(offsets, dict):
            return table

        cleaned: Dict[str, Dict[str, float]] = {}
        for receiver_id, entries in offsets.items():
            if not isinstance(entries, dict):
                continue
            per_frequency: Dict[str, float] = {}
            for key, value in entries.items():
                try:
                    numeric = float(value)
                except (TypeError, ValueError):
                    logger.warning("Dropping non-numeric calibration entry %s/%s", receiver_id, key)
                    continue
                if math.isfinite(numeric):
                    per_frequency[str(key)] = numeric
            if per_frequency:
                cleaned[str(receiver_id)] = per_frequency

        with table._lock:
            table.offsets = cleaned
            table.updated_at = raw.get("updated_at") if isinstance(raw, dict) else None
        return table

    # ------------------------------------------------------------------ save

    def save(self) -> Optional[str]:
        """Persist atomically: write a temp file in the same dir, then rename."""
        if not self.path:
            return None
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            payload = {
                "offsets": self.offsets,
                "updated_at": datetime.utcnow().isoformat(),
            }
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory, prefix=".calibration-", suffix=".tmp", delete=False
        )
        try:
            with handle:
                json.dump(payload, handle, indent=2)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(handle.name, self.path)
        except OSError as exc:
            try:
                os.unlink(handle.name)
            except OSError:
                pass
            raise ConfigurationError(f"failed to persist calibration: {exc}") from exc
        return self.path

    # ---------------------------------------------------------------- access

    def offset_for(self, receiver_id: str, frequency_hz: float) -> float:
        """Offset for a receiver/frequency, clamped to a sane range.

        Missing entries mean "not calibrated", which is 0 dB. An out-of-range
        stored value is treated the same way as a missing one rather than being
        trusted: a mis-keyed offset is worse than no offset.
        """
        with self._lock:
            raw = self.offsets.get(str(receiver_id), {}).get(frequency_key(frequency_hz), 0.0)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if not math.isfinite(value) or abs(value) > MAX_ABS_OFFSET_DB:
            return 0.0
        return value

    def apply(self, receiver_id: str, frequency_hz: float, value_db: float) -> None:
        """Store an offset for a receiver/frequency."""
        numeric = float(value_db)
        if not math.isfinite(numeric):
            raise ConfigurationError("calibration offset must be finite")
        clamped = max(-MAX_ABS_OFFSET_DB, min(MAX_ABS_OFFSET_DB, numeric))
        key = frequency_key(frequency_hz)
        with self._lock:
            self.offsets.setdefault(str(receiver_id), {})[key] = clamped
            self.updated_at = datetime.utcnow().isoformat()

    def clear_receiver(self, receiver_id: str) -> None:
        """Drop every offset for one receiver."""
        with self._lock:
            self.offsets.pop(str(receiver_id), None)

    def clear_all(self) -> None:
        """Drop every offset."""
        with self._lock:
            self.offsets = {}

    def receivers(self) -> List[str]:
        with self._lock:
            return sorted(self.offsets.keys())

    def frequencies_for(self, receiver_id: str) -> List[float]:
        """Calibrated frequencies for a receiver, numerically ascending."""
        with self._lock:
            keys = list(self.offsets.get(str(receiver_id), {}).keys())
        try:
            return sorted(parse_frequency_key(key) for key in keys)
        except ConfigurationError:
            return []

    def to_dict(self) -> Dict[str, Any]:
        with self._lock:
            return {"offsets": json.loads(json.dumps(self.offsets))}

    def merge(self, other: Dict[str, Dict[str, float]]) -> None:
        """Merge a ``previewOffsets``-shaped mapping in, honouring ``used`` flags."""
        with self._lock:
            for receiver_id, entries in (other or {}).items():
                if not isinstance(entries, dict):
                    continue
                target = self.offsets.setdefault(str(receiver_id), {})
                for key, entry in entries.items():
                    if isinstance(entry, dict):
                        if not entry.get("used", True):
                            continue
                        value = entry.get("offset", 0.0)
                    else:
                        value = entry
                    try:
                        numeric = float(value)
                    except (TypeError, ValueError):
                        continue
                    if math.isfinite(numeric):
                        target[str(key)] = max(
                            -MAX_ABS_OFFSET_DB, min(MAX_ABS_OFFSET_DB, numeric)
                        )
            self.updated_at = datetime.utcnow().isoformat()


@dataclass
class PathLossFitPoint:
    """One (transmitter, receiver) observation used in a PLE fit."""

    transmitter: str
    frequency_hz: float
    receiver_id: str
    receiver_name: str
    measured_dbfs: float
    adjusted_dbfs: float
    distance_meters: float
    used_in_fit: bool = True
    weight: float = 0.0
    residual_db: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transmitter": self.transmitter,
            "frequencyHz": self.frequency_hz,
            "receiverId": self.receiver_id,
            "receiverName": self.receiver_name,
            "measured": round(self.measured_dbfs, 4),
            "adjusted": round(self.adjusted_dbfs, 4),
            "distMeters": round(self.distance_meters, 2),
            "usedInFit": self.used_in_fit,
            "weight": round(self.weight, 6),
            "residualDb": self.residual_db,
        }


@dataclass
class PathLossFitResult:
    """Outcome of a path-loss-exponent fit."""

    path_loss_exponent: float
    reference_rssi_dbm: float
    r_squared: float
    points: List[PathLossFitPoint] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warning: Optional[str] = None
    fallback_applied: bool = False
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pathLossExponent": self.path_loss_exponent,
            "referenceRssiDbm": self.reference_rssi_dbm,
            "r2": self.r_squared,
            "points": [p.to_dict() for p in self.points],
            "errors": list(self.errors),
            "warning": self.warning,
            "fallbackApplied": self.fallback_applied,
            "details": dict(self.details),
        }


def _median(values: Sequence[float]) -> Optional[float]:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    ordered = sorted(finite)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def compute_relative_offsets(
    readings: Sequence[Dict[str, Any]], frequency_hz: float
) -> Tuple[Dict[str, float], List[str]]:
    """Offsets that equalise all receivers for one frequency.

    RSSI differences between receivers at one frequency come from antenna gain,
    cable loss and dongle calibration. Subtracting the median across receivers
    removes the common-mode part, leaving only per-receiver error.

    Rejects outliers with a MAD-based threshold so one bad receiver cannot drag
    the whole table.
    """
    key = frequency_key(frequency_hz)
    usable = [
        r
        for r in readings
        if r.get("rssi_dbfs") is not None and math.isfinite(float(r["rssi_dbfs"]))
    ]
    if not usable:
        return {}, []

    values = [float(r["rssi_dbfs"]) for r in usable]
    center = _median(values)
    if center is None:
        return {}, []

    deviations = [abs(float(r["rssi_dbfs"]) - center) for r in usable]
    mad = _median(deviations) or 1.0
    threshold = max(3.0 * mad, MIN_REJECTION_THRESHOLD_DB)

    offsets: Dict[str, float] = {}
    rejected: List[str] = []
    for reading in usable:
        value = float(reading["rssi_dbfs"])
        receiver_id = str(reading["receiver_id"])
        if abs(value - center) > threshold:
            rejected.append(
                f"{reading.get('receiver_name', receiver_id)}: "
                f"{value - center:+.1f} dB from median (>{threshold:.1f} dB)"
            )
            continue
        offsets[receiver_id] = {
            "offset": max(-MAX_ABS_OFFSET_DB, min(MAX_ABS_OFFSET_DB, center - value)),
            "used": True,
            "frequencyKey": key,
        }
    return offsets, rejected


def _weighted_linear_fit(
    xs: Sequence[float], ys: Sequence[float], weights: Sequence[float]
) -> Optional[Dict[str, float]]:
    """Weighted least squares fit of ``y = a + b*x``."""
    n = len(xs)
    if n < 2 or len(ys) != n or len(weights) != n:
        return None

    weight_sum = float(sum(weights))
    if weight_sum <= 0:
        return None

    mean_x = sum(w * x for w, x in zip(weights, xs)) / weight_sum
    mean_y = sum(w * y for w, y in zip(weights, ys)) / weight_sum

    covariance = 0.0
    variance_x = 0.0
    for w, x, y in zip(weights, xs, ys):
        dx = x - mean_x
        covariance += w * dx * (y - mean_y)
        variance_x += w * dx * dx

    if variance_x == 0:
        return None

    slope = covariance / variance_x
    return {
        "a": mean_y - slope * mean_x,
        "b": slope,
        "meanX": mean_x,
        "meanY": mean_y,
    }


def fit_path_loss_exponent(
    observations: Sequence[Dict[str, Any]],
    default_frequency_hz: Optional[float] = None,
    max_rounds: int = 3,
) -> PathLossFitResult:
    """Fit ``RSSI = P0 - 10*n*log10(d)`` from known transmitter positions.

    ``observations`` entries need ``frequency_hz``, ``distance_meters`` and
    ``adjusted_dbfs`` (calibration already applied). Optional ``transmitter``,
    ``receiver_id`` and ``receiver_name`` are carried into the output for
    display.

    Weighting is ``1000/d`` clipped to ``[0.1, 10]``: nearer receivers have
    better ground-truth distance and their readings deserve more influence.
    Rejection is iterative MAD-based clipping, so the fit is not dominated by
    multipath-faded distant receivers.

    Falls back to band defaults whenever the data cannot support a real fit, and
    says so in ``warning`` / ``fallback_applied`` rather than silently returning
    a number the caller might trust.
    """
    points: List[PathLossFitPoint] = []
    errors: List[str] = []
    nearest_hz = default_frequency_hz

    for item in observations:
        try:
            frequency_hz = float(item["frequency_hz"])
            distance_meters = float(item["distance_meters"])
            adjusted = float(item["adjusted_dbfs"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"skipping malformed observation: {item!r}")
            continue

        if not all(math.isfinite(v) for v in (frequency_hz, distance_meters, adjusted)):
            errors.append(f"skipping non-finite observation: {item!r}")
            continue

        # Inside 10 m the log-distance model breaks down (near field, mutual
        # coupling); including those points biases the intercept badly.
        if distance_meters <= 10.0:
            errors.append(
                f"skipping {item.get('transmitter', 'observation')} at {distance_meters:.1f} m (too close)"
            )
            continue

        if nearest_hz is None:
            nearest_hz = frequency_hz

        raw_weight = max(0.1, min(10.0, 1000.0 / max(10.0, distance_meters)))
        points.append(
            PathLossFitPoint(
                transmitter=str(item.get("transmitter") or f"{frequency_hz/1e6:.3f} MHz"),
                frequency_hz=frequency_hz,
                receiver_id=str(item.get("receiver_id", "")),
                receiver_name=str(item.get("receiver_name", "")),
                measured_dbfs=float(item.get("measured_dbfs", adjusted)),
                adjusted_dbfs=adjusted,
                distance_meters=distance_meters,
                weight=raw_weight,
            )
        )

    reference_hz = nearest_hz or 100e6
    defaults = frequency_default_path_loss(reference_hz)

    if not points:
        return PathLossFitResult(
            path_loss_exponent=defaults.exponent,
            reference_rssi_dbm=defaults.reference_rssi_dbm,
            r_squared=0.0,
            points=points,
            errors=errors,
            warning=(
                "No valid measurements collected for fitting. "
                f"Using band default n={defaults.exponent}."
            ),
            fallback_applied=True,
        )

    distances = [p.distance_meters for p in points]
    min_d, max_d = min(distances), max(distances)
    spread_ratio = max_d / max(1.0, min_d)

    # Fewer than 1.5x distance variation means the slope is dominated by noise
    # and by gain drift. Keep the band exponent, derive P0 from the data.
    if spread_ratio < 1.5:
        candidates = sorted(
            p.adjusted_dbfs + 10.0 * defaults.exponent * math.log10(p.distance_meters)
            for p in points
        )
        median_p0 = candidates[len(candidates) // 2]
        p0 = min(median_p0, MAX_REFERENCE_RSSI_DBM)
        for point in points:
            point.used_in_fit = True
            point.weight = max(0.1, min(10.0, 1000.0 / max(10.0, point.distance_meters)))
            point.residual_db = 0.0
        return PathLossFitResult(
            path_loss_exponent=defaults.exponent,
            reference_rssi_dbm=round(p0, 4),
            r_squared=0.0,
            points=points,
            errors=errors,
            warning=(
                f"Receivers all at similar distances (ratio {spread_ratio:.2f}:1). "
                f"Using band default n={defaults.exponent} and deriving P0 from the data."
            ),
            fallback_applied=False,
        )

    xs = [math.log10(p.distance_meters) for p in points]
    ys = [p.adjusted_dbfs for p in points]
    weights = [p.weight for p in points]

    best_fit: Optional[Dict[str, float]] = None
    best_indices: List[int] = list(range(len(points)))

    for _round in range(max(1, max_rounds)):
        fit = _weighted_linear_fit(xs, ys, weights)
        if fit is None:
            break

        residuals = [y - (fit["a"] + fit["b"] * x) for x, y in zip(xs, ys)]
        abs_residuals = [abs(r) for r in residuals]
        median_abs = _median(abs_residuals) or 1.0
        threshold = max(3.0 * MAD_TO_SIGMA * median_abs, MIN_REJECTION_THRESHOLD_DB)
        keep = [abs(r) <= threshold for r in abs_residuals]

        best_fit = fit
        kept = [i for i, k in enumerate(keep) if k]
        if len(kept) < 2 or len(kept) == len(points):
            break
        best_indices = kept
        xs = [xs[i] for i in kept]
        ys = [ys[i] for i in kept]
        weights = [weights[i] for i in kept]

    if best_fit is None:
        best_fit = _weighted_linear_fit(
            [math.log10(p.distance_meters) for p in points],
            [p.adjusted_dbfs for p in points],
            [p.weight for p in points],
        )
        best_indices = list(range(len(points)))

    if best_fit is None:
        return PathLossFitResult(
            path_loss_exponent=defaults.exponent,
            reference_rssi_dbm=defaults.reference_rssi_dbm,
            r_squared=0.0,
            points=points,
            errors=errors + ["Insufficient distance variation to fit path-loss exponent."],
            warning="Fit failed; using band defaults.",
            fallback_applied=True,
        )

    slope = best_fit["b"]
    intercept = best_fit["a"]
    exponent = -slope / 10.0
    reference_rssi = intercept

    used = set(best_indices)
    for index, point in enumerate(points):
        point.used_in_fit = index in used
        point.residual_db = round(
            point.adjusted_dbfs - (intercept + slope * math.log10(point.distance_meters)), 3
        )

    physically_valid = (
        math.isfinite(exponent)
        and MIN_PATH_LOSS_EXPONENT <= exponent <= MAX_PATH_LOSS_EXPONENT
        and slope < 0
        and math.isfinite(reference_rssi)
        and reference_rssi <= MAX_REFERENCE_RSSI_DBM
    )

    if not physically_valid:
        if math.isfinite(intercept) and intercept <= -10:
            fallback_p0 = max(-70.0, min(MAX_REFERENCE_RSSI_DBM, intercept))
        else:
            fallback_p0 = defaults.reference_rssi_dbm
        return PathLossFitResult(
            path_loss_exponent=defaults.exponent,
            reference_rssi_dbm=round(fallback_p0, 4),
            r_squared=0.0,
            points=points,
            errors=errors,
            warning=(
                f"Unphysical fit (n={exponent:.3f}, slope={slope:.3f}, "
                f"reference={reference_rssi:.1f} dBm); using band default "
                f"n={defaults.exponent}, reference={fallback_p0:.1f} dBm."
            ),
            fallback_applied=True,
            details={"slope": slope, "intercept": intercept, "rawPathLossExponent": exponent},
        )

    # R^2 over the surviving points.
    final_x = [math.log10(points[i].distance_meters) for i in sorted(used)]
    final_y = [points[i].adjusted_dbfs for i in sorted(used)]
    if len(final_y) >= 2:
        mean_y = sum(final_y) / len(final_y)
        ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(final_x, final_y))
        ss_tot = sum((y - mean_y) ** 2 for y in final_y)
        r_squared = 1.0 if ss_tot == 0 else max(0.0, 1.0 - ss_res / ss_tot)
    else:
        r_squared = 0.0

    return PathLossFitResult(
        path_loss_exponent=round(exponent, 6),
        reference_rssi_dbm=round(reference_rssi, 4),
        r_squared=round(r_squared, 6),
        points=points,
        errors=errors,
        fallback_applied=False,
    )


def build_observations(
    transmitters: Iterable[Dict[str, Any]],
    receivers: Iterable[Dict[str, Any]],
    measurements: Dict[Tuple[str, float], float],
    calibration: Optional[CalibrationTable] = None,
) -> List[Dict[str, Any]]:
    """Pair transmitters with receivers and pull measured levels out of a map.

    ``measurements`` is keyed by ``(receiver_id, frequency_hz)``. Distances come
    from great-circle separation between the transmitter and receiver positions.
    """
    observations: List[Dict[str, Any]] = []
    for transmitter in transmitters:
        try:
            frequency_hz = float(transmitter["frequency_hz"])
            tx_lat = float(transmitter["lat"])
            tx_lon = float(transmitter["lon"])
        except (KeyError, TypeError, ValueError):
            continue

        for receiver in receivers:
            receiver_id = str(receiver.get("id", ""))
            if not receiver_id or receiver.get("enabled") is False:
                continue
            if receiver.get("manual_rssi_db") is not None:
                # A pinned manual level is not a measurement; including it would
                # fit the model to a constant.
                continue
            try:
                rx_lat = float(receiver["lat"])
                rx_lon = float(receiver["lon"])
            except (KeyError, TypeError, ValueError):
                continue

            level = measurements.get((receiver_id, frequency_hz))
            if level is None or not math.isfinite(float(level)):
                continue

            adjusted = float(level) + (
                calibration.offset_for(receiver_id, frequency_hz) if calibration else 0.0
            )
            observations.append(
                {
                    "transmitter": transmitter.get("name") or f"{frequency_hz/1e6:.3f} MHz",
                    "frequency_hz": frequency_hz,
                    "receiver_id": receiver_id,
                    "receiver_name": receiver.get("name", receiver_id),
                    "measured_dbfs": float(level),
                    "adjusted_dbfs": adjusted,
                    "distance_meters": haversine_meters(tx_lat, tx_lon, rx_lat, rx_lon),
                }
            )
    return observations