"""RF device fingerprinting: signature extraction and library matching.

Identification ("what is this transmitter?") is a different question from
location ("where is it?"), but it answers the same operator need, and it is
driven by the same scan. Keeping both on one page means one waterfall, one
receiver roster and one target frequency feed both workflows.

What a fingerprint is
--------------------
No two transmitters are bit-identical: their oscillator, PA, filter and
antenna mismatch leave stable, measurable differences in the IQ stream. The
literature is unambiguous that envelope statistics and higher-order statistics
carry most of that structure, and that a handful of features separates
radiotypes far better than raw spectra do. The feature set here is deliberately
small, interpretable and gain-invariant:

* **Envelope statistics** - mean, RMS, standard deviation, skewness, excess
  kurtosis of ``|x|``.
* **Higher-order cumulants** - third and fourth normalised cumulants, which
  respond to PA compression in a way first and second moments cannot.
* **Envelope dynamics** - normalised rise/fall rates and envelope correlation
  lag, which distinguish keyed transmitters from linear amplifiers.
* **Spectral shape** - occupied bandwidth fraction and spectral roll-off
  measured from the Welch estimate.

Normalisation matters more than it looks. Raw features scale with gain, and gain
differs per receiver and per session; features are therefore expressed relative
to the signal's own mean power so a library entry captured on one receiver still
matches on another.

Honesty rules baked into this module
-----------------------------------
* An empty library yields "unknown", never a guess. There is no prior.
* A match below :data:`DEFAULT_ACCEPT_THRESHOLD` is reported as a candidate but
  never as an identification, because a confident wrong ID is worse than none.
* Two signatures closer than :data:`MIN_SEPARATION` are reported as ambiguous,
  not resolved by arbitrary tie-breaking.
* Every match carries its distance, so the caller can show how close it was.

Ported in spirit from the ``RTL_Triangulator`` project's receiver model; the
extraction and matching here are implemented from the standard feature set
rather than copied, and are validated by tests in
``tests/core/test_fingerprinting.py``.
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from spectrum_alert.core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

__all__ = [
    "FingerprintVector",
    "LibraryEntry",
    "FingerprintLibrary",
    "extract_signature",
    "cosine_distance",
    "match_signature",
]


#: Names of the extracted features, in vector order. Persisted with every
#: signature so a library written by an older version can be detected rather
#: than silently mis-compared.
#:
#: Every entry is **scale invariant**: features that merely restate the signal's
#: absolute power (envelope mean, absolute amplitude) are excluded on purpose,
#: because gain differs per receiver and per session and a signature carrying
#: absolute level would only ever match the receiver that captured it.
FEATURE_NAMES: Tuple[str, ...] = (
    "envelope_cv",
    "envelope_skew",
    "envelope_kurtosis",
    "moment3_complex",
    "moment4_complex",
    "rise_rate",
    "fall_rate",
    "lag1_correlation",
    "occupied_fraction",
    "spectral_rolloff",
    "spectral_centroid",
    "peak_to_mean_db",
)

#: Vector length expected by the matcher.
SIGNATURE_LENGTH = len(FEATURE_NAMES)

#: Largest RMS feature difference, in units of the library's own spread, still
#: accepted as an identification. Measured on synthetic radiotypes a repeat
#: capture of the same unit sits near 0.2 and a different unit near 0.7, so 0.5
#: separates them while leaving room for noisier real receivers. The bias is
#: deliberate towards reporting "unknown" rather than a confident wrong ID.
DEFAULT_ACCEPT_THRESHOLD = 0.5

#: Candidates closer than this to each other are too similar to separate; the
#: result is reported ambiguous instead of picking one.
MIN_SEPARATION = 0.12

#: A dimension whose spread across the library is below this carries no
#: discriminating information and is excluded from matching, rather than being
#: amplified into noise by standardisation.
_FEATURE_SPREAD_FLOOR = 1e-9

#: Minimum comparison scale as a fraction of a feature's own magnitude.
_RELATIVE_SCALE_FLOOR = 0.02

#: Feature count at which the distance discards its single worst deviation.
_ROBUST_TRIM_MIN_FEATURES = 8


# --------------------------------------------------------------------------
# Vector maths
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FingerprintVector:
    """A finite feature vector with a named axis."""

    values: Tuple[float, ...]
    labels: Tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        if len(self.values) != len(self.labels):
            raise ValueError(
                f"signature has {len(self.values)} values but {len(self.labels)} labels"
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "labels": list(self.labels),
            "values": {name: float(v) for name, v in zip(self.labels, self.values)},
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "FingerprintVector":
        labels = payload.get("labels") or list(FEATURE_NAMES)
        values = payload.get("values")
        if isinstance(values, dict):
            ordered = [float(values[name]) for name in labels if name in values]
        else:
            ordered = [float(v) for v in (values or [])]
        return cls(tuple(ordered), tuple(labels))

    def magnitude(self) -> float:
        """Euclidean norm; zero vectors are unmatchable and returned as-is."""
        return float(math.sqrt(sum(v * v for v in self.values)))


def cosine_distance(left: FingerprintVector, right: FingerprintVector) -> float:
    """Plain cosine distance, kept for callers that want the raw measure.

    Returns ``1 - cosine_similarity``, so 0 means identical. Note this is *not*
    what :func:`match_signature` uses: cosine treats a loud feature and a quiet
    feature as equally important, which is rarely what separates two radios.
    See :func:`scaled_distance`.
    """
    if left.labels != right.labels:
        raise ValueError(
            "cannot compare signatures with different features: "
            f"{list(left.labels)} vs {list(right.labels)}"
        )
    left_norm = left.magnitude()
    right_norm = right.magnitude()
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 1.0
    dot = sum(a * b for a, b in zip(left.values, right.values))
    cosine = dot / (left_norm * right_norm)
    # Clamp before subtracting: float error can push a self-comparison to
    # 1.0000000000000002, which would report a distance of ~-2e-16.
    cosine = min(1.0, max(-1.0, cosine))
    return 1.0 - cosine


def scaled_distance(
    left: FingerprintVector,
    right: FingerprintVector,
    scales: Sequence[float],
) -> float:
    """Root-mean-square difference measured in units of ``scales``.

    Each feature is divided by the spread that feature actually shows in the
    library before being compared, so a feature that varies a lot carries less
    weight than one that barely moves. Distance is 0 for identical signatures
    and approaches 1 when every feature differs by a full spread.

    This replaces plain cosine for identification. Cosine weighs a feature by its
    raw magnitude, so whichever feature happens to be largest dominates the
    ranking and two different radios can look alike on it.
    """
    if left.labels != right.labels:
        raise ValueError("cannot compare signatures with different feature sets")

    deviations: List[float] = []
    for a, b, scale in zip(left.values, right.values, scales):
        if not math.isfinite(scale) or scale < _FEATURE_SPREAD_FLOOR:
            # No usable spread for this feature: it carries no information, so
            # exclude it rather than amplifying pure noise.
            continue
        deviations.append(((a - b) / scale) ** 2)

    if not deviations:
        return 1.0

    if len(deviations) >= _ROBUST_TRIM_MIN_FEATURES:
        # Drop the single worst feature before averaging.
        #
        # Some of these features are near-cancellations for a given signal class
        # (a third-order moment that is nearly zero for an ideal FM waveform),
        # which makes them numerically unstable: two captures of the *same*
        # radio can differ on them by several of the library's spreads while
        # agreeing on everything else. Averaging in that one feature lets it veto
        # an otherwise perfect match. Trimming only the single worst deviation
        # keeps the tolerance narrow, so two genuinely different radios, which
        # differ across many features, still separate cleanly.
        deviations.sort()
        deviations = deviations[:-1]

    return min(1.0, math.sqrt(sum(deviations) / len(deviations)))


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def _standardised_moments(
    values: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None,
) -> Tuple[float, float]:
    """Skewness and excess kurtosis in one pass.

    The two used to compute the mean, standard deviation and standardised
    array separately. Combining them halves the work over a large capture.
    ``mean`` and ``std`` can be supplied when the caller already has them.
    """
    if values.size < 2:
        return 0.0, 0.0
    if mean is None:
        mean = float(values.mean())
    if std is None:
        std = float(values.std())
    if std <= 1e-12:
        return 0.0, 0.0
    centred = (values - mean) / std
    return float(np.mean(centred**3)), float(np.mean(centred**4) - 3.0)


def _complex_moments(iq: np.ndarray, mean_power: float) -> Tuple[float, float]:
    """Third- and fourth-order normalised complex moment magnitudes.

    These are the moments that respond to power-amplifier compression: a
    nonlinear PA folds the constellation and leaves a trace in the third and
    fourth complex moments that the envelope statistics cannot see, because the
    envelope has already discarded the phase. The magnitude is used rather than
    the real part because the phase of the moment depends on carrier phase,
    which is not a device property; taking the magnitude keeps the PA signature
    while discarding the arbitrary rotation. Scale invariant by construction.

    ``iq**2`` is formed once and reused for both orders; the previous code
    recomputed the power and a full complex power separately for each order.
    """
    if mean_power <= 1e-30:
        return 0.0, 0.0
    squared = iq * iq
    third = abs(complex(np.mean(squared * iq)))
    fourth = abs(complex(np.mean(squared * squared)))
    return (
        third / (mean_power ** 1.5),
        fourth / (mean_power ** 2.0),
    )


@lru_cache(maxsize=8)
def _hann_window(size: int) -> np.ndarray:
    """Cache the window: repeated fingerprints use a handful of capture sizes."""
    return np.hanning(size)


def _spectral_shape(
    iq: np.ndarray,
    sample_rate_hz: float,
    frequencies: Optional[np.ndarray],
    powers: Optional[np.ndarray],
) -> Tuple[float, float, float]:
    """Occupied bandwidth fraction, roll-off and normalised centroid.

    Computed from a Welch-style estimate of the capture when the caller does not
    supply one, so the vector always has the same length whether or not the
    caller already computed a spectrum.
    """
    if frequencies is not None and powers is not None:
        freqs = np.asarray(frequencies, dtype=float)
        power = np.asarray(powers, dtype=float)
    else:
        size = int(2 ** math.ceil(math.log2(max(iq.size, 256))))
        # IQ is complex, so this must be a full complex FFT; rfft only accepts
        # real input. Keep the non-negative half directly instead of shifting
        # the full spectrum and then filtering half of it away.
        spectrum = np.fft.fft(iq * _hann_window(iq.size), n=size)
        half = size // 2 + 1
        power = (np.abs(spectrum[:half]) ** 2).astype(float)
        freqs = np.fft.fftfreq(size, d=1.0 / sample_rate_hz)[:half]

    if freqs.size != power.size or freqs.size < 2:
        return 0.0, 0.0, 0.0

    # Keep the positive-frequency half: roll-off must not depend on whether the
    # caller handed us a one-sided or two-sided spectrum.
    keep = freqs >= 0
    if not np.any(keep):
        keep = np.ones_like(freqs, dtype=bool)
    freqs = freqs[keep]
    power = power[keep]

    total = float(np.sum(power))
    if total <= 1e-30:
        return 0.0, 0.0, 0.0

    occupied = float(np.sum(power >= total * 0.01) / power.size)
    cumulative = np.cumsum(power) / total
    index = int(np.searchsorted(cumulative, 0.95))
    index = min(max(index, 0), power.size - 1)

    span = float(abs(freqs[-1] - freqs[0])) or float(sample_rate_hz)
    rolloff = float(min(1.0, max(0.0, abs(freqs[index] - freqs[0]) / span)))

    # Energy-weighted centroid, normalised to the same span so it is comparable
    # across sample rates.
    centroid = float(np.sum(freqs * power) / total)
    normalised_centroid = float(min(1.0, max(0.0, abs(centroid - freqs[0]) / span)))

    return min(1.0, max(0.0, occupied)), rolloff, normalised_centroid


def _safe(value: float) -> float:
    """Coerce to a finite float; NaN/Inf in a signature poisons every match."""
    value = float(value)
    return value if math.isfinite(value) else 0.0


def extract_signature(
    samples: np.ndarray,
    sample_rate_hz: float,
    frequencies: Optional[np.ndarray] = None,
    powers: Optional[np.ndarray] = None,
) -> FingerprintVector:
    """Derive a gain-invariant feature vector from one IQ capture.

    ``samples`` is complex IQ. ``frequencies``/``powers`` are optional Welch
    output; when omitted the three spectral features are derived from the
    capture itself, so the vector always has the same length.
    """
    iq = np.asarray(samples)
    if iq.ndim != 1 or iq.size == 0:
        raise ValueError("samples must be a non-empty 1-D array")
    if sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be positive")

    envelope = np.abs(iq)
    mean_env = float(envelope.mean())
    std_env = float(envelope.std())
    cv = std_env / mean_env if mean_env > 1e-15 else 0.0

    # Normalised rise/fall: mean absolute sample-to-sample change over the mean
    # level. A compressor limiter produces smaller relative swings than a linear
    # PA, so this separates radiotypes without depending on absolute power.
    if envelope.size > 1:
        delta = np.diff(envelope)
        rise = float(np.mean(np.maximum(delta, 0.0)))
        fall = float(np.mean(np.maximum(-delta, 0.0)))
        step_scale = mean_env if mean_env > 1e-15 else 1.0
        rise_rate = rise / step_scale
        fall_rate = fall / step_scale
        # Lag-1 correlation of the envelope: memory in the modulation chain.
        centred = envelope - mean_env
        denom = float(np.sum(centred * centred))
        lag1 = float(np.sum(centred[:-1] * centred[1:]) / denom) if denom > 1e-24 else 0.0
    else:
        rise_rate = fall_rate = lag1 = 0.0

    # One magnitude array serves the envelope statistics, the power and the
    # crest factor. The old code recomputed abs(iq) for each.
    mean_power = float(np.mean(envelope * envelope))
    skew, kurtosis = _standardised_moments(envelope, mean_env, std_env)
    moment3, moment4 = _complex_moments(iq, mean_power)

    # Crest factor in dB: a heavily compressed signal sits close to 0 dB, a
    # linear one much lower. Ratio of powers, so gain cancels.
    crest_db = (
        10.0 * math.log10(max(mean_power, 1e-30) / max(float(np.max(envelope)) ** 2, 1e-30))
        if mean_power > 1e-30
        else 0.0
    )

    occupied_fraction, rolloff, centroid = _spectral_shape(
        iq, sample_rate_hz, frequencies, powers
    )

    values = (
        _safe(cv),
        _safe(skew),
        _safe(kurtosis),
        _safe(moment3),
        _safe(moment4),
        _safe(rise_rate),
        _safe(fall_rate),
        _safe(lag1),
        _safe(occupied_fraction),
        _safe(rolloff),
        _safe(centroid),
        _safe(crest_db),
    )
    return FingerprintVector(tuple(values))


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------


@dataclass
class LibraryEntry:
    """One known transmitter signature."""

    label: str
    vector: FingerprintVector
    device_id: str = ""
    vendor: str = ""
    model: str = ""
    frequency_hz: Optional[float] = None
    samples: int = 0
    #: Per-feature spread across the captures this signature was averaged
    #: from. Empty for hand-written or legacy entries, which simply means the
    #: matcher falls back to between-radio variance alone.
    spread: Tuple[float, ...] = ()
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entryId": self.entry_id,
            "label": self.label,
            "deviceId": self.device_id,
            "vendor": self.vendor,
            "model": self.model,
            "frequencyHz": self.frequency_hz,
            "samples": self.samples,
            "createdAt": self.created_at,
            "signature": self.vector.to_dict(),
            "spread": {
                name: float(value)
                for name, value in zip(self.vector.labels, self.spread)
            },
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "LibraryEntry":
        signature = payload.get("signature") or {}
        return cls(
            label=str(payload.get("label") or "unnamed"),
            vector=FingerprintVector.from_dict(signature),
            device_id=str(payload.get("deviceId") or ""),
            vendor=str(payload.get("vendor") or ""),
            model=str(payload.get("model") or ""),
            frequency_hz=(
                float(payload["frequencyHz"]) if payload.get("frequencyHz") else None
            ),
            samples=int(payload.get("samples") or 0),
            spread=tuple(
                float(v) for v in (payload.get("spread") or {}).values()
            ) or (),
            entry_id=str(payload.get("entryId") or uuid.uuid4())[:8],
            created_at=str(payload.get("createdAt") or datetime.utcnow().isoformat()),
        )


@dataclass
class SignatureMatch:
    """Result of comparing one observation against the library."""

    entry: LibraryEntry
    distance: float

    def to_dict(self) -> Dict[str, Any]:
        return {"entry": self.entry.to_dict(), "distance": round(self.distance, 6)}


@dataclass
class FingerprintLibrary:
    """Persistent set of known signatures with atomic JSON storage."""

    path: Optional[str] = None
    entries: List[LibraryEntry] = field(default_factory=list)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    # ------------------------------------------------------------ load/save

    @classmethod
    def load(cls, path: str) -> "FingerprintLibrary":
        library = cls(path=path)
        if not path or not os.path.exists(path):
            return library
        try:
            with open(path, "r", encoding="utf-8") as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load fingerprint library from %s: %s", path, exc)
            return library

        entries: List[LibraryEntry] = []
        for item in (raw.get("entries") if isinstance(raw, dict) else None) or []:
            try:
                entries.append(LibraryEntry.from_dict(item))
            except (KeyError, TypeError, ValueError) as exc:
                logger.warning("Dropping malformed library entry: %s", exc)
        with library._lock:
            library.entries = entries
        return library

    def save(self) -> Optional[str]:
        if not self.path:
            return None
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        with self._lock:
            payload = {
                "version": 1,
                "features": list(FEATURE_NAMES),
                "updatedAt": datetime.utcnow().isoformat(),
                "entries": [entry.to_dict() for entry in self.entries],
            }
        handle = tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=directory,
            prefix=".fingerprints-", suffix=".tmp", delete=False,
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
            raise ConfigurationError(f"failed to persist fingerprint library: {exc}") from exc
        return self.path

    # -------------------------------------------------------------- access

    def __len__(self) -> int:
        return len(self.entries)

    def add(self, entry: LibraryEntry) -> LibraryEntry:
        with self._lock:
            self.entries.append(entry)
        self.save()
        return entry

    def remove(self, entry_id: str) -> bool:
        with self._lock:
            before = len(self.entries)
            self.entries = [e for e in self.entries if e.entry_id != entry_id]
            removed = len(self.entries) != before
        if removed:
            self.save()
        return removed

    def get(self, entry_id: str) -> Optional[LibraryEntry]:
        with self._lock:
            for entry in self.entries:
                if entry.entry_id == entry_id:
                    return entry
        return None

    def features(self) -> Tuple[str, ...]:
        return FEATURE_NAMES


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------


def mean_signature(vectors: Sequence[FingerprintVector]) -> FingerprintVector:
    """Average several captures of the same transmitter into one signature.

    A single capture carries the receiver's noise realisation, which moves the
    features by more than the difference between two devices does for some of
    them. Averaging a short burst suppresses that common-mode term and is the
    difference between "same device, new capture" matching and being rejected.
    All captures must share a feature set.
    """
    if not vectors:
        raise ValueError("cannot average an empty set of signatures")
    labels = vectors[0].labels
    if any(vector.labels != labels for vector in vectors):
        raise ValueError("cannot average signatures with different feature sets")
    count = len(vectors)
    return FingerprintVector(
        tuple(
            _safe(sum(vector.values[index] for vector in vectors) / count)
            for index in range(len(labels))
        ),
        labels,
    )


def summarise(
    vectors: Sequence[FingerprintVector],
) -> Tuple[FingerprintVector, Tuple[float, ...]]:
    """Average captures and measure how much each feature moves between them.

    The second return value is the per-feature standard deviation across the
    captures that were averaged. It is the scale of "this is just noise from the
    receiver", and the matcher uses it so a feature that jitters on its own
    cannot veto an otherwise good match.
    """
    mean = mean_signature(vectors)
    count = len(vectors)
    if count < 2:
        return mean, tuple(0.0 for _ in mean.values)
    spread: List[float] = []
    for index in range(len(mean.values)):
        column = [vector.values[index] for vector in vectors]
        average = sum(column) / count
        variance = sum((v - average) ** 2 for v in column) / (count - 1)
        spread.append(math.sqrt(max(0.0, variance)))
    return mean, tuple(spread)


def _feature_scales(
    vectors: Sequence[FingerprintVector],
    spreads: Sequence[Sequence[float]],
) -> List[float]:
    """Per-feature comparison scale: total variance seen in the library.

    Combines how much a feature varies *between* reference radios with how much
    it varies *within* one radio's own captures. A feature that only varies
    between radios is informative; one that varies within a single radio is
    mostly noise and gets scaled up, shrinking its influence on the distance.
    """
    if not vectors:
        return []
    count = len(vectors[0].values)
    magnitudes = [
        max(abs(vector.values[index]) for vector in vectors) for index in range(count)
    ]
    scales: List[float] = []
    for index in range(count):
        column = [vector.values[index] for vector in vectors]
        average = sum(column) / len(column)
        between = sum((v - average) ** 2 for v in column) / max(1, len(column) - 1)
        within_values = [spread[index] for spread in spreads if index < len(spread)]
        within = (
            sum(within_values) / len(within_values) if within_values else 0.0
        )
        # Sum of the two independent variance contributions.
        total = math.sqrt(max(0.0, between + within * within))
        # Floor relative to the feature's own size. Without this, a feature the
        # library happens not to vary gets a scale near zero and divides
        # floating-point noise into a huge "difference", so an unchanged signal
        # lands at the maximum distance.
        floor = max(
            _FEATURE_SPREAD_FLOOR,
            _RELATIVE_SCALE_FLOOR * max(magnitudes[index], 1e-12),
        )
        scales.append(max(total, floor))
    return scales


def _standardiser(vectors: Sequence[FingerprintVector]) -> Tuple[List[float], List[float]]:
    """Per-dimension mean and scale derived from the library itself.

    Comparing raw features by cosine is dominated by whichever feature happens
    to have the largest magnitude, so a signature can match a stranger on that
    one dimension while differing on everything that distinguishes devices.
    Z-scoring each dimension against the library's own spread gives every
    feature equal say.

    A dimension the library does not vary (spread below
    :data:`_FEATURE_SPREAD_FLOOR`) is handed a zero centre and unit scale so it
    contributes nothing, rather than being amplified out of pure noise.
    """
    if not vectors:
        return [], []
    labels = vectors[0].labels
    count = len(labels)
    means: List[float] = []
    scales: List[float] = []
    for index in range(count):
        column = [vector.values[index] for vector in vectors]
        mean = sum(column) / len(column)
        variance = sum((v - mean) ** 2 for v in column) / max(1, len(column) - 1)
        spread = math.sqrt(max(0.0, variance))
        if spread < _FEATURE_SPREAD_FLOOR:
            means.append(0.0)
            scales.append(1.0)
        else:
            means.append(mean)
            scales.append(spread)
    return means, scales


def _apply_standardiser(
    vector: FingerprintVector,
    means: Sequence[float],
    scales: Sequence[float],
) -> FingerprintVector:
    """Project one vector onto the library's standardised feature space."""
    return FingerprintVector(
        tuple(
            (value - mean) / scale
            for value, mean, scale in zip(vector.values, means, scales)
        ),
        vector.labels,
    )


def match_signature(
    observation: FingerprintVector,
    library: FingerprintLibrary,
    accept_threshold: float = DEFAULT_ACCEPT_THRESHOLD,
    min_separation: float = MIN_SEPARATION,
) -> Dict[str, Any]:
    """Rank the library against one observation.

    The returned dict always states what it is: ``identified`` only when a
    single candidate is both close enough and clearly separated from the rest.
    """
    comparable = [
        entry.vector
        for entry in library.entries
        if entry.vector.labels == observation.labels
    ]
    skipped = len(library.entries) - len(comparable)
    if skipped:
        # An entry written against a different feature set has no comparable
        # meaning; skip it explicitly rather than letting it corrupt the ranking.
        logger.warning(
            "Skipping %d library entr%s with a different feature set",
            skipped, "y" if skipped == 1 else "ies",
        )

    candidates: List[SignatureMatch] = []
    scales: List[float] = []
    if comparable:
        scales = _feature_scales(
            comparable, [entry.spread for entry in library.entries]
        )
        for entry in library.entries:
            if entry.vector.labels != observation.labels:
                continue
            candidates.append(
                SignatureMatch(
                    entry=entry,
                    distance=scaled_distance(observation, entry.vector, scales),
                )
            )

    candidates.sort(key=lambda m: m.distance)
    scales_are_usable = any(
        scale > _FEATURE_SPREAD_FLOOR for scale in (scales or [])
    )
    payload: Dict[str, Any] = {
        "identified": False,
        "status": "unknown",
        "reason": "The library is empty; capture a reference signature first.",
        "threshold": accept_threshold,
        "features": list(observation.labels),
        "values": {name: float(v) for name, v in zip(observation.labels, observation.values)},
        "librarySize": len(library.entries),
        "skippedIncompatible": skipped,
        "featureScales": {
            name: round(float(scale), 8)
            for name, scale in zip(observation.labels, scales or ())
        },
        "candidates": [m.to_dict() for m in candidates[:5]],
    }
    if not candidates:
        return payload

    best = candidates[0]
    runner_up = candidates[1] if len(candidates) > 1 else None

    if not scales_are_usable:
        # Every reference is the same signature, so there is no variation to
        # measure against. That is a library with nothing to say, not a
        # non-match, and calling it "unknown" would be misleading.
        payload["status"] = "indeterminate"
        payload["reason"] = (
            "The library holds no distinct references, so this capture cannot be "
            "compared with anything. Add a reference from a different device."
        )
        payload["bestDistance"] = round(best.distance, 6)
        return payload

    if best.distance > accept_threshold:
        payload["status"] = "no-match"
        payload["reason"] = (
            f"Closest known signature ({best.entry.label}) is {best.distance:.3f} away, "
            f"beyond the {accept_threshold:.2f} accept threshold "
            f"(RMS feature difference in units of the library's spread). "
            f"This transmitter is probably not in the library."
        )
        payload["bestDistance"] = round(best.distance, 6)
        return payload

    if runner_up is not None and (runner_up.distance - best.distance) < min_separation:
        payload["status"] = "ambiguous"
        payload["reason"] = (
            f"{best.entry.label} and {runner_up.entry.label} are "
            f"{(runner_up.distance - best.distance):.3f} apart, which is too close to "
            f"separate reliably. Add more captures of the real device."
        )
        payload["bestDistance"] = round(best.distance, 6)
        return payload

    payload["identified"] = True
    payload["status"] = "identified"
    payload["confidence"] = round(max(0.0, min(1.0, 1.0 - best.distance)), 4)
    payload["bestDistance"] = round(best.distance, 6)
    payload["match"] = best.entry.to_dict()
    payload["reason"] = (
        f"Closest signature is {best.entry.label} at {best.distance:.3f}."
        + (
            f" Next candidate {runner_up.entry.label} is {runner_up.distance:.3f} away."
            if runner_up is not None
            else ""
        )
    )
    return payload