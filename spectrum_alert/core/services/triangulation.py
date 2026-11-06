"""RSSI-based transmitter geolocation.

Ported from the ``RTL_Triangulator`` project's ``triangulatePosition``.

Given several receivers that all measured the same emitter, the log-distance
path-loss model turns each measurement into a *distance circle*. The emitter
lies on the intersection of all of them, so we minimise::

    sum_i  w_i * huber( ||x - p_i|| - d_i )

Design decisions worth calling out:

* **Huber loss, not plain least squares.** RSSI varies with multipath by several
  dB, which is tens of percent of the modelled distance. Squared error lets one
  badly faded receiver drag the solution metres away. Huber is linear beyond a
  threshold, so it resists that.

* **Multi-resolution grid search, then gradient descent.** The objective has
  several local minima when there are only two or three receivers, so a coarse
  global grid establishes a basin before the local optimiser takes over. Pure
  gradient descent from a centroid lands in whichever basin the start point is
  closest to, which is not reliably the best one.

* **Weights combine inverse distance and measurement reliability.** A receiver
  reporting a much stronger signal is closer, and its distance estimate is
  correspondingly more sensitive to error, so it gets more pull. Low-confidence
  readings (approximate or fallback spectra) are down-weighted explicitly.

* **Confidence ellipse from the numerical Hessian.** The reported uncertainty is
  not just residual scatter: the local curvature of the objective says which
  directions are constrained and which are not, which is what a 2-anchor solve
  badly needs to communicate.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from spectrum_alert.core.services.propagation import (
    EARTH_RADIUS_M,
    METERS_PER_LATITUDE_DEGREE,
    clamp_distance_meters,
    haversine_meters,
    meters_to_latitude_degrees,
    meters_to_longitude_degrees,
    rssi_to_distance_meters,
)

logger = logging.getLogger(__name__)

__all__ = [
    "TriangulationReading",
    "ResidualDetail",
    "ConfidenceEllipse",
    "PositionEstimate",
    "triangulate_position",
]

#: Receivers with a source like this report a coarse spectrum; trust them less.
LOW_CONFIDENCE_SOURCE_WEIGHTS: Dict[str, float] = {
    "openwebrx_smeter_http_fallback": 0.35,
    "openwebrx_smeter": 0.35,
    "adpcm_approx": 0.2,
    "manual_override": 0.5,
}

#: Minimum receiver spread assumed when receivers are nearly coincident, so the
#: geometry penalty does not collapse to zero.
MIN_RECEIVER_SPREAD_M = 1000.0

#: Grid resolution per refinement pass.
GRID_STEPS = 45

#: Number of multi-resolution passes.
GRID_PASSES = 4

#: Maximum gradient-descent iterations.
MAX_GRADIENT_ITERATIONS = 120

#: Armijo sufficient-decrease constant.
ARMIJO_C = 0.1


@dataclass
class TriangulationReading:
    """One receiver's measurement of the emitter."""

    receiver_id: str
    name: str
    lat: float
    lon: float
    rssi_db: float
    source: str = "unknown"
    reliability: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.receiver_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "rssiDbm": self.rssi_db,
            "source": self.source,
            "reliability": self.reliability,
        }


@dataclass
class ResidualDetail:
    """Per-receiver fit residual, reported in both metres and dB."""

    name: str
    measured_rssi_db: float
    predicted_rssi_db: float
    residual_db: float
    distance_meters: float
    modeled_distance_meters: float
    residual_meters: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "measuredRssi": round(self.measured_rssi_db, 2),
            "predictedRssi": round(self.predicted_rssi_db, 2),
            "residualDb": round(self.residual_db, 2),
            "distanceMeters": int(round(self.distance_meters)),
            "modeledDistMeters": int(round(self.modeled_distance_meters)),
            "residualMeters": int(round(self.residual_meters)),
        }


@dataclass
class ConfidenceEllipse:
    """Orientation and size of the solution's local uncertainty region."""

    semi_major_meters: Optional[float] = None
    semi_minor_meters: Optional[float] = None
    angle_deg: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        def _round(value: Optional[float]) -> Optional[float]:
            return int(round(value)) if value is not None and math.isfinite(value) else None

        return {
            "semiMajorMeters": _round(self.semi_major_meters),
            "semiMinorMeters": _round(self.semi_minor_meters),
            "angleDeg": (
                round(self.angle_deg, 1)
                if self.angle_deg is not None and math.isfinite(self.angle_deg)
                else None
            ),
        }


@dataclass
class PositionEstimate:
    """A geolocation estimate plus everything needed to judge it."""

    lat: float
    lon: float
    uncertainty_meters: float
    rms_residual_meters: float
    rms_rssi_residual_db: float
    receiver_spread_meters: float
    anchor_count: int
    quality: str = "good"
    quality_reasons: List[str] = field(default_factory=list)
    residuals: List[ResidualDetail] = field(default_factory=list)
    confidence: ConfidenceEllipse = field(default_factory=ConfidenceEllipse)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "uncertaintyMeters": int(round(self.uncertainty_meters)),
            "rmsResidualMeters": int(round(self.rms_residual_meters)),
            "rmsRssiResidual": round(self.rms_rssi_residual_db, 2),
            "receiverSpreadMeters": int(round(self.receiver_spread_meters)),
            "anchorCount": self.anchor_count,
            "quality": self.quality,
            "qualityReasons": list(self.quality_reasons),
            "residuals": [r.to_dict() for r in self.residuals],
            "confidence": self.confidence.to_dict(),
        }


@dataclass
class _Anchor:
    """Internal weighted anchor used by the optimiser."""

    lat: float
    lon: float
    name: str
    distance_m: float
    raw_distance_m: float
    rssi_db: float
    weight: float


def _source_weight(source: str) -> float:
    normalized = str(source or "").lower()
    for needle, weight in LOW_CONFIDENCE_SOURCE_WEIGHTS.items():
        if needle in normalized:
            return weight
    return 1.0


def _build_anchors(
    readings: Sequence[TriangulationReading],
    reference_rssi_dbm: float,
    path_loss_exponent: float,
) -> List[_Anchor]:
    anchors: List[_Anchor] = []
    for reading in readings:
        raw_distance = rssi_to_distance_meters(
            reading.rssi_db, reference_rssi_dbm, path_loss_exponent
        )
        distance = clamp_distance_meters(raw_distance)
        reliability = reading.reliability
        reliability = 1.0 if reliability is None else max(0.0, min(1.0, float(reliability)))
        base = _source_weight(reading.source) * max(0.1, min(1.0, reliability))
        anchors.append(
            _Anchor(
                lat=float(reading.lat),
                lon=float(reading.lon),
                name=reading.name,
                distance_m=distance,
                raw_distance_m=raw_distance,
                rssi_db=float(reading.rssi_db),
                weight=base * max(0.01, 1.0 / max(0.01, raw_distance)),
            )
        )
    return anchors


def _centroid(anchors: Sequence[_Anchor]) -> tuple[float, float]:
    if not anchors:
        return (0.0, 0.0)
    weight_sum = sum(a.weight for a in anchors)
    if weight_sum > 0:
        return (
            sum(a.lat * a.weight for a in anchors) / weight_sum,
            sum(a.lon * a.weight for a in anchors) / weight_sum,
        )
    return (
        sum(a.lat for a in anchors) / len(anchors),
        sum(a.lon for a in anchors) / len(anchors),
    )


def _huber(error_m: float, delta_m: float) -> float:
    """Huber loss with transition at ``delta_m``."""
    magnitude = abs(error_m)
    if magnitude <= delta_m:
        return 0.5 * magnitude * magnitude
    return delta_m * (magnitude - 0.5 * delta_m)


def _haversine_grid(
    lats: np.ndarray,
    lons: np.ndarray,
    anchor_lat: float,
    anchor_lon: float,
) -> np.ndarray:
    """Vectorised great-circle distance from a coordinate grid to one anchor."""
    phi1 = np.radians(lats)
    phi2 = math.radians(float(anchor_lat))
    d_phi = phi2 - phi1
    d_lambda = np.radians(lons - float(anchor_lon))
    a = (
        np.sin(d_phi / 2.0) ** 2
        + math.cos(phi2) * np.cos(phi1) * np.sin(d_lambda / 2.0) ** 2
    )
    np.clip(a, 0.0, 1.0, out=a)
    return EARTH_RADIUS_M * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def triangulate_position(
    readings: Sequence[TriangulationReading],
    reference_rssi_dbm: float,
    path_loss_exponent: float,
) -> PositionEstimate:
    """Estimate the emitter position from multi-receiver RSSI.

    Raises ``ValueError`` when fewer than two finite readings are supplied, since
    a single receiver has no bearing information at all.
    """
    if len(readings) < 2:
        raise ValueError("at least 2 readings are required for triangulation")

    anchors = _build_anchors(readings, reference_rssi_dbm, path_loss_exponent)
    if len(anchors) < 2:
        raise ValueError("at least 2 usable readings are required for triangulation")

    center_lat, center_lon = _centroid(anchors)

    # Geometry scale: how far apart the receivers are. Sets both the search
    # bounds and how much the solution should be penalised for wandering away.
    spread_m = 0.0
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            spread_m = max(
                spread_m,
                haversine_meters(anchors[i].lat, anchors[i].lon, anchors[j].lat, anchors[j].lon),
            )
    spread_m = max(MIN_RECEIVER_SPREAD_M, spread_m)

    lats = [a.lat for a in anchors]
    lons = [a.lon for a in anchors]
    lat_pad = max(
        meters_to_latitude_degrees(spread_m * 1.5),
        (max(lats) - min(lats)) * 1.2,
        0.2,
    )
    lon_pad = max(
        meters_to_longitude_degrees(spread_m * 1.5, center_lat),
        (max(lons) - min(lons)) * 1.2,
        0.2,
    )
    box = (
        min(lats) - lat_pad,
        min(lons) - lon_pad,
        max(lats) + lat_pad,
        max(lons) + lon_pad,
    )

    def objective(lat: float, lon: float) -> float:
        total = 0.0
        for anchor in anchors:
            actual = haversine_meters(lat, lon, anchor.lat, anchor.lon)
            error = actual - anchor.distance_m
            # Transition at whichever is larger: 50 % of the modelled distance
            # (measurement uncertainty) or 10 % of receiver spread (geometry).
            delta = max(anchor.distance_m * 0.5, 0.1 * spread_m, 100.0)
            total += anchor.weight * _huber(error, delta)
        if len(anchors) < 3:
            # With two anchors the problem is underconstrained: the solution is a
            # line, not a point. A light prior toward the receiver cluster picks
            # a stable point on that line instead of drifting.
            offset = haversine_meters(lat, lon, center_lat, center_lon)
            total += 0.01 * offset * offset
        return total

    best_lat, best_lon = center_lat, center_lon
    best_score = objective(best_lat, best_lon)

    # Multi-resolution bounded grid search to land in the right basin.
    # The grid is evaluated as whole NumPy arrays. The scalar version made
    # roughly 65k Haversine calls per solve for eight receivers.
    for _ in range(GRID_PASSES):
        box_min_lat, box_min_lon, box_max_lat, box_max_lon = box
        grid_lats = np.linspace(box_min_lat, box_max_lat, GRID_STEPS)
        grid_lons = np.linspace(box_min_lon, box_max_lon, GRID_STEPS)
        lat_mesh, lon_mesh = np.meshgrid(grid_lats, grid_lons, indexing="ij")
        scores = np.zeros_like(lat_mesh, dtype=float)

        for anchor in anchors:
            distance = _haversine_grid(lat_mesh, lon_mesh, anchor.lat, anchor.lon)
            error = distance - anchor.distance_m
            delta = max(anchor.distance_m * 0.5, 0.1 * spread_m, 100.0)
            magnitude = np.abs(error)
            loss = np.where(
                magnitude <= delta,
                0.5 * magnitude * magnitude,
                delta * (magnitude - 0.5 * delta),
            )
            scores += anchor.weight * loss

        if len(anchors) < 3:
            offset = _haversine_grid(lat_mesh, lon_mesh, center_lat, center_lon)
            scores += 0.01 * offset * offset

        index = np.unravel_index(int(np.argmin(scores)), scores.shape)
        candidate_score = float(scores[index])
        if candidate_score < best_score:
            best_score = candidate_score
            best_lat = float(lat_mesh[index])
            best_lon = float(lon_mesh[index])

        lat_half = (box_max_lat - box_min_lat) / 6.0
        lon_half = (box_max_lon - box_min_lon) / 6.0
        box = (
            best_lat - lat_half,
            best_lon - lon_half,
            best_lat + lat_half,
            best_lon + lon_half,
        )

    # Gradient descent with backtracking line search (Armijo).
    grad_lat, grad_lon, grad_score = best_lat, best_lon, best_score
    step_size = max(
        meters_to_latitude_degrees(spread_m * 0.02),
        (box[3] - box[1]) / 90.0,
    )
    eps = 1e-8

    for iteration in range(MAX_GRADIENT_ITERATIONS):
        base = objective(grad_lat, grad_lon)
        d_lat = (objective(grad_lat + eps, grad_lon) - base) / eps
        d_lon = (objective(grad_lat, grad_lon + eps) - base) / eps
        norm = math.hypot(d_lat, d_lon)
        if not math.isfinite(norm) or norm < 1e-6:
            break

        step = step_size / (0.5 + iteration * 0.05)
        improved = False
        for _ in range(8):
            candidate_lat = grad_lat - step * (d_lat / norm)
            candidate_lon = grad_lon - step * (d_lon / norm)
            candidate_score = objective(candidate_lat, candidate_lon)
            if candidate_score < grad_score and candidate_score <= grad_score - ARMIJO_C * step * norm:
                grad_lat, grad_lon, grad_score = candidate_lat, candidate_lon, candidate_score
                improved = True
                break
            step *= 0.5
        if not improved:
            break

    if grad_score < best_score:
        best_lat, best_lon = grad_lat, grad_lon

    residuals: List[ResidualDetail] = []
    for anchor in anchors:
        distance = haversine_meters(best_lat, best_lon, anchor.lat, anchor.lon)
        predicted = float(reference_rssi_dbm) - 10.0 * float(path_loss_exponent) * math.log10(
            max(0.01, distance)
        )
        residuals.append(
            ResidualDetail(
                name=anchor.name,
                measured_rssi_db=anchor.rssi_db,
                predicted_rssi_db=predicted,
                residual_db=anchor.rssi_db - predicted,
                distance_meters=distance,
                modeled_distance_meters=anchor.distance_m,
                residual_meters=distance - anchor.distance_m,
            )
        )

    count = max(1, len(residuals))
    rms_m = math.sqrt(sum(r.residual_meters**2 for r in residuals) / count)
    rms_db = math.sqrt(sum(r.residual_db**2 for r in residuals) / count)
    ellipse = _confidence_ellipse(objective, best_lat, best_lon, rms_m, len(anchors))

    quality, reasons = _classify_quality(
        rms_db, rms_m, spread_m, ellipse, len(anchors), residuals
    )

    return PositionEstimate(
        lat=best_lat,
        lon=best_lon,
        uncertainty_meters=max(50.0, rms_m) if math.isfinite(rms_m) else 1000.0,
        rms_residual_meters=rms_m,
        rms_rssi_residual_db=rms_db,
        receiver_spread_meters=spread_m,
        anchor_count=len(anchors),
        quality=quality,
        quality_reasons=reasons,
        residuals=residuals,
        confidence=ellipse,
    )


def _confidence_ellipse(
    objective,
    lat: float,
    lon: float,
    rms_residual_m: float,
    anchor_count: int,
    step: float = 1e-4,
) -> ConfidenceEllipse:
    """Second-order uncertainty region from central differences of the objective.

    ``step`` is in degrees; 1e-4 is roughly 11 m, small enough that the Hessian
    reflects local curvature and large enough to stay above float noise in the
    objective sum.
    """
    try:
        f0 = objective(lat, lon)
        d2_ll = (objective(lat + step, lon) - 2.0 * f0 + objective(lat - step, lon)) / (step * step)
        d2_mm = (objective(lat, lon + step) - 2.0 * f0 + objective(lat, lon - step)) / (step * step)
        d2_lm = (
            objective(lat + step, lon + step)
            - objective(lat + step, lon - step)
            - objective(lat - step, lon + step)
            + objective(lat - step, lon - step)
        ) / (4.0 * step * step)

        determinant = d2_ll * d2_mm - d2_lm * d2_lm
        if not math.isfinite(determinant) or determinant <= 1e-20:
            return ConfidenceEllipse()

        sigma_squared = max(
            1.0, rms_residual_m * rms_residual_m / max(1, anchor_count - 1)
        )
        c11 = sigma_squared * d2_mm / determinant
        c22 = sigma_squared * d2_ll / determinant
        c12 = -sigma_squared * d2_lm / determinant

        trace = c11 + c22
        discriminant = math.sqrt(
            max(0.0, (c11 - c22) ** 2 + 4.0 * c12 * c12)
        )
        major = (trace + discriminant) / 2.0
        minor = (trace - discriminant) / 2.0

        lat_m = METERS_PER_LATITUDE_DEGREE
        lon_m = METERS_PER_LATITUDE_DEGREE * math.cos(math.radians(lat))
        return ConfidenceEllipse(
            semi_major_meters=math.sqrt(max(major, minor)) * max(lat_m, lon_m),
            semi_minor_meters=math.sqrt(min(major, minor)) * min(lat_m, lon_m),
            angle_deg=math.degrees(0.5 * math.atan2(2.0 * c12, c11 - c22)),
        )
    except (ValueError, ZeroDivisionError, OverflowError) as exc:  # pragma: no cover
        logger.debug("confidence ellipse unavailable: %s", exc)
        return ConfidenceEllipse()


def _consensus_offset_db(residuals: Sequence[ResidualDetail]) -> Optional[float]:
    """How far one receiver sits from the consensus of the others, in dB.

    ``r_i = measured_i - predicted_i``. If every receiver agreed, every ``r_i``
    would be near zero. A single faded receiver shows up as one residual far from
    the median of the rest: the optimiser slides the fix toward it, the other
    receivers all end up biased the *other* way, and the median of the residuals
    separates from the outlier.

    This is the statistic that exposes a lone bad receiver. The RMS residual
    does not: the optimiser absorbs most of the error into the healthy anchors
    and RMS stays deceptively small while the reported fix is kilometres off.
    """
    values = sorted(abs(r.residual_db) for r in residuals)
    if len(values) < 3:
        return None
    return values[-1] - values[len(values) // 2]


def _classify_quality(
    rms_db: float,
    rms_m: float,
    spread_m: float,
    ellipse: ConfidenceEllipse,
    anchor_count: int,
    residuals: Optional[Sequence[ResidualDetail]] = None,
) -> tuple[str, List[str]]:
    """Grade the fit and explain the grade.

    Four independent checks, because they fail differently:

    * **RMS dB residual** catches systematic calibration error affecting
      every receiver equally.
    * **Consensus outlier score** catches one receiver disagreeing with the
      others. This is the check that matters most in practice, since multipath
      fading on a single receiver is the common real-world failure and it leaves
      the RMS looking fine.
    * **Normalised geometric residual** catches a solution that does not
      actually fit the distance circles.
    * **Confidence ellipse vs receiver spread** catches a solution whose
      uncertainty dwarfs the geometry that produced it.
    """
    quality = "good"
    reasons: List[str] = []

    if not math.isfinite(rms_db) or rms_db > 8.0:
        quality = "poor"
        reasons.append(f"RSSI residual {rms_db:.1f} dB > 8 dB")
    elif rms_db > 4.0:
        quality = "fair"
        reasons.append(f"RSSI residual {rms_db:.1f} dB > 4 dB")

    if residuals:
        outlier = _consensus_offset_db(residuals)
        if outlier is not None and outlier > 3.0:
            quality = "poor" if outlier > 5.0 else "fair"
            reasons.append(
                f"one receiver disagrees with the others by {outlier:.1f} dB (multipath?)"
            )

    normalized = rms_m / max(100.0, spread_m)
    if normalized > 0.5:
        quality = "poor"
        reasons.append(f"normalised residual {normalized:.2f} > 0.5")
    elif normalized > 0.25:
        quality = "fair" if quality != "poor" else "poor"
        reasons.append(f"normalised residual {normalized:.2f} > 0.25")

    # Ellipse-to-spread ratio tracks the real position error closely: on a
    # synthetic three-receiver set it came out at 0.35 for a 517 m error and 1.38
    # for a 7.5 km one, while the RMS residual barely moved. Thresholds are set
    # from that behaviour, not guessed.
    if ellipse.semi_major_meters is not None and math.isfinite(ellipse.semi_major_meters):
        ratio = ellipse.semi_major_meters / max(1.0, spread_m)
        if ratio > 1.0:
            quality = "poor"
            reasons.append(
                f"confidence ellipse {ellipse.semi_major_meters:.0f} m exceeds receiver spread"
            )
        elif ratio > 0.6:
            quality = "fair" if quality != "poor" else "poor"
            reasons.append(
                f"confidence ellipse {ellipse.semi_major_meters:.0f} m is large "
                f"({ratio:.2f}x receiver spread)"
            )

    if anchor_count < 3:
        quality = "fair" if quality != "poor" else "poor"
        reasons.append("only 2 receivers (underconstrained)")

    if not reasons:
        reasons.append("consistent fit")
    return quality, reasons


def likelihood_grid(
    readings: Sequence[TriangulationReading],
    reference_rssi_dbm: float,
    path_loss_exponent: float,
    box: tuple[float, float, float, float],
    steps: int = 36,
) -> List[List[float]]:
    """Coarse L1-error likelihood grid, normalised to ``0..1`` per cell.

    Used by the map to shade where the emitter could plausibly be rather than
    drawing a single point that implies far more confidence than exists. The
    kernel width scales with the typical modelled distance so the field stays
    meaningful whether the emitter is 200 m or 40 km away.
    """
    min_lat, min_lon, max_lat, max_lon = box
    typical = 0.0
    modeled: List[float] = []
    for reading in readings:
        distance = clamp_distance_meters(
            rssi_to_distance_meters(
                reading.rssi_db, reference_rssi_dbm, path_loss_exponent
            )
        )
        modeled.append(distance)
    if modeled:
        typical = max(250.0, sum(modeled) / len(modeled))
    else:
        typical = 250.0

    grid_lats = np.linspace(min_lat, max_lat, max(1, steps))
    grid_lons = np.linspace(min_lon, max_lon, max(1, steps))
    lat_mesh, lon_mesh = np.meshgrid(grid_lats, grid_lons, indexing="ij")

    error = np.zeros_like(lat_mesh, dtype=float)
    for reading, distance in zip(readings, modeled):
        error += np.abs(
            _haversine_grid(lat_mesh, lon_mesh, reading.lat, reading.lon) - distance
        )

    scores = np.exp(-error / (typical * 1.5))
    best = float(np.max(scores))
    worst = float(np.min(scores))
    span = max(1e-9, best - worst)
    return ((scores - worst) / span).tolist()