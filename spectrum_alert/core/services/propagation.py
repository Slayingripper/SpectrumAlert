"""Radio path-loss and geodesic helper functions.

Ported from the ``RTL_Triangulator`` project so that SpectrumAlert can convert
between RSSI and distance, pick sane propagation defaults per band, and convert
between metres and WGS84 degrees.

The log-distance path-loss model used throughout is::

    RSSI(d) = P0 - 10 * n * log10(d)

where ``P0`` is the received power at 1 m (dBm) and ``n`` is the path-loss
exponent. Inverting gives the distance estimate used for geolocation::

    d = 10 ** ((P0 - RSSI) / (10 * n))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

__all__ = [
    "EARTH_RADIUS_M",
    "METERS_PER_LATITUDE_DEGREE",
    "PathLossDefaults",
    "haversine_meters",
    "meters_to_latitude_degrees",
    "meters_to_longitude_degrees",
    "rssi_to_distance_meters",
    "distance_to_rssi",
    "clamp_distance_meters",
    "frequency_band_defaults",
    "frequency_default_path_loss",
    "validate_path_loss_exponent",
]

#: Mean Earth radius (IUGG) in metres.
EARTH_RADIUS_M = 6_371_000.0

#: Metres per degree of latitude (spherical approximation).
METERS_PER_LATITUDE_DEGREE = 111_320.0

#: Distance floor used by the inverse model. Below roughly this the log-distance
#: model is invalid anyway (near-field), but keeping a floor stops ``log10(0)``.
MIN_MODEL_DISTANCE_M = 0.5

#: Distance ceiling used by the inverse model (500 km).
MAX_MODEL_DISTANCE_M = 500_000.0


@dataclass(frozen=True)
class PathLossDefaults:
    """Frequency-band specific path-loss defaults."""

    exponent: float
    reference_rssi_dbm: float
    band: str


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two WGS84 coordinates in metres."""
    phi1 = math.radians(float(lat1))
    phi2 = math.radians(float(lat2))
    d_phi = phi2 - phi1
    d_lambda = math.radians(float(lon2) - float(lon1))

    a = (
        math.sin(d_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    )
    # Guard against floating point overshoot past 1.0.
    a = min(1.0, max(0.0, a))
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_M * c


def meters_to_latitude_degrees(meters: float) -> float:
    """Convert a north/south offset in metres to degrees of latitude."""
    return float(meters) / METERS_PER_LATITUDE_DEGREE


def meters_to_longitude_degrees(meters: float, latitude: float) -> float:
    """Convert an east/west offset in metres to degrees of longitude at ``latitude``."""
    denominator = METERS_PER_LATITUDE_DEGREE * math.cos(math.radians(float(latitude)))
    if denominator == 0:
        return 0.0
    return float(meters) / denominator


def rssi_to_distance_meters(
    rssi_dbm: float, reference_rssi_dbm: float, path_loss_exponent: float
) -> float:
    """Invert the log-distance path-loss model.

    ``reference_rssi_dbm`` is the power received at 1 m. Returns the *raw*
    (unclamped) distance so callers can weight by proximity.
    """
    exponent = 10.0 * float(path_loss_exponent)
    if exponent == 0:
        raise ValueError("path_loss_exponent must be non-zero")
    return math.pow(10.0, (float(reference_rssi_dbm) - float(rssi_dbm)) / exponent)


def distance_to_rssi(
    distance_meters: float, reference_rssi_dbm: float, path_loss_exponent: float
) -> float:
    """Forward log-distance path-loss model, in dBm."""
    return float(reference_rssi_dbm) - 10.0 * float(path_loss_exponent) * math.log10(
        max(0.01, float(distance_meters))
    )


def clamp_distance_meters(
    distance_meters: float, minimum: float = MIN_MODEL_DISTANCE_M, maximum: float = MAX_MODEL_DISTANCE_M
) -> float:
    """Clamp a modelled distance into the range the path-loss model is trusted over."""
    return max(float(minimum), min(float(maximum), float(distance_meters)))


def frequency_band_defaults(frequency_hz: float) -> str:
    """Return a coarse band label for a centre frequency."""
    f_mhz = float(frequency_hz) / 1e6
    if f_mhz < 30:
        return "hf"
    if f_mhz < 100:
        return "vhf-low"
    if f_mhz < 300:
        return "vhf-high"
    if f_mhz < 1000:
        return "uhf"
    return "shf"


def frequency_default_path_loss(frequency_hz: float) -> PathLossDefaults:
    """Frequency-appropriate default path-loss exponent and 1 m reference power.

    Typical values, blending urban/rural literature for line-of-sight terrestrial
    paths:

    ============  =====  ========
    Band          n      P0 dBm
    ============  =====  ========
    < 100 MHz     3.2   -35
    100-300 MHz   3.0   -40
    300-1000 MHz  3.5   -45
    > 1000 MHz    4.0   -50
    ============  =====  ========
    """
    f_mhz = float(frequency_hz) / 1e6
    band = frequency_band_defaults(frequency_hz)
    if f_mhz < 100:
        return PathLossDefaults(exponent=3.2, reference_rssi_dbm=-35.0, band=band)
    if f_mhz < 300:
        return PathLossDefaults(exponent=3.0, reference_rssi_dbm=-40.0, band=band)
    if f_mhz < 1000:
        return PathLossDefaults(exponent=3.5, reference_rssi_dbm=-45.0, band=band)
    return PathLossDefaults(exponent=4.0, reference_rssi_dbm=-50.0, band=band)


def validate_path_loss_exponent(
    exponent: float, frequency_hz: float, low: float = 1.5, high: float = 4.5
) -> float:
    """Return ``exponent`` if physically plausible, otherwise the band default."""
    value = float(exponent)
    if not math.isfinite(value) or value < low or value > high:
        return frequency_default_path_loss(frequency_hz).exponent
    return value


def bounding_box(
    points: Tuple[Tuple[float, float], ...], pad_ratio: float = 0.35, pad_meters: float = 500.0
) -> Tuple[float, float, float, float]:
    """Return a padded ``(min_lat, min_lon, max_lat, max_lon)`` around ``points``."""
    if not points:
        raise ValueError("at least one point is required")

    lats = [float(p[0]) for p in points]
    lons = [float(p[1]) for p in points]
    min_lat, max_lat = min(lats), max(lats)
    min_lon, max_lon = min(lons), max(lons)
    mid_lat = (min_lat + max_lat) / 2.0

    lat_pad = max((max_lat - min_lat) * pad_ratio, meters_to_latitude_degrees(pad_meters))
    lon_pad = max(
        (max_lon - min_lon) * pad_ratio,
        meters_to_longitude_degrees(pad_meters, mid_lat),
    )
    return (
        min_lat - lat_pad,
        min_lon - lon_pad,
        max_lat + lat_pad,
        max_lon + lon_pad,
    )