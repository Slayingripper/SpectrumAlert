"""Domain models for geolocated receivers.

A *receiver* is a monitoring site: somewhere an SDR is listening. A receiver
carries a position, a transport (local dongle, remote ``rtl_tcp`` server) and
tuning parameters. Multiple receivers observing the same emitter is what makes
geolocation possible, so this is the aggregate the calibration and triangulation
services operate on.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator

from spectrum_alert.core.services.propagation import haversine_meters


class ReceiverMode(str, Enum):
    """Transport used to reach a receiver's SDR."""

    #: RTL-SDR attached directly to this machine (pyrtlsdr / rtl_power).
    LOCAL = "local"
    #: A remote host running ``rtl_tcp``; IQ is streamed over TCP.
    RTLTCP = "rtltcp"


class ReceiverSource(str, Enum):
    """Provenance of a scan, used to weight its trustworthiness."""

    RTL_POWER = "rtl_power"
    RTL_TCP_IQ = "rtl_tcp_iq"
    MANUAL_OVERRIDE = "manual_override"
    MOCK = "mock"
    UNKNOWN = "unknown"


class SDRReceiver(BaseModel):
    """A single monitoring site."""

    id: str = Field(..., description="Stable identifier")
    name: str = Field(..., min_length=1, description="Human-readable label")
    mode: ReceiverMode = Field(default=ReceiverMode.LOCAL, description="Transport to the SDR")
    lat: float = Field(..., ge=-90.0, le=90.0, description="Latitude in degrees")
    lon: float = Field(..., ge=-180.0, le=180.0, description="Longitude in degrees")

    # Transport-specific
    device_index: Optional[int] = Field(None, ge=0, description="Local dongle index")
    host: Optional[str] = Field(None, description="rtl_tcp host")
    port: Optional[int] = Field(None, ge=1, le=65535, description="rtl_tcp port")

    # Tuning
    gain_db: Optional[float] = Field(None, description="Manual gain in dB")
    ppm_error: Optional[float] = Field(None, description="Crystal error correction in ppm")
    bandwidth_hz: Optional[float] = Field(None, gt=0, description="Analog front-end bandwidth")

    #: When set, every scan for this receiver reports this level instead of
    #: measuring. Useful for sites that are known but currently unreachable.
    manual_rssi_db: Optional[float] = Field(
        None, description="Pin the receiver to a fixed dBFS level instead of measuring"
    )

    enabled: bool = Field(default=True, description="Include in scans and geolocation")
    altitude_m: Optional[float] = Field(None, description="Optional site altitude")
    notes: Optional[str] = Field(None, description="Free-text notes")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")

    @validator("mode", pre=True)
    @classmethod
    def _normalise_mode(cls, value: Any) -> Any:
        if isinstance(value, str):
            lowered = value.strip().lower()
            aliases = {
                "rtltcp": ReceiverMode.RTLTCP,
                "rtl_tcp": ReceiverMode.RTLTCP,
                "tcp": ReceiverMode.RTLTCP,
                "remote": ReceiverMode.RTLTCP,
                "local": ReceiverMode.LOCAL,
                "usb": ReceiverMode.LOCAL,
            }
            if lowered in aliases:
                return aliases[lowered]
        return value

    @validator("id", pre=True)
    @classmethod
    def _require_id(cls, value: Any) -> Any:
        if value is None or str(value).strip() == "":
            raise ValueError("receiver id is required")
        return str(value).strip()

    @validator("name", pre=True)
    @classmethod
    def _require_name(cls, value: Any) -> Any:
        if value is None or str(value).strip() == "":
            raise ValueError("receiver name is required")
        return str(value).strip()

    def distance_to(self, other: "SDRReceiver") -> float:
        """Great-circle distance to another receiver in metres."""
        return haversine_meters(self.lat, self.lon, other.lat, other.lon)

    def validate_transport(self) -> None:
        """Raise ``ValueError`` if the mode's required fields are missing.

        Kept separate from Pydantic validation so partially-configured receivers
        can be persisted and edited in the UI without being rejected outright.
        """
        if self.mode is ReceiverMode.RTLTCP:
            if not self.host:
                raise ValueError(f"receiver {self.name!r}: host is required for rtl_tcp mode")
            if self.port is None:
                raise ValueError(f"receiver {self.name!r}: port is required for rtl_tcp mode")

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly payload."""
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "lat": self.lat,
            "lon": self.lon,
            "deviceIndex": self.device_index,
            "host": self.host,
            "port": self.port,
            "gainDb": self.gain_db,
            "ppmError": self.ppm_error,
            "bandwidthHz": self.bandwidth_hz,
            "manualRssiDb": self.manual_rssi_db,
            "enabled": self.enabled,
            "altitudeM": self.altitude_m,
            "notes": self.notes,
            "createdAt": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SDRReceiver":
        """Build from a camelCase payload as served by the API."""
        return cls(
            id=payload.get("id"),
            name=payload.get("name"),
            mode=payload.get("mode", ReceiverMode.LOCAL),
            lat=payload.get("lat"),
            lon=payload.get("lon"),
            device_index=payload.get("deviceIndex", payload.get("device_index")),
            host=payload.get("host"),
            port=payload.get("port"),
            gain_db=payload.get("gainDb", payload.get("gain_db")),
            ppm_error=payload.get("ppmError", payload.get("ppm_error")),
            bandwidth_hz=payload.get("bandwidthHz", payload.get("bandwidth_hz")),
            manual_rssi_db=payload.get("manualRssiDb", payload.get("manual_rssi_db")),
            enabled=payload.get("enabled", True),
            altitude_m=payload.get("altitudeM", payload.get("altitude_m")),
            notes=payload.get("notes"),
            created_at=payload.get("createdAt", payload.get("created_at")),
        )


class ReceiverReading(BaseModel):
    """A measured level at one receiver for one frequency."""

    receiver_id: str = Field(..., description="Receiver that took the measurement")
    name: str = Field(..., description="Receiver label")
    lat: float = Field(..., description="Receiver latitude")
    lon: float = Field(..., description="Receiver longitude")
    rssi_db: float = Field(..., description="Calibrated level in dBFS/dBm")
    raw_rssi_db: Optional[float] = Field(None, description="Level before calibration")
    calibration_offset_db: float = Field(0.0, description="Offset applied")
    noise_floor_db: Optional[float] = Field(None, description="Noise floor at that receiver")
    prominence_db: Optional[float] = Field(None, description="Signal prominence over the floor")
    reliability: float = Field(1.0, ge=0.0, le=1.0, description="Measurement confidence 0-1")
    source: str = Field(default="unknown", description="How the scan was produced")
    scan_ms: Optional[float] = Field(None, description="Scan duration in milliseconds")
    warning: Optional[str] = Field(None, description="Non-fatal measurement warning")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.receiver_id,
            "name": self.name,
            "lat": self.lat,
            "lon": self.lon,
            "rssiDbm": round(self.rssi_db, 2),
            "rawRssiDbm": round(self.raw_rssi_db, 2) if self.raw_rssi_db is not None else None,
            "calOffsetDb": round(self.calibration_offset_db, 2),
            "noiseFloorDbm": round(self.noise_floor_db, 2) if self.noise_floor_db is not None else None,
            "prominenceDb": round(self.prominence_db, 2) if self.prominence_db is not None else None,
            "reliability": self.reliability,
            "source": self.source,
            "scanMs": self.scan_ms,
            "warning": self.warning,
        }


class TriangulationResult(BaseModel):
    """A geolocation outcome plus its provenance."""

    frequency_hz: float = Field(..., gt=0, description="Frequency that was located")
    estimate: Dict[str, Any] = Field(..., description="Position estimate")
    readings: List[Dict[str, Any]] = Field(default_factory=list)
    model: Dict[str, Any] = Field(default_factory=dict, description="Path-loss model used")
    warnings: List[str] = Field(default_factory=list)
    used_low_confidence_fallback: bool = Field(default=False)


def receiver_spread(receivers: List[SDRReceiver]) -> float:
    """Largest pairwise distance between receivers, in metres."""
    spread = 0.0
    for i in range(len(receivers)):
        for j in range(i + 1, len(receivers)):
            spread = max(spread, receivers[i].distance_to(receivers[j]))
    return spread