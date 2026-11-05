"""Ports (interfaces) for the hexagonal boundary of SpectrumAlert.

The domain and application layers define *what* they need; infrastructure
supplies *how*. Concretely, the application layer depends on these protocols
rather than on ``RTLSDRInterface`` or ``DataStorage`` directly, which is what
makes the use cases testable without hardware and without touching disk.

Use ``typing.Protocol`` rather than ABC so implementations need not inherit:
a backend is free to be a dataclass, a closure or a third-party object as long as
it satisfies the shape.
"""

from __future__ import annotations

from typing import (
    Any,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from spectrum_alert.core.domain.receivers import SDRReceiver
from spectrum_alert.core.services.spectrum_analysis import ScanResult

__all__ = [
    "SpectrumScanPort",
    "ReceiverStorePort",
    "CalibrationStorePort",
]


@runtime_checkable
class SpectrumScanPort(Protocol):
    """Produces a frequency-domain scan for one receiver."""

    def scan(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        span_hz: float,
        bin_hz: Optional[float] = None,
        integration_seconds: float = 1.0,
        bins: int = 256,
    ) -> ScanResult:
        """Return a scan centred on ``center_hz`` covering ``span_hz``.

        Implementations must not fabricate data: a receiver that cannot be
        reached should raise, or return a scan whose ``source`` and ``warning``
        make the degradation explicit.
        """
        ...

    def check(self, receiver: SDRReceiver, center_hz: float) -> Dict[str, Any]:
        """Probe reachability. Returns a mapping with at least ``ok`` and ``message``."""
        ...


@runtime_checkable
class ReceiverStorePort(Protocol):
    """Persistence for the receiver roster."""

    def list(self) -> List[SDRReceiver]:
        """All configured receivers, disabled ones included."""
        ...

    def save(self, receiver: SDRReceiver) -> SDRReceiver:
        """Insert or update one receiver; returns the stored form."""
        ...

    def delete(self, receiver_id: str) -> bool:
        """Remove a receiver. Returns whether anything was removed."""
        ...

    def clear(self) -> None:
        """Remove every receiver."""
        ...

    def enabled(self) -> List[SDRReceiver]:
        """Only the receivers currently enabled for scanning."""
        ...


@runtime_checkable
class CalibrationStorePort(Protocol):
    """Persistence and lookup of per-receiver calibration offsets."""

    def offset_for(self, receiver_id: str, frequency_hz: float) -> float:
        """Offset in dB for a receiver/frequency pair."""
        ...

    def apply(self, receiver_id: str, frequency_hz: float, value_db: float) -> None:
        """Store an offset."""
        ...

    def clear_receiver(self, receiver_id: str) -> None:
        """Drop every offset for one receiver."""
        ...

    def save(self) -> None:
        """Flush to durable storage."""
        ...