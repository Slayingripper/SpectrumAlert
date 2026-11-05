"""SpectrumAlert v1.1 - Advanced RF Spectrum Monitoring and Anomaly Detection System.

This package provides a comprehensive solution for RF spectrum monitoring,
anomaly detection, and signal analysis using Software-Defined Radio (SDR) devices.

Key Features:
- Real-time spectrum monitoring with band sweeping
- Anomaly detection on a windowed periodogram with a robust noise floor
- Multi-receiver transmitter geolocation with calibration and path-loss fitting
- Multi-transport scanning: local rtl_power, remote rtl_tcp, or pinned levels
- Band plans for PMR446, ISM, amateur and air bands
- Docker deployment support
- MQTT integration for alerts
- Comprehensive logging and monitoring

Units note: an RTL-SDR reports dBFS, not dBm. Levels stay labelled dBFS until a
per-receiver calibration offset is applied.

Architecture:
- Clean Architecture principles with dependency inversion
- Domain-driven design with clear separation of concerns
- Modern Python practices with type hints and async support
- Comprehensive testing with unit, integration, and e2e tests
"""

__version__ = "1.1.0"
__author__ = "SpectrumAlert Team"
__email__ = "spectrum-alert@example.com"
__license__ = "MIT"

# Public API exports
from spectrum_alert.core.domain.models import (
    SpectrumData,
    AnomalyDetection,
    RFFingerprint,
    MonitoringSession,
)
from spectrum_alert.core.domain.receivers import ReceiverMode, SDRReceiver
from spectrum_alert.core.services.propagation import (
    distance_to_rssi,
    rssi_to_distance_meters,
)
from spectrum_alert.core.services.spectrum_analysis import ScanResult, analyze_spectrum
from spectrum_alert.core.exceptions import (
    SpectrumAlertError,
    SDRError,
    ModelError,
    ConfigurationError,
)

__all__ = [
    "SpectrumData",
    "AnomalyDetection",
    "RFFingerprint",
    "MonitoringSession",
    "SDRReceiver",
    "ReceiverMode",
    "ScanResult",
    "analyze_spectrum",
    "rssi_to_distance_meters",
    "distance_to_rssi",
    "SpectrumAlertError",
    "SDRError",
    "ModelError",
    "ConfigurationError",
]
