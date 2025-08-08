"""SpectrumAlert v1.1 - Advanced RF Spectrum Monitoring and Anomaly Detection System.

This package provides a comprehensive solution for RF spectrum monitoring,
anomaly detection, and signal analysis using Software-Defined Radio (SDR) devices.

Key Features:
- Real-time spectrum monitoring
- Machine learning-based anomaly detection
- RF fingerprinting and classification
- Docker deployment support
- MQTT integration for alerts
- Comprehensive logging and monitoring

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
    "SpectrumAlertError",
    "SDRError",
    "ModelError",
    "ConfigurationError",
]
