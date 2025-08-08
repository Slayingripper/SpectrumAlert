"""Custom exceptions for SpectrumAlert system."""

from typing import Any, Dict, Optional


class SpectrumAlertError(Exception):
    """Base exception for all SpectrumAlert errors."""
    
    def __init__(
        self, 
        message: str, 
        details: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}
        self.cause = cause


class SDRError(SpectrumAlertError):
    """Raised when SDR device operations fail."""
    pass


class ModelError(SpectrumAlertError):
    """Raised when ML model operations fail."""
    pass


class ConfigurationError(SpectrumAlertError):
    """Raised when configuration is invalid or missing."""
    pass


class DataValidationError(SpectrumAlertError):
    """Raised when data validation fails."""
    pass


class FeatureExtractionError(SpectrumAlertError):
    """Raised when feature extraction fails."""
    pass


class MonitoringError(SpectrumAlertError):
    """Raised when monitoring operations fail."""
    pass


class StorageError(SpectrumAlertError):
    """Raised when data storage operations fail."""
    pass


class NetworkError(SpectrumAlertError):
    """Raised when network operations fail."""
    pass
