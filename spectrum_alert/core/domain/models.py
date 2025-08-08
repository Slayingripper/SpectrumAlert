"""Domain models for SpectrumAlert system."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel, Field, validator


class DetectionMode(str, Enum):
    """Detection modes for spectrum analysis."""
    LITE = "lite"
    FULL = "full"


class AnomalyType(str, Enum):
    """Types of anomalies that can be detected."""
    FREQUENCY = "frequency"
    AMPLITUDE = "amplitude"
    BANDWIDTH = "bandwidth"
    MODULATION = "modulation"
    PATTERN = "pattern"
    UNKNOWN = "unknown"


class MonitoringStatus(str, Enum):
    """Status of monitoring sessions."""
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"


class SpectrumData(BaseModel):
    """Represents RF spectrum data from SDR device."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frequency_hz: float = Field(..., gt=0, description="Center frequency in Hz")
    sample_rate_hz: float = Field(..., gt=0, description="Sample rate in Hz")
    gain_db: float = Field(..., description="Gain in dB")
    samples: List[complex] = Field(..., description="IQ samples")
    power_spectrum: Optional[List[float]] = Field(None, description="Power spectrum")
    duration_seconds: float = Field(..., gt=0, description="Capture duration")
    
    class Config:
        arbitrary_types_allowed = True
    
    @validator('samples')
    def validate_samples(cls, v):
        if not v:
            raise ValueError("Samples cannot be empty")
        return v
    
    @property
    def sample_count(self) -> int:
        """Number of samples in the data."""
        return len(self.samples)
    
    @property
    def frequency_range_hz(self) -> tuple[float, float]:
        """Frequency range covered by this data."""
        bandwidth = self.sample_rate_hz
        return (
            self.frequency_hz - bandwidth / 2,
            self.frequency_hz + bandwidth / 2
        )


class FeatureVector(BaseModel):
    """Feature vector extracted from spectrum data."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spectrum_data_id: str = Field(..., description="ID of source spectrum data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    mode: DetectionMode = Field(..., description="Detection mode used")
    features: Dict[str, float] = Field(..., description="Feature values")
    feature_names: List[str] = Field(..., description="Names of features")
    
    @validator('features')
    def validate_features(cls, v):
        if not v:
            raise ValueError("Features cannot be empty")
        return v
    
    @property
    def feature_vector(self) -> List[float]:
        """Get feature values as ordered list."""
        return [self.features[name] for name in self.feature_names]


class AnomalyDetection(BaseModel):
    """Represents an anomaly detection result."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    spectrum_data_id: str = Field(..., description="ID of analyzed spectrum data")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frequency_hz: float = Field(..., gt=0, description="Frequency where anomaly detected")
    anomaly_type: AnomalyType = Field(..., description="Type of anomaly")
    confidence_score: float = Field(..., ge=0, le=1, description="Confidence score 0-1")
    severity: str = Field(..., description="Anomaly severity level")
    description: str = Field(..., description="Human-readable description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    detection_mode: DetectionMode = Field(..., description="Mode used for detection")
    
    @validator('severity')
    def validate_severity(cls, v):
        valid_severities = ['low', 'medium', 'high', 'critical']
        if v.lower() not in valid_severities:
            raise ValueError(f"Severity must be one of: {valid_severities}")
        return v.lower()


class RFFingerprint(BaseModel):
    """RF fingerprint for device identification."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    device_id: Optional[str] = Field(None, description="Known device identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    frequency_hz: float = Field(..., gt=0, description="Operating frequency")
    signature: Dict[str, float] = Field(..., description="Unique RF signature")
    confidence_score: float = Field(..., ge=0, le=1, description="Identification confidence")
    classification: Optional[str] = Field(None, description="Device classification")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('signature')
    def validate_signature(cls, v):
        if not v:
            raise ValueError("Signature cannot be empty")
        return v


class MonitoringSession(BaseModel):
    """Represents a monitoring session."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Session name")
    start_time: datetime = Field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = Field(None)
    status: MonitoringStatus = Field(default=MonitoringStatus.STARTING)
    frequency_range_hz: tuple[float, float] = Field(..., description="Monitored frequency range")
    detection_mode: DetectionMode = Field(..., description="Detection mode")
    configuration: Dict[str, Any] = Field(default_factory=dict, description="Session config")
    statistics: Dict[str, Any] = Field(default_factory=dict, description="Session stats")
    
    class Config:
        use_enum_values = True
    
    @validator('frequency_range_hz')
    def validate_frequency_range(cls, v):
        start_freq, end_freq = v
        if start_freq >= end_freq:
            raise ValueError("Start frequency must be less than end frequency")
        if start_freq <= 0:
            raise ValueError("Frequencies must be positive")
        return v
    
    @property
    def duration_seconds(self) -> Optional[float]:
        """Session duration in seconds."""
        if self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return None
    
    @property
    def is_active(self) -> bool:
        """Check if session is currently active."""
        return self.status in [MonitoringStatus.STARTING, MonitoringStatus.RUNNING]


class ModelMetadata(BaseModel):
    """Metadata for ML models."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Model name")
    version: str = Field(..., description="Model version")
    model_type: str = Field(..., description="Type of model")
    detection_mode: DetectionMode = Field(..., description="Compatible detection mode")
    training_date: datetime = Field(..., description="When model was trained")
    performance_metrics: Dict[str, float] = Field(default_factory=dict)
    feature_names: List[str] = Field(..., description="Expected feature names")
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def feature_count(self) -> int:
        """Number of features expected by this model."""
        return len(self.feature_names)
