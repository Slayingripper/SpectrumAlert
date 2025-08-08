"""Configuration settings models using Pydantic."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, validator


class SDRSettings(BaseModel):
    """SDR device configuration."""
    
    device_index: int = Field(0, description="SDR device index")
    frequency_hz: float = Field(100e6, gt=0, description="Default center frequency")
    sample_rate_hz: float = Field(2048000, gt=0, description="Sample rate")
    gain_db: float = Field(20, description="RF gain in dB")
    bandwidth_hz: Optional[float] = Field(None, description="Bandwidth filter")
    buffer_size: int = Field(1024, gt=0, description="Buffer size for samples")
    
    @validator('frequency_hz')
    def validate_frequency(cls, v):
        if not (1e6 <= v <= 6000e6):  # 1 MHz to 6 GHz
            raise ValueError("Frequency must be between 1 MHz and 6 GHz")
        return v


class MonitoringSettings(BaseModel):
    """Monitoring configuration."""
    
    frequency_start_hz: float = Field(88e6, gt=0, description="Start frequency")
    frequency_end_hz: float = Field(108e6, gt=0, description="End frequency")
    frequency_step_hz: float = Field(1e6, gt=0, description="Frequency step size")
    scan_delay_seconds: float = Field(0.01, ge=0, description="Delay between scans")
    capture_duration_seconds: float = Field(0.1, gt=0, description="Capture duration")
    detection_mode: str = Field("full", description="Detection mode (lite/full)")
    confidence_threshold: float = Field(0.5, ge=0, le=1, description="Confidence threshold")
    
    @validator('frequency_end_hz')
    def validate_frequency_range(cls, v, values):
        if 'frequency_start_hz' in values and v <= values['frequency_start_hz']:
            raise ValueError("End frequency must be greater than start frequency")
        return v


class ModelSettings(BaseModel):
    """Machine learning model configuration."""
    
    model_directory: Path = Field(Path("models"), description="Model storage directory")
    anomaly_model_name: str = Field("anomaly_detection_model_lite.pkl", description="Anomaly model file")
    fingerprint_model_name: str = Field("rf_fingerprinting_model_lite.pkl", description="Fingerprint model file")
    contamination_rate: float = Field(0.1, gt=0, lt=1, description="Anomaly contamination rate")
    auto_retrain: bool = Field(True, description="Enable automatic retraining")
    retrain_interval_hours: int = Field(24, gt=0, description="Retraining interval")
    min_training_samples: int = Field(1000, gt=0, description="Minimum samples for training")


class DataSettings(BaseModel):
    """Data storage and management configuration."""
    
    data_directory: Path = Field(Path("data"), description="Data storage directory")
    log_directory: Path = Field(Path("logs"), description="Log file directory")
    max_file_size_mb: int = Field(100, gt=0, description="Maximum file size in MB")
    retention_days: int = Field(30, gt=0, description="Data retention period")
    compression_enabled: bool = Field(True, description="Enable data compression")
    backup_enabled: bool = Field(False, description="Enable data backup")


class MQTTSettings(BaseModel):
    """MQTT broker configuration."""
    
    enabled: bool = Field(True, description="Enable MQTT publishing")
    broker_host: str = Field("localhost", description="MQTT broker hostname")
    broker_port: int = Field(1883, gt=0, le=65535, description="MQTT broker port")
    username: Optional[str] = Field(None, description="MQTT username")
    password: Optional[str] = Field(None, description="MQTT password")
    client_id: str = Field("spectrum_alert", description="MQTT client ID")
    topic_prefix: str = Field("spectrum_alert", description="Topic prefix")
    qos: int = Field(1, ge=0, le=2, description="Quality of service level")
    keepalive_seconds: int = Field(60, gt=0, description="Keepalive interval")
    tls_enabled: bool = Field(False, description="Enable TLS encryption")


class LoggingSettings(BaseModel):
    """Logging configuration."""
    
    level: str = Field("INFO", description="Log level")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    file_enabled: bool = Field(True, description="Enable file logging")
    console_enabled: bool = Field(True, description="Enable console logging")
    max_file_size_mb: int = Field(10, gt=0, description="Maximum log file size")
    backup_count: int = Field(5, gt=0, description="Number of backup log files")
    
    @validator('level')
    def validate_log_level(cls, v):
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()


class SpectrumAlertSettings(BaseModel):
    """Main configuration settings for SpectrumAlert."""
    
    # Core settings
    environment: str = Field("development", description="Environment (dev/prod/test)")
    debug: bool = Field(False, description="Enable debug mode")
    
    # Component settings - use Optional and provide defaults in methods
    sdr: Optional[SDRSettings] = None
    monitoring: Optional[MonitoringSettings] = None
    models: Optional[ModelSettings] = None
    data: Optional[DataSettings] = None
    mqtt: Optional[MQTTSettings] = None
    logging: Optional[LoggingSettings] = None
    
    # Additional settings
    custom_settings: Dict[str, Any] = Field(default_factory=dict, description="Custom settings")
    
    class Config:
        env_prefix = "SPECTRUM_ALERT_"
        env_nested_delimiter = "__"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize with defaults if not provided
        if self.sdr is None:
            self.sdr = SDRSettings()
        if self.monitoring is None:
            self.monitoring = MonitoringSettings()
        if self.models is None:
            self.models = ModelSettings()
        if self.data is None:
            self.data = DataSettings()
        if self.mqtt is None:
            self.mqtt = MQTTSettings()
        if self.logging is None:
            self.logging = LoggingSettings()
    
    @validator('environment')
    def validate_environment(cls, v):
        valid_envs = ['development', 'production', 'testing']
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {valid_envs}")
        return v.lower()
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to a model file."""
        return self.models.model_directory / model_name
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to a data file."""
        return self.data.data_directory / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get full path to a log file."""
        return self.data.log_directory / filename
