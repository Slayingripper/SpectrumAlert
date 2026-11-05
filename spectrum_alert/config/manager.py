"""Configuration management for SpectrumAlert."""

import os
import json
import configparser
from pathlib import Path
from typing import Any, Dict, Optional, Union

from spectrum_alert.core.exceptions import ConfigurationError


class ConfigurationManager:
    """Manages application configuration from multiple sources."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to configuration file. If None, looks for config.ini
        """
        self._config = {}
        self._config_path = Path(config_path) if config_path else Path("config/config.ini")
        self._load_defaults()
        self._load_from_file()
        self._load_from_environment()
    
    def _load_defaults(self) -> None:
        """Load default configuration values."""
        self._config = {
            # SDR Settings
            'sdr': {
                'device_index': 0,
                'frequency_hz': 100e6,
                'sample_rate_hz': 2048000,
                'gain_db': 20,
                'buffer_size': 1024,
            },
            
            # Monitoring Settings
            'monitoring': {
                'frequency_start_hz': 88e6,
                'frequency_end_hz': 108e6,
                'frequency_step_hz': 1e6,
                'scan_delay_seconds': 0.01,
                'capture_duration_seconds': 0.1,
                'detection_mode': 'full',
                'confidence_threshold': 0.5,
            },
            
            # Model Settings
            'models': {
                'model_directory': 'models',
                'anomaly_model_name': 'anomaly_detection_model_lite.pkl',
                'fingerprint_model_name': 'rf_fingerprinting_model_lite.pkl',
                'contamination_rate': 0.1,
                'auto_retrain': True,
                'retrain_interval_hours': 24,
                'min_training_samples': 1000,
            },
            
            # Data Settings
            'data': {
                'data_directory': 'data',
                'log_directory': 'logs',
                'max_file_size_mb': 100,
                'retention_days': 30,
                'compression_enabled': True,
                'backup_enabled': False,
            },
            
            # MQTT Settings
            'mqtt': {
                'enabled': True,
                'broker_host': 'localhost',
                'broker_port': 1883,
                'username': None,
                'password': None,
                'client_id': 'spectrum_alert',
                'topic_prefix': 'spectrum_alert',
                'qos': 1,
                'keepalive_seconds': 60,
                'tls_enabled': False,
            },
            
            # Logging Settings
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_enabled': True,
                'console_enabled': True,
                'max_file_size_mb': 10,
                'backup_count': 5,
            },
            
            # Core Settings
            'core': {
                'environment': 'development',
                'debug': False,
            }
        }
    
    def _load_from_file(self) -> None:
        """Load configuration from INI file."""
        if not self._config_path.exists():
            return
        
        try:
            parser = configparser.ConfigParser()
            parser.read(self._config_path)
            
            for section_name in parser.sections():
                if section_name.lower() not in self._config:
                    self._config[section_name.lower()] = {}
                
                for key, value in parser[section_name].items():
                    # Try to convert to appropriate type
                    converted_value = self._convert_value(value)
                    self._config[section_name.lower()][key] = converted_value
                    
        except Exception as e:
            raise ConfigurationError(f"Error loading config file: {e}")
    
    def _load_from_environment(self) -> None:
        """Load configuration from environment variables."""
        prefix = "SPECTRUM_ALERT_"
        
        for key, value in os.environ.items():
            if key.startswith(prefix):
                # Parse nested keys: SPECTRUM_ALERT_SDR__FREQUENCY_HZ
                config_key = key[len(prefix):].lower()
                parts = config_key.split('__')
                
                if len(parts) == 2:
                    section, setting = parts
                    if section not in self._config:
                        self._config[section] = {}
                    self._config[section][setting] = self._convert_value(value)
                elif len(parts) == 1:
                    # Direct setting
                    self._config['core'][parts[0]] = self._convert_value(value)
    
    def _convert_value(self, value: str) -> Any:
        """Convert string value to appropriate type."""
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        if value.lower() in ('null', 'none', ''):
            return None
        
        # Try numeric conversion
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a configuration setting.
        
        Args:
            key: Setting key in format 'section.setting' or 'setting'
            default: Default value if setting not found
            
        Returns:
            Setting value or default
        """
        parts = key.split('.')
        
        if len(parts) == 1:
            # Direct key
            return self._config.get('core', {}).get(parts[0], default)
        elif len(parts) == 2:
            # Section.key
            section, setting = parts
            return self._config.get(section, {}).get(setting, default)
        else:
            raise ConfigurationError(f"Invalid setting key format: {key}")
    
    def set_setting(self, key: str, value: Any) -> None:
        """Set a configuration setting.
        
        Args:
            key: Setting key in format 'section.setting' or 'setting'
            value: Value to set
        """
        parts = key.split('.')
        
        if len(parts) == 1:
            if 'core' not in self._config:
                self._config['core'] = {}
            self._config['core'][parts[0]] = value
        elif len(parts) == 2:
            section, setting = parts
            if section not in self._config:
                self._config[section] = {}
            self._config[section][setting] = value
        else:
            raise ConfigurationError(f"Invalid setting key format: {key}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get all settings for a section.
        
        Args:
            section: Section name
            
        Returns:
            Dictionary of section settings
        """
        return self._config.get(section, {}).copy()
    
    def save_configuration(self, config_path: Optional[Union[str, Path]] = None) -> None:
        """Save current configuration to file.
        
        Args:
            config_path: Path to save config. If None, uses default path.
        """
        save_path = Path(config_path) if config_path else self._config_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            parser = configparser.ConfigParser()
            
            for section_name, section_data in self._config.items():
                if section_name == 'core':
                    section_name = 'GENERAL'
                else:
                    section_name = section_name.upper()
                
                parser.add_section(section_name)
                for key, value in section_data.items():
                    parser.set(section_name, key, str(value))
            
            with open(save_path, 'w') as f:
                parser.write(f)
                
        except Exception as e:
            raise ConfigurationError(f"Error saving config file: {e}")
    
    def validate_configuration(self) -> bool:
        """Validate current configuration.
        
        Returns:
            True if configuration is valid
            
        Raises:
            ConfigurationError: If configuration is invalid
        """
        errors = []
        
        # Validate frequency ranges
        freq_start = self.get_setting('monitoring.frequency_start_hz')
        freq_end = self.get_setting('monitoring.frequency_end_hz')
        if freq_start >= freq_end:
            errors.append("End frequency must be greater than start frequency")
        
        # Validate paths exist
        model_dir = Path(self.get_setting('models.model_directory'))
        if not model_dir.exists():
            errors.append(f"Model directory does not exist: {model_dir}")
        
        # Validate MQTT settings if enabled
        if self.get_setting('mqtt.enabled'):
            broker_host = self.get_setting('mqtt.broker_host')
            if not broker_host:
                errors.append("MQTT broker host is required when MQTT is enabled")
        
        if errors:
            raise ConfigurationError("Configuration validation failed: " + "; ".join(errors))
        
        return True
    
    def get_model_path(self, model_name: str) -> Path:
        """Get full path to a model file."""
        model_dir = self.get_setting('models.model_directory')
        return Path(model_dir) / model_name
    
    def get_data_path(self, filename: str) -> Path:
        """Get full path to a data file."""
        data_dir = self.get_setting('data.data_directory')
        return Path(data_dir) / filename
    
    def get_log_path(self, filename: str) -> Path:
        """Get full path to a log file."""
        log_dir = self.get_setting('data.log_directory')
        return Path(log_dir) / filename
    
    def to_dict(self) -> Dict[str, Any]:
        """Get configuration as dictionary."""
        return self._config.copy()
    
    def from_dict(self, config_dict: Dict[str, Any]) -> None:
        """Load configuration from dictionary."""
        self._config.update(config_dict)
