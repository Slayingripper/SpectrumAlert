"""
Configuration manager for SpectrumAlert
"""

import configparser
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HamBand:
    """Represents a ham radio frequency band"""
    start_freq: float
    end_freq: float
    name: str = ""
    
    def __post_init__(self):
        if self.start_freq >= self.end_freq:
            raise ValueError(f"Start frequency ({self.start_freq}) must be less than end frequency ({self.end_freq})")
        if self.start_freq <= 0 or self.end_freq <= 0:
            raise ValueError("Frequencies must be positive")


@dataclass
class GeneralConfig:
    """General configuration settings"""
    freq_step: float
    sample_rate: float
    runs_per_freq: int
    sdr_type: str
    lite_mode: bool = False
    
    def __post_init__(self):
        if self.freq_step <= 0:
            raise ValueError("Frequency step must be positive")
        if self.sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
        if self.runs_per_freq <= 0:
            raise ValueError("Runs per frequency must be positive")
        if self.sdr_type not in ['rtlsdr', 'limesdr', 'hackrf', 'rsp1a', 'usrp']:
            raise ValueError(f"Unsupported SDR type: {self.sdr_type}")


@dataclass
class ReceiverConfig:
    """Receiver location configuration"""
    latitude: float
    longitude: float
    
    def __post_init__(self):
        if not (-90 <= self.latitude <= 90):
            raise ValueError("Latitude must be between -90 and 90 degrees")
        if not (-180 <= self.longitude <= 180):
            raise ValueError("Longitude must be between -180 and 180 degrees")


@dataclass
class MQTTConfig:
    """MQTT configuration settings"""
    broker: str
    port: int
    topic_anomalies: str
    topic_modulation: str
    topic_signal_strength: str
    topic_coordinates: str
    username: Optional[str] = None
    password: Optional[str] = None
    
    def __post_init__(self):
        if not (1 <= self.port <= 65535):
            raise ValueError("Port must be between 1 and 65535")
        if not self.broker:
            raise ValueError("MQTT broker address cannot be empty")


class ConfigManager:
    """Manages configuration for SpectrumAlert"""
    
    def __init__(self, config_file: str = 'config/config.ini'):
        self.config_file = config_file
        self.config = configparser.ConfigParser()
        self._ham_bands: List[HamBand] = []
        self._general: Optional[GeneralConfig] = None
        self._receiver: Optional[ReceiverConfig] = None
        self._mqtt: Optional[MQTTConfig] = None
        
    def load_config(self) -> None:
        """Load configuration from file"""
        if not os.path.exists(self.config_file):
            logger.error(f"Config file '{self.config_file}' not found")
            raise FileNotFoundError(f"Config file '{self.config_file}' not found")
        
        try:
            self.config.read(self.config_file)
            self._load_ham_bands()
            self._load_general_config()
            self._load_receiver_config()
            self._load_mqtt_config()
            logger.info(f"Configuration loaded successfully from {self.config_file}")
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def _load_ham_bands(self) -> None:
        """Load ham band configuration"""
        if 'HAM_BANDS' not in self.config:
            raise ValueError("'HAM_BANDS' section missing in config file")
        
        bands_str = self.config['HAM_BANDS'].get('bands')
        if not bands_str:
            raise ValueError("Missing 'bands' entry in 'HAM_BANDS' section")
        
        self._ham_bands = []
        for band_str in bands_str.split(','):
            band_str = band_str.strip()
            if '-' not in band_str:
                raise ValueError(f"Invalid frequency range format: {band_str}. Expected 'start-end'")
            
            try:
                start_str, end_str = band_str.split('-')
                start_freq = self._parse_frequency(start_str.strip())
                end_freq = self._parse_frequency(end_str.strip())
                band = HamBand(start_freq, end_freq, f"Band_{len(self._ham_bands) + 1}")
                self._ham_bands.append(band)
            except ValueError as e:
                raise ValueError(f"Invalid frequency range {band_str}: {e}")
    
    def _parse_frequency(self, freq_str: str) -> float:
        """Parse frequency string with units (e.g., '144e6', '144MHz')"""
        freq_str = freq_str.upper().replace('HZ', '').replace('MHZ', 'e6').replace('GHZ', 'e9')
        try:
            return float(eval(freq_str))  # Safe for numerical expressions
        except:
            raise ValueError(f"Cannot parse frequency: {freq_str}")
    
    def _load_general_config(self) -> None:
        """Load general configuration"""
        if 'GENERAL' not in self.config:
            raise ValueError("'GENERAL' section missing in config file")
        
        section = self.config['GENERAL']
        self._general = GeneralConfig(
            freq_step=float(section.get('freq_step', 500e3)),
            sample_rate=float(section.get('sample_rate', 2.048e6)),
            runs_per_freq=int(section.get('runs_per_freq', 5)),
            sdr_type=section.get('sdr_type', 'rtlsdr'),
            lite_mode=section.getboolean('lite_mode', False)
        )
    
    def _load_receiver_config(self) -> None:
        """Load receiver configuration"""
        if 'RECEIVER' not in self.config:
            raise ValueError("'RECEIVER' section missing in config file")
        
        section = self.config['RECEIVER']
        self._receiver = ReceiverConfig(
            latitude=float(section.get('latitude', 0.0)),
            longitude=float(section.get('longitude', 0.0))
        )
    
    def _load_mqtt_config(self) -> None:
        """Load MQTT configuration"""
        if 'MQTT' not in self.config:
            raise ValueError("'MQTT' section missing in config file")
        
        section = self.config['MQTT']
        self._mqtt = MQTTConfig(
            broker=section.get('broker', 'localhost'),
            port=int(section.get('port', 1883)),
            topic_anomalies=section.get('topic_anomalies', 'hamradio/anomalies'),
            topic_modulation=section.get('topic_modulation', 'hamradio/modulation'),
            topic_signal_strength=section.get('topic_signal_strength', 'hamradio/signal_strength'),
            topic_coordinates=section.get('topic_coordinates', 'hamradio/coordinates'),
            username=section.get('username'),
            password=section.get('password')
        )
    
    @property
    def ham_bands(self) -> List[HamBand]:
        """Get ham bands configuration"""
        if not self._ham_bands:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._ham_bands
    
    @property
    def general(self) -> GeneralConfig:
        """Get general configuration"""
        if not self._general:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._general
    
    @property
    def receiver(self) -> ReceiverConfig:
        """Get receiver configuration"""
        if not self._receiver:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._receiver
    
    @property
    def mqtt(self) -> MQTTConfig:
        """Get MQTT configuration"""
        if not self._mqtt:
            raise RuntimeError("Configuration not loaded. Call load_config() first.")
        return self._mqtt
    
    def validate_config(self) -> bool:
        """Validate the complete configuration"""
        try:
            # Check if all required sections are present
            _ = self.ham_bands
            _ = self.general
            _ = self.receiver
            _ = self.mqtt
            
            logger.info("Configuration validation successful")
            return True
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def create_default_config(self, filename: Optional[str] = None) -> None:
        """Create a default configuration file"""
        if filename is None:
            filename = self.config_file
        
        default_config = configparser.ConfigParser()
        
        # Ham bands section
        default_config['HAM_BANDS'] = {
            'bands': '144e6-146e6,420e6-440e6'
        }
        
        # General section
        default_config['GENERAL'] = {
            'freq_step': '500000',
            'sample_rate': '2.048e6',
            'runs_per_freq': '5',
            'sdr_type': 'rtlsdr',
            'lite_mode': 'false'
        }
        
        # Receiver section
        default_config['RECEIVER'] = {
            'latitude': '0.0',
            'longitude': '0.0'
        }
        
        # MQTT section
        default_config['MQTT'] = {
            'broker': 'localhost',
            'port': '1883',
            'topic_anomalies': 'hamradio/anomalies',
            'topic_modulation': 'hamradio/modulation',
            'topic_signal_strength': 'hamradio/signal_strength',
            'topic_coordinates': 'hamradio/coordinates'
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w') as configfile:
            default_config.write(configfile)
        
        logger.info(f"Default configuration created at {filename}")
