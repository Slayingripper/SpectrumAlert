"""
Test configuration and utilities for SpectrumAlert web interface tests
"""

import os
import sys
import tempfile
import pytest
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

# Add the project root to the Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test configuration
TEST_CONFIG = {
    "frequency_range": "88-108",
    "sample_rate": 2048000,
    "gain": 20,
    "threshold": 0.8,
    "scan_interval": 5,
    "strict_mode": False
}

TEST_MULTIBAND_CONFIG = {
    "bands": [
        {"start": 88, "end": 108, "name": "FM"},
        {"start": 400, "end": 500, "name": "UHF"}
    ],
    "scan_interval": 5,
    "threshold": 0.8
}

TEST_MQTT_CONFIG = {
    "broker": "localhost",
    "port": 1883,
    "topic": "spectrum/alerts",
    "username": "test",
    "password": "test"
}


class MockRTLSDRInterface:
    """Mock RTL-SDR interface for testing"""
    
    def __init__(self, device_index=0, device_args=None):
        self.device_index = device_index
        self.device_args = device_args or {}
        self._is_open = False
        self._center_freq = 100e6
        self._sample_rate = 2048000
        self._gain = 20
        
    def open(self):
        self._is_open = True
        
    def close(self):
        self._is_open = False
        
    def set_center_freq(self, frequency):
        self._center_freq = frequency
        
    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate
        
    def set_gain(self, gain):
        self._gain = gain
        
    def read_samples(self, num_samples):
        import numpy as np
        # Return mock complex samples
        return np.random.random(num_samples) + 1j * np.random.random(num_samples)


class MockDataStorage:
    """Mock data storage for testing"""
    
    def __init__(self):
        self.spectrum_data = []
        self.anomalies = []
        self.features = []
        
    def save_spectrum_data(self, spectrum_data):
        self.spectrum_data.append(spectrum_data)
        return spectrum_data.id
        
    def save_anomaly(self, anomaly):
        self.anomalies.append(anomaly)
        return anomaly.id
        
    def save_features(self, features):
        self.features.append(features)
        return features.spectrum_data_id
        
    def get_recent_anomalies(self, limit=10):
        return self.anomalies[-limit:]
        
    def get_anomaly_stats(self):
        return {
            "total_anomalies": len(self.anomalies),
            "last_24h": len(self.anomalies),
            "severity_breakdown": {"high": 5, "medium": 3, "low": 2}
        }


class MockSystemMonitor:
    """Mock system monitor for testing"""
    
    def get_system_status(self):
        return {
            "cpu_usage": 25.5,
            "memory_usage": 60.2,
            "disk_usage": 45.8,
            "temperature": 42.1,
            "uptime": "2 days, 3 hours",
            "monitoring_active": False
        }
        
    def get_advanced_status(self):
        return {
            "cpu_usage": 25.5,
            "memory_usage": 60.2,
            "disk_usage": 45.8,
            "temperature": 42.1,
            "uptime": "2 days, 3 hours",
            "monitoring_active": False,
            "network_usage": {"tx": 1024, "rx": 2048},
            "processes": {"total": 156, "active": 23},
            "load_average": [1.2, 1.1, 1.0]
        }


class MockAnomalyDetectionUseCase:
    """Mock anomaly detection use case"""
    
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor
        self.threshold = 0.8
        self.strict_mode = False
        
    def set_threshold(self, threshold):
        self.threshold = threshold
        
    def set_strict_threshold_mode(self, enabled, min_snr=15.0):
        self.strict_mode = enabled
        
    def detect_anomalies(self, spectrum_data):
        from spectrum_alert.core.domain.models import AnomalyDetection, AnomalyType, DetectionMode
        from datetime import datetime
        
        # Mock detection result
        anomaly = AnomalyDetection(
            spectrum_data_id=spectrum_data.id,
            frequency_hz=spectrum_data.frequency_hz,
            anomaly_type=AnomalyType.UNKNOWN,
            confidence_score=0.9,
            severity="high",
            description="Test anomaly detected",
            detection_mode=DetectionMode.LITE,
            metadata={"power_dbm": -50.0, "snr_db": 20.0}
        )
        return [anomaly]


class MockFeatureExtractor:
    """Mock feature extractor"""
    
    def __init__(self, lite_mode=True):
        self.lite_mode = lite_mode
        
    def extract_features(self, samples, frequency):
        # Mock feature extraction
        class MockFeatureSet:
            def __init__(self):
                self.features = [1.0, 2.0, 3.0, 4.0, 5.0]
                self.feature_names = ["feature1", "feature2", "feature3", "feature4", "feature5"]
                
        return MockFeatureSet()


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory for test data"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def mock_sdr():
    """Mock RTL-SDR interface"""
    return MockRTLSDRInterface()


@pytest.fixture
def mock_storage():
    """Mock data storage"""
    return MockDataStorage()


@pytest.fixture
def mock_system_monitor():
    """Mock system monitor"""
    return MockSystemMonitor()


@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor"""
    return MockFeatureExtractor()


@pytest.fixture
def mock_anomaly_detector(mock_feature_extractor):
    """Mock anomaly detector"""
    return MockAnomalyDetectionUseCase(mock_feature_extractor)
