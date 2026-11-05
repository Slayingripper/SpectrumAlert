"""
Tests for backend components and use cases used by the web interface
"""

import pytest
import tempfile
import asyncio
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta
from pathlib import Path

from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.infrastructure.monitoring import SystemMonitor
from spectrum_alert.core.domain.models import SpectrumData, AnomalyDetection, AnomalyType, DetectionMode
from tests.web.conftest import MockRTLSDRInterface, MockFeatureExtractor


class TestDataStorage:
    """Test suite for DataStorage backend component"""

    @pytest.fixture
    def temp_data_dir(self):
        """Create temporary directory for test data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def storage(self, temp_data_dir):
        """Create DataStorage instance with temporary directory"""
        return DataStorage(data_dir=temp_data_dir)

    def test_storage_initialization(self, storage, temp_data_dir):
        """Test storage initializes correctly"""
        assert storage.data_dir == temp_data_dir
        
        # Check that required directories are created
        required_dirs = ["spectrum", "anomalies", "features", "logs"]
        for dir_name in required_dirs:
            assert (Path(temp_data_dir) / dir_name).exists()

    def test_save_spectrum_data(self, storage):
        """Test saving spectrum data"""
        spectrum_data = SpectrumData(
            frequency_hz=100e6,
            sample_rate_hz=2048000,
            gain_db=20,
            samples=[1+1j, 2+2j, 3+3j],
            duration_seconds=1.0,
            power_spectrum=[1.0, 2.0, 3.0]
        )
        
        # save_* return the file path, which is what the callers use to locate the
        # artifact. (The test previously asserted the record id, which no
        # implementation returns.)
        result_path = storage.save_spectrum_data(spectrum_data)
        assert result_path.endswith(".json")
        assert Path(result_path).exists()
        # The record id must still appear in the stored document.
        assert spectrum_data.id[:8] in Path(result_path).name

        spectrum_files = list(Path(storage.data_dir, "spectrum").glob("*.json"))
        assert len(spectrum_files) > 0

    def test_load_spectrum_data_roundtrip(self, storage):
        """Saved captures must load back, including the true duration."""
        spectrum_data = SpectrumData(
            frequency_hz=100e6,
            sample_rate_hz=2048000,
            gain_db=20,
            samples=[1 + 1j, 2 + 2j, 3 + 3j],
            duration_seconds=0.5,
        )
        path = storage.save_spectrum_data(spectrum_data)
        loaded = storage.load_spectrum_data(path)

        assert loaded.id == spectrum_data.id
        assert loaded.sample_count == 3
        # duration_seconds must come from the record, not be recomputed from the
        # truncated sample count (which reported 1.5 us instead of 0.5 s).
        assert loaded.duration_seconds == pytest.approx(0.5)

    def test_load_spectrum_data_accepts_legacy_schema(self, storage, tmp_path):
        """The 706 archived files use samples_real/samples_imag."""
        import json

        legacy = {
            "id": "legacy-1",
            "timestamp": datetime.utcnow().isoformat(),
            "frequency_hz": 145e6,
            "sample_rate_hz": 1.024e6,
            "gain_db": 20.0,
            "samples_real": [0.1, 0.2],
            "samples_imag": [0.3, 0.4],
        }
        path = tmp_path / "spectrum_20250809_101621_legacy.json"
        path.write_text(json.dumps(legacy))

        loaded = storage.load_spectrum_data(str(path))
        assert loaded.id == "legacy-1"
        assert loaded.sample_count == 2
        assert loaded.samples[0] == complex(0.1, 0.3)

    def test_save_anomaly(self, storage):
        """Test saving anomaly detection result"""
        anomaly = AnomalyDetection(
            spectrum_data_id="test-spectrum-id",
            frequency_hz=100e6,
            anomaly_type=AnomalyType.UNKNOWN,
            confidence_score=0.9,
            severity="high",
            description="Test anomaly",
            detection_mode=DetectionMode.LITE
        )
        
        result_path = storage.save_anomaly(anomaly)
        assert result_path.endswith(".csv")
        assert Path(result_path).exists()

        anomaly_files = list(Path(storage.data_dir, "anomalies").glob("*.csv"))
        assert len(anomaly_files) > 0

    def test_get_recent_anomalies(self, storage):
        """Test retrieving recent anomalies"""
        # Create test anomalies
        for i in range(5):
            anomaly = AnomalyDetection(
                spectrum_data_id=f"test-spectrum-id-{i}",
                frequency_hz=100e6 + i * 1e6,
                anomaly_type=AnomalyType.FREQUENCY,
                confidence_score=0.8 + i * 0.02,
                severity="medium",
                description=f"Test anomaly {i}",
                detection_mode=DetectionMode.LITE
            )
            storage.save_anomaly(anomaly)
        
        recent_anomalies = storage.get_recent_anomalies(limit=3)
        assert len(recent_anomalies) <= 3

    def test_get_anomaly_stats(self, storage):
        """Test getting anomaly statistics"""
        stats = storage.get_anomaly_stats()
        assert isinstance(stats, dict)
        assert stats["total"] == 0
        assert stats["days"] == 7
        assert "by_severity" in stats
        assert "by_type" in stats
        assert "by_mode" in stats

    def test_get_anomaly_stats_counts_saved_anomalies(self, storage):
        """Stats must reflect what was actually written."""
        for severity in ("low", "high", "high"):
            storage.save_anomaly(
                AnomalyDetection(
                    spectrum_data_id="test-spectrum-id",
                    frequency_hz=100e6,
                    anomaly_type=AnomalyType.AMPLITUDE,
                    confidence_score=0.9,
                    severity=severity,
                    description="Test anomaly",
                    detection_mode=DetectionMode.LITE,
                )
            )

        stats = storage.get_anomaly_stats(days=7)
        assert stats["total"] == 3
        assert stats["by_severity"]["high"] == 2
        assert stats["by_severity"]["low"] == 1

    def test_get_anomaly_count(self, storage):
        """get_anomaly_count is called by `monitor status` in the CLI."""
        storage.save_anomaly(
            AnomalyDetection(
                spectrum_data_id="test-spectrum-id",
                frequency_hz=100e6,
                anomaly_type=AnomalyType.AMPLITUDE,
                confidence_score=0.9,
                severity="high",
                description="Test anomaly",
                detection_mode=DetectionMode.LITE,
            )
        )
        assert storage.get_anomaly_count(days=1) == 1
        # A window that ends in the past cannot contain anything written now.
        assert storage.get_anomaly_count(days=0) == 0


class TestSystemMonitor:
    """Test suite for SystemMonitor backend component"""

    @pytest.fixture
    def monitor(self):
        """Create SystemMonitor instance"""
        return SystemMonitor()

    def test_get_system_status(self, monitor):
        """Test getting system status"""
        status = monitor.get_system_status()
        
        assert isinstance(status, dict)
        required_fields = ["cpu_usage", "memory_usage", "disk_usage", "uptime"]
        for field in required_fields:
            assert field in status

    def test_get_advanced_status(self, monitor):
        """Test getting advanced system status"""
        status = monitor.get_advanced_status()
        
        assert isinstance(status, dict)
        required_fields = ["cpu_usage", "memory_usage", "disk_usage", "temperature", "uptime"]
        for field in required_fields:
            assert field in status

    def test_status_values_are_numeric(self, monitor):
        """Test that status values are numeric where expected"""
        status = monitor.get_system_status()
        
        numeric_fields = ["cpu_usage", "memory_usage", "disk_usage"]
        for field in numeric_fields:
            if field in status:
                assert isinstance(status[field], (int, float))
                assert 0 <= status[field] <= 100  # Assuming percentages


class TestRTLSDRInterface:
    """Test suite for RTL-SDR interface"""

    @pytest.fixture
    def sdr(self):
        """Create mock RTL-SDR interface"""
        return MockRTLSDRInterface()

    def test_sdr_initialization(self, sdr):
        """Test SDR initializes correctly"""
        assert sdr.device_index == 0
        assert sdr._is_open is False

    def test_sdr_open_close(self, sdr):
        """Test SDR open and close operations"""
        assert sdr._is_open is False
        
        sdr.open()
        assert sdr._is_open is True
        
        sdr.close()
        assert sdr._is_open is False

    def test_sdr_configuration(self, sdr):
        """Test SDR configuration methods"""
        sdr.set_center_freq(100e6)
        assert sdr._center_freq == 100e6
        
        sdr.set_sample_rate(2048000)
        assert sdr._sample_rate == 2048000
        
        sdr.set_gain(20)
        assert sdr._gain == 20

    def test_sdr_read_samples(self, sdr):
        """Test reading samples from SDR"""
        samples = sdr.read_samples(1024)
        assert len(samples) == 1024
        assert samples.dtype.kind == 'c'  # Complex data type


class TestFeatureExtractor:
    """Test suite for FeatureExtractor"""

    @pytest.fixture
    def extractor(self):
        """Create mock feature extractor"""
        return MockFeatureExtractor(lite_mode=True)

    def test_feature_extraction(self, extractor):
        """Test feature extraction"""
        import numpy as np
        samples = np.random.random(1024) + 1j * np.random.random(1024)
        
        features = extractor.extract_features(samples, 100e6)
        
        assert hasattr(features, 'features')
        assert hasattr(features, 'feature_names')
        assert len(features.features) == len(features.feature_names)

    def test_lite_mode_setting(self, extractor):
        """Test lite mode setting"""
        assert extractor.lite_mode is True
        
        full_extractor = MockFeatureExtractor(lite_mode=False)
        assert full_extractor.lite_mode is False


class TestWebAppBackgroundTasks:
    """Test suite for web app background tasks and async operations"""

    def test_background_update_task_creation(self):
        """Test that background update tasks can be created"""
        async def mock_background_task():
            await asyncio.sleep(0.1)
            return "completed"
        
        # Test that async function can be called
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(mock_background_task())
            assert result == "completed"
        finally:
            loop.close()

    def test_websocket_message_handling(self):
        """Test WebSocket message handling logic"""
        async def handle_websocket_message(message):
            """Mock WebSocket message handler"""
            if message.get("type") == "anomaly_detected":
                return {"status": "processed", "type": "anomaly"}
            elif message.get("type") == "spectrum_update":
                return {"status": "processed", "type": "spectrum"}
            else:
                return {"status": "unknown", "type": "unknown"}
        
        # Test different message types
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            # Test anomaly message
            anomaly_msg = {"type": "anomaly_detected", "data": {"frequency": 100e6}}
            result = loop.run_until_complete(handle_websocket_message(anomaly_msg))
            assert result["type"] == "anomaly"
            
            # Test spectrum message
            spectrum_msg = {"type": "spectrum_update", "data": {"power": [1, 2, 3]}}
            result = loop.run_until_complete(handle_websocket_message(spectrum_msg))
            assert result["type"] == "spectrum"
        finally:
            loop.close()

    def test_monitoring_loop_logic(self):
        """Test monitoring loop logic"""
        class MockMonitoringLoop:
            def __init__(self):
                self.active = False
                self.iterations = 0
                
            async def run_monitoring_iteration(self):
                """Single monitoring iteration"""
                self.iterations += 1
                # Simulate some work
                await asyncio.sleep(0.01)
                return {"status": "ok", "iteration": self.iterations}
                
            async def monitoring_loop(self, max_iterations=5):
                """Main monitoring loop"""
                self.active = True
                try:
                    while self.active and self.iterations < max_iterations:
                        result = await self.run_monitoring_iteration()
                        if result["status"] != "ok":
                            break
                finally:
                    self.active = False
                return self.iterations
        
        # Test monitoring loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            monitor = MockMonitoringLoop()
            iterations = loop.run_until_complete(monitor.monitoring_loop(max_iterations=3))
            assert iterations == 3
            assert monitor.active is False
        finally:
            loop.close()


class TestConfigurationHandling:
    """Test suite for configuration handling"""

    def test_default_config_values(self):
        """Test default configuration values"""
        default_config = {
            "frequency_range": "88-108",
            "sample_rate": 2048000,
            "gain": 20,
            "threshold": 0.8,
            "scan_interval": 5,
            "strict_mode": False
        }
        
        # Validate configuration structure
        assert "frequency_range" in default_config
        assert isinstance(default_config["sample_rate"], int)
        assert isinstance(default_config["gain"], (int, float))
        assert 0 <= default_config["threshold"] <= 1
        assert default_config["scan_interval"] > 0

    def test_config_validation(self):
        """Test configuration validation"""
        def validate_config(config):
            """Validate configuration parameters"""
            errors = []
            
            if "frequency_range" in config:
                freq_range = config["frequency_range"]
                if not isinstance(freq_range, str) or "-" not in freq_range:
                    errors.append("Invalid frequency_range format")
                    
            if "sample_rate" in config:
                if not isinstance(config["sample_rate"], int) or config["sample_rate"] <= 0:
                    errors.append("Invalid sample_rate value")
                    
            if "threshold" in config:
                threshold = config["threshold"]
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    errors.append("Invalid threshold value")
                    
            return errors
        
        # Test valid config
        valid_config = {
            "frequency_range": "88-108",
            "sample_rate": 2048000,
            "threshold": 0.8
        }
        assert len(validate_config(valid_config)) == 0
        
        # Test invalid config
        invalid_config = {
            "frequency_range": "invalid",
            "sample_rate": -1000,
            "threshold": 1.5
        }
        errors = validate_config(invalid_config)
        assert len(errors) == 3

    def test_multiband_config_validation(self):
        """Test multiband configuration validation"""
        def validate_multiband_config(config):
            """Validate multiband configuration"""
            errors = []
            
            if "bands" not in config:
                errors.append("Missing bands configuration")
                return errors
                
            bands = config["bands"]
            if not isinstance(bands, list) or len(bands) == 0:
                errors.append("Bands must be a non-empty list")
                return errors
                
            for i, band in enumerate(bands):
                if not isinstance(band, dict):
                    errors.append(f"Band {i} must be a dictionary")
                    continue
                    
                required_fields = ["start", "end", "name"]
                for field in required_fields:
                    if field not in band:
                        errors.append(f"Band {i} missing field: {field}")
                        
                if "start" in band and "end" in band:
                    if band["start"] >= band["end"]:
                        errors.append(f"Band {i} start frequency must be less than end frequency")
                        
            return errors
        
        # Test valid multiband config
        valid_config = {
            "bands": [
                {"start": 88, "end": 108, "name": "FM"},
                {"start": 400, "end": 500, "name": "UHF"}
            ]
        }
        assert len(validate_multiband_config(valid_config)) == 0
        
        # Test invalid multiband config
        invalid_config = {
            "bands": [
                {"start": 108, "end": 88, "name": "Invalid"},  # start > end
                {"start": 400, "name": "Missing end"}  # missing end
            ]
        }
        errors = validate_multiband_config(invalid_config)
        assert len(errors) >= 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
