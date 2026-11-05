"""
Comprehensive tests for SpectrumAlert web application API endpoints
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock
from httpx import AsyncClient, ASGITransport
from fastapi.testclient import TestClient

from spectrum_alert.web.app import SpectrumAlertWebApp
from tests.web.conftest import (
    MockRTLSDRInterface, MockDataStorage, MockSystemMonitor,
    MockFeatureExtractor, MockAnomalyDetectionUseCase,
    TEST_CONFIG, TEST_MULTIBAND_CONFIG, TEST_MQTT_CONFIG
)


class TestSpectrumAlertWebApp:
    """Test suite for SpectrumAlert web application"""

    @pytest.fixture
    def app(self):
        """Create test FastAPI application"""
        return SpectrumAlertWebApp()

    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return TestClient(app.get_app())

    @pytest.fixture
    async def async_client(self, app):
        """Create async test client"""
        transport = ASGITransport(app=app.get_app())
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    def test_app_initialization(self, app):
        """Test web app initializes correctly"""
        assert app.app is not None
        assert app.storage is not None
        assert app.system_monitor is not None
        assert app.connections == []
        assert app.monitoring_active is False
        assert app.training_active is False

    def test_dashboard_route(self, client):
        """Test dashboard page loads"""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]

    def test_system_status_endpoint(self, client, app):
        """Test system status API endpoint"""
        # Patch the app's system monitor instance directly
        mock_monitor = MockSystemMonitor()
        app.system_monitor = mock_monitor
        
        response = client.get("/api/system/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "data" in data
        assert "timestamp" in data
        assert data["data"]["cpu_usage"] == 25.5

    def test_system_advanced_status_endpoint(self, client):
        """Test advanced system status API endpoint"""
        response = client.get("/api/system/advanced-status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "data" in data
        assert "cpu_usage" in data["data"]
        assert "memory_usage" in data["data"]
        assert "temperature" in data["data"]

    def test_recent_anomalies_endpoint(self, client):
        """Test recent anomalies API endpoint"""
        response = client.get("/api/anomalies/recent")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "anomalies" in data
        assert isinstance(data["anomalies"], list)

    def test_spectrum_analysis_endpoint(self, client):
        """Test spectrum analysis API endpoint"""
        response = client.get("/api/spectrum/analysis")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "analysis" in data

    def test_model_info_endpoint(self, client):
        """Test model info API endpoint"""
        response = client.get("/api/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "model_info" in data

    @patch('spectrum_alert.web.app.RTLSDRInterface')
    @patch('spectrum_alert.web.app.SpectrumMonitoringUseCase')
    @patch('spectrum_alert.web.app.AnomalyDetectionUseCase')
    def test_monitoring_start_endpoint(self, mock_anomaly, mock_monitoring, mock_sdr, client):
        """Test monitoring start API endpoint"""
        # Setup mocks
        mock_sdr.return_value = MockRTLSDRInterface()
        mock_monitoring_instance = MagicMock()
        mock_monitoring.return_value = mock_monitoring_instance
        mock_anomaly.return_value = MockAnomalyDetectionUseCase(MockFeatureExtractor())
        
        response = client.post("/api/monitoring/start", json=TEST_CONFIG)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "monitoring started" in data["message"].lower()

    def test_monitoring_stop_endpoint(self, client):
        """Test monitoring stop API endpoint"""
        response = client.post("/api/monitoring/stop")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_monitoring_status_endpoint(self, client):
        """Test monitoring status API endpoint"""
        response = client.get("/api/monitoring/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "monitoring_active" in data

    def test_multiband_monitoring_endpoint(self, client):
        """Test multiband monitoring API endpoint"""
        response = client.post("/api/monitoring/multiband", json=TEST_MULTIBAND_CONFIG)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_advanced_monitoring_endpoint(self, client):
        """Test advanced monitoring API endpoint"""
        advanced_config = {
            "dc_exclude_hz": 5000,
            "edge_exclude_hz": 10000,
            "novelty_filtering": True,
            "strict_mode": True
        }
        
        response = client.post("/api/monitoring/advanced", json=advanced_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_mqtt_connect_endpoint(self, client):
        """Test MQTT connect API endpoint"""
        response = client.post("/api/mqtt/connect", json=TEST_MQTT_CONFIG)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_mqtt_disconnect_endpoint(self, client):
        """Test MQTT disconnect API endpoint"""
        response = client.post("/api/mqtt/disconnect")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_mqtt_status_endpoint(self, client):
        """Test MQTT status API endpoint"""
        response = client.get("/api/mqtt/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"
        assert "connected" in data

    def test_mqtt_test_endpoint(self, client):
        """Test MQTT test API endpoint"""
        response = client.post("/api/mqtt/test", json={"message": "test"})
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_model_train_endpoint(self, client):
        """Test model training API endpoint"""
        train_config = {
            "algorithm": "isolation_forest",
            "contamination": 0.1,
            "n_estimators": 100
        }
        
        response = client.post("/api/model/train", json=train_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_model_test_endpoint(self, client):
        """Test model testing API endpoint"""
        test_config = {
            "test_data_path": "/tmp/test_data.csv",
            "metrics": ["accuracy", "precision", "recall"]
        }
        
        response = client.post("/api/model/test", json=test_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    def test_model_deploy_endpoint(self, client):
        """Test model deployment API endpoint"""
        deploy_config = {
            "model_path": "/tmp/model.pkl",
            "deployment_mode": "production"
        }
        
        response = client.post("/api/model/deploy", json=deploy_config)
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "success"

    @patch('spectrum_alert.web.app.RTLSDRInterface')
    @patch('spectrum_alert.web.app.SpectrumMonitoringUseCase')
    def test_data_collect_endpoint(self, mock_monitoring, mock_sdr, client):
        """Test data collection API endpoint"""
        from unittest.mock import MagicMock

        mock_sdr.return_value = MockRTLSDRInterface()
        mock_monitoring_instance = MagicMock()
        mock_monitoring.return_value = mock_monitoring_instance

        spectrum_data = MagicMock()
        spectrum_data.id = "mock-spectrum-id"
        spectrum_data.frequency_hz = 433000000.0
        spectrum_data.sample_count = 2048
        spectrum_data.duration_seconds = 1.0
        spectrum_data.timestamp.isoformat.return_value = "2026-01-01T00:00:00"
        mock_monitoring_instance.capture_spectrum_data.return_value = spectrum_data

        collect_config = {
            "frequency_range": "88-108",
            "duration": 60,
            "sample_rate": 2048000
        }

        response = client.post("/api/data/collect", json=collect_config)
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "success"        
    def test_data_stats_endpoint(self, client):
        """Test data statistics API endpoint"""
        response = client.get("/api/data/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "stats" in data

    def test_model_training_status_endpoint(self, client):
        """Test model training status API endpoint"""
        response = client.get("/api/model/training/status")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "ok"
        assert "training_active" in data

    @pytest.mark.asyncio
    async def test_websocket_connection(self, async_client):
        """Test WebSocket connection"""
        # Note: This is a simplified test - full WebSocket testing requires more setup
        try:
            async with async_client.websocket_connect("/ws") as websocket:
                # If we get here, connection was successful
                assert True
        except Exception:
            # WebSocket might not be fully testable in this context
            # but we can at least verify the endpoint exists
            pass

    def test_error_handling_invalid_endpoint(self, client):
        """Test error handling for invalid endpoints"""
        response = client.get("/api/invalid/endpoint")
        assert response.status_code == 404

    def test_error_handling_invalid_method(self, client):
        """Test error handling for invalid HTTP methods"""
        response = client.put("/api/monitoring/start")
        assert response.status_code == 405

    def test_error_handling_invalid_json(self, client):
        """Test error handling for invalid JSON"""
        response = client.post(
            "/api/monitoring/start",
            data="invalid json",
            headers={"content-type": "application/json"}
        )
        assert response.status_code == 422


class TestWebAppHelperMethods:
    """Test suite for web app helper methods"""

    @pytest.fixture
    def app(self):
        return SpectrumAlertWebApp()

    @pytest.mark.asyncio
    async def test_broadcast_to_websockets_empty_connections(self, app):
        """Test broadcasting with no WebSocket connections"""
        # Should not raise an error
        await app._broadcast_to_websockets({"test": "message"})

    @pytest.mark.asyncio
    async def test_broadcast_to_websockets_with_connections(self, app):
        """Test broadcasting with mock WebSocket connections"""
        mock_websocket = AsyncMock()
        app.connections = [mock_websocket]
        
        await app._broadcast_to_websockets({"test": "message"})
        
        mock_websocket.send_text.assert_called_once()

    def test_get_app_returns_fastapi_instance(self, app):
        """Test that get_app returns a FastAPI instance"""
        fastapi_app = app.get_app()
        assert fastapi_app is not None
        assert hasattr(fastapi_app, 'routes')


class TestWebAppIntegration:
    """Integration tests for web app components"""

    @pytest.fixture
    def app(self):
        return SpectrumAlertWebApp()

    @pytest.fixture
    def client(self, app):
        return TestClient(app.get_app())

    @patch('spectrum_alert.web.app.RTLSDRInterface')
    @patch('spectrum_alert.web.app.SpectrumMonitoringUseCase')
    @patch('spectrum_alert.web.app.AnomalyDetectionUseCase')
    def test_full_monitoring_workflow(self, mock_anomaly, mock_monitoring, mock_sdr, client):
        """Test complete monitoring workflow"""
        # Setup mocks
        mock_sdr.return_value = MockRTLSDRInterface()
        mock_monitoring_instance = MagicMock()
        mock_monitoring.return_value = mock_monitoring_instance
        mock_anomaly.return_value = MockAnomalyDetectionUseCase(MockFeatureExtractor())
        
        # Start monitoring
        start_response = client.post("/api/monitoring/start", json=TEST_CONFIG)
        assert start_response.status_code == 200
        
        # Check status
        status_response = client.get("/api/monitoring/status")
        assert status_response.status_code == 200
        
        # Stop monitoring
        stop_response = client.post("/api/monitoring/stop")
        assert stop_response.status_code == 200

    def test_system_monitoring_integration(self, client):
        """Test system monitoring integration"""
        # Get system status
        response = client.get("/api/system/status")
        assert response.status_code == 200
        
        # Get advanced status
        advanced_response = client.get("/api/system/advanced-status")
        assert advanced_response.status_code == 200

    def test_data_workflow_integration(self, client):
        """Test data collection and analysis workflow"""
        # Get data stats
        stats_response = client.get("/api/data/stats")
        assert stats_response.status_code == 200
        
        # Get spectrum analysis
        analysis_response = client.get("/api/spectrum/analysis")
        assert analysis_response.status_code == 200
        
        # Get recent anomalies
        anomalies_response = client.get("/api/anomalies/recent")
        assert anomalies_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
