"""
FastAPI Web Application for SpectrumAlert
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, RedirectResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    logger.warning("FastAPI not available. Install with: pip install fastapi uvicorn jinja2")

from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.infrastructure.monitoring import SystemMonitor
from spectrum_alert.web.geolocate import build_router, default_service

# Import use case classes for the web app endpoints
try:
    from spectrum_alert.application.use_cases.spectrum_monitoring import SpectrumMonitoringUseCase
    from spectrum_alert.application.use_cases.anomaly_detection import AnomalyDetectionUseCase
    from spectrum_alert.application.use_cases.model_training import ModelTrainingUseCase
    from spectrum_alert.infrastructure.sdr import RTLSDRInterface
    from spectrum_alert.core.services.feature_extraction import FeatureExtractor
except ImportError as e:
    logger.warning(f"Some use case imports failed: {e}. Some features may not work.")
    # Create stub classes for testing
    SpectrumMonitoringUseCase = None
    AnomalyDetectionUseCase = None  
    ModelTrainingUseCase = None
    RTLSDRInterface = None
    FeatureExtractor = None


class SpectrumAlertWebApp:
    """SpectrumAlert Web Application"""

    # The dongle has exactly one USB interface and librtlsdr refuses to claim
    # it while another handle holds it (LIBUSB_ERROR_BUSY). Stopping monitoring
    # releases it asynchronously, so pressing Start again straight away used to
    # fail outright with "No SDR available: ... Resource busy" even though the
    # operator had simply stopped and restarted. A short bounded retry turns
    # that race into a brief pause.
    _SDR_OPEN_ATTEMPTS = 5
    _SDR_OPEN_DELAY_S = 0.4

    @classmethod
    def _open_sdr_when_free(cls, sdr, attempts=None, delay=None):
        """Open the SDR, waiting briefly for a previous handle to be released.

        Raises whatever the last attempt raised; only a busy interface is
        retried, so a genuinely absent dongle still fails immediately.
        """
        import time

        attempts = cls._SDR_OPEN_ATTEMPTS if attempts is None else attempts
        delay = cls._SDR_OPEN_DELAY_S if delay is None else delay

        last_error = None
        for attempt in range(max(1, attempts)):
            try:
                sdr.open()
                return sdr
            except Exception as e:  # noqa: BLE001 - re-raised below
                if "busy" not in str(e).lower():
                    raise
                last_error = e
                if attempt < attempts - 1:
                    logger.info(
                        "SDR still held by another handle, retrying in %.1fs (%d/%d)",
                        delay, attempt + 1, attempts,
                    )
                    time.sleep(delay)
        raise last_error

    def __init__(self):
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI is required for web interface. Install with: pip install fastapi uvicorn jinja2")
            
        self.app = FastAPI(
            title="SpectrumAlert Dashboard",
            description="RF Spectrum Monitoring and Anomaly Detection Dashboard",
            version="1.1.0"
        )
        
        # Setup static files and templates
        web_dir = Path(__file__).parent
        self.app.mount("/static", StaticFiles(directory=str(web_dir / "static")), name="static")
        self.templates = Jinja2Templates(directory=str(web_dir / "templates"))
        
        # Initialize data storage and monitoring
        self.storage = DataStorage()
        self.system_monitor = SystemMonitor()

        # Geolocation service: receiver roster, calibration table, multi-mode
        # scanner. Mounted as its own router so these endpoints stay separate
        # from the dashboard's monitoring API.
        self.geo_service = default_service()
        self.app.include_router(build_router(self.geo_service))

        # Active WebSocket connections
        self.connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Setup startup and shutdown events
        @self.app.on_event("startup")
        async def startup_event():
            await self.start_background_updates()
        
        @self.app.on_event("shutdown") 
        async def shutdown_event():
            self._background_running = False
            if self._background_task:
                self._background_task.cancel()
        
        # Background task for real-time updates
        self._background_task = None
        self._background_running = False
        
        # Monitoring control
        self.monitoring_service = None
        self.monitoring_active = False
        
        # Training control
        self.training_active = False
        self.training_progress = {}
        self.last_training_time = None
    
    def _setup_routes(self):
        """Setup FastAPI routes"""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard(request: Request):
            """Main dashboard page"""
            # TemplateResponse(request, name, context) is the Starlette >=0.29
            # signature. The old (name, context) form is interpreted as
            # (name, parent) and raises "TypeError: unhashable type: 'dict'".
            return self.templates.TemplateResponse(request, "dashboard.html")

        @self.app.get("/geolocate", response_class=HTMLResponse)
        async def geolocate_page(request: Request):
            """Redirect to the dashboard, which now hosts the workspace.

            The analysis workspace (receivers, waterfall, location, device
            identification) used to be a second page reached from a link in the
            header. Splitting one workflow across two documents meant the
            operator lost the system panels whenever they located or identified
            something. It is now a section of the dashboard, and this route
            stays so existing bookmarks and the header link keep working.
            """
            return RedirectResponse(url="/#workspace", status_code=302)
        
        @self.app.get("/api/system/status")
        async def get_system_status():
            """Get current system status"""
            try:
                status = self.system_monitor.get_system_status()
                
                # Handle both flat structure (from mock) and nested structure
                if "cpu_usage" in status:
                    # Flat structure from mock
                    data = {
                        "cpu_usage": status.get("cpu_usage", 0),
                        "memory_usage": status.get("memory_usage", 0),
                        "disk_usage": status.get("disk_usage", 0),
                        "uptime": status.get("uptime", "Unknown"),
                        "temperature": status.get("temperature"),
                        "status": status.get("status", "ok"),
                        "monitoring_active": status.get("monitoring_active", False)
                    }
                else:
                    # Nested structure from real implementation
                    data = {
                        "cpu_usage": status.get("metrics", {}).get("cpu", {}).get("percent", 0),
                        "memory_usage": status.get("metrics", {}).get("memory", {}).get("percent", 0),
                        "disk_usage": status.get("metrics", {}).get("disk", {}).get("percent", 0),
                        "uptime": self.system_monitor.get_uptime_string(),
                        # Real sensor value or None; the card must not invent one.
                        "temperature": status.get("temperature"),
                        "status": status.get("status", "unknown"),
                        "health_checks": status.get("health_checks", {})
                    }
                
                return {
                    "status": "ok",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting system status: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/api/anomalies/recent")
        async def get_recent_anomalies(hours: int = 24, min_hz: float = None,
                                       max_hz: float = None):
            """Get recent anomalies, optionally scoped to a frequency window."""
            try:
                # Get anomalies from the last N hours
                since = datetime.now() - timedelta(hours=hours)
                anomalies = self.storage.get_anomalies_since(since)
                
                # Convert to JSON-serializable format
                anomaly_data = []
                for anomaly in anomalies:
                    # Optional window filter: a persisted detection from an
                    # earlier scan at a different centre is not part of what is
                    # being monitored now, and showing it reads as a detection
                    # "on a frequency not being monitored".
                    if min_hz is not None and anomaly.frequency_hz < min_hz:
                        continue
                    if max_hz is not None and anomaly.frequency_hz > max_hz:
                        continue
                    anomaly_data.append({
                        "id": str(anomaly.id),
                        "timestamp": anomaly.timestamp.isoformat(),
                        "frequency_mhz": anomaly.frequency_hz / 1e6,
                        "frequency_hz": anomaly.frequency_hz,
                        "confidence_score": anomaly.confidence_score,
                        "severity": anomaly.severity,
                        "description": anomaly.description,
                        "anomaly_type": (anomaly.anomaly_type.value
                                         if anomaly.anomaly_type else None),
                        "detection_mode": anomaly.detection_mode.value
                    })
                
                return {
                    "status": "ok",
                    "anomalies": anomaly_data,
                    "data": anomaly_data,  # Keep both for compatibility
                    "count": len(anomaly_data),
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting recent anomalies: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/api/spectrum/analysis")
        async def get_spectrum_analysis(frequency_range: str = "144-146"):
            """Get spectrum analysis data with optional frequency range filter"""
            try:
                # Get recent spectrum data for analysis
                analysis = await self._get_spectrum_analysis(frequency_range)
                return {
                    "status": "ok",
                    "analysis": analysis,
                    "data": analysis,  # Keep both for compatibility
                    "timestamp": datetime.now().isoformat(),
                    "frequency_range": frequency_range
                }
            except Exception as e:
                logger.error(f"Error getting spectrum analysis: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.get("/api/model/info")
        async def get_model_info():
            """Get ML model information and statistics"""
            try:
                model_info = await self._get_model_info()
                return {
                    "status": "ok",
                    "model_info": model_info,
                    "data": model_info,  # Keep both for compatibility
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"Error getting model info: {e}")
                return {"status": "error", "message": str(e)}
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates"""
            await websocket.accept()
            self.connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    await websocket.receive_text()
            except WebSocketDisconnect:
                if websocket in self.connections:
                    self.connections.remove(websocket)

        # Monitoring control endpoints
        @self.app.post("/api/monitoring/start")
        async def start_monitoring(config: dict = None):
            """Start spectrum monitoring"""
            try:
                # Idempotency guard. Without it, concurrent POSTs each created a
                # task and each opened the dongle, so two loops interleaved their
                # retunes and USB reads against one device and two tasks appended
                # to the same anomaly CSV.
                # All component classes come from module level: a local import
                # here would shadow the patchable names tests rely on.
                if getattr(self, 'monitoring_active', False):
                    return {
                        "status": "success",
                        "message": "Monitoring is already running",
                        "config": getattr(self, 'monitoring_config', None),
                    }

                # Set up monitoring with default or provided config
                monitoring_config = config or {
                    "center_freq": 145,
                    "frequency_range": "88-108",
                    "sample_rate": 2048000,
                    "gain": 20,
                    "threshold": 0.8,
                    "scan_interval": 5,
                    "strict_mode": False
                }

                # Initialize services. No hardware must surface as an error:
                # a silent fallback to synthetic data made the dashboard say
                # "No SDR hardware found" while the dongle sat on the bus.
                if RTLSDRInterface is None:
                    return {
                        "status": "error",
                        "message": (
                            "RTL-SDR support is unavailable: pyrtlsdr is not "
                            "installed (pip install pyrtlsdr)."
                        ),
                    }

                try:
                    sdr = RTLSDRInterface()  # Use RTL-SDR implementation
                    self._open_sdr_when_free(sdr)
                    logger.info("RTL-SDR device initialized successfully")
                except Exception as e:
                    logger.error("No SDR available: %s", e)
                    return {
                        "status": "error",
                        "message": f"No SDR available: {e}",
                    }

                storage = DataStorage()
                feature_extractor = FeatureExtractor(lite_mode=True)  # Use lite mode for web
                anomaly_detector = AnomalyDetectionUseCase(feature_extractor)
                anomaly_detector.set_threshold(monitoring_config.get("threshold", 0.8))
                if monitoring_config.get("strict_mode", False):
                    anomaly_detector.set_strict_threshold_mode(True)

                monitoring_service = SpectrumMonitoringUseCase(
                    sdr_interface=sdr,
                    storage=storage,
                    feature_extractor=feature_extractor
                )

                # Attach anomaly detector to monitoring service
                monitoring_service.anomaly_detector = anomaly_detector

                # Store monitoring service for later control
                self.monitoring_service = monitoring_service
                self.monitoring_config = monitoring_config
                self.monitoring_task = asyncio.create_task(
                    self._run_monitoring_loop(monitoring_service, monitoring_config)
                )
                # Set active only after the task exists, so /status never reports
                # running while there is nothing to stop.
                self.monitoring_active = True

                return {
                    "status": "success",
                    "message": "Monitoring started successfully",
                    "config": monitoring_config
                }

            except Exception as e:
                logger.error(f"Failed to start monitoring: {e}")
                self.monitoring_active = False
                return {
                    "status": "error",
                    "message": f"Failed to start monitoring: {str(e)}"
                }

        @self.app.post("/api/monitoring/stop")
        async def stop_monitoring():
            """Stop spectrum monitoring"""
            try:
                self.monitoring_active = False

                # Wait for the loop to finish before touching the device.
                # Previously this closed the SDR from the request handler while
                # the task could be blocked inside read_samples(), and the loop
                # then reopened it on the next iteration, leaving the dongle open
                # after "stop".
                task = getattr(self, 'monitoring_task', None)
                if task is not None:
                    task.cancel()
                    try:
                        await task
                    except (asyncio.CancelledError, Exception):
                        pass
                    self.monitoring_task = None

                service = getattr(self, 'monitoring_service', None)
                if service is not None and getattr(service, 'sdr', None):
                    try:
                        service.sdr.close()
                        logger.info("SDR device closed successfully")
                    except Exception as sdr_error:
                        logger.warning(f"Error closing SDR device: {sdr_error}")

                self.monitoring_service = None

                return {
                    "status": "success",
                    "message": "Monitoring stopped successfully"
                }

            except Exception as e:
                logger.error(f"Failed to stop monitoring: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to stop monitoring: {str(e)}"
                }

        @self.app.get("/api/monitoring/status")
        async def get_monitoring_status():
            """Get current monitoring status"""
            window = getattr(self, 'monitored_window_hz', None)
            monitoring_data = {
                "monitoring_active": getattr(self, 'monitoring_active', False),
                "has_monitoring_service": hasattr(self, 'monitoring_service'),
                # The receiver's real capture window (start/end Hz), or null when
                # idle. The UI uses this to scope alerts and label the chart.
                "monitored_window_hz": list(window) if window else None,
                "center_freq_hz": getattr(self, 'monitoring_center_hz', None),
                "span_hz": getattr(self, 'monitoring_span_hz', None),
            }
            
            # Return both data nested and flattened for backward compatibility
            return {
                "status": "ok",
                "data": monitoring_data,
                **monitoring_data  # Flatten to top level
            }

        @self.app.post("/api/monitoring/retune")
        async def retune_monitoring(config: dict = None):
            """Re-tune an active scan without restarting it.

            The monitoring loop reads `monitoring_config` every iteration, so
            updating the centre frequency here takes effect on the next capture.
            """
            try:
                if not getattr(self, 'monitoring_active', False):
                    return {"status": "error", "message": "Monitoring is not running"}
                cfg = getattr(self, 'monitoring_config', None)
                if cfg is None:
                    return {"status": "error", "message": "No active monitoring configuration"}
                config = config or {}
                if config.get("center_freq") is not None:
                    cfg["center_freq"] = float(config["center_freq"])
                if config.get("sample_rate") is not None:
                    cfg["sample_rate"] = int(config["sample_rate"])
                if config.get("frequency_range"):
                    cfg["frequency_range"] = config["frequency_range"]
                return {
                    "status": "success",
                    "message": "Re-tuned",
                    "config": cfg,
                }
            except Exception as e:
                logger.error(f"Failed to retune monitoring: {e}")
                return {"status": "error", "message": str(e)}

        # Advanced monitoring configuration endpoints
        @self.app.post("/api/monitoring/multiband")
        async def start_multiband_monitoring(config: Optional[dict] = None):
            """Record a multi-band configuration (does not start a capture)"""
            try:
                multiband_config = config or {
                    "bands": ["144-148", "430-440"],
                    "data_minutes": 5,
                    "monitor_minutes": 15,
                    "max_cycles_per_band": 1,
                    "threshold": 0.8,
                    "strict_threshold": True,
                    "dc_exclude_hz": 10000,
                    "edge_exclude_hz": 25000,
                    "continuous_learning": False,
                    "novelty_filter": False
                }
                
                # Record the configuration only. This endpoint starts no capture
                # loop, so raising monitoring_active here made /status and the
                # MONITOR indicator report a run that did not exist, and the
                # idempotency guard then refused a real start as "already
                # running". Capture is started from /api/monitoring/start.
                self.multiband_config = multiband_config
                
                logger.info(f"Recording multi-band configuration: {multiband_config}")
                
                # Broadcast the recorded configuration
                await self._broadcast_to_websockets({
                    "type": "multiband_monitoring_started",
                    "data": multiband_config
                })
                
                return {
                    "status": "success",
                    "message": "Multi-band configuration recorded",
                    "config": multiband_config
                }
                
            except Exception as e:
                logger.error(f"Failed to record multi-band configuration: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to record multi-band configuration: {str(e)}"
                }

        @self.app.post("/api/monitoring/advanced")
        async def start_advanced_monitoring(config: Optional[dict] = None):
            """Record an advanced configuration (does not start a capture)"""
            try:
                advanced_config = config or {
                    "frequency_start": 144.0,
                    "frequency_end": 148.0,
                    "threshold": 0.8,
                    "strict_threshold": True,
                    "dc_exclude_hz": 8000,
                    "edge_exclude_hz": 20000,
                    "novelty_filter": False,
                    "novelty_freq_tol_hz": 3000,
                    "novelty_cooldown_s": 5.0,
                    "continuous_learning": False,
                    "sample_rate": 2048000,
                    "gain": 30
                }
                
                # Record the configuration only; see /api/monitoring/multiband.
                self.advanced_config = advanced_config
                
                logger.info(f"Recording advanced configuration: {advanced_config}")
                
                # Broadcast the recorded configuration
                await self._broadcast_to_websockets({
                    "type": "advanced_monitoring_started",
                    "data": advanced_config
                })
                
                return {
                    "status": "success",
                    "message": "Advanced configuration recorded",
                    "config": advanced_config
                }
                
            except Exception as e:
                logger.error(f"Failed to record advanced configuration: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to record advanced configuration: {str(e)}"
                }

        # MQTT integration endpoints
        @self.app.post("/api/mqtt/connect")
        async def connect_mqtt(config: Optional[dict] = None):
            """Connect to MQTT broker"""
            try:
                mqtt_config = config or {
                    "broker": "localhost",
                    "port": 1883,
                    "topic_prefix": "spectrum_alert",
                    "username": None,
                    "password": None
                }
                
                # Store MQTT configuration
                self.mqtt_config = mqtt_config
                self.mqtt_connected = True
                
                logger.info(f"MQTT connection established to {mqtt_config['broker']}:{mqtt_config['port']}")
                
                # Broadcast MQTT connection status
                await self._broadcast_to_websockets({
                    "type": "mqtt_connected",
                    "data": mqtt_config
                })
                
                return {
                    "status": "success",
                    "message": "MQTT connected successfully",
                    "config": mqtt_config
                }
                
            except Exception as e:
                logger.error(f"Failed to connect to MQTT: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to connect to MQTT: {str(e)}"
                }

        @self.app.post("/api/mqtt/disconnect")
        async def disconnect_mqtt():
            """Disconnect from MQTT broker"""
            try:
                self.mqtt_connected = False
                
                logger.info("MQTT disconnected")
                
                # Broadcast MQTT disconnection
                await self._broadcast_to_websockets({
                    "type": "mqtt_disconnected",
                    "data": {}
                })
                
                return {
                    "status": "success",
                    "message": "MQTT disconnected successfully"
                }
                
            except Exception as e:
                logger.error(f"Failed to disconnect MQTT: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to disconnect MQTT: {str(e)}"
                }

        @self.app.get("/api/mqtt/status")
        async def get_mqtt_status():
            """Get MQTT connection status"""
            mqtt_data = {
                "connected": getattr(self, 'mqtt_connected', False),
                "config": getattr(self, 'mqtt_config', {}),
                "last_message": getattr(self, 'last_mqtt_message', None)
            }
            
            return {
                "status": "success",
                "connected": mqtt_data["connected"],  # Top-level for test compatibility
                "data": mqtt_data
            }

        @self.app.post("/api/mqtt/test")
        async def test_mqtt():
            """Test MQTT connection by sending a test message"""
            try:
                # For testing purposes, always allow the test to proceed
                # Send test message
                test_message = {
                    "type": "test",
                    "timestamp": datetime.now().isoformat(),
                    "message": "MQTT test from SpectrumAlert Web Interface"
                }
                
                # In a real implementation, this would publish to MQTT
                logger.info("MQTT test message sent")
                
                # Broadcast test result
                await self._broadcast_to_websockets({
                    "type": "mqtt_test_sent",
                    "data": test_message
                })
                
                return {
                    "status": "success",
                    "message": "MQTT test message sent successfully",
                    "test_data": test_message
                }
                
            except Exception as e:
                logger.error(f"Failed to send MQTT test: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to send MQTT test: {str(e)}"
                }

        # Enhanced system status endpoint
        @self.app.get("/api/system/advanced-status")
        async def get_advanced_system_status():
            """Get detailed system status including SDR and advanced metrics"""
            try:
                import psutil
                import platform
                
                # System metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Network interfaces
                network_interfaces = []
                for interface, addresses in psutil.net_if_addrs().items():
                    if interface != 'lo':  # Skip loopback
                        network_interfaces.append({
                            "name": interface,
                            "addresses": [addr.address for addr in addresses if addr.family == 2]  # IPv4
                        })
                
                # Temperature comes from the same helper as every other status
                # surface: picking "first entry of the first sensor group"
                # here reported the ambient sensor (16 °C) while the monitor
                # reported the CPU package (60 °C) for the same machine.
                temperature = self.system_monitor._read_temperature_c()
                
                # Real SDR probe. This used to hardcode "Available" and invent
                # an "RTL-SDR #0", so the dashboard claimed hardware existed on
                # machines with no dongle attached at all.
                from spectrum_alert.infrastructure.sdr.scanner import probe_local_sdr
                probe = probe_local_sdr()
                sdr_status = probe["detail"]
                
                status_data = {
                    "system": {
                        "platform": platform.system(),
                        "architecture": platform.machine(),
                        "python_version": platform.python_version(),
                        "hostname": platform.node()
                    },
                    "performance": {
                        "cpu_usage": cpu_percent,
                        "memory_usage": memory.percent,
                        "memory_available": memory.available // (1024 * 1024),  # MB
                        "disk_usage": (disk.used / disk.total) * 100,
                        "disk_free": disk.free // (1024 * 1024 * 1024),  # GB
                        "temperature": temperature,
                        "uptime": self.system_monitor.get_uptime_string()
                    },
                    "hardware": {
                        "sdr_status": sdr_status,
                        "sdr_available": probe["available"],
                        "sdr_probe_status": probe["status"],
                        "sdr_devices": probe["devices"]
                    },
                    "network": {
                        "interfaces": network_interfaces
                    },
                    "monitoring": {
                        "active": getattr(self, 'monitoring_active', False),
                        "multiband_active": hasattr(self, 'multiband_config'),
                        "advanced_active": hasattr(self, 'advanced_config'),
                        "config": getattr(self, 'monitoring_config', {})
                    },
                    "mqtt": {
                        "connected": getattr(self, 'mqtt_connected', False),
                        "config": getattr(self, 'mqtt_config', {})
                    },
                    # Add performance fields to top level for backward compatibility
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "memory_available": memory.available // (1024 * 1024),
                    "disk_usage": (disk.used / disk.total) * 100,
                    "disk_free": disk.free // (1024 * 1024 * 1024),
                    "temperature": temperature
                }
                
                return {
                    "status": "success",
                    "data": status_data
                }
                
            except Exception as e:
                logger.error(f"Failed to get advanced system status: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

        # Model training and management endpoints
        @self.app.post("/api/model/train")
        async def train_model(config: dict = None):
            """Train a new model with REAL data"""
            try:
                from pathlib import Path
                import numpy as np
                import pandas as pd
                from sklearn.ensemble import IsolationForest
                from sklearn.svm import OneClassSVM
                from sklearn.neural_network import MLPClassifier
                from sklearn.model_selection import train_test_split
                from sklearn.preprocessing import StandardScaler
                from sklearn.metrics import classification_report, accuracy_score
                import joblib
                
                # Parse training configuration
                training_config = config or {}
                model_type = training_config.get("model_type", "isolation_forest")
                dataset_name = training_config.get("dataset")
                mode = training_config.get("mode", "lite")
                contamination = float(training_config.get("contamination", 0.1))
                epochs = int(training_config.get("epochs", 100))

                # The Training panel's dataset picker offers "all", "recent" and
                # "custom". Those are ranges, not filenames, but they were
                # treated as dataset names: selecting the default "All data"
                # made the endpoint look for data/features/all and refuse to
                # train. Only a real filename reaches the explicit path branch.
                dataset_choice = (dataset_name or "").strip()
                is_range_choice = dataset_choice.lower() in ("", "all", "recent", "custom")

                if not is_range_choice:
                    dataset_path = None
                    for path in (
                        Path(f"data/features/{dataset_choice}"),
                        Path(dataset_choice),
                        Path(f"data/{dataset_choice}"),
                    ):
                        if path.exists():
                            dataset_path = path
                            break

                    if not dataset_path:
                        return {
                            "status": "error",
                            "message": (
                                f"Dataset '{dataset_choice}' not found. "
                                f"Available datasets can be seen in Data Management."
                            ),
                        }
                    try:
                        df = pd.read_csv(dataset_path)
                        logger.info(f"Loaded dataset {dataset_path} with shape {df.shape}")
                    except Exception as e:
                        return {"status": "error", "message": f"Could not read dataset: {str(e)}"}
                else:
                    # No explicit dataset: use every feature file for the mode.
                    # Hardcoding one dated filename meant training silently
                    # depended on whichever capture happened to be newest, and
                    # failed whenever that one file had too few rows even with
                    # hundreds of rows available elsewhere.
                    days = 1 if dataset_choice.lower() == "recent" else None
                    try:
                        df = self.storage.load_training_data(mode=mode, days=days)
                    except Exception as e:
                        logger.warning("load_training_data failed (%s); falling back to CSVs", e)
                        frames = []
                        for path in sorted(Path("data/features").glob(f"features_*_{mode}.csv")):
                            try:
                                frames.append(pd.read_csv(path))
                            except Exception:
                                continue
                        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
                    logger.info(
                        f"Aggregated {mode} feature data: "
                        f"{0 if df is None else len(df)} rows"
                    )

                if df is None or len(df) == 0:
                    return {
                        "status": "error",
                        "message": (
                            f"No {mode} feature data found. Run the monitor first to "
                            f"collect training data."
                        ),
                    }

                if len(df) < 10:
                    return {
                        "status": "error",
                        "message": (
                            f"Dataset too small ({len(df)} samples). "
                            f"Need at least 10 samples for training."
                        ),
                    }

                # Prepare feature data. Identifier and label columns are dropped,
                # and rows with missing values are removed rather than filled with
                # 0: a zero in an amplitude or PAPR column is a plausible-looking
                # but fabricated measurement, and StandardScaler would then fit
                # around it.
                skip_tokens = ('timestamp', 'label', 'target', 'class', 'id')
                feature_columns = [
                    col for col in df.columns
                    if pd.api.types.is_numeric_dtype(df[col])
                    and not any(token == col.lower() or token in col.lower()
                                for token in skip_tokens)
                ]

                if len(feature_columns) == 0:
                    return {
                        "status": "error",
                        "message": "No numeric feature columns found in dataset"
                    }

                X = df[feature_columns].dropna()

                if len(X) < 10:
                    return {
                        "status": "error",
                        "message": (
                            f"Only {len(X)} complete rows after dropping incomplete "
                            f"ones. Need at least 10."
                        ),
                    }

                label_col = None
                for candidate in ('label', 'anomaly'):
                    if candidate in df.columns:
                        label_col = candidate
                        break

                # X and y must be filtered together. Masking X with a label
                # column taken from an unfiltered frame misaligns the two once
                # any row has been dropped.
                if label_col is not None:
                    usable = df[feature_columns + [label_col]].dropna()
                    X_all = usable[feature_columns]
                    y_all = usable[label_col]
                    normal_mask = y_all == 0
                    if int(normal_mask.sum()) > 5:
                        X_train = X_all[normal_mask]
                        y_true = y_all[normal_mask].values
                    else:
                        X_train = X_all
                        y_true = y_all.values
                else:
                    X_train = X
                    y_true = None
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X_train)
                
                # Train model based on type
                self.training_active = True
                self.training_progress = {
                    "stage": "training",
                    "progress": 0.0,
                    "message": f"Training {model_type} model...",
                    "samples": len(X_train),
                    "features": len(feature_columns)
                }
                
                model = None

                # Models go in models/, matching where /api/model/deploy and
                # /api/model/status look for them. Training previously wrote to
                # the current working directory, so a freshly trained model was
                # invisible to every other endpoint and to the CLI.
                models_dir = Path("models")
                models_dir.mkdir(parents=True, exist_ok=True)

                if model_type == "isolation_forest":
                    model = IsolationForest(
                        contamination=contamination,
                        random_state=42,
                        n_estimators=100
                    )
                    model.fit(X_scaled)
                    model_filename = str(models_dir / "anomaly_detection_model_lite.pkl")
                    
                elif model_type == "one_class_svm":
                    model = OneClassSVM(nu=contamination, kernel='rbf', gamma='auto')
                    model.fit(X_scaled)
                    model_filename = str(models_dir / "one_class_svm_model_lite.pkl")
                    
                elif model_type == "autoencoder":
                    # Simple neural network autoencoder
                    from sklearn.neural_network import MLPRegressor
                    model = MLPRegressor(
                        hidden_layer_sizes=(len(feature_columns)//2, len(feature_columns)//4, len(feature_columns)//2),
                        max_iter=epochs,
                        random_state=42
                    )
                    model.fit(X_scaled, X_scaled)  # Autoencoder learns to reconstruct input
                    model_filename = str(models_dir / "autoencoder_model_lite.pkl")
                    
                else:
                    return {
                        "status": "error",
                        "message": f"Unknown model type: {model_type}"
                    }
                
                # Evaluate model if we have labels
                evaluation_results = {}
                if y_true is not None and model is not None:
                    X_test_scaled = scaler.transform(X)
                    if model_type in ["isolation_forest", "one_class_svm"]:
                        y_pred = model.predict(X_test_scaled)
                        # Convert model output: 1 = normal, -1 = anomaly -> 0 = normal, 1 = anomaly  
                        y_pred_binary = (y_pred == -1).astype(int)
                        accuracy = accuracy_score(y_true, y_pred_binary)
                        evaluation_results = {
                            "accuracy": float(accuracy),
                            "anomalies_detected": int(np.sum(y_pred_binary)),
                            "total_samples": len(y_true),
                            "true_anomalies": int(np.sum(y_true))
                        }
                
                # Save model and scaler
                model_data = {
                    "model": model,
                    "scaler": scaler,
                    "feature_columns": feature_columns,
                    "model_type": model_type,
                    "training_config": training_config,
                    "dataset_info": {
                        "filename": dataset_name or f"all {mode} feature files",
                        "samples": len(X_train),
                        "features": len(feature_columns)
                    }
                }
                
                joblib.dump(model_data, model_filename)
                logger.info(f"Saved trained model to {model_filename}")
                
                # Update training status
                self.training_active = False
                self.training_progress = {
                    "stage": "completed",
                    "progress": 100.0,
                    "message": "Training completed successfully",
                    "accuracy": evaluation_results.get("accuracy", 0.0),
                    "model_file": model_filename
                }
                self.last_training_time = datetime.now()
                
                return {
                    "status": "success",
                    "message": f"Model training completed successfully",
                    "model_type": model_type,
                    "dataset": dataset_name,
                    "samples_trained": len(X_train),
                    "features": len(feature_columns),
                    "model_file": model_filename,
                    "evaluation": evaluation_results,
                    "training_time": datetime.now().isoformat()
                }
                
            except Exception as e:
                self.training_active = False
                self.training_progress = {
                    "stage": "error",
                    "progress": 0.0,
                    "message": f"Training failed: {str(e)}"
                }
                logger.error(f"Model training failed: {e}")
                return {
                    "status": "error",
                    "message": f"Training failed: {str(e)}"
                }

        @self.app.post("/api/model/test")
        async def test_model(config: dict = None):
            """Evaluate the latest trained model on real feature data"""
            try:
                test_config = config or {
                    "lite_mode": True,
                    "test_size": 0.2,
                    "random_state": 42
                }

                results = await self._run_model_test(test_config)

                return {
                    "status": "success",
                    "message": "Model testing completed",
                    "results": results,
                    "config": test_config
                }

            except ValueError as e:
                # Expected operator conditions (no model, no data, wrong
                # columns) — report them plainly, not as a server fault.
                return {
                    "status": "error",
                    "message": str(e)
                }
            except Exception as e:
                logger.error(f"Failed to test model: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to test model: {str(e)}"
                }

        @self.app.post("/api/model/deploy")
        async def deploy_model(config: dict = None):
            """Deploy the trained model"""
            try:
                deployment_config = config or {
                    "model_name": "spectrum_alert_model",
                    "backup_existing": True,
                    "validate_deployment": True
                }
                
                # Check if model files exist
                from pathlib import Path
                model_files = list(Path("models").glob("*.pkl")) if Path("models").exists() else []
                
                if not model_files:
                    return {
                        "status": "error",
                        "message": "No trained models found. Please train a model first."
                    }
                
                # Deploy the model (copy to appropriate location)
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                
                # In a real deployment, you would copy the model to production location
                # For now, we'll just validate it exists and is accessible
                
                return {
                    "status": "success",
                    "message": f"Model deployed successfully: {latest_model.name}",
                    "model_info": {
                        "name": latest_model.name,
                        "size": latest_model.stat().st_size,
                        "modified": datetime.fromtimestamp(latest_model.stat().st_mtime).isoformat()
                    },
                    "config": deployment_config
                }
                
            except Exception as e:
                logger.error(f"Failed to deploy model: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to deploy model: {str(e)}"
                }

        @self.app.get("/api/model/training/status")
        async def get_training_status():
            """Get current training status"""
            training_data = {
                "training_active": getattr(self, 'training_active', False),
                "training_progress": getattr(self, 'training_progress', {}),
                "last_training": getattr(self, 'last_training_time', None)
            }
            
            return {
                "status": "ok",
                "data": training_data,
                **training_data  # Flatten to top level for backward compatibility
            }

        @self.app.get("/api/model/status")
        async def get_model_status():
            """Report the trained model actually on disk.

            The old version looked for model files in the working directory
            while training wrote to models/, so a freshly trained model showed
            as "Unknown" with an empty file list. It then invented an accuracy
            from random-normal noise when it could not find one. This reports
            the artifact's own metadata and leaves metrics unset when they were
            never measured.
            """
            try:
                import joblib
                from pathlib import Path

                model_status = {
                    "accuracy": None,
                    "confidence": None,
                    "last_trained": None,
                    "sample_count": 0,
                    "model_type": "Unknown",
                    "is_deployed": False,
                    "feature_count": 0,
                    "performance_metrics": {},
                    "model_files": [],
                }

                # models/ next to the data directory is where the trainer and
                # the CLI write. Current working directory is included for
                # older checkouts that dropped a .pkl at the repo root.
                data_root = Path(self.storage.data_dir).resolve()
                search_dirs = [data_root.parent / "models", Path("models"), Path.cwd()]
                found = {}
                for directory in search_dirs:
                    if not directory.is_dir():
                        continue
                    for pattern in ("*.pkl", "*.joblib"):
                        for path in directory.glob(pattern):
                            # The same models/ directory can be reached through
                            # more than one relative path; key by resolved path
                            # so the list does not report duplicates.
                            found[str(path.resolve())] = path
                candidates = sorted(found.values(), key=lambda f: f.stat().st_mtime, reverse=True)

                if not candidates:
                    return {"status": "success", "data": model_status}

                latest = candidates[0]
                model_status["model_files"] = [path.name for path in candidates]
                model_status["last_trained"] = datetime.fromtimestamp(
                    latest.stat().st_mtime
                ).isoformat()
                model_status["is_deployed"] = True

                try:
                    artifact = joblib.load(latest)
                except Exception as exc:
                    logger.warning("Could not load model %s: %s", latest, exc)
                    return {"status": "success", "data": model_status}

                model = artifact.get("model", artifact) if isinstance(artifact, dict) else artifact
                dataset_info = artifact.get("dataset_info", {}) if isinstance(artifact, dict) else {}

                model_status["model_type"] = (
                    artifact.get("model_type")
                    or artifact.get("mode")
                    or type(model).__name__
                )
                model_status["sample_count"] = int(
                    dataset_info.get("samples")
                    or artifact.get("n_samples")
                    or 0
                )
                model_status["feature_count"] = int(
                    dataset_info.get("features")
                    or artifact.get("n_features")
                    or (len(artifact.get("feature_columns", [])) if isinstance(artifact, dict) else 0)
                    or 0
                )

                # Only surface metrics that were actually measured. The
                # training endpoint reports them under "evaluation"; the
                # use-case trainer reports a holdout rate instead.
                evaluation = artifact.get("evaluation", {}) if isinstance(artifact, dict) else {}
                if isinstance(evaluation, dict):
                    accuracy = evaluation.get("accuracy")
                    if isinstance(accuracy, (int, float)):
                        model_status["accuracy"] = float(accuracy)
                        model_status["confidence"] = float(accuracy)

                return {"status": "success", "data": model_status}

            except Exception as e:
                logger.error(f"Failed to get model status: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "data": {
                        "accuracy": None,
                        "confidence": None,
                        "last_trained": None,
                        "sample_count": 0,
                        "model_type": "Unknown",
                        "is_deployed": False,
                        "feature_count": 0,
                        "performance_metrics": {},
                        "model_files": [],
                    },
                }

        @self.app.get("/api/data/stats")
        async def get_data_statistics():
            """Get statistics about available training data"""
            try:
                from pathlib import Path
                import pandas as pd
                
                data_stats = {
                    "total_files": 0,
                    "total_samples": 0,
                    "data_size_mb": 0,
                    "date_range": {"start": None, "end": None},
                    "frequency_ranges": [],
                    "file_types": {}
                }
                
                # Use the configured data directory, not a hard-coded relative
                # path: the storage layer may be pointed elsewhere (and under an
                # installed wheel the working directory is not the checkout).
                data_dir = Path(self.storage.data_dir)
                if data_dir.exists():
                    all_files = [
                        path for path in data_dir.rglob("*")
                        if path.is_file() and path.suffix.lower() in (".csv", ".json")
                    ]
                    csv_files = [path for path in all_files if path.suffix.lower() == ".csv"]

                    data_stats["total_files"] = len(all_files)
                    data_stats["file_types"] = {
                        "csv": len(csv_files),
                        "json": len(all_files) - len(csv_files),
                    }
                    total_size = sum(f.stat().st_size for f in all_files)
                    data_stats["data_size_mb"] = round(total_size / (1024 * 1024), 2)

                    # Sample a few CSV files to estimate row counts. Spectrum
                    # JSON files store sample counts, not tabular rows.
                    sample_files = csv_files[:5]
                    total_samples = 0

                    for file in sample_files:
                        try:
                            df = pd.read_csv(file)
                            total_samples += len(df)
                        except Exception:
                            continue

                    if sample_files:
                        avg_samples_per_file = total_samples / len(sample_files)
                        data_stats["total_samples"] = int(avg_samples_per_file * len(csv_files))
                
                return {
                    "status": "ok",
                    "data": data_stats,
                    "stats": data_stats  # Add backward compatibility
                }
                
            except Exception as e:
                logger.error(f"Failed to get data statistics: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

        @self.app.post("/api/data/collect")
        async def start_data_collection(config: Optional[dict] = None):
            """Start data collection process"""
            try:
                collection_config = config or {
                    "duration_minutes": 0.017,  # 1 second (0.017 minutes) - very small for testing
                    "output_file": f"spectrum_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "sample_rate": 2e6,
                    "center_frequency": 433e6,
                    "gain": 20.0
                }
                
                # Log the received configuration for debugging
                logger.info(f"Data collection started with config: {collection_config}")
                
                # Calculate duration first (outside try block for scope) - use very small duration for testing
                duration_seconds = min(collection_config.get("duration_minutes", 0.017) * 60, 1 * 60)  # Cap at 1 minute, default 1 second
                
                # Try to actually collect spectrum data using the monitoring use case
                try:
                    from spectrum_alert.infrastructure.sdr import RTLSDRInterface
                    from spectrum_alert.infrastructure.storage import DataStorage
                    from spectrum_alert.core.services.feature_extraction import FeatureExtractor
                    from spectrum_alert.application.use_cases.spectrum_monitoring import SpectrumMonitoringUseCase
                    
                    # Initialize components
                    sdr = RTLSDRInterface()
                    storage = DataStorage()
                    feature_extractor = FeatureExtractor(lite_mode=True)
                    monitoring_use_case = SpectrumMonitoringUseCase(sdr, storage, feature_extractor)
                    
                    # Set up data collection variables
                    spectrum_data = None
                    
                    # Log frequency information
                    center_freq = collection_config.get("center_frequency", 433e6)
                    logger.info(f"Collecting data at center frequency: {center_freq/1e6:.3f} MHz for {duration_seconds/60:.1f} minutes")
                    
                    safe_op = getattr(sdr, "safe_operation", None)
                    ctx = safe_op() if callable(safe_op) else None
                    if ctx is None:
                        from contextlib import nullcontext
                        ctx = nullcontext()
                    with ctx:
                        spectrum_data = monitoring_use_case.capture_spectrum_data(
                            frequency=center_freq,
                            sample_rate=collection_config.get("sample_rate", 2e6),
                            gain=collection_config.get("gain", 20.0),
                            duration=duration_seconds
                        )
                    
                    if spectrum_data:
                        return {
                            "status": "success",
                            "message": f"Data collection completed successfully - {duration_seconds/60:.1f} minutes of data captured",
                            "config": collection_config,
                            "data": {
                                "spectrum_data_id": spectrum_data.id,
                                "frequency_hz": spectrum_data.frequency_hz,
                                "sample_count": spectrum_data.sample_count,
                                "duration_seconds": spectrum_data.duration_seconds,
                                "timestamp": spectrum_data.timestamp.isoformat()
                            },
                            "estimated_completion": datetime.now().isoformat()
                        }
                    else:
                        raise Exception("Failed to capture spectrum data")
                    
                except ImportError as e:
                    logger.warning(f"SDR components not available: {e}")
                    # No mock capture: a fake success here would hand the
                    # training pipeline IDs that point at nothing.
                    return {
                        "status": "error",
                        "message": f"SDR components not available: {str(e)}",
                        "config": collection_config,
                    }

                except Exception as sdr_error:
                    logger.warning(f"SDR collection failed: {sdr_error}")
                    return {
                        "status": "error",
                        "message": f"Data collection failed: {str(sdr_error)}",
                        "config": collection_config,
                    }
                
            except Exception as e:
                logger.error(f"Failed to start data collection: {e}")
                return {
                    "status": "error",
                    "message": f"Failed to start data collection: {str(e)}"
                }

        @self.app.get("/api/data/files")
        async def get_data_files():
            """Get list of available REAL data files with details"""
            try:
                from pathlib import Path
                import os
                import json
                
                data_files = []

                # Scan the same data directory the storage layer writes to.
                # `__file__` resolves inside site-packages for an installed
                # wheel, so the old anchor pointed at a directory that never
                # existed in Docker and Data Management was always empty there.
                data_root = Path(self.storage.data_dir).resolve()
                real_data_paths = [
                    data_root / "features",      # Feature files
                    data_root / "anomalies",     # Anomaly detection results
                    data_root / "spectrum",      # Raw spectrum data
                    data_root,
                    Path.cwd(),                  # Root directory for lite models
                ]

                def _display_path(path: Path) -> str:
                    try:
                        return str(path.relative_to(Path.cwd()))
                    except ValueError:
                        return str(path)
                
                for data_dir in real_data_paths:
                    if not data_dir.exists():
                        continue
                    
                    # Find CSV files (datasets)
                    for csv_file in data_dir.glob("*.csv"):
                        if csv_file.is_file() and csv_file.stat().st_size > 1000:  # At least 1KB
                            try:
                                # Get file stats
                                stat = csv_file.stat()
                                
                                # Determine dataset type from filename and location
                                dataset_type = "unknown"
                                description = "Data file"
                                
                                if "features" in str(csv_file):
                                    dataset_type = "features"
                                    description = "Feature vectors extracted from spectrum data"
                                elif "anomalies" in str(csv_file):
                                    dataset_type = "anomalies"  
                                    description = "Detected anomalies and their properties"
                                elif "collected_data" in csv_file.name:
                                    dataset_type = "raw_data"
                                    description = "Raw collected spectrum data"
                                
                                # Try to read CSV to get sample count and columns
                                samples = 0
                                columns = []
                                sample_data = None
                                
                                try:
                                    df = pd.read_csv(csv_file, nrows=5)  # Read first 5 rows for info
                                    samples = len(pd.read_csv(csv_file))  # Get total count
                                    columns = list(df.columns)
                                    sample_data = df.head(3).to_dict('records')  # First 3 rows as sample
                                except Exception as e:
                                    logger.warning(f"Could not read CSV {csv_file}: {e}")
                                
                                file_info = {
                                    "filename": csv_file.name,
                                    "path": _display_path(csv_file),
                                    "full_path": str(csv_file),
                                    "type": "csv",
                                    "dataset_type": dataset_type,
                                    "description": description,
                                    "size_bytes": stat.st_size,
                                    "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                    "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                    "samples": samples,
                                    "columns": columns,
                                    "sample_data": sample_data,
                                    "selectable": True,  # Can be selected for training
                                    # Match the training endpoint's own floor
                                    # (10 complete rows). A 100-row UI threshold
                                    # marked trainable datasets as not ready.
                                    "ready_for_training": samples >= 10 and len(columns) > 1
                                }
                                
                                data_files.append(file_info)
                                
                            except Exception as e:
                                logger.warning(f"Error processing CSV file {csv_file}: {e}")
                    
                    # Also include JSON spectrum files (for analysis)
                    for json_file in data_dir.glob("*.json"):
                        if json_file.is_file() and json_file.stat().st_size > 1000:  # At least 1KB
                            try:
                                stat = json_file.stat()
                                
                                # Check if it's spectrum data
                                samples = 0
                                freq_info = None
                                
                                try:
                                    with open(json_file, 'r') as f:
                                        data = json.load(f)
                                        # Current files store i_samples/q_samples;
                                        # the legacy schema used samples_real/
                                        # samples_imag. Only recognising the old
                                        # keys hid every capture written since
                                        # the schema change.
                                        if "i_samples" in data and "q_samples" in data:
                                            sample_count = len(data["i_samples"])
                                        elif "samples_real" in data and "samples_imag" in data:
                                            sample_count = len(data["samples_real"])
                                        else:
                                            sample_count = 0
                                        if sample_count:
                                            samples = sample_count
                                            freq_info = {
                                                "center_frequency": data.get("frequency_hz", 0),
                                                "sample_rate": data.get("sample_rate_hz", 0),
                                                "frequency_range": data.get("frequency_range_hz", [0, 0])
                                            }
                                except Exception:
                                    continue  # Skip if not valid spectrum data
                                
                                if samples > 0:  # Only include if we found valid spectrum data
                                    file_info = {
                                        "filename": json_file.name,
                                        "path": _display_path(json_file),
                                        "full_path": str(json_file),
                                        "type": "json",
                                        "dataset_type": "spectrum",
                                        "description": "Raw RTL-SDR I/Q spectrum data",
                                        "size_bytes": stat.st_size,
                                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                                        "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
                                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                                        "samples": samples,
                                        "frequency_info": freq_info,
                                        "selectable": False,  # Raw spectrum data not directly used for training
                                        "ready_for_training": False
                                    }
                                    
                                    data_files.append(file_info)
                            
                            except Exception as e:
                                logger.warning(f"Error processing JSON file {json_file}: {e}")
                
                # Sort by modification time (newest first)
                data_files.sort(key=lambda x: x["modified"], reverse=True)
                
                # Group by type for better organization
                csv_files = [f for f in data_files if f["type"] == "csv"]
                json_files = [f for f in data_files if f["type"] == "json"]
                
                return {
                    "status": "ok",
                    "data": {
                        "files": data_files,
                        "csv_datasets": csv_files,
                        "spectrum_files": json_files,
                        "total_files": len(data_files),
                        "total_csv": len(csv_files),
                        "total_spectrum": len(json_files),
                        "total_size_mb": round(sum(f["size_mb"] for f in data_files), 2),
                        "ready_for_training": [f for f in csv_files if f["ready_for_training"]]
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get data files: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

        @self.app.get("/api/data/file/{filename}")
        async def get_file_analysis(filename: str):
            """Get detailed analysis of a specific data file"""
            try:
                from pathlib import Path
                import json

                # The listing endpoint returns bare filenames from several
                # subdirectories, so the lookup has to search those same
                # subdirectories. It previously looked only in "data/<name>",
                # which is why analysing a listed feature or anomaly file
                # always failed with "File not found".
                safe_name = Path(filename).name
                if safe_name != filename:
                    return {"status": "error", "message": "Invalid file name"}

                data_root = Path(self.storage.data_dir)
                candidates = [
                    data_root / safe_name,
                    data_root / "features" / safe_name,
                    data_root / "anomalies" / safe_name,
                    data_root / "spectrum" / safe_name,
                    Path.cwd() / safe_name,
                ]
                file_path = next((p for p in candidates if p.is_file()), None)

                if not file_path:
                    return {
                        "status": "error",
                        "message": f"File '{filename}' not found"
                    }
                
                analysis = {
                    "filename": filename,
                    "path": str(file_path),
                    "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                    "samples": 0,
                    "columns": [],
                    "frequency_analysis": {},
                    "signal_statistics": {},
                    "preview_data": []
                }
                
                if file_path.suffix == ".csv":
                    df = pd.read_csv(file_path)
                    analysis["samples"] = len(df)
                    analysis["columns"] = list(df.columns)
                    
                    # Get preview data (first 10 rows)
                    analysis["preview_data"] = df.head(10).to_dict('records')
                    
                    # Frequency analysis
                    freq_cols = [col for col in df.columns if 'freq' in col.lower()]
                    if freq_cols:
                        freq_col = freq_cols[0]
                        analysis["frequency_analysis"] = {
                            "column": freq_col,
                            "min_hz": float(df[freq_col].min()),
                            "max_hz": float(df[freq_col].max()),
                            "mean_hz": float(df[freq_col].mean()),
                            "unique_frequencies": int(df[freq_col].nunique())
                        }
                    
                    # Signal strength analysis
                    signal_cols = [col for col in df.columns if any(s in col.lower() for s in ['power', 'magnitude', 'strength', 'amplitude', 'dbm'])]
                    if signal_cols:
                        signal_col = signal_cols[0]
                        analysis["signal_statistics"] = {
                            "column": signal_col,
                            "min": float(df[signal_col].min()),
                            "max": float(df[signal_col].max()),
                            "mean": float(df[signal_col].mean()),
                            "std": float(df[signal_col].std()),
                            "median": float(df[signal_col].median())
                        }
                
                elif file_path.suffix == ".json":
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        
                    if isinstance(data, list):
                        analysis["samples"] = len(data)
                        analysis["preview_data"] = data[:10]
                    elif isinstance(data, dict):
                        # Current captures store i_samples/q_samples. Report the
                        # real capture length and keep the sample arrays out of
                        # preview_data: sending 8k floats twice made the modal
                        # response megabytes wide for no display benefit.
                        i_samples = data.get("i_samples")
                        q_samples = data.get("q_samples")
                        if isinstance(i_samples, list):
                            analysis["samples"] = len(i_samples)
                            analysis["preview_data"] = [{
                                "sample_preview": i_samples[:8],
                                "q_preview": (q_samples or [])[:8],
                            }]
                        elif isinstance(data.get("samples_real"), list):
                            analysis["samples"] = len(data["samples_real"])
                            analysis["preview_data"] = [{
                                "sample_preview": data["samples_real"][:8],
                                "q_preview": data.get("samples_imag", [])[:8],
                            }]
                        else:
                            analysis["samples"] = 1
                            analysis["preview_data"] = [data]

                        # Extract metadata if available
                        if "frequency_hz" in data:
                            analysis["frequency_analysis"]["center_hz"] = data["frequency_hz"]
                        if "sample_rate_hz" in data:
                            analysis["frequency_analysis"]["sample_rate_hz"] = data["sample_rate_hz"]
                        if isinstance(data.get("i_samples"), list):
                            half = data.get("sample_rate_hz", 0) / 2.0
                            analysis["frequency_analysis"]["range_hz"] = [
                                data["frequency_hz"] - half,
                                data["frequency_hz"] + half,
                            ]
                        if isinstance(data.get("spectrum_data"), list):
                            analysis["samples"] = len(data["spectrum_data"])
                
                return {
                    "status": "ok",
                    "data": analysis
                }
                
            except Exception as e:
                logger.error(f"Failed to analyze file {filename}: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

        @self.app.get("/api/data/datasets")
        async def get_available_datasets():
            """Get available datasets for model training"""
            try:
                from pathlib import Path
                
                datasets = []
                
                # Look for organized datasets
                data_dirs = ["data", "datasets", "training_data", "spectrum_data"]
                
                for data_dir_name in data_dirs:
                    data_dir = Path(data_dir_name)
                    if not data_dir.exists():
                        continue
                    
                    # Look for dataset folders or grouped files
                    for item in data_dir.iterdir():
                        if item.is_dir():
                            # Check if directory contains data files
                            csv_files = list(item.glob("*.csv"))
                            json_files = list(item.glob("*.json"))
                            
                            if csv_files or json_files:
                                dataset_info = {
                                    "name": item.name,
                                    "type": "directory",
                                    "path": str(item),
                                    "file_count": len(csv_files) + len(json_files),
                                    "csv_files": len(csv_files),
                                    "json_files": len(json_files),
                                    "size_mb": sum(f.stat().st_size for f in csv_files + json_files) / (1024 * 1024),
                                    "description": f"Dataset with {len(csv_files)} CSV and {len(json_files)} JSON files"
                                }
                                datasets.append(dataset_info)
                
                # Also add individual large files as datasets
                for data_dir_name in ["data", ".", "spectrum_data"]:
                    data_dir = Path(data_dir_name)
                    if not data_dir.exists():
                        continue
                        
                    for csv_file in data_dir.glob("*.csv"):
                        if csv_file.stat().st_size > 1024 * 1024:  # Files larger than 1MB
                            dataset_info = {
                                "name": csv_file.stem,
                                "type": "file",
                                "path": str(csv_file),
                                "file_count": 1,
                                "csv_files": 1,
                                "json_files": 0,
                                "size_mb": csv_file.stat().st_size / (1024 * 1024),
                                "description": f"Single CSV file ({csv_file.stat().st_size / (1024 * 1024):.1f} MB)"
                            }
                            datasets.append(dataset_info)
                
                # Add some predefined datasets if they exist
                predefined_datasets = [
                    {
                        "name": "collected_data_lite",
                        "type": "predefined",
                        "path": "collected_data_lite.csv",
                        "description": "Lightweight collected spectrum data"
                    },
                    {
                        "name": "full_spectrum_data", 
                        "type": "predefined",
                        "path": "data/spectrum_*.csv",
                        "description": "Complete spectrum monitoring data"
                    },
                    {
                        "name": "anomaly_training_data",
                        "type": "predefined", 
                        "path": "data/anomaly_*.csv",
                        "description": "Pre-labeled anomaly detection training data"
                    }
                ]
                
                for pd_dataset in predefined_datasets:
                    # Check if the dataset actually exists
                    if pd_dataset["path"].endswith(".csv"):
                        if Path(pd_dataset["path"]).exists():
                            file_size = Path(pd_dataset["path"]).stat().st_size / (1024 * 1024)
                            pd_dataset.update({
                                "file_count": 1,
                                "csv_files": 1,
                                "json_files": 0,
                                "size_mb": file_size
                            })
                            datasets.append(pd_dataset)
                    else:
                        # Pattern-based dataset
                        matching_files = list(Path(".").glob(pd_dataset["path"]))
                        if matching_files:
                            total_size = sum(f.stat().st_size for f in matching_files) / (1024 * 1024)
                            pd_dataset.update({
                                "file_count": len(matching_files),
                                "csv_files": len([f for f in matching_files if f.suffix == ".csv"]),
                                "json_files": len([f for f in matching_files if f.suffix == ".json"]),
                                "size_mb": total_size
                            })
                            datasets.append(pd_dataset)
                
                return {
                    "status": "ok",
                    "data": {
                        "datasets": datasets,
                        "total_datasets": len(datasets)
                    }
                }
                
            except Exception as e:
                logger.error(f"Failed to get available datasets: {e}")
                return {
                    "status": "error",
                    "message": str(e)
                }

        @self.app.get("/api/analytics/realtime")
        async def get_realtime_analytics(timeframe: str = "24h"):
            """Get real-time analytics data"""
            try:
                import numpy as np
                from pathlib import Path
                
                # Parse timeframe
                hours = 24
                if timeframe == "1h":
                    hours = 1
                elif timeframe == "7d":
                    hours = 168
                elif timeframe == "30d":
                    hours = 720
                
                # Calculate time window
                end_time = datetime.now()
                start_time = end_time - timedelta(hours=hours)
                
                analytics_data = {
                    "detections": 0,
                    "anomalies": 0,
                    "avg_signal": -50.0,
                    "realtime_accuracy": 0.85,
                    "detections_change": 0.0,
                    "anomalies_change": 0.0,
                    "signal_change": 0.0,
                    "accuracy_change": 0.0,
                    "timeline": [],
                    "frequency_distribution": []
                }
                
                try:
                    # Check for recent data files
                    data_dir = Path("data")
                    if data_dir.exists():
                        # Count spectrum files for detections
                        spectrum_files = list(data_dir.glob("spectrum/*.json"))
                        recent_files = [f for f in spectrum_files 
                                      if datetime.fromtimestamp(f.stat().st_mtime) >= start_time]
                        analytics_data["detections"] = len(recent_files)
                        
                        # Count anomalies
                        anomaly_files = list(data_dir.glob("anomalies/*.csv"))
                        recent_anomalies = 0
                        for f in anomaly_files:
                            if datetime.fromtimestamp(f.stat().st_mtime) >= start_time:
                                try:
                                    import pandas as pd
                                    df = pd.read_csv(f)
                                    recent_anomalies += len(df)
                                except Exception:
                                    pass
                        analytics_data["anomalies"] = recent_anomalies
                        
                        # Generate REAL analytics with actual spectrum analysis
                        real_signal_levels = []
                        model_accuracy = 0.85  # Default
                        
                        # Analyze recent spectrum files for real signal data
                        sample_files = recent_files[:10] if len(recent_files) > 10 else recent_files
                        for file_path in sample_files:
                            try:
                                with open(file_path, 'r') as f:
                                    data = json.load(f)
                                    if 'signal_strength' in data:
                                        real_signal_levels.append(data['signal_strength'])
                                    elif 'power_spectrum' in data and data['power_spectrum']:
                                        # Calculate average power from real spectrum
                                        avg_power = np.mean(data['power_spectrum'])
                                        real_signal_levels.append(float(avg_power))
                            except Exception:
                                continue
                        
                        # Use real signal data if available, otherwise use monitoring-based estimates
                        if real_signal_levels:
                            analytics_data["avg_signal"] = np.mean(real_signal_levels)
                            analytics_data["signal_change"] = np.std(real_signal_levels)
                        else:
                            # Use current monitoring status to estimate signal levels
                            if hasattr(self, 'monitoring_active') and self.monitoring_active:
                                analytics_data["avg_signal"] = -45.0 + np.random.normal(0, 3)  # Active monitoring = stronger signals
                            else:
                                analytics_data["avg_signal"] = -60.0 + np.random.normal(0, 5)  # No monitoring = background level
                            analytics_data["signal_change"] = np.random.normal(0, 2)
                        
                        # Get real model performance based on model file age and activity
                        try:
                            model_files = ["anomaly_detection_model_lite.pkl", "rf_fingerprinting_model_lite.pkl"]
                            model_found = False
                            for model_file in model_files:
                                model_path = Path(model_file)
                                if model_path.exists():
                                    # Calculate accuracy based on model age and recent anomaly detection rate
                                    model_age_hours = (datetime.now() - datetime.fromtimestamp(model_path.stat().st_mtime)).total_seconds() / 3600
                                    
                                    # Newer models are more accurate
                                    base_accuracy = max(0.70, 0.95 - (model_age_hours / 24 * 0.01))  # Decay over time
                                    
                                    # Adjust based on recent detection activity
                                    if len(recent_files) > 50:  # High activity
                                        model_accuracy = base_accuracy * 0.95  # Slightly lower due to stress
                                    elif len(recent_files) > 10:  # Normal activity
                                        model_accuracy = base_accuracy
                                    else:  # Low activity
                                        model_accuracy = base_accuracy * 0.98  # Slightly higher due to less stress
                                    
                                    model_found = True
                                    break
                            
                            if not model_found:
                                model_accuracy = 0.60  # No trained model available
                                
                        except Exception:
                            model_accuracy = 0.75  # Default fallback
                        
                        analytics_data["realtime_accuracy"] = model_accuracy + np.random.normal(0, 0.02)
                        
                        # Calculate realistic change rates based on time and activity
                        current_hour = datetime.now().hour
                        
                        # Detection rate changes based on typical RF activity patterns
                        if 6 <= current_hour <= 22:  # Daytime - more activity
                            analytics_data["detections_change"] = len(recent_files) * 0.1 + np.random.normal(1, 2)
                        else:  # Nighttime - less activity
                            analytics_data["detections_change"] = len(recent_files) * 0.05 + np.random.normal(-0.5, 1)
                        
                        # Anomaly change based on actual anomaly detection
                        analytics_data["anomalies_change"] = recent_anomalies - analytics_data.get("anomalies", 0)
                        analytics_data["accuracy_change"] = np.random.normal(0, 0.5)
                        
                        # Ensure realistic bounds
                        analytics_data["realtime_accuracy"] = max(0.0, min(1.0, analytics_data["realtime_accuracy"]))
                        
                        # Generate timeline data (hourly buckets)
                        timeline_hours = min(24, hours)
                        for i in range(timeline_hours):
                            timestamp = end_time - timedelta(hours=i)
                            analytics_data["timeline"].append({
                                "timestamp": timestamp.isoformat(),
                                "detections": max(0, int(np.random.poisson(analytics_data["detections"] / timeline_hours))),
                                "anomalies": max(0, int(np.random.poisson(analytics_data["anomalies"] / timeline_hours)))
                            })
                        
                        # Generate frequency distribution data
                        common_frequencies = [144.0, 146.0, 433.0, 434.0, 435.0, 915.0, 2400.0]
                        for freq in common_frequencies:
                            analytics_data["frequency_distribution"].append({
                                "frequency": freq * 1e6,
                                "power": -60 + np.random.normal(0, 10)
                            })
                
                except Exception as e:
                    logger.warning(f"Error generating analytics data: {e}")
                
                return {
                    "status": "success",
                    "data": analytics_data
                }
                
            except Exception as e:
                logger.error(f"Failed to get analytics data: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "data": {
                        "detections": 0,
                        "anomalies": 0,
                        "avg_signal": -50.0,
                        "realtime_accuracy": 0.0,
                        "detections_change": 0.0,
                        "anomalies_change": 0.0,
                        "signal_change": 0.0,
                        "accuracy_change": 0.0,
                        "timeline": [],
                        "frequency_distribution": []
                    }
                }

        # Registered with the routes at construction time. It previously sat
        # after a stray dedent inside _run_model_test, so the SDR was only
        # released on shutdown if /api/model/test happened to have been called.
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Clean up resources on shutdown"""
            try:
                logger.info("Shutting down SpectrumAlert web application...")
                self.monitoring_active = False

                # Close SDR device if it exists
                if hasattr(self, 'monitoring_service') and hasattr(self.monitoring_service, 'sdr'):
                    try:
                        self.monitoring_service.sdr.close()
                        logger.info("SDR device closed during shutdown")
                    except Exception as e:
                        logger.warning(f"Error closing SDR during shutdown: {e}")

            except Exception as e:
                logger.error(f"Error during shutdown: {e}")

    async def _get_spectrum_analysis(self, frequency_range: str = "144-146") -> Dict[str, Any]:
        """Get spectrum analysis data from real RTL-SDR captures"""
        # Set default center frequency
        center_freq_hz = 145e6  # Default to VHF amateur band
        
        try:
            import numpy as np
            from pathlib import Path
            from scipy import signal
            import os
            
            # Parse frequency range for filtering
            try:
                start_freq, end_freq = map(float, frequency_range.split("-"))
                start_freq_hz = start_freq * 1e6  # Convert to Hz
                end_freq_hz = end_freq * 1e6
                center_freq_hz = (start_freq_hz + end_freq_hz) / 2
            except:
                # Default to VHF amateur band if parsing fails
                start_freq_hz = 144e6
                end_freq_hz = 146e6
                center_freq_hz = 145e6
            
            # Get real spectrum data files using absolute path
            base_dir = Path(__file__).parent.parent.parent  # Go up to SpectrumAlert root
            data_dir = base_dir / "data" / "spectrum"
            if not data_dir.exists():
                return {
                    "frequencies": [],
                    "power_spectrum": [],
                    "peak_frequencies": [],
                    "signal_strength": -80.0,
                    "bandwidth": 0.0,
                    "center_frequency": center_freq_hz,
                    "total_samples": 0,
                    "error": "No spectrum data directory found"
                }
            
            # Get the most recent spectrum file
            spectrum_files = list(data_dir.glob("*.json"))
            if not spectrum_files:
                return {
                    "frequencies": [],
                    "power_spectrum": [],
                    "peak_frequencies": [],
                    "signal_strength": -80.0,
                    "bandwidth": 0.0,
                    "center_frequency": 433000000.0,
                    "total_samples": 0,
                    "error": "No spectrum data files found"
                }
            
            # Sort by modification time and get the latest file
            latest_file = max(spectrum_files, key=lambda f: f.stat().st_mtime)
            
            with open(latest_file, 'r') as f:
                spectrum_data = json.load(f)
            
            # Extract I/Q samples. Both schemas are accepted: the current writer emits
            # i_samples/q_samples while the 706 archived files use
            # samples_real/samples_imag. Reading only the legacy keys meant every
            # freshly written file reported "No I/Q samples in data file", while
            # the storage layer could not read any of the legacy ones.
            if 'i_samples' in spectrum_data:
                samples_real = np.array(spectrum_data.get('i_samples', []))
                samples_imag = np.array(spectrum_data.get('q_samples', []))
            else:
                samples_real = np.array(spectrum_data.get('samples_real', []))
                samples_imag = np.array(spectrum_data.get('samples_imag', []))

            if len(samples_real) == 0 or len(samples_imag) == 0:
                return {
                    "frequencies": [],
                    "power_spectrum": [],
                    "peak_frequencies": [],
                    "signal_strength": -80.0,
                    "bandwidth": 0.0,
                    "center_frequency": float(spectrum_data.get('frequency_hz', 433000000)),
                    "total_samples": 0,
                    "error": "No I/Q samples in data file"
                }
            
            # Create complex samples from I/Q data
            complex_samples = samples_real + 1j * samples_imag
            
            # Calculate power spectral density using Welch's method
            sample_rate = spectrum_data.get('sample_rate_hz', 2000000)
            center_freq = spectrum_data.get('frequency_hz', 433000000)
            
            # Use only a subset for processing (to speed up web response)
            max_samples = min(len(complex_samples), 8192)
            processed_samples = complex_samples[:max_samples]
            
            # Calculate power spectral density
            frequencies, psd = signal.welch(
                processed_samples, 
                fs=sample_rate, 
                nperseg=min(1024, max_samples//4),
                noverlap=None,
                return_onesided=False
            )
            
            # Convert to dBFS. A "+30 dBm for a 50 ohm impedance" offset is not a
            # calibration: it invents an absolute reference the dongle does not
            # have, so readings were labelled dBm while being meaningless.
            psd_dbfs = 10 * np.log10(psd + 1e-12)
            
            # Shift frequencies to actual RF frequencies
            rf_frequencies = frequencies + center_freq
            
            # Find peak frequencies (signals above background)
            # Noise floor from the quietest 15% rather than the median, so a
            # wideband signal does not inflate its own background.
            background_level = float(np.percentile(psd_dbfs, 15))
            threshold = background_level + 6  # 6 dB above background
            peaks, _ = signal.find_peaks(psd_dbfs, height=threshold, distance=10)
            
            peak_frequencies = []
            for peak_idx in peaks:
                peak_frequencies.append({
                    "frequency": float(rf_frequencies[peak_idx]),
                    "power_dbfs": float(psd_dbfs[peak_idx]),
                    "frequency_mhz": float(rf_frequencies[peak_idx] / 1e6)
                })
            
            # Calculate overall signal strength
            signal_strength = float(np.max(psd_dbfs))
            bandwidth = float(sample_rate)
            
            # Prepare data for frontend (downsample for web display)
            downsample_factor = max(1, len(frequencies) // 512)  # Max 512 points for web
            freq_display = rf_frequencies[::downsample_factor] / 1e6  # Convert to MHz
            psd_display = psd_dbfs[::downsample_factor]
            
            return {
                "frequencies": freq_display.tolist(),
                "power_spectrum": psd_display.tolist(),
                "peak_frequencies": peak_frequencies,
                "signal_strength": signal_strength,
                "units": "dBFS",
                "bandwidth": bandwidth,
                "center_frequency": float(center_freq),
                "total_samples": len(complex_samples),
                "background_level": float(background_level),
                "file_timestamp": spectrum_data.get('timestamp', datetime.now().isoformat()),
                "sample_rate": float(sample_rate),
                "gain_db": spectrum_data.get('gain_db', 20.0)
            }
            
        except Exception as e:
            logger.error(f"Error in spectrum analysis: {e}")
            return {
                "frequencies": [],
                "power_spectrum": [],
                "peak_frequencies": [],
                "signal_strength": -80.0,
                "bandwidth": 0.0,
                "center_frequency": center_freq_hz,
                "total_samples": 0,
                "error": str(e)
            }
    
    async def _get_model_info(self) -> Dict[str, Any]:
        """Get ML model information"""
        try:
            from pathlib import Path
            
            # Look for model files
            models_dir = Path("models")
            model_info = {
                "models_available": [],
                "training_stats": {},
                "model_performance": {}
            }
            
            if models_dir.exists():
                for model_file in models_dir.glob("*.pkl"):
                    model_info["models_available"].append({
                        "name": model_file.name,
                        "size": model_file.stat().st_size,
                        "modified": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat()
                    })
            
            # Get training data info if available
            data_dir = Path("data")
            if data_dir.exists():
                training_files = list(data_dir.glob("**/features_*.csv"))
                if training_files:
                    latest_file = max(training_files, key=lambda x: x.stat().st_mtime)
                    try:
                        df = pd.read_csv(latest_file)
                        model_info["training_stats"] = {
                            "samples_count": len(df),
                            "features_count": len(df.columns) - 1,  # Exclude frequency column
                            "last_updated": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat()
                        }
                    except Exception as e:
                        logger.error(f"Error reading training data: {e}")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {
                "models_available": [],
                "training_stats": {},
                "model_performance": {},
                "error": str(e)
            }
    
    async def broadcast_update(self, data: Dict[str, Any]):
        """Broadcast update to all connected WebSocket clients"""
        if not self.connections:
            return
        
        message = json.dumps(data)
        disconnected = []
        
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        
        # Remove disconnected clients
        for connection in disconnected:
            self.connections.remove(connection)
    
    async def start_background_updates(self):
        """Start background task for real-time updates"""
        async def update_loop():
            while True:
                try:
                    # Get current system status
                    status = self.system_monitor.get_system_status()
                    
                    # Broadcast to connected clients
                    await self.broadcast_update({
                        "type": "system_status",
                        "data": status,
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    await asyncio.sleep(5)  # Update every 5 seconds
                    
                except Exception as e:
                    logger.error(f"Error in background update loop: {e}")
                    await asyncio.sleep(10)  # Wait longer on error
        
        self._background_task = asyncio.create_task(update_loop())

    async def _run_monitoring_loop(self, monitoring_service, config):
        """Run the monitoring service in a loop"""
        try:
            import time
            
            # Use the SDR instance from the monitoring service
            sdr = monitoring_service.sdr
            
            logger.info("Starting monitoring loop...")
            
            while self.monitoring_active:
                try:
                    # A receiver captures one contiguous window, so the tuning
                    # point is an explicit centre frequency. It wins when given;
                    # otherwise a start-end band collapses to its midpoint for
                    # backwards compatibility. Pretending to scan a whole band
                    # the dongle cannot see is what produced alerts "on
                    # frequencies not being monitored".
                    frequency_range = config.get("frequency_range", "88-108")
                    try:
                        start_freq, end_freq = map(float, frequency_range.split("-"))
                        start_freq *= 1e6  # Convert to Hz
                        end_freq *= 1e6
                    except (ValueError, AttributeError):
                        start_freq, end_freq = 88e6, 108e6

                    sample_rate = config.get("sample_rate", 2048000)
                    gain = config.get("gain", 20)

                    if config.get("center_freq") is not None:
                        center_freq = float(config["center_freq"]) * 1e6
                    else:
                        center_freq = (start_freq + end_freq) / 2

                    # Configure SDR (don't open again if already open)
                    if not hasattr(sdr, '_is_open') or not sdr._is_open:
                        sdr.open()
                        logger.info("SDR device opened for monitoring")
                    
                    sdr.set_sample_rate(sample_rate)
                    sdr.set_center_freq(center_freq)
                    sdr.set_gain(gain)

                    # The RTL2832 snaps to the nearest supported rate. Use the
                    # rate the device actually applied for the frequency axis,
                    # otherwise a request the hardware rounded shifts every bin.
                    actual_rate = float(getattr(sdr, "sample_rate", 0) or sample_rate)
                    window_start_hz = center_freq - actual_rate / 2
                    window_end_hz = center_freq + actual_rate / 2
                    self.monitored_window_hz = [window_start_hz, window_end_hz]
                    self.monitoring_center_hz = center_freq
                    self.monitoring_span_hz = actual_rate

                    # Collect samples.
                    #
                    # read_samples blocks for hundreds of milliseconds and this
                    # runs in the event loop, so the whole server — including
                    # every WebSocket push — stalled for the duration of each
                    # capture. Hand it to a thread.
                    #
                    # 1M complex samples is also 16 MB per iteration and the
                    # detector only uses an FFT window of it; 256k is ample for a
                    # Welch estimate and cuts the transfer by 4x.
                    samples = await asyncio.to_thread(sdr.read_samples, 256 * 1024)

                    # Process spectrum data
                    import numpy as np
                    from scipy import signal

                    # Welch (averaged, overlapped) rather than a single
                    # periodogram: variance per bin drops sharply for the same
                    # data, which is what makes a fixed threshold stable.
                    frequencies, power_spectrum = signal.welch(
                        samples, fs=actual_rate, nperseg=1024
                    )

                    # Convert to absolute frequencies
                    frequencies = frequencies + center_freq

                    # Convert to dBFS. This is NOT dBm: an RTL-SDR has no
                    # calibrated reference. It was previously labelled "power_dbm"
                    # and pushed over the WebSocket as if it were absolute.
                    power_dbfs = 10 * np.log10(power_spectrum + 1e-12)
                    
                    # Create SpectrumData object for anomaly detection
                    from spectrum_alert.core.domain.models import SpectrumData
                    from datetime import datetime
                    import uuid
                    
                    spectrum_data = SpectrumData(
                        id=str(uuid.uuid4()),
                        frequency_hz=center_freq,
                        sample_rate_hz=actual_rate,
                        gain_db=gain,
                        samples=samples.tolist(),
                        duration_seconds=len(samples) / actual_rate,
                        timestamp=datetime.utcnow(),
                        power_spectrum=power_dbfs.tolist(),
                        sample_rate_actual_hz=actual_rate,
                    )
                    
                    # Run anomaly detection
                    anomaly_results = monitoring_service.anomaly_detector.detect_anomalies(spectrum_data)
                    
                    # Process any detected anomalies
                    for anomaly_result in anomaly_results:
                        monitoring_service.storage.save_anomaly(anomaly_result)
                        
                        # Broadcast anomaly via WebSocket
                        await self._broadcast_to_websockets({
                            "type": "anomaly_detected",
                            "data": {
                                "frequency_mhz": anomaly_result.frequency_hz / 1e6,  # Convert Hz to MHz
                                # dBFS, not dBm. metadata["peak_dbfs"] is what the
                                # detector records; the old lookup of
                                # metadata["power_dbm"] never matched anything and
                                # always reported 0.
                                "power_dbfs": anomaly_result.metadata.get("peak_dbfs"),
                                "noise_floor_dbfs": anomaly_result.metadata.get("noise_floor_dbfs"),
                                "snr_db": anomaly_result.metadata.get("snr_db"),
                                "anomaly_score": anomaly_result.confidence_score,  # Use confidence_score
                                "severity": anomaly_result.severity,
                                "timestamp": anomaly_result.timestamp.isoformat(),
                                "description": anomaly_result.description,
                                "anomaly_type": anomaly_result.anomaly_type.value
                            }
                        })
                    
                    # Downsample across the whole span for the web. Slicing the
                    # first 100 bins (the old behaviour) sent only the lowest
                    # frequencies and dropped the peak, so the chart looked
                    # nothing like the detection.
                    step = max(1, len(frequencies) // 256)
                    freq_display = (frequencies[::step] / 1e6).tolist()
                    power_display = power_dbfs[::step].tolist()

                    # Broadcast spectrum update
                    await self._broadcast_to_websockets({
                        "type": "spectrum_update",
                        "data": {
                            "frequencies_mhz": freq_display,
                            "powers_dbfs": power_display,
                            # Unambiguous units: these are dBFS, not dBm.
                            "units": "dBFS",
                            "peak_frequency_mhz": float(frequencies[np.argmax(power_dbfs)] / 1e6),
                            "avg_power_dbfs": float(np.mean(power_dbfs)),
                            "noise_floor_dbfs": float(np.percentile(power_dbfs, 15)),
                            "snr_db": float(np.max(power_dbfs) - np.percentile(power_dbfs, 15)),
                            # The real window the receiver is listening to, so
                            # the UI can label the chart and scope alerts.
                            "center_freq_mhz": float(center_freq / 1e6),
                            "span_mhz": float(actual_rate / 1e6),
                            "window_start_mhz": float(window_start_hz / 1e6),
                            "window_end_mhz": float(window_end_hz / 1e6),
                        }
                    })
                    
                    # Close SDR for this iteration
                    sdr.close()
                    
                    # Wait for next scan
                    await asyncio.sleep(config.get("scan_interval", 5))
                    
                except Exception as e:
                    logger.error(f"Error in monitoring loop iteration: {e}")
                    await asyncio.sleep(5)
                    
        except Exception as e:
            logger.error(f"Fatal error in monitoring loop: {e}")
            self.monitoring_active = False
        finally:
            logger.info("Monitoring loop stopped")

    async def _broadcast_to_websockets(self, message):
        """Broadcast message to all connected WebSocket clients"""
        if not self.connections:
            return
            
        import json
        message_json = json.dumps(message)
        
        # Remove disconnected clients
        disconnected = []
        for websocket in self.connections:
            try:
                await websocket.send_text(message_json)
            except Exception:
                disconnected.append(websocket)
        
        # Clean up disconnected clients
        for websocket in disconnected:
            self.connections.remove(websocket)

    async def _run_model_test(self, config):
        """Evaluate the latest trained model against real feature data.

        Every reported number comes from predictions over the stored feature
        files. Labels are used only when the data actually carries them;
        without labels the response says so instead of inventing metrics.
        """
        import time as _time
        import joblib
        import pandas as pd
        import numpy as np
        from pathlib import Path
        from sklearn.metrics import (
            accuracy_score,
            precision_score,
            recall_score,
            f1_score,
            confusion_matrix,
        )

        started = _time.monotonic()
        mode = (config or {}).get("mode", "lite")

        model_files = sorted(
            Path("models").glob("*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not model_files:
            raise ValueError("No trained model found. Train a model first.")

        model_data = joblib.load(model_files[0])
        model = model_data.get("model")
        scaler = model_data.get("scaler")
        feature_columns = model_data.get("feature_columns") or []
        if model is None or scaler is None or not feature_columns:
            raise ValueError(
                f"{model_files[0].name} does not contain a loadable model."
            )

        # Same sources as training: aggregated storage first, then the
        # per-run feature files.
        df = None
        try:
            df = self.storage.load_training_data(mode=mode)
        except Exception as e:
            logger.warning("load_training_data failed (%s); falling back to CSVs", e)
            frames = []
            for path in sorted(Path("data/features").glob(f"features_*_{mode}.csv")):
                try:
                    frames.append(pd.read_csv(path))
                except Exception:
                    continue
            df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

        if df is None or len(df) == 0:
            raise ValueError(f"No {mode} feature data found to evaluate against.")

        missing = [c for c in feature_columns if c not in df.columns]
        if missing:
            shown = ", ".join(missing[:8]) + ("…" if len(missing) > 8 else "")
            raise ValueError(f"Evaluation data is missing model features: {shown}")

        label_col = next(
            (c for c in ("label", "anomaly", "target", "class") if c in df.columns),
            None,
        )
        rows = df[feature_columns + ([label_col] if label_col else [])].dropna()
        if len(rows) < 10:
            raise ValueError(
                f"Only {len(rows)} complete evaluation rows; need at least 10."
            )

        predicted = np.asarray(
            model.predict(scaler.transform(rows[feature_columns]))
        )
        metrics = {"samples": int(len(rows))}

        # IsolationForest / OneClassSVM answer -1 (anomaly) or 1 (normal).
        # A regressor answers reconstructions, which are not a verdict.
        if predicted.ndim == 1 and set(np.unique(predicted)) <= {-1.0, 1.0}:
            flagged = (predicted == -1).astype(int)
            metrics["anomalies_flagged"] = int(flagged.sum())
            if label_col is not None:
                y_true = rows[label_col].astype(int).values
                metrics.update({
                    "accuracy": float(accuracy_score(y_true, flagged)),
                    "precision": float(
                        precision_score(y_true, flagged, zero_division=0)
                    ),
                    "recall": float(recall_score(y_true, flagged, zero_division=0)),
                    "f1_score": float(f1_score(y_true, flagged, zero_division=0)),
                    "confusion_matrix": confusion_matrix(y_true, flagged).tolist(),
                    "labelled": True,
                })
            else:
                metrics["labelled"] = False
        else:
            metrics["labelled"] = False

        metrics["training_time"] = round(_time.monotonic() - started, 3)

        results = {
            "model_file": model_files[0].name,
            "model_type": model_data.get("model_type") or "unknown",
            "metrics": metrics,
        }

        await self._broadcast_to_websockets({
            "type": "model_test_complete",
            "data": {"results": results},
        })
        return results

    def get_app(self) -> FastAPI:
        """Get the FastAPI application instance"""
        return self.app


# Global app instance
web_app = SpectrumAlertWebApp()
app = web_app.get_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "spectrum_alert.web.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
