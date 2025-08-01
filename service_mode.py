#!/usr/bin/env python3
"""
SpectrumAlert Service Mode - Continuous Spectrum Monitoring
Designed to run as a Docker service for automated monitoring
"""

import os
import sys
import time
import signal
import logging
import json
from pathlib import Path
from datetime import datetime

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.utils.config_manager import ConfigManager
    from src.core.model_manager import ModelManager
    from src.core.data_manager import DataManager
    from src.core.spectrum_monitor import SpectrumMonitor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class SpectrumAlertService:
    """Service mode for continuous spectrum monitoring"""
    
    def __init__(self):
        self.monitor = None
        self.running = False
        self.config = None
        self.model_manager = None
        self.data_manager = None
        
        # Service configuration from environment
        self.service_model = os.getenv('SERVICE_MODEL', 'latest')
        self.lite_mode = os.getenv('SERVICE_LITE_MODE', 'false').lower() == 'true'
        self.alert_threshold = float(os.getenv('SERVICE_ALERT_THRESHOLD', '0.7'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        self.setup_logging()
        self.setup_signal_handlers()
    
    def setup_logging(self):
        """Setup logging for service mode"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/app/logs/service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('SpectrumService')
        self.logger.info("SpectrumAlert Service Mode starting...")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        if self.monitor:
            self.monitor.stop_monitoring()
        sys.exit(0)
    
    def initialize_system(self):
        """Initialize the monitoring system"""
        try:
            # Initialize configuration
            self.config = ConfigManager()
            self.config.load_config()
            
            # Initialize managers
            self.model_manager = ModelManager()
            self.data_manager = DataManager()
            
            # Create necessary directories
            os.makedirs("/app/logs", exist_ok=True)
            os.makedirs("/app/data", exist_ok=True)
            os.makedirs("/app/models", exist_ok=True)
            os.makedirs("/app/config", exist_ok=True)
            
            self.logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def find_best_model(self):
        """Find the best model for monitoring"""
        if not self.model_manager:
            self.logger.error("Model manager not initialized")
            return None, None
            
        model_files = self.model_manager.list_models()
        anomaly_models = [f for f in model_files if 'anomaly' in f.lower()]
        
        if not anomaly_models:
            self.logger.error("No anomaly detection models found")
            return None, None
        
        # Select model based on service configuration
        selected_model = None
        
        if self.service_model == 'latest':
            # Sort by modification time (newest first)
            model_times = []
            for model in anomaly_models:
                try:
                    model_path = os.path.join(self.model_manager.model_dir, model)
                    mtime = os.path.getmtime(model_path)
                    model_times.append((model, mtime))
                except:
                    continue
            
            if model_times:
                model_times.sort(key=lambda x: x[1], reverse=True)
                selected_model = model_times[0][0]
        
        elif self.service_model in anomaly_models:
            selected_model = self.service_model
        
        else:
            # Fallback to first available model
            selected_model = anomaly_models[0]
        
        if not selected_model:
            self.logger.error("No suitable anomaly model found")
            return None, None
        
        # Find corresponding RF model
        if "anomaly_detection" in selected_model:
            rf_model = selected_model.replace("anomaly_detection", "rf_fingerprinting")
        else:
            # Look for any RF model
            rf_models = [f for f in model_files if 'rf' in f.lower() or 'fingerprint' in f.lower()]
            rf_model = rf_models[0] if rf_models else None
        
        if not rf_model:
            self.logger.error("No RF fingerprinting model found")
            return None, None
        
        self.logger.info(f"Selected models - Anomaly: {selected_model}, RF: {rf_model}")
        return selected_model, rf_model
    
    def determine_lite_mode(self, anomaly_model):
        """Determine lite mode from model metadata or filename"""
        if not self.model_manager:
            return "lite" in anomaly_model.lower()
            
        try:
            metadata = self.model_manager.load_model_metadata(anomaly_model)
            if metadata and 'lite_mode' in metadata:
                return metadata['lite_mode']
        except:
            pass
        
        # Fallback to filename check
        return "lite" in anomaly_model.lower()
    
    def save_service_status(self, status_data):
        """Save service status to JSON file"""
        try:
            status_file = "/app/logs/service_status.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save service status: {e}")
    
    def run_service(self):
        """Main service loop"""
        self.logger.info("Starting continuous spectrum monitoring service...")
        
        # Find models
        anomaly_model, rf_model = self.find_best_model()
        if not (anomaly_model and rf_model):
            self.logger.error("Cannot start service without models")
            return False
        
        # Determine lite mode
        actual_lite_mode = self.determine_lite_mode(anomaly_model)
        if self.lite_mode != actual_lite_mode:
            self.logger.warning(f"Overriding lite_mode from {self.lite_mode} to {actual_lite_mode} based on model")
            self.lite_mode = actual_lite_mode
        
        # Save initial status
        status_data = {
            "service_started": datetime.now(),
            "anomaly_model": anomaly_model,
            "rf_model": rf_model,
            "lite_mode": self.lite_mode,
            "alert_threshold": self.alert_threshold,
            "status": "starting"
        }
        self.save_service_status(status_data)
        
        self.running = True
        restart_count = 0
        max_restarts = 5
        
        while self.running:
            try:
                self.logger.info(f"Starting monitoring session (attempt {restart_count + 1})")
                
                # Initialize monitor
                if not self.config:
                    raise Exception("Configuration not initialized")
                    
                self.monitor = SpectrumMonitor(
                    self.config,
                    lite_mode=self.lite_mode,
                    rf_model_file=rf_model,
                    anomaly_model_file=anomaly_model
                )
                
                # Update status
                status_data["status"] = "monitoring"
                status_data["last_restart"] = datetime.now()
                status_data["restart_count"] = restart_count
                self.save_service_status(status_data)
                
                # Start monitoring
                self.monitor.start_monitoring()
                
                # If we get here, monitoring stopped normally
                self.logger.info("Monitoring session completed normally")
                break
                
            except KeyboardInterrupt:
                self.logger.info("Service interrupted by user")
                break
                
            except Exception as e:
                restart_count += 1
                self.logger.error(f"Monitoring failed (attempt {restart_count}): {e}")
                
                if restart_count >= max_restarts:
                    self.logger.error(f"Max restart attempts ({max_restarts}) reached. Stopping service.")
                    status_data["status"] = "failed"
                    status_data["error"] = str(e)
                    self.save_service_status(status_data)
                    break
                
                # Wait before restart
                wait_time = min(30 * restart_count, 300)  # Progressive backoff, max 5 minutes
                self.logger.info(f"Waiting {wait_time} seconds before restart...")
                time.sleep(wait_time)
                
                # Update status
                status_data["status"] = "restarting"
                status_data["last_error"] = str(e)
                self.save_service_status(status_data)
            
            finally:
                if self.monitor:
                    self.monitor.stop_monitoring()
                    self.monitor = None
        
        # Final status update
        status_data["status"] = "stopped"
        status_data["service_stopped"] = datetime.now()
        self.save_service_status(status_data)
        
        self.logger.info("SpectrumAlert Service stopped")
        return True
    
    def check_prerequisites(self):
        """Check if system prerequisites are met"""
        self.logger.info("Checking system prerequisites...")
        
        # Check RTL-SDR availability
        try:
            from src.core.robust_collector import SafeRTLSDR
            test_sdr = SafeRTLSDR()
            if test_sdr.open():
                self.logger.info("✓ RTL-SDR device accessible")
                test_sdr.close()
            else:
                self.logger.error("✗ RTL-SDR device not accessible")
                return False
        except Exception as e:
            self.logger.error(f"✗ RTL-SDR error: {e}")
            return False
        
        # Check for models
        if not self.model_manager:
            self.logger.error("✗ Model manager not initialized")
            return False
            
        model_files = self.model_manager.list_models()
        if not model_files:
            self.logger.error("✗ No trained models found")
            return False
        
        self.logger.info(f"✓ Found {len(model_files)} trained models")
        return True


def main():
    """Main entry point for service mode"""
    service = SpectrumAlertService()
    
    if not service.initialize_system():
        sys.exit(1)
    
    if not service.check_prerequisites():
        service.logger.error("Prerequisites not met. Please ensure RTL-SDR is connected and models are trained.")
        sys.exit(1)
    
    try:
        success = service.run_service()
        sys.exit(0 if success else 1)
    except Exception as e:
        service.logger.error(f"Service failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
