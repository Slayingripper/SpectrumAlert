#!/usr/bin/env python3
"""
SpectrumAlert Autonomous Service - Full Automated Workflow
Captures data for 24 hours, trains models, then monitors continuously
"""

import os
import sys
import time
import signal
import logging
import json
import threading
from pathlib import Path
from datetime import datetime, timedelta
from enum import Enum

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.utils.config_manager import ConfigManager
    from src.core.robust_collector import RobustDataCollector
    from src.core.model_manager import RFFingerprintingTrainer, AnomalyDetectionTrainer, ModelManager
    from src.core.data_manager import DataManager
    from src.core.spectrum_monitor import SpectrumMonitor
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

class ServiceState(Enum):
    INITIALIZING = "initializing"
    DATA_COLLECTION = "data_collection"
    MODEL_TRAINING = "model_training"
    MONITORING = "monitoring"
    ERROR = "error"
    STOPPED = "stopped"

class AutonomousSpectrumService:
    """Fully autonomous spectrum monitoring service"""
    
    def __init__(self):
        self.state = ServiceState.INITIALIZING
        self.running = False
        self.config = None
        self.model_manager = None
        self.data_manager = None
        self.collector = None
        self.monitor = None
        
        # Service configuration from environment
        self.collection_hours = float(os.getenv('COLLECTION_HOURS', '24'))
        self.lite_mode = os.getenv('SERVICE_LITE_MODE', 'false').lower() == 'true'
        self.alert_threshold = float(os.getenv('SERVICE_ALERT_THRESHOLD', '0.7'))
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.retrain_interval_hours = float(os.getenv('RETRAIN_INTERVAL_HOURS', '168'))  # Weekly
        self.min_training_samples = int(os.getenv('MIN_TRAINING_SAMPLES', '1000'))
        
        self.setup_logging()
        self.setup_signal_handlers()
        
        # Status tracking
        self.start_time = None
        self.collection_start_time = None
        self.training_start_time = None
        self.monitoring_start_time = None
        self.last_retrain_time = None
        
    def setup_logging(self):
        """Setup logging for autonomous service"""
        log_level = getattr(logging, self.log_level.upper(), logging.INFO)
        
        os.makedirs("/app/logs", exist_ok=True)
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('/app/logs/autonomous_service.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('AutonomousService')
        self.logger.info("SpectrumAlert Autonomous Service starting...")
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        self.logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
        self.state = ServiceState.STOPPED
        
        if self.collector:
            self.collector.stop_collection()
        if self.monitor:
            self.monitor.stop_monitoring()
            
        self.save_service_status()
        sys.exit(0)
    
    def initialize_system(self):
        """Initialize the autonomous system"""
        try:
            self.logger.info("Initializing autonomous spectrum monitoring system...")
            
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
            
            self.start_time = datetime.now()
            self.logger.info("System initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            self.state = ServiceState.ERROR
            return False
    
    def save_service_status(self):
        """Save comprehensive service status to JSON file"""
        try:
            status_data = {
                "service_type": "autonomous",
                "state": self.state.value,
                "start_time": self.start_time,
                "collection_hours_configured": self.collection_hours,
                "lite_mode": self.lite_mode,
                "alert_threshold": self.alert_threshold,
                "retrain_interval_hours": self.retrain_interval_hours,
                "min_training_samples": self.min_training_samples,
                "collection_start_time": self.collection_start_time,
                "training_start_time": self.training_start_time,
                "monitoring_start_time": self.monitoring_start_time,
                "last_retrain_time": self.last_retrain_time,
                "last_update": datetime.now()
            }
            
            # Add state-specific information
            if self.state == ServiceState.DATA_COLLECTION and self.collection_start_time:
                elapsed = datetime.now() - self.collection_start_time
                remaining = timedelta(hours=self.collection_hours) - elapsed
                status_data["collection_progress"] = {
                    "elapsed_hours": elapsed.total_seconds() / 3600,
                    "remaining_hours": max(0, remaining.total_seconds() / 3600),
                    "completion_percentage": min(100, (elapsed.total_seconds() / (self.collection_hours * 3600)) * 100)
                }
            
            if self.data_manager:
                data_files = self.data_manager.list_data_files()
                status_data["data_files_count"] = len(data_files)
                status_data["latest_data_files"] = data_files[-5:] if data_files else []
            
            if self.model_manager:
                model_files = self.model_manager.list_models()
                status_data["model_files_count"] = len(model_files)
                status_data["available_models"] = model_files
            
            status_file = "/app/logs/autonomous_status.json"
            with open(status_file, 'w') as f:
                json.dump(status_data, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.warning(f"Failed to save service status: {e}")
    
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
                return True
            else:
                self.logger.error("✗ RTL-SDR device not accessible")
                return False
        except Exception as e:
            self.logger.error(f"✗ RTL-SDR error: {e}")
            return False
    
    def collect_training_data(self):
        """Collect data for the specified duration"""
        self.logger.info(f"Starting autonomous data collection for {self.collection_hours} hours...")
        self.state = ServiceState.DATA_COLLECTION
        self.collection_start_time = datetime.now()
        
        try:
            # Generate filename with timestamp
            timestamp = self.collection_start_time.strftime("%Y%m%d_%H%M%S")
            mode_suffix = "_lite" if self.lite_mode else "_full"
            filename = f"data/autonomous_training_{timestamp}{mode_suffix}.csv"
            
            self.logger.info(f"Data will be saved to: {filename}")
            self.logger.info(f"Collection mode: {'Lite' if self.lite_mode else 'Full'}")
            
            # Initialize collector
            self.collector = RobustDataCollector(self.config)
            
            # Start collection in background thread to allow status updates
            collection_thread = threading.Thread(
                target=self._collection_worker,
                args=(filename,)
            )
            collection_thread.daemon = True
            collection_thread.start()
            
            # Monitor collection progress
            while collection_thread.is_alive() and self.running:
                self.save_service_status()
                time.sleep(60)  # Update status every minute
                
                # Log progress every hour
                if self.collection_start_time:
                    elapsed = datetime.now() - self.collection_start_time
                    if elapsed.total_seconds() % 3600 < 60:  # Every hour
                        hours_elapsed = elapsed.total_seconds() / 3600
                        hours_remaining = max(0, self.collection_hours - hours_elapsed)
                        self.logger.info(f"Data collection progress: {hours_elapsed:.1f}/{self.collection_hours} hours completed, {hours_remaining:.1f} hours remaining")
            
            collection_thread.join(timeout=10)
            
            if self.running:
                self.logger.info("✓ Data collection completed successfully")
                return True
            else:
                self.logger.warning("Data collection interrupted")
                return False
                
        except Exception as e:
            self.logger.error(f"Data collection failed: {e}")
            return False
        finally:
            self.collector = None
    
    def _collection_worker(self, filename):
        """Background worker for data collection"""
        try:
            success = self.collector.collect_data(
                duration_minutes=self.collection_hours * 60,
                output_filename=filename,
                lite_mode=self.lite_mode
            )
            return success
        except Exception as e:
            self.logger.error(f"Collection worker failed: {e}")
            return False
    
    def train_models_automatically(self):
        """Automatically train models on collected data"""
        self.logger.info("Starting autonomous model training...")
        self.state = ServiceState.MODEL_TRAINING
        self.training_start_time = datetime.now()
        
        try:
            # Find the latest data file
            if not self.data_manager:
                self.logger.error("Data manager not initialized")
                return False
                
            data_files = self.data_manager.list_data_files()
            if not data_files:
                self.logger.error("No data files found for training")
                return False
            
            # Use the most recent data file
            latest_data_file = max(data_files, key=lambda f: os.path.getmtime(
                os.path.join(self.data_manager.data_dir, f)
            ))
            
            self.logger.info(f"Training models using: {latest_data_file}")
            
            # Load features
            features = self.data_manager.load_features_csv(latest_data_file)
            if features is None or len(features) == 0:
                self.logger.error("No features loaded from data file")
                return False
            
            if len(features) < self.min_training_samples:
                self.logger.warning(f"Only {len(features)} samples available (minimum: {self.min_training_samples})")
                self.logger.warning("Training anyway, but model quality may be reduced")
            
            self.logger.info(f"Loaded {len(features)} feature vectors for training")
            
            # Generate model names with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            mode_suffix = "_lite" if self.lite_mode else "_full"
            
            # Train RF fingerprinting model
            self.logger.info("Training RF fingerprinting model...")
            rf_trainer = RFFingerprintingTrainer(lite_mode=self.lite_mode)
            rf_model, rf_metadata = rf_trainer.train_model(features)
            
            if rf_model:
                rf_model_file = f"autonomous_rf_{timestamp}{mode_suffix}.pkl"
                rf_trainer.model_manager.save_model(rf_model, rf_model_file, rf_metadata)
                self.logger.info(f"✓ RF fingerprinting model saved: {rf_model_file}")
            else:
                self.logger.error("RF fingerprinting model training failed")
                return False
            
            # Train anomaly detection model
            self.logger.info("Training anomaly detection model...")
            anomaly_trainer = AnomalyDetectionTrainer(lite_mode=self.lite_mode)
            anomaly_model, anomaly_metadata = anomaly_trainer.train_model(features)
            
            if anomaly_model:
                anomaly_model_file = f"autonomous_anomaly_{timestamp}{mode_suffix}.pkl"
                anomaly_trainer.model_manager.save_model(anomaly_model, anomaly_model_file, anomaly_metadata)
                self.logger.info(f"✓ Anomaly detection model saved: {anomaly_model_file}")
            else:
                self.logger.error("Anomaly detection model training failed")
                return False
            
            self.logger.info("✓ Model training completed successfully!")
            self.last_retrain_time = datetime.now()
            return True
            
        except Exception as e:
            self.logger.error(f"Model training failed: {e}")
            return False
    
    def start_monitoring(self):
        """Start continuous anomaly monitoring"""
        self.logger.info("Starting autonomous anomaly monitoring...")
        self.state = ServiceState.MONITORING
        self.monitoring_start_time = datetime.now()
        
        try:
            # Find the latest trained models
            if not self.model_manager:
                self.logger.error("Model manager not initialized")
                return False
                
            model_files = self.model_manager.list_models()
            anomaly_models = [f for f in model_files if 'anomaly' in f.lower()]
            
            if not anomaly_models:
                self.logger.error("No anomaly detection models found")
                return False
            
            # Use the most recent anomaly model
            latest_anomaly_model = max(anomaly_models, key=lambda f: os.path.getmtime(
                os.path.join(self.model_manager.model_dir, f)
            ))
            
            # Find corresponding RF model
            if "autonomous_anomaly" in latest_anomaly_model:
                latest_rf_model = latest_anomaly_model.replace("autonomous_anomaly", "autonomous_rf")
            else:
                # Fallback: find any RF model
                rf_models = [f for f in model_files if 'rf' in f.lower() or 'fingerprint' in f.lower()]
                if rf_models:
                    latest_rf_model = max(rf_models, key=lambda f: os.path.getmtime(
                        os.path.join(self.model_manager.model_dir, f)
                    ))
                else:
                    self.logger.error("No RF fingerprinting model found")
                    return False
            
            self.logger.info(f"Using anomaly model: {latest_anomaly_model}")
            self.logger.info(f"Using RF model: {latest_rf_model}")
            
            # Start monitoring
            if not self.config:
                self.logger.error("Configuration not initialized")
                return False
                
            self.monitor = SpectrumMonitor(
                self.config,
                lite_mode=self.lite_mode,
                rf_model_file=latest_rf_model,
                anomaly_model_file=latest_anomaly_model
            )
            
            # Monitor with periodic retraining check
            self._monitoring_loop()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Monitoring startup failed: {e}")
            return False
    
    def _monitoring_loop(self):
        """Main monitoring loop with periodic retraining"""
        self.logger.info("Starting monitoring loop with periodic retraining...")
        
        while self.running:
            try:
                # Start monitoring in a separate thread
                if not self.monitor:
                    self.logger.error("Monitor not initialized")
                    return
                    
                monitor_thread = threading.Thread(target=self.monitor.start_monitoring)
                monitor_thread.daemon = True
                monitor_thread.start()
                
                # Check for retraining needs periodically
                while self.running and monitor_thread.is_alive():
                    self.save_service_status()
                    time.sleep(300)  # Check every 5 minutes
                    
                    # Check if it's time for retraining
                    if self.should_retrain():
                        self.logger.info("Initiating scheduled retraining...")
                        self.monitor.stop_monitoring()
                        monitor_thread.join(timeout=30)
                        
                        # Perform retraining
                        if self.retrain_models():
                            self.logger.info("Retraining successful, restarting monitoring...")
                            # Restart monitoring with new models
                            self.start_monitoring()
                            return
                        else:
                            self.logger.warning("Retraining failed, continuing with existing models")
                            # Continue with existing models
                            break
                
                monitor_thread.join(timeout=10)
                
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                time.sleep(60)  # Wait before retrying
    
    def should_retrain(self):
        """Check if models should be retrained"""
        if not self.last_retrain_time:
            return False
        
        time_since_retrain = datetime.now() - self.last_retrain_time
        return time_since_retrain.total_seconds() > (self.retrain_interval_hours * 3600)
    
    def retrain_models(self):
        """Retrain models with accumulated data"""
        self.logger.info("Starting scheduled model retraining...")
        
        # Collect additional data for a shorter period (1 hour)
        short_collection_hours = 1.0
        self.logger.info(f"Collecting {short_collection_hours} hours of fresh data for retraining...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_suffix = "_lite" if self.lite_mode else "_full"
        filename = f"data/retrain_data_{timestamp}{mode_suffix}.csv"
        
        # Quick data collection
        collector = RobustDataCollector(self.config)
        success = collector.collect_data(
            duration_minutes=short_collection_hours * 60,
            output_filename=filename,
            lite_mode=self.lite_mode
        )
        
        if success:
            self.logger.info("Fresh data collected, training updated models...")
            return self.train_models_automatically()
        else:
            self.logger.warning("Fresh data collection failed, skipping retraining")
            return False
    
    def run_autonomous_service(self):
        """Main autonomous service workflow"""
        self.logger.info("Starting autonomous spectrum monitoring workflow...")
        self.running = True
        
        try:
            # Step 1: Check if we already have trained models
            existing_models = self.model_manager.list_models() if self.model_manager else []
            anomaly_models = [f for f in existing_models if 'anomaly' in f.lower()]
            
            if anomaly_models:
                self.logger.info(f"Found {len(anomaly_models)} existing trained models")
                self.logger.info("Skipping data collection and training, starting monitoring...")
                self.last_retrain_time = datetime.now()  # Set retrain time to now
                return self.start_monitoring()
            
            # Step 2: Collect training data
            self.logger.info("No existing models found, starting full autonomous workflow...")
            if not self.collect_training_data():
                self.logger.error("Data collection failed, cannot proceed")
                self.state = ServiceState.ERROR
                return False
            
            # Step 3: Train models
            if not self.train_models_automatically():
                self.logger.error("Model training failed, cannot proceed")
                self.state = ServiceState.ERROR
                return False
            
            # Step 4: Start monitoring
            if not self.start_monitoring():
                self.logger.error("Monitoring startup failed")
                self.state = ServiceState.ERROR
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Autonomous service failed: {e}")
            self.state = ServiceState.ERROR
            return False
        finally:
            self.save_service_status()


def main():
    """Main entry point for autonomous service"""
    service = AutonomousSpectrumService()
    
    if not service.initialize_system():
        sys.exit(1)
    
    if not service.check_prerequisites():
        service.logger.error("Prerequisites not met. Please ensure RTL-SDR is connected.")
        sys.exit(1)
    
    try:
        success = service.run_autonomous_service()
        sys.exit(0 if success else 1)
    except Exception as e:
        service.logger.error(f"Autonomous service failed with unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
