"""
Real-time spectrum monitoring for SpectrumAlert
"""

import time
import logging
import threading
from typing import Optional, Dict, Any
from datetime import datetime
from src.core.sdr_interface import SDRFactory, sdr_context
from src.core.feature_extraction import FeatureExtractor, FeatureValidator
from src.core.model_manager import ModelManager, ModelPredictor
from src.core.mqtt_manager import MQTTManager
from src.utils.config_manager import ConfigManager
from src.utils.logger import log_performance

logger = logging.getLogger(__name__)


class SpectrumMonitor:
    """Real-time spectrum monitoring and anomaly detection"""
    
    def __init__(self, config_manager: ConfigManager, lite_mode: bool = False, 
                 rf_model_file: Optional[str] = None, anomaly_model_file: Optional[str] = None):
        self.config = config_manager
        self.lite_mode = lite_mode
        self.rf_model_file = rf_model_file
        self.anomaly_model_file = anomaly_model_file
        self.feature_extractor = FeatureExtractor(lite_mode=False)  # Always use full extractor for compatibility
        self.model_manager = ModelManager()
        self.predictor = ModelPredictor(self.model_manager)
        self.mqtt_manager = MQTTManager(config_manager.mqtt)
        self._stop_event = threading.Event()
        self._monitoring_stats = {
            'samples_processed': 0,
            'anomalies_detected': 0,
            'mqtt_messages_sent': 0,
            'errors': 0,
            'start_time': None
        }
        # Will be set after loading models
        self._expected_features = None
        self._load_models()
    
    def _load_models(self) -> bool:
        """Load trained models"""
        try:
            # Use provided model files or fall back to default naming
            if self.rf_model_file and self.anomaly_model_file:
                rf_model_file = self.rf_model_file
                anomaly_model_file = self.anomaly_model_file
            else:
                mode_suffix = "lite" if self.lite_mode else "full"
                rf_model_file = f"rf_model_{mode_suffix}.pkl"
                anomaly_model_file = f"anomaly_model_{mode_suffix}.pkl"
            
            success = self.predictor.load_models(rf_model_file, anomaly_model_file)
            
            if success:
                # Get expected features from anomaly model metadata
                anomaly_metadata = self.model_manager.load_model_metadata(anomaly_model_file)
                if anomaly_metadata and 'feature_names' in anomaly_metadata:
                    self._expected_features = anomaly_metadata['feature_names']
                    logger.info(f"Model expects {len(self._expected_features)} features: {self._expected_features}")
                else:
                    logger.warning("No feature names found in model metadata, using default extraction")
                    self._expected_features = None
                
                logger.info("Models loaded successfully for monitoring")
                return True
            else:
                logger.error("Failed to load required models")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    @log_performance
    def start_monitoring(self) -> None:
        """Start real-time spectrum monitoring"""
        if not self._load_models():
            logger.error("Cannot start monitoring without trained models")
            return
        
        logger.info("Starting spectrum monitoring...")
        self._monitoring_stats['start_time'] = datetime.now()
        
        try:
            # Connect to MQTT broker
            if not self.mqtt_manager.connect():
                logger.warning("Failed to connect to MQTT broker, continuing without MQTT")
            
            # Start monitoring loop
            with sdr_context(self.config.general.sdr_type) as sdr:
                self._configure_sdr(sdr)
                self._monitoring_loop(sdr)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        except Exception as e:
            logger.error(f"Error during monitoring: {e}")
        finally:
            self._stop_monitoring()
    
    def _configure_sdr(self, sdr) -> None:
        """Configure SDR for monitoring"""
        try:
            # Use lite mode settings if enabled
            if self.lite_mode:
                sample_rate = min(self.config.general.sample_rate, 1.024e6)
                sample_size = 64 * 1024
            else:
                sample_rate = self.config.general.sample_rate
                sample_size = 128 * 1024
            
            sdr.set_sample_rate(sample_rate)
            sdr.set_gain('auto')
            
            self.sample_size = sample_size
            logger.info(f"SDR configured for monitoring: {sample_rate} Hz, {sample_size} samples")
            
        except Exception as e:
            logger.error(f"Error configuring SDR: {e}")
            raise
    
    def _monitoring_loop(self, sdr) -> None:
        """Main monitoring loop"""
        logger.info("Monitoring loop started - Press Ctrl+C to stop")
        
        while not self._stop_event.is_set():
            try:
                for band in self.config.ham_bands:
                    if self._stop_event.is_set():
                        break
                    
                    current_freq = band.start_freq
                    while current_freq <= band.end_freq and not self._stop_event.is_set():
                        self._monitor_frequency(sdr, current_freq)
                        current_freq += self.config.general.freq_step
                        
                        # Small delay to prevent overwhelming the system
                        time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                self._monitoring_stats['errors'] += 1
                time.sleep(1)  # Wait before retrying
    
    def _monitor_frequency(self, sdr, frequency: float) -> None:
        """Monitor a single frequency"""
        try:
            # Set frequency and collect samples
            sdr.set_center_freq(frequency)
            iq_samples = sdr.read_samples(self.sample_size)
            
            # Extract features compatible with the model
            if self._expected_features:
                feature_set = self.feature_extractor.extract_features_compatible(
                    iq_samples, frequency, self._expected_features)
            else:
                # Fallback to standard extraction
                feature_set = self.feature_extractor.extract_features(iq_samples, frequency)
            
            # Validate features
            if not FeatureValidator.validate_features(feature_set):
                feature_set = FeatureValidator.sanitize_features(feature_set)
            
            # Detect anomaly
            anomaly_result = self.predictor.detect_anomaly(feature_set)
            
            # Calculate signal strength
            signal_strength = self._calculate_signal_strength(iq_samples)
            
            # Update stats
            self._monitoring_stats['samples_processed'] += 1
            
            # Handle anomaly detection
            if anomaly_result.get('is_anomaly', False):
                self._handle_anomaly(frequency, anomaly_result, signal_strength)
            
            # Send regular telemetry
            self._send_telemetry(frequency, signal_strength, anomaly_result)
            
            # Log monitoring activity (reduced frequency)
            if self._monitoring_stats['samples_processed'] % 100 == 0:
                logger.info(f"Processed {self._monitoring_stats['samples_processed']} samples, "
                          f"detected {self._monitoring_stats['anomalies_detected']} anomalies")
            
        except Exception as e:
            logger.error(f"Error monitoring frequency {frequency}: {e}")
            self._monitoring_stats['errors'] += 1
    
    def _calculate_signal_strength(self, iq_samples) -> float:
        """Calculate signal strength in dB"""
        try:
            import numpy as np
            amplitude = np.abs(iq_samples)
            power = np.mean(amplitude ** 2)
            if power > 0:
                return 10 * np.log10(power)
            else:
                return -100.0  # Very low signal
        except:
            return -100.0
    
    def _handle_anomaly(self, frequency: float, anomaly_result: Dict[str, Any], 
                       signal_strength: float) -> None:
        """Handle detected anomaly"""
        try:
            self._monitoring_stats['anomalies_detected'] += 1
            
            logger.warning(f"ANOMALY DETECTED at {frequency/1e6:.3f} MHz")
            logger.warning(f"  Score: {anomaly_result.get('score', 0):.3f}")
            logger.warning(f"  Confidence: {anomaly_result.get('confidence', 0):.3f}")
            logger.warning(f"  Signal Strength: {signal_strength:.1f} dB")
            
            # Send MQTT alert
            coordinates = (self.config.receiver.latitude, self.config.receiver.longitude)
            metadata = {
                'signal_strength_db': signal_strength,
                'confidence': anomaly_result.get('confidence', 0),
                'lite_mode': self.lite_mode
            }
            
            success = self.mqtt_manager.publish_anomaly(
                frequency, 
                anomaly_result.get('score', 0),
                coordinates,
                metadata
            )
            
            if success:
                self._monitoring_stats['mqtt_messages_sent'] += 1
            
        except Exception as e:
            logger.error(f"Error handling anomaly: {e}")
    
    def _send_telemetry(self, frequency: float, signal_strength: float, 
                       anomaly_result: Dict[str, Any]) -> None:
        """Send regular telemetry data"""
        try:
            # Send signal strength (every 10th sample to reduce MQTT traffic)
            if self._monitoring_stats['samples_processed'] % 10 == 0:
                coordinates = (self.config.receiver.latitude, self.config.receiver.longitude)
                
                success = self.mqtt_manager.publish_signal_strength(
                    frequency, signal_strength, coordinates
                )
                
                if success:
                    self._monitoring_stats['mqtt_messages_sent'] += 1
            
        except Exception as e:
            logger.debug(f"Error sending telemetry: {e}")
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        logger.info("Stopping spectrum monitoring...")
        self._stop_event.set()
    
    def _stop_monitoring(self) -> None:
        """Clean up after monitoring"""
        try:
            self.mqtt_manager.disconnect()
            self._log_monitoring_stats()
            logger.info("Spectrum monitoring stopped")
        except Exception as e:
            logger.error(f"Error during monitoring cleanup: {e}")
    
    def _log_monitoring_stats(self) -> None:
        """Log monitoring statistics"""
        stats = self._monitoring_stats
        if stats['start_time']:
            duration = datetime.now() - stats['start_time']
            logger.info(f"Monitoring session completed:")
            logger.info(f"  Duration: {duration}")
            logger.info(f"  Samples processed: {stats['samples_processed']}")
            logger.info(f"  Anomalies detected: {stats['anomalies_detected']}")
            logger.info(f"  MQTT messages sent: {stats['mqtt_messages_sent']}")
            logger.info(f"  Errors: {stats['errors']}")
            
            if duration.total_seconds() > 0:
                rate = stats['samples_processed'] / duration.total_seconds()
                logger.info(f"  Processing rate: {rate:.2f} samples/second")
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get current monitoring statistics"""
        return self._monitoring_stats.copy()
    
    def is_monitoring(self) -> bool:
        """Check if currently monitoring"""
        return not self._stop_event.is_set()


class SpectrumAnalyzer:
    """Analyzes spectrum data for patterns and trends"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
    
    def analyze_frequency_usage(self, features_data: list) -> Dict[str, Any]:
        """Analyze frequency usage patterns"""
        try:
            import numpy as np
            from collections import defaultdict
            
            frequency_stats = defaultdict(list)
            
            # Group features by frequency
            for feature_set in features_data:
                freq_mhz = feature_set.frequency / 1e6
                frequency_stats[freq_mhz].extend(feature_set.features)
            
            # Analyze each frequency
            analysis = {}
            for freq, features in frequency_stats.items():
                if features:
                    analysis[freq] = {
                        'mean_activity': float(np.mean(features)),
                        'std_activity': float(np.std(features)),
                        'sample_count': len(features)
                    }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing frequency usage: {e}")
            return {}
    
    def generate_activity_report(self, monitoring_stats: Dict[str, Any]) -> str:
        """Generate monitoring activity report"""
        try:
            report = []
            report.append("=== Spectrum Monitoring Report ===")
            report.append(f"Samples Processed: {monitoring_stats.get('samples_processed', 0)}")
            report.append(f"Anomalies Detected: {monitoring_stats.get('anomalies_detected', 0)}")
            report.append(f"MQTT Messages Sent: {monitoring_stats.get('mqtt_messages_sent', 0)}")
            report.append(f"Errors: {monitoring_stats.get('errors', 0)}")
            
            if monitoring_stats.get('start_time'):
                start_time = monitoring_stats['start_time']
                if isinstance(start_time, str):
                    start_time = datetime.fromisoformat(start_time)
                report.append(f"Started: {start_time}")
            
            return "\n".join(report)
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return "Error generating report"
