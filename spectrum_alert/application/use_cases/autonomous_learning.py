"""
Autonomous continuous learning use case for SpectrumAlert v1.1
"""

import logging
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass
from spectrum_alert.core.domain.models import DetectionMode, MonitoringStatus, SpectrumData
from spectrum_alert.infrastructure.sdr import SDRInterface
from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.application.use_cases.spectrum_monitoring import SpectrumMonitoringUseCase
from spectrum_alert.core.exceptions import MonitoringError, SDRError

logger = logging.getLogger(__name__)


@dataclass
class AutonomousCycleConfig:
    """Configuration for autonomous cycle"""
    data_collection_minutes: int = 10
    training_interval_hours: int = 1  
    monitoring_interval_minutes: int = 30
    frequency_start: float = 88.0e6
    frequency_end: float = 108.0e6
    detection_mode: DetectionMode = DetectionMode.LITE
    sample_rate: float = 2.048e6
    gain: float = 30.0
    max_cycles: Optional[int] = None  # None = infinite
    retention_days: int = 7  # how long to keep raw/feature/anomaly files
    detect_during_collection: bool = False  # disable detection in Phase 1 by default
    monitor_window_ms: float = 200.0  # analysis window per read during monitoring
    per_frequency_hold_seconds: float = 10.0  # how long to stay on each freq before hopping


class AutonomousLearningUseCase:
    """Use case for autonomous continuous learning and monitoring"""
    
    def __init__(
        self,
        sdr: SDRInterface,
        storage: DataStorage,
        feature_extractor: FeatureExtractor,
        config: AutonomousCycleConfig,
        status_callback: Optional[Callable] = None
    ):
        self.sdr = sdr
        self.storage = storage
        self.feature_extractor = feature_extractor
        self.config = config
        self.status_callback = status_callback
        
        # Initialize use cases
        self.monitoring_use_case = SpectrumMonitoringUseCase(sdr, storage, feature_extractor)
        # Import anomaly detection use case with multiple fallbacks
        try:
            from .anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        except Exception:
            try:
                from spectrum_alert.application.use_cases import AnomalyDetectionUseCase  # type: ignore
            except Exception:
                from spectrum_alert.application.use_cases.anomaly_detection import AnomalyDetectionUseCase  # type: ignore
        self.anomaly_use_case = AnomalyDetectionUseCase(feature_extractor)
        
        # State tracking
        self.is_running = False
        self.current_cycle = 0
        self.last_training_time = None
        self.total_data_collected = 0
        self.cycle_start_time = None
        self._stop_event = threading.Event()
        
        # MQTT (configured via config manager)
        self.mqtt_mgr = None
        try:
            from spectrum_alert.config.manager import ConfigurationManager
            from spectrum_alert.infrastructure.messaging import MQTTManager
            cfg = ConfigurationManager()
            if cfg.get_setting('mqtt.enabled'):
                self.mqtt_mgr = MQTTManager(
                    broker=cfg.get_setting('mqtt.broker_host'),
                    port=cfg.get_setting('mqtt.broker_port'),
                    username=cfg.get_setting('mqtt.username'),
                    password=cfg.get_setting('mqtt.password'),
                    client_id=cfg.get_setting('mqtt.client_id'),
                    topic_prefix=cfg.get_setting('mqtt.topic_prefix'),
                    qos=cfg.get_setting('mqtt.qos'),
                    keepalive_seconds=cfg.get_setting('mqtt.keepalive_seconds'),
                    tls_enabled=cfg.get_setting('mqtt.tls_enabled'),
                )
        except Exception:
            # Config or MQTT not available; proceed without MQTT
            self.mqtt_mgr = None
    
    def _log_status(self, message: str, level: str = "info"):
        """Log status message and call callback if provided"""
        if level == "info":
            logger.info(message)
        elif level == "warning":
            logger.warning(message)
        elif level == "error":
            logger.error(message)
            
        if self.status_callback:
            self.status_callback(message, level)
        
        # Also publish status to MQTT if available
        try:
            if (self.mqtt_mgr is not None) and getattr(self.mqtt_mgr, 'connected', False):
                # Best-effort, ignore publish errors here
                self.mqtt_mgr.publish_status(status=level, details={"message": message})
        except Exception:
            pass
    
    def start_autonomous_mode(self) -> None:
        """Start the autonomous learning cycle"""
        self.is_running = True
        self.cycle_start_time = datetime.now()
        self._stop_event.clear()
        
        # Connect MQTT if configured
        if self.mqtt_mgr:
            try:
                self.mqtt_mgr.connect()
            except Exception as e:
                self._log_status(f"MQTT connect failed: {e}", "warning")
        
        self._log_status("🚀 Starting autonomous learning mode")
        self._log_status(f"📊 Data collection: {self.config.data_collection_minutes} min")
        self._log_status(f"🧠 Training interval: {self.config.training_interval_hours} hours")
        self._log_status(f"📡 Monitoring interval: {self.config.monitoring_interval_minutes} min")
        self._log_status(f"📻 Frequency range: {self.config.frequency_start/1e6:.1f} - {self.config.frequency_end/1e6:.1f} MHz")
        
        try:
            while self.is_running and not self._stop_event.is_set():
                if self.config.max_cycles and self.current_cycle >= self.config.max_cycles:
                    self._log_status(f"✅ Completed {self.config.max_cycles} cycles")
                    break
                    
                self.current_cycle += 1
                self._run_cycle()
                
        except KeyboardInterrupt:
            self._log_status("⏹️ Autonomous mode stopped by user")
        except Exception as e:
            self._log_status(f"❌ Autonomous mode failed: {e}", "error")
            raise MonitoringError(f"Autonomous mode failed: {e}")
        finally:
            # Disconnect MQTT
            try:
                if (self.mqtt_mgr is not None) and getattr(self.mqtt_mgr, 'connected', False):
                    self.mqtt_mgr.disconnect()
            except Exception:
                pass
            self.is_running = False
    
    def _run_cycle(self) -> None:
        """Run a single autonomous cycle"""
        cycle_start = datetime.now()
        self._log_status(f"🔄 Starting cycle {self.current_cycle}")
        
        try:
            # Phase 1: Data Collection
            self._data_collection_phase()
            
            # Phase 2: Training (if needed)
            if self._should_train():
                self._training_phase()
            
            # Phase 3: Monitoring
            self._monitoring_phase()
            
            cycle_duration = datetime.now() - cycle_start
            self._log_status(f"✅ Cycle {self.current_cycle} completed in {cycle_duration}")
            
        except Exception as e:
            self._log_status(f"❌ Cycle {self.current_cycle} failed: {e}", "error")
            # Continue to next cycle after brief delay
            time.sleep(30)
    
    def _data_collection_phase(self) -> None:
        """Phase 1: Collect baseline spectrum data with enhanced quality"""
        self._log_status(f"📊 Phase 1: Collecting data for {self.config.data_collection_minutes} minutes")
        self._log_status(f"🎯 Target: {self.config.frequency_start/1e6:.1f}-{self.config.frequency_end/1e6:.1f} MHz at {self.config.sample_rate/1e6:.1f} MSps")
        
        try:
            # Open SDR device with retry logic
            sdr_opened = False
            max_retries = 3
            
            for attempt in range(max_retries):
                try:
                    # Close any existing connection first
                    try:
                        self.sdr.close()
                        time.sleep(1)  # Wait for cleanup
                    except:
                        pass
                    
                    self.sdr.open()
                    sdr_opened = True
                    self._log_status(f"✅ SDR opened successfully (attempt {attempt + 1})")
                    break
                except Exception as sdr_error:
                    self._log_status(f"⚠️ SDR open attempt {attempt + 1} failed: {sdr_error}", "warning")
                    if attempt < max_retries - 1:
                        time.sleep(2)  # Wait before retry
                    else:
                        raise SDRError(f"Failed to open SDR after {max_retries} attempts: {sdr_error}")
            
            if not sdr_opened:
                raise SDRError("Unable to open SDR device")
            
            # Configure SDR for optimal data collection with conservative settings
            center_freq = (self.config.frequency_start + self.config.frequency_end) / 2
            
            # Set parameters one by one with error checking
            try:
                self.sdr.set_sample_rate(self.config.sample_rate)
                time.sleep(0.1)  # Brief delay between settings
                
                self.sdr.set_center_freq(center_freq)
                time.sleep(0.1)
                
                self.sdr.set_gain(self.config.gain)
                time.sleep(0.1)
                
            except Exception as config_error:
                self._log_status(f"⚠️ SDR configuration issue: {config_error}", "warning")
                # Continue with defaults if configuration fails
            
            self._log_status(f"🔧 SDR configured: {center_freq/1e6:.1f} MHz center, {self.config.sample_rate/1e6:.1f} MSps, {self.config.gain} dB gain")
            
            # Start monitoring session for data collection
            session_name = f"autonomous_data_collection_cycle_{self.current_cycle}"
            session = self.monitoring_use_case.start_monitoring_session(
                session_name=session_name,
                frequency_range=(self.config.frequency_start, self.config.frequency_end),
                detection_mode=self.config.detection_mode
            )
            
            # Enhanced data collection with continuous sampling
            collection_start = datetime.now()
            collection_end = collection_start + timedelta(minutes=self.config.data_collection_minutes)
            samples_collected = 0
            data_batches = 0
            
            # Use smaller, more conservative batch sizes to prevent USB overflow
            # Start with very small batches and gradually increase if stable
            base_batch_size = 1024  # Very conservative starting point
            max_batch_size = min(int(self.config.sample_rate * 0.01), 8192)  # Max 10ms or 8K samples
            current_batch_size = base_batch_size
            consecutive_successes = 0
            usb_error_count = 0
            max_usb_errors = 5
            
            self._log_status(f"🚀 Starting conservative data collection (batch size: {current_batch_size})...")
            
            # Enhanced data collection loop with proper error handling
            while datetime.now() < collection_end and not self._stop_event.is_set():
                try:
                    # Read samples from SDR with adaptive batch size
                    samples = self.sdr.read_samples(current_batch_size)
                    
                    # Only process every 10th batch to reduce CPU load and create meaningful chunks
                    if data_batches % 10 == 0:
                        # Calculate power spectrum from samples (reduce size for storage)
                        fft_samples = samples[:min(512, len(samples))]
                        power_spectrum = np.abs(np.fft.fft(fft_samples)).tolist()
                        
                        # Create spectrum data object with reduced sample storage
                        spectrum_data = SpectrumData(
                            timestamp=datetime.now(),
                            frequency_hz=center_freq,
                            sample_rate_hz=self.config.sample_rate,
                            gain_db=self.config.gain,
                            samples=fft_samples.tolist() if hasattr(fft_samples, 'tolist') else list(fft_samples),
                            power_spectrum=power_spectrum,
                            duration_seconds=current_batch_size / self.config.sample_rate
                        )
                        
                        # Store the data using correct method and extract features
                        try:
                            self.storage.save_spectrum_data(spectrum_data)
                            self.monitoring_use_case.extract_features(spectrum_data)
                            
                            # Optional anomaly detection during collection (disabled by default)
                            if self.config.detect_during_collection:
                                anomalies = self.anomaly_use_case.detect_anomalies(spectrum_data)
                                if anomalies:
                                    for anomaly in anomalies:
                                        try:
                                            self.storage.save_anomaly(anomaly)
                                        except Exception as sa_err:
                                            self._log_status(f"⚠️ Failed to save anomaly: {sa_err}", "warning")
                                        # Publish via MQTT with precise frequency
                                        try:
                                            if (self.mqtt_mgr is not None) and getattr(self.mqtt_mgr, 'connected', False):
                                                self.mqtt_mgr.publish_anomaly(
                                                    frequency=anomaly.frequency_hz,
                                                    score=anomaly.confidence_score,
                                                    details={
                                                        "severity": getattr(anomaly, 'severity', 'unknown'),
                                                        "description": getattr(anomaly, 'description', ''),
                                                        "mode": getattr(anomaly, 'detection_mode', self.config.detection_mode).value if hasattr(anomaly, 'detection_mode') else self.config.detection_mode.value,
                                                        "phase": "collection",
                                                        "cycle": self.current_cycle,
                                                        "exact_frequency_hz": anomaly.frequency_hz,
                                                        "frequency_mhz_precise": anomaly.frequency_hz / 1e6,
                                                    }
                                                )
                                        except Exception:
                                            pass
                                    # Log precise frequencies
                                    peaks = ", ".join(f"{a.frequency_hz/1e6:.6f} MHz (score {a.confidence_score:.2f})" for a in anomalies)
                                    self._log_status(f"🚨 Detected {len(anomalies)} anomaly(ies): {peaks}")
                        except Exception as storage_error:
                            self._log_status(f"⚠️ Storage error: {storage_error}", "warning")
                    
                    # Update counters and adapt batch size only on successful read/process
                    samples_collected += len(samples)
                    data_batches += 1
                    consecutive_successes += 1
                    
                    # Adaptive batch size - increase if stable
                    if consecutive_successes >= 50 and current_batch_size < max_batch_size:
                        current_batch_size = min(current_batch_size * 2, max_batch_size)
                        consecutive_successes = 0
                        self._log_status(f"📈 Increased batch size to {current_batch_size}")
                    
                    # Progress update every 100 batches (less frequent to reduce overhead)
                    if data_batches % 100 == 0:
                        remaining_time = (collection_end - datetime.now()).total_seconds()
                        data_rate_mbps = (samples_collected * 8) / (1024 * 1024) / ((datetime.now() - collection_start).total_seconds())
                        self._log_status(f"📊 Progress: {samples_collected:,} samples, {data_rate_mbps:.1f} MB/s, {remaining_time/60:.1f} min remaining")
                    
                    # Longer pause to prevent USB overflow - critical for stability
                    time.sleep(0.05)
                
                except Exception as sample_error:
                    error_msg = str(sample_error).lower()
                    if "overflow" in error_msg or "usb" in error_msg:
                        usb_error_count += 1
                        # Reduce batch size on USB errors
                        current_batch_size = max(base_batch_size, current_batch_size // 2)
                        consecutive_successes = 0
                        
                        self._log_status(f"⚠️ USB error #{usb_error_count}: {sample_error}", "warning")
                        self._log_status(f"🔧 Reduced batch size to {current_batch_size}")
                        
                        if usb_error_count >= max_usb_errors:
                            self._log_status(f"❌ Too many USB errors ({usb_error_count}), stopping collection", "error")
                            break
                        
                        # Longer recovery time for USB errors
                        time.sleep(1.0)
                    else:
                        self._log_status(f"⚠️ Sample collection error: {sample_error}", "warning")
                        time.sleep(0.2)
                    continue
            
            # Stop the session and close SDR
            self.monitoring_use_case.stop_monitoring_session()
            self.sdr.close()
            
            # Calculate collection statistics
            actual_duration = (datetime.now() - collection_start).total_seconds() / 60
            data_size_mb = samples_collected * 8 / (1024 * 1024)  # Complex samples = 8 bytes
            
            # Calculate rates safely
            if actual_duration > 0:
                avg_rate = samples_collected / (actual_duration * 60)  # samples per second
            else:
                avg_rate = 0
            
            # Calculate effective data rate (stored data vs collected)
            stored_batches = data_batches // 10  # Only every 10th batch was stored
            
            self.total_data_collected += samples_collected
            
            # Enhanced statistics with USB error information
            if samples_collected > 0:
                self._log_status(f"✅ Data collection completed!")
                self._log_status(f"   • Duration: {actual_duration:.1f} minutes")
                self._log_status(f"   • Samples collected: {samples_collected:,}")
                self._log_status(f"   • Samples stored: {stored_batches * 512:,}")  # 512 samples per stored batch
                self._log_status(f"   • Data size: {data_size_mb:.1f} MB collected") 
                self._log_status(f"   • Rate: {avg_rate/1e6:.2f} MSps average")
                self._log_status(f"   • Batches: {data_batches} total, {stored_batches} stored")
                self._log_status(f"   • USB errors: {usb_error_count}")
                self._log_status(f"   • Final batch size: {current_batch_size}")
            else:
                self._log_status(f"⚠️ Data collection completed with no samples", "warning")
                self._log_status(f"   • Duration: {actual_duration:.1f} minutes")
                self._log_status(f"   • USB errors: {usb_error_count}")
                if usb_error_count >= max_usb_errors:
                    self._log_status("   • Terminated due to excessive USB errors", "warning")
            
        except Exception as e:
            # Ensure SDR is closed on error
            try:
                self.sdr.close()
            except:
                pass
            self._log_status(f"❌ Data collection failed: {e}", "error")
            raise
    
    def _training_phase(self) -> None:
        """Phase 2: Train/update models with new data"""
        self._log_status("🧠 Phase 2: Training models with collected data")
        
        try:
            # Optional: clean up old data before training to keep disk in check
            try:
                removed = self.storage.cleanup_old_data(max_age_days=self.config.retention_days)
                self._log_status(f"🧹 Cleaned old data: {removed}")
            except Exception as e:
                self._log_status(f"⚠️ Cleanup skipped: {e}", "warning")
            
            # Load recent training data
            training_data = self.storage.load_training_data(
                mode=self.config.detection_mode.value,
                days=max(1, min(self.config.retention_days, 30))
            )
            
            if training_data is None or len(training_data) < 10:
                self._log_status("⚠️ Insufficient training data, skipping training", "warning")
                return
            
            # Train and persist anomaly model
            from spectrum_alert.application.use_cases.model_training import ModelTrainingUseCase
            trainer = ModelTrainingUseCase(self.storage)
            if trainer.train_models(self.config.detection_mode, days=max(1, min(self.config.retention_days, 30))):
                self.last_training_time = datetime.now()
                self._log_status("✅ Models trained and saved successfully")
                # Notify via MQTT
                try:
                    if (self.mqtt_mgr is not None) and getattr(self.mqtt_mgr, 'connected', False):
                        self.mqtt_mgr.publish_status("info", {"event": "training_complete", "cycle": self.current_cycle})
                except Exception:
                    pass
            else:
                self._log_status("⚠️ Model training returned False", "warning")
            
        except Exception as e:
            self._log_status(f"❌ Training failed: {e}", "error")
            # Continue without training
    
    def _monitoring_phase(self) -> None:
        """Phase 3: Active monitoring (near real-time streaming)"""
        self._log_status(f"📡 Phase 3: Active monitoring for {self.config.monitoring_interval_minutes} minutes")
        
        try:
            # Open SDR device
            self.sdr.open()
            # Configure static params once
            try:
                self.sdr.set_sample_rate(self.config.sample_rate)
                self.sdr.set_gain(self.config.gain)
            except Exception as config_error:
                self._log_status(f"⚠️ SDR configuration issue: {config_error}", "warning")
            
            # Start monitoring session
            session_name = f"autonomous_monitoring_cycle_{self.current_cycle}"
            session = self.monitoring_use_case.start_monitoring_session(
                session_name=session_name,
                frequency_range=(self.config.frequency_start, self.config.frequency_end),
                detection_mode=self.config.detection_mode
            )
            
            monitoring_end = datetime.now() + timedelta(minutes=self.config.monitoring_interval_minutes)
            key_frequencies = [
                self.config.frequency_start,
                (self.config.frequency_start + self.config.frequency_end) / 2.0,
                self.config.frequency_end,
            ]
            
            # Compute samples per window
            window_seconds = max(0.02, float(self.config.monitor_window_ms) / 1000.0)
            samples_per_window = max(2048, int(self.config.sample_rate * window_seconds))
            
            # Live loop across frequencies until time runs out
            while datetime.now() < monitoring_end and not self._stop_event.is_set():
                for freq in key_frequencies:
                    if datetime.now() >= monitoring_end or self._stop_event.is_set():
                        break
                    # Tune to frequency (keep SR/Gain as set)
                    try:
                        self.sdr.set_center_freq(freq)
                    except Exception as tune_err:
                        self._log_status(f"⚠️ Tune error @ {freq/1e6:.3f} MHz: {tune_err}", "warning")
                        continue
                    
                    stay_until = datetime.now() + timedelta(seconds=self.config.per_frequency_hold_seconds)
                    while datetime.now() < stay_until and datetime.now() < monitoring_end and not self._stop_event.is_set():
                        try:
                            samples = self.sdr.read_samples(samples_per_window)
                            # Build minimal SpectrumData without disk IO
                            sd = SpectrumData(
                                frequency_hz=freq,
                                sample_rate_hz=self.config.sample_rate,
                                gain_db=self.config.gain,
                                samples=samples.tolist(),
                                power_spectrum=None,
                                duration_seconds=samples_per_window / self.config.sample_rate,
                            )
                            # Detect and publish immediately
                            anomalies = self.anomaly_use_case.detect_anomalies(sd)
                            if anomalies:
                                for anomaly in anomalies:
                                    # Persist anomaly only (avoid writing every spectrum window)
                                    try:
                                        self.storage.save_anomaly(anomaly)
                                    except Exception as sa_err:
                                        self._log_status(f"⚠️ Failed to save anomaly: {sa_err}", "warning")
                                    # Publish via MQTT with precise frequency
                                    try:
                                        if (self.mqtt_mgr is not None) and getattr(self.mqtt_mgr, 'connected', False):
                                            self.mqtt_mgr.publish_anomaly(
                                                frequency=anomaly.frequency_hz,
                                                score=anomaly.confidence_score,
                                                details={
                                                    "severity": getattr(anomaly, 'severity', 'unknown'),
                                                    "description": getattr(anomaly, 'description', ''),
                                                    "mode": getattr(anomaly, 'detection_mode', self.config.detection_mode).value if hasattr(anomaly, 'detection_mode') else self.config.detection_mode.value,
                                                    "phase": "monitoring",
                                                    "cycle": self.current_cycle,
                                                    "at_freq_hz": freq,
                                                    "exact_frequency_hz": anomaly.frequency_hz,
                                                    "frequency_mhz_precise": anomaly.frequency_hz / 1e6,
                                                }
                                            )
                                    except Exception:
                                        pass
                                # Log precise frequencies
                                peaks = ", ".join(f"{a.frequency_hz/1e6:.6f} MHz (score {a.confidence_score:.2f})" for a in anomalies)
                                self._log_status(f"🚨 Detected {len(anomalies)} anomaly(ies): {peaks}")
                        except KeyboardInterrupt:
                            self._stop_event.set()
                            break
                        except Exception as read_err:
                            self._log_status(f"⚠️ Monitoring read error @ {freq/1e6:.3f} MHz: {read_err}", "warning")
                            time.sleep(0.02)
                            continue
                        # Tiny sleep to yield
                        time.sleep(0.005)
            
            # Stop monitoring and close SDR
            self.monitoring_use_case.stop_monitoring_session()
            self.sdr.close()
            self._log_status("✅ Monitoring complete")
            
        except Exception as e:
            # Ensure SDR is closed on error
            try:
                self.sdr.close()
            except:
                pass
            self._log_status(f"❌ Monitoring failed: {e}", "error")
            # Continue to next cycle
    
    def _should_train(self) -> bool:
        """Determine if training should be performed"""
        if self.last_training_time is None:
            return True  # First training
        
        time_since_training = datetime.now() - self.last_training_time
        return time_since_training >= timedelta(hours=self.config.training_interval_hours)
    
    def stop_autonomous_mode(self) -> None:
        """Stop the autonomous learning cycle"""
        self.is_running = False
        self._stop_event.set()
        self._log_status("🛑 Stopping autonomous mode...")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get autonomous mode statistics"""
        runtime = None
        if self.cycle_start_time:
            runtime = datetime.now() - self.cycle_start_time
        
        return {
            "cycles_completed": self.current_cycle,
            "total_data_collected": self.total_data_collected,
            "last_training": self.last_training_time.isoformat() if self.last_training_time else None,
            "runtime": str(runtime) if runtime else None,
            "is_running": self.is_running,
            "config": {
                "data_collection_minutes": self.config.data_collection_minutes,
                "training_interval_hours": self.config.training_interval_hours,
                "monitoring_interval_minutes": self.config.monitoring_interval_minutes,
                "frequency_range_mhz": f"{self.config.frequency_start/1e6:.1f}-{self.config.frequency_end/1e6:.1f}"
            }
        }
