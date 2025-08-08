"""
Spectrum monitoring use case for SpectrumAlert v3.0
"""

import logging
from typing import Optional, Dict, Any
from spectrum_alert.core.domain.models import SpectrumData, MonitoringSession, DetectionMode, MonitoringStatus
from spectrum_alert.infrastructure.sdr import SDRInterface
from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.core.exceptions import MonitoringError, SDRError

logger = logging.getLogger(__name__)


class SpectrumMonitoringUseCase:
    """Use case for spectrum monitoring operations"""
    
    def __init__(
        self,
        sdr_interface: SDRInterface,
        storage: DataStorage,
        feature_extractor: FeatureExtractor
    ):
        self.sdr = sdr_interface
        self.storage = storage
        self.feature_extractor = feature_extractor
        self.current_session: Optional[MonitoringSession] = None
    
    def start_monitoring_session(
        self,
        session_name: str,
        frequency_range: tuple[float, float],
        detection_mode: DetectionMode,
        configuration: Optional[Dict[str, Any]] = None
    ) -> MonitoringSession:
        """Start a new monitoring session"""
        try:
            session = MonitoringSession(
                name=session_name,
                frequency_range_hz=frequency_range,
                detection_mode=detection_mode,
                configuration=configuration or {}
            )
            
            self.current_session = session
            logger.info(f"Started monitoring session: {session_name}")
            return session
        except Exception as e:
            logger.error(f"Error starting monitoring session: {e}")
            raise MonitoringError(f"Failed to start monitoring session: {e}")
    
    def capture_spectrum_data(
        self,
        frequency: float,
        sample_rate: float,
        gain: float,
        duration: float
    ) -> SpectrumData:
        """Capture spectrum data from SDR"""
        try:
            # Configure SDR
            self.sdr.set_center_freq(frequency)
            self.sdr.set_sample_rate(sample_rate)
            self.sdr.set_gain(gain)
            
            # Calculate number of samples
            num_samples = int(sample_rate * duration)
            
            # Read samples
            samples = self.sdr.read_samples(num_samples)
            
            # Create spectrum data object
            spectrum_data = SpectrumData(
                frequency_hz=frequency,
                sample_rate_hz=sample_rate,
                gain_db=gain,
                samples=samples.tolist(),
                duration_seconds=duration
            )
            
            # Save to storage
            self.storage.save_spectrum_data(spectrum_data)
            
            logger.debug(f"Captured {len(samples)} samples at {frequency} Hz")
            return spectrum_data
        except Exception as e:
            logger.error(f"Error capturing spectrum data: {e}")
            raise SDRError(f"Failed to capture spectrum data: {e}")
    
    def extract_features(self, spectrum_data: SpectrumData) -> None:
        """Extract features from spectrum data"""
        try:
            # Convert samples back to numpy array
            import numpy as np
            samples = np.array([complex(s) for s in spectrum_data.samples])
            
            # Extract features
            feature_set = self.feature_extractor.extract_features(
                samples, spectrum_data.frequency_hz
            )
            
            # Create feature vector domain object
            from spectrum_alert.core.domain.models import FeatureVector
            feature_vector = FeatureVector(
                spectrum_data_id=spectrum_data.id,
                mode=DetectionMode.LITE if self.feature_extractor.lite_mode else DetectionMode.FULL,
                features={name: value for name, value in zip(feature_set.feature_names, feature_set.features)},
                feature_names=feature_set.feature_names
            )
            
            # Save features
            self.storage.save_features(feature_vector)
            
            logger.debug(f"Extracted {len(feature_set.features)} features")
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise MonitoringError(f"Failed to extract features: {e}")
    
    def stop_monitoring_session(self) -> Optional[MonitoringSession]:
        """Stop the current monitoring session"""
        try:
            if self.current_session:
                from datetime import datetime
                self.current_session.end_time = datetime.utcnow()
                self.current_session.status = MonitoringStatus.STOPPED
                
                logger.info(f"Stopped monitoring session: {self.current_session.name}")
                session = self.current_session
                self.current_session = None
                return session
            else:
                logger.warning("No active monitoring session to stop")
                return None
        except Exception as e:
            logger.error(f"Error stopping monitoring session: {e}")
            raise MonitoringError(f"Failed to stop monitoring session: {e}")
    
    def get_session_status(self) -> Optional[Dict[str, Any]]:
        """Get current session status"""
        if not self.current_session:
            return None
        
        return {
            'id': self.current_session.id,
            'name': self.current_session.name,
            'status': self.current_session.status,
            'start_time': self.current_session.start_time.isoformat(),
            'duration_seconds': self.current_session.duration_seconds,
            'frequency_range_hz': self.current_session.frequency_range_hz,
            'detection_mode': self.current_session.detection_mode.value,
            'is_active': self.current_session.is_active
        }
