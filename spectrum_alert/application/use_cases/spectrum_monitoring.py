"""
Spectrum monitoring use case for SpectrumAlert v3.0
"""

import logging
from typing import Optional, Dict, Any
from spectrum_alert.core.domain.models import (
    FeatureVector,
    MonitoringSession,
    MonitoringStatus,
    SpectrumData,
)
from spectrum_alert.core.domain.models import DetectionMode
from spectrum_alert.infrastructure.sdr import SDRInterface
from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.core.exceptions import MonitoringError, SDRError

logger = logging.getLogger(__name__)


def _enum_value(value: Any) -> Any:
    """Unwrap an enum to its value, tolerating a value that is already a str."""
    return getattr(value, "value", value)


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
        self.anomaly_detector: Optional[Any] = None  # Will be set externally
    
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
            num_samples = int(sample_rate * duration)

            # Configure the SDR. Applying tuning once per capture meant three USB
            # control transfers plus a retune for every frame; tuning is now
            # skipped when the device is already where it needs to be.
            #
            # Gain is compared against `requested_gain`, not `gain`. `gain` is
            # what the tuner reports back, and the tuner snaps a request to the
            # nearest supported step, so 20.0 comes back as 19.7 and the
            # comparison never matched: the gain was re-programmed on every
            # frame until libusb failed the second one with a PIPE error.
            if self.sdr.center_freq != frequency:
                self.sdr.set_center_freq(frequency)
            if self.sdr.sample_rate != sample_rate:
                self.sdr.set_sample_rate(sample_rate)
            requested_gain = getattr(self.sdr, 'requested_gain', None)
            if requested_gain is None:
                requested_gain = self.sdr.gain
            if requested_gain != gain:
                self.sdr.set_gain(gain)

            samples = self.sdr.read_samples(num_samples)

            # Record what the tuner actually applied, not what was requested.
            # The RTL2832 snaps to the nearest supported rate, so caching the
            # request makes every stored frequency axis wrong.
            actual_rate = self.sdr.sample_rate or sample_rate

            spectrum_data = SpectrumData(
                frequency_hz=frequency,
                sample_rate_hz=sample_rate,
                gain_db=gain if isinstance(gain, (int, float)) else 0.0,
                samples=samples.tolist(),
                duration_seconds=duration,
                sample_rate_actual_hz=float(actual_rate),
            )

            self.storage.save_spectrum_data(spectrum_data)

            logger.debug(f"Captured {len(samples)} samples at {frequency} Hz")
            return spectrum_data
        except Exception as e:
            logger.error(f"Error capturing spectrum data: {e}")
            raise SDRError(f"Failed to capture spectrum data: {e}")

    def extract_features(self, spectrum_data: SpectrumData) -> Optional[FeatureVector]:
        """Extract features from spectrum data and persist them.

        Returns the persisted ``FeatureVector``, or ``None`` if the capture was
        too short to describe. Previously this raised on every call because the
        storage method was named ``save_features`` on one side and
        ``save_feature_vector`` on the other, which broke the entire monitoring
        loop.
        """
        import numpy as np

        try:
            samples = np.asarray(spectrum_data.samples)
            if samples.size == 0:
                return None
            if not np.iscomplexobj(samples):
                samples = samples.astype(np.complex64, copy=False)

            feature_set = self.feature_extractor.extract_features(
                samples, spectrum_data.frequency_hz
            )

            feature_vector = FeatureVector(
                spectrum_data_id=spectrum_data.id,
                mode=DetectionMode.LITE if self.feature_extractor.lite_mode else DetectionMode.FULL,
                features={
                    name: value
                    for name, value in zip(feature_set.feature_names, feature_set.features)
                },
                feature_names=feature_set.feature_names,
            )

            self.storage.save_feature_vector(feature_vector)

            logger.debug(f"Extracted {len(feature_set.features)} features")
            return feature_vector
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise MonitoringError(f"Failed to extract features: {e}")
    
    def stop_monitoring_session(self) -> Optional[MonitoringSession]:
        """Stop the current monitoring session"""
        try:
            if self.current_session:
                from datetime import datetime
                self.current_session.end_time = datetime.utcnow()
                # Assign the enum, not its .value: Pydantic re-validates on
                # assignment only when validate_assignment is on, so writing the
                # raw string here would leave the field the wrong type.
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
            'start_time': self.current_session.start_time.isoformat(),
            'duration_seconds': self.current_session.duration_seconds,
            'frequency_range_hz': list(self.current_session.frequency_range_hz),
            # .value on a str: MonitoringSession used to set use_enum_values=True,
            # so detection_mode arrived here as a bare string and this raised
            # AttributeError on every call.
            'detection_mode': _enum_value(self.current_session.detection_mode),
            'status': _enum_value(self.current_session.status),
            'is_active': self.current_session.is_active
        }
