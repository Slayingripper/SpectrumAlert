"""
RTL-SDR implementation of SDR interface
"""

import logging
import numpy as np
from typing import Optional, Dict, Any, Union

from .base import SDRInterface
from spectrum_alert.core.exceptions import SDRError

logger = logging.getLogger(__name__)


class RTLSDRInterface(SDRInterface):
    """RTL-SDR implementation of SDR interface"""
    
    def __init__(self, device_index: int = 0, device_args: Optional[Dict[str, Any]] = None):
        super().__init__(device_args)
        self.device_index = device_index
        self.sdr = None
    
    def open(self) -> None:
        """Open the RTL-SDR device"""
        try:
            from rtlsdr import RtlSdr
            self.sdr = RtlSdr(device_index=self.device_index)
            self._is_open = True
            logger.info(f"RTL-SDR device {self.device_index} opened successfully")
        except Exception as e:
            logger.error(f"Failed to open RTL-SDR device {self.device_index}: {e}")
            raise SDRError(f"Failed to open RTL-SDR: {e}")
    
    def close(self) -> None:
        """Close the RTL-SDR device"""
        if self.sdr:
            try:
                self.sdr.close()
                self._is_open = False
                logger.info("RTL-SDR device closed")
            except Exception as e:
                logger.error(f"Error closing RTL-SDR: {e}")
            finally:
                self.sdr = None
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from RTL-SDR"""
        if not self.sdr:
            raise SDRError("SDR device not open")
        
        try:
            return self.sdr.read_samples(num_samples)
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            raise SDRError(f"Failed to read samples: {e}")
    
    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the sample rate"""
        if not self.sdr:
            raise SDRError("SDR device not open")
        
        try:
            self.sdr.sample_rate = sample_rate
            self._sample_rate = sample_rate
            logger.debug(f"Sample rate set to {sample_rate} Hz")
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            raise SDRError(f"Failed to set sample rate: {e}")
    
    def set_center_freq(self, frequency: float) -> None:
        """Set the center frequency"""
        if not self.sdr:
            raise SDRError("SDR device not open")
        
        try:
            self.sdr.center_freq = frequency
            self._center_freq = frequency
            logger.debug(f"Center frequency set to {frequency} Hz")
        except Exception as e:
            logger.error(f"Error setting center frequency: {e}")
            raise SDRError(f"Failed to set center frequency: {e}")
    
    def set_gain(self, gain: Union[float, str]) -> None:
        """Set the gain"""
        if not self.sdr:
            raise SDRError("SDR device not open")
        
        try:
            if gain == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(gain)
            self._gain = gain
            logger.debug(f"Gain set to {gain}")
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            raise SDRError(f"Failed to set gain: {e}")
