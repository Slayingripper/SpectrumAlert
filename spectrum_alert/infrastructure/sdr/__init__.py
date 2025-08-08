"""
SDR interface for SpectrumAlert v1.1 with enhanced error handling
"""

import logging
import numpy as np
import time
import signal
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from contextlib import contextmanager
from spectrum_alert.core.exceptions import SDRError

if TYPE_CHECKING:
    from rtlsdr import RtlSdr

logger = logging.getLogger(__name__)


class SDRConnectionError(SDRError):
    """SDR connection/communication error"""
    pass


class SDRConfigurationError(SDRError):
    """SDR configuration error"""
    pass


class SDRInterface(ABC):
    """Abstract base class for SDR devices with robust error handling"""
    
    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        self.device_args = device_args or {}
        self._sample_rate: Optional[float] = None
        self._center_freq: Optional[float] = None
        self._gain: Optional[Union[float, str]] = None
        self._is_open = False
        self._retry_count = 0
        self._max_retries = 3
        self._retry_delay = 1.0  # seconds
    
    def _retry_operation(self, operation, *args, **kwargs):
        """Retry an operation with exponential backoff"""
        for attempt in range(self._max_retries + 1):
            try:
                return operation(*args, **kwargs)
            except Exception as e:
                if attempt == self._max_retries:
                    logger.error(f"Operation failed after {self._max_retries} attempts: {e}")
                    raise SDRConnectionError(f"SDR operation failed: {e}")
                
                wait_time = self._retry_delay * (2 ** attempt)
                logger.warning(f"Operation failed (attempt {attempt + 1}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
    
    @contextmanager
    def safe_operation(self):
        """Context manager for safe SDR operations"""
        try:
            if not self._is_open:
                self.open()
            yield self
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
            raise
        except Exception as e:
            logger.error(f"SDR operation failed: {e}")
            # Try to recover
            try:
                self.close()
                time.sleep(1)
                self.open()
                logger.info("SDR recovered successfully")
            except Exception as recovery_error:
                logger.error(f"SDR recovery failed: {recovery_error}")
                raise SDRConnectionError(f"SDR operation and recovery failed: {e}")
    
    @abstractmethod
    def open(self) -> None:
        """Open the SDR device"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the SDR device"""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from the SDR"""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the sample rate"""
        pass
    
    @abstractmethod
    def set_center_freq(self, frequency: float) -> None:
        """Set the center frequency"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: Union[float, str]) -> None:
        """Set the gain"""
        pass


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
            raise SDRConnectionError(f"Failed to open RTL-SDR: {e}")
    
    def close(self) -> None:
        """Close the RTL-SDR device"""
        if self.sdr:
            try:
                self.sdr.close()
                self._is_open = False
                logger.info("RTL-SDR device closed")
            except Exception as e:
                logger.error(f"Error closing RTL-SDR: {e}")
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from RTL-SDR"""
        if not self.sdr:
            raise SDRConnectionError("SDR device not open")
        
        try:
            return self.sdr.read_samples(num_samples)
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            raise SDRConnectionError(f"Failed to read samples: {e}")
    
    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the sample rate"""
        if not self.sdr:
            raise SDRConnectionError("SDR device not open")
        
        try:
            self.sdr.sample_rate = sample_rate
            self._sample_rate = sample_rate
            logger.debug(f"Sample rate set to {sample_rate} Hz")
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            raise SDRConfigurationError(f"Failed to set sample rate: {e}")
    
    def set_center_freq(self, frequency: float) -> None:
        """Set the center frequency"""
        if not self.sdr:
            raise SDRConnectionError("SDR device not open")
        
        try:
            self.sdr.center_freq = frequency
            self._center_freq = frequency
            logger.debug(f"Center frequency set to {frequency} Hz")
        except Exception as e:
            logger.error(f"Error setting center frequency: {e}")
            raise SDRConfigurationError(f"Failed to set center frequency: {e}")
    
    def set_gain(self, gain: Union[float, str]) -> None:
        """Set the gain"""
        if not self.sdr:
            raise SDRConnectionError("SDR device not open")
        
        try:
            if gain == 'auto':
                self.sdr.gain = 'auto'
            else:
                self.sdr.gain = float(gain)
            self._gain = gain
            logger.debug(f"Gain set to {gain}")
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            raise SDRConfigurationError(f"Failed to set gain: {e}")


# Export the main classes
__all__ = ['SDRInterface', 'RTLSDRInterface', 'SDRConnectionError', 'SDRConfigurationError']
