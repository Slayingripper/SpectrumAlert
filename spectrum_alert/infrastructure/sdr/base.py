"""
Base SDR interface and error classes
"""

import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Optional, Dict, Any, Union

from spectrum_alert.core.exceptions import SDRError

logger = logging.getLogger(__name__)


class SDRInterface(ABC):
    """Abstract base class for SDR interfaces"""
    
    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        self.device_args = device_args or {}
        self._is_open = False
        self._sample_rate = None
        self._center_freq = None
        self._gain = None
        self._max_retries = 3
        self._retry_delay = 1.0
    
    @property
    def is_open(self) -> bool:
        """Check if the SDR device is open"""
        return self._is_open
    
    @property
    def sample_rate(self) -> Optional[float]:
        """Get current sample rate"""
        return self._sample_rate
    
    @property
    def center_freq(self) -> Optional[float]:
        """Get current center frequency"""
        return self._center_freq
    
    @property
    def gain(self) -> Optional[Union[float, str]]:
        """Get current gain"""
        return self._gain
    
    def with_retry(self, operation_func, *args, **kwargs):
        """Execute operation with retry logic"""
        for attempt in range(self._max_retries):
            try:
                return operation_func(*args, **kwargs)
            except Exception as e:
                if attempt == self._max_retries - 1:
                    logger.error(f"Operation failed after {self._max_retries} attempts: {e}")
                    raise SDRError(f"SDR operation failed: {e}")
                
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
                raise SDRError(f"SDR operation and recovery failed: {e}")
    
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
