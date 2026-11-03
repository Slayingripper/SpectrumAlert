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
        # What the caller last asked for, which is not the same as _gain: the
        # tuner snaps a request to the nearest step it supports, so asking for
        # 20.0 dB reports back 19.7. Comparing a request against the reported
        # value therefore never matches, and the caller re-programs the tuner on
        # every single frame until libusb gives up with a PIPE error.
        self._gain_requested = None
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
        """The gain the tuner actually applied, as read back from the device."""
        return self._gain

    @property
    def requested_gain(self) -> Optional[Union[float, str]]:
        """The gain last requested.

        Use this to decide whether a new gain needs sending. ``gain`` is the
        tuner's own answer and will differ from the request whenever the
        request lands between two supported steps, which makes an equality
        check against ``gain`` wrong.
        """
        return self._gain_requested
    
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
    def safe_operation(self, recover: bool = True, close_on_exit: bool = False):
        """Context manager for SDR operations with optional device recovery.

        ``recover``
            On failure, close and reopen the device once, then **re-raise**. The
            original implementation returned normally after a "successful"
            recovery, which silently swallowed the exception and let the caller
            proceed as if the operation had worked; worse, it re-ran the same
            failing body with no bound. Callers now always see the error.

        ``close_on_exit``
            Close the device when the block exits, including on the success
            path. Without this the device leaked whenever a ``with`` block ended
            normally.
        """
        opened_here = False
        if not self._is_open:
            self.open()
            opened_here = True

        try:
            yield self
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
            raise
        except SDRError:
            if not recover:
                raise
            logger.warning("SDR operation failed; attempting one device reopen")
            try:
                self.close()
                time.sleep(1.0)
                self.open()
                opened_here = True
            except Exception as recovery_error:
                logger.error("SDR recovery failed: %s", recovery_error)
                raise
            raise
        finally:
            if close_on_exit or (opened_here and not recover):
                try:
                    self.close()
                except Exception as close_error:  # pragma: no cover - best effort
                    logger.warning("error closing SDR during cleanup: %s", close_error)
    
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
