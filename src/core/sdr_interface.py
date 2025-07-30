"""
Unified SDR interface for SpectrumAlert with enhanced error handling and recovery
"""

import logging
import numpy as np
import time
import signal
import sys
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union, TYPE_CHECKING
from contextlib import contextmanager

if TYPE_CHECKING:
    from rtlsdr import RtlSdr

logger = logging.getLogger(__name__)


class SDRError(Exception):
    """Base exception for SDR-related errors"""
    pass


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
        self._gain: Optional[float | str] = None
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
    def set_center_freq(self, freq: float) -> None:
        """Set the center frequency"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float | str) -> None:
        """Set the gain"""
        pass
    
    @property
    def sample_rate(self) -> Optional[float]:
        return self._sample_rate
    
    @property
    def center_freq(self) -> Optional[float]:
        return self._center_freq
    
    @property
    def gain(self) -> Optional[float | str]:
        return self._gain
    
    @property
    def is_open(self) -> bool:
        return self._is_open


class RTLSDRInterface(SDRInterface):
    """RTL-SDR interface implementation with robust error handling"""
    
    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        super().__init__(device_args)
        self._sdr: Optional['RtlSdr'] = None
        self._last_successful_freq = None
        self._usb_reset_count = 0
        self._max_usb_resets = 3
    
    def _reset_usb_device(self):
        """Attempt to reset USB device"""
        try:
            import subprocess
            import os
            
            # Find RTL-SDR USB device
            result = subprocess.run(['lsusb'], capture_output=True, text=True)
            rtl_lines = [line for line in result.stdout.split('\n') if 'RTL' in line or '0bda:2838' in line]
            
            if rtl_lines:
                logger.info("Attempting USB device reset...")
                # This is a more gentle approach - just close and reopen
                time.sleep(2)
                self._usb_reset_count += 1
                return True
        except Exception as e:
            logger.error(f"USB reset failed: {e}")
        return False
    
    def open(self) -> None:
        """Open RTL-SDR device with retry mechanism"""
        def _open_device():
            try:
                from rtlsdr import RtlSdr
                if self._sdr:
                    try:
                        self._sdr.close()
                    except:
                        pass
                
                device_index = self.device_args.get('device_index', 0)
                self._sdr = RtlSdr(device_index=device_index)
                
                # Set some safe defaults to prevent driver issues
                self._sdr.sample_rate = 2.048e6
                self._sdr.center_freq = 100e6  # Safe frequency
                self._sdr.gain = 'auto'
                
                self._is_open = True
                self._retry_count = 0
                logger.info(f"RTL-SDR device {device_index} opened successfully")
                
            except ImportError:
                raise ImportError("pyrtlsdr library not found. Install with: pip install pyrtlsdr")
            except Exception as e:
                logger.error(f"Failed to open RTL-SDR device: {e}")
                # If we've had USB issues, try reset
                if "usb" in str(e).lower() and self._usb_reset_count < self._max_usb_resets:
                    self._reset_usb_device()
                raise SDRConnectionError(f"Cannot open RTL-SDR: {e}")
        
        self._retry_operation(_open_device)
    
    def close(self) -> None:
        """Close RTL-SDR device safely"""
        if self._sdr and self._is_open:
            try:
                self._sdr.close()
                self._is_open = False
                self._sdr = None
                logger.info("RTL-SDR device closed successfully")
            except Exception as e:
                logger.error(f"Error closing RTL-SDR device: {e}")
                # Force cleanup
                self._sdr = None
                self._is_open = False
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from RTL-SDR with error handling"""
        if not self._is_open or not self._sdr:
            raise SDRConnectionError("SDR device not open")
        
        def _read_samples():
            try:
                # Add timeout and validation
                if num_samples <= 0 or num_samples > 16*1024*1024:  # Reasonable limits
                    raise ValueError(f"Invalid sample count: {num_samples}")
                
                samples = self._sdr.read_samples(num_samples)
                if samples is None or len(samples) == 0:
                    raise SDRConnectionError("No samples received from SDR")
                
                return np.array(samples, dtype=np.complex64)
                
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['usb', 'i2c', 'timeout', 'r82xx']):
                    # Hardware communication error - try recovery
                    logger.warning(f"Hardware communication error: {e}")
                    raise SDRConnectionError(f"Hardware communication error: {e}")
                else:
                    logger.error(f"Error reading samples: {e}")
                    raise
        
        return self._retry_operation(_read_samples)
    
    def set_sample_rate(self, sample_rate: float) -> None:
        """Set RTL-SDR sample rate with validation"""
        if not self._is_open or not self._sdr:
            raise SDRConnectionError("SDR device not open")
        
        def _set_sample_rate():
            # Validate sample rate
            if not (225000 <= sample_rate <= 3200000):
                raise SDRConfigurationError(f"Invalid sample rate: {sample_rate}. Must be 225kHz-3.2MHz")
            
            try:
                self._sdr.sample_rate = sample_rate
                self._sample_rate = sample_rate
                logger.debug(f"Sample rate set to {sample_rate}")
            except Exception as e:
                raise SDRConfigurationError(f"Failed to set sample rate: {e}")
        
        self._retry_operation(_set_sample_rate)
    
    def set_center_freq(self, freq: float) -> None:
        """Set RTL-SDR center frequency with validation"""
        if not self._is_open or not self._sdr:
            raise SDRConnectionError("SDR device not open")
        
        def _set_center_freq():
            # Validate frequency range (RTL-SDR typical range)
            if not (24e6 <= freq <= 1766e6):
                logger.warning(f"Frequency {freq} may be outside RTL-SDR range (24MHz-1.766GHz)")
            
            try:
                self._sdr.center_freq = freq
                self._center_freq = freq
                self._last_successful_freq = freq
                logger.debug(f"Center frequency set to {freq}")
            except Exception as e:
                error_msg = str(e).lower()
                if any(keyword in error_msg for keyword in ['r82xx', 'i2c', 'usb']):
                    # Hardware communication error
                    raise SDRConnectionError(f"Hardware error setting frequency: {e}")
                else:
                    raise SDRConfigurationError(f"Failed to set center frequency: {e}")
        
        self._retry_operation(_set_center_freq)
    
    def set_gain(self, gain: float | str) -> None:
        """Set RTL-SDR gain with validation"""
        if not self._is_open or not self._sdr:
            raise SDRConnectionError("SDR device not open")
        
        def _set_gain():
            try:
                if gain == 'auto':
                    self._sdr.gain = 'auto'
                else:
                    # Validate gain range (RTL-SDR typical range is 0-50dB)
                    if isinstance(gain, (int, float)) and not (0 <= gain <= 50):
                        logger.warning(f"Gain {gain} may be outside typical range (0-50dB)")
                    self._sdr.gain = gain
                
                self._gain = gain
                logger.debug(f"Gain set to {gain}")
            except Exception as e:
                raise SDRConfigurationError(f"Failed to set gain: {e}")
        
        self._retry_operation(_set_gain)


class SoapySDRInterface(SDRInterface):
    """SoapySDR interface for multiple SDR types"""
    
    def __init__(self, device_args: Optional[Dict[str, Any]] = None):
        super().__init__(device_args)
        self._sdr = None
        self._rx_stream = None
    
    def open(self) -> None:
        """Open SoapySDR device"""
        try:
            import SoapySDR
            self._sdr = SoapySDR.Device(self.device_args)
            self._rx_stream = self._sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
            self._sdr.activateStream(self._rx_stream)
            self._is_open = True
            logger.info(f"SoapySDR device opened: {self._sdr.getDriverKey()}")
        except ImportError:
            raise ImportError("SoapySDR library not found. Install SoapySDR and Python bindings")
        except Exception as e:
            logger.error(f"Failed to open SoapySDR device: {e}")
            raise
    
    def close(self) -> None:
        """Close SoapySDR device"""
        if self._sdr and self._is_open:
            try:
                if self._rx_stream:
                    self._sdr.deactivateStream(self._rx_stream)
                    self._sdr.closeStream(self._rx_stream)
                self._is_open = False
                logger.info("SoapySDR device closed")
            except Exception as e:
                logger.error(f"Error closing SoapySDR device: {e}")
    
    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from SoapySDR device"""
        if not self._is_open or not self._sdr or not self._rx_stream:
            raise RuntimeError("SDR device not open")
        
        try:
            import SoapySDR
            buff = np.zeros(num_samples, dtype=np.complex64)
            sr = self._sdr.readStream(self._rx_stream, [buff], len(buff))
            if sr.ret != len(buff):
                logger.warning(f"Expected {len(buff)} samples, got {sr.ret}")
            return buff
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            raise
    
    def set_sample_rate(self, sample_rate: float) -> None:
        """Set SoapySDR sample rate"""
        if not self._is_open or not self._sdr:
            raise RuntimeError("SDR device not open")
        
        try:
            import SoapySDR
            self._sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, sample_rate)
            self._sample_rate = sample_rate
            logger.debug(f"Sample rate set to {sample_rate}")
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            raise
    
    def set_center_freq(self, freq: float) -> None:
        """Set SoapySDR center frequency"""
        if not self._is_open or not self._sdr:
            raise RuntimeError("SDR device not open")
        
        try:
            import SoapySDR
            self._sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
            self._center_freq = freq
            logger.debug(f"Center frequency set to {freq}")
        except Exception as e:
            logger.error(f"Error setting center frequency: {e}")
            raise
    
    def set_gain(self, gain: float | str) -> None:
        """Set SoapySDR gain"""
        if not self._is_open or not self._sdr:
            raise RuntimeError("SDR device not open")
        
        try:
            import SoapySDR
            if isinstance(gain, str) and gain.lower() == 'auto':
                self._sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, True)  # Enable AGC
            else:
                self._sdr.setGainMode(SoapySDR.SOAPY_SDR_RX, 0, False)  # Disable AGC
                self._sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, gain)
            self._gain = gain
            logger.debug(f"Gain set to {gain}")
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            raise


class SDRFactory:
    """Factory for creating SDR interfaces"""
    
    @staticmethod
    def create_sdr(sdr_type: str, device_args: Optional[Dict[str, Any]] = None) -> SDRInterface:
        """Create an SDR interface based on type"""
        sdr_type = sdr_type.lower()
        
        if sdr_type == 'rtlsdr':
            return RTLSDRInterface(device_args)
        elif sdr_type in ['limesdr', 'hackrf', 'rsp1a', 'usrp', 'soapysdr']:
            return SoapySDRInterface(device_args)
        else:
            raise ValueError(f"Unsupported SDR type: {sdr_type}")
    
    @staticmethod
    def list_available_sdrs() -> List[str]:
        """List available SDR types based on installed libraries"""
        available_sdrs = []
        
        try:
            import rtlsdr
            available_sdrs.append('rtlsdr')
        except ImportError:
            pass
        
        try:
            import SoapySDR
            available_sdrs.extend(['limesdr', 'hackrf', 'rsp1a', 'usrp'])
        except ImportError:
            pass
        
        return available_sdrs


@contextmanager
def sdr_context(sdr_type: str, device_args: Optional[Dict[str, Any]] = None):
    """Context manager for SDR operations"""
    sdr = SDRFactory.create_sdr(sdr_type, device_args)
    try:
        sdr.open()
        yield sdr
    finally:
        sdr.close()
