"""
RTL-SDR implementation of SDR interface
"""

import logging
import threading
import numpy as np
from typing import Optional, Dict, Any, Union

from .base import SDRInterface
from spectrum_alert.core.exceptions import SDRError

logger = logging.getLogger(__name__)


class RTLSDRInterface(SDRInterface):
    """RTL-SDR implementation of SDR interface"""

    # One libusb handle per dongle, shared by every instance. stop_monitoring
    # cancels the loop while a read may still be running in a worker thread
    # and then closes the device; without serialization libusb aborted with
    # "refcnt >= 2" and took the whole server down. Class level (not per
    # instance) so a workspace scan opening the same dongle queues instead of
    # interleaving USB traffic. RLock: open() calls close() on re-open.
    _device_lock = threading.RLock()

    def __init__(self, device_index: int = 0, device_args: Optional[Dict[str, Any]] = None):
        super().__init__(device_args)
        self.device_index = device_index
        self.sdr = None
        self._bandwidth = None
        self._freq_correction = None

    def open(self) -> None:
        """Open the RTL-SDR device"""
        try:
            from rtlsdr import RtlSdr
        except ImportError as e:
            raise SDRError(
                "pyrtlsdr is not installed. Install with: pip install pyrtlsdr"
            ) from e

        with self._device_lock:
            if self.sdr is not None:
                self.close()

            try:
                self.sdr = RtlSdr(device_index=self.device_index)
            except Exception as e:
                self.sdr = None
                self._is_open = False
                logger.error(f"Failed to open RTL-SDR device {self.device_index}: {e}")
                raise SDRError(f"Failed to open RTL-SDR: {e}") from e

            self._is_open = True
            try:
                identity = (
                    f"{getattr(self.sdr, 'vendor_name', '?')}/"
                    f"{getattr(self.sdr, 'product_name', '?')} "
                    f"tuner={getattr(self.sdr, 'tuner', '?')} "
                    f"serial={getattr(self.sdr, 'serial_number', '?')}"
                )
                logger.info(f"RTL-SDR device {self.device_index} opened: {identity}")
            except Exception:
                logger.info(f"RTL-SDR device {self.device_index} opened")

    def close(self) -> None:
        """Close the RTL-SDR device"""
        with self._device_lock:
            if self.sdr:
                try:
                    self.sdr.close()
                    logger.info("RTL-SDR device closed")
                except Exception as e:
                    logger.error(f"Error closing RTL-SDR: {e}")
                finally:
                    # Clear state regardless of whether close() raised. Leaving
                    # _is_open True after a failed close makes every later operation
                    # believe the device is usable when it is not.
                    self.sdr = None
                    self._is_open = False
                    self._gain = None
                    self._gain_requested = None

    def read_samples(self, num_samples: int) -> np.ndarray:
        """Read IQ samples from RTL-SDR"""
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")

            try:
                return self.sdr.read_samples(num_samples)
            except Exception as e:
                logger.error(f"Error reading samples: {e}")
                raise SDRError(f"Failed to read samples: {e}") from e

    def set_sample_rate(self, sample_rate: float) -> None:
        """Set the sample rate"""
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")

            try:
                self.sdr.sample_rate = sample_rate
                # Read back rather than caching the request: the tuner snaps to the
                # nearest rate it supports, so 2.048 MSps is really ~2.0 MSps.
                # Reporting the requested value makes every reported frequency axis
                # subtly wrong.
                self._sample_rate = float(self.sdr.sample_rate)
                logger.debug(f"Sample rate set to {sample_rate} Hz (actual {self._sample_rate} Hz)")
            except Exception as e:
                logger.error(f"Error setting sample rate: {e}")
                raise SDRError(f"Failed to set sample rate: {e}") from e

    def set_bandwidth(self, bandwidth: float) -> None:
        """Set the analog front-end bandwidth.

        Separated from the sample rate: the sample rate sets the digitised span,
        the bandwidth sets the analog filter. Leaving the analog filter wide while
        digitising narrow is what lets strong out-of-band signals fold into the
        passband and masquerade as anomalies.
        """
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")
            if bandwidth <= 0:
                raise ValueError("bandwidth must be positive")
            try:
                self.sdr.bandwidth = bandwidth
                self._bandwidth = float(self.sdr.bandwidth)
                logger.debug(f"Bandwidth set to {bandwidth} Hz (actual {self._bandwidth} Hz)")
            except AttributeError:
                logger.debug("this pyrtlsdr build does not expose bandwidth control")
            except Exception as e:
                logger.error(f"Error setting bandwidth: {e}")
                raise SDRError(f"Failed to set bandwidth: {e}") from e

    def set_freq_correction(self, ppm_error: int) -> None:
        """Set the crystal error correction in ppm.

        Without this, absolute frequency accuracy is the dongle's factory
        calibration, commonly +/-20-100 ppm. At 145 MHz that is several kHz of
        error, which is more than enough to miss a narrow channel entirely.
        """
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")
            try:
                self.sdr.freq_correction = ppm_error
                self._freq_correction = int(self.sdr.freq_correction)
                logger.debug(f"Frequency correction set to {ppm_error} ppm")
            except AttributeError:
                logger.debug("this pyrtlsdr build does not expose freq_correction")
            except Exception as e:
                logger.error(f"Error setting frequency correction: {e}")
                raise SDRError(f"Failed to set frequency correction: {e}") from e

    def set_center_freq(self, frequency: float) -> None:
        """Set the center frequency"""
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")

            try:
                self.sdr.center_freq = frequency
                self._center_freq = frequency
                logger.debug(f"Center frequency set to {frequency} Hz")
            except Exception as e:
                logger.error(f"Error setting center frequency: {e}")
                raise SDRError(f"Failed to set center frequency: {e}") from e

    def set_gain(self, gain: Union[float, str]) -> None:
        """Set the gain"""
        with self._device_lock:
            if not self.sdr:
                raise SDRError("SDR device not open")

            # Idempotent against the *request*, not against what the tuner
            # reports. The R820T snaps a request to the nearest supported step,
            # so asking for 20.0 dB comes back as 19.7. A caller that compared
            # the two re-sent the gain on every frame, and re-programming the
            # tuner straight after a read eventually tripped libusb's I2C
            # bridge: "rtlsdr_demod_write_reg failed with -9 ... Could not get
            # gain mode", which killed the monitoring loop on its second frame.
            if self._gain_requested is not None and self._gain_requested == gain:
                return

            try:
                if gain == 'auto':
                    self.sdr.gain = 'auto'
                    # pyrtlsdr's 'auto' means "disable manual gain", which leaves the
                    # tuner near 0 dB. It is not AGC, and the RTL2832 does not really
                    # have one. Read back the real value so reports are honest.
                    self._gain = self._read_gain()
                else:
                    self.sdr.gain = float(gain)
                    self._gain = self._read_gain()
                self._gain_requested = gain
                logger.debug(f"Gain set to {gain} (actual {self._gain})")
            except Exception as e:
                logger.error(f"Error setting gain: {e}")
                raise SDRError(f"Failed to set gain: {e}") from e

    def _read_gain(self) -> Optional[float]:
        """Read the gain the tuner actually applied, if it will tell us."""
        try:
            value = self.sdr.gain
        except Exception:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return value

    @property
    def bandwidth(self) -> Optional[float]:
        """Current analog bandwidth in Hz, if known."""
        return self._bandwidth

    @property
    def freq_correction(self) -> Optional[int]:
        """Current crystal correction in ppm, if set."""
        return self._freq_correction

    def info(self) -> dict:
        """Hardware identity and current configuration."""
        if not self.sdr:
            return {"open": False, "device_index": self.device_index}
        return {
            "open": self._is_open,
            "device_index": self.device_index,
            "manufacturer": getattr(self.sdr, "manufacturer", None),
            "product": getattr(self.sdr, "product", None),
            "serial": getattr(self.sdr, "serial_number", None),
            "tuner": getattr(self.sdr, "tuner", None),
            "tuner_gain_count": getattr(self.sdr, "tuner_gain_count", None),
            "vendor_name": getattr(self.sdr, "vendor_name", None),
            "product_name": getattr(self.sdr, "product_name", None),
            "center_freq": self._center_freq,
            "sample_rate": self._sample_rate,
            "gain": self._gain,
            "bandwidth": self._bandwidth,
            "freq_correction": self._freq_correction,
        }
