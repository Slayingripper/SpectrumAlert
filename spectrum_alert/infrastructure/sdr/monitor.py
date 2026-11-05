"""
Real-time spectrum monitoring system
"""

import time
import logging
import threading
from typing import Optional, Dict, Any, List, Callable
from threading import Thread, Event

from .base import SDRInterface

logger = logging.getLogger(__name__)


class SpectrumMonitor:
    """Real-time spectrum monitoring system.

    Runs the capture loop on a background thread. The previous version imported
    ``Thread`` and created ``_monitor_thread`` but then called the loop
    synchronously from ``start()``, so ``start()`` never returned until the loop
    finished and nothing was actually concurrent. Signal handlers are also no
    longer registered here: ``signal.signal`` only works on the main thread, so
    constructing a monitor from a worker raised ``ValueError``.
    """

    def __init__(self, sdr_interface: SDRInterface, close_on_stop: bool = False):
        self.sdr = sdr_interface
        self.is_running = False
        self._should_stop = Event()
        self._monitor_thread: Optional[Thread] = None
        self._latest_data = None
        self._callbacks: List[Callable] = []
        self._lock = threading.RLock()
        # Leave the device open by default: a monitor is often the long-lived
        # owner of the dongle, and closing on stop made restart impossible.
        self.close_on_stop = close_on_stop

    def add_callback(self, callback: Callable):
        """Add a callback function to receive spectrum data"""
        self._callbacks.append(callback)

    def start(
        self,
        sample_rate: float = 2.048e6,
        num_samples: int = 8192,
        update_interval: float = 0.1,
        center_freq: Optional[float] = None,
        gain=None,
        background: bool = True,
    ):
        """Start spectrum monitoring.

        Returns a thread handle when ``background`` is true (the default), else
        ``None`` after running to completion synchronously.
        """
        if self.is_running:
            logger.warning("Monitor is already running")
            return self._monitor_thread

        try:
            self.sdr.set_sample_rate(sample_rate)
            if center_freq is not None:
                self.sdr.set_center_freq(center_freq)
            if gain is not None:
                self.sdr.set_gain(gain)
        except Exception as e:
            logger.error(f"Failed to configure SDR: {e}")
            raise

        self.is_running = True
        self._should_stop.clear()

        logger.info(
            f"Starting spectrum monitor (rate: {sample_rate/1e6:.2f} MHz, "
            f"samples: {num_samples}, interval: {update_interval}s)"
        )

        if not background:
            self._monitor_loop(num_samples, update_interval)
            return None

        self._monitor_thread = Thread(
            target=self._monitor_loop,
            args=(num_samples, update_interval),
            name="spectrum-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        return self._monitor_thread

    def _monitor_loop(self, num_samples: int, update_interval: float):
        """Main monitoring loop"""
        try:
            while not self._should_stop.is_set():
                start_time = time.time()

                try:
                    samples = self.sdr.read_samples(num_samples)

                    frame = {
                        'samples': samples,
                        'timestamp': start_time,
                        'sample_rate': self.sdr.sample_rate,
                        'center_freq': self.sdr.center_freq,
                        'gain': self.sdr.gain
                    }
                    with self._lock:
                        self._latest_data = frame

                    for callback in list(self._callbacks):
                        try:
                            callback(frame)
                        except Exception as e:
                            # One bad callback must not kill the capture loop.
                            logger.error(f"Callback error: {e}")

                    elapsed = time.time() - start_time
                    if elapsed < update_interval:
                        # Event.wait rather than sleep so stop() takes effect
                        # immediately instead of after a full interval.
                        self._should_stop.wait(update_interval - elapsed)

                except Exception as e:
                    if self._should_stop.is_set():
                        break
                    logger.error(f"Error in monitoring loop: {e}")
                    self._should_stop.wait(1.0)

        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        finally:
            self.is_running = False
            if self.close_on_stop:
                try:
                    self.sdr.close()
                except Exception as e:  # pragma: no cover - best effort
                    logger.warning(f"Error closing SDR on stop: {e}")

    def stop(self, timeout: float = 5.0):
        """Stop spectrum monitoring"""
        if not self.is_running:
            self._should_stop.set()
            return

        logger.info("Stopping spectrum monitor...")
        self._should_stop.set()

        thread = self._monitor_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=timeout)
            if thread.is_alive():
                logger.warning("Monitor thread did not stop within %.1fs", timeout)

        self.is_running = False
        logger.info("Spectrum monitor stopped")

    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest spectrum data"""
        with self._lock:
            return self._latest_data