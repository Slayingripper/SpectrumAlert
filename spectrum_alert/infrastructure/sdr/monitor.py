"""
Real-time spectrum monitoring system
"""

import time
import signal
import logging
from typing import Optional, Dict, Any, List, Callable
from threading import Thread

from .base import SDRInterface

logger = logging.getLogger(__name__)


class SpectrumMonitor:
    """Real-time spectrum monitoring system"""
    
    def __init__(self, sdr_interface: SDRInterface):
        self.sdr = sdr_interface
        self.is_running = False
        self._should_stop = False
        self._monitor_thread = None
        self._latest_data = None
        self._callbacks: List[Callable] = []
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, stopping monitor...")
        self.stop()
    
    def add_callback(self, callback: Callable):
        """Add a callback function to receive spectrum data"""
        self._callbacks.append(callback)
    
    def start(self, sample_rate: float = 2.048e6, num_samples: int = 8192, update_interval: float = 0.1):
        """Start spectrum monitoring"""
        if self.is_running:
            logger.warning("Monitor is already running")
            return
        
        try:
            # Configure SDR
            self.sdr.set_sample_rate(sample_rate)
            
            self.is_running = True
            self._should_stop = False
            
            logger.info(f"Starting spectrum monitor (rate: {sample_rate/1e6:.2f} MHz, samples: {num_samples})")
            
            # Start monitoring loop
            self._monitor_loop(num_samples, update_interval)
            
        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            self.is_running = False
            raise
    
    def _monitor_loop(self, num_samples: int, update_interval: float):
        """Main monitoring loop"""
        try:
            while not self._should_stop:
                start_time = time.time()
                
                try:
                    # Read samples from SDR
                    samples = self.sdr.read_samples(num_samples)
                    
                    # Store latest data
                    self._latest_data = {
                        'samples': samples,
                        'timestamp': start_time,
                        'sample_rate': self.sdr.sample_rate,
                        'center_freq': self.sdr.center_freq,
                        'gain': self.sdr.gain
                    }
                    
                    # Call registered callbacks
                    for callback in self._callbacks:
                        try:
                            callback(self._latest_data)
                        except Exception as e:
                            logger.error(f"Callback error: {e}")
                    
                    # Rate limiting
                    elapsed = time.time() - start_time
                    if elapsed < update_interval:
                        time.sleep(update_interval - elapsed)
                        
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(1.0)  # Wait before retrying
                    
        except KeyboardInterrupt:
            logger.info("Monitoring interrupted by user")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop spectrum monitoring"""
        if not self.is_running:
            return
        
        logger.info("Stopping spectrum monitor...")
        self._should_stop = True
        
        # Wait for monitor to stop
        timeout = 5.0
        start_time = time.time()
        while self.is_running and (time.time() - start_time) < timeout:
            time.sleep(0.1)
        
        if self.is_running:
            logger.warning("Monitor did not stop gracefully")
        
        logger.info("Spectrum monitor stopped")
    
    def get_latest_data(self) -> Optional[Dict[str, Any]]:
        """Get the latest spectrum data"""
        return self._latest_data
