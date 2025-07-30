"""
Robust RTL-SDR data collector with comprehensive error handling
"""

import logging
import numpy as np
import time
import csv
import os
from typing import List, Tuple, Optional
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class SafeRTLSDR:
    """Safe RTL-SDR wrapper with error handling and recovery"""
    
    def __init__(self, device_index: int = 0):
        self.device_index = device_index
        self._sdr = None
        self._is_open = False
        self._lock = threading.Lock()
        self._consecutive_errors = 0
        self._max_consecutive_errors = 3
        
    def _reset_device(self):
        """Reset the RTL-SDR device"""
        if self._sdr:
            try:
                self._sdr.close()
            except:
                pass
        self._sdr = None
        self._is_open = False
        time.sleep(1)  # Give device time to reset
        
    def open(self, sample_rate: float = 2.048e6, gain: str = 'auto') -> bool:
        """Open RTL-SDR device with error handling"""
        with self._lock:
            try:
                if self._is_open and self._sdr:
                    return True
                
                from rtlsdr import RtlSdr
                
                # Reset if we had consecutive errors
                if self._consecutive_errors >= self._max_consecutive_errors:
                    logger.warning("Too many consecutive errors, resetting device")
                    self._reset_device()
                    self._consecutive_errors = 0
                
                self._sdr = RtlSdr(device_index=self.device_index)
                
                # Set safe defaults first
                self._sdr.sample_rate = sample_rate
                self._sdr.center_freq = 100e6  # Safe frequency
                self._sdr.gain = gain
                
                self._is_open = True
                self._consecutive_errors = 0
                logger.info(f"RTL-SDR device {self.device_index} opened successfully")
                return True
                
            except Exception as e:
                self._consecutive_errors += 1
                logger.error(f"Failed to open RTL-SDR: {e}")
                self._reset_device()
                return False
    
    def close(self):
        """Close RTL-SDR device safely"""
        with self._lock:
            if self._sdr and self._is_open:
                try:
                    self._sdr.close()
                    logger.info("RTL-SDR device closed")
                except Exception as e:
                    logger.error(f"Error closing RTL-SDR: {e}")
                finally:
                    self._sdr = None
                    self._is_open = False
    
    def set_frequency(self, freq: float) -> bool:
        """Set center frequency with error handling"""
        with self._lock:
            if not self._is_open or not self._sdr:
                return False
            
            try:
                # Validate frequency
                if not (24e6 <= freq <= 1766e6):
                    logger.warning(f"Frequency {freq} may be outside RTL-SDR range")
                
                self._sdr.center_freq = freq
                # Small delay to let frequency settle
                time.sleep(0.01)
                return True
                
            except Exception as e:
                self._consecutive_errors += 1
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ['r82xx', 'i2c', 'usb']):
                    logger.error(f"Hardware communication error: {e}")
                    # This is likely a hardware issue, need to reset
                    self._reset_device()
                else:
                    logger.error(f"Error setting frequency: {e}")
                
                return False
    
    def read_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Read samples with error handling"""
        with self._lock:
            if not self._is_open or not self._sdr:
                return None
            
            try:
                # Validate sample count
                if num_samples <= 0 or num_samples > 16*1024*1024:
                    logger.error(f"Invalid sample count: {num_samples}")
                    return None
                
                samples = self._sdr.read_samples(num_samples)
                
                if samples is None or len(samples) == 0:
                    logger.error("No samples received from SDR")
                    self._consecutive_errors += 1
                    return None
                
                self._consecutive_errors = 0  # Reset error counter on success
                return np.array(samples, dtype=np.complex64)
                
            except Exception as e:
                self._consecutive_errors += 1
                error_msg = str(e).lower()
                
                if any(keyword in error_msg for keyword in ['usb', 'i2c', 'timeout', 'r82xx']):
                    logger.error(f"Hardware communication error: {e}")
                    # Critical error - reset device
                    self._reset_device()
                else:
                    logger.error(f"Error reading samples: {e}")
                
                return None
    
    @contextmanager
    def safe_operation(self):
        """Context manager for safe operations"""
        try:
            yield self
        except KeyboardInterrupt:
            logger.info("Operation interrupted by user")
            raise
        except Exception as e:
            logger.error(f"Operation failed: {e}")
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._max_consecutive_errors:
                logger.warning("Too many errors, resetting device")
                self._reset_device()
            raise


class RobustDataCollector:
    """Robust data collector with comprehensive error handling"""
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.sdr = SafeRTLSDR(device_index=0)
        self._collection_active = False
        self._stop_collection = threading.Event()
        
    def extract_features(self, iq_data: np.ndarray, lite_mode: bool = False) -> List[float]:
        """Extract features from IQ data"""
        try:
            if len(iq_data) == 0:
                return []
            
            I = np.real(iq_data)
            Q = np.imag(iq_data)
            amplitude = np.sqrt(I**2 + Q**2)
            
            if lite_mode:
                # Basic features only
                mean_amplitude = np.mean(amplitude)
                std_amplitude = np.std(amplitude)
                return [mean_amplitude, std_amplitude]
            else:
                # Full feature set
                phase = np.unwrap(np.angle(iq_data))
                fft_values = np.fft.fft(iq_data)
                fft_magnitude = np.abs(fft_values)
                
                # Calculate all features
                mean_amplitude = np.mean(amplitude)
                std_amplitude = np.std(amplitude)
                mean_fft_magnitude = np.mean(fft_magnitude)
                std_fft_magnitude = np.std(fft_magnitude)
                
                # Handle edge cases
                if std_amplitude > 0:
                    skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
                    kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
                else:
                    skew_amplitude = 0
                    kurt_amplitude = 0
                
                # Phase statistics
                std_phase = np.std(phase)
                mean_phase = np.mean(phase)
                if std_phase > 0:
                    skew_phase = np.mean((phase - mean_phase) ** 3) / (std_phase ** 3)
                    kurt_phase = np.mean((phase - mean_phase) ** 4) / (std_phase ** 4)
                else:
                    skew_phase = 0
                    kurt_phase = 0
                
                # Additional features
                if len(amplitude) > 1:
                    cyclo_autocorr = np.abs(np.correlate(amplitude, amplitude, mode='full')[len(amplitude) // 2:]).mean()
                else:
                    cyclo_autocorr = 0
                
                # Spectral entropy
                fft_magnitude_sum = np.sum(fft_magnitude)
                if fft_magnitude_sum > 0:
                    normalized_fft = fft_magnitude / fft_magnitude_sum
                    spectral_entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))
                else:
                    spectral_entropy = 0
                
                # PAPR
                if mean_amplitude > 0:
                    papr = np.max(amplitude) ** 2 / np.mean(amplitude ** 2)
                else:
                    papr = 0
                
                # Band energy ratio
                fft_magnitude_half = fft_magnitude[:len(fft_magnitude) // 2]
                if fft_magnitude_sum > 0:
                    band_energy_ratio = np.sum(fft_magnitude_half) / fft_magnitude_sum
                else:
                    band_energy_ratio = 0
                
                return [
                    mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
                    skew_amplitude, kurt_amplitude, skew_phase, kurt_phase, cyclo_autocorr,
                    spectral_entropy, papr, band_energy_ratio
                ]
                
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            # Return zeros for the expected number of features
            if lite_mode:
                return [0.0, 0.0]
            else:
                return [0.0] * 12
    
    def collect_data(self, duration_minutes: float, output_filename: str, lite_mode: bool = False) -> bool:
        """Collect RF data with robust error handling"""
        logger.info(f"Starting data collection for {duration_minutes} minutes (lite_mode={lite_mode})")
        
        # Prepare output file
        os.makedirs(os.path.dirname(output_filename) if os.path.dirname(output_filename) else '.', exist_ok=True)
        
        # Open SDR
        sample_rate = 1.024e6 if lite_mode else 2.048e6
        if not self.sdr.open(sample_rate=sample_rate):
            logger.error("Failed to open RTL-SDR device")
            return False
        
        try:
            ham_bands = self.config.ham_bands
            freq_step = self.config.general.freq_step
            runs_per_freq = 3 if lite_mode else self.config.general.runs_per_freq
            sample_size = 128 * 1024 if lite_mode else 256 * 1024
            
            start_time = time.time()
            duration_seconds = duration_minutes * 60
            
            header_written = False
            samples_collected = 0
            total_frequencies = sum((band.end_freq - band.start_freq) / freq_step + 1 for band in ham_bands)
            
            with open(output_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                while time.time() - start_time < duration_seconds and not self._stop_collection.is_set():
                    for band in ham_bands:
                        current_freq = band.start_freq
                        
                        while current_freq <= band.end_freq and not self._stop_collection.is_set():
                            if not self.sdr.set_frequency(current_freq):
                                logger.warning(f"Failed to set frequency {current_freq}, skipping")
                                current_freq += freq_step
                                continue
                            
                            # Collect multiple samples per frequency
                            run_features = []
                            for run in range(runs_per_freq):
                                try:
                                    samples = self.sdr.read_samples(sample_size)
                                    if samples is not None:
                                        features = self.extract_features(samples, lite_mode)
                                        if features:
                                            run_features.append(features)
                                    
                                    # Small delay between runs
                                    time.sleep(0.01)
                                    
                                except Exception as e:
                                    logger.warning(f"Error in run {run} at {current_freq}: {e}")
                                    continue
                            
                            if run_features:
                                # Average features across runs
                                avg_features = np.mean(run_features, axis=0)
                                
                                # Write header if needed
                                if not header_written:
                                    if lite_mode:
                                        header = ['Frequency', 'Mean_Amplitude', 'Std_Amplitude']
                                    else:
                                        header = ['Frequency', 'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 
                                                'Std_FFT_Magnitude', 'Skew_Amplitude', 'Kurt_Amplitude', 'Skew_Phase', 
                                                'Kurt_Phase', 'Cyclo_Autocorr', 'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio']
                                    writer.writerow(header)
                                    header_written = True
                                
                                # Write data
                                row = [current_freq] + avg_features.tolist()
                                writer.writerow(row)
                                samples_collected += 1
                                
                                # Progress update
                                progress = (samples_collected / total_frequencies) * 100
                                if samples_collected % 10 == 0:
                                    logger.info(f"Progress: {progress:.1f}% - Collected {samples_collected} samples")
                            
                            current_freq += freq_step
                            
                            # Check for timeout
                            if time.time() - start_time >= duration_seconds:
                                break
            
            logger.info(f"Data collection completed. Collected {samples_collected} samples in {output_filename}")
            return True
            
        except KeyboardInterrupt:
            logger.info("Data collection interrupted by user")
            return False
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            return False
        finally:
            self._stop_collection.clear()
            self.sdr.close()
    
    def stop_collection(self):
        """Stop data collection"""
        self._stop_collection.set()
        logger.info("Stop signal sent to data collector")
