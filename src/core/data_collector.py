"""
Data collection module for SpectrumAlert
"""

import time
import logging
import threading
from typing import List, Optional, Callable
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.core.sdr_interface import SDRInterface, SDRFactory, sdr_context
from src.core.feature_extraction import FeatureExtractor, FeatureSet
from src.core.data_manager import DataManager, StreamingDataWriter
from src.utils.config_manager import ConfigManager, HamBand
from src.utils.logger import log_performance

logger = logging.getLogger(__name__)


class DataCollector:
    """Collects RF data and extracts features"""
    
    def __init__(self, config_manager: ConfigManager, lite_mode: bool = False):
        self.config = config_manager
        self.lite_mode = lite_mode
        self.feature_extractor = FeatureExtractor(lite_mode)
        self.data_manager = DataManager()
        self._stop_event = threading.Event()
        self._collection_stats = {
            'samples_collected': 0,
            'features_extracted': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    @log_performance
    def collect_data(self, duration_minutes: float, 
                    output_filename: str,
                    progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        """Collect data for specified duration"""
        logger.info(f"Starting data collection for {duration_minutes} minutes")
        
        try:
            self._reset_stats()
            self._collection_stats['start_time'] = datetime.now()
            
            # Calculate sample size based on mode
            sample_size = 128 * 1024 if self.lite_mode else 256 * 1024
            
            # Create streaming writer
            writer = StreamingDataWriter(output_filename, self.data_manager)
            
            try:
                with sdr_context(self.config.general.sdr_type) as sdr:
                    self._configure_sdr(sdr)
                    
                    # Collect data using threading for better performance
                    success = self._collect_with_threading(
                        sdr, duration_minutes, sample_size, writer, progress_callback
                    )
                    
                    writer.flush()  # Ensure all data is written
                    return success
                    
            finally:
                writer.close()
                
        except Exception as e:
            logger.error(f"Error during data collection: {e}")
            return False
        finally:
            self._collection_stats['end_time'] = datetime.now()
            self._log_collection_stats()
    
    def _configure_sdr(self, sdr: SDRInterface) -> None:
        """Configure SDR parameters"""
        try:
            # Use lite mode settings if enabled
            if self.lite_mode:
                sample_rate = min(self.config.general.sample_rate, 1.024e6)
            else:
                sample_rate = self.config.general.sample_rate
            
            sdr.set_sample_rate(sample_rate)
            sdr.set_gain('auto')  # Use automatic gain control
            
            logger.info(f"SDR configured: sample_rate={sample_rate}, gain=auto")
        except Exception as e:
            logger.error(f"Error configuring SDR: {e}")
            raise
    
    def _collect_with_threading(self, sdr: SDRInterface, duration_minutes: float,
                               sample_size: int, writer: StreamingDataWriter,
                               progress_callback: Optional[Callable[[float], None]]) -> bool:
        """Collect data using threading for parallel processing"""
        duration_seconds = duration_minutes * 60
        start_time = time.time()
        
        # Use a smaller number of threads to avoid overwhelming the SDR
        max_workers = 2 if self.lite_mode else 4
        
        try:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = []
                
                while (time.time() - start_time) < duration_seconds and not self._stop_event.is_set():
                    for band in self.config.ham_bands:
                        if self._stop_event.is_set():
                            break
                        
                        # Submit band collection task
                        future = executor.submit(
                            self._collect_band_data,
                            sdr, band, sample_size, writer
                        )
                        futures.append(future)
                        
                        # Limit number of concurrent futures
                        if len(futures) >= max_workers * 2:
                            self._process_completed_futures(futures)
                    
                    # Update progress
                    if progress_callback:
                        elapsed = time.time() - start_time
                        progress = min(elapsed / duration_seconds, 1.0)
                        progress_callback(progress)
                
                # Process remaining futures
                self._process_completed_futures(futures, wait_for_all=True)
                
            return True
            
        except Exception as e:
            logger.error(f"Error in threaded collection: {e}")
            return False
    
    def _collect_band_data(self, sdr: SDRInterface, band: HamBand,
                          sample_size: int, writer: StreamingDataWriter) -> None:
        """Collect data for a single band"""
        try:
            current_freq = band.start_freq
            runs_per_freq = self.config.general.runs_per_freq
            
            while current_freq <= band.end_freq and not self._stop_event.is_set():
                # Collect multiple runs for averaging
                features_list = []
                
                for run in range(runs_per_freq):
                    if self._stop_event.is_set():
                        break
                    
                    try:
                        # Set frequency and collect samples
                        sdr.set_center_freq(current_freq)
                        iq_samples = sdr.read_samples(sample_size)
                        
                        # Extract features
                        feature_set = self.feature_extractor.extract_features(
                            iq_samples, current_freq
                        )
                        features_list.append(feature_set.features)
                        
                        self._collection_stats['samples_collected'] += 1
                        
                    except Exception as e:
                        logger.warning(f"Error collecting sample at {current_freq}: {e}")
                        self._collection_stats['errors'] += 1
                        continue
                
                # Average features across runs
                if features_list:
                    avg_features = self._average_features(features_list)
                    avg_feature_set = FeatureSet(
                        frequency=current_freq,
                        features=avg_features,
                        feature_names=self.feature_extractor.feature_names,
                        timestamp=datetime.now().isoformat()
                    )
                    
                    writer.add_features(avg_feature_set)
                    self._collection_stats['features_extracted'] += 1
                
                current_freq += self.config.general.freq_step
                
        except Exception as e:
            logger.error(f"Error collecting band data: {e}")
            self._collection_stats['errors'] += 1
    
    def _average_features(self, features_list: List[List[float]]) -> List[float]:
        """Average features across multiple runs"""
        if not features_list:
            return []
        
        import numpy as np
        features_array = np.array(features_list)
        return np.mean(features_array, axis=0).tolist()
    
    def _process_completed_futures(self, futures: List, wait_for_all: bool = False):
        """Process completed futures and remove them from the list"""
        if wait_for_all:
            # Wait for all futures to complete
            for future in as_completed(futures):
                try:
                    future.result()  # This will raise any exceptions
                except Exception as e:
                    logger.error(f"Future completed with error: {e}")
            futures.clear()
        else:
            # Process only completed futures
            completed = []
            for i, future in enumerate(futures):
                if future.done():
                    completed.append(i)
                    try:
                        future.result()
                    except Exception as e:
                        logger.error(f"Future completed with error: {e}")
            
            # Remove completed futures in reverse order to maintain indices
            for i in reversed(completed):
                futures.pop(i)
    
    def stop_collection(self) -> None:
        """Stop data collection"""
        logger.info("Stopping data collection...")
        self._stop_event.set()
    
    def _reset_stats(self) -> None:
        """Reset collection statistics"""
        self._stop_event.clear()
        self._collection_stats = {
            'samples_collected': 0,
            'features_extracted': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    def _log_collection_stats(self) -> None:
        """Log collection statistics"""
        stats = self._collection_stats
        if stats['start_time'] and stats['end_time']:
            duration = stats['end_time'] - stats['start_time']
            logger.info(f"Collection completed in {duration}")
            logger.info(f"Samples collected: {stats['samples_collected']}")
            logger.info(f"Features extracted: {stats['features_extracted']}")
            logger.info(f"Errors encountered: {stats['errors']}")
            
            if duration.total_seconds() > 0:
                rate = stats['samples_collected'] / duration.total_seconds()
                logger.info(f"Collection rate: {rate:.2f} samples/second")
    
    def get_collection_stats(self) -> dict:
        """Get current collection statistics"""
        return self._collection_stats.copy()


class BandScanner:
    """Scans specific frequency bands"""
    
    def __init__(self, config_manager: ConfigManager, lite_mode: bool = False):
        self.config = config_manager
        self.lite_mode = lite_mode
        self.feature_extractor = FeatureExtractor(lite_mode)
    
    def scan_frequency(self, sdr: SDRInterface, frequency: float,
                      num_samples: Optional[int] = None) -> Optional[FeatureSet]:
        """Scan a single frequency"""
        if num_samples is None:
            num_samples = 128 * 1024 if self.lite_mode else 256 * 1024
        
        try:
            sdr.set_center_freq(frequency)
            iq_samples = sdr.read_samples(num_samples)
            
            feature_set = self.feature_extractor.extract_features(iq_samples, frequency)
            feature_set.timestamp = datetime.now().isoformat()
            
            return feature_set
            
        except Exception as e:
            logger.error(f"Error scanning frequency {frequency}: {e}")
            return None
    
    def scan_band(self, sdr: SDRInterface, band: HamBand,
                 step_size: Optional[float] = None) -> List[FeatureSet]:
        """Scan an entire frequency band"""
        if step_size is None:
            step_size = self.config.general.freq_step
        
        features = []
        current_freq = band.start_freq
        
        while current_freq <= band.end_freq:
            feature_set = self.scan_frequency(sdr, current_freq)
            if feature_set:
                features.append(feature_set)
            current_freq += step_size
        
        logger.info(f"Scanned {len(features)} frequencies in band {band.start_freq}-{band.end_freq}")
        return features
    
    def quick_scan(self, sdr: SDRInterface, frequencies: List[float]) -> List[FeatureSet]:
        """Quick scan of specific frequencies"""
        features = []
        
        for freq in frequencies:
            feature_set = self.scan_frequency(sdr, freq)
            if feature_set:
                features.append(feature_set)
        
        logger.info(f"Quick scan completed for {len(features)} frequencies")
        return features


class CollectionScheduler:
    """Schedules automatic data collection"""
    
    def __init__(self, data_collector: DataCollector):
        self.collector = data_collector
        self._scheduler_thread = None
        self._stop_event = threading.Event()
        self._is_running = False
    
    def schedule_collection(self, interval_hours: float, duration_minutes: float,
                          output_prefix: str = "scheduled_data") -> None:
        """Schedule automatic data collection"""
        if self._is_running:
            logger.warning("Scheduler already running")
            return
        
        logger.info(f"Scheduling collection every {interval_hours} hours for {duration_minutes} minutes")
        
        self._scheduler_thread = threading.Thread(
            target=self._run_scheduler,
            args=(interval_hours, duration_minutes, output_prefix),
            daemon=True
        )
        self._is_running = True
        self._scheduler_thread.start()
    
    def _run_scheduler(self, interval_hours: float, duration_minutes: float,
                      output_prefix: str) -> None:
        """Run the collection scheduler"""
        interval_seconds = interval_hours * 3600
        
        while not self._stop_event.is_set():
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{output_prefix}_{timestamp}.csv"
            
            logger.info(f"Starting scheduled collection: {filename}")
            
            try:
                success = self.collector.collect_data(duration_minutes, filename)
                if success:
                    logger.info(f"Scheduled collection completed: {filename}")
                else:
                    logger.error(f"Scheduled collection failed: {filename}")
            except Exception as e:
                logger.error(f"Error in scheduled collection: {e}")
            
            # Wait for next collection
            self._stop_event.wait(interval_seconds)
        
        self._is_running = False
    
    def stop_scheduler(self) -> None:
        """Stop the collection scheduler"""
        if self._is_running:
            logger.info("Stopping collection scheduler...")
            self._stop_event.set()
            if self._scheduler_thread:
                self._scheduler_thread.join(timeout=10)
            self._is_running = False
    
    def is_running(self) -> bool:
        """Check if scheduler is running"""
        return self._is_running
