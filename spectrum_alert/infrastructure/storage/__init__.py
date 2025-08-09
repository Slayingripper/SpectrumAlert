"""
Data storage infrastructure for SpectrumAlert v3.0
"""

import os
import json
import csv
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from spectrum_alert.core.exceptions import StorageError
from spectrum_alert.core.domain.models import SpectrumData, AnomalyDetection, FeatureVector

logger = logging.getLogger(__name__)


class DataStorage:
    """Handles data storage operations for SpectrumAlert"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure data directories exist"""
        directories = [
            self.data_dir,
            os.path.join(self.data_dir, "spectrum"),
            os.path.join(self.data_dir, "anomalies"),
            os.path.join(self.data_dir, "features"),
            os.path.join(self.data_dir, "logs")
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_spectrum_data(self, spectrum_data: SpectrumData) -> str:
        """Save spectrum data to file with optimized storage"""
        try:
            filename = f"spectrum_{spectrum_data.timestamp.strftime('%Y%m%d_%H%M%S')}_{spectrum_data.id[:8]}.json"
            filepath = os.path.join(self.data_dir, "spectrum", filename)
            
            # Convert to serializable format with reduced precision and smaller sample storage
            data = {
                'id': spectrum_data.id,
                'timestamp': spectrum_data.timestamp.isoformat(),
                'frequency_hz': spectrum_data.frequency_hz,
                'sample_rate_hz': spectrum_data.sample_rate_hz,
                'gain_db': spectrum_data.gain_db,
                'duration_seconds': spectrum_data.duration_seconds,
                'sample_count': spectrum_data.sample_count,
                'frequency_range_hz': spectrum_data.frequency_range_hz,
                # Store only first 256 samples with reduced precision to save space
                'samples_real': [round(s.real, 4) for s in spectrum_data.samples[:256]],
                'samples_imag': [round(s.imag, 4) for s in spectrum_data.samples[:256]],
                'power_spectrum': [round(p, 4) for p in spectrum_data.power_spectrum[:256]] if spectrum_data.power_spectrum else None
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, separators=(',', ':'))  # Compact JSON format
            
            logger.debug(f"Spectrum data saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving spectrum data: {e}")
            raise StorageError(f"Failed to save spectrum data: {e}")
    
    def save_anomaly(self, anomaly: AnomalyDetection) -> str:
        """Save anomaly detection result"""
        try:
            date_str = anomaly.timestamp.strftime('%Y%m%d')
            filename = f"anomalies_{date_str}.csv"
            filepath = os.path.join(self.data_dir, "anomalies", filename)
            
            # Prepare data for CSV
            data = {
                'id': anomaly.id,
                'timestamp': anomaly.timestamp.isoformat(),
                'frequency_hz': anomaly.frequency_hz,
                'anomaly_type': anomaly.anomaly_type.value,
                'confidence_score': anomaly.confidence_score,
                'severity': anomaly.severity,
                'description': anomaly.description,
                'detection_mode': anomaly.detection_mode.value,
                'spectrum_data_id': anomaly.spectrum_data_id,
                'metadata': json.dumps(anomaly.metadata)
            }
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            
            logger.debug(f"Anomaly saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving anomaly: {e}")
            raise StorageError(f"Failed to save anomaly: {e}")
    
    def save_features(self, features: FeatureVector) -> str:
        """Save feature vector"""
        try:
            date_str = features.timestamp.strftime('%Y%m%d')
            filename = f"features_{date_str}_{features.mode.value}.csv"
            filepath = os.path.join(self.data_dir, "features", filename)
            
            # Prepare data for CSV
            data = {
                'id': features.id,
                'timestamp': features.timestamp.isoformat(),
                'spectrum_data_id': features.spectrum_data_id,
                'mode': features.mode.value,
                **features.features  # Unpack feature values
            }
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=data.keys())
                if not file_exists:
                    writer.writeheader()
                writer.writerow(data)
            
            logger.debug(f"Features saved to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving features: {e}")
            raise StorageError(f"Failed to save features: {e}")
    
    def load_training_data(self, mode: str = "full", days: int = 7) -> Optional[pd.DataFrame]:
        """Load training data from saved features"""
        try:
            feature_dir = os.path.join(self.data_dir, "features")
            
            # Find feature files for the specified mode
            files = []
            cutoff = datetime.now() - timedelta(days=days)
            for filename in os.listdir(feature_dir):
                if filename.startswith(f"features_") and filename.endswith(f"_{mode}.csv"):
                    # Optional: filter by date prefix in filename if present
                    try:
                        # filename format: features_YYYYMMDD_mode.csv
                        date_str = filename.split('_')[1]
                        file_date = datetime.strptime(date_str, '%Y%m%d')
                        if file_date < cutoff:
                            continue
                    except Exception:
                        # If parsing fails, include the file
                        pass
                    filepath = os.path.join(feature_dir, filename)
                    files.append(filepath)
            
            if not files:
                logger.warning(f"No training data files found for mode: {mode}")
                return None
            
            # Load and combine data
            dataframes = []
            for filepath in files:
                try:
                    df = pd.read_csv(filepath)
                    dataframes.append(df)
                except Exception as e:
                    logger.warning(f"Error loading {filepath}: {e}")
            
            if not dataframes:
                logger.warning("No valid training data found")
                return None
            
            combined_df = pd.concat(dataframes, ignore_index=True)
            logger.info(f"Loaded {len(combined_df)} training samples for mode: {mode}")
            return combined_df
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            raise StorageError(f"Failed to load training data: {e}")
    
    def get_anomaly_count(self, days: int = 1) -> int:
        """Get count of anomalies in the last N days"""
        try:
            anomaly_dir = os.path.join(self.data_dir, "anomalies")
            total_count = 0
            
            # Generate date strings for the last N days
            current_date = datetime.now()
            
            for i in range(days):
                date = current_date - timedelta(days=i)
                date_str = date.strftime('%Y%m%d')
                filename = f"anomalies_{date_str}.csv"
                filepath = os.path.join(anomaly_dir, filename)
                
                if os.path.exists(filepath):
                    try:
                        df = pd.read_csv(filepath)
                        total_count += len(df)
                    except Exception as e:
                        logger.warning(f"Error reading {filepath}: {e}")
            
            return total_count
        except Exception as e:
            logger.error(f"Error getting anomaly count: {e}")
            return 0

    def cleanup_old_data(self, max_age_days: int = 7) -> Dict[str, int]:
        """Remove data files older than specified days across all data folders.
        Defaults to 7 days for more aggressive cleanup.
        Returns a dict with counts of removed files per category.
        """
        removed = {"spectrum": 0, "features": 0, "anomalies": 0, "logs": 0}
        try:
            cutoff_ts = (datetime.now() - timedelta(days=max_age_days)).timestamp()
            # Map subfolders to counters key
            folders = {
                os.path.join(self.data_dir, "spectrum"): "spectrum",
                os.path.join(self.data_dir, "features"): "features", 
                os.path.join(self.data_dir, "anomalies"): "anomalies",
                os.path.join(self.data_dir, "logs"): "logs",
            }
            for folder, key in folders.items():
                if not os.path.isdir(folder):
                    continue
                for name in os.listdir(folder):
                    path = os.path.join(folder, name)
                    try:
                        if os.path.isfile(path) and os.path.getmtime(path) < cutoff_ts:
                            os.remove(path)
                            removed[key] += 1
                    except Exception as e:
                        logger.debug(f"Skip removing {path}: {e}")
            logger.info(
                f"Cleanup complete (>{max_age_days}d): "
                f"spectrum={removed['spectrum']}, features={removed['features']}, "
                f"anomalies={removed['anomalies']}, logs={removed['logs']}"
            )
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
        return removed
