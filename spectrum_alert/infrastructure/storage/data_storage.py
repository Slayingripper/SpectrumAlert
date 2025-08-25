"""
Base data storage classes and interfaces
"""

import os
import json
import csv
import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from datetime import datetime
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
                'i_samples': [float(x.real) for x in spectrum_data.samples[:1000]],  # Limit samples
                'q_samples': [float(x.imag) for x in spectrum_data.samples[:1000]],
                'metadata': {}  # Remove metadata field for now
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=None, separators=(',', ':'))
            
            logger.debug(f"Saved spectrum data to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving spectrum data: {e}")
            raise StorageError(f"Failed to save spectrum data: {e}")
    
    def load_spectrum_data(self, filepath: str) -> SpectrumData:
        """Load spectrum data from file"""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Reconstruct complex samples
            i_samples = data['i_samples']
            q_samples = data['q_samples']
            iq_samples = [complex(i, q) for i, q in zip(i_samples, q_samples)]
            
            return SpectrumData(
                id=data['id'],
                timestamp=datetime.fromisoformat(data['timestamp']),
                frequency_hz=data['frequency_hz'],
                sample_rate_hz=data['sample_rate_hz'],
                gain_db=data['gain_db'],
                samples=iq_samples,
                power_spectrum=None,
                duration_seconds=len(iq_samples) / data['sample_rate_hz']
            )
        except Exception as e:
            logger.error(f"Error loading spectrum data from {filepath}: {e}")
            raise StorageError(f"Failed to load spectrum data: {e}")
    
    def save_anomaly(self, anomaly: AnomalyDetection) -> str:
        """Save anomaly detection result to CSV file"""
        try:
            # Create daily anomaly file
            date_str = anomaly.timestamp.strftime('%Y%m%d')
            filename = f"anomalies_{date_str}.csv"
            filepath = os.path.join(self.data_dir, "anomalies", filename)
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            with open(filepath, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'id', 'timestamp', 'frequency_hz', 'anomaly_type', 
                    'confidence_score', 'severity', 'description', 
                    'detection_mode', 'spectrum_data_id', 'metadata'
                ])
                
                if not file_exists:
                    writer.writeheader()
                
                writer.writerow({
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
                })
            
            logger.debug(f"Saved anomaly to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving anomaly: {e}")
            raise StorageError(f"Failed to save anomaly: {e}")
    
    def get_anomalies_since(self, since: datetime) -> List[AnomalyDetection]:
        """Get anomalies since a specific datetime"""
        try:
            anomalies = []
            anomalies_dir = os.path.join(self.data_dir, "anomalies")
            
            if not os.path.exists(anomalies_dir):
                return anomalies
            
            # Get all anomaly CSV files
            for filename in os.listdir(anomalies_dir):
                if filename.startswith("anomalies_") and filename.endswith(".csv"):
                    filepath = os.path.join(anomalies_dir, filename)
                    
                    try:
                        df = pd.read_csv(filepath)
                        
                        for _, row in df.iterrows():
                            timestamp = datetime.fromisoformat(row['timestamp'])
                            
                            # Filter by time
                            if timestamp >= since:
                                from spectrum_alert.core.domain.models import AnomalyType, DetectionMode
                                
                                anomaly = AnomalyDetection(
                                    id=row['id'],
                                    timestamp=timestamp,
                                    frequency_hz=row['frequency_hz'],
                                    anomaly_type=AnomalyType(row['anomaly_type']),
                                    confidence_score=row['confidence_score'],
                                    severity=row['severity'],
                                    description=row['description'],
                                    detection_mode=DetectionMode(row['detection_mode']),
                                    spectrum_data_id=row['spectrum_data_id'],
                                    metadata=json.loads(row['metadata']) if pd.notna(row['metadata']) else {}
                                )
                                anomalies.append(anomaly)
                    
                    except Exception as e:
                        logger.error(f"Error reading anomaly file {filepath}: {e}")
                        continue
            
            # Sort by timestamp (newest first)
            anomalies.sort(key=lambda x: x.timestamp, reverse=True)
            return anomalies
            
        except Exception as e:
            logger.error(f"Error getting anomalies: {e}")
            raise StorageError(f"Failed to get anomalies: {e}")
    
    def save_feature_vector(self, feature_vector: FeatureVector) -> str:
        """Save feature vector to CSV file"""
        try:
            # Create daily feature file
            date_str = feature_vector.timestamp.strftime('%Y%m%d')
            filename = f"features_{date_str}.csv"
            filepath = os.path.join(self.data_dir, "features", filename)
            
            # Check if file exists to determine if we need headers
            file_exists = os.path.exists(filepath)
            
            # Prepare features dict
            features_dict = {
                'id': feature_vector.id,
                'timestamp': feature_vector.timestamp.isoformat(),
                'spectrum_data_id': feature_vector.spectrum_data_id,
                **feature_vector.features  # Unpack features dict
            }
            
            with open(filepath, 'a', newline='') as f:
                if not file_exists:
                    # Write header
                    fieldnames = list(features_dict.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(features_dict)
                else:
                    # Read existing fieldnames
                    with open(filepath, 'r') as read_f:
                        reader = csv.DictReader(read_f)
                        existing_fieldnames = reader.fieldnames or []
                    
                    # Merge fieldnames
                    all_fieldnames = list(dict.fromkeys(list(existing_fieldnames) + list(features_dict.keys())))
                    
                    # If new fields were added, rewrite the file
                    if len(all_fieldnames) > len(existing_fieldnames):
                        # Read all existing data
                        df = pd.read_csv(filepath)
                        
                        # Add new row
                        new_row = pd.DataFrame([features_dict])
                        df = pd.concat([df, new_row], ignore_index=True)
                        
                        # Write back with all columns
                        df.to_csv(filepath, index=False)
                    else:
                        # Simple append
                        writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                        writer.writerow(features_dict)
            
            logger.debug(f"Saved feature vector to {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"Error saving feature vector: {e}")
            raise StorageError(f"Failed to save feature vector: {e}")
    
    def get_feature_files(self) -> List[str]:
        """Get all feature CSV files"""
        try:
            features_dir = os.path.join(self.data_dir, "features")
            if not os.path.exists(features_dir):
                return []
            
            files = []
            for filename in os.listdir(features_dir):
                if filename.startswith("features_") and filename.endswith(".csv"):
                    files.append(os.path.join(features_dir, filename))
            
            return sorted(files)
        except Exception as e:
            logger.error(f"Error getting feature files: {e}")
            raise StorageError(f"Failed to get feature files: {e}")
    
    def get_spectrum_files(self) -> List[str]:
        """Get all spectrum JSON files"""
        try:
            spectrum_dir = os.path.join(self.data_dir, "spectrum")
            if not os.path.exists(spectrum_dir):
                return []
            
            files = []
            for filename in os.listdir(spectrum_dir):
                if filename.startswith("spectrum_") and filename.endswith(".json"):
                    files.append(os.path.join(spectrum_dir, filename))
            
            return sorted(files)
        except Exception as e:
            logger.error(f"Error getting spectrum files: {e}")
            raise StorageError(f"Failed to get spectrum files: {e}")
    
    def cleanup_old_data(self, days_to_keep: int = 30) -> None:
        """Clean up old data files"""
        try:
            cutoff_time = datetime.now() - pd.Timedelta(days=days_to_keep)
            
            # Clean spectrum files
            for filepath in self.get_spectrum_files():
                try:
                    stat = os.stat(filepath)
                    file_time = datetime.fromtimestamp(stat.st_mtime)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.debug(f"Removed old spectrum file: {filepath}")
                except Exception as e:
                    logger.error(f"Error removing {filepath}: {e}")
            
            # Clean feature files
            for filepath in self.get_feature_files():
                try:
                    stat = os.stat(filepath)
                    file_time = datetime.fromtimestamp(stat.st_mtime)
                    if file_time < cutoff_time:
                        os.remove(filepath)
                        logger.debug(f"Removed old feature file: {filepath}")
                except Exception as e:
                    logger.error(f"Error removing {filepath}: {e}")
            
            logger.info(f"Cleanup completed - removed files older than {days_to_keep} days")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            raise StorageError(f"Failed to cleanup old data: {e}")
