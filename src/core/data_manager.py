"""
Data management utilities for SpectrumAlert
"""

import csv
import os
import json
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Iterator
from datetime import datetime
from threading import Lock
from contextlib import contextmanager
from src.core.feature_extraction import FeatureSet

logger = logging.getLogger(__name__)


class DataManager:
    """Manages data storage and retrieval for SpectrumAlert"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self._file_locks: Dict[str, Lock] = {}
        self._ensure_data_dir()
    
    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists"""
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _get_file_lock(self, filename: str) -> Lock:
        """Get or create a file lock"""
        if filename not in self._file_locks:
            self._file_locks[filename] = Lock()
        return self._file_locks[filename]
    
    @contextmanager
    def _file_context(self, filename: str):
        """Context manager for thread-safe file operations"""
        lock = self._get_file_lock(filename)
        with lock:
            yield
    
    def save_features_csv(self, features: List[FeatureSet], filename: str, append: bool = False) -> None:
        """Save features to CSV file"""
        if not features:
            logger.warning("No features to save")
            return
        
        filepath = os.path.join(self.data_dir, filename)
        
        with self._file_context(filepath):
            try:
                mode = 'a' if append and os.path.exists(filepath) else 'w'
                write_header = mode == 'w' or not os.path.exists(filepath)
                
                with open(filepath, mode, newline='') as f:
                    # Use the first feature set to determine structure
                    fieldnames = ['frequency', 'timestamp'] + features[0].feature_names
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if write_header:
                        writer.writeheader()
                    
                    for feature_set in features:
                        row = feature_set.to_dict()
                        if row['timestamp'] is None:
                            row['timestamp'] = datetime.now().isoformat()
                        writer.writerow(row)
                
                logger.info(f"Saved {len(features)} feature sets to {filepath}")
            except Exception as e:
                logger.error(f"Error saving features to CSV: {e}")
                raise
    
    def load_features_csv(self, filename: str) -> List[FeatureSet]:
        """Load features from CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"CSV file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                features = []
                
                for row in reader:
                    # Handle case sensitivity for frequency column
                    frequency_key = 'frequency' if 'frequency' in row else 'Frequency'
                    frequency = float(row[frequency_key])
                    timestamp = row.get('timestamp') or row.get('Timestamp')
                    
                    # Extract feature values (skip frequency and timestamp)
                    feature_names = [k for k in row.keys() if k.lower() not in ['frequency', 'timestamp']]
                    feature_values = [float(row[name]) for name in feature_names]
                    
                    feature_set = FeatureSet(
                        frequency=frequency,
                        features=feature_values,
                        feature_names=feature_names,
                        timestamp=timestamp
                    )
                    features.append(feature_set)
                
                logger.info(f"Loaded {len(features)} feature sets from {filepath}")
                return features
        except Exception as e:
            logger.error(f"Error loading features from CSV: {e}")
            raise
    
    def save_features_json(self, features: List[FeatureSet], filename: str) -> None:
        """Save features to JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            data = [feature_set.to_dict() for feature_set in features]
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(features)} feature sets to {filepath}")
        except Exception as e:
            logger.error(f"Error saving features to JSON: {e}")
            raise
    
    def load_features_json(self, filename: str) -> List[FeatureSet]:
        """Load features from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"JSON file not found: {filepath}")
            return []
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            features = []
            for item in data:
                frequency = float(item['frequency'])
                timestamp = item.get('timestamp')
                
                # Extract feature values
                feature_names = [k for k in item.keys() if k not in ['frequency', 'timestamp']]
                feature_values = [float(item[name]) for name in feature_names]
                
                feature_set = FeatureSet(
                    frequency=frequency,
                    features=feature_values,
                    feature_names=feature_names,
                    timestamp=timestamp
                )
                features.append(feature_set)
            
            logger.info(f"Loaded {len(features)} feature sets from {filepath}")
            return features
        except Exception as e:
            logger.error(f"Error loading features from JSON: {e}")
            raise
    
    def get_features_dataframe(self, filename: str) -> Optional[pd.DataFrame]:
        """Load features as pandas DataFrame"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return None
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.json'):
                df = pd.read_json(filepath)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            logger.info(f"Loaded DataFrame with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading DataFrame: {e}")
            return None
    
    def save_dataframe(self, df: pd.DataFrame, filename: str) -> None:
        """Save DataFrame to file"""
        filepath = os.path.join(self.data_dir, filename)
        
        try:
            if filename.endswith('.csv'):
                df.to_csv(filepath, index=False)
            elif filename.endswith('.json'):
                df.to_json(filepath, orient='records', indent=2)
            else:
                raise ValueError(f"Unsupported file format: {filename}")
            
            logger.info(f"Saved DataFrame with shape {df.shape} to {filepath}")
        except Exception as e:
            logger.error(f"Error saving DataFrame: {e}")
            raise
    
    def append_feature_csv(self, feature_set: FeatureSet, filename: str) -> None:
        """Append a single feature set to CSV file"""
        filepath = os.path.join(self.data_dir, filename)
        
        with self._file_context(filepath):
            try:
                file_exists = os.path.exists(filepath)
                
                with open(filepath, 'a', newline='') as f:
                    fieldnames = ['frequency', 'timestamp'] + feature_set.feature_names
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    
                    if not file_exists:
                        writer.writeheader()
                    
                    row = feature_set.to_dict()
                    if row['timestamp'] is None:
                        row['timestamp'] = datetime.now().isoformat()
                    writer.writerow(row)
                
                logger.debug(f"Appended feature set to {filepath}")
            except Exception as e:
                logger.error(f"Error appending feature to CSV: {e}")
                raise
    
    def get_data_stats(self, filename: str) -> Dict[str, Any]:
        """Get statistics about the data file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return {"exists": False}
        
        try:
            file_size = os.path.getsize(filepath)
            modified_time = datetime.fromtimestamp(os.path.getmtime(filepath))
            
            stats = {
                "exists": True,
                "file_size": file_size,
                "modified_time": modified_time.isoformat(),
                "readable": os.access(filepath, os.R_OK)
            }
            
            # Try to get row count for CSV files
            if filename.endswith('.csv'):
                try:
                    with open(filepath, 'r') as f:
                        row_count = sum(1 for _ in f) - 1  # Subtract header
                    stats["row_count"] = max(0, row_count)
                except:
                    stats["row_count"] = "unknown"
            
            return stats
        except Exception as e:
            logger.error(f"Error getting data stats: {e}")
            return {"exists": True, "error": str(e)}
    
    def cleanup_old_data(self, max_age_days: int = 30) -> None:
        """Remove data files older than specified days"""
        try:
            cutoff_time = datetime.now().timestamp() - (max_age_days * 24 * 3600)
            
            for filename in os.listdir(self.data_dir):
                filepath = os.path.join(self.data_dir, filename)
                if os.path.isfile(filepath):
                    if os.path.getmtime(filepath) < cutoff_time:
                        os.remove(filepath)
                        logger.info(f"Removed old data file: {filename}")
        except Exception as e:
            logger.error(f"Error during data cleanup: {e}")
    
    def list_data_files(self) -> List[str]:
        """List all data files in the data directory"""
        try:
            files = [f for f in os.listdir(self.data_dir) 
                    if os.path.isfile(os.path.join(self.data_dir, f))]
            return sorted(files)
        except Exception as e:
            logger.error(f"Error listing data files: {e}")
            return []
    
    def migrate_old_data(self, old_filename: str, new_filename: str) -> bool:
        """Migrate data from old format to new format"""
        try:
            old_path = old_filename if os.path.exists(old_filename) else os.path.join(self.data_dir, old_filename)
            
            if not os.path.exists(old_path):
                logger.warning(f"Old data file not found: {old_path}")
                return False
            
            # Load old format data
            features = self.load_features_csv(old_filename)
            
            if features:
                # Save in new format
                self.save_features_csv(features, new_filename)
                logger.info(f"Migrated {len(features)} records from {old_filename} to {new_filename}")
                return True
            else:
                logger.warning(f"No data found in {old_filename}")
                return False
        except Exception as e:
            logger.error(f"Error migrating data: {e}")
            return False


class StreamingDataWriter:
    """Writes data in streaming fashion for real-time applications"""
    
    def __init__(self, filename: str, data_manager: DataManager):
        self.filename = filename
        self.data_manager = data_manager
        self.buffer: List[FeatureSet] = []
        self.buffer_size = 100  # Write to disk every N samples
        self._lock = Lock()
    
    def add_features(self, feature_set: FeatureSet) -> None:
        """Add features to buffer"""
        with self._lock:
            self.buffer.append(feature_set)
            
            if len(self.buffer) >= self.buffer_size:
                self.flush()
    
    def flush(self) -> None:
        """Flush buffer to disk"""
        if not self.buffer:
            return
        
        try:
            self.data_manager.save_features_csv(self.buffer, self.filename, append=True)
            self.buffer.clear()
            logger.debug(f"Flushed {len(self.buffer)} features to {self.filename}")
        except Exception as e:
            logger.error(f"Error flushing data: {e}")
    
    def close(self) -> None:
        """Close writer and flush remaining data"""
        with self._lock:
            if self.buffer:
                self.flush()
