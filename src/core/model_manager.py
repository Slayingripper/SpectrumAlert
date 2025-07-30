"""
Model management utilities for SpectrumAlert
"""

import os
import joblib
import logging
import numpy as np
from typing import Any, Optional, Dict, List, Tuple
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from src.core.feature_extraction import FeatureSet
from src.utils.logger import log_performance

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages machine learning models for SpectrumAlert"""
    
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._ensure_model_dir()
    
    def _ensure_model_dir(self) -> None:
        """Ensure model directory exists"""
        os.makedirs(self.model_dir, exist_ok=True)
    
    def save_model(self, model: Any, filename: str, metadata: Optional[Dict] = None) -> None:
        """Save a model with optional metadata"""
        filepath = os.path.join(self.model_dir, filename)
        
        try:
            # Save the model
            joblib.dump(model, filepath)
            
            # Save metadata if provided
            if metadata:
                metadata_file = filepath.replace('.pkl', '_metadata.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filename: str) -> Optional[Any]:
        """Load a model from file"""
        filepath = os.path.join(self.model_dir, filename)
        
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return None
        
        try:
            model = joblib.load(filepath)
            logger.info(f"Model loaded from {filepath}")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None
    
    def load_model_metadata(self, filename: str) -> Optional[Dict]:
        """Load model metadata"""
        filepath = os.path.join(self.model_dir, filename)
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        
        if not os.path.exists(metadata_file):
            return None
        
        try:
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            logger.error(f"Error loading model metadata: {e}")
            return None
    
    def list_models(self) -> List[str]:
        """List all available models"""
        try:
            models = [f for f in os.listdir(self.model_dir) 
                     if f.endswith('.pkl')]
            return sorted(models)
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    def delete_model(self, filename: str) -> bool:
        """Delete a model and its metadata"""
        filepath = os.path.join(self.model_dir, filename)
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
                logger.info(f"Deleted model: {filename}")
            
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
                logger.info(f"Deleted model metadata: {filename}")
            
            return True
        except Exception as e:
            logger.error(f"Error deleting model: {e}")
            return False


class RFFingerprintingTrainer:
    """Trains RF fingerprinting models"""
    
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self.model_manager = ModelManager()
        self.scaler = StandardScaler()
        self.pca = None
    
    @log_performance
    def train_model(self, features: List[FeatureSet], 
                   n_devices: int = 10) -> Tuple[Optional[Any], Optional[Dict]]:
        """Train RF fingerprinting model"""
        if len(features) < 10:
            logger.error("Insufficient data for training (minimum 10 samples required)")
            return None, None
        
        # Automatically adjust n_devices based on available data
        # Need at least 5 samples per device for proper train/test split
        max_devices = max(2, len(features) // 5)
        n_devices = min(n_devices, max_devices)
        logger.info(f"Adjusted n_devices to {n_devices} based on {len(features)} samples")
        
        try:
            # Prepare data
            X, y = self._prepare_training_data(features, n_devices)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Apply PCA if not in lite mode
            if not self.lite_mode and X_train_scaled.shape[1] > 5:
                n_components = min(10, X_train_scaled.shape[1], X_train_scaled.shape[0] // 2)
                self.pca = PCA(n_components=n_components)
                X_train_scaled = self.pca.fit_transform(X_train_scaled)
                X_test_scaled = self.pca.transform(X_test_scaled)
            
            # Configure model
            if self.lite_mode:
                model = RandomForestClassifier(
                    n_estimators=50,
                    max_depth=10,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = self._get_optimized_model(X_train_scaled, y_train)
            
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Create metadata
            metadata = {
                'model_type': 'RandomForestClassifier',
                'lite_mode': self.lite_mode,
                'accuracy': float(accuracy),
                'n_features': X_train_scaled.shape[1],
                'n_samples': len(X),
                'n_devices': n_devices,
                'created_at': datetime.now().isoformat(),
                'feature_names': features[0].feature_names if features else []
            }
            
            logger.info(f"Model trained with accuracy: {accuracy:.2%}")
            logger.info(f"Classification Report:\n{classification_report(y_test, y_pred)}")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error training RF fingerprinting model: {e}")
            return None, None
    
    def _prepare_training_data(self, features: List[FeatureSet], 
                              n_devices: int) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from features"""
        X = np.array([f.features for f in features])
        
        # Generate simulated device labels
        y = np.array([f"Device_{i % n_devices}" for i in range(len(features))])
        
        return X, y
    
    def _get_optimized_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Get optimized model using grid search"""
        # Determine cross-validation folds
        class_counts = Counter(y_train)
        min_samples = min(class_counts.values())
        n_splits = min(5, min_samples)
        
        if n_splits < 2:
            logger.warning("Insufficient data for cross-validation, using default parameters")
            return RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search parameters
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        # Reduced parameter grid for smaller datasets
        if len(X_train) < 1000:
            param_grid = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20],
                'min_samples_split': [2, 5]
            }
        
        rf = RandomForestClassifier(random_state=42, n_jobs=-1)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        
        grid_search = GridSearchCV(
            rf, param_grid, cv=skf, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return grid_search.best_estimator_


class AnomalyDetectionTrainer:
    """Trains anomaly detection models"""
    
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self.model_manager = ModelManager()
        self.scaler = StandardScaler()
    
    @log_performance
    def train_model(self, features: List[FeatureSet],
                   contamination: float = 0.05) -> Tuple[Optional[Any], Optional[Dict]]:
        """Train anomaly detection model"""
        if len(features) < 10:
            logger.error("Insufficient data for training (minimum 10 samples required)")
            return None, None
        
        try:
            # Prepare data
            X = np.array([f.features for f in features])
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Configure model
            if self.lite_mode:
                model = IsolationForest(
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1
                )
            else:
                model = IsolationForest(
                    contamination=contamination,
                    n_estimators=200,
                    max_samples='auto',
                    random_state=42,
                    n_jobs=-1
                )
            
            # Train model
            model.fit(X_scaled)
            
            # Evaluate model
            predictions = model.predict(X_scaled)
            anomaly_score = model.decision_function(X_scaled)
            n_anomalies = np.sum(predictions == -1)
            
            # Calculate silhouette score if possible
            silhouette = None
            if len(np.unique(predictions)) > 1:
                try:
                    silhouette = silhouette_score(X_scaled, predictions)
                except:
                    pass
            
            # Create metadata
            metadata = {
                'model_type': 'IsolationForest',
                'lite_mode': self.lite_mode,
                'contamination': contamination,
                'n_features': X_scaled.shape[1],
                'n_samples': len(X),
                'n_anomalies': int(n_anomalies),
                'anomaly_ratio': float(n_anomalies / len(X)),
                'silhouette_score': float(silhouette) if silhouette is not None else None,
                'created_at': datetime.now().isoformat(),
                'feature_names': features[0].feature_names if features else []
            }
            
            logger.info(f"Anomaly detection model trained")
            logger.info(f"Detected {n_anomalies}/{len(X)} anomalies ({n_anomalies/len(X):.2%})")
            if silhouette is not None:
                logger.info(f"Silhouette score: {silhouette:.3f}")
            
            return model, metadata
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            return None, None


class ModelPredictor:
    """Makes predictions using trained models"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.rf_model = None
        self.anomaly_model = None
        self.scaler = None
        self.pca = None
        self._rf_metadata = None
        self._anomaly_metadata = None
    
    def load_models(self, rf_model_file: str, anomaly_model_file: str,
                   scaler_file: Optional[str] = None, pca_file: Optional[str] = None) -> bool:
        """Load trained models"""
        try:
            # Load RF fingerprinting model
            self.rf_model = self.model_manager.load_model(rf_model_file)
            self._rf_metadata = self.model_manager.load_model_metadata(rf_model_file)
            
            # Load anomaly detection model
            self.anomaly_model = self.model_manager.load_model(anomaly_model_file)
            self._anomaly_metadata = self.model_manager.load_model_metadata(anomaly_model_file)
            
            # Load scaler if provided
            if scaler_file:
                self.scaler = self.model_manager.load_model(scaler_file)
            
            # Load PCA if provided
            if pca_file:
                self.pca = self.model_manager.load_model(pca_file)
            
            if self.rf_model and self.anomaly_model:
                logger.info("Models loaded successfully")
                return True
            else:
                logger.error("Failed to load required models")
                return False
                
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def predict_device(self, feature_set: FeatureSet) -> Optional[str]:
        """Predict device type from features"""
        if not self.rf_model:
            logger.error("RF fingerprinting model not loaded")
            return None
        
        try:
            X = np.array([feature_set.features])
            
            # Check feature dimension compatibility
            expected_features = getattr(self.rf_model, 'n_features_in_', None)
            if expected_features and X.shape[1] != expected_features:
                logger.error(f"Feature dimension mismatch: got {X.shape[1]} features, RF model expects {expected_features} features")
                return None
            
            # Apply scaling if available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Apply PCA if available
            if self.pca:
                X = self.pca.transform(X)
            
            prediction = self.rf_model.predict(X)[0]
            return str(prediction)
            
        except Exception as e:
            logger.error(f"Error predicting device: {e}")
            return None
    
    def detect_anomaly(self, feature_set: FeatureSet) -> Dict[str, Any]:
        """Detect anomaly in features"""
        if not self.anomaly_model:
            logger.error("Anomaly detection model not loaded")
            return {'is_anomaly': False, 'score': 0.0, 'error': 'Model not loaded'}
        
        try:
            X = np.array([feature_set.features])
            
            # Check feature dimension compatibility
            expected_features = getattr(self.anomaly_model, 'n_features_in_', None)
            if expected_features and X.shape[1] != expected_features:
                error_msg = f"Feature dimension mismatch: got {X.shape[1]} features, model expects {expected_features} features"
                logger.error(error_msg)
                return {'is_anomaly': False, 'score': 0.0, 'error': error_msg}
            
            # Apply scaling if available
            if self.scaler:
                X = self.scaler.transform(X)
            
            # Make prediction
            prediction = self.anomaly_model.predict(X)[0]
            score = self.anomaly_model.decision_function(X)[0]
            
            return {
                'is_anomaly': prediction == -1,
                'score': float(score),
                'confidence': abs(float(score))
            }
            
        except Exception as e:
            error_msg = f"Error detecting anomaly: {e}"
            logger.error(error_msg)
            return {'is_anomaly': False, 'score': 0.0, 'error': error_msg}
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            'rf_model_loaded': self.rf_model is not None,
            'anomaly_model_loaded': self.anomaly_model is not None,
            'rf_metadata': self._rf_metadata,
            'anomaly_metadata': self._anomaly_metadata
        }
