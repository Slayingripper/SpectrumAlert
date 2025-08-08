"""
Model training use case for SpectrumAlert v1.1
"""

import logging
from typing import Optional, Tuple, List, Any
import pandas as pd
from datetime import datetime
from pathlib import Path

from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.core.domain.models import DetectionMode
from spectrum_alert.core.exceptions import ModelError

logger = logging.getLogger(__name__)


class ModelTrainingUseCase:
    """Use case for model training operations"""
    
    def __init__(self, storage: DataStorage):
        self.storage = storage
    
    def _prepare_features(self, df: pd.DataFrame) -> Tuple[Any, List[str]]:
        # Drop non-feature columns if present
        cols = [c for c in df.columns if c not in {"id", "timestamp", "spectrum_data_id", "mode"}]
        X = df[cols].values.astype(float)
        return X, cols
    
    def train_models(self, mode: DetectionMode, days: int = 7) -> bool:
        """Train ML models using collected data and save them under models/"""
        try:
            # Lazy imports to avoid hard dependency for non-training workflows
            from sklearn.ensemble import IsolationForest  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
            import joblib  # type: ignore
            import json
            
            # Load training data
            training_data = self.storage.load_training_data(mode=mode.value, days=days)
            
            if training_data is None or len(training_data) == 0:
                logger.warning(f"No training data available for mode: {mode.value}")
                return False
            
            logger.info(f"Training models with {len(training_data)} samples for {mode.value} mode")
            X, feature_names = self._prepare_features(training_data)
            try:
                n_samples = int(getattr(X, 'shape', [len(X) if hasattr(X, '__len__') else 0])[0])
                n_features = int(getattr(X, 'shape', [0, len(X[0]) if n_samples > 0 else 0])[1])
            except Exception:
                n_samples = len(X) if hasattr(X, '__len__') else 0
                n_features = 0
            if n_samples < 10:
                logger.warning("Insufficient rows to train model (<10)")
                return False
            
            # Scale
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)
            
            # Anomaly detection model (IsolationForest)
            model = IsolationForest(
                contamination=0.1 if mode == DetectionMode.LITE else 0.05,
                random_state=42,
                n_jobs=-1
            )
            model.fit(Xs)
            
            # Persist models
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = mode.value
            model_path = models_dir / f"anomaly_model_{suffix}_{ts}.pkl"
            scaler_path = models_dir / f"anomaly_scaler_{suffix}_{ts}.pkl"
            meta_path = models_dir / f"anomaly_model_{suffix}_{ts}.meta.json"
            
            joblib.dump(model, model_path)
            joblib.dump(scaler, scaler_path)
            meta = {
                "mode": mode.value,
                "created_at": ts,
                "n_samples": int(n_samples),
                "n_features": int(n_features),
                "feature_names": feature_names,
            }
            meta_path.write_text(json.dumps(meta, indent=2))
            
            logger.info(f"Saved anomaly model: {model_path.name}")
            return True
        except ImportError as ie:
            logger.error(f"Missing ML dependencies for training: {ie}")
            raise ModelError("scikit-learn and joblib are required for training")
        except Exception as e:
            logger.error(f"Error training models: {e}")
            raise ModelError(f"Failed to train models: {e}")
    
    def validate_models(self, mode: DetectionMode) -> bool:
        """Validate trained models (placeholder)"""
        try:
            logger.info(f"Validating models for {mode.value} mode")
            return True
        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return False
