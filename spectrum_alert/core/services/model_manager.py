"""
Model management service for SpectrumAlert v3.0
"""

import os
import joblib
import logging
import numpy as np
from typing import Any, Optional, Dict, List
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from collections import Counter
from spectrum_alert.core.services.feature_extraction import FeatureSet
from spectrum_alert.core.exceptions import ModelError

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages machine learning models for SpectrumAlert"""
    def __init__(self, model_dir: str = "models"):
        self.model_dir = model_dir
        self._ensure_model_dir()
    def _ensure_model_dir(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)
    def save_model(self, model: Any, filename: str, metadata: Optional[Dict] = None) -> None:
        filepath = os.path.join(self.model_dir, filename)
        try:
            joblib.dump(model, filepath)
            if metadata:
                metadata_file = filepath.replace('.pkl', '_metadata.json')
                import json
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise ModelError(str(e))
    def load_model(self, filename: str) -> Optional[Any]:
        filepath = os.path.join(self.model_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Model file not found: {filepath}")
            return None
        try:
            return joblib.load(filepath)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise ModelError(str(e))
