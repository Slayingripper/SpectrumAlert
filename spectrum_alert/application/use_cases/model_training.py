"""
Model training use case for SpectrumAlert v1.1
"""

import logging
from typing import Optional, Tuple, List, Any, Dict
import pandas as pd
from datetime import datetime
from pathlib import Path

from spectrum_alert.infrastructure.storage import DataStorage
from spectrum_alert.core.domain.models import DetectionMode
from spectrum_alert.core.exceptions import ModelError

logger = logging.getLogger(__name__)

#: Rows below this cannot produce a meaningful fit.
MIN_TRAINING_ROWS = 10

#: Fraction of rows held out for validation.
VALIDATION_FRACTION = 0.2


class ModelTrainingUseCase:
    """Trains and persists anomaly-detection models.

    The model and its scaler are saved as a *single* artifact. Previously they
    were written to two independently timestamped files, so nothing recorded
    which scaler belonged to which model and a consumer had to guess the pairing
    from two unrelated timestamps.
    """

    def __init__(self, storage: DataStorage, model_dir: Optional[str] = None):
        self.storage = storage
        self.model_dir = Path(model_dir or "models")

    def _prepare_features(self, df: pd.DataFrame) -> Tuple[Any, List[str]]:
        """Extract the feature matrix, dropping identifier columns.

        Non-numeric and constant columns are removed here: a constant column
        divides by zero in ``StandardScaler`` and produces NaN downstream.
        """
        drop = {"id", "timestamp", "spectrum_data_id", "mode"}
        candidates = [c for c in df.columns if c not in drop]

        numeric = df[candidates].apply(pd.to_numeric, errors="coerce")
        numeric = numeric.dropna(axis=1, how="any")
        keep = [c for c in numeric.columns if numeric[c].nunique(dropna=True) > 1]

        if not keep:
            raise ModelError("No usable feature columns in the training data")

        X = numeric[keep].values.astype(float)
        return X, list(keep)

    def train_models(self, mode: DetectionMode, days: int = 7) -> bool:
        """Train an IsolationForest on collected features and persist it."""
        try:
            from sklearn.ensemble import IsolationForest  # type: ignore
            from sklearn.preprocessing import StandardScaler  # type: ignore
            from sklearn.model_selection import train_test_split  # type: ignore
            import joblib  # type: ignore
        except ImportError as ie:
            logger.error(f"Missing ML dependencies for training: {ie}")
            raise ModelError("scikit-learn and joblib are required for training")

        mode_value = getattr(mode, "value", mode)
        training_data = self.storage.load_training_data(mode=mode_value, days=days)

        if training_data is None or len(training_data) == 0:
            logger.warning(f"No training data available for mode: {mode_value}")
            return False

        logger.info(f"Training models with {len(training_data)} samples for {mode_value} mode")

        X, feature_names = self._prepare_features(training_data)
        n_samples, n_features = X.shape

        if n_samples < MIN_TRAINING_ROWS:
            logger.warning(
                "Insufficient rows to train model (%d < %d)", n_samples, MIN_TRAINING_ROWS
            )
            return False

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Hold out a slice so the reported score is measured, not assumed.
        if n_samples >= MIN_TRAINING_ROWS * 2:
            X_train, X_test = train_test_split(
                X_scaled, test_size=VALIDATION_FRACTION, random_state=42
            )
        else:
            X_train, X_test = X_scaled, None

        model = IsolationForest(
            # LITE mode has only 2 features, so a 5 % contamination assumption
            # would mark very few rows; keep the wider value there.
            contamination=0.1 if mode_value == "lite" else 0.05,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train)

        holdout_rate = None
        if X_test is not None and len(X_test):
            scores = model.score_samples(X_test)
            holdout_rate = float((scores < 0).mean())

        self.model_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        artifact = {
            "model": model,
            "scaler": scaler,
            "feature_names": list(feature_names),
            "mode": mode_value,
            "created_at": stamp,
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "holdout_anomaly_rate": holdout_rate,
        }

        path = self.model_dir / f"anomaly_model_{mode_value}_{stamp}.joblib"
        joblib.dump(artifact, path)

        # Stable "latest" pointers so consumers do not have to glob for timestamps.
        joblib.dump(artifact, self.model_dir / f"anomaly_model_{mode_value}_latest.joblib")
        (self.model_dir / f"anomaly_model_{mode_value}_latest.json").write_text(
            pd.Series(
                {
                    "mode": mode_value,
                    "created_at": stamp,
                    "artifact": path.name,
                    "n_samples": int(n_samples),
                    "n_features": int(n_features),
                    "feature_names": list(feature_names),
                    "holdout_anomaly_rate": holdout_rate,
                }
            ).to_json(orient="values", indent=2)
        )

        logger.info(
            "Saved anomaly model: %s (%d samples, %d features, holdout anomaly rate %s)",
            path.name, n_samples, n_features,
            "n/a" if holdout_rate is None else f"{holdout_rate:.3f}",
        )
        return True

    def load_model(self, mode: DetectionMode) -> Optional[Dict[str, Any]]:
        """Load the most recent artifact for ``mode``, or ``None``."""
        try:
            import joblib  # type: ignore
        except ImportError as ie:
            raise ModelError("joblib is required to load models") from ie

        mode_value = getattr(mode, "value", mode)
        latest = self.model_dir / f"anomaly_model_{mode_value}_latest.joblib"
        if not latest.exists():
            logger.info(f"No model found for mode {mode_value}")
            return None
        try:
            return joblib.load(latest)
        except Exception as e:
            logger.error(f"Failed to load {latest}: {e}")
            return None

    def list_models(self) -> List[Dict[str, Any]]:
        """Summaries of every persisted model artifact, newest first."""
        if not self.model_dir.exists():
            return []
        models: List[Dict[str, Any]] = []
        for path in sorted(self.model_dir.glob("anomaly_model_*.joblib"), reverse=True):
            try:
                stat = path.stat()
                models.append({
                    "file": path.name,
                    "size_bytes": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except OSError:
                continue
        return models

    def validate_models(self, mode: DetectionMode) -> bool:
        """Check that a model for ``mode`` loads and can score a sample.

        The previous version was a stub that unconditionally returned ``True``,
        so a corrupt or mismatched model looked healthy.
        """
        try:
            artifact = self.load_model(mode)
            if artifact is None:
                logger.warning(f"No model available to validate for mode {mode}")
                return False

            required = {"model", "scaler", "feature_names"}
            missing = required - set(artifact)
            if missing:
                logger.error(f"Model artifact is missing {sorted(missing)}")
                return False

            n_features = len(artifact["feature_names"])
            if not n_features:
                logger.error("Model artifact declares no features")
                return False

            # Score a synthetic row to prove the pipeline is wired end to end.
            probe = artifact["scaler"].transform(
                pd.DataFrame([[0.0] * n_features], columns=artifact["feature_names"])
            )
            artifact["model"].score_samples(probe)

            logger.info(f"Validated model for {getattr(mode, 'value', mode)}: {n_features} features")
            return True
        except Exception as e:
            logger.error(f"Error validating models: {e}")
            return False