"""
Feature extraction service for RF signal analysis (SpectrumAlert v3.0)
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import signal, stats
from spectrum_alert.core.exceptions import FeatureExtractionError

logger = logging.getLogger(__name__)

@dataclass
class FeatureSet:
    frequency: float
    features: List[float]
    feature_names: List[str]
    timestamp: Optional[str] = None
    def to_dict(self) -> Dict[str, Any]:
        result = {'frequency': self.frequency, 'timestamp': self.timestamp}
        for name, value in zip(self.feature_names, self.features):
            result[name] = value
        return result

class FeatureExtractor:
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self.feature_names = self._get_feature_names()
    def _get_feature_names(self) -> List[str]:
        if self.lite_mode:
            return ['mean_amplitude', 'std_amplitude']
        else:
            return [
                'mean_amplitude', 'std_amplitude', 'mean_fft_magnitude', 'std_fft_magnitude',
                'skew_amplitude', 'kurt_amplitude', 'skew_phase', 'kurt_phase',
                'cyclo_autocorr', 'spectral_entropy', 'papr', 'band_energy_ratio'
            ]
    def extract_features(self, iq_data: np.ndarray, frequency: float) -> FeatureSet:
        if len(iq_data) == 0:
            raise FeatureExtractionError("Empty IQ data provided")
        if self.lite_mode:
            features = self._extract_lite_features(iq_data)
            feature_names = self._get_feature_names()
        else:
            features = self._extract_full_features(iq_data)
            feature_names = self._get_feature_names()
        return FeatureSet(frequency=frequency, features=features, feature_names=feature_names)
    def _extract_lite_features(self, iq_data: np.ndarray) -> List[float]:
        amplitude = np.abs(iq_data)
        return [float(np.mean(amplitude)), float(np.std(amplitude))]
    def _extract_full_features(self, iq_data: np.ndarray) -> List[float]:
        amplitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        fft_magnitude = np.abs(np.fft.fft(iq_data))
        features = [
            float(np.mean(amplitude)),
            float(np.std(amplitude)),
            float(np.mean(fft_magnitude)),
            float(np.std(fft_magnitude)),
            float(stats.skew(amplitude)),
            float(stats.kurtosis(amplitude)),
            float(stats.skew(phase)),
            float(stats.kurtosis(phase)),
            float(np.mean(np.correlate(amplitude, amplitude, mode='full'))),
            float(stats.entropy(np.abs(fft_magnitude))),
            float(np.max(amplitude) ** 2 / (np.mean(amplitude) ** 2 + 1e-8)),
            float(np.sum(amplitude[:len(amplitude)//2]) / (np.sum(amplitude) + 1e-8)),
        ]
        return features
