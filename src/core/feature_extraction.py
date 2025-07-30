"""
Feature extraction utilities for RF signal analysis
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import signal, stats
from src.utils.logger import log_performance

logger = logging.getLogger(__name__)


@dataclass
class FeatureSet:
    """Container for extracted features"""
    frequency: float
    features: List[float]
    feature_names: List[str]
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format"""
        result = {
            'frequency': self.frequency,
            'timestamp': self.timestamp
        }
        for name, value in zip(self.feature_names, self.features):
            result[name] = value
        return result


class FeatureExtractor:
    """Unified feature extraction for RF signals"""
    
    def __init__(self, lite_mode: bool = False):
        self.lite_mode = lite_mode
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names based on mode"""
        if self.lite_mode:
            return ['mean_amplitude', 'std_amplitude']
        else:
            return [
                'mean_amplitude', 'std_amplitude', 'mean_fft_magnitude', 'std_fft_magnitude',
                'skew_amplitude', 'kurt_amplitude', 'skew_phase', 'kurt_phase',
                'cyclo_autocorr', 'spectral_entropy', 'papr', 'band_energy_ratio',
                'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate',
                'rms_amplitude', 'crest_factor', 'spectral_flatness'
            ]
    
    @log_performance
    def extract_features(self, iq_data: np.ndarray, frequency: float) -> FeatureSet:
        """Extract features from IQ data"""
        if len(iq_data) == 0:
            raise ValueError("Empty IQ data provided")
        
        if self.lite_mode:
            features = self._extract_lite_features(iq_data)
        else:
            features = self._extract_full_features(iq_data)
        
        return FeatureSet(
            frequency=frequency,
            features=features,
            feature_names=self.feature_names,
            timestamp=None  # Will be set by caller if needed
        )
    
    def extract_features_compatible(self, iq_data: np.ndarray, frequency: float, 
                                  target_feature_names: List[str]) -> FeatureSet:
        """Extract features compatible with a specific model's expected features"""
        if len(iq_data) == 0:
            raise ValueError("Empty IQ data provided")
        
        # First extract all available features (full mode)
        all_features = self._extract_full_features(iq_data)
        all_feature_names = [
            'mean_amplitude', 'std_amplitude', 'mean_fft_magnitude', 'std_fft_magnitude',
            'skew_amplitude', 'kurt_amplitude', 'skew_phase', 'kurt_phase',
            'cyclo_autocorr', 'spectral_entropy', 'papr', 'band_energy_ratio',
            'spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate',
            'rms_amplitude', 'crest_factor', 'spectral_flatness'
        ]
        
        # Create mapping from feature name to value (case-insensitive)
        feature_map = {}
        for name, value in zip(all_feature_names, all_features):
            # Normalize to lowercase for matching
            feature_map[name.lower()] = value
        
        # Extract only the features the model expects
        compatible_features = []
        compatible_names = []
        
        for target_name in target_feature_names:
            # Normalize target name to lowercase for matching
            target_key = target_name.lower()
            if target_key in feature_map:
                compatible_features.append(feature_map[target_key])
                compatible_names.append(target_name)  # Keep original case from model
            else:
                logger.warning(f"Target feature '{target_name}' not found in available features")
                # Use 0.0 as default for missing features
                compatible_features.append(0.0)
                compatible_names.append(target_name)
        
        return FeatureSet(
            frequency=frequency,
            features=compatible_features,
            feature_names=compatible_names,
            timestamp=None
        )
    
    def _extract_lite_features(self, iq_data: np.ndarray) -> List[float]:
        """Extract basic features for lite mode"""
        try:
            I = np.real(iq_data)
            Q = np.imag(iq_data)
            amplitude = np.sqrt(I**2 + Q**2)
            
            # Basic amplitude statistics
            mean_amplitude = float(np.mean(amplitude))
            std_amplitude = float(np.std(amplitude))
            
            # Ensure no negative values due to numerical issues
            std_amplitude = abs(std_amplitude)
            
            return [mean_amplitude, std_amplitude]
        except Exception as e:
            logger.error(f"Error extracting lite features: {e}")
            return [0.0, 0.0]
    
    def _extract_full_features(self, iq_data: np.ndarray) -> List[float]:
        """Extract comprehensive features for full mode"""
        try:
            I = np.real(iq_data)
            Q = np.imag(iq_data)
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.unwrap(np.angle(iq_data))
            
            # FFT analysis
            fft_values = np.fft.fft(iq_data)
            fft_magnitude = np.abs(fft_values)
            fft_freq = np.fft.fftfreq(len(iq_data))
            
            # Basic amplitude statistics
            mean_amplitude = float(np.mean(amplitude))
            std_amplitude = float(np.std(amplitude))
            rms_amplitude = float(np.sqrt(np.mean(amplitude**2)))
            
            # FFT magnitude statistics
            mean_fft_magnitude = float(np.mean(fft_magnitude))
            std_fft_magnitude = float(np.std(fft_magnitude))
            
            # Higher-order moments for amplitude
            skew_amplitude, kurt_amplitude = self._safe_moments(amplitude, mean_amplitude, std_amplitude)
            
            # Phase statistics
            mean_phase = float(np.mean(phase))
            std_phase = float(np.std(phase))
            skew_phase, kurt_phase = self._safe_moments(phase, mean_phase, std_phase)
            
            # Cyclostationary features
            cyclo_autocorr = self._calculate_cyclo_autocorr(amplitude)
            
            # Spectral features
            spectral_entropy = self._calculate_spectral_entropy(fft_magnitude)
            spectral_centroid = self._calculate_spectral_centroid(fft_magnitude, fft_freq)
            spectral_rolloff = self._calculate_spectral_rolloff(fft_magnitude, fft_freq)
            spectral_flatness = self._calculate_spectral_flatness(fft_magnitude)
            
            # Power-based features
            papr = self._calculate_papr(amplitude, mean_amplitude)
            crest_factor = self._calculate_crest_factor(amplitude, rms_amplitude)
            
            # Band energy ratio
            band_energy_ratio = self._calculate_band_energy_ratio(fft_magnitude)
            
            # Zero crossing rate
            zero_crossing_rate = self._calculate_zero_crossing_rate(I)
            
            return [
                mean_amplitude, std_amplitude, mean_fft_magnitude, std_fft_magnitude,
                skew_amplitude, kurt_amplitude, skew_phase, kurt_phase,
                cyclo_autocorr, spectral_entropy, papr, band_energy_ratio,
                spectral_centroid, spectral_rolloff, zero_crossing_rate,
                rms_amplitude, crest_factor, spectral_flatness
            ]
        except Exception as e:
            logger.error(f"Error extracting full features: {e}")
            return [0.0] * len(self.feature_names)
    
    def _safe_moments(self, data: np.ndarray, mean_val: float, std_val: float) -> tuple:
        """Safely calculate skewness and kurtosis"""
        try:
            if std_val > 1e-10 and len(data) > 2:
                skew = float(stats.skew(data))
                kurt = float(stats.kurtosis(data))
                return skew, kurt
            else:
                return 0.0, 0.0
        except:
            return 0.0, 0.0
    
    def _calculate_cyclo_autocorr(self, amplitude: np.ndarray) -> float:
        """Calculate cyclostationary autocorrelation"""
        try:
            if len(amplitude) > 1:
                autocorr = np.correlate(amplitude, amplitude, mode='full')
                mid_point = len(autocorr) // 2
                return float(np.mean(np.abs(autocorr[mid_point:])))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_spectral_entropy(self, fft_magnitude: np.ndarray) -> float:
        """Calculate spectral entropy"""
        try:
            fft_sum = np.sum(fft_magnitude)
            if fft_sum > 1e-10:
                normalized_fft = fft_magnitude / fft_sum
                # Avoid log(0) by adding small epsilon
                entropy = -np.sum(normalized_fft * np.log2(normalized_fft + 1e-12))
                return float(entropy)
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_spectral_centroid(self, fft_magnitude: np.ndarray, fft_freq: np.ndarray) -> float:
        """Calculate spectral centroid"""
        try:
            fft_sum = np.sum(fft_magnitude)
            if fft_sum > 1e-10:
                centroid = np.sum(fft_freq * fft_magnitude) / fft_sum
                return float(abs(centroid))
            else:
                return 0.0
        except:
            return 0.0
    
    def _calculate_spectral_rolloff(self, fft_magnitude: np.ndarray, fft_freq: np.ndarray, percentile: float = 0.85) -> float:
        """Calculate spectral rolloff"""
        try:
            cumsum = np.cumsum(fft_magnitude)
            total_energy = cumsum[-1]
            if total_energy > 1e-10:
                rolloff_point = total_energy * percentile
                rolloff_idx = np.where(cumsum >= rolloff_point)[0]
                if len(rolloff_idx) > 0:
                    return float(abs(fft_freq[rolloff_idx[0]]))
            return 0.0
        except:
            return 0.0
    
    def _calculate_spectral_flatness(self, fft_magnitude: np.ndarray) -> float:
        """Calculate spectral flatness (Wiener entropy)"""
        try:
            fft_magnitude = fft_magnitude[fft_magnitude > 1e-10]  # Remove zeros
            if len(fft_magnitude) > 0:
                geometric_mean = float(stats.gmean(fft_magnitude))
                arithmetic_mean = float(np.mean(fft_magnitude))
                if arithmetic_mean > 1e-10:
                    return geometric_mean / arithmetic_mean
            return 0.0
        except:
            return 0.0
    
    def _calculate_papr(self, amplitude: np.ndarray, mean_amplitude: float) -> float:
        """Calculate Peak-to-Average Power Ratio"""
        try:
            if mean_amplitude > 1e-10:
                peak_power = float(np.max(amplitude) ** 2)
                avg_power = float(np.mean(amplitude ** 2))
                if avg_power > 1e-10:
                    return peak_power / avg_power
            return 0.0
        except:
            return 0.0
    
    def _calculate_crest_factor(self, amplitude: np.ndarray, rms_amplitude: float) -> float:
        """Calculate crest factor"""
        try:
            if rms_amplitude > 1e-10:
                peak_amplitude = float(np.max(amplitude))
                return peak_amplitude / rms_amplitude
            return 0.0
        except:
            return 0.0
    
    def _calculate_band_energy_ratio(self, fft_magnitude: np.ndarray) -> float:
        """Calculate band energy ratio (lower half vs total)"""
        try:
            fft_sum = np.sum(fft_magnitude)
            if fft_sum > 1e-10:
                half_point = len(fft_magnitude) // 2
                lower_half_energy = np.sum(fft_magnitude[:half_point])
                return float(lower_half_energy / fft_sum)
            return 0.0
        except:
            return 0.0
    
    def _calculate_zero_crossing_rate(self, signal_data: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        try:
            if len(signal_data) > 1:
                zero_crossings = np.sum(np.diff(np.signbit(signal_data)))
                return float(zero_crossings / len(signal_data))
            return 0.0
        except:
            return 0.0


class FeatureValidator:
    """Validates extracted features"""
    
    @staticmethod
    def validate_features(feature_set: FeatureSet) -> bool:
        """Validate feature set for consistency"""
        try:
            # Check for NaN or infinite values
            for feature in feature_set.features:
                if not np.isfinite(feature):
                    logger.warning(f"Invalid feature value detected: {feature}")
                    return False
            
            # Check feature count matches expected
            if len(feature_set.features) != len(feature_set.feature_names):
                logger.warning("Feature count mismatch")
                return False
            
            return True
        except Exception as e:
            logger.error(f"Feature validation error: {e}")
            return False
    
    @staticmethod
    def sanitize_features(feature_set: FeatureSet) -> FeatureSet:
        """Sanitize features by replacing invalid values"""
        sanitized_features = []
        for feature in feature_set.features:
            if np.isnan(feature) or np.isinf(feature):
                sanitized_features.append(0.0)
            else:
                sanitized_features.append(float(feature))
        
        return FeatureSet(
            frequency=feature_set.frequency,
            features=sanitized_features,
            feature_names=feature_set.feature_names,
            timestamp=feature_set.timestamp
        )
