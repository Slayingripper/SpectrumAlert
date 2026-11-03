"""
Feature extraction service for RF signal analysis (SpectrumAlert v3.0)
"""

from __future__ import annotations

import numpy as np
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from scipy import stats
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
    """Computes per-capture descriptors used for classification.

    Several of the original features were misnamed or meaningless and have been
    corrected:

    * ``skew_phase`` / ``kurt_phase`` used linear statistics on ``np.angle``. The
      phase of a noisy complex sample is near-uniform on [-pi, pi] regardless of
      the signal, so these were constants carrying no information. They are now
      ``phase_resultant`` and ``phase_circvar``: circular statistics computed on
      the *differential* phase, which distinguish a coherent carrier from noise.
    * ``cyclo_autocorr`` averaged the full autocorrelation of the amplitude. That
      result is a function of N and total energy alone, so it scaled with sample
      count and carried no signal information. It is now a normalised lag-1
      autocorrelation, which measures how amplitude-correlated successive
      samples are.
    * ``band_energy_ratio`` summed the *time-domain* amplitude over the first
      half of the capture, so it was always near 0.5 and had nothing to do with
      frequency bands. It is now the share of spectral energy inside a band
      around the centre frequency.
    * ``papr`` was returned as a raw linear ratio (values around 11800). It is
      now reported in dB, which is the form peak-to-average power ratio is
      normally expressed in.
    """

    def __init__(self, lite_mode: bool = False, band_fraction: float = 0.5):
        self.lite_mode = lite_mode
        self.band_fraction = float(np.clip(band_fraction, 0.05, 1.0))
        self.feature_names = self._get_feature_names()

    def _get_feature_names(self) -> List[str]:
        if self.lite_mode:
            return ['mean_amplitude', 'std_amplitude']
        return [
            'mean_amplitude', 'std_amplitude', 'mean_fft_magnitude', 'std_fft_magnitude',
            'skew_amplitude', 'kurt_amplitude', 'phase_resultant', 'phase_circvar',
            'cyclo_autocorr', 'spectral_entropy', 'papr_db', 'band_energy_ratio'
        ]

    def extract_features(self, iq_data: np.ndarray, frequency: float) -> FeatureSet:
        data = np.asarray(iq_data)
        if data.size == 0:
            raise FeatureExtractionError("Empty IQ data provided")
        if self.lite_mode:
            features = self._extract_lite_features(data)
        else:
            features = self._extract_full_features(data)
        feature_names = self._get_feature_names()
        if len(features) != len(feature_names):
            # A mismatch here used to be silently accepted and then truncated by
            # a downstream zip(), which quietly trained on the wrong columns.
            raise FeatureExtractionError(
                f"feature count {len(features)} does not match {len(feature_names)} names"
            )
        return FeatureSet(frequency=frequency, features=features, feature_names=feature_names)

    def _extract_lite_features(self, iq_data: np.ndarray) -> List[float]:
        amplitude = np.abs(iq_data)
        return [float(np.mean(amplitude)), float(np.std(amplitude))]

    def _extract_full_features(self, iq_data: np.ndarray) -> List[float]:
        amplitude = np.abs(iq_data)
        phase = np.angle(iq_data)

        # Window before the FFT. Without one, spectral leakage inflates both FFT
        # statistics and the entropy measure.
        windowed = iq_data * np.hanning(len(iq_data))
        fft_magnitude = np.abs(np.fft.fftshift(np.fft.fft(windowed)))
        power = fft_magnitude ** 2

        # Circular statistics on the *differential* phase.
        #
        # Taking them on the absolute phase looks correct but is not: a 50 kHz
        # tone sampled over 4 ms sweeps ~200 cycles, so its phase distribution is
        # uniform and the resultant length collapses to zero — identical to
        # noise. A carrier is characterised by a *constant phase increment*
        # (constant instantaneous frequency), so differencing first gives ~1 for
        # a coherent tone and ~0 for noise. This is the discriminator form.
        differential_phase = np.diff(phase)
        resultant_length = float(np.abs(np.mean(np.exp(1j * differential_phase))))
        circular_variance = float(max(0.0, 1.0 - resultant_length))

        # Normalised lag-1 autocorrelation of the amplitude. O(n), unlike the
        # full autocorrelation this replaces.
        cyclo_autocorr = self._lag_one_autocorrelation(amplitude)

        # Spectral entropy over normalised power, so the value is in nats and
        # bounded by log2(n_bins) for a flat spectrum.
        total_power = float(np.sum(power))
        if total_power > 0:
            normalised = power / total_power
            nonzero = normalised[normalised > 0]
            spectral_entropy = float(-np.sum(nonzero * np.log2(nonzero)))
        else:
            spectral_entropy = 0.0

        # PAPR in dB.
        mean_amplitude = float(np.mean(amplitude))
        peak_power = float(np.max(amplitude) ** 2)
        papr_db = float(10.0 * np.log10((peak_power / (mean_amplitude ** 2)) + 1e-12))

        band_energy_ratio = self._band_energy_ratio(power)

        return [
            float(np.mean(amplitude)),
            float(np.std(amplitude)),
            float(np.mean(fft_magnitude)),
            float(np.std(fft_magnitude)),
            float(stats.skew(amplitude)) if len(amplitude) > 2 else 0.0,
            float(stats.kurtosis(amplitude)) if len(amplitude) > 3 else 0.0,
            resultant_length,
            circular_variance,
            cyclo_autocorr,
            spectral_entropy,
            papr_db,
            band_energy_ratio,
        ]

    @staticmethod
    def _lag_one_autocorrelation(amplitude: np.ndarray) -> float:
        """Normalised lag-1 autocorrelation of a real series."""
        if amplitude.size < 3:
            return 0.0
        centered = amplitude - amplitude.mean()
        denominator = float(np.dot(centered, centered))
        if denominator <= 0:
            return 0.0
        return float(np.dot(centered[:-1], centered[1:]) / denominator)

    def _band_energy_ratio(self, power: np.ndarray) -> float:
        """Share of spectral energy inside a band around the FFT centre.

        The previous version sliced the time-domain amplitude array, so it always
        returned approximately 0.5 regardless of the signal. This is the
        frequency-domain quantity its name implies.
        """
        total = float(np.sum(power))
        if total <= 0:
            return 0.0
        bins = power.size
        # Centre band spans band_fraction of the spectrum, centred on DC.
        half = max(1, int(bins * self.band_fraction / 2.0))
        centre = bins // 2
        lo = max(0, centre - half)
        hi = min(bins, centre + half + 1)
        return float(np.sum(power[lo:hi]) / total)