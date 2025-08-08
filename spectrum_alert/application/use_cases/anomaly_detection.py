"""
Anomaly detection use case for SpectrumAlert v1.1
"""

import logging
from typing import List
import numpy as np
import time
from collections import deque
from spectrum_alert.core.domain.models import SpectrumData, AnomalyDetection, DetectionMode, AnomalyType
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.core.exceptions import ModelError

logger = logging.getLogger(__name__)

# Cache for windows to avoid re-allocations each call
_WINDOW_CACHE: dict[int, np.ndarray] = {}

def _hann_window(n: int) -> np.ndarray:
    w = _WINDOW_CACHE.get(n)
    if w is None:
        w = np.hanning(n).astype(np.float32)
        _WINDOW_CACHE[n] = w
    return w


after_window_power_norm_cache: dict[int, float] = {}


class AnomalyDetectionUseCase:
    """Use case for anomaly detection operations"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.anomaly_threshold = 0.7  # Default threshold
        # Exclude DC/LO spur around center
        self.dc_exclude_hz: float = 5000.0
        # Exclude edges of passband near Nyquist (FFT edges)
        self.edge_exclude_hz: float = 10000.0

        # Novelty/persistence filter (enabled by default)
        self.novelty_enabled: bool = True
        self.novelty_freq_tol_hz: float = 5000.0
        self.novelty_cooldown_s: float = 3.0
        self.novelty_score_delta: float = 0.15
        # Keep a small history of recent alerts (freq_hz, ts, score)
        self._recent_alerts: deque[tuple[float, float, float]] = deque(maxlen=128)
    
    def set_dc_exclude_hz(self, hz: float) -> None:
        """Set bandwidth around DC to ignore (helps avoid LO/DC spike false positives)."""
        if hz >= 0:
            self.dc_exclude_hz = float(hz)
    
    def set_edge_exclude_hz(self, hz: float) -> None:
        """Set bandwidth to exclude near Nyquist edges (avoids false peaks at edges of passband)."""
        if hz >= 0:
            self.edge_exclude_hz = float(hz)

    def configure_novelty(
        self,
        enabled: bool = True,
        freq_tol_hz: float = 5000.0,
        cooldown_s: float = 3.0,
        score_delta: float = 0.15,
    ) -> None:
        """Configure persistence/novelty filter.
        - enabled: turn filter on/off
        - freq_tol_hz: frequencies within this tolerance are considered the same signal
        - cooldown_s: suppress re-alerts for the same signal within this time window
        - score_delta: allow re-alert within cooldown only if score rises by at least this amount
        """
        self.novelty_enabled = bool(enabled)
        if freq_tol_hz >= 0:
            self.novelty_freq_tol_hz = float(freq_tol_hz)
        if cooldown_s >= 0:
            self.novelty_cooldown_s = float(cooldown_s)
        if score_delta >= 0:
            self.novelty_score_delta = float(score_delta)

    def _is_novel_peak(self, freq_hz: float, score: float, now_ts: float) -> bool:
        if not self.novelty_enabled:
            return True
        # Prune very old entries beyond 3x cooldown to keep deque small
        cutoff = now_ts - max(self.novelty_cooldown_s * 3.0, 1.0)
        while self._recent_alerts and self._recent_alerts[0][1] < cutoff:
            self._recent_alerts.popleft()
        # Find latest matching frequency within tolerance
        latest_match = None
        tol = abs(self.novelty_freq_tol_hz)
        for f, ts, sc in reversed(self._recent_alerts):
            if abs(f - freq_hz) <= tol:
                latest_match = (f, ts, sc)
                break
        if latest_match is None:
            return True
        _, last_ts, last_score = latest_match
        # If beyond cooldown, treat as novel again
        if (now_ts - last_ts) >= self.novelty_cooldown_s:
            return True
        # Within cooldown, allow only if score rose significantly
        if (score - last_score) >= self.novelty_score_delta:
            return True
        return False

    def detect_anomalies(self, spectrum_data: SpectrumData) -> List[AnomalyDetection]:
        """Detect anomalies in spectrum data with precise frequency estimation"""
        try:
            # Fast, vectorized conversion; avoid Python loops
            samples = np.asarray(spectrum_data.samples)
            if samples.size == 0:
                return []
            if not np.iscomplexobj(samples):
                samples = samples.astype(np.complex64, copy=False)
            n = int(samples.shape[0])
            if n < 8:
                return []
            
            # Windowing to reduce spectral leakage (cached window)
            window = _hann_window(n)
            xw = samples * window
            
            # FFT and power spectrum (float32 to reduce memory and improve speed)
            fft_vals = np.fft.fftshift(np.fft.fft(xw, n=n))
            power = (np.abs(fft_vals) ** 2).astype(np.float32, copy=False)
            
            # Optional light smoothing to reduce variance (3-bin boxcar)
            if n >= 17:
                kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
                power = np.convolve(power, kernel, mode='same')
            
            # Frequency bins (shifted to match fftshift)
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / spectrum_data.sample_rate_hz))
            df = float(freqs[1] - freqs[0]) if n > 1 else spectrum_data.sample_rate_hz
            
            # Robust baseline: median of spectrum excluding top 1% highest bins
            if n > 100:
                k = max(1, n // 100)
                thresh = np.partition(power, -k)[-k]
                mask = power < thresh
                if mask.any():
                    baseline = float(np.median(power[mask]))
                else:
                    baseline = float(np.median(power))
            else:
                baseline = float(np.median(power))
            baseline = max(baseline, 1e-12)

            # Notch DC region
            if self.dc_exclude_hz > 0 and df > 0 and n > 8:
                exclude_bins = max(1, int(self.dc_exclude_hz / df))
                mid = n // 2
                l = max(0, mid - exclude_bins)
                r = min(n, mid + exclude_bins + 1)
                power[l:r] = baseline
            
            # Notch edges near Nyquist
            if self.edge_exclude_hz > 0 and df > 0 and n > 8:
                edge_bins = max(1, int(self.edge_exclude_hz / df))
                edge_bins = min(edge_bins, (n // 2) - 1)
                power[:edge_bins] = baseline
                power[-edge_bins:] = baseline
            
            # Peak detection with parabolic interpolation for sub-bin accuracy
            idx = int(np.argmax(power))
            peak_power = float(power[idx])
            if peak_power <= 0.0:
                return []
            
            # Parabolic interpolation around the peak
            delta = 0.0
            if 0 < idx < n - 1:
                alpha = power[idx - 1]
                beta = power[idx]
                gamma = power[idx + 1]
                denom = (alpha - 2.0 * beta + gamma)
                if denom != 0.0:
                    delta = 0.5 * (alpha - gamma) / denom
                    # Clamp delta to [-0.5, 0.5] to stay within bin vicinity
                    if delta > 0.5:
                        delta = 0.5
                    elif delta < -0.5:
                        delta = -0.5
            
            # SNR-based scoring (in dB), mapped to 0..1
            snr_db = float(10.0 * np.log10((peak_power + 1e-12) / baseline))
            # Map: 3 dB -> 0.0, 23 dB -> ~1.0
            score = (snr_db - 3.0) / 20.0
            if score < 0.0:
                score = 0.0
            elif score > 1.0:
                score = 1.0
            
            # Precise anomaly frequency (center + offset with sub-bin interpolation)
            peak_freq_offset = float(freqs[idx] + delta * df)
            peak_freq_hz = float(spectrum_data.frequency_hz + peak_freq_offset)

            # Final sanity guards: drop peaks too close to DC or Nyquist edges
            # This double-checks even if notching failed due to small FFT size or driver limits
            half_bw = float(spectrum_data.sample_rate_hz) / 2.0
            dc_guard = max(self.dc_exclude_hz * 1.2, 2.0 * abs(df))
            edge_guard = max(self.edge_exclude_hz * 1.2, 2.0 * abs(df))
            if abs(peak_freq_offset) <= dc_guard:
                return []
            if (half_bw - abs(peak_freq_offset)) <= edge_guard:
                return []

            # Persistence/novelty gating to reduce repeated alerts for stable carriers
            now_ts = time.time()
            if not self._is_novel_peak(peak_freq_hz, score, now_ts):
                return []
            
            anomalies: List[AnomalyDetection] = []
            if score >= self.anomaly_threshold:
                # Extract features (optional, retained for pipeline consistency)
                try:
                    self.feature_extractor.extract_features(samples, spectrum_data.frequency_hz)
                except Exception:
                    # Don't fail anomaly detection if features fail
                    pass
                
                anomaly = AnomalyDetection(
                    spectrum_data_id=spectrum_data.id,
                    frequency_hz=peak_freq_hz,
                    anomaly_type=AnomalyType.AMPLITUDE,
                    confidence_score=score,
                    severity="medium",
                    description=f"Peak at {peak_freq_hz/1e6:.6f} MHz, SNR={snr_db:.1f} dB, score={score:.2f}",
                    detection_mode=DetectionMode.LITE if self.feature_extractor.lite_mode else DetectionMode.FULL,
                    metadata={
                        "peak_power": float(peak_power),
                        "baseline_power": float(baseline),
                        "bin_index": idx,
                        "fft_size": n,
                        "interp_delta_bins": float(delta),
                        "freq_bin_hz": df,
                        "dc_exclude_hz": float(self.dc_exclude_hz),
                        "edge_exclude_hz": float(self.edge_exclude_hz),
                        "novelty_enabled": bool(self.novelty_enabled),
                        "novelty_freq_tol_hz": float(self.novelty_freq_tol_hz),
                        "novelty_cooldown_s": float(self.novelty_cooldown_s),
                        "novelty_score_delta": float(self.novelty_score_delta),
                        "method": "periodogram:hann+interp+dc_notch+edge_notch+guards+novelty",
                    },
                )
                anomalies.append(anomaly)
                # Record this alert for future novelty filtering
                self._recent_alerts.append((peak_freq_hz, now_ts, score))
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise ModelError(f"Failed to detect anomalies: {e}")

    def set_threshold(self, threshold: float) -> None:
        """Set anomaly detection threshold (0..1)"""
        if 0.0 <= threshold <= 1.0:
            self.anomaly_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
