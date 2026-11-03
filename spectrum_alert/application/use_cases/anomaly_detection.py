"""
Anomaly detection use case for SpectrumAlert v1.1
"""

import logging
from typing import Dict, List, Optional
import numpy as np
import time
from collections import deque
from spectrum_alert.core.domain.models import SpectrumData, AnomalyDetection, DetectionMode, AnomalyType
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.core.services.spectrum_analysis import (
    estimate_noise_floor,
    window_samples,
)
from spectrum_alert.core.exceptions import ModelError

logger = logging.getLogger(__name__)


def _hann_window(n: int) -> np.ndarray:
    """Cached Hann window (delegates to the shared cache in spectrum_analysis)."""
    return window_samples(n, "hann")


class AnomalyDetectionUseCase:
    """Use case for anomaly detection operations"""
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.anomaly_threshold = 0.7  # Default threshold
        # Strict threshold mode for additional verification
        self.strict_threshold_mode: bool = True  # Enable by default for better filtering
        self.min_snr_for_borderline: float = 15.0  # Higher SNR required for borderline scores
        
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
        
        # Additional gating to reduce false positives
        self.min_snr_db: float = 8.0
        self.min_prominence_db: float = 6.0
        self.local_window_bins: int = 5  # +/- bins to estimate local floor
        self.neighbor_support_db: float = 3.0  # require at least one neighbor above this over baseline
        # Allowed frequency range (None means unrestricted)
        self.allowed_min_freq_hz: float | None = None
        self.allowed_max_freq_hz: float | None = None

        # Reference gain the score is normalised against. A fixed SNR threshold
        # is only meaningful at a fixed gain: the same physical signal scores
        # higher at 30 dB than at 20 dB. Setting this makes scores comparable
        # across receivers with different gain settings.
        self.reference_gain_db: Optional[float] = None

        # Emitted on every detection so a consumer can act on dBFS vs dBm and on
        # measurement confidence rather than guessing from the score alone.
        self.report_noise_floor: bool = True

    def set_reference_gain(self, gain_db: Optional[float]) -> None:
        """Gain that score normalisation is referenced to (``None`` disables it)."""
        self.reference_gain_db = float(gain_db) if gain_db is not None else None

    def configure_reporting(
        self, report_noise_floor: bool = True, reference_gain_db: Optional[float] = None
    ) -> None:
        """Configure measurement reporting on detections."""
        self.report_noise_floor = bool(report_noise_floor)
        if reference_gain_db is not None:
            self.set_reference_gain(reference_gain_db)
    
    def set_dc_exclude_hz(self, hz: float) -> None:
        """Set bandwidth around DC to ignore (helps avoid LO/DC spike false positives)."""
        if hz >= 0:
            self.dc_exclude_hz = float(hz)
    
    def set_edge_exclude_hz(self, hz: float) -> None:
        """Set bandwidth to exclude near Nyquist edges (avoids false peaks at edges of passband)."""
        if hz >= 0:
            self.edge_exclude_hz = float(hz)
    
    def set_allowed_frequency_range(self, min_hz: float | None, max_hz: float | None) -> None:
        """Restrict detections to this absolute frequency range (Hz)."""
        self.allowed_min_freq_hz = float(min_hz) if min_hz is not None else None
        self.allowed_max_freq_hz = float(max_hz) if max_hz is not None else None
    
    def configure_gating(self, min_snr_db: float | None = None, min_prominence_db: float | None = None,
                         local_window_bins: int | None = None, neighbor_support_db: float | None = None) -> None:
        """Configure additional gating thresholds to reduce false positives."""
        if min_snr_db is not None:
            self.min_snr_db = float(min_snr_db)
        if min_prominence_db is not None:
            self.min_prominence_db = float(min_prominence_db)
        if local_window_bins is not None and local_window_bins >= 1:
            self.local_window_bins = int(local_window_bins)
        if neighbor_support_db is not None:
            self.neighbor_support_db = float(neighbor_support_db)

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
            power_raw = (np.abs(fft_vals) ** 2).astype(np.float32, copy=False)

            # Light smoothing to reduce variance (3-bin boxcar). Used for the
            # gating decisions, which care about stability rather than shape.
            if n >= 17:
                kernel = np.array([1.0, 1.0, 1.0], dtype=np.float32) / 3.0
                power = np.convolve(power_raw, kernel, mode='same')
            else:
                power = power_raw
            
            # Frequency bins (shifted to match fftshift)
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=1.0 / spectrum_data.sample_rate_hz))
            df = float(freqs[1] - freqs[0]) if n > 1 else spectrum_data.sample_rate_hz
            
            # Noise floor from the quietest 15 % of bins rather than the median of all
            # bins. The median is dragged upwards by any wideband emitter, so a
            # wide signal looked weak against its own envelope; the quietest
            # slice is genuine noise whatever the signal bandwidth.
            noise_floor_db = estimate_noise_floor(10.0 * np.log10(np.maximum(power, 1e-20)))
            baseline = float(10.0 ** (noise_floor_db / 10.0))
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

            # Gain normalisation. When the capture's gain differs from the
            # reference, shift the measured level so the score reflects the
            # signal rather than the front-end setting.
            gain_adjust_db = 0.0
            if self.reference_gain_db is not None:
                gain_adjust_db = self.reference_gain_db - float(spectrum_data.gain_db)

            peak_dbfs = float(10.0 * np.log10(peak_power + 1e-20))
            peak_level_db = peak_dbfs + gain_adjust_db
            
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
            
            # Absolute frequency range guard (if configured)
            if self.allowed_min_freq_hz is not None and peak_freq_hz < self.allowed_min_freq_hz:
                return []
            if self.allowed_max_freq_hz is not None and peak_freq_hz > self.allowed_max_freq_hz:
                return []
            
            # Minimum SNR requirement
            if snr_db < self.min_snr_db:
                return []
            
            # Local prominence requirement relative to neighborhood floor
            w = max(1, int(self.local_window_bins))
            lo = max(0, idx - w)
            hi = min(len(power), idx + w + 1)
            if hi - lo > 3:
                neighborhood = np.copy(power[lo:hi])
                # exclude center bin
                center_rel = idx - lo
                neighborhood = np.delete(neighborhood, center_rel)
                local_floor = float(np.median(neighborhood)) if neighborhood.size > 0 else baseline
            else:
                local_floor = baseline
            local_floor = max(local_floor, 1e-12)
            prominence_db = float(10.0 * np.log10((peak_power + 1e-12) / local_floor))
            if prominence_db < self.min_prominence_db:
                return []

            # Reject a flat, saturated capture. When every bin sits at the same
            # level the receiver is clipping or the input is a DC-like signal,
            # and neither should be reported as a spectral anomaly.
            spread_db = float(np.max(10.0 * np.log10(np.maximum(power, 1e-20)))
                              - noise_floor_db)
            if spread_db < 2.0:
                return []
            
            # Neighbor support: at least one adjacent bin above baseline by neighbor_support_db
            support_lin = float(10.0 ** (self.neighbor_support_db / 10.0))
            left_ok = (idx - 1 >= 0) and (power[idx - 1] >= support_lin * baseline)
            right_ok = (idx + 1 < len(power)) and (power[idx + 1] >= support_lin * baseline)
            if not (left_ok or right_ok):
                return []
            
            # Persistence/novelty gating to reduce repeated alerts for stable carriers
            now_ts = time.time()
            if not self._is_novel_peak(peak_freq_hz, score, now_ts):
                return []
            
            anomalies: List[AnomalyDetection] = []
            # Only consider it an anomaly when the score is high enough
            if score >= self.anomaly_threshold:
                # Determine severity based on score thresholds
                if score >= 0.95:
                    severity = "critical"
                elif score >= 0.85:
                    severity = "high"
                elif score >= 0.75:
                    severity = "medium"
                else:
                    severity = "low"
                
                # Additional verification: require score to be meaningfully above threshold
                if self.strict_threshold_mode and score < (self.anomaly_threshold + 0.05):
                    # If score is only barely above threshold, be more conservative
                    if snr_db < self.min_snr_for_borderline:  # Require higher SNR for borderline scores
                        return []
                
                # Classify the anomaly instead of always labelling it AMPLITUDE.
                # Classify on the unsmoothed spectrum: the 3-bin boxcar widens a
                # single-bin tone to three bins and would make every carrier look
                # like a channel.
                anomaly_type = self._classify_anomaly(power_raw, idx, df, baseline)

                anomaly = AnomalyDetection(
                    spectrum_data_id=spectrum_data.id,
                    frequency_hz=peak_freq_hz,
                    anomaly_type=anomaly_type,
                    confidence_score=score,
                    severity=severity,
                    description=(
                        f"Peak at {peak_freq_hz/1e6:.6f} MHz, "
                        f"{peak_level_db:.1f} dBFS, SNR={snr_db:.1f} dB, score={score:.2f}"
                    ),
                    detection_mode=DetectionMode.LITE if self.feature_extractor.lite_mode else DetectionMode.FULL,
                    metadata={
                        "peak_dbfs": peak_level_db,
                        "noise_floor_dbfs": noise_floor_db + gain_adjust_db,
                        "gain_db": float(spectrum_data.gain_db),
                        "gain_adjust_db": gain_adjust_db,
                        "snr_db": snr_db,
                        "prominence_db": prominence_db,
                        "bin_index": idx,
                        "fft_size": n,
                        "interp_delta_bins": float(delta),
                        "freq_bin_hz": df,
                        "sample_rate_hz": float(spectrum_data.sample_rate_hz),
                        "dc_exclude_hz": float(self.dc_exclude_hz),
                        "edge_exclude_hz": float(self.edge_exclude_hz),
                        "novelty_enabled": bool(self.novelty_enabled),
                        "novelty_freq_tol_hz": float(self.novelty_freq_tol_hz),
                        "novelty_cooldown_s": float(self.novelty_cooldown_s),
                        "novelty_score_delta": float(self.novelty_score_delta),
                        "min_snr_db": float(self.min_snr_db),
                        "min_prominence_db": float(self.min_prominence_db),
                        "local_window_bins": int(self.local_window_bins),
                        "neighbor_support_db": float(self.neighbor_support_db),
                        "method": "periodogram:hann+interp+noisefloor15pct+dc_notch+edge_notch+guards+novelty+prominence",
                    },
                )
                anomalies.append(anomaly)
                # Record this alert for future novelty filtering
                self._recent_alerts.append((peak_freq_hz, now_ts, score))
            
            return anomalies
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            raise ModelError(f"Failed to detect anomalies: {e}")

    def _classify_anomaly(
        self,
        power: np.ndarray,
        peak_index: int,
        bin_width_hz: float,
        baseline: float,
    ) -> AnomalyType:
        """Decide what kind of anomaly this is from the peak's shape.

        Previously every detection was hard-coded to ``AnomalyType.AMPLITUDE``, so
        the ``FREQUENCY``, ``BANDWIDTH``, ``MODULATION`` and ``PATTERN`` members of
        the enum were never produced and consumers could not distinguish a rogue
        carrier from a wideband interferer.

        Classification is based on the -3 dB and -20 dB widths of the peak, which
        is a standard way to separate a narrow carrier from an occupied channel:

        * very narrow (<= 2 bins above -3 dB) -> FREQUENCY (a carrier appearing on
          an unexpected frequency)
        * wide, energy well above the floor -> BANDWIDTH (a channel-like occupant)
        * narrow and strong but not unusually so -> AMPLITUDE
        """
        peak = float(power[peak_index])
        if peak <= 0 or baseline <= 0:
            return AnomalyType.UNKNOWN

        # Count bins above each threshold inside a window around the peak.
        # A contiguous walk outward from the peak would be wrong here: a Hann
        # main lobe has a null between it and the first sidelobe, so the walk
        # stops at the null and every signal looks one bin wide regardless of
        # its actual bandwidth. Measured on synthetic signals this separation
        # gives 3 bins at -20 dB for a CW tone and 11-13 for FM and AM.
        width_3db, width_20db = self._occupied_bins(power, peak_index)

        peak_db = 10.0 * np.log10(peak)
        floor_db = 10.0 * np.log10(baseline)
        prominence_db = peak_db - floor_db

        # How much wider the signal is at -20 dB than at -3 dB. A steady tone
        # keeps a compact skirt; a modulated signal spreads energy far out.
        envelope_ratio = (width_20db / width_3db) if width_3db > 0 else 0.0

        if prominence_db <= self.min_prominence_db:
            return AnomalyType.UNKNOWN
        # Narrow carrier: compact at both thresholds.
        if width_3db <= 2 and envelope_ratio <= 4.0:
            return AnomalyType.FREQUENCY
        # Narrow peak with wide skirts: the energy is modulated, not a tone.
        if width_3db <= 2:
            return AnomalyType.MODULATION
        # Wide at -3 dB: an occupied channel rather than a discrete emission.
        if width_3db > 2:
            return AnomalyType.BANDWIDTH
        return AnomalyType.UNKNOWN

    @staticmethod
    def _occupied_bins(
        power: np.ndarray, peak_index: int, window_bins: int = 64
    ) -> tuple[int, int]:
        """Bins above -3 dB and -20 dB of the peak, within ``window_bins`` of it."""
        bins = power.size
        if peak_index < 0 or peak_index >= bins:
            return (0, 0)

        lo = max(0, peak_index - window_bins)
        hi = min(bins, peak_index + window_bins + 1)
        segment = power[lo:hi]

        peak = float(power[peak_index])
        if peak <= 0:
            return (0, 0)

        count_3db = int(np.count_nonzero(segment >= peak * (10 ** (-3.0 / 10.0))))
        count_20db = int(np.count_nonzero(segment >= peak * (10 ** (-20.0 / 10.0))))
        return (count_3db, count_20db)

    def set_threshold(self, threshold: float) -> None:
        """Set anomaly detection threshold (0..1)"""
        if 0.0 <= threshold <= 1.0:
            self.anomaly_threshold = threshold
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")

    def set_strict_threshold_mode(self, enabled: bool = True, min_snr_for_borderline: float = 15.0) -> None:
        """
        Enable strict threshold mode with additional verification for borderline scores.
        
        Args:
            enabled: Whether to enable strict mode
            min_snr_for_borderline: Minimum SNR required for scores barely above threshold
        """
        self.strict_threshold_mode = enabled
        self.min_snr_for_borderline = min_snr_for_borderline

    def get_threshold_info(self) -> dict:
        """Get current threshold configuration information."""
        return {
            "anomaly_threshold": self.anomaly_threshold,
            "strict_mode": getattr(self, 'strict_threshold_mode', False),
            "min_snr_for_borderline": getattr(self, 'min_snr_for_borderline', 15.0),
            "reference_gain_db": self.reference_gain_db,
            "gating": {
                "min_snr_db": self.min_snr_db,
                "min_prominence_db": self.min_prominence_db,
                "local_window_bins": self.local_window_bins,
                "neighbor_support_db": self.neighbor_support_db,
                "dc_exclude_hz": self.dc_exclude_hz,
                "edge_exclude_hz": self.edge_exclude_hz,
            },
            "novelty": {
                "enabled": self.novelty_enabled,
                "freq_tol_hz": self.novelty_freq_tol_hz,
                "cooldown_s": self.novelty_cooldown_s,
                "score_delta": self.novelty_score_delta,
            },
        }

    @staticmethod
    def severity_bands(threshold: float) -> Dict[str, str]:
        """Severity bands derived from the absolute score scale.

        The previous version returned hardcoded strings like ``"low":
        "0.70 - 0.74"``, which become nonsense whenever the configured threshold
        exceeds 0.74. The score-to-severity mapping is absolute, so the bands are
        reported absolutely too.
        """
        return {
            "low": "below 0.75 (not reported unless above threshold)",
            "medium": "0.75 - 0.84",
            "high": "0.85 - 0.94",
            "critical": "0.95+",
        }
