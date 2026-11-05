"""Tests for SDR locking, storage hardening, and the anomaly-detection pipeline."""

import json
import os
import threading
import time

import numpy as np
import pytest

from spectrum_alert.application.use_cases.anomaly_detection import AnomalyDetectionUseCase
from spectrum_alert.core.domain.models import (
    AnomalyType,
    DetectionMode,
    MonitoringSession,
    MonitoringStatus,
    SpectrumData,
)
from spectrum_alert.core.services.feature_extraction import FeatureExtractor
from spectrum_alert.infrastructure.sdr.base import SDRInterface
from spectrum_alert.infrastructure.sdr.lock import (
    DeviceLock,
    DeviceLockRegistry,
    device_file_lock,
    get_registry,
    hold_device,
)
from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner
from spectrum_alert.infrastructure.sdr.rtl_tcp import RtltcpClient, scan_with_sample_rate
from spectrum_alert.infrastructure.storage.receiver_store import JsonFileStore, ReceiverStore

SAMPLE_RATE = 2_048_000.0


# ----------------------------------------------------------------- device locks


class TestDeviceLock:
    def test_mutual_exclusion(self):
        """Two threads must never be inside the same lock at once."""
        registry = DeviceLockRegistry()
        lock = registry.for_key("usb:0")
        events = []
        deadline = time.monotonic() + 0.4

        def worker():
            while time.monotonic() < deadline:
                with lock:
                    events.append("in")
                    time.sleep(0.001)
                    events.append("out")

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        overlaps = sum(
            1 for i in range(len(events) - 1)
            if events[i] == "in" and events[i + 1] == "in"
        )
        assert overlaps == 0

    def test_reentrant(self):
        lock = DeviceLock("x")
        lock.acquire()
        lock.acquire()
        lock.release()
        lock.release()
        # Balanced acquire/release must leave the lock free.
        assert lock.acquire(timeout=0.1)
        lock.release()

    def test_registry_shares_one_lock_per_key(self):
        registry = get_registry()
        assert registry.for_key("usb:9") is registry.for_key("usb:9")
        assert registry.for_key("usb:9") is not registry.for_key("usb:8")

    def test_combined_lock_is_reentrant(self):
        # flock is per open-file-description, so a nested acquisition would
        # deadlock if the file lock were not reentrant within the process.
        with hold_device("rtltcp:host:1234"):
            with hold_device("rtltcp:host:1234"):
                with hold_device("rtltcp:host:1234"):
                    pass

    def test_file_lock_is_reentrant(self):
        with device_file_lock("reentrant:key", timeout=1.0):
            with device_file_lock("reentrant:key", timeout=1.0):
                pass


# --------------------------------------------------------------- storage locks


class TestJsonFileStore:
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / "doc.json")
        store = JsonFileStore(path)
        store.write({"a": 1})
        assert JsonFileStore(path).read() == {"a": 1}

    def test_missing_file_reads_empty(self, tmp_path):
        assert JsonFileStore(str(tmp_path / "absent.json")).read() == {}

    def test_corrupt_file_is_quarantined(self, tmp_path):
        path = tmp_path / "doc.json"
        path.write_text("{broken")
        assert JsonFileStore(str(path)).read() == {}
        assert (tmp_path / "doc.json.corrupt").exists()

    def test_write_leaves_no_temp_files(self, tmp_path):
        store = JsonFileStore(str(tmp_path / "doc.json"))
        store.write({"a": 1})
        store.write({"a": 2})
        leftovers = [n for n in os.listdir(tmp_path) if n.startswith(".tmp-")]
        assert leftovers == []

    def test_concurrent_writes_keep_the_file_valid(self, tmp_path):
        path = str(tmp_path / "doc.json")
        store = JsonFileStore(path)

        def writer(index):
            store.write({"index": index, "payload": "x" * 200})

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(12)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # A torn write would leave unparsable JSON.
        assert json.loads(open(path).read())["index"] in range(12)


# ------------------------------------------------------------- receiver store


class TestReceiverStore:
    def _receiver(self, **overrides):
        payload = {"id": "r1", "name": "Home", "mode": "local", "lat": 35.1, "lon": 33.3}
        payload.update(overrides)
        return payload

    def test_save_and_reload(self, tmp_path):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        path = str(tmp_path / "receivers.json")
        store = ReceiverStore(path)
        store.save(SDRReceiver.from_dict(self._receiver()))

        reloaded = ReceiverStore(path)
        assert len(reloaded.list()) == 1
        assert reloaded.get("r1").name == "Home"

    def test_persistence_survives_a_new_instance(self, tmp_path):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        path = str(tmp_path / "receivers.json")
        ReceiverStore(path).save(SDRReceiver.from_dict(self._receiver()))
        assert len(ReceiverStore(path).list()) == 1

    def test_upsert_replaces(self, tmp_path):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        store = ReceiverStore(str(tmp_path / "receivers.json"))
        store.save(SDRReceiver.from_dict(self._receiver(name="First")))
        store.save(SDRReceiver.from_dict(self._receiver(name="Second")))
        assert [r.name for r in store.list()] == ["Second"]

    def test_delete(self, tmp_path):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        store = ReceiverStore(str(tmp_path / "receivers.json"))
        store.save(SDRReceiver.from_dict(self._receiver()))
        assert store.delete("r1") is True
        assert store.delete("r1") is False

    def test_enabled_filters_disabled(self, tmp_path):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        store = ReceiverStore(str(tmp_path / "receivers.json"))
        store.save(SDRReceiver.from_dict(self._receiver(enabled=True)))
        store.save(SDRReceiver.from_dict(self._receiver(id="r2", name="Off", enabled=False)))
        assert [r.id for r in store.enabled()] == ["r1"]

    def test_invalid_entry_is_dropped_not_fatal(self, tmp_path):
        path = tmp_path / "receivers.json"
        path.write_text(json.dumps({
            "receivers": [
                self._receiver(),
                {"id": "bad", "name": "Bad", "lat": 999, "lon": 0},
            ]
        }))
        store = ReceiverStore(str(path))
        assert [r.id for r in store.list()] == ["r1"]


# --------------------------------------------------------------------- scanner


class TestSpectrumScanner:
    def _receiver(self, **overrides):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        payload = {"id": "r1", "name": "Test", "mode": "local", "lat": 35.1, "lon": 33.3}
        payload.update(overrides)
        return SDRReceiver.from_dict(payload)

    def test_manual_override_is_tagged(self):
        """A pinned receiver must never look like a measurement."""
        scan = SpectrumScanner().scan(self._receiver(manual_rssi_db=-62.0), 446.1e6, 300e3)
        assert scan.source == "manual_override"
        assert scan.warning and "no hardware" in scan.warning
        assert scan.peak_dbfs == pytest.approx(-62.0)

    def test_manual_override_check_reports_ok(self):
        result = SpectrumScanner().check(self._receiver(manual_rssi_db=-62.0), 446.1e6)
        assert result["ok"] is True
        assert result["source"] == "manual_override"

    def test_unreachable_rtl_tcp_raises(self):
        receiver = self._receiver(mode="rtltcp", host="127.0.0.1", port=9)
        with pytest.raises(Exception):
            SpectrumScanner().scan(receiver, 446.1e6, 200e3, integration_seconds=0.1)

    def test_check_never_raises(self):
        receiver = self._receiver(mode="rtltcp", host="127.0.0.1", port=9)
        result = SpectrumScanner().check(receiver, 446.1e6)
        assert result["ok"] is False
        assert result["message"]

    def test_invalid_span_is_rejected(self):
        with pytest.raises(Exception):
            SpectrumScanner().scan(self._receiver(manual_rssi_db=-60.0), 446.1e6, 0)

    def test_rtl_power_csv_parsing(self, tmp_path):
        """Overlapping blocks must be averaged, not concatenated.

        rtl_power's documented CSV columns are:
            date, time, Hz low, Hz high, Hz step, samples, dbm, dbm, ...
        so the levels start at index 6, "Hz low" is index 2 and "Hz step"
        index 4.
        """
        csv = (
            "2025-08-09,10:16:21,-1000,7000,2000,4096,-75.0,-74.0,-73.0,-72.0\n"
            "2025-08-09,10:16:21,1000,9000,2000,4096,-72.0,-74.5,-75.1,-75.0\n"
        )
        path = tmp_path / "out.csv"
        path.write_text(csv)

        freqs, dbm = SpectrumScanner()._parse_rtl_power_csv(str(path))
        # Row 1 covers -1000..5000, row 2 covers 1000..7000 at the same step, so
        # 1000/3000/5000 are covered twice and the union has five distinct bins.
        assert freqs.tolist() == [-1000.0, 1000.0, 3000.0, 5000.0, 7000.0]
        assert freqs[0] < freqs[-1]
        # Duplicated bins are averaged rather than one copy winning:
        # 1000 Hz is -74.0 (row 1) and -72.0 (row 2).
        assert dbm[1] == pytest.approx(-73.0)
        # 3000 Hz is -73.0 and -74.5.
        assert dbm[2] == pytest.approx(-73.75)

    def test_rtl_power_csv_ignores_garbage(self, tmp_path):
        path = tmp_path / "out.csv"
        path.write_text("not,a,valid\nrow\n")
        freqs, _ = SpectrumScanner()._parse_rtl_power_csv(str(path))
        assert freqs.size == 0


class TestRtltcpScan:
    def test_scan_places_a_tone_correctly(self):
        # 4096 points at 240 kS/s gives ~58.6 Hz per bin, so a tone 100 Hz
        # above centre lands in bin 1.
        iq = (0.2 * np.exp(2j * np.pi * 100 * np.arange(4096) / 240_000.0)).astype(np.complex64)
        freqs, dbfs, rms = scan_with_sample_rate(iq, 100e6, 240_000.0, 256)
        assert len(freqs) == len(dbfs) == 4096
        bin_hz = 240_000.0 / 4096
        assert freqs[int(np.argmax(dbfs))] == pytest.approx(
            100e6 + 100.0, abs=bin_hz
        )
        assert np.all(np.diff(freqs) > 0)
        assert rms < 0

    def test_client_requires_host_and_port(self):
        with pytest.raises(Exception):
            RtltcpClient("", 1234)
        with pytest.raises(Exception):
            RtltcpClient("host", 0)

    def test_probe_on_a_closed_port(self):
        ok, message = RtltcpClient("127.0.0.1", 9, timeout=0.5).probe()
        assert ok is False
        assert message


# -------------------------------------------------------------- anomaly detection


def _capture(signal=None, n=16384, noise=0.002, gain=30.0, seed=3):
    rng = np.random.default_rng(seed)
    base = noise * (rng.normal(size=n) + 1j * rng.normal(size=n))
    samples = (base if signal is None else base + signal).astype(np.complex64)
    return SpectrumData(
        frequency_hz=100e6,
        sample_rate_hz=SAMPLE_RATE,
        gain_db=gain,
        samples=samples.tolist(),
        duration_seconds=n / SAMPLE_RATE,
    )


def _tone(offset_hz, amplitude=0.5, n=16384):
    t = np.arange(n) / SAMPLE_RATE
    return amplitude * np.exp(2j * np.pi * offset_hz * t)


def _detector(**gating):
    detector = AnomalyDetectionUseCase(FeatureExtractor(lite_mode=True))
    defaults = dict(min_snr_db=10.0, min_prominence_db=8.0, neighbor_support_db=4.0)
    defaults.update(gating)
    detector.configure_gating(**defaults)
    return detector


class TestAnomalyDetection:
    def test_detects_a_carrier_at_the_right_frequency(self):
        anomalies = _detector().detect_anomalies(_capture(_tone(200_000.0)))
        assert len(anomalies) == 1
        anomaly = anomalies[0]
        assert anomaly.frequency_hz == pytest.approx(100.2e6, abs=2_000.0)

    def test_reports_noise_floor_and_dbfs(self):
        anomalies = _detector().detect_anomalies(_capture(_tone(200_000.0)))
        metadata = anomalies[0].metadata
        assert "noise_floor_dbfs" in metadata
        assert metadata["noise_floor_dbfs"] < metadata["peak_dbfs"]
        assert metadata["snr_db"] > 10

    def test_plain_noise_produces_nothing(self):
        assert _detector().detect_anomalies(_capture()) == []

    def test_dc_only_signal_is_rejected(self):
        """The DC notch exists precisely for this."""
        n = 16384
        rng = np.random.default_rng(4)
        dc = 0.5 + 0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))
        assert _detector().detect_anomalies(_capture(dc.astype(np.complex64))) == []

    def test_classifies_a_narrow_carrier_as_frequency(self):
        anomalies = _detector().detect_anomalies(_capture(_tone(200_000.0)))
        assert anomalies[0].anomaly_type is AnomalyType.FREQUENCY

    def test_classifies_a_dense_cluster_as_bandwidth(self):
        """Many closely spaced tones fill the -3 dB width the classifier measures.

        20 tones 80 kHz apart still look like a narrow peak to the -3 dB
        measure; 200 tones 8 kHz apart merge into a genuinely wide occupant.
        """
        n = 16384
        t = np.arange(n) / SAMPLE_RATE
        dense = sum(
            0.15 * np.exp(2j * np.pi * (100e3 + offset) * t)
            for offset in np.linspace(-800e3, 800e3, 200)
        )
        anomalies = _detector().detect_anomalies(_capture(dense.astype(np.complex64)))
        assert anomalies, "wideband cluster should still be detected"
        assert anomalies[0].anomaly_type is AnomalyType.BANDWIDTH

    def test_classifies_a_sparse_cluster_as_frequency(self):
        n = 16384
        t = np.arange(n) / SAMPLE_RATE
        sparse = sum(
            0.15 * np.exp(2j * np.pi * (100e3 + offset) * t)
            for offset in np.linspace(-800e3, 800e3, 20)
        )
        anomalies = _detector().detect_anomalies(_capture(sparse.astype(np.complex64)))
        assert anomalies[0].anomaly_type is AnomalyType.FREQUENCY

    def test_novelty_filter_suppresses_repeats(self):
        detector = _detector()
        capture = _capture(_tone(200_000.0))
        first = detector.detect_anomalies(capture)
        second = detector.detect_anomalies(capture)
        assert len(first) == 1
        assert second == [], "a stable carrier must not re-alert within the cooldown"

    def test_novelty_filter_allows_a_score_rise(self):
        detector = _detector()
        capture = _capture(_tone(200_000.0))
        assert len(detector.detect_anomalies(capture)) == 1

        # With a zero delta required, an equal-score repeat counts as novel, so
        # this exercises the score-rise path rather than the cooldown alone.
        detector.configure_novelty(score_delta=0.0)
        assert len(detector.detect_anomalies(capture)) == 1

    def test_novelty_filter_suppresses_a_different_frequency_too_close(self):
        """A 1 kHz shift is well inside the 5 kHz tolerance, so it is suppressed."""
        detector = _detector()
        assert len(detector.detect_anomalies(_capture(_tone(200_000.0)))) == 1
        assert detector.detect_anomalies(_capture(_tone(201_000.0))) == []

    def test_novelty_filter_alows_a_frequency_outside_the_tolerance(self):
        detector = _detector()
        assert len(detector.detect_anomalies(_capture(_tone(200_000.0)))) == 1
        # 20 kHz away, beyond the 5 kHz tolerance: a different signal entirely.
        assert len(detector.detect_anomalies(_capture(_tone(220_000.0)))) == 1

    def test_score_saturates_for_any_real_signal(self):
        """Documents the score mapping: 3 dB -> 0, 23 dB -> 1.0.

        A synthetic tone clears 23 dB by a wide margin, so nearly every genuine
        detection scores 1.0. The score is therefore a coarse gate rather than a
        graded confidence for clean signals.
        """
        detector = _detector()
        anomaly = detector.detect_anomalies(_capture(_tone(200_000.0)))[0]
        assert anomaly.confidence_score == pytest.approx(1.0)

    def test_allowed_frequency_range_is_enforced(self):
        detector = _detector()
        detector.set_allowed_frequency_range(150e6, 200e6)
        assert detector.detect_anomalies(_capture(_tone(200_000.0))) == []

    def test_reference_gain_shifts_the_reported_level(self):
        plain = _detector().detect_anomalies(_capture(_tone(200_000.0), gain=30.0))
        referenced = _detector()
        referenced.set_reference_gain(20.0)
        normalised = referenced.detect_anomalies(_capture(_tone(200_000.0), gain=30.0))

        assert plain[0].metadata["gain_adjust_db"] == 0.0
        # Reference is 20 dB, capture ran at 30 dB, so adjust by -10 dB.
        assert normalised[0].metadata["gain_adjust_db"] == pytest.approx(-10.0)

    def test_threshold_info_is_internally_consistent(self):
        detector = _detector()
        detector.set_threshold(0.9)
        info = detector.get_threshold_info()
        assert info["anomaly_threshold"] == 0.9
        assert "gating" in info and "novelty" in info

    def test_severity_bands_do_not_depend_on_threshold(self):
        bands = AnomalyDetectionUseCase.severity_bands(0.95)
        assert "0.75" in bands["medium"]
        assert "0.85" in bands["high"]


# ---------------------------------------------------------------- feature sets


class TestFeatureExtraction:
    def test_feature_count_matches_names(self):
        extractor = FeatureExtractor(lite_mode=False)
        n = 4096
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        result = extractor.extract_features(iq, 144.5e6)
        assert len(result.features) == len(result.feature_names) == 12

    def test_lite_mode_returns_two_features(self):
        extractor = FeatureExtractor(lite_mode=True)
        n = 1024
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        assert len(extractor.extract_features(iq, 144.5e6).features) == 2

    def test_papr_is_reported_in_db(self):
        """A raw linear ratio gave values around 10^4, which is unreadable."""
        extractor = FeatureExtractor(lite_mode=False)
        n = 8192
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        features = dict(
            zip(
                extractor.feature_names,
                extractor.extract_features(iq, 144.5e6).features,
            )
        )
        assert 0 < features["papr_db"] < 40

    def test_differential_phase_statistics_separate_carrier_from_noise(self):
        """Circular statistics must be computed on the differential phase.

        On absolute phase a 50 kHz tone sweeps ~200 cycles over the window, so
        its phase distribution is uniform and the statistic collapses to the
        same near-zero value noise gives. Differencing first is what makes the
        feature discriminate.
        """
        extractor = FeatureExtractor(lite_mode=False)
        n = 8192
        rng = np.random.default_rng(1)
        t = np.arange(n) / SAMPLE_RATE

        noise = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        carrier = (0.4 * np.exp(2j * np.pi * 50e3 * t)).astype(np.complex64)

        def resultant(samples):
            values = dict(
                zip(
                    extractor.feature_names,
                    extractor.extract_features(samples, 144.5e6).features,
                )
            )
            return values["phase_resultant"]

        assert resultant(carrier) > 0.9
        assert resultant(noise) < 0.2

    def test_band_energy_ratio_differs_between_signals(self):
        """It must be a frequency-domain quantity, not a time-domain slice."""
        extractor = FeatureExtractor(lite_mode=False)
        n = 8192
        t = np.arange(n) / SAMPLE_RATE

        narrow = (0.4 * np.exp(2j * np.pi * 0 * t)).astype(np.complex64)
        offcentre = (0.4 * np.exp(2j * np.pi * 700e3 * t)).astype(np.complex64)

        def ratio(samples):
            values = dict(
                zip(
                    extractor.feature_names,
                    extractor.extract_features(samples, 144.5e6).features,
                )
            )
            return values["band_energy_ratio"]

        assert ratio(narrow) > 0.9
        assert ratio(offcentre) < 0.3

    def test_autocorrelation_is_scale_free(self):
        """It must not simply track sample count, as the old one did."""
        extractor = FeatureExtractor(lite_mode=False)
        n = 32768
        rng = np.random.default_rng(1)
        iq = (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)

        def autocorr(samples):
            values = dict(
                zip(
                    extractor.feature_names,
                    extractor.extract_features(samples, 144.5e6).features,
                )
            )
            return values["cyclo_autocorr"]

        short = autocorr(iq[:4096])
        long = autocorr(iq)
        # Both near zero for uncorrelated noise, regardless of length.
        assert abs(short) < 0.2
        assert abs(long) < 0.2

    def test_empty_input_raises(self):
        from spectrum_alert.core.exceptions import FeatureExtractionError

        with pytest.raises(FeatureExtractionError):
            FeatureExtractor().extract_features(np.array([]), 144.5e6)


# ------------------------------------------------------------- domain regression


class TestDomainModels:
    def test_session_enum_values_are_enums(self):
        """use_enum_values under Pydantic v2 left defaulted fields as enums.

        Callers doing .value on a str raised AttributeError from
        get_session_status().
        """
        session = MonitoringSession(
            name="test",
            frequency_range_hz=(144e6, 146e6),
            detection_mode=DetectionMode.LITE,
        )
        assert session.detection_mode is DetectionMode.LITE
        assert session.status is MonitoringStatus.STARTING
        assert session.detection_mode.value == "lite"
        assert session.status.value == "starting"

    def test_session_rejects_inverted_range(self):
        with pytest.raises(Exception):
            MonitoringSession(
                name="test",
                frequency_range_hz=(146e6, 144e6),
                detection_mode=DetectionMode.LITE,
            )

    def test_session_is_active(self):
        session = MonitoringSession(
            name="test",
            frequency_range_hz=(144e6, 146e6),
            detection_mode=DetectionMode.LITE,
        )
        assert session.is_active is True


# --------------------------------------------------------------- SDR interface


class _FakeSDR(SDRInterface):
    def __init__(self):
        super().__init__()
        self.samples = np.ones(64, dtype=np.complex64)
        self.opened = 0

    def open(self):
        self.opened += 1
        self._is_open = True

    def close(self):
        self._is_open = False

    def read_samples(self, num_samples):
        return self.samples[:num_samples]

    def set_sample_rate(self, sample_rate):
        self._sample_rate = sample_rate

    def set_center_freq(self, frequency):
        self._center_freq = frequency

    def set_gain(self, gain):
        self._gain = gain


class TestSafeOperation:
    def test_opens_if_closed(self):
        sdr = _FakeSDR()
        with sdr.safe_operation():
            pass
        assert sdr.opened == 1

    def test_failure_is_reraised_after_recovery(self):
        """The old version swallowed the exception after a 'successful' reopen.

        The contract under test is that the caller still sees the error; whether
        a reopen was attempted in between is an implementation detail.
        """
        sdr = _FakeSDR()
        with pytest.raises(ValueError, match="boom"):
            with sdr.safe_operation():
                raise ValueError("boom")

    def test_recovery_is_skipped_when_disabled(self):
        sdr = _FakeSDR()
        with pytest.raises(ValueError):
            with sdr.safe_operation(recover=False):
                raise ValueError("boom")
        # Only the initial open happened.
        assert sdr.opened == 1

    def test_close_on_exit_closes_the_device(self):
        """Without this the device leaked on the success path."""
        sdr = _FakeSDR()
        with sdr.safe_operation(close_on_exit=True):
            assert sdr.is_open
        assert sdr.is_open is False