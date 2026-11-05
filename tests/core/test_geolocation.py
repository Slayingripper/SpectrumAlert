"""Tests for the multi-receiver geolocation and calibration services."""

import math

import numpy as np
import pytest

from spectrum_alert.core.domain.receivers import ReceiverMode, SDRReceiver, receiver_spread
from spectrum_alert.core.services import band_plans
from spectrum_alert.core.services.calibration import (
    MAX_ABS_OFFSET_DB,
    CalibrationTable,
    compute_relative_offsets,
    fit_path_loss_exponent,
    frequency_key,
    parse_frequency_key,
)
from spectrum_alert.core.services.propagation import (
    clamp_distance_meters,
    distance_to_rssi,
    frequency_default_path_loss,
    haversine_meters,
    rssi_to_distance_meters,
    validate_path_loss_exponent,
)
from spectrum_alert.core.services.spectrum_analysis import (
    analyze_spectrum,
    compute_psd,
    dbm_at_frequency,
    downsample_points,
    estimate_noise_floor,
)
from spectrum_alert.core.services.triangulation import (
    TriangulationReading,
    likelihood_grid,
    triangulate_position,
)

SAMPLE_RATE = 2_048_000.0


# --------------------------------------------------------------------- helpers


def synth(signal=None, n=32768, noise=0.002, seed=5):
    """Noise plus an optional complex signal."""
    rng = np.random.default_rng(seed)
    base = noise * (rng.normal(size=n) + 1j * rng.normal(size=n))
    if signal is None:
        return base.astype(np.complex64)
    return (base + signal).astype(np.complex64)


def tone(offset_hz, amplitude=0.5, n=32768):
    t = np.arange(n) / SAMPLE_RATE
    return amplitude * np.exp(2j * np.pi * offset_hz * t)


# ----------------------------------------------------------------- propagation


class TestPropagation:
    def test_round_trip(self):
        for rssi in (-50.0, -70.0, -90.0):
            for n in (2.0, 3.0, 4.0):
                for p0 in (-30.0, -45.0):
                    distance = rssi_to_distance_meters(rssi, p0, n)
                    assert distance_to_rssi(distance, p0, n) == pytest.approx(rssi, abs=1e-9)

    def test_distance_decreases_with_rssi(self):
        assert rssi_to_distance_meters(-60, -40, 3) < rssi_to_distance_meters(-80, -40, 3)

    def test_higher_exponent_shortens_reach(self):
        # d = 10 ** ((P0 - RSSI) / (10n)): a larger exponent divides by more,
        # so the same RSSI implies a shorter distance.
        assert rssi_to_distance_meters(-70, -40, 4.0) < rssi_to_distance_meters(-70, -40, 2.0)

    def test_clamp_bounds(self):
        assert clamp_distance_meters(0.0) == 0.5
        assert clamp_distance_meters(1e12) == 500_000.0
        assert clamp_distance_meters(1234.0) == 1234.0

    @pytest.mark.parametrize(
        "hz,expected_n",
        [(50e6, 3.2), (200e6, 3.0), (446e6, 3.5), (2400e6, 4.0)],
    )
    def test_band_defaults(self, hz, expected_n):
        defaults = frequency_default_path_loss(hz)
        assert defaults.exponent == pytest.approx(expected_n)

    def test_validate_rejects_unphysical(self):
        # 0.5 is steeper than free space and 9 is shallower than nothing.
        assert validate_path_loss_exponent(0.5, 446e6) == 3.5
        assert validate_path_loss_exponent(9.0, 446e6) == 3.5
        assert validate_path_loss_exponent(2.5, 446e6) == 2.5

    def test_haversine_known_distance(self):
        # Nicosia to Limassol is roughly 60 km.
        distance = haversine_meters(35.1694, 33.3828, 34.7071, 33.0223)
        assert 55_000 < distance < 65_000

    def test_haversine_zero_for_same_point(self):
        assert haversine_meters(35.0, 33.0, 35.0, 33.0) == pytest.approx(0.0, abs=1e-6)


# ------------------------------------------------------------ spectrum analysis


class TestSpectrumAnalysis:
    def test_psd_finds_exact_tone(self):
        sig = synth(tone(200_000.0))
        freqs, dbfs = compute_psd(sig, SAMPLE_RATE, center_hz=100e6)
        peak = freqs[int(np.argmax(dbfs))]
        assert peak == pytest.approx(100.2e6, abs=2.0)

    def test_psd_is_monotonic_in_frequency(self):
        freqs, _ = compute_psd(synth(tone(0.0)), SAMPLE_RATE, center_hz=100e6)
        assert np.all(np.diff(freqs) > 0)

    def test_psd_reports_dc_at_the_centre(self):
        """A DC-offset signal peaks at the capture centre.

        Notching DC is the detector's job (see AnomalyDetectionUseCase's
        dc_exclude_hz), not the PSD's: the PSD reports what is actually there.
        """
        n = 32768
        rng = np.random.default_rng(2)
        dc = 0.5 + 0.01 * (rng.normal(size=n) + 1j * rng.normal(size=n))
        freqs, dbfs = compute_psd(dc.astype(np.complex64), SAMPLE_RATE, center_hz=100e6)
        peak = freqs[int(np.argmax(dbfs))]
        assert peak == pytest.approx(100e6, abs=2.0)

    def test_noise_floor_ignores_a_wideband_signal(self):
        """The floor must not be dragged up by the signal it sits next to.

        This is why the floor comes from the quietest slice rather than the
        median: a wide emitter otherwise looks weak against its own skirt.
        """
        n = 32768

        # Band-limited noise over the middle 70 % of the span, built in the
        # frequency domain: a genuinely wideband occupant, which is the case
        # the median estimator gets wrong.
        rng = np.random.default_rng(9)
        spectrum = np.zeros(n, dtype=np.complex128)
        offsets = np.fft.fftfreq(n, d=1.0 / SAMPLE_RATE)
        inside = np.abs(offsets) < 700e3 / 2.0
        # Amplitude high enough that each occupied bin sits well above the
        # per-bin noise level: spreading fixed power over ~11k bins leaves each
        # one below the noise floor, which is not the situation under test.
        spectrum[inside] = 2.0 * (
            rng.normal(size=int(inside.sum())) + 1j * rng.normal(size=int(inside.sum()))
        )
        wide = np.fft.ifft(np.fft.ifftshift(spectrum)).astype(np.complex64)

        noise_only = synth(n=n)

        # compute_psd already returns dBFS; converting again would floor it.
        freqs, db = compute_psd(synth(wide, n=n), SAMPLE_RATE, center_hz=100e6)
        floor = estimate_noise_floor(db)
        median = float(np.median(db))

        # The signal really does occupy a third of the span.
        occupied = int(np.count_nonzero(db > floor + 6))
        assert occupied > 0.25 * db.size, "signal should occupy a large part of the span"

        # The point of the quietest-slice estimator: with the wideband signal
        # present the floor still matches the noise-only floor, whereas the
        # median is dragged up into the signal.
        _, clean_db = compute_psd(noise_only, SAMPLE_RATE, center_hz=100e6)
        noise_floor = estimate_noise_floor(clean_db)

        assert abs(floor - noise_floor) < 3.0
        assert median > floor + 1.0, "median should be pulled up by the signal"

    def test_noise_floor_matches_a_median_based_one_when_quiet(self):
        """With one narrow tone present the two estimators should agree closely."""
        freqs, db = compute_psd(synth(tone(200_000.0)), SAMPLE_RATE, center_hz=100e6)
        quiet_slice = estimate_noise_floor(db)
        median = float(np.median(db))
        assert abs(quiet_slice - median) < 1.0

    def test_dbm_at_frequency_picks_the_strongest_bin(self):
        freqs = np.array([99.99e6, 100.0e6, 100.01e6, 100.02e6])
        values = np.array([-60.0, -40.0, -55.0, -58.0])
        assert dbm_at_frequency(freqs, values, 100.0e6) == -40.0

    def test_dbm_at_frequency_falls_back_to_nearest(self):
        freqs = np.array([100.0e6, 100.01e6])
        values = np.array([-40.0, -50.0])
        # Target far outside the span: nearest bin rather than an error.
        # 100.01 MHz is the closest to 200 MHz.
        assert dbm_at_frequency(freqs, values, 200e6) == -50.0

    def test_analyze_reports_prominence_and_reliability(self):
        freqs, db = compute_psd(synth(tone(150_000.0)), SAMPLE_RATE, center_hz=100e6)
        analysis = analyze_spectrum(freqs, db, 100.15e6)
        assert analysis.prominence_db > 40
        assert analysis.reliability == pytest.approx(1.0)
        assert analysis.overloaded is False

    def test_analyze_handles_empty_input(self):
        analysis = analyze_spectrum([], [], 100e6)
        assert analysis.reliability == 0.0

    def test_downsample_keeps_the_peak(self):
        points = [{"hz": float(i), "dbm": -70.0} for i in range(1000)]
        points[517] = {"hz": 517.0, "dbm": -20.0}
        reduced = downsample_points(points, 64)
        assert len(reduced) <= 64
        assert max(reduced, key=lambda p: p["dbm"])["dbm"] == -20.0

    def test_downsample_passes_through_short_input(self):
        points = [{"hz": 1.0, "dbm": -50.0}]
        assert downsample_points(points, 256) == points


# ------------------------------------------------------------------ calibration


class TestCalibration:
    def test_frequency_key_round_trip(self):
        assert parse_frequency_key(frequency_key(446_100_000)) == 446_100_000

    def test_frequency_key_rounds_to_khz(self):
        # Readings a few Hz apart must share one offset.
        assert frequency_key(446_100_000) == frequency_key(446_100_004)

    def test_offset_for_missing_is_zero(self, tmp_path):
        table = CalibrationTable(str(tmp_path / "cal.json"))
        assert table.offset_for("nobody", 446.1e6) == 0.0

    def test_apply_then_offset(self, tmp_path):
        table = CalibrationTable(str(tmp_path / "cal.json"))
        table.apply("r1", 446.1e6, 3.5)
        assert table.offset_for("r1", 446.1e6) == pytest.approx(3.5)

    def test_offset_is_clamped(self, tmp_path):
        table = CalibrationTable(str(tmp_path / "cal.json"))
        table.apply("r1", 446.1e6, 500.0)
        assert table.offset_for("r1", 446.1e6) == MAX_ABS_OFFSET_DB

    def test_save_and_load_round_trip(self, tmp_path):
        path = str(tmp_path / "cal.json")
        table = CalibrationTable(path)
        table.apply("r1", 446.1e6, -2.25)
        table.save()

        reloaded = CalibrationTable.load(path)
        assert reloaded.offset_for("r1", 446.1e6) == pytest.approx(-2.25)

    def test_corrupt_file_does_not_raise(self, tmp_path):
        path = tmp_path / "cal.json"
        path.write_text("{not json")
        assert CalibrationTable.load(str(path)).offset_for("r1", 446e6) == 0.0

    def test_frequencies_sorted_numerically(self, tmp_path):
        table = CalibrationTable(str(tmp_path / "cal.json"))
        for hz in (88_200_000, 103_700_000, 91_100_000):
            table.apply("r1", hz, 1.0)
        assert table.frequencies_for("r1") == [88_200_000.0, 91_100_000.0, 103_700_000.0]

    def test_relative_offsets_equalise_receivers(self):
        readings = [
            {"receiver_id": "a", "receiver_name": "A", "rssi_dbfs": -70.0},
            {"receiver_id": "b", "receiver_name": "B", "rssi_dbfs": -75.0},
            {"receiver_id": "c", "receiver_name": "C", "rssi_dbfs": -72.0},
        ]
        offsets, rejected = compute_relative_offsets(readings, 446.1e6)
        assert not rejected
        # Median is -72. An offset is added to a reading to bring it onto the
        # median, so a reader above the median needs a negative offset.
        assert offsets["a"]["offset"] == pytest.approx(-2.0)
        assert offsets["b"]["offset"] == pytest.approx(3.0)
        assert offsets["c"]["offset"] == pytest.approx(0.0)

    def test_relative_offsets_reject_an_outlier(self):
        readings = [
            {"receiver_id": "a", "receiver_name": "A", "rssi_dbfs": -70.0},
            {"receiver_id": "b", "receiver_name": "B", "rssi_dbfs": -70.5},
            {"receiver_id": "c", "receiver_name": "C", "rssi_dbfs": -70.2},
            {"receiver_id": "z", "receiver_name": "Z", "rssi_dbfs": -20.0},
        ]
        offsets, rejected = compute_relative_offsets(readings, 446.1e6)
        assert "z" not in offsets
        assert len(rejected) == 1


class TestPathLossFit:
    def _observations(self, slope=-30.0, noise=0.0, seed=1):
        rng = np.random.default_rng(seed)
        observations = []
        for i, distance in enumerate([800.0, 2500.0, 6000.0, 15_000.0, 40_000.0, 90_000.0]):
            # RSSI = P0 - 10*n*log10(d): slope is -10*n, intercept is P0.
            # P0 must be negative; a broadcast transmitter is never above
            # -20 dBm at 1 m, and the fitter rejects the fit otherwise.
            measured = slope * math.log10(distance) - 30.0
            if noise:
                measured += rng.normal(scale=noise)
            observations.append({
                "transmitter": f"T{i}",
                "frequency_hz": 100e6,
                "receiver_id": f"r{i}",
                "receiver_name": f"R{i}",
                "adjusted_dbfs": measured,
                "distance_meters": distance,
            })
        return observations

    def test_recovers_a_known_slope(self):
        # slope = -30 dB/decade => n = 3.0; intercept -30 => P0 = -30 dBm.
        result = fit_path_loss_exponent(self._observations(slope=-30.0))
        assert result.path_loss_exponent == pytest.approx(3.0, abs=0.01)
        assert result.reference_rssi_dbm == pytest.approx(-30.0, abs=0.01)
        assert result.r_squared == pytest.approx(1.0, abs=1e-6)
        assert result.fallback_applied is False

    def test_survives_one_bad_measurement(self):
        observations = self._observations(noise=0.5)
        # A badly multipath-faded receiver, 25 dB low.
        observations[2]["adjusted_dbfs"] -= 25.0
        result = fit_path_loss_exponent(observations)
        assert result.path_loss_exponent == pytest.approx(3.0, abs=0.1)
        assert observations[2]["transmitter"] == "T2"

    def test_falls_back_when_distance_spread_is_too_small(self):
        observations = []
        for i in range(8):
            distance = 5000.0 + i * 5.0  # ratio 1.001, cannot fit a slope
            observations.append({
                "frequency_hz": 446.1e6,
                "adjusted_dbfs": distance_to_rssi(distance, -45.0, 3.5),
                "distance_meters": distance,
                "receiver_id": f"r{i}",
            })
        result = fit_path_loss_exponent(observations)
        # UHF band default, since the observations are at 446 MHz.
        assert result.path_loss_exponent == pytest.approx(3.5)
        assert "similar distances" in (result.warning or "")

    def test_falls_back_on_unphysical_slope(self):
        # RSSI rising with distance is impossible for free-space-ish paths.
        observations = self._observations(slope=+5.0)
        result = fit_path_loss_exponent(observations)
        assert result.fallback_applied is True
        assert 1.5 <= result.path_loss_exponent <= 4.5
        assert result.reference_rssi_dbm <= -20.0

    def test_skips_observations_too_close_to_the_transmitter(self):
        observations = self._observations() + [{
            "transmitter": "near",
            "frequency_hz": 100e6,
            "adjusted_dbfs": -5.0,
            "distance_meters": 3.0,
            "receiver_id": "rn",
        }]
        result = fit_path_loss_exponent(observations)
        assert all(p.transmitter != "near" for p in result.points)
        assert any("too close" in error for error in result.errors)

    def test_empty_input_falls_back(self):
        result = fit_path_loss_exponent([])
        assert result.fallback_applied is True
        assert result.path_loss_exponent == pytest.approx(3.0)  # default: 100 MHz


# --------------------------------------------------------------- triangulation

TX_LAT, TX_LON = 35.100, 33.350
P0 = -30.0
EXPONENT = 3.0
RECEIVER_SITES = [
    ("A", 35.130, 33.310),
    ("B", 35.080, 33.380),
    ("C", 35.160, 33.340),
]


def readings_from_truth(offsets_db=None, reliability=1.0):
    offsets_db = offsets_db or {}
    readings = []
    for name, lat, lon in RECEIVER_SITES:
        distance = haversine_meters(TX_LAT, TX_LON, lat, lon)
        level = distance_to_rssi(distance, P0, EXPONENT) + offsets_db.get(name, 0.0)
        readings.append(
            TriangulationReading(
                receiver_id=name, name=name, lat=lat, lon=lon,
                rssi_db=level, source="rtl_power", reliability=reliability,
            )
        )
    return readings


class TestTriangulation:
    def test_recovers_a_known_position(self):
        estimate = triangulate_position(readings_from_truth(), P0, EXPONENT)
        error = haversine_meters(estimate.lat, estimate.lon, TX_LAT, TX_LON)
        assert error < 1000
        assert estimate.quality in {"good", "fair"}

    def test_reports_residuals_for_every_anchor(self):
        estimate = triangulate_position(readings_from_truth(), P0, EXPONENT)
        assert len(estimate.residuals) == len(RECEIVER_SITES)
        assert estimate.anchor_count == 3

    def test_requires_at_least_two_readings(self):
        with pytest.raises(ValueError):
            triangulate_position(readings_from_truth()[:1], P0, EXPONENT)

    def test_quality_degrades_with_a_faded_receiver(self):
        """A lone faded receiver must not be reported as a good fit.

        The RMS residual barely moves when the optimiser slides toward the bad
        anchor, so the confidence ellipse is what catches it.
        """
        clean = triangulate_position(readings_from_truth(), P0, EXPONENT)
        faded = triangulate_position(
            readings_from_truth({"B": -14.0}), P0, EXPONENT
        )
        assert faded.quality in {"fair", "poor"}
        assert clean.quality == "good"

    def test_two_receivers_are_marked_underconstrained(self):
        estimate = triangulate_position(readings_from_truth()[:2], P0, EXPONENT)
        assert any("underconstrained" in reason for reason in estimate.quality_reasons)

    def test_confidence_ellipse_is_present_and_ordered(self):
        estimate = triangulate_position(readings_from_truth(), P0, EXPONENT)
        confidence = estimate.confidence
        assert confidence.semi_major_meters is not None
        assert confidence.semi_minor_meters is not None
        assert confidence.semi_major_meters >= confidence.semi_minor_meters

    def test_estimate_serialises(self):
        payload = triangulate_position(readings_from_truth(), P0, EXPONENT).to_dict()
        assert set(payload) >= {
            "lat", "lon", "uncertaintyMeters", "quality", "residuals", "confidence"
        }

    def test_low_confidence_source_is_downweighted(self):
        """The same three readings, differing only in the third one's source.

        C's level is the outlier in both cases. When C is a real measurement it
        pulls the solution toward itself; when it comes from a pinned manual
        override it is discounted, so the solution settles nearer the A/B
        consensus.
        """
        anchor_a = ("A", 35.130, 33.310)
        anchor_b = ("B", 35.080, 33.380)
        anchor_c = ("C", 35.160, 33.340)

        def build(source_c):
            return [
                TriangulationReading("A", *anchor_a, -75.0, "rtl_power", 1.0),
                TriangulationReading("B", *anchor_b, -78.0, "rtl_power", 1.0),
                TriangulationReading("C", *anchor_c, -70.0, source_c, 1.0),
            ]

        trusted = triangulate_position(build("rtl_power"), -40.0, 3.5)
        discounted = triangulate_position(build("manual_override"), -40.0, 3.5)

        midpoint = ((anchor_a[1] + anchor_b[1]) / 2.0, (anchor_a[2] + anchor_b[2]) / 2.0)
        trusted_offset = haversine_meters(trusted.lat, trusted.lon, *midpoint)
        discounted_offset = haversine_meters(discounted.lat, discounted.lon, *midpoint)

        assert discounted_offset < trusted_offset

    def test_likelihood_grid_is_normalised(self):
        readings = readings_from_truth()
        grid = likelihood_grid(readings, P0, EXPONENT, (35.0, 33.2, 35.3, 33.5), steps=12)
        assert len(grid) == 12
        assert all(len(row) == 12 for row in grid)
        values = [value for row in grid for value in row]
        assert min(values) == pytest.approx(0.0)
        assert max(values) == pytest.approx(1.0)


# ------------------------------------------------------------------- receivers


class TestSDRReceiver:
    def test_round_trip(self):
        receiver = SDRReceiver(
            id="r1", name="Home", mode="local", lat=35.13, lon=33.31,
            gain_db=30.0, ppm_error=20.0,
        )
        restored = SDRReceiver.from_dict(receiver.to_dict())
        assert restored == receiver

    def test_mode_aliases(self):
        assert SDRReceiver(id="a", name="A", mode="RTL_TCP", lat=0, lon=0).mode is ReceiverMode.RTLTCP
        assert SDRReceiver(id="a", name="A", mode="usb", lat=0, lon=0).mode is ReceiverMode.LOCAL

    def test_coordinates_are_validated(self):
        with pytest.raises(Exception):
            SDRReceiver(id="a", name="A", lat=95.0, lon=0.0)
        with pytest.raises(Exception):
            SDRReceiver(id="a", name="A", lat=0.0, lon=200.0)

    def test_name_is_required(self):
        with pytest.raises(Exception):
            SDRReceiver(id="a", name="  ", lat=0.0, lon=0.0)

    def test_rtltcp_requires_host_and_port(self):
        receiver = SDRReceiver(id="a", name="A", mode="rtltcp", lat=0.0, lon=0.0)
        with pytest.raises(ValueError):
            receiver.validate_transport()

    def test_rtltcp_with_host_validates(self):
        receiver = SDRReceiver(
            id="a", name="A", mode="rtltcp", lat=0.0, lon=0.0, host="h", port=1234
        )
        receiver.validate_transport()

    def test_distance_to(self):
        a = SDRReceiver(id="a", name="A", lat=35.0, lon=33.0)
        b = SDRReceiver(id="b", name="B", lat=35.01, lon=33.0)
        assert 1000 < a.distance_to(b) < 1300

    def test_receiver_spread(self):
        receivers = [
            SDRReceiver(id="a", name="A", lat=35.0, lon=33.0),
            SDRReceiver(id="b", name="B", lat=35.1, lon=33.0),
        ]
        assert receiver_spread(receivers) > 10_000


# ------------------------------------------------------------------ band plans


class TestBandPlans:
    def test_pmr446_channel_count_and_spacing(self):
        plan = band_plans.get_band_plan("pmr446")
        assert len(plan.channels) == 16
        for first, second in zip(plan.channels, plan.channels[1:]):
            assert second.frequency_hz - first.frequency_hz == 12_500.0

    def test_pmr446_fits_one_capture(self):
        plan = band_plans.get_band_plan("pmr446")
        assert plan.span_hz < 2_000_000

    def test_channel_lookup(self):
        channel = band_plans.pmr446_channel_at(446_043_750)
        assert channel is not None and channel.number == 4
        assert band_plans.pmr446_channel_at(900e6) is None

    def test_nearest_channel(self):
        plan = band_plans.get_band_plan("pmr446")
        channel = band_plans.nearest_channel(plan, 446_045_000)
        assert channel.number == 4

    def test_unknown_plan_returns_none(self):
        assert band_plans.get_band_plan("nope") is None

    def test_all_plans_serialise(self):
        for key in band_plans.BAND_PLANS:
            payload = band_plans.get_band_plan(key).to_dict()
            assert payload["channelCount"] == len(payload["channels"])