"""Tests for the geolocation HTTP API.

These exercise the router against mocked hardware: the point is the request
contract, validation, and error reporting, none of which should require a dongle.
"""


import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from spectrum_alert.core.exceptions import SDRError
from spectrum_alert.core.services.calibration import CalibrationTable
from spectrum_alert.core.services.propagation import distance_to_rssi, haversine_meters
from spectrum_alert.core.services.spectrum_analysis import ScanResult
from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
from spectrum_alert.web.geolocate import (
    GeolocationService,
    ReceiverPayload,
    build_router,
)


class FakeScanner:
    """Stands in for :class:`SpectrumScanner` with deterministic levels.

    Levels follow the path-loss model from a fixed emitter position, so the
    triangulation endpoint can be checked against a known answer.
    """

    EMITTER = (35.100, 33.350)

    # Must match frequency_default_path_loss(446.1 MHz), since that is what the
    # endpoint uses when the request does not override the model. Using a
    # different exponent here would make every reading consistent under a
    # *wrong* model and the fix would land off-target for that reason alone.
    EMITTER_FREQUENCY = 446.1e6
    REFERENCE = -45.0
    EXPONENT = 3.5

    def __init__(self, fail_ids=()):
        self.fail_ids = set(fail_ids)
        self.scans = []

    def _level(self, receiver, center_hz, span_hz, bins=64, integration=1.0):
        distance = haversine_meters(*self.EMITTER, receiver.lat, receiver.lon)
        # No deliberate offset: a uniform bias would shrink every modelled
        # distance by the same factor and bias the fix toward the centroid,
        # which would test the bias rather than the pipeline.
        peak = distance_to_rssi(distance, self.REFERENCE, self.EXPONENT)
        floor = peak - 45.0
        step = span_hz / max(1, bins - 1)
        start = center_hz - span_hz / 2.0
        points = []
        for i in range(bins):
            hz = start + i * step
            points.append({"hz": hz, "dbm": peak if abs(hz - center_hz) < step else floor})
        return ScanResult(
            center_hz=center_hz,
            span_hz=span_hz,
            bin_hz=step,
            points=points,
            peak_dbfs=peak,
            noise_floor_dbfs=floor,
            integration_seconds=integration,
            source="rtl_power",
        )

    def scan(self, receiver, center_hz, span_hz, bin_hz=None, integration_seconds=1.0, bins=64):
        self.scans.append(receiver.id)
        if receiver.id in self.fail_ids:
            # SDRError, matching the real scanner: the endpoint handles that
            # type specifically rather than masking every exception as
            # "hardware unavailable".
            raise SDRError(f"simulated hardware failure for {receiver.id}")
        if receiver.manual_rssi_db is not None:
            return ScanResult(
                center_hz=center_hz, span_hz=span_hz, bin_hz=span_hz / 63,
                points=[{"hz": center_hz, "dbm": receiver.manual_rssi_db}],
                peak_dbfs=receiver.manual_rssi_db,
                noise_floor_dbfs=receiver.manual_rssi_db,
                source="manual_override", warning="manual level pinned",
            )
        return self._level(receiver, center_hz, span_hz, bins, integration_seconds)

    async def scan_async(self, *args, **kwargs):
        return self.scan(*args, **kwargs)

    async def scan_all_async(self, receivers, *args, **kwargs):
        results = {}
        for receiver in receivers:
            try:
                results[receiver.id] = self.scan(receiver, *args, **kwargs)
            except Exception:
                continue
        return results

    async def check_async(self, receiver, center_hz):
        try:
            scan = self.scan(receiver, center_hz, 64_000.0, bins=8)
        except Exception as exc:
            return {"id": receiver.id, "ok": False, "message": str(exc)}
        return {"id": receiver.id, "ok": True, "message": f"peak {scan.peak_dbfs:.1f} dBFS"}


@pytest.fixture
def service(tmp_path):
    store = ReceiverStore(str(tmp_path / "receivers.json"))
    calibration = CalibrationTable(str(tmp_path / "calibration.json"))
    return GeolocationService(store, calibration, FakeScanner())


@pytest.fixture
def client(service):
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app)


def _add(client, **overrides):
    payload = {
        "id": "r1", "name": "Alpha", "mode": "local",
        "lat": 35.130, "lon": 33.310, "gainDb": 30.0,
    }
    payload.update(overrides)
    return client.post("/api/sdrs", json=payload)


class TestReceiverEndpoints:
    def test_add_and_list(self, client):
        assert _add(client).status_code == 200
        listed = client.get("/api/sdrs").json()["sdrs"]
        assert len(listed) == 1
        assert listed[0]["name"] == "Alpha"

    def test_upsert_replaces_by_id(self, client):
        _add(client)
        _add(client, name="Alpha renamed")
        listed = client.get("/api/sdrs").json()["sdrs"]
        assert len(listed) == 1
        assert listed[0]["name"] == "Alpha renamed"

    def test_rtltcp_requires_host(self, client):
        response = _add(client, id="r2", mode="rtltcp", host=None)
        assert response.status_code == 400
        assert "host" in response.json()["detail"]

    def test_invalid_latitude_is_rejected(self, client):
        # Rejected by the request model, so 422 rather than the 400 used for
        # semantic failures such as a missing host.
        assert _add(client, lat=95.0).status_code == 422

    def test_missing_name_is_rejected(self, client):
        assert _add(client, name="").status_code == 422

    def test_create_without_an_id_generates_one(self, client):
        """The UI's Add Receiver button does not supply an id.

        The domain model requires one, so the service must mint it; otherwise
        every create fails, which is the normal path through the UI.
        """
        response = client.post(
            "/api/sdrs", json={"name": "No Id", "mode": "local", "lat": 35.1, "lon": 33.3}
        )
        assert response.status_code == 200
        created = response.json()["sdr"]
        assert created["id"]
        assert [r["id"] for r in client.get("/api/sdrs").json()["sdrs"]] == [created["id"]]

    def test_create_ids_are_unique(self, client):
        first = client.post("/api/sdrs", json={"name": "A", "lat": 35.1, "lon": 33.3})
        second = client.post("/api/sdrs", json={"name": "B", "lat": 35.2, "lon": 33.4})
        assert first.json()["sdr"]["id"] != second.json()["sdr"]["id"]

    def test_delete(self, client):
        _add(client)
        assert client.delete("/api/sdrs/r1").status_code == 200
        assert client.get("/api/sdrs").json()["sdrs"] == []

    def test_delete_unknown_is_404(self, client):
        assert client.delete("/api/sdrs/nope").status_code == 404

    def test_clear_all(self, client):
        _add(client)
        _add(client, id="r2", name="Bravo", lat=35.08, lon=33.38)
        assert client.delete("/api/sdrs").status_code == 200
        assert client.get("/api/sdrs").json()["sdrs"] == []

    def test_check_endpoint(self, client):
        _add(client)
        response = client.post("/api/sdrs/r1/check", json={"centerHz": 446.1e6})
        assert response.status_code == 200
        assert response.json()["ok"] is True

    def test_check_unknown_receiver_is_404(self, client):
        assert client.post("/api/sdrs/nope/check", json={"centerHz": 446.1e6}).status_code == 404


class TestSpectrumEndpoint:
    def test_get_is_rejected(self, client):
        """A GET cannot select a receiver, so it must not silently pick one."""
        assert client.get("/api/spectrum").status_code == 405

    def test_hardware_failure_is_503_not_500(self, client, service):
        """Absent hardware is expected, so it must not surface as a server fault.

        Left unhandled this returned 500 "Internal Server Error", which tells an
        operator nothing about what to fix.
        """
        _add(client)
        service.scanner = FakeScanner(fail_ids={"r1"})
        response = client.post("/api/spectrum", json={"centerHz": 446.1e6, "bins": 64})
        assert response.status_code == 503
        assert "Alpha" in response.json()["detail"]

    def test_scan_requires_a_receiver(self, client):
        response = client.post("/api/spectrum", json={"centerHz": 446.1e6})
        assert response.status_code == 400

    def test_scan_returns_points(self, client):
        _add(client)
        payload = client.post(
            "/api/spectrum", json={"centerHz": 446.1e6, "spanHz": 200e3, "bins": 64}
        ).json()
        assert len(payload["points"]) == 64
        assert payload["peakDbm"] > payload["noiseFloorDbm"]
        assert payload["source"] == "rtl_power"

    def test_scan_unknown_receiver_is_404(self, client):
        _add(client)
        response = client.post("/api/spectrum", json={"centerHz": 446.1e6, "sdrId": "nope"})
        assert response.status_code == 404

    def test_bins_are_bounded(self, client):
        _add(client)
        assert client.post(
            "/api/spectrum", json={"centerHz": 446.1e6, "bins": 100000}
        ).status_code == 422


class TestTriangulation:
    def _three_receivers(self, client):
        _add(client, id="r1", name="A", lat=35.130, lon=33.310)
        _add(client, id="r2", name="B", lat=35.080, lon=33.380)
        _add(client, id="r3", name="C", lat=35.160, lon=33.340)

    def test_requires_two_receivers(self, client):
        _add(client)
        response = client.post("/api/triangulate", json={"frequencyHz": 446.1e6})
        assert response.status_code == 400

    def test_locates_a_known_emitter(self, client):
        self._three_receivers(client)
        payload = client.post(
            "/api/triangulate", json={"frequencyHz": 446.1e6}
        ).json()

        estimate = payload["estimate"]
        error = haversine_meters(
            estimate["lat"], estimate["lon"], *FakeScanner.EMITTER
        )
        assert error < 3000, f"fix was {error:.0f} m from the truth"
        assert payload["model"]["band"] == "uhf"
        assert len(payload["readings"]) == 3

    def test_each_reading_reports_its_measurement(self, client):
        self._three_receivers(client)
        readings = client.post(
            "/api/triangulate", json={"frequencyHz": 446.1e6}
        ).json()["readings"]
        for reading in readings:
            assert "noiseFloorDbm" in reading
            assert "prominenceDb" in reading
            assert "reliability" in reading
            assert "source" in reading

    def test_warns_when_a_receiver_fails(self, client, service):
        self._three_receivers(client)
        service.scanner = FakeScanner(fail_ids={"r2"})
        payload = client.post("/api/triangulate", json={"frequencyHz": 446.1e6}).json()
        assert "r2" not in [r["id"] for r in payload["readings"]]
        assert any("B: scan failed" in w for w in payload["measurement"]["warnings"])

    def test_warns_with_only_two_receivers(self, client):
        _add(client, id="r1", name="A", lat=35.130, lon=33.310)
        _add(client, id="r2", name="B", lat=35.080, lon=33.380)
        payload = client.post("/api/triangulate", json={"frequencyHz": 446.1e6}).json()
        warnings = " ".join(payload["measurement"]["warnings"])
        assert "underconstrained" in warnings or "Only 2" in warnings

    def test_out_of_range_exponent_falls_back(self, client):
        self._three_receivers(client)
        payload = client.post(
            "/api/triangulate", json={"frequencyHz": 446.1e6, "pathLossExponent": 12.0}
        ).json()
        assert payload["model"]["pathLossExponent"] < 12.0
        assert any("default" in w for w in payload["measurement"]["warnings"])

    def test_explicit_model_is_respected(self, client):
        self._three_receivers(client)
        payload = client.post(
            "/api/triangulate",
            json={
                "frequencyHz": 446.1e6,
                "pathLossExponent": FakeScanner.EXPONENT,
                "referenceRssiDbm": FakeScanner.REFERENCE,
            },
        ).json()
        assert payload["model"]["pathLossExponent"] == FakeScanner.EXPONENT
        assert payload["model"]["referenceRssiDbm"] == FakeScanner.REFERENCE

    def test_all_receivers_failing_is_503(self, client, service):
        """Every receiver unreadable means unavailable hardware, not a bad request."""
        self._three_receivers(client)
        service.scanner = FakeScanner(fail_ids={"r1", "r2", "r3"})
        response = client.post("/api/triangulate", json={"frequencyHz": 446.1e6})
        assert response.status_code == 503
        assert "scanned" in response.json()["detail"]


class TestTriangulateFromReadings:
    def _body(self):
        return {
            "readings": [
                {"id": "a", "name": "A", "lat": 35.130, "lon": 33.310, "rssiDbm": -70.0},
                {"id": "b", "name": "B", "lat": 35.080, "lon": 33.380, "rssiDbm": -78.0},
                {"id": "c", "name": "C", "lat": 35.160, "lon": 33.340, "rssiDbm": -73.0},
            ],
            "referenceRssiDbm": -40.0,
            "pathLossExponent": 3.5,
        }

    def test_returns_an_estimate(self, client):
        payload = client.post("/api/triangulate/readings", json=self._body()).json()
        assert payload["estimate"]["anchorCount"] == 3
        assert payload["model"]["pathLossExponent"] == 3.5

    def test_includes_a_likelihood_grid_on_request(self, client):
        body = dict(self._body(), includeLikelihood=True)
        payload = client.post("/api/triangulate/readings", json=body).json()
        grid = payload["likelihood"]["grid"]
        assert len(grid) == 36
        assert len(grid[0]) == 36

    def test_requires_two_readings(self, client):
        body = dict(self._body(), readings=self._body()["readings"][:1])
        assert client.post("/api/triangulate/readings", json=body).status_code == 400

    def test_reliability_is_range_checked(self, client):
        body = dict(self._body())
        body["readings"][0]["reliability"] = 5.0
        assert client.post("/api/triangulate/readings", json=body).status_code == 422


class TestCalibrationEndpoints:
    def test_get_empty(self, client):
        assert client.get("/api/calibration").json()["offsets"] == {}

    def test_set_and_read(self, client):
        _add(client)
        response = client.post(
            "/api/calibration",
            json={"receiverId": "r1", "frequencyHz": 446.1e6, "offsetDb": 3.5},
        )
        assert response.status_code == 200
        offsets = client.get("/api/calibration").json()["offsets"]
        assert offsets["r1"]["k446100"] == pytest.approx(3.5)

    def test_offsets_are_clamped(self, client):
        _add(client)
        client.post(
            "/api/calibration",
            json={"receiverId": "r1", "frequencyHz": 446.1e6, "offsetDb": 999.0},
        )
        offsets = client.get("/api/calibration").json()["offsets"]
        assert abs(offsets["r1"]["k446100"]) <= 20.0

    def test_clear_receiver(self, client):
        _add(client)
        client.post(
            "/api/calibration",
            json={"receiverId": "r1", "frequencyHz": 446.1e6, "offsetDb": 3.5},
        )
        assert client.delete("/api/calibration/r1").json()["offsets"] == {}

    def test_clear_all(self, client):
        _add(client)
        client.post(
            "/api/calibration",
            json={"receiverId": "r1", "frequencyHz": 446.1e6, "offsetDb": 3.5},
        )
        assert client.delete("/api/calibration").json()["offsets"] == {}

    def test_deleting_a_receiver_drops_its_offsets(self, client):
        _add(client)
        client.post(
            "/api/calibration",
            json={"receiverId": "r1", "frequencyHz": 446.1e6, "offsetDb": 3.5},
        )
        client.delete("/api/sdrs/r1")
        assert client.get("/api/calibration").json()["offsets"] == {}

    def test_preview_does_not_persist(self, client):
        _add(client)
        payload = client.post("/api/calibrate", json={"apply": False}).json()
        assert payload["applied"] is False
        assert client.get("/api/calibration").json()["offsets"] == {}

    def test_apply_persists(self, client):
        _add(client, id="r1", name="A", lat=35.130, lon=33.310)
        _add(client, id="r2", name="B", lat=35.080, lon=33.380)
        payload = client.post("/api/calibrate", json={"apply": True}).json()
        assert payload["applied"] is True
        assert client.get("/api/calibration").json()["offsets"]


class TestPmrAndBandPlans:
    def _receivers(self, client):
        _add(client, id="r1", name="A", lat=35.130, lon=33.310)
        _add(client, id="r2", name="B", lat=35.080, lon=33.380)

    def test_pmr_scan_covers_all_channels(self, client):
        self._receivers(client)
        payload = client.post("/api/pmr/scan", json={}).json()
        assert payload["plan"]["key"] == "pmr446"
        for receiver in payload["receivers"]:
            assert len(receiver["channels"]) == 16

    def test_pmr_scan_reports_noise_floor(self, client):
        self._receivers(client)
        receiver = client.post("/api/pmr/scan", json={}).json()["receivers"][0]
        assert receiver["noiseFloorDbm"] is not None

    def test_pmr_scan_requires_receivers(self, client):
        assert client.post("/api/pmr/scan", json={}).status_code == 400

    def test_pmr_scan_with_no_readable_receiver_is_503(self, client, service):
        _add(client)
        service.scanner = FakeScanner(fail_ids={"r1"})
        assert client.post("/api/pmr/scan", json={}).status_code == 503

    def test_bandplan_list(self, client):
        keys = [p["key"] for p in client.get("/api/bandplans").json()["bandPlans"]]
        assert "pmr446" in keys

    def test_single_bandplan(self, client):
        payload = client.get("/api/bandplans/pmr446").json()
        assert payload["channelCount"] == 16
        assert payload["channels"][0]["frequencyHz"] == 446_006_250.0

    def test_unknown_bandplan_is_404(self, client):
        assert client.get("/api/bandplans/nope").status_code == 404

    def test_tools(self, client):
        payload = client.get("/api/tools").json()
        assert "rtlPowerAvailable" in payload
        assert payload["receiverCount"] == 0


class TestReceiverPayload:
    def test_rejects_out_of_range_coordinates(self):
        with pytest.raises(ValidationError):
            ReceiverPayload(name="A", lat=100.0, lon=0.0)

    def test_accepts_a_minimal_receiver(self):
        payload = ReceiverPayload(name="A", lat=35.0, lon=33.0)
        assert payload.mode == "local"
        assert payload.enabled is True

# --------------------------------------------------------------------------
# Fingerprinting
# --------------------------------------------------------------------------


class FingerprintScanner:
    """Stands in for hardware by returning synthetic captures per device.

    Each "device" is a different modulation; the noise realisation changes per
    capture exactly as it would on a real receiver, which is what makes the
    averaging step necessary rather than optional.
    """

    DEVICES = {
        "alpha": dict(bandwidth=4e3, compression=1.0),
        "beta": dict(bandwidth=9e3, compression=0.3),
        "gamma": dict(bandwidth=15e3, compression=0.7),
    }

    def __init__(self):
        self.device = "alpha"
        self.calls = 0

    def capture_iq_burst(self, receiver, center_hz, sample_rate_hz, sample_count,
                         captures=1):
        return [
            self.capture_iq(receiver, center_hz, sample_rate_hz, sample_count)
            for _ in range(max(1, int(captures)))
        ]

    def capture_iq(self, receiver, center_hz, sample_rate_hz, sample_count,
                   integration_seconds=0.0):
        import numpy as np

        from spectrum_alert.core.exceptions import SDRError

        # Mirror the real guard: a pinned receiver has no IQ. The guard itself is
        # covered in tests/infrastructure; this checks the endpoint turns the
        # failure into an honest 503 rather than an empty success.
        if getattr(receiver, "manual_rssi_db", None) is not None:
            raise SDRError(f"{receiver.name}: a manual level override supplies no IQ samples.")

        self.calls += 1
        kwargs = self.DEVICES[self.device]
        rng = np.random.default_rng(self.calls)
        n = int(sample_count)
        t = np.arange(n) / float(sample_rate_hz)
        dev = 4e3
        swing = np.sin(2 * np.pi * dev * t) * kwargs["bandwidth"]
        phase = 2 * np.pi * 120e3 * t + np.cumsum(swing) / float(sample_rate_hz)
        envelope = (1 - kwargs["compression"]) + kwargs["compression"] * np.abs(
            np.cos(np.pi * swing / (2 * kwargs["bandwidth"]))
        )
        noise = (rng.normal(size=n) + 1j * rng.normal(size=n)) * 0.12
        return (envelope * np.exp(1j * phase) * 0.6 + noise).astype(np.complex64)


@pytest.fixture
def fp_client(tmp_path):
    from spectrum_alert.web.geolocate import GeolocationService, build_router
    from spectrum_alert.core.services.fingerprinting import FingerprintLibrary
    from spectrum_alert.core.services.calibration import CalibrationTable
    from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore
    from fastapi import FastAPI

    scanner = FingerprintScanner()
    service = GeolocationService(
        ReceiverStore(str(tmp_path / "receivers.json")),
        CalibrationTable(),
        scanner,
        FingerprintLibrary(str(tmp_path / "fingerprints.json")),
    )
    app = FastAPI()
    app.include_router(build_router(service))
    return TestClient(app), scanner, service


def _fp_receiver(client, receiver_id):
    client.post("/api/sdrs", json={
        "id": receiver_id, "name": receiver_id.upper(), "mode": "local",
        "lat": 51.5, "lon": -2.6,
    })


class TestFingerprintEndpoints:
    def test_empty_library_reports_zero_entries(self, fp_client):
        client, _, _ = fp_client
        payload = client.get("/api/fingerprints").json()
        assert payload["entries"] == []
        assert len(payload["featureNames"]) == 12

    def test_capture_enrols_when_a_label_is_given(self, fp_client):
        client, scanner, _ = fp_client
        _fp_receiver(client, "rx0")
        scanner.device = "alpha"
        response = client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6, "label": "Base station",
            "vendor": "Acme", "model": "TX-1",
        })
        assert response.status_code == 200
        payload = response.json()
        assert payload["enrolled"] is not None
        assert payload["enrolled"]["label"] == "Base station"
        assert payload["enrolled"]["vendor"] == "Acme"
        # librarySize is measured before the new entry is added.
        assert payload["librarySize"] == 0
        assert len(client.get("/api/fingerprints").json()["entries"]) == 1

    def test_known_device_is_identified(self, fp_client):
        client, scanner, _ = fp_client
        _fp_receiver(client, "rx0")
        for index, device in enumerate(("alpha", "beta")):
            _fp_receiver(client, f"rx{index}")
            scanner.device = device
            client.post("/api/fingerprints/capture", json={
                "sdrId": f"rx{index}", "frequencyHz": 145.5e6,
                "label": f"TX-{device}",
            })

        scanner.device = "alpha"
        result = client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6,
        }).json()
        assert result["identified"] is True, result["reason"]
        assert result["match"]["label"] == "TX-alpha"
        assert 0.0 < result["confidence"] <= 1.0

    def test_unknown_device_is_not_identified(self, fp_client):
        client, scanner, _ = fp_client
        _fp_receiver(client, "rx0")
        scanner.device = "alpha"
        client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6, "label": "TX-alpha",
        })

        scanner.device = "gamma"
        result = client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6,
        }).json()
        assert result["identified"] is False
        assert result["status"] in {"no-match", "ambiguous"}

    def test_pinned_receiver_cannot_be_fingerprinted(self, fp_client):
        """A pinned level has no IQ, so this must fail loudly and explain why."""
        client, _, _ = fp_client
        client.post("/api/sdrs", json={
            "id": "pinned", "name": "Pinned", "mode": "local",
            "lat": 51.5, "lon": -2.6, "manualRssiDb": -70.0,
        })
        response = client.post("/api/fingerprints/capture", json={
            "sdrId": "pinned", "frequencyHz": 145.5e6,
        })
        assert response.status_code == 503
        assert "manual level override" in response.json()["detail"]
        assert client.get("/api/fingerprints").json()["entries"] == []

    def test_unknown_receiver_is_404(self, fp_client):
        client, _, _ = fp_client
        assert client.post("/api/fingerprints/capture", json={
            "sdrId": "nope", "frequencyHz": 145.5e6,
        }).status_code == 404

    def test_capture_validates_its_arguments(self, fp_client):
        client, _, _ = fp_client
        _fp_receiver(client, "rx0")
        assert client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 0,
        }).status_code == 422
        assert client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6, "captures": 99,
        }).status_code == 422

    def test_library_entries_can_be_listed_and_deleted(self, fp_client):
        client, scanner, service = fp_client
        _fp_receiver(client, "rx0")
        scanner.device = "alpha"
        client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6, "label": "TX-alpha",
        })
        entry = client.get("/api/fingerprints").json()["entries"][0]
        assert client.delete(f"/api/fingerprints/{entry['entryId']}").status_code == 200
        assert client.get("/api/fingerprints").json()["entries"] == []
        assert client.delete(f"/api/fingerprints/{entry['entryId']}").status_code == 404

    def test_enrolled_signature_is_persisted_to_the_configured_path(
        self, fp_client, tmp_path
    ):
        """Regression: an empty library is falsy, so `library or FingerprintLibrary()`
        threw away the configured path and nothing was ever written to disk.
        """
        client, scanner, service = fp_client
        assert len(service.fingerprints) == 0
        _fp_receiver(client, "rx0")
        scanner.device = "alpha"
        client.post("/api/fingerprints/capture", json={
            "sdrId": "rx0", "frequencyHz": 145.5e6, "label": "TX-alpha",
        })
        stored = tmp_path / "fingerprints.json"
        assert stored.exists(), "enrolment was not written to disk"

        from spectrum_alert.core.services.fingerprinting import FingerprintLibrary
        assert [e.label for e in FingerprintLibrary.load(str(stored)).entries] == [
            "TX-alpha"
        ]
