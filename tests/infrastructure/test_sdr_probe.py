"""Tests for the local SDR probe used by the status surfaces.

The dashboard and the CLI both used to invent hardware: the web endpoint
hardcoded ``"Available"`` plus an ``"RTL-SDR #0"`` device, and the CLI echoed
whatever pyrtlsdr threw. The probe exists so both can tell the truth, so these
tests pin down that it never reports a device that rtl_test did not list.
"""

import subprocess

import pytest

from spectrum_alert.infrastructure.sdr import scanner
from spectrum_alert.infrastructure.sdr.scanner import probe_local_sdr


@pytest.fixture(autouse=True)
def clear_probe_cache(monkeypatch):
    """Every test gets a cold cache; otherwise one probe would serve all."""
    monkeypatch.setattr(scanner, "_probe_cache", {"at": 0.0, "result": None})
    yield


def _run_result(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess(
        args=["rtl_test", "-t"], returncode=returncode, stdout=stdout, stderr=stderr
    )


class TestProbeLocalSdr:
    def test_no_device_is_reported_honestly(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run_result(stdout="No supported devices found.\n"),
        )
        result = probe_local_sdr(force=True)
        assert result["status"] == "no-device"
        assert result["available"] is False
        assert result["devices"] == []
        assert "No RTL-SDR device detected" in result["detail"]

    def test_missing_binary_is_tools_missing(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: False)
        result = probe_local_sdr(force=True)
        assert result["status"] == "tools-missing"
        assert result["available"] is False

    def test_detected_device_is_listed(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)
        output = (
            "Found 1 device(s):\n"
            "  0:  Realtek, RTL2838UHIDIR, SN: 00000001\n"
            "Using device 0: Generic RTL2832U OEM\n"
        )
        monkeypatch.setattr(
            subprocess, "run", lambda *a, **k: _run_result(stdout=output)
        )
        result = probe_local_sdr(force=True)
        assert result["status"] == "available"
        assert result["available"] is True
        assert result["devices"] == [
            {"index": 0, "name": "Realtek, RTL2838UHIDIR, SN: 00000001"}
        ]

    def test_undefined_symbol_is_named_not_generic(self, monkeypatch):
        """A librtlsdr version mismatch must say what is actually wrong.

        The raw loader message ("undefined symbol: rtlsdr_set_dithering")
        tells an operator nothing; the probe must translate it into "your
        installed library is older than what rtl_test was built against".
        """
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)
        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **k: _run_result(
                stderr="/lib/x86_64-linux-gnu/librtlsdr.so: undefined symbol: "
                       "rtlsdr_set_dithering\n"
            ),
        )
        result = probe_local_sdr(force=True)
        assert result["status"] == "error"
        assert result["available"] is False
        assert "rtlsdr_set_dithering" in result["detail"]
        assert "version mismatch" in result["detail"]

    def test_timeout_is_an_error_not_a_hang(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)

        def slow(*a, **k):
            raise subprocess.TimeoutExpired(cmd="rtl_test", timeout=4)

        monkeypatch.setattr(subprocess, "run", slow)
        result = probe_local_sdr(force=True)
        assert result["status"] == "error"
        assert "did not answer" in result["detail"]

    def test_result_is_cached_within_ttl(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)
        calls = {"n": 0}

        def counting_run(*a, **k):
            calls["n"] += 1
            return _run_result(stdout="No supported devices found.\n")

        monkeypatch.setattr(subprocess, "run", counting_run)
        probe_local_sdr()
        probe_local_sdr()
        probe_local_sdr()
        assert calls["n"] == 1, "status polling must not re-run rtl_test every call"

    def test_force_bypasses_the_cache(self, monkeypatch):
        monkeypatch.setattr(scanner, "rtl_test_available", lambda: True)
        calls = {"n": 0}

        def counting_run(*a, **k):
            calls["n"] += 1
            return _run_result(stdout="No supported devices found.\n")

        monkeypatch.setattr(subprocess, "run", counting_run)
        probe_local_sdr()
        probe_local_sdr(force=True)
        assert calls["n"] == 2


class TestCaptureIq:
    """``capture_iq`` feeds fingerprinting, which needs the raw waveform.

    The guard that matters is the manual-override refusal: a pinned level is one
    number, and returning a flat trace for it would produce a signature that
    matches nothing while looking like a transmitter.
    """

    def _receiver(self, **overrides):
        from spectrum_alert.core.domain.receivers import SDRReceiver

        fields = dict(id="r1", name="R1", mode="local", lat=51.5, lon=-2.6)
        fields.update(overrides)
        return SDRReceiver(**fields)

    def test_pinned_level_is_refused(self):
        from spectrum_alert.core.exceptions import SDRError
        from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner

        receiver = self._receiver(manual_rssi_db=-70.0)
        with pytest.raises(SDRError) as excinfo:
            SpectrumScanner().capture_iq(receiver, 145e6, 240e3, 1024)
        assert "manual level override" in str(excinfo.value)

    def test_rtltcp_requires_a_host_and_port(self):
        from spectrum_alert.core.exceptions import SDRError
        from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner

        with pytest.raises(SDRError):
            SpectrumScanner().capture_iq(self._receiver(mode="rtltcp"), 145e6, 240e3, 1024)

    def test_rejects_impossible_arguments(self):
        from spectrum_alert.core.exceptions import SDRError
        from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner

        scanner = SpectrumScanner()
        with pytest.raises(SDRError):
            scanner.capture_iq(self._receiver(), 0, 240e3, 1024)
        with pytest.raises(SDRError):
            scanner.capture_iq(self._receiver(), 145e6, 0, 1024)
        with pytest.raises(SDRError):
            scanner.capture_iq(self._receiver(), 145e6, 240e3, 0)
