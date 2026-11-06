"""Unified spectrum scanner across every receiver transport.

This is the adapter that satisfies
:class:`~spectrum_alert.core.interfaces.ports.SpectrumScanPort`. It selects a
transport per receiver and normalises the result into a single
:class:`~spectrum_alert.core.services.spectrum_analysis.ScanResult`.

Transports:

``rtl_power``
    Shells out to ``rtl_power`` for a locally attached dongle. Preferred: it
    averages internally and reports genuine dBm-ish levels.

``rtl_tcp``
    Streams IQ from a remote ``rtl_tcp`` server and computes the FFT here.

``manual_override``
    The receiver is pinned to a fixed level and cannot be measured. Emits a flat
    scan tagged so downstream weighting can discount it.

Every path returns ``source`` and, when something was degraded, ``warning``.
Nothing fabricates data silently, which is the whole point: an operator needs to
know whether a reading came from hardware.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from spectrum_alert.core.domain.receivers import SDRReceiver, ReceiverMode, ReceiverSource
from spectrum_alert.core.exceptions import SDRError
from spectrum_alert.core.services.spectrum_analysis import (
    ScanResult,
    analyze_spectrum,
    downsample_points,
    flat_spectrum_scan,
    scan_result_from_arrays,
)
from spectrum_alert.infrastructure.sdr.lock import hold_device
from spectrum_alert.infrastructure.sdr.rtl_tcp import RtltcpClient, scan_with_sample_rate

logger = logging.getLogger(__name__)

__all__ = ["SpectrumScanner", "rtl_power_available", "rtl_test_available", "probe_local_sdr"]

#: An RTL2832 tops out around 2.4 MSps, so anything wider is impossible and is
#: clamped rather than silently accepted.
MAX_SAMPLE_RATE_HZ = 2_400_000.0
MIN_SAMPLE_RATE_HZ = 225_000.0

#: Default bins returned to the UI.
DEFAULT_BINS = 256

#: Timeout for one ``rtl_power`` invocation.
RTL_POWER_TIMEOUT_S = 15.0


def rtl_power_available() -> bool:
    """Whether the ``rtl_power`` binary is on PATH."""
    return shutil.which("rtl_power") is not None


def rtl_test_available() -> bool:
    """Whether the ``rtl_test`` binary is on PATH."""
    return shutil.which("rtl_test") is not None


# ---------------------------------------------------------------------
# Device probe
# ---------------------------------------------------------------------

#: How long a probe result stays valid. The web dashboard polls status every
#: few seconds; shelling out to ``rtl_test`` on every poll would keep the USB
#: bus (and the CPU) busy for nothing.
_PROBE_CACHE_TTL_S = 30.0

#: Hard cap on one probe. ``rtl_test -t`` returns in well under a second when
#: a dongle answers; anything longer means the USB subsystem is stuck.
_PROBE_TIMEOUT_S = 4.0

_probe_cache: Dict[str, Any] = {"at": 0.0, "result": None}


def probe_local_sdr(force: bool = False) -> Dict[str, Any]:
    """Report the truth about locally attached RTL-SDR hardware.

    Returns a dict with ``status`` (``available`` / ``no-device`` /
    ``tools-missing`` / ``error``), a human ``detail`` string, and a
    ``devices`` list. Never fabricates a device: when nothing answers, the
    list is empty and the status says why.

    The result is cached for ``_PROBE_CACHE_TTL_S`` seconds because callers
    (the status endpoint) poll far more often than hardware changes.
    """
    now = time.monotonic()
    cached = _probe_cache["result"]
    if not force and cached is not None and now - _probe_cache["at"] < _PROBE_CACHE_TTL_S:
        return cached

    result = _probe_local_sdr_uncached()
    _probe_cache["at"] = now
    _probe_cache["result"] = result
    return result


def _probe_local_sdr_uncached() -> Dict[str, Any]:
    if not rtl_test_available():
        return {
            "status": "tools-missing",
            "available": False,
            "detail": "rtl_test is not installed; install the rtl-sdr package",
            "devices": [],
        }

    try:
        proc = subprocess.run(
            ["rtl_test", "-t"],
            capture_output=True,
            text=True,
            timeout=_PROBE_TIMEOUT_S,
        )
        output = (proc.stdout or "") + (proc.stderr or "")
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "available": False,
            "detail": f"rtl_test did not answer within {_PROBE_TIMEOUT_S:.0f}s; "
                      "the USB subsystem may be stuck",
            "devices": [],
        }
    except OSError as exc:
        return {
            "status": "error",
            "available": False,
            "detail": f"rtl_test could not be executed: {exc}",
            "devices": [],
        }

    if "No supported devices found" in output:
        return {
            "status": "no-device",
            "available": False,
            "detail": "No RTL-SDR device detected. Attach a dongle or add a "
                      "remote rtl_tcp receiver.",
            "devices": [],
        }

    # A library mismatch (binary built against newer librtlsdr headers than
    # the installed shared object) surfaces as a loader error. Name the actual
    # problem instead of reporting a generic failure.
    if "undefined symbol" in output:
        symbol = output.split("undefined symbol:", 1)[1].splitlines()[0].strip()
        return {
            "status": "error",
            "available": False,
            "detail": f"librtlsdr version mismatch: the installed library lacks "
                      f"'{symbol}'. Install a librtlsdr that provides it "
                      f"(or rebuild rtl-sdr against the installed one).",
            "devices": [],
        }

    devices = []
    for line in output.splitlines():
        # rtl_test prints e.g. "  0:  Realtek, RTL2838UHIDIR, SN: 00000001"
        stripped = line.strip()
        if stripped[:1].isdigit() and ":" in stripped:
            index, _, name = stripped.partition(":")
            devices.append({"index": int(index.strip()), "name": name.strip()})

    if devices:
        return {
            "status": "available",
            "available": True,
            "detail": f"{len(devices)} RTL-SDR device(s) detected",
            "devices": devices,
        }

    # rtl_test exited without finding a device but also without its usual
    # message: report the tail of its output so the operator sees what it saw.
    tail = " ".join(output.strip().splitlines()[-2:])[:160] or f"exit code {proc.returncode}"
    return {
        "status": "error",
        "available": False,
        "detail": f"rtl_test gave an unexpected answer: {tail}",
        "devices": [],
    }


class SpectrumScanner:
    """Scans receivers, whatever transport each one uses."""

    def __init__(self, rtl_power_bin: str = "rtl_power"):
        self.rtl_power_bin = rtl_power_bin

    # ------------------------------------------------------------------ scan

    def scan(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        span_hz: float,
        bin_hz: Optional[float] = None,
        integration_seconds: float = 1.0,
        bins: int = DEFAULT_BINS,
    ) -> ScanResult:
        """Produce one scan for ``receiver``.

        Raises :class:`SDRError` when the hardware cannot be reached. A pinned
        ``manual_rssi_db`` never raises; it returns a tagged flat scan.
        """
        center_hz = float(center_hz)
        span_hz = float(span_hz)
        if not math.isfinite(center_hz) or center_hz <= 0:
            raise SDRError("center_hz must be a positive frequency")
        if not math.isfinite(span_hz) or span_hz <= 0:
            raise SDRError("span_hz must be positive")

        if receiver.manual_rssi_db is not None:
            return flat_spectrum_scan(
                center_hz=center_hz,
                span_hz=span_hz,
                level_db=float(receiver.manual_rssi_db),
                bins=bins,
                source=ReceiverSource.MANUAL_OVERRIDE.value,
                warning=(
                    f"Using manual level override ({receiver.manual_rssi_db:.1f} dB) for "
                    f"{receiver.name}; no hardware was read."
                ),
            )

        if receiver.mode is ReceiverMode.RTLTCP:
            return self._scan_rtl_tcp(receiver, center_hz, span_hz, integration_seconds, bins)
        return self._scan_local(receiver, center_hz, span_hz, bin_hz, integration_seconds, bins)

    def scan_all(
        self,
        receivers: Sequence[SDRReceiver],
        center_hz: float,
        span_hz: float,
        bin_hz: Optional[float] = None,
        integration_seconds: float = 1.0,
        bins: int = DEFAULT_BINS,
    ) -> Dict[str, ScanResult]:
        """Scan several receivers, one at a time.

        Sequential on purpose: parallel scans over the same physical dongle would
        contend for USB bandwidth and produce corrupt spectra. Remote receivers
        on different hosts would be safe to parallelise, but serial is
        predictable and fast enough for the handful of sites involved.
        """
        results: Dict[str, ScanResult] = {}
        for receiver in receivers:
            started = time.monotonic()
            try:
                results[receiver.id] = self.scan(
                    receiver, center_hz, span_hz, bin_hz, integration_seconds, bins
                )
                logger.debug(
                    "scanned %s in %.0f ms", receiver.name, (time.monotonic() - started) * 1000
                )
            except SDRError as exc:
                logger.warning("scan failed for %s: %s", receiver.name, exc)
        return results

    # --------------------------------------------------------------- transports

    def _device_key(self, receiver: SDRReceiver) -> str:
        """Lock key. Remote receivers key on host:port, locals on the dongle index."""
        if receiver.mode is ReceiverMode.RTLTCP:
            return f"rtltcp:{receiver.host}:{receiver.port}"
        return f"usb:{receiver.device_index or 0}"

    def _scan_rtl_tcp(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        span_hz: float,
        integration_seconds: float,
        bins: int,
    ) -> ScanResult:
        if not receiver.host or not receiver.port:
            raise SDRError(f"{receiver.name}: rtl_tcp mode requires host and port")

        sample_rate = self._clamp_sample_rate(span_hz * 1.25)
        # More samples than the FFT can use is wasted bandwidth; the FFT is capped
        # at 4096 points, so request a little more than that for window edge.
        fft_size = 4096
        sample_count = min(
            max(2048, int(math.ceil(sample_rate * max(0.05, float(integration_seconds))))),
            fft_size,
        )

        client = RtltcpClient(receiver.host, receiver.port)
        with hold_device(self._device_key(receiver)):
            iq = client.capture(
                center_hz=center_hz,
                sample_rate_hz=sample_rate,
                sample_count=sample_count,
                gain_db=receiver.gain_db,
                ppm_error=receiver.ppm_error,
            )

        freqs, power_dbfs, rms_dbfs = scan_with_sample_rate(
            iq, center_hz, sample_rate, bins
        )
        result = scan_result_from_arrays(
            freqs,
            power_dbfs,
            source=ReceiverSource.RTL_TCP_IQ.value,
            integration_seconds=integration_seconds,
            warning=(
                f"Level is dBFS from an uncalibrated {receiver.host}; apply a calibration "
                f"offset before comparing across receivers."
            ),
        )
        result.rms_dbfs = rms_dbfs
        result.points = downsample_points(result.points, bins)
        return result

    def _scan_local(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        span_hz: float,
        bin_hz: Optional[float],
        integration_seconds: float,
        bins: int,
    ) -> ScanResult:
        if not rtl_power_available():
            raise SDRError(
                "rtl_power is not installed on this host; local receivers cannot be scanned"
            )

        effective_bin = max(500.0, float(bin_hz or 2000.0))
        effective_span = max(effective_bin * 4.0, span_hz)

        workdir = tempfile.mkdtemp(prefix="spectrum-alert-")
        csv_path = os.path.join(workdir, f"rtl_power_{int(time.time() * 1000)}.csv")
        try:
            args = self._rtl_power_args(
                receiver, center_hz, effective_span, effective_bin, integration_seconds, csv_path
            )
            with hold_device(self._device_key(receiver)):
                self._run_rtl_power(args)

            freqs, dbm = self._parse_rtl_power_csv(csv_path)
            if freqs.size == 0:
                raise SDRError("rtl_power produced no parsable data")

            result = scan_result_from_arrays(
                freqs,
                dbm,
                source=ReceiverSource.RTL_POWER.value,
                integration_seconds=integration_seconds,
            )
            result.center_hz = center_hz
            result.span_hz = max(effective_span, float(freqs[-1] - freqs[0]))
            result.bin_hz = effective_bin
            result.points = downsample_points(result.points, bins)
            return result
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def _rtl_power_args(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        span_hz: float,
        bin_hz: float,
        integration_seconds: float,
        csv_path: str,
    ) -> List[str]:
        start_hz = center_hz - span_hz / 2.0
        end_hz = center_hz + span_hz / 2.0
        kHz = max(1, int(round(bin_hz / 1000.0)))

        args = [
            "-f",
            f"{start_hz / 1e6:.6f}M:{end_hz / 1e6:.6f}M:{kHz}k",
            "-i",
            str(max(1, int(round(float(integration_seconds))))),
            "-1",
            "-d",
            str(receiver.device_index or 0),
        ]
        if receiver.gain_db is not None:
            args += ["-g", str(receiver.gain_db)]
        if receiver.ppm_error is not None:
            args += ["-p", str(receiver.ppm_error)]
        args.append(csv_path)
        return args

    def _run_rtl_power(self, args: Sequence[str]) -> None:
        try:
            completed = subprocess.run(
                [self.rtl_power_bin, *args],
                capture_output=True,
                timeout=RTL_POWER_TIMEOUT_S,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SDRError("rtl_power timed out") from exc
        except FileNotFoundError as exc:
            raise SDRError("rtl_power is not installed on this host") from exc

        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", "replace").strip()
            raise SDRError(f"rtl_power exited {completed.returncode}: {stderr or 'no detail'}")

    def _parse_rtl_power_csv(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        """Parse ``rtl_power`` CSV output into sorted ``(freqs_hz, dbm)``.

        ``rtl_power`` writes wide rows: a header row then blocks where columns
        2 and 4 hold the block start frequency and step, and columns 6 onward are
        the levels. Overlapping blocks are averaged rather than concatenated,
        which keeps the frequency axis monotonic.
        """
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = [line.strip() for line in handle if line.strip()]
        except OSError as exc:
            raise SDRError(f"could not read rtl_power output: {exc}") from exc

        points: List[tuple[float, float]] = []
        for line in lines:
            parts = [p.strip() for p in line.split(",") if p.strip()]
            if len(parts) < 7:
                continue
            try:
                start_hz = float(parts[2])
                step_hz = float(parts[4])
            except ValueError:
                continue
            levels: List[float] = []
            for cell in parts[6:]:
                try:
                    value = float(cell)
                except ValueError:
                    continue
                if math.isfinite(value):
                    levels.append(value)
            if not levels:
                continue
            points.extend((start_hz + step_hz * i, level) for i, level in enumerate(levels))

        if not points:
            return np.array([]), np.array([])

        # Averaging duplicates keeps overlapping blocks from creating steps.
        merged: dict[int, List[float]] = {}
        for hz, level in points:
            merged.setdefault(int(round(hz)), []).append(level)

        freqs = np.array(sorted(merged.keys()), dtype=np.float64)
        dbm = np.array([float(np.mean(merged[int(h)])) for h in freqs], dtype=np.float64)
        return freqs, dbm

    # ------------------------------------------------------------------ iq

    def capture_iq_burst(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        sample_rate_hz: float,
        sample_count: int = 16384,
        captures: int = 1,
    ) -> list[np.ndarray]:
        """Capture several IQ blocks, keeping a local dongle open throughout.

        Fingerprinting averages several short bursts. Calling ``capture_iq``
        once per burst reopened and reconfigured the USB device each time,
        which cost more than the fingerprint maths. Remote ``rtl_tcp`` receivers
        still go through the usual per-capture path.
        """
        captures = max(1, int(captures))
        if captures == 1:
            return [self.capture_iq(receiver, center_hz, sample_rate_hz, sample_count)]
        if receiver.mode.value == "rtltcp":
            return [
                self.capture_iq(receiver, center_hz, sample_rate_hz, sample_count)
                for _ in range(captures)
            ]

        if receiver.manual_rssi_db is not None:
            raise SDRError(
                f"{receiver.name}: a manual level override supplies no IQ samples. "
                f"Clear the pinned level to fingerprint this receiver."
            )

        from spectrum_alert.infrastructure.sdr.rtlsdr import RTLSDRInterface

        device = RTLSDRInterface(device_index=receiver.device_index or 0)
        device.open()
        try:
            device.set_center_freq(float(center_hz))
            device.set_sample_rate(float(sample_rate_hz))
            if receiver.gain_db is not None:
                device.set_gain(receiver.gain_db)
            return [np.asarray(device.read_samples(int(sample_count))) for _ in range(captures)]
        finally:
            device.close()

    def capture_iq(
        self,
        receiver: SDRReceiver,
        center_hz: float,
        sample_rate_hz: float,
        sample_count: int = 16384,
        integration_seconds: float = 0.0,
    ) -> np.ndarray:
        """Return raw complex IQ from ``receiver``.

        Distinct from :meth:`scan`, which reduces the stream to a power
        spectrum: fingerprinting needs the waveform itself, because the
        differences that identify a transmitter live in the envelope shape and
        the higher-order moments that a spectrum has already discarded.

        Raises :class:`SDRError` when the transport cannot supply samples.
        """
        center_hz = float(center_hz)
        sample_rate_hz = float(sample_rate_hz)
        sample_count = int(sample_count)

        if not math.isfinite(center_hz) or center_hz <= 0:
            raise SDRError("center_hz must be a positive frequency")
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise SDRError("sample_rate_hz must be positive")
        if sample_count <= 0:
            raise SDRError("sample_count must be positive")

        if receiver.manual_rssi_db is not None:
            # A pinned level is a single number. There is no modulation, no
            # envelope and no moment structure to fingerprint, so this must fail
            # loudly rather than return a flat trace that would match nothing
            # and look like a broken transmitter.
            raise SDRError(
                f"{receiver.name}: a manual level override supplies no IQ samples. "
                f"Clear the pinned level to fingerprint this receiver."
            )

        if receiver.mode.value == "rtltcp":
            if not receiver.host or not receiver.port:
                raise SDRError(f"{receiver.name}: rtl_tcp mode requires host and port")
            client = RtltcpClient(receiver.host, receiver.port)
            return np.asarray(
                client.capture(
                    center_hz=center_hz,
                    sample_rate_hz=sample_rate_hz,
                    sample_count=sample_count,
                    gain_db=receiver.gain_db,
                    ppm_error=receiver.ppm_error,
                )
            )

        from spectrum_alert.infrastructure.sdr.rtlsdr import RTLSDRInterface

        device = RTLSDRInterface(device_index=receiver.device_index or 0)
        device.open()
        try:
            device.set_center_freq(center_hz)
            device.set_sample_rate(sample_rate_hz)
            if receiver.gain_db is not None:
                device.set_gain(receiver.gain_db)
            return np.asarray(device.read_samples(sample_count))
        finally:
            device.close()

    # ----------------------------------------------------------------- checks

    def check(self, receiver: SDRReceiver, center_hz: float) -> Dict[str, Any]:
        """Probe whether ``receiver`` can be read right now.

        Returns a mapping with ``ok`` and ``message``. Never raises: the point is
        to report reachability to a UI.
        """
        if receiver.manual_rssi_db is not None:
            return {
                "id": receiver.id,
                "ok": True,
                "message": (
                    f"Manual level override active ({receiver.manual_rssi_db:.1f} dB); "
                    f"no hardware read performed."
                ),
                "source": ReceiverSource.MANUAL_OVERRIDE.value,
            }

        try:
            scan = self.scan(receiver, center_hz=float(center_hz), span_hz=64_000.0, bins=64)
        except SDRError as exc:
            return {"id": receiver.id, "ok": False, "message": str(exc)}
        except Exception as exc:  # defensive: the UI must never 500 on a probe
            logger.exception("unexpected error probing %s", receiver.name)
            return {"id": receiver.id, "ok": False, "message": f"unexpected error: {exc}"}

        peak = scan.peak_dbfs
        peak_hz = scan.peak_hz or center_hz
        return {
            "id": receiver.id,
            "ok": True,
            "message": (
                f"Measured {peak:.1f} dBFS peak at {peak_hz/1e6:.6f} MHz "
                f"(noise floor {scan.noise_floor_dbfs:.1f} dBFS) via {scan.source}"
            ),
            "peakDbfs": peak,
            "peakHz": peak_hz,
            "noiseFloorDbfs": scan.noise_floor_dbfs,
            "source": scan.source,
        }

    # ------------------------------------------------------------- measurement

    def measure_level(
        self,
        receiver: SDRReceiver,
        frequency_hz: float,
        window_hz: float = 12_000.0,
        span_hz: Optional[float] = None,
        integration_seconds: float = 1.0,
    ) -> Dict[str, Any]:
        """Measure level, noise floor and reliability at one frequency."""
        effective_span = float(span_hz or max(120_000.0, window_hz * 6.0))
        started = time.monotonic()
        scan = self.scan(
            receiver,
            center_hz=frequency_hz,
            span_hz=effective_span,
            integration_seconds=integration_seconds,
            bins=128,
        )
        elapsed_ms = (time.monotonic() - started) * 1000.0

        freqs = [p["hz"] for p in scan.points]
        values = [p["dbm"] for p in scan.points]
        analysis = analyze_spectrum(freqs, values, frequency_hz, window_hz=window_hz)

        return {
            "receiverId": receiver.id,
            "name": receiver.name,
            "lat": receiver.lat,
            "lon": receiver.lon,
            "rssiDbfs": analysis.target_dbfs,
            "noiseFloorDbfs": analysis.noise_floor_dbfs,
            "prominenceDb": analysis.prominence_db,
            "reliability": analysis.reliability,
            "overloaded": analysis.overloaded,
            "source": scan.source,
            "scanMs": round(elapsed_ms, 1),
            "warning": scan.warning,
        }

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clamp_sample_rate(requested: float) -> float:
        return float(min(MAX_SAMPLE_RATE_HZ, max(MIN_SAMPLE_RATE_HZ, float(requested))))

    # -------------------------------------------------------------- async glue

    async def scan_async(self, *args, **kwargs) -> ScanResult:
        """Async wrapper: ``read_samples`` blocks for hundreds of ms.

        Running a scan directly in a coroutine stalls the whole event loop, which
        is what froze the dashboard. Handing it to a thread keeps the server
        responsive.
        """
        return await asyncio.to_thread(self.scan, *args, **kwargs)

    async def scan_all_async(
        self,
        receivers: Sequence[SDRReceiver],
        center_hz: float,
        span_hz: float,
        bin_hz: Optional[float] = None,
        integration_seconds: float = 1.0,
        bins: int = DEFAULT_BINS,
    ) -> Dict[str, ScanResult]:
        return await asyncio.to_thread(
            self.scan_all, receivers, center_hz, span_hz, bin_hz, integration_seconds, bins
        )

    async def check_async(self, receiver: SDRReceiver, center_hz: float) -> Dict[str, Any]:
        return await asyncio.to_thread(self.check, receiver, center_hz)