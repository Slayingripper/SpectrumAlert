"""FastAPI routes for multi-receiver geolocation.

Ported and adapted from the ``RTL_Triangulator`` Express server, restructured onto
FastAPI. Endpoints are grouped as:

``/api/sdrs``                receiver roster CRUD and reachability probes
``/api/spectrum``            real spectrum scans (POST only; GET cannot choose a receiver)
``/api/triangulate``         geolocate a frequency from live scans
``/api/triangulate/readings``geolocate from readings the client already has
``/api/calibration``         per-receiver offsets: read, compute, apply, clear
``/api/ple/fit``             fit the path-loss exponent from known transmitters
``/api/pmr/scan``            PMR446 channel activity plus opportunistic geolocation
``/api/bandplans``           channel tables for the UI

Every scan runs through ``asyncio.to_thread``. An RTL-SDR read blocks for
hundreds of milliseconds and doing that in the event loop froze the entire
dashboard.
"""

from __future__ import annotations

import logging
import uuid
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from spectrum_alert.core.domain.receivers import SDRReceiver
from spectrum_alert.core.exceptions import SDRError
from spectrum_alert.core.services import band_plans as band_plan_service
from spectrum_alert.core.services.calibration import (
    CalibrationTable,
    build_observations,
    compute_relative_offsets,
    fit_path_loss_exponent,
    frequency_key,
)
from spectrum_alert.core.services.propagation import (
    frequency_default_path_loss,
    rssi_to_distance_meters,
    validate_path_loss_exponent,
)
from spectrum_alert.core.services.fingerprinting import (
    FEATURE_NAMES as FINGERPRINT_FEATURES,
    FingerprintLibrary,
    LibraryEntry,
    extract_signature,
    match_signature,
    summarise,
)
from spectrum_alert.core.services.spectrum_analysis import analyze_spectrum
from spectrum_alert.core.services.triangulation import (
    TriangulationReading,
    likelihood_grid,
    triangulate_position,
)
from spectrum_alert.infrastructure.sdr.scanner import SpectrumScanner, rtl_power_available
from spectrum_alert.infrastructure.storage.receiver_store import ReceiverStore

logger = logging.getLogger(__name__)

__all__ = ["build_router", "GeolocationService"]


# --------------------------------------------------------------------- models


class ReceiverPayload(BaseModel):
    """Upsert payload for a receiver."""

    id: Optional[str] = None
    name: str = Field(..., min_length=1)
    mode: str = Field("local")
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)
    deviceIndex: Optional[int] = None
    host: Optional[str] = None
    port: Optional[int] = None
    gainDb: Optional[float] = None
    ppmError: Optional[float] = None
    bandwidthHz: Optional[float] = None
    manualRssiDb: Optional[float] = None
    enabled: bool = True
    altitudeM: Optional[float] = None
    notes: Optional[str] = None


class SpectrumRequest(BaseModel):
    centerHz: float = Field(433_920_000.0, gt=0)
    spanHz: float = Field(2_000_000.0, gt=0)
    binHz: Optional[float] = None
    bins: int = Field(256, ge=64, le=1024)
    integrationSec: float = Field(1.0, gt=0, le=60.0)
    sdrId: Optional[str] = None


class CheckRequest(BaseModel):
    centerHz: float = Field(433_920_000.0, gt=0)


class TriangulateRequest(BaseModel):
    frequencyHz: float = Field(433_920_000.0, gt=0)
    referenceRssiDbm: Optional[float] = None
    pathLossExponent: Optional[float] = None
    minReliability: float = Field(0.2, ge=0.0, le=1.0)
    windowHz: float = Field(12_000.0, gt=0)
    #: Short integration: triangulation only needs enough averaging to resolve a
    #: carrier, and a long one makes the endpoint feel unresponsive.
    integrationSec: float = Field(0.5, gt=0, le=10.0)


class ReadingIn(BaseModel):
    id: str
    name: str = ""
    lat: float
    lon: float
    rssiDbm: float
    source: str = "unknown"
    reliability: Optional[float] = Field(None, ge=0.0, le=1.0)


class FingerprintRequest(BaseModel):
    """Capture-and-identify (and optionally enrol) one receiver."""

    sdrId: str = Field(..., min_length=1)
    frequencyHz: float = Field(..., gt=0)
    #: Number of short captures averaged into the signature. A single capture is
    #: dominated by the receiver's own noise realisation, so the default of 4 is
    #: what makes a repeat capture of the same radio match its reference.
    captures: int = Field(4, ge=1, le=16)
    sampleRateHz: float = Field(1_024_000.0, gt=0)
    samplesPerCapture: int = Field(16_384, ge=1024, le=262_144)
    #: Enrol the captured signature as a new reference in one step.
    label: Optional[str] = Field(None, max_length=120)
    vendor: str = Field("", max_length=80)
    model: str = Field("", max_length=80)
    deviceId: str = Field("", max_length=80)


class TriangulateReadingsRequest(BaseModel):
    readings: List[ReadingIn]
    referenceRssiDbm: Optional[float] = None
    pathLossExponent: Optional[float] = None
    includeLikelihood: bool = False
    #: Optional: lets the endpoint pick the same band-aware propagation
    #: defaults the live path uses. Without it the UHF fallback (n=3.5) is
    #: all that can be assumed, which silently mismodels VHF readings.
    frequencyHz: Optional[float] = Field(default=None, gt=0)


class Transmitter(BaseModel):
    name: str = ""
    frequencyHz: float = Field(..., gt=0)
    lat: float
    lon: float


class CalibrateRequest(BaseModel):
    transmitters: Optional[List[Transmitter]] = None
    apply: bool = False


class FitPlerRequest(BaseModel):
    transmitters: Optional[List[Transmitter]] = None
    apply: bool = False


class PmrScanRequest(BaseModel):
    triggerDeltaDb: float = Field(20.0, gt=0)
    minStrongReceivers: int = Field(3, ge=1)
    maxSimultaneousChannels: int = Field(4, ge=1)
    referenceRssiDbm: float = -40.0
    pathLossExponent: float = 3.5
    bins: int = Field(256, ge=64, le=1024)


class CalibrationOffsetPayload(BaseModel):
    receiverId: str
    frequencyHz: float = Field(..., gt=0)
    offsetDb: float


# -------------------------------------------------------------------- service


class GeolocationService:
    """Holds the receiver roster, calibration table and scanner for the router."""

    def __init__(
        self,
        receiver_store: ReceiverStore,
        calibration: CalibrationTable,
        scanner: Optional[SpectrumScanner] = None,
        fingerprints: Optional[FingerprintLibrary] = None,
    ) -> None:
        self.receivers = receiver_store
        self.calibration = calibration
        self.scanner = scanner or SpectrumScanner()
        # `fingerprints or FingerprintLibrary()` would discard the caller's
        # library whenever it is empty, and FingerprintLibrary defines __len__ so
        # an empty one is falsy. The result was a pathless library that never
        # persisted a single enrolled signature.
        self.fingerprints = (
            fingerprints if fingerprints is not None else FingerprintLibrary()
        )

    # ------------------------------------------------------------- receivers

    def list_receivers(self) -> List[Dict[str, Any]]:
        return [r.to_dict() for r in self.receivers.list()]

    def upsert_receiver(self, payload: ReceiverPayload) -> Dict[str, Any]:
        # Generate the id here when the client did not supply one. The domain
        # model requires an id, so building it before this point made every
        # create-without-id request fail: exactly what the UI's "Add receiver"
        # button does.
        fields = payload.model_dump(exclude_none=True)
        if not fields.get("id"):
            fields["id"] = str(uuid.uuid4())

        try:
            receiver = SDRReceiver.from_dict(fields)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid receiver: {exc}") from exc

        try:
            receiver.validate_transport()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return self.receivers.save(receiver).to_dict()

    def delete_receiver(self, receiver_id: str) -> bool:
        removed = self.receivers.delete(receiver_id)
        if removed:
            # Offsets for a deleted receiver are dead weight and would reappear
            # if the id were ever reused.
            self.calibration.clear_receiver(receiver_id)
            self.calibration.save()
        return removed

    # ---------------------------------------------------------------- scans

    async def spectrum(self, request: SpectrumRequest) -> Dict[str, Any]:
        receivers = self.receivers.enabled()
        if request.sdrId:
            match = next((r for r in receivers if r.id == request.sdrId), None)
            if match is None:
                raise HTTPException(status_code=404, detail="Receiver not found or disabled")
            targets = [match]
        else:
            if not receivers:
                raise HTTPException(
                    status_code=400,
                    detail="Add at least one enabled receiver before scanning spectrum.",
                )
            targets = [receivers[0]]

        # A missing or unreachable dongle is an expected condition, not a
        # server fault. Left unhandled it surfaced as a 500 "Internal Server
        # Error" with no explanation, which tells an operator nothing about
        # what to fix.
        try:
            result = await self.scanner.scan_async(
                targets[0],
                request.centerHz,
                request.spanHz,
                request.binHz,
                request.integrationSec,
                request.bins,
            )
        except SDRError as exc:
            raise HTTPException(
                status_code=503,
                detail=f"{targets[0].name} could not be scanned: {exc}",
            ) from exc
        payload = result.to_dict()
        payload.update(
            {
                "sdrId": targets[0].id,
                "sdrName": targets[0].name,
                "rtlPowerAvailable": rtl_power_available(),
            }
        )
        return payload

    async def check_receiver(self, receiver_id: str, request: CheckRequest) -> Dict[str, Any]:
        receiver = self.receivers.get(receiver_id)
        if receiver is None:
            raise HTTPException(status_code=404, detail="Receiver not found")
        return await self.scanner.check_async(receiver, request.centerHz)

    # -------------------------------------------------------- triangulation

    async def triangulate(self, request: TriangulateRequest) -> Dict[str, Any]:
        receivers = self.receivers.enabled()
        if len(receivers) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 enabled receivers are required for triangulation.",
            )

        warnings: List[str] = []
        defaults = frequency_default_path_loss(request.frequencyHz)
        exponent = (
            validate_path_loss_exponent(request.pathLossExponent, request.frequencyHz)
            if request.pathLossExponent is not None
            else defaults.exponent
        )
        if request.pathLossExponent is not None and exponent != request.pathLossExponent:
            warnings.append(
                f"Path-loss exponent {request.pathLossExponent} is outside the plausible "
                f"range; using band default {exponent}."
            )
        reference = (
            float(request.referenceRssiDbm) if request.referenceRssiDbm is not None
            else defaults.reference_rssi_dbm
        )

        # scan_all_async already skips receivers that fail, so reaching here means
        # none of them could be read at all. Report that as unavailable
        # hardware rather than an empty result.
        results = await self.scanner.scan_all_async(
            receivers,
            request.frequencyHz,
            max(120_000.0, request.windowHz * 6.0),
            None,
            request.integrationSec,
            128,
        )
        if not results:
            raise HTTPException(
                status_code=503,
                detail=(
                    "No receiver could be scanned. Check that the dongles are "
                    "connected and that the receiver configuration is correct."
                ),
            )

        readings: List[Dict[str, Any]] = []
        for receiver in receivers:
            scan = results.get(receiver.id)
            if scan is None:
                warnings.append(f"{receiver.name}: scan failed; excluded from the fix.")
                continue

            freqs = [p["hz"] for p in scan.points]
            values = [p["dbm"] for p in scan.points]
            analysis = analyze_spectrum(
                freqs, values, request.frequencyHz, window_hz=request.windowHz
            )

            offset = self.calibration.offset_for(receiver.id, request.frequencyHz)
            raw = analysis.target_dbfs
            if analysis.prominence_db < 1.5:
                warnings.append(
                    f"{receiver.name}: weak signal at "
                    f"{request.frequencyHz/1e6:.4f} MHz "
                    f"(prominence {analysis.prominence_db:.2f} dB)."
                )
            if scan.warning:
                warnings.append(f"{receiver.name}: {scan.warning}")

            readings.append(
                {
                    "id": receiver.id,
                    "name": receiver.name,
                    "lat": receiver.lat,
                    "lon": receiver.lon,
                    "rssiDbm": raw + offset,
                    "rawRssiDbm": raw,
                    "calOffsetDb": offset,
                    "noiseFloorDbm": analysis.noise_floor_dbfs + offset,
                    "prominenceDb": analysis.prominence_db,
                    "reliability": analysis.reliability,
                    "source": scan.source,
                    "manualOverride": receiver.manual_rssi_db is not None,
                }
            )

        if len(readings) < 2:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Only {len(readings)} receiver(s) reported at "
                    f"{request.frequencyHz/1e6:.4f} MHz; at least 2 are needed."
                ),
            )

        used_low_confidence = False
        reliable = [r for r in readings if r["reliability"] >= request.minReliability]
        selected = reliable
        if len(selected) < 2:
            # Fall back to the best available readings rather than refusing:
            # a weak signal is still worth locating, provided it is labelled.
            selected = sorted(readings, key=lambda r: r["reliability"], reverse=True)[:3]
            used_low_confidence = True
            warnings.append(
                f"Weak signal at {request.frequencyHz/1e6:.4f} MHz; using low-confidence "
                f"fallback with the best available receivers."
            )

        estimate = self._solve(selected, reference, exponent)
        for reading in readings:
            modeled = rssi_to_distance_meters(reading["rssiDbm"], reference, exponent)
            reading["modeledDistMeters"] = round(modeled, 2)

        if len(receivers) == 2:
            warnings.append(
                "Only 2 receivers: the estimate is underconstrained. Add a third for "
                "usable accuracy."
            )

        return {
            "frequencyHz": request.frequencyHz,
            "estimate": estimate,
            "readings": readings,
            "model": {
                "referenceRssiDbm": reference,
                "pathLossExponent": exponent,
                "band": defaults.band,
            },
            "measurement": {
                "usedLowConfidenceFallback": used_low_confidence,
                "warnings": warnings,
                "rtlPowerAvailable": rtl_power_available(),
            },
        }

    def _solve(
        self, readings: List[Dict[str, Any]], reference: float, exponent: float
    ) -> Dict[str, Any]:
        triples = [
            TriangulationReading(
                receiver_id=r["id"],
                name=r["name"],
                lat=r["lat"],
                lon=r["lon"],
                rssi_db=r["rssiDbm"],
                source=r.get("source", "unknown"),
                reliability=r.get("reliability"),
            )
            for r in readings
        ]
        try:
            estimate = triangulate_position(triples, reference, exponent)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return estimate.to_dict()

    def triangulate_from_readings(self, request: TriangulateReadingsRequest) -> Dict[str, Any]:
        if len(request.readings) < 2:
            raise HTTPException(status_code=400, detail="At least 2 readings are required.")

        # When the caller supplies the frequency, use the same band-aware
        # defaults as the live endpoint so both paths agree. Without it, 3.5
        # (the UHF default) is the only defensible fallback.
        if request.frequencyHz is not None:
            defaults = frequency_default_path_loss(request.frequencyHz)
            default_exponent = defaults.exponent
            default_reference = defaults.reference_rssi_dbm
        else:
            default_exponent = 3.5
            default_reference = -40.0

        exponent = (
            validate_path_loss_exponent(
                request.pathLossExponent, request.frequencyHz or 446_100_000.0
            )
            if request.pathLossExponent is not None
            else default_exponent
        )
        reference = (
            float(request.referenceRssiDbm)
            if request.referenceRssiDbm is not None
            else default_reference
        )

        triples = [
            TriangulationReading(
                receiver_id=r.id,
                name=r.name or r.id,
                lat=r.lat,
                lon=r.lon,
                rssi_db=r.rssiDbm,
                source=r.source,
                reliability=r.reliability,
            )
            for r in request.readings
        ]
        try:
            estimate = triangulate_position(triples, reference, exponent)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        payload = {
            "estimate": estimate.to_dict(),
            "readings": [r.model_dump() for r in request.readings],
            "model": {"referenceRssiDbm": reference, "pathLossExponent": exponent},
        }

        if request.includeLikelihood:
            box = _likelihood_box(triples)
            payload["likelihood"] = {
                "box": list(box),
                "steps": 36,
                "grid": likelihood_grid(triples, reference, exponent, box),
            }
        return payload

    # ---------------------------------------------------------- fingerprinting

    def capture_signature(self, request: FingerprintRequest) -> Dict[str, Any]:
        """Capture IQ, derive a signature, and identify it against the library.

        Returns the signature, the match verdict and any enrolment performed in
        the same request. Hardware failures surface as 503 with the reason
        rather than as an empty result.
        """
        receiver = self.receivers.get(request.sdrId)
        if receiver is None:
            raise HTTPException(status_code=404, detail=f"Unknown receiver {request.sdrId!r}")

        sample_rate = float(request.sampleRateHz)
        vectors = []
        for _ in range(max(1, int(request.captures))):
            try:
                samples = self.scanner.capture_iq(
                    receiver,
                    center_hz=float(request.frequencyHz),
                    sample_rate_hz=sample_rate,
                    sample_count=int(request.samplesPerCapture),
                )
            except SDRError as exc:
                raise HTTPException(
                    status_code=503,
                    detail=f"{receiver.name} could not be captured for fingerprinting: {exc}",
                ) from exc
            if samples is None or len(samples) == 0:
                raise HTTPException(
                    status_code=503,
                    detail=f"{receiver.name} returned no samples.",
                )
            vectors.append(extract_signature(samples, sample_rate))

        # Averaging the burst gives the signature; the spread across those same
        # captures tells the matcher which features are just receiver noise on
        # this particular signal.
        signature, spread = summarise(vectors)
        verdict = match_signature(signature, self.fingerprints)

        enrolled = None
        if request.label:
            entry = LibraryEntry(
                label=request.label,
                vector=signature,
                spread=spread,
                device_id=request.deviceId or request.label,
                vendor=request.vendor,
                model=request.model,
                frequency_hz=float(request.frequencyHz),
                samples=int(request.samplesPerCapture) * int(request.captures),
            )
            self.fingerprints.add(entry)
            enrolled = entry.to_dict()

        verdict.update(
            {
                "receiver": receiver.to_dict(),
                "captures": int(request.captures),
                "sampleRateHz": sample_rate,
                "samplesPerCapture": int(request.samplesPerCapture),
                "enrolled": enrolled,
                "featureNames": list(FINGERPRINT_FEATURES),
            }
        )
        return verdict

    def list_fingerprints(self) -> Dict[str, Any]:
        return {
            "entries": [entry.to_dict() for entry in self.fingerprints.entries],
            "featureNames": list(FINGERPRINT_FEATURES),
        }

    def delete_fingerprint(self, entry_id: str) -> Dict[str, Any]:
        if not self.fingerprints.remove(entry_id):
            raise HTTPException(status_code=404, detail=f"Unknown library entry {entry_id!r}")
        return {"status": "ok", "entryId": entry_id}

    # ----------------------------------------------------------- calibration

    def get_calibration(self) -> Dict[str, Any]:
        return self.calibration.to_dict()

    def set_calibration_offset(self, payload: CalibrationOffsetPayload) -> Dict[str, Any]:
        self.calibration.apply(payload.receiverId, payload.frequencyHz, payload.offsetDb)
        self.calibration.save()
        return self.calibration.to_dict()

    def clear_calibration_receiver(self, receiver_id: str) -> Dict[str, Any]:
        self.calibration.clear_receiver(receiver_id)
        self.calibration.save()
        return self.calibration.to_dict()

    def clear_calibration(self) -> Dict[str, Any]:
        self.calibration.clear_all()
        self.calibration.save()
        return self.calibration.to_dict()

    async def calibrate(self, request: CalibrateRequest) -> Dict[str, Any]:
        """Measure known transmitters and derive equalisation offsets.

        ``apply=False`` computes a preview only, so an operator can inspect the
        numbers before committing them. Offsets outside the MAD-based rejection
        threshold are reported and skipped.
        """
        receivers = self.receivers.enabled()
        if not receivers:
            raise HTTPException(
                status_code=400, detail="Add at least one enabled receiver before calibration."
            )

        transmitters = [
            {"name": t.name, "frequency_hz": t.frequencyHz, "lat": t.lat, "lon": t.lon}
            for t in (request.transmitters or [])
        ]
        if not transmitters:
            transmitters = DEFAULT_TRANSMITTERS

        summary: List[Dict[str, Any]] = []
        preview: Dict[str, Dict[str, float]] = {}
        rejected: List[str] = []

        for transmitter in transmitters:
            frequency_hz = transmitter["frequency_hz"]
            readings: List[Dict[str, Any]] = []

            for receiver in receivers:
                try:
                    scan = await self.scanner.scan_async(
                        receiver, frequency_hz, 160_000.0, None, 1.0, 128
                    )
                except SDRError as exc:
                    rejected.append(f"{receiver.name} @ {frequency_hz/1e6:.3f} MHz: {exc}")
                    continue
                if receiver.manual_rssi_db is not None:
                    continue
                if scan.source in {"manual_override"}:
                    continue

                freqs = [p["hz"] for p in scan.points]
                values = [p["dbm"] for p in scan.points]
                analysis = analyze_spectrum(freqs, values, frequency_hz, window_hz=12_000.0)
                readings.append(
                    {
                        "receiver_id": receiver.id,
                        "receiver_name": receiver.name,
                        "rssi_dbfs": analysis.target_dbfs,
                        "prominence_db": analysis.prominence_db,
                        "noise_floor_dbfs": analysis.noise_floor_dbfs,
                        "reliability": analysis.reliability,
                        "source": scan.source,
                    }
                )

            offsets, rejected_here = compute_relative_offsets(readings, frequency_hz)
            rejected.extend(rejected_here)

            values = [r["rssi_dbfs"] for r in readings]
            centre = sorted(values)[len(values) // 2] if values else None
            summary.append(
                {
                    "transmitter": transmitter.get("name") or f"{frequency_hz/1e6:.3f} MHz",
                    "frequencyHz": frequency_hz,
                    "centerDbfs": centre,
                    "readings": readings,
                    "rejected": rejected_here,
                }
            )

            for receiver_id, entry in offsets.items():
                preview.setdefault(receiver_id, {})[frequency_key(frequency_hz)] = entry

        if request.apply and preview:
            self.calibration.merge(preview)
            self.calibration.save()

        return {
            "ok": True,
            "applied": bool(request.apply),
            "previewOffsets": preview,
            "summary": summary,
            "rejected": rejected,
            "calibration": self.calibration.to_dict() if request.apply else None,
        }

    async def fit_ple(self, request: FitPlerRequest) -> Dict[str, Any]:
        """Fit the path-loss exponent from known transmitter geometry."""
        receivers = self.receivers.enabled()
        if not receivers:
            raise HTTPException(
                status_code=400, detail="Add at least one enabled receiver before fitting PLE."
            )

        transmitters = [
            {"name": t.name, "frequency_hz": t.frequencyHz, "lat": t.lat, "lon": t.lon}
            for t in (request.transmitters or [])
        ]
        if not transmitters:
            transmitters = DEFAULT_TRANSMITTERS

        measurements: Dict[Any, float] = {}
        errors: List[str] = []

        for transmitter in transmitters:
            frequency_hz = transmitter["frequency_hz"]
            for receiver in receivers:
                if receiver.manual_rssi_db is not None:
                    continue
                try:
                    scan = await self.scanner.scan_async(
                        receiver, frequency_hz, 160_000.0, None, 1.0, 128
                    )
                except SDRError as exc:
                    errors.append(
                        f"{receiver.name} @ {frequency_hz/1e6:.3f} MHz: {exc}"
                    )
                    continue
                freqs = [p["hz"] for p in scan.points]
                values = [p["dbm"] for p in scan.points]
                analysis = analyze_spectrum(freqs, values, frequency_hz, window_hz=12_000.0)
                if analysis.reliability < 0.2:
                    errors.append(
                        f"{receiver.name} @ {frequency_hz/1e6:.3f} MHz: weak or unreliable reading"
                    )
                    continue
                measurements[(receiver.id, frequency_hz)] = analysis.target_dbfs

        observations = build_observations(
            transmitters, [r.to_dict() for r in receivers], measurements, self.calibration
        )
        result = fit_path_loss_exponent(
            observations, default_frequency_hz=transmitters[0]["frequency_hz"]
        )

        payload = result.to_dict()
        payload["errors"] = result.errors + errors
        payload["applied"] = bool(request.apply)

        if request.apply and not result.fallback_applied:
            # Report the accepted model so the caller can store it as the default.
            # Offsets are per-frequency equalisation, not the path-loss model, so
            # a fitted exponent deliberately does not go into the offset table.
            payload["model"] = {
                "referenceRssiDbm": result.reference_rssi_dbm,
                "pathLossExponent": result.path_loss_exponent,
                "r2": result.r_squared,
            }
        return payload

    # ------------------------------------------------------------- PMR446

    async def pmr_scan(self, request: PmrScanRequest) -> Dict[str, Any]:
        """One wide capture covering all 16 PMR446 channels.

        A single 300 kHz span contains the whole plan, so all channels are
        measured from one tune per receiver instead of sixteen.
        """
        receivers = self.receivers.enabled()
        if not receivers:
            raise HTTPException(status_code=400, detail="No enabled receivers.")

        plan = band_plan_service.BAND_PLANS["pmr446"]
        centre = (plan.low_hz + plan.high_hz) / 2.0
        span = plan.span_hz + 40_000.0

        results = await self.scanner.scan_all_async(receivers, centre, span, None, 0.15, request.bins)
        if not results:
            raise HTTPException(
                status_code=503,
                detail="No receiver could be scanned; check the dongles and receiver configuration.",
            )

        rows: List[Dict[str, Any]] = []
        for receiver in receivers:
            scan = results.get(receiver.id)
            if scan is None:
                rows.append({
                    "sdrId": receiver.id,
                    "name": receiver.name,
                    "lat": receiver.lat,
                    "lon": receiver.lon,
                    "error": "scan failed",
                    "channels": None,
                })
                continue

            freqs = [p["hz"] for p in scan.points]
            values = [p["dbm"] for p in scan.points]
            floor = None
            channels: List[Dict[str, Any]] = []
            for channel in plan.channels:
                analysis = analyze_spectrum(
                    freqs, values, channel.frequency_hz,
                    window_hz=channel.bandwidth_hz / 2.0,
                )
                offset = self.calibration.offset_for(receiver.id, channel.frequency_hz)
                floor = analysis.noise_floor_dbfs + offset if floor is None else floor
                channels.append(
                    {
                        "ch": channel.number,
                        "name": channel.name,
                        "freq": channel.frequency_hz,
                        "rssiDbm": round(analysis.target_dbfs + offset, 2),
                        "prominenceDb": round(analysis.prominence_db, 2),
                        "color": channel.color,
                    }
                )
            rows.append(
                {
                    "sdrId": receiver.id,
                    "name": receiver.name,
                    "lat": receiver.lat,
                    "lon": receiver.lon,
                    "source": scan.source,
                    "noiseFloorDbm": floor,
                    "channels": channels,
                }
            )

        return {
            "centerHz": centre,
            "spanHz": span,
            "timestamp": datetime.utcnow().isoformat(),
            "plan": {"key": plan.key, "label": plan.label},
            "triggerDeltaDb": request.triggerDeltaDb,
            "minStrongReceivers": request.minStrongReceivers,
            "receivers": rows,
        }


def _likelihood_box(
    readings: List[TriangulationReading], pad_ratio: float = 0.35, pad_meters: float = 500.0
) -> tuple[float, float, float, float]:
    lats = [r.lat for r in readings]
    lons = [r.lon for r in readings]
    from spectrum_alert.core.services.propagation import (
        meters_to_latitude_degrees,
        meters_to_longitude_degrees,
    )

    mid_lat = (min(lats) + max(lats)) / 2.0
    lat_pad = max((max(lats) - min(lats)) * pad_ratio, meters_to_latitude_degrees(pad_meters))
    lon_pad = max(
        (max(lons) - min(lons)) * pad_ratio,
        meters_to_longitude_degrees(pad_meters, mid_lat),
    )
    return (
        min(lats) - lat_pad,
        min(lons) - lon_pad,
        max(lats) + lat_pad,
        max(lons) + lon_pad,
    )


#: Broadcast transmitters used as calibration and path-loss anchors by default.
#: Coordinates are the CyBC Troodos main site plus two nearer Cyprus sites,
#: chosen so the distance spread is wide enough to fit a slope.
DEFAULT_TRANSMITTERS: List[Dict[str, Any]] = [
    {"name": "CyBC Radio 4 (88.2 MHz)", "frequency_hz": 88_200_000.0,
     "lat": 34.9369, "lon": 32.8600},
    {"name": "CyBC Radio 2 (91.1 MHz)", "frequency_hz": 91_100_000.0,
     "lat": 34.9369, "lon": 32.8600},
    {"name": "CyBC Radio 3 (94.8 MHz)", "frequency_hz": 94_800_000.0,
     "lat": 34.9369, "lon": 32.8600},
    {"name": "CyBC Radio 1 (97.2 MHz)", "frequency_hz": 97_200_000.0,
     "lat": 34.9369, "lon": 32.8600},
    {"name": "Antenna FM (103.7 MHz)", "frequency_hz": 103_700_000.0,
     "lat": 34.8600, "lon": 33.2810},
    {"name": "Radio Lefkosia (107.2 MHz)", "frequency_hz": 107_200_000.0,
     "lat": 35.0825, "lon": 33.3850},
]


# --------------------------------------------------------------------- router


def build_router(service: GeolocationService) -> APIRouter:
    """Build the geolocation router bound to ``service``."""
    router = APIRouter(prefix="/api", tags=["geolocation"])

    @router.get("/sdrs")
    async def list_receivers():
        return {"sdrs": service.list_receivers()}

    @router.post("/sdrs")
    async def upsert_receiver(payload: ReceiverPayload):
        return {"sdr": service.upsert_receiver(payload)}

    @router.delete("/sdrs/{receiver_id}")
    async def delete_receiver(receiver_id: str):
        if not service.delete_receiver(receiver_id):
            raise HTTPException(status_code=404, detail="Receiver not found")
        return {"ok": True, "deleted": receiver_id}

    @router.delete("/sdrs")
    async def clear_receivers():
        for receiver in service.receivers.list():
            service.calibration.clear_receiver(receiver.id)
        service.calibration.save()
        service.receivers.clear()
        return {"ok": True}

    @router.post("/sdrs/{receiver_id}/check")
    async def check_receiver(receiver_id: str, request: CheckRequest):
        return await service.check_receiver(receiver_id, request)

    @router.get("/spectrum")
    async def spectrum_get():
        # A GET cannot select a receiver or an integration time, and silently
        # scanning the first receiver would hide that from the caller.
        raise HTTPException(
            status_code=405,
            detail="Use POST /api/spectrum with a receiver and integration time.",
        )

    @router.post("/spectrum")
    async def spectrum(request: SpectrumRequest):
        return await service.spectrum(request)

    @router.post("/triangulate")
    async def triangulate(request: TriangulateRequest):
        return await service.triangulate(request)

    @router.post("/triangulate/readings")
    async def triangulate_readings(request: TriangulateReadingsRequest):
        return service.triangulate_from_readings(request)

    @router.get("/calibration")
    async def get_calibration():
        return service.get_calibration()

    @router.post("/calibration")
    async def set_calibration(payload: CalibrationOffsetPayload):
        return service.set_calibration_offset(payload)

    @router.delete("/calibration/{receiver_id}")
    async def clear_calibration_receiver(receiver_id: str):
        return service.clear_calibration_receiver(receiver_id)

    @router.delete("/calibration")
    async def clear_calibration():
        return service.clear_calibration()

    @router.post("/calibrate")
    async def calibrate(request: CalibrateRequest):
        return await service.calibrate(request)

    @router.post("/ple/fit")
    async def fit_ple(request: FitPlerRequest):
        return await service.fit_ple(request)

    @router.post("/pmr/scan")
    async def pmr_scan(request: PmrScanRequest):
        return await service.pmr_scan(request)

    @router.get("/bandplans")
    async def bandplans():
        return {"bandPlans": band_plan_service.list_band_plans()}

    @router.get("/bandplans/{key}")
    async def bandplan(key: str):
        plan = band_plan_service.get_band_plan(key)
        if plan is None:
            raise HTTPException(status_code=404, detail="Unknown band plan")
        return plan.to_dict()

    # ------------------------------------------------------- fingerprinting

    @router.get("/fingerprints")
    async def list_fingerprints():
        return service.list_fingerprints()

    @router.post("/fingerprints/capture")
    async def capture_fingerprint(request: FingerprintRequest):
        """Capture one receiver, identify it, and optionally enrol it."""
        return service.capture_signature(request)

    @router.delete("/fingerprints/{entry_id}")
    async def delete_fingerprint(entry_id: str):
        return service.delete_fingerprint(entry_id)

    @router.get("/tools")
    async def tools():
        from spectrum_alert.infrastructure.sdr.scanner import rtl_test_available

        return {
            "rtlPowerAvailable": rtl_power_available(),
            "rtlTestAvailable": rtl_test_available(),
            "receiverCount": len(service.receivers.list()),
            "enabledCount": len(service.receivers.enabled()),
        }

    return router


def default_service(data_dir: str = "data") -> GeolocationService:
    """Build the service with on-disk defaults."""
    os.makedirs(data_dir, exist_ok=True)
    return GeolocationService(
        ReceiverStore(os.path.join(data_dir, "receivers.json")),
        CalibrationTable.load(os.path.join(data_dir, "calibration.json")),
        FingerprintLibrary.load(os.path.join(data_dir, "fingerprints.json")),
    )