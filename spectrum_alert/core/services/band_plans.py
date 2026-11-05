"""Band plans and frequency presets.

Ported from the ``RTL_Triangulator`` project's hardcoded PMR446 channel table,
extended with the other band plans that matter for unattended monitoring.

A band plan is a list of channels. Knowing the channels turns "something is
present somewhere in 446 MHz" into "PMR446 channel 7 just went active", which is
what makes an alert actionable instead of merely noisy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

__all__ = [
    "Channel",
    "BandPlan",
    "BAND_PLANS",
    "get_band_plan",
    "list_band_plans",
    "nearest_channel",
    "pmr446_channel_at",
]


@dataclass(frozen=True)
class Channel:
    """One channel within a band plan."""

    number: int
    frequency_hz: float
    name: str = ""
    bandwidth_hz: float = 12_500.0
    color: str = "#22c55e"
    mode: str = "fm"

    def to_dict(self) -> Dict[str, object]:
        return {
            "number": self.number,
            "frequencyHz": self.frequency_hz,
            "name": self.name or f"CH{self.number}",
            "bandwidthHz": self.bandwidth_hz,
            "color": self.color,
            "mode": self.mode,
        }


@dataclass(frozen=True)
class BandPlan:
    """A named collection of channels."""

    key: str
    label: str
    description: str
    channels: List[Channel]
    default_step_hz: float = 12_500.0

    @property
    def low_hz(self) -> float:
        return min(c.frequency_hz for c in self.channels)

    @property
    def high_hz(self) -> float:
        return max(c.frequency_hz for c in self.channels)

    @property
    def span_hz(self) -> float:
        return self.high_hz - self.low_hz

    def to_dict(self) -> Dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "lowHz": self.low_hz,
            "highHz": self.high_hz,
            "spanHz": self.span_hz,
            "channelCount": len(self.channels),
            "channels": [c.to_dict() for c in self.channels],
        }


def _pmr446() -> BandPlan:
    """PMR446: 16 channels, 12.5 kHz spacing from 446.00625 MHz.

    Licence-exempt UHF PMR in the EU. Low power (0.5 W ERP), so a received
    signal stands well clear of the noise floor, which makes it a good target
    for receiver-driven detection.
    """
    colors = [
        "#ef4444", "#f97316", "#f59e0b", "#eab308", "#84cc16", "#22c55e",
        "#10b981", "#14b8a6", "#06b6d4", "#0ea5e9", "#3b82f6", "#6366f1",
        "#8b5cf6", "#a855f7", "#d946ef", "#ec4899",
    ]
    base = 446_006_250.0
    channels = [
        Channel(
            number=i + 1,
            frequency_hz=base + i * 12_500.0,
            name=f"CH{i + 1}",
            bandwidth_hz=12_500.0,
            color=colors[i % len(colors)],
            mode="fm",
        )
        for i in range(16)
    ]
    return BandPlan(
        key="pmr446",
        label="PMR446",
        description="EU licence-exempt UHF PMR radio, 16 channels at 12.5 kHz.",
        channels=channels,
    )


def _ism868() -> BandPlan:
    """EU 863-870 MHz short-range device band.

    A dense, legally open band. One single 2.4 MSps capture covers 868.0-869.7
    MHz, so this is a cheap thing to watch.
    """
    channels = [
        Channel(number=868_000 + i * 200_000, frequency_hz=868_000_000 + i * 200_000,
                name=f"868.{i * 0.2:.1f}", bandwidth_hz=200_000.0, mode="gfsk")
        for i in range(20)
    ]
    return BandPlan(
        key="ism868",
        label="EU ISM 863-870 MHz",
        description="EU short-range device band; telemetry, alarms, trackers.",
        channels=channels,
        default_step_hz=200_000.0,
    )


def _ism915() -> BandPlan:
    """US 902-928 MHz ISM band."""
    channels = [
        Channel(number=i, frequency_hz=902_000_000 + i * 200_000,
                name=f"902.{i * 0.2:.1f}", bandwidth_hz=200_000.0, mode="gfsk")
        for i in range(130)
    ]
    return BandPlan(
        key="ism915",
        label="US ISM 902-928 MHz",
        description="US ISM band; widely used for telemetry and mesh nodes.",
        channels=channels,
        default_step_hz=200_000.0,
    )


def _ism433() -> BandPlan:
    """EU 433-434 MHz short-range device band."""
    channels = [
        Channel(number=i, frequency_hz=433_050_000 + i * 25_000,
                name=f"433.{0.050 + i * 0.025:.3f}", bandwidth_hz=25_000.0, mode="nfm")
        for i in range(40)
    ]
    return BandPlan(
        key="ism433",
        label="EU ISM 433-434 MHz",
        description="EU SRD band; weather sensors, remotes, garage doors.",
        channels=channels,
        default_step_hz=25_000.0,
    )


def _ham2m() -> BandPlan:
    """2 m amateur band, common repeater and beacon frequencies."""
    spots = [
        ("144.000", 144_000_000), ("144.174", 144_174_000), ("144.474", 144_474_000),
        ("145.000", 145_000_000), ("145.2125", 145_212_500), ("145.500", 145_500_000),
        ("145.600", 145_600_000), ("145.675", 145_675_000), ("145.800", 145_800_000),
    ]
    channels = [
        Channel(number=i + 1, frequency_hz=hz, name=label, bandwidth_hz=12_500.0, mode="ssb/fm")
        for i, (label, hz) in enumerate(spots)
    ]
    return BandPlan(
        key="ham2m",
        label="2 m amateur",
        description="144-146 MHz; FM voice, repeaters and packet.",
        channels=channels,
        default_step_hz=12_500.0,
    )


def _ham70cm() -> BandPlan:
    """70 cm amateur band, common repeater and beacon frequencies."""
    spots = [
        ("430.000", 430_000_000), ("430.400", 430_400_000), ("431.000", 431_000_000),
        ("432.000", 432_000_000), ("432.100", 432_100_000), ("432.200", 432_200_000),
        ("433.000", 433_000_000), ("433.500", 433_500_000), ("434.000", 434_000_000),
    ]
    channels = [
        Channel(number=i + 1, frequency_hz=hz, name=label, bandwidth_hz=12_500.0, mode="ssb/fm")
        for i, (label, hz) in enumerate(spots)
    ]
    return BandPlan(
        key="ham70cm",
        label="70 cm amateur",
        description="430-440 MHz; FM voice and repeaters.",
        channels=channels,
        default_step_hz=12_500.0,
    )


def _airband() -> BandPlan:
    """Civil aviation AM band."""
    spots = [
        ("118.000", 118_000_000), ("119.025", 119_025_000), ("120.725", 120_725_000),
        ("121.500", 121_500_000), ("121.700", 121_700_000), ("124.350", 124_350_000),
        ("127.800", 127_800_000), ("132.550", 132_550_000), ("135.800", 135_800_000),
    ]
    channels = [
        Channel(number=i + 1, frequency_hz=hz, name=label, bandwidth_hz=25_000.0, mode="am")
        for i, (label, hz) in enumerate(spots)
    ]
    return BandPlan(
        key="airband",
        label="Air band (118-136 MHz)",
        description="Civil aviation AM voice and navaids.",
        channels=channels,
        default_step_hz=25_000.0,
    )


def _broadband_fm() -> BandPlan:
    """FM broadcast band, sampled at broadcast channel centres."""
    channels = [
        Channel(number=i, frequency_hz=87_500_000 + i * 200_000,
                name=f"{87.5 + i * 0.2:.1f}", bandwidth_hz=200_000.0, mode="fm")
        for i in range(53)
    ]
    return BandPlan(
        key="fm_broadcast",
        label="FM broadcast 87.5-108 MHz",
        description="Broadcast FM; strong, wideband, useful as a path-loss anchor.",
        channels=channels,
        default_step_hz=200_000.0,
    )


BAND_PLANS: Dict[str, BandPlan] = {
    plan.key: plan
    for plan in (
        _pmr446(),
        _ism433(),
        _ism868(),
        _ism915(),
        _ham2m(),
        _ham70cm(),
        _airband(),
        _broadband_fm(),
    )
}


def get_band_plan(key: str) -> Optional[BandPlan]:
    """Look up a band plan by key, or ``None``."""
    return BAND_PLANS.get(str(key or "").strip().lower())


def list_band_plans() -> List[Dict[str, object]]:
    """Summaries of every known band plan, for UI population."""
    return [
        {
            "key": plan.key,
            "label": plan.label,
            "description": plan.description,
            "lowHz": plan.low_hz,
            "highHz": plan.high_hz,
            "channelCount": len(plan.channels),
        }
        for plan in BAND_PLANS.values()
    ]


def nearest_channel(plan: BandPlan, frequency_hz: float) -> Optional[Channel]:
    """Channel in ``plan`` closest to ``frequency_hz``."""
    if not plan.channels:
        return None
    return min(plan.channels, key=lambda c: abs(c.frequency_hz - float(frequency_hz)))


def pmr446_channel_at(frequency_hz: float) -> Optional[Channel]:
    """PMR446 channel containing ``frequency_hz``, or ``None`` if outside the plan."""
    plan = BAND_PLANS["pmr446"]
    half = plan.default_step_hz / 2.0
    for channel in plan.channels:
        if abs(channel.frequency_hz - float(frequency_hz)) <= half:
            return channel
    return None