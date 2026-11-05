"""Remote SDR support over the ``rtl_tcp`` protocol.

``rtl_tcp`` exposes an RTL-SDR as a TCP server that streams raw 8-bit IQ once
tuned. Talking to it directly (rather than shelling out to ``rtl_power
rtl_tcp=host:port``) gives access to a real receive chain on a machine that does
not have the SDR attached.

Protocol, from the wire up:

1. The server greets with a 12-byte ``rtl_tcp`` header: magic ``"RTL0"``, a tuner
   type, then four big-endian ``uint32`` values (tuner gain count, crystal rate,
   tuner gain count again, flags). We skip it.
2. Commands are 5 bytes: one opcode byte followed by a big-endian ``uint32``.
   We send SET_SAMPLE_RATE, SET_FREQ_CORRECTION, SET_GAIN_MODE, SET_GAIN,
   SET_AGC_MODE and SET_TUNER_FREQ.
3. Everything after our commands is interleaved IQ as unsigned bytes centred on
   127.5.

The scan is computed from that IQ with a windowed FFT, which is why this lives
in infrastructure rather than reusing :mod:`spectrum_analysis`'s Welch path: we
control the window and want a single low-latency transform, not an average.
"""

from __future__ import annotations

import logging
import socket
import struct
from typing import Optional

import numpy as np

from spectrum_alert.core.exceptions import SDRError

logger = logging.getLogger(__name__)

__all__ = ["RtltcpClient", "RTL_TCP_HEADER_SIZE", "RTL_TCP_MAGIC"]

RTL_TCP_HEADER_SIZE = 12
RTL_TCP_MAGIC = b"RTL0"

# Opcodes from rtl_tcp's protocol (rtl-sdr/src/rtl_tcp.c).
CMD_SET_FREQ = 0x01
CMD_SET_SAMPLE_RATE = 0x02
CMD_SET_GAIN_MODE = 0x03
CMD_SET_GAIN = 0x04
CMD_SET_FREQ_CORRECTION = 0x05
CMD_SET_AGC_MODE = 0x09

#: FFT sizes used for a scan. Kept modest so a scan stays well under a second.
MIN_FFT_SIZE = 512
MAX_FFT_SIZE = 4096


class RtltcpClient:
    """Minimal ``rtl_tcp`` client that yields IQ bytes."""

    def __init__(self, host: str, port: int, timeout: float = 4.0):
        if not host:
            raise SDRError("rtl_tcp host is required")
        if not port:
            raise SDRError("rtl_tcp port is required")
        self.host = str(host)
        self.port = int(port)
        self.timeout = float(timeout)

    def _command(self, opcode: int, value: int) -> bytes:
        return bytes([opcode]) + struct.pack(">I", int(value) & 0xFFFFFFFF)

    def capture(
        self,
        center_hz: float,
        sample_rate_hz: float,
        sample_count: int,
        gain_db: Optional[float] = None,
        ppm_error: Optional[float] = None,
    ) -> np.ndarray:
        """Tune, read ``sample_count`` complex samples, and return them.

        ``gain_db=None`` requests manual-gain-off (approximately 0 dB); it does
        *not* request AGC, which the RTL2832 tuner does not really implement.
        """
        if sample_count <= 0:
            raise SDRError("sample_count must be positive")

        sample_count = int(sample_count)
        needed_bytes = sample_count * 2  # 8-bit I + 8-bit Q

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)

        chunks: list[bytes] = []
        collected = 0
        setup_sent = False

        try:
            sock.connect((self.host, self.port))

            buffer = b""
            while collected < needed_bytes:
                try:
                    chunk = sock.recv(65536)
                except socket.timeout as exc:
                    raise SDRError(
                        f"rtl_tcp {self.host}:{self.port} timed out after "
                        f"{collected}/{needed_bytes} bytes"
                    ) from exc
                if not chunk:
                    raise SDRError(
                        f"rtl_tcp {self.host}:{self.port} closed after "
                        f"{collected}/{needed_bytes} bytes"
                    )

                buffer += chunk

                if not setup_sent:
                    if len(buffer) < RTL_TCP_HEADER_SIZE:
                        continue
                    if not buffer.startswith(RTL_TCP_MAGIC):
                        raise SDRError(
                            f"{self.host}:{self.port} did not send an rtl_tcp header"
                        )
                    buffer = buffer[RTL_TCP_HEADER_SIZE:]

                    # Tuner freq must follow sample rate and gain so the
                    # frontend is settled before the capture begins.
                    commands = [
                        self._command(CMD_SET_SAMPLE_RATE, round(sample_rate_hz)),
                        self._command(CMD_SET_GAIN_MODE, 1 if gain_db is not None else 0),
                        self._command(CMD_SET_AGC_MODE, 0),
                    ]
                    if gain_db is not None:
                        # Gain is in tenths of a dB on the wire.
                        commands.append(self._command(CMD_SET_GAIN, round(float(gain_db) * 10.0)))
                    if ppm_error is not None:
                        commands.append(self._command(CMD_SET_FREQ_CORRECTION, round(float(ppm_error))))
                    commands.append(self._command(CMD_SET_FREQ, round(center_hz)))
                    sock.sendall(b"".join(commands))
                    setup_sent = True

                if buffer:
                    chunks.append(buffer)
                    collected += len(buffer)
                    buffer = b""

            raw = b"".join(chunks)[:needed_bytes]
        except socket.timeout as exc:
            raise SDRError(f"rtl_tcp {self.host}:{self.port} connection timed out") from exc
        except (ConnectionRefusedError, ConnectionResetError, OSError) as exc:
            raise SDRError(f"rtl_tcp {self.host}:{self.port} failed: {exc}") from exc
        finally:
            try:
                sock.close()
            except OSError:
                pass

        # Unsigned bytes centred on 127.5 -> complex in roughly [-1, 1).
        iq = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        iq -= 127.5
        iq /= 128.0
        return (iq[0::2] + 1j * iq[1::2]).astype(np.complex64)

    def probe(self) -> tuple[bool, str]:
        """Check reachability. Returns ``(ok, message)`` and never raises."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(min(2.0, self.timeout))
        try:
            sock.connect((self.host, self.port))
            return True, f"Connected to rtl_tcp at {self.host}:{self.port}"
        except socket.timeout:
            return False, f"Timed out connecting to {self.host}:{self.port}"
        except OSError as exc:
            return False, f"Cannot reach {self.host}:{self.port}: {exc}"
        finally:
            try:
                sock.close()
            except OSError:
                pass


def _nearest_power_of_two(value: int) -> int:
    value = max(1, int(value))
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def scan_with_sample_rate(
    iq: np.ndarray,
    center_hz: float,
    sample_rate_hz: float,
    requested_bins: int = 256,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Like :func:`scan_from_iq` but with an explicit sample rate for the axis."""
    samples = np.asarray(iq)
    if samples.size == 0:
        raise SDRError("no IQ samples to analyse")

    sample_count = int(samples.shape[0])
    fft_size = max(MIN_FFT_SIZE, min(MAX_FFT_SIZE, _nearest_power_of_two(sample_count)))

    window = np.hanning(fft_size).astype(np.float32)
    block = np.zeros(fft_size, dtype=np.complex64)
    usable = min(fft_size, sample_count)
    block[:usable] = samples[:usable]
    block *= window

    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    rms_dbfs = 10.0 * np.log10(rms + 1e-15)

    spectrum = np.fft.fftshift(np.fft.fft(block))
    power = (np.abs(spectrum) ** 2).astype(np.float64) / fft_size
    power_dbfs = 10.0 * np.log10(np.maximum(power, 1e-20))

    bin_hz = float(sample_rate_hz) / fft_size
    start_hz = float(center_hz) - float(sample_rate_hz) / 2.0
    freqs = start_hz + np.arange(fft_size, dtype=np.float64) * bin_hz
    return freqs, power_dbfs, rms_dbfs