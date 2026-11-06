# Installation

## Requirements

- Python 3.8 or newer
- An RTL-SDR dongle for live capture (optional for UI work and synthetic fixtures)
- Linux is recommended. macOS and Windows require suitable USB drivers.

## Local install

```bash
git clone https://github.com/Slayingripper/SpectrumAlert.git
cd SpectrumAlert

python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Verify the CLI:

```bash
spectrum-alert --help
spectrum-alert version
```

## Receiver access on Linux

Confirm the dongle is visible:

```bash
lsusb | grep 0bda:2838
rtl_test -t
```

If `rtl_test` reports a permissions error, add your user to `plugdev` and reload
the session:

```bash
sudo usermod -aG plugdev "$USER"
```

The RTL-SDR packages ship udev rules on most distributions. Reconnect the dongle
after installing them.

Check that SpectrumAlert sees the receiver:

```bash
spectrum-alert monitor status
```

A missing dongle is not a startup failure. The dashboard still loads and reports
that no SDR is attached.

## Docker install

```bash
docker compose up -d
```

Open <http://localhost:8088>. See [Docker](Docker.md) for USB access, ports and
headless profiles.

## Remote receivers

A receiver on another host can be used through `rtl_tcp`:

```bash
# On the remote host
rtl_tcp -a 0.0.0.0 -p 1234

# In SpectrumAlert
spectrum-alert geo add-receiver Hill --lat 35.08 --lon 33.38 \
  --mode rtltcp --host 10.0.0.5 --port 1234
```

## Verify the install

```bash
spectrum-alert monitor status
spectrum-alert geo receivers
pytest
```

The UI contract and backend test suite runs without an SDR. The single
hardware-dependent collection test requires a real dongle.

## Next

- [Quick Start](Quick-Start.md)
- [Web Dashboard](Web-Dashboard.md)
- [CLI Reference](CLI-Reference.md)
