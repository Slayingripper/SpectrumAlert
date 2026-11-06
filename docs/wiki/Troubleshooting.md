# Troubleshooting

## The dashboard does not start

Check the process directly:

```bash
spectrum-alert web --host 0.0.0.0 --port 8000
```

Common causes:

- **Port already in use** — choose another port with `--port`; Docker uses host
  port 8088 by default.
- **Missing web dependencies** — reinstall with `pip install -e .`.
- **Wrong working directory in Docker** — the image serves package assets, not
  files from the checkout.

Check the container:

```bash
docker compose ps
docker compose logs -f spectrumalert
```

## No RTL-SDR detected

Check the host first:

```bash
lsusb | grep 0bda:2838
rtl_test -t
```

If `lsusb` shows nothing:

- Reconnect the dongle.
- Try a different USB port or a powered hub.
- Avoid USB 3 ports that have known RTL-SDR compatibility problems.

If `rtl_test` reports a permission error:

```bash
sudo usermod -aG plugdev "$USER"
```

Then log out and back in.

## `LIBUSB_ERROR_BUSY` or “Resource busy”

A physical dongle can only be open in one process. Stop dashboard monitoring
before running a CLI scan on the same receiver, or use a second dongle for a
second process.

Check for a stale process:

```bash
ps -ef | grep -E "spectrum-alert|rtl_"
```

## Waterfall is blank or the receiver appears wedged

If `read_samples` hangs while `rtl_test -t` still reports the device, the dongle
is in the known stuck state after an abnormal exit. Reset it:

```bash
usbreset 0bda:2838
```

If `usbreset` is unavailable, unplug and reconnect the dongle. A reset can change
the device path from `/dev/bus/usb/001/005` to another number, which is why the
Docker configuration maps the whole USB bus rather than one device path.

## Data Management is empty

Confirm the dashboard is using the expected working directory. The data
directory is relative to where the app was started:

```bash
pwd
ls data/features data/anomalies data/spectrum
```

In Docker, the data directory is mounted from the checkout:

```bash
docker compose exec spectrumalert ls /app/data
```

## Training says “No feature data found”

Training needs feature CSVs for the selected mode:

```bash
ls data/features/features_*_lite.csv
ls data/features/features_*_full.csv
```

A model needs at least ten usable feature rows. Generate test data with:

```bash
python scripts/generate_iq_samples.py --data-dir /tmp/demo/data
```

## Training succeeds but model status shows “Not deployed”

The model status polls every 30 seconds. Training completion triggers an
immediate refresh in current builds. If the panel still looks stale, reload the
dashboard or check:

```bash
ls models/
```

A model artifact must exist under `models/` in the same working directory as the
app.

## UI looks stale after an update

Hard-reload the page or rebuild the container:

```bash
docker compose build
docker compose up -d
```

The dashboard assets are served by the backend; a rebuilt image is required for
container deployments.

## Chart is empty while data exists

The Monitor chart requires a live `/api/spectrum` response. Feature and anomaly
archives populate Analytics and Data Management, not the live spectrum chart.
Use the Workspace scan or start monitoring to populate the waterfall.

## Related

- [Installation](Installation.md)
- [Docker](Docker.md)
- [FAQ](FAQ.md)
