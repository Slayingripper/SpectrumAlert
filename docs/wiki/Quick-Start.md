# Quick Start

## 1. Start the dashboard

```bash
spectrum-alert web --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000>.

For Docker:

```bash
docker compose up -d
```

Open <http://localhost:8088>.

## 2. Confirm the receiver

The **System status** panel reports the detected RTL-SDR and host metrics. The
same information is available from the terminal:

```bash
spectrum-alert monitor status
```

If the panel reports no receiver, see [Troubleshooting](Troubleshooting.md).

## 3. Start a scan

In the dashboard, open **Monitor → Spectrum analysis** and press **Start**. The
centre frequency defaults to 145 MHz with a 2.048 MHz window.

From the terminal, the equivalent scan is:

```bash
spectrum-alert monitor start --freq-start 144 --freq-end 148 --mode full
```

## 4. Review detections

Open the **Threats** module for the severity summary, timeline and recent alerts.
An alert can be acknowledged or sent to the workspace for locating.

The boundaries are:

- **Spectrum data** — raw captures and feature rows
- **Detection** — SNR and prominence over the noise floor
- **Anomaly** — a detection that passed the configured threshold and gating

## 5. Open the analysis workspace

Select **Workspace** in the app bar. The workspace contains:

- Receiver roster and add/edit controls
- Calibration and path-loss fitting
- Live waterfall and peak table
- Emitter map with confidence ellipse
- RF fingerprint capture and reference library
- PMR446 channel scanner
- Activity log

Triangulation needs at least two enabled receivers with coordinates. The
requirement indicator under the target frequency states how many are available.

## 6. Try it without a dongle

Generate deterministic IQ fixtures:

```bash
python scripts/generate_iq_samples.py \
  --data-dir /tmp/spectrumalert-demo/data \
  --raw-dir  /tmp/spectrumalert-demo/raw
```

Run the dashboard from that data directory:

```bash
cd /tmp/spectrumalert-demo
PYTHONPATH=/path/to/SpectrumAlert python3 -m uvicorn \
  spectrum_alert.web.run:create_app --factory --port 8000
```

The fixture generator writes clean captures, four detectable signals, full and
lite feature rows, and anomaly history.

## Next

- [Web Dashboard](Web-Dashboard.md)
- [CLI Reference](CLI-Reference.md)
- [Data and Training](Data-and-Training.md)
