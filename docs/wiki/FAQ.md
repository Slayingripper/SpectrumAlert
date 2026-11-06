# FAQ

### Do I have to use the GUI now?

No. The terminal interface is fully supported and uses the same backend, storage
and detector as the dashboard.

```bash
spectrum-alert monitor status
spectrum-alert monitor start --freq-start 144 --freq-end 148
spectrum-alert geo triangulate 446100000
```

### Did the Docker default change?

The default `docker compose up` service now runs the web dashboard instead of a
one-shot `monitor status` command. For CLI commands:

```bash
docker compose run --rm spectrumalert monitor status
```

The headless services remain available through the `service` and `autonomous`
profiles.

### Can I use the GUI and CLI at the same time?

Yes, but not on the same physical dongle at the same time. One process must own a
receiver while it is capturing. Use a second receiver for concurrent work.

### Why does the dashboard use port 8088?

Host port 8088 maps to container port 8000 by default to avoid a common conflict
with other local services. Override it:

```bash
SPECTRUM_ALERT_WEB_PORT=9000 docker compose up -d
```

A local `spectrum-alert web` defaults to port 8000.

### Is the reported level dBm?

No. The RTL-SDR has no calibrated power reference, so all levels are **dBFS**.
Apply per-receiver calibration before comparing absolute levels, and even then
the result depends on the calibration reference.

### Why does the model status show “--” for accuracy?

Accuracy needs labelled evaluation data. If a model was trained without labels,
accuracies and confidence cannot be measured. The panel shows `--` rather than a
fabricated score.

### Where is my data stored?

```text
data/spectrum/    JSON captures with I/Q samples
data/features/    feature CSVs for training
data/anomalies/   daily anomaly history
data/receivers.json
data/calibration.json
```

In Docker this maps to the checkout's `data/` directory.

### Can it run without internet?

The dashboard assets, Bootstrap and charts required by the core UI are served
locally or vendored. Leaflet map tiles are fetched from OpenStreetMap and will
not render offline; the rest of the dashboard remains usable.

### Can I test without an SDR?

Yes. Generate deterministic IQ fixtures:

```bash
python scripts/generate_iq_samples.py \
  --data-dir /tmp/spectrumalert-demo/data \
  --raw-dir  /tmp/spectrumalert-demo/raw
```

The fixture set exercises storage, feature extraction, anomaly detection,
training and the dashboard.

### Is a Raspberry Pi supported?

Yes. Use Lite mode, a lower sample rate and a modest FFT/window size. The UI is
responsive and the backend runs within the normal Python dependency set.

### How many receivers are needed for geolocation?

At least two enabled receivers with coordinates. Two produce an underconstrained
fix; three or more produce a usable estimate with a confidence ellipse.

### Where do I report a bug?

Open an issue at
<https://github.com/Slayingripper/SpectrumAlert/issues> and include:

- Operating system and Python version
- Receiver model and whether `rtl_test -t` works
- The command or dashboard action used
- Relevant log lines from `data/logs/` or `docker compose logs`
